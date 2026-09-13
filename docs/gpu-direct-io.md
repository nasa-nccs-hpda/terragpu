# Experimental GPU raster I/O on PRISM

TerraGPU now has a small geospatial GPU cache reader/writer. It preserves CRS,
affine transform, band names and nodata while allowing KvikIO to read and write
raster payloads in CuPy device buffers. It is an experimental working format,
not a replacement for Rasterio's GeoTIFF codecs. Initial ingestion and final
GeoTIFF export use Rasterio on the CPU, and their costs are reported separately.

## Run in an existing GPU allocation

From the repository root, with `uv` installed:

```bash
git pull --ff-only
srun --ntasks=1 bash scripts/setup_prism_gpu_io.sh
srun --ntasks=1 --export=ALL bash -ec '
  arch=$(uname -m)
  if [[ "$arch" == arm64 ]]; then arch=aarch64; fi
  source ".venv-io-$arch/bin/activate"
  bash scripts/run_prism_gpu_io.sh results/prism-gpu-io-01
'
tar -czf prism-gpu-io-01-results.tar.gz -C results prism-gpu-io-01
```

The setup creates a separate architecture-specific environment, uses the existing
CUDA 12/CuPy setup, and installs KvikIO CUDA 12 version 26.8.0. Linux ARM64 and
x86-64 are supported by the selected wheels. PRISM's installed driver and
storage configuration still need to pass the on-node probes.

The default example automatically downloads the public WorldView sample (~257 MB),
or verifies its existing cache. **No Earthdata token is needed for this example.**
It first produces QA-masked NDVI/NDWI on CPU, then benchmarks multi-scale spatial
means and variances at 15- and 31-pixel windows. Centered moments accumulate in float64;
outputs are float32 on both backends. This is a new experiment alongside the
existing five-product paper suite, which remains available unchanged.

Use a fresh results directory for each run. `TERRAGPU_REPEAT` defaults to 5.
`TERRAGPU_DATA_ROOT` defaults to `data`. `TERRAGPU_IO_ROOT` defaults to
`data/gpu-io`; set it to an existing writable scratch location to test that
filesystem. Input caches and intermediate output caches live on that filesystem
and are removed when the experiment exits. Allow several GB of scratch space.
Test different storage locations in separate runs and retain each mount report.
An H100/GH200 alone does not establish that a filesystem supports GPUDirect.

## What is measured

Identical cache layouts are processed with four modes:

| Mode | Payload path |
| --- | --- |
| `numpy` | Host read, CPU computation, host write |
| `cupy` | Host read, H2D copy, GPU computation, D2H copy, host write |
| `kvikio-compat` | KvikIO device-buffer API with compatibility mode enabled |
| `kvikio-cufile` | KvikIO device-buffer API with cuFile requested |

Each tile stays on the GPU through all requested spatial operations. There is
no application-level host copy between these operations. KvikIO may still stage
data internally, and cuFile can use compatibility paths. This version does not
implement asynchronous read/compute/write overlap, GPU compression or native
compressed GeoTIFF/NetCDF decoding.

The cuFile read/write probe runs on the actual scratch filesystem. If it fails,
that mode is explicitly skipped; other modes continue. Use `--require-cufile`
with the Python CLI to make that failure fatal. All reports retain
`gds_verified: false`: neither successful cuFile I/O nor `gdscheck` output proves
that the measured operations bypassed host memory. Mount-specific runtime
telemetry is required before making a GPUDirect claim.

JSON stores every repetition, first-run timings, read/compute/write phase timings,
bootstrap summaries, correctness errors, preparation/export costs, source hash,
package versions, GPU identity and thread settings. CSV contains plotting metrics
and matching CPU/backend ratios for each tile size. Validation is outside timing.
The runner also captures CUDA configuration, mount information and available
`gdscheck` diagnostics. Review logs for private system paths before publishing.

Filesystem caches are uncontrolled, and writes do not include an `fsync` durability
barrier. First-run timings are not guaranteed cold-disk measurements. Ratios describe
the **prepared-cache pipeline**, not complete native-product processing. Charge
native preparation, cache conversion and required GeoTIFF export when calculating
end-to-end time or the number of repeated analyses needed to amortize preparation.
The CPU baseline uses one thread; it does not establish a win over a tuned
multi-core implementation. Peak GPU memory is not yet measured.

For larger real imagery or another sweep, activate the environment and run:

```bash
python -m terragpu.benchmark_gpu_io \
  --source /path/to/georeferenced.tif \
  --work-root data/gpu-io \
  --tiles 512 1024 2048 --sizes 15 31 63 \
  --repeat 7 --warmup 1 --output results/larger-raster.json
```

Use representative larger scenes and more substantial analysis, not duplicated
pixels to manufacture a speedup. The suite validates every output against a
float64 reference. The variance estimator centers each tile before accumulating moments, reducing
cancellation when values have a large offset. Extreme within-tile dynamic range
can still require a different algorithm; numerical validation remains mandatory.

## Reader/writer API

```python
from terragpu.gpu_io import pack_raster, RasterCache, CacheWriter

pack_raster("input.tif", "input-cache", tile_size=512, halo=15)
cache = RasterCache("input-cache", verify=True)
# Tile includes the halo; arrays have (band, y, x) axes.
tile = cache.read_tile(0, 0, mode="kvikio-cufile")

# Example identity copy, with device buffers passed directly to KvikIO.
with CacheWriter("output-cache", cache.meta, mode="kvikio-cufile") as writer:
    for row, col in cache.tiles():
        writer.write_tile(row, col, cache.read_tile(row, col, "kvikio-cufile"))
RasterCache("output-cache").to_geotiff("output.tif")  # Explicit host export
```

The versioned cache consists of JSON metadata and uncompressed little-endian
float32 chunks padded to 4096-byte file lengths. Halo duplication and edge
padding consume storage; logical and physical input sizes are recorded. This
layout trades storage for simple contiguous reads, and is not a standardized
interchange format. Ingestion applies source scale/offset and maps masks to NaNs.
Input checksums are verified before benchmarking. GPU-produced outputs are
numerically validated outside timing; the writer does not create host-side
checksums for those payloads. Existing destinations are rejected and incomplete
writes are cleaned up. Reader and writer metadata checks reject nonfinite or
singular affine transforms, missing/invalid CRS, invalid band labels and invalid
dimensions before chunk I/O. Rotated and skewed invertible transforms are retained.

Relevant upstream documentation: [KvikIO quickstart](https://docs.nvidia.com/kvikio/latest/quickstart/)
and [KvikIO CUDA 12 package](https://pypi.org/project/kvikio-cu12/26.8.0/).

## Diagnose GDS without repeating the benchmark

From the repository in an active PRISM allocation:

```bash
srun --ntasks=1 --export=ALL bash scripts/check_prism_gds.sh data/gpu-io \
  2>&1 | tee results/prism-gds-check.txt
```

This read-only check locates both `gdscheck` and `gdscheck.py`, including
`CUDA_HOME` and versioned installations under `/usr/local`. It reports the
filesystem and relevant loaded kernel modules. Missing tools do not prove GDS
is unavailable: NCCS may provide them through a module or another installation
path. Ask NCCS which exact mount is enabled for GDS and which CUDA/GDS module
qualifies the GH200 nodes. A positive platform check must still be followed by
mount-specific read/write validation with cuFile telemetry. Do not infer direct
storage from a successful compatibility-mode operation.

## Auditable plotting metrics

New PRISM runs generate a `metrics/` directory automatically after the benchmark.
For an existing completed run, including one downloaded from PRISM:

```bash
python -m terragpu.gpu_io_metrics \
  results/prism-gpu-io-nvme-02/gpu-io.json \
  results/prism-gpu-io-nvme-02/metrics
```

The exporter needs only Python's standard library. It validates completion,
clean revision, the successful-mode/tile matrix, numerical-validation flags,
repetition counts, finite nonnegative phase times, phase totals and stored medians.
CPU/backend ratios require matching workload, shape, scope and input sizes.
It rejects incomplete or inconsistent reports before creating output and refuses
to overwrite an existing metrics directory.

`summary.csv` contains recomputed medians and within-run min/max values;
`samples.csv` contains every measured repetition and phase for plotting.
`same-tile-ratios.csv` compares each GPU mode with its matching CPU tile size,
when CPU results are present. `preparation.csv` retains separately measured
conversion/export costs. `provenance.json` records source identifiers, thread
settings and interpretation limits. These are checks of recorded evidence, not
a rerun of numerical validation. The exporter cannot establish GDS, independent
job uncertainty, tuned CPU performance or native-product end-to-end speedups.
Phase medians must not be added and presented as the median total.
