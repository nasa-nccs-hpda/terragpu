# Expanded benchmark runs inside an existing PRISM allocation

Use the `codex/geospatial-revival` branch. From the repository root, with `uv`
installed, run:

```bash
git switch codex/geospatial-revival
git pull --ff-only
srun --ntasks=1 bash scripts/setup_prism_uv.sh gpu
source .venv-prism/bin/activate
srun --ntasks=1 bash scripts/run_prism_paper.sh results/prism-h100 both
tar -czf prism-h100-results.tar.gz -C results prism-h100
```

Use `results/prism-v100` and `prism-v100-results.tar.gz` for a V100 allocation.
`srun` ensures the work runs on an allocated compute node; no new `salloc`,
account, partition, or hard-coded PRISM path is required. The Python environment
and repository must be on a filesystem visible to the allocated node.
The runner uses the first visible CUDA device and runs CPU and GPU sequentially
on that same node. It does not use all allocated GPUs.

`setup_prism_uv.sh` creates a Python 3.12 environment and installs the benchmark
extras with `uv pip`, NumPy <2.5 / SciPy <1.18 (within CuPy 14's tested API range),
and `cupy-cuda12x[ctk]==14.2.0`. CUDA 12 supports the shared V100/H100 baseline;
CUDA 13 is unsuitable for the V100 compilation path. CUDA components are
provided by Python wheels; the host must have a compatible NVIDIA driver.
The script performs a real CUDA compilation/computation and fails early if
the environment is incompatible. Keep that error log if preflight fails;
do not silently switch to CPU. The resolved environment is saved in the venv,
and the runner records `pip freeze` again with the results.
References: [uv environments](https://docs.astral.sh/uv/pip/environments/),
[CuPy installation](https://docs.cupy.dev/en/stable/install.html).

For a CPU-only allocation, use `setup_prism_uv.sh cpu` and
`run_prism_paper.sh results/prism-cpu cpu`. Use a new output directory for every
run. Inputs download automatically before timing (about 275 MB for the public
WorldView and SatStereo samples); no Earthdata credentials are required.
To prefetch on a network-enabled node, run `terragpu-data worldview` and
`terragpu-data satstereo` first. Transfer the resulting `data/` directories to
the shared filesystem if compute nodes cannot reach the internet.

## Workloads

| Workload | Data | What it measures |
|---|---|---|
| NDVI | Fixed-seed synthetic 4096² bands | NumPy, CuPy, Dask and Dask-CuPy, resident data |
| Focal mean | WorldView-3 ARD red band, 4251² | NaN-aware 15×15 neighborhood reduction |
| Spectral angle | Fixed-seed 512×512×136 cube | Dot products, norms and angular distance to a supplied spectrum |
| Stereo matching | SatStereo MP1, 1286×1298 pair | Census descriptors, Hamming costs, spatial aggregation, disparity search |
| WorldView NDVI/NDWI I/O | WorldView-3 ARD and QA | Open, read, QA alignment, transfers, compute, compressed write and close |

The first three complex workloads report resident-input timing, cold execution,
and input transfer separately. Host output copies for correctness checks are
outside resident timings. The I/O workload includes transfers and file close;
its CPU/GPU ordering alternates. The runner does not flush filesystem caches.
CPU numerical libraries are limited to one thread: this is a defined baseline,
not a claim that a GPU beats every optimized multicore CPU configuration.
The existing Dask benchmark uses one execution thread and a prebuilt graph;
it is not a multi-GPU or cluster throughput measurement.

### Tiled focal processing

For a separate streaming experiment on a georeferenced index raster:

```bash
python -m terragpu.focal indices.tif focal-mean.tif --band 1 --size 15 --tile-size 512 --backend cupy
```

Install the `benchmark` extra for its SciPy CPU reference. Each output tile reads
a halo of `size//2` pixels, computes the neighborhood mean, and writes only its
core. Outside-image pixels contribute nothing. Nodata holes are filled when a
window has finite neighbors; all-invalid windows remain NaN. Stored scale/offset
are applied, but product QA/calibration are not inferred—use a prepared index or
reflectance raster. Tests compare seams and edge windows with independent manual
neighborhood means, including tile sizes smaller than the halo. The paper suite's
existing focal measurement is still resident-input timing; this streaming CLI
is a separate path and does not silently change its scope.

Each operation is checked outside timing. NumPy/CuPy results are compared on
the entire output. Focal samples and the spectral-angle result have independent
CPU arithmetic checks; synthetic tests check stereo recovery of known signed
shifts and invalid support. CPU/GPU Census disparities must match exactly.
The runner's WorldView I/O check compares against the CPU adapter; its independent
geospatial/QA validator is `scripts/validate_worldview.py` and is tested separately.

## Photogrammetry dataset and accuracy

The pinned [Purdue SatStereo sample](https://engineering.purdue.edu/RVL/Database/SatStereo/index.html)
contains two rectified WorldView-3 images, two disparity maps, building masks,
and acquisition metadata. The archive is 17,906,618 bytes, SHA-256
`700398ac6c8b918c8359afc94422905fd74944404ea397dc1d7806fa74e44217`.
The download page specifies GPL v3 for the dataset; original files and
attribution stay in the local cache, not in the software repository.
Cite Patil et al., *A New Stereo Benchmarking Dataset for Satellite Images*
([2019 paper](https://arxiv.org/abs/1907.04404)). Their reference disparities are
derived using LiDAR and fused DSM alignment; they are not error-free truth.

The Census baseline uses a 5×5 descriptor, 5×5 cost aggregation, integer
disparities -128…128, and deterministic lowest-disparity tie breaking. It stores
one disparity's costs at a time instead of a full H×W×D cost volume. There is
no learned component, subpixel refinement, occlusion consistency check, RPC
triangulation or bundle adjustment. Disparity convention is
`x_right = x_left + disparity`. A check of reciprocal reference maps supported
this sign: median reciprocal residual 0.0124 pixels versus 2.78 for the opposite
sign on deterministic subsamples. This sign check does not tune matcher settings.

Quality is reported for all finite reference pixels and separately for the
provided building mask, including coverage, MAE and bad-3-pixel percentage.
Reference disparities outside the fixed search range are counted explicitly;
they are not silently removed from error statistics. The images are 152 days
apart, so radiometry, shadows and scene changes make this a challenging case.
One pair is sufficient for harness development, not broad reconstruction claims.
Expand to more SatStereo AOIs/pairs before final paper comparisons; ETH3D is
another option for terrestrial stereo, but not a substitute for satellite data.

## Results to bring back

`run_prism_paper.sh` defaults to two warmups and seven measured repetitions;
set `TERRAGPU_REPEAT=15` for more timing samples. It writes:

- `suite.json`: raw timings, cold/transfer times, bootstrap intervals, input
  hashes, hardware, software versions, revision, correctness and stereo quality.
- `ndvi-*.json`: explicit execution/scheduler metadata for the NDVI comparisons.
- `summary.csv`, `samples.csv`, `speedups.json`: plot-ready exports.
- `tests.txt`, `nvidia-smi.txt`, `cupy-config.txt`, `pip-freeze.txt`, and logs.
- `run-status.txt`: 0 for completion; retain nonzero failures for diagnosis.

Bring back each `.tar.gz` results bundle. It contains metrics/logs, not imagery
or credentials. We can then plot throughput, CPU/GPU ratios, runtime distributions,
and stereo quality, keeping resident and end-to-end scopes separate. Bootstrap
intervals describe timing repetitions on one input, not dataset generalization.
An interrupted suite leaves `.partial.json`; partial results are not a completed
paper run. Publishable claims still require matched V100/H100 runs, multiple
scenes/sizes, an optimized multicore CPU comparison, memory measurements and
appropriate external baselines (e.g. established classical stereo implementations).
