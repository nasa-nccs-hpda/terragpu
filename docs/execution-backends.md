# Execution backends: decision and evidence

Decision, 2026-09-11: keep current NumPy/CuPy, xarray and Rasterio/rioxarray.
Make Dask/distributed optional. Add a direct, tile-at-a-time path as the
single-device reference. Do not claim that one backend always wins.

| Workload | First choice | Reason / limitation |
|---|---|---|
| Small array fitting comfortably in RAM | NumPy | Avoid scheduler overhead |
| Array fitting in GPU memory | CuPy | Explicit device-resident arithmetic |
| Single-scene pointwise processing beyond device memory | Rasterio windows + CuPy | No task graph; application arrays bounded by tile dimensions |
| Large lazy expressions, collections, multiple GPUs | Dask arrays + optional Dask-CUDA | Composable graphs and distributed scheduling; tune and benchmark chunks |

Dask's own [best practices](https://docs.dask.org/en/latest/array-best-practices.html)
recommend NumPy when suitable and caution against small/misaligned chunks.
[Rasterio windows](https://rasterio.readthedocs.io/en/latest/topics/windowed-rw.html)
provide direct partial reads/writes, but actual read granularity depends on source
blocks. Streaming tile size should be a multiple of source block size where
possible. Current streaming is synchronous and single-device; it does not overlap
I/O/transfers or implement halo exchange. Focal filters will require halos;
reprojection requires different source-window logic and is not implemented here.

## Alternatives assessed

- **Cubed:** xarray supports this alternative with an emphasis on predictable
  memory. Worth an experiment for rechunking and array-store workloads, but no
  PRISM GPU evidence currently justifies replacing Dask.
  [xarray overview](https://xarray.dev/blog/cubed-xarray).
- **cuSpatial:** do not adopt as a new core dependency. RAPIDS stopped publishing
  cuSpatial packages in release 25.06.
  [RAPIDS notices](https://docs.rapids.ai/notices/).
- **Fused CuPy kernels:** prioritize after profiling demonstrates that elementwise
  allocations/launches dominate. Preserve a numerical reference and measure JIT
  separately. [CuPy performance guidance](https://docs.cupy.dev/en/stable/user_guide/performance.html).
- **Zarr and product-native readers:** add when PACE/AVIRIS product formats and
  layouts are identified. A storage change is not automatically a compute speedup.

No new unqualified scheduler or JIT dependency is added solely for novelty.
Novel research should come from a demonstrated method, reproducible performance,
and correct geospatial results.

## Measurement

`python -m terragpu.benchmark_io` compares actual compressed GeoTIFF read → NDVI
→ write, matching output tile layout. Inputs and validation reference are
synthetic; repeats alternate backend order. Filesystem cache is uncontrolled.
Dask uses one thread so the first experiment isolates scheduler/array overhead,
not maximum CPU throughput. Timings include transfers and writer close; correctness
is checked outside timing. Streaming also includes output-validation setup and
an atomic rename, so its overhead remains visible. Run both CPU and CuPy modes
on PRISM, then add tuned threaded/distributed Dask lanes before choosing a default
for throughput. This harness is not yet a publication-grade memory benchmark.

The earlier resident 2048² test measured ~4.0 ms NumPy versus ~5.9 ms Dask locally;
those timings alone do not establish end-to-end or GPU performance.

Local end-to-end smoke run: 2048² synthetic GeoTIFF, 1024² tiles, 3 repeats,
NumPy streaming median 119.7 ms and single-threaded Dask median 126.0 ms.
Both passed the float64 numerical reference. This small uncontrolled-cache
sample supports retaining both paths; it is insufficient to claim a general
speedup. Raw output: results/io-comparison.json (local ignored artifact).
