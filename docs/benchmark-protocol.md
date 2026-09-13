# Running the initial benchmark

Install the core and run:

```bash
terragpu-benchmark --backend numpy --size 2048 --repeat 10 --output results/numpy.json
terragpu-benchmark --backend dask --size 2048 --chunk 1024 --repeat 10 --output results/dask.json
# In an allocated GPU environment:
terragpu-benchmark --backend cupy --size 2048 --repeat 10 --output results/cupy.json
terragpu-benchmark --backend dask-cupy --size 2048 --chunk 1024 --repeat 10 --output results/dask-cupy.json
```

The fixed-seed workload is two float32 reflectance bands and NDVI. Each result
is checked against independently computed float64 NumPy output (rtol=1e-5,
atol=1e-6). GPU requests fail if CUDA is unavailable. CUDA device synchronization
bounds timed work. JSON records samples, warmups, cold execution, separate
transfer and graph-build times, versions, hardware, revision and dirty status.

These are **resident-input microbenchmarks**: input generation, disk I/O and
transfers are excluded from repeat timings. Dask uses a prebuilt graph and a
single-threaded scheduler. Its full input and output fit in memory; this does
not test streaming, distributed execution or multiple GPUs. NumPy execution
includes xarray expression construction; Dask graph construction is reported
separately. CPU transfer fields largely measure timer/call overhead. Cold timing
is the first operation in this process, not a guarantee of an empty CUDA JIT cache.

2026-09-11 local smoke results (Apple Silicon, Python 3.13.1, 2048² pixels,
10 repeats, 2 warmups): NumPy median 3.994 ms; Dask with 1024² chunks median
5.932 ms. Maximum absolute error was 1.37e-7 for both. Raw files are in
results/local-numpy.json and results/local-dask.json (ignored local artifacts).
This verifies the harness; it does not establish a GPU speedup or a historical
performance improvement. The working tree was uncommitted at measurement time.

Use scripts/prism_benchmark.sh inside a PRISM allocation for the initial matrix.
Before publishable runs, commit the source and archive the complete environment
and input manifests. See modernization-plan.md for the real-data, memory,
statistical, multi-GPU and paper requirements still to implement.

## Geospatial scope update

Dask is now optional (`pip install '.[parallel]'`). Run
`python -m terragpu.benchmark_io --device numpy --output results/io.json` to
compare windowed and Dask read/compute/write. Use `--device cupy` inside a GPU
allocation. See execution-backends.md for boundaries and limitations. Training
and classifier benchmarks are no longer part of the plan.
