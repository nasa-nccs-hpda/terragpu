# Installation and qualification

CPU raster core (Python 3.11+):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,parallel]'
python -m pytest
terragpu-benchmark --backend numpy --output results/numpy.json
```

The base install has no Dask, distributed, or model-framework requirement.
Install `.[parallel]` for Dask arrays and clusters. Use `backend="numpy"` or
`backend="cupy"` for direct execution and `terragpu.streaming.process_indices`
for single-device window processing without Dask.

GPU execution requires Linux and a working NVIDIA environment. On PRISM,
start with the site's supported CUDA 12 environment for V100 and a controlled
H100 comparison. Install exactly one matching CuPy distribution, e.g.
`python -m pip install 'cupy-cuda12x>=14,<15'`, then test an actual computation.
Use the [CuPy installation guide](https://docs.cupy.dev/en/stable/install.html)
for toolkit/header requirements. For Dask-CUDA, use the matching
[RAPIDS installation selector](https://docs.rapids.ai/install/); independently
upgrading Dask after RAPIDS installation can invalidate compatibility.

`environment_gpu.yml` is a CUDA 12 raster-only candidate, not a qualified RAPIDS
lock. CUDA 13 removes compilation support for Volta/V100; use a separate H100
environment for CUDA 13 experiments. Record `pip freeze`, `conda list --explicit`
(if applicable), `nvidia-smi`, and the source revision for every hardware run.
The local CPU freeze is platform-specific evidence, not a Linux/GPU lock.

Run `bash scripts/prism_benchmark.sh results/prism-smoke` within an allocated
PRISM GPU job, with this package installed in the active environment. No site
partition/account/module names are assumed. Current benchmark scope is synthetic,
resident, single-GPU NDVI; see docs/modernization-plan.md for real-data milestones.

The Dockerfile is now a CPU reference recipe; it has not been built locally.
GitLab CI now tests the package rather than publishing the old model container.
