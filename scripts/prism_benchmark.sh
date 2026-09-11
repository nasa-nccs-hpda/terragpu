#!/usr/bin/env bash
# Run inside an allocated PRISM GPU job with the environment already activated.
set -euo pipefail
out=${1:-results/prism-smoke}
mkdir -p "$out"
# Keep failures inspectable without automatically uploading results.
trap 'status=$?; printf "%s\n" "$status" > "$out/run-status.txt"' EXIT
exec > >(tee "$out/run.log") 2>&1
nvidia-smi > "$out/nvidia-smi.txt"
python -m pip freeze > "$out/pip-freeze.txt"
git rev-parse HEAD > "$out/git-commit.txt"
git status --short > "$out/git-status.txt"
python -c 'import cupy as cp; print(cp.show_config()); assert cp.cuda.runtime.getDeviceCount() > 0; print(cp.arange(10).sum().get())' > "$out/cupy-config.txt" 2>&1
python -m pytest tests -q > "$out/tests.txt" 2>&1
# Full suite must pass, and GPU test must run (CuPy preflight above is mandatory).
for backend in numpy dask cupy dask-cupy; do
  for size in 512 2048 4096; do
    python -m terragpu.benchmark --backend "$backend" --size "$size" \
      --chunk 1024 --warmup 3 --repeat 10 --output "$out/${backend}-${size}.json"
  done
done

python -m terragpu.benchmark_io --device numpy --repeat 5 --output "$out/io-numpy.json"
python -m terragpu.benchmark_io --device cupy --repeat 5 --output "$out/io-cupy.json"
