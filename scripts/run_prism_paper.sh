#!/usr/bin/env bash
# Run inside an existing salloc session. No partition/account/site paths assumed.
set -euo pipefail
out=${1:-results/prism-$(date -u +%Y%m%dT%H%M%SZ)}
mode=${2:-both}
repeat=${TERRAGPU_REPEAT:-7}
case "$mode" in both) backends=(numpy cupy);; cpu) backends=(numpy);; *) echo 'Mode must be both or cpu' >&2;exit 2;; esac
if [[ -e "$out" ]]; then echo "Use a new output directory: $out" >&2;exit 1;fi
mkdir -p "$out"
trap 'status=$?; printf "%s\n" "$status" > "$out/run-status.txt"' EXIT
exec > >(tee "$out/run.log") 2>&1
# A defined single-thread CPU baseline; use a separate experiment for scaling.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -m pip freeze > "$out/pip-freeze.txt"
lscpu > "$out/lscpu.txt"
git rev-parse HEAD > "$out/git-commit.txt"
git status --short > "$out/git-status.txt"
if [[ -s "$out/git-status.txt" ]]; then echo 'Commit or stash source changes before a benchmark run.' >&2;exit 1;fi
if [[ "$mode" == both ]]; then
  nvidia-smi > "$out/nvidia-smi.txt"
  python -c 'import cupy as cp; cp.show_config(); assert cp.cuda.runtime.getDeviceCount()>0; print(cp.arange(10).sum().get())' > "$out/cupy-config.txt" 2>&1
fi
python -m pytest tests -q > "$out/tests.txt" 2>&1
python -m terragpu.paper_benchmark --backends "${backends[@]}" --repeat "$repeat" --warmup 2 --output "$out/suite.json"
# Retain Dask as an explicit comparison, with scheduler/scope in each JSON.
for backend in numpy dask; do
  python -m terragpu.benchmark --backend "$backend" --size 4096 --repeat "$repeat" --warmup 2 --output "$out/ndvi-$backend.json"
done
if [[ "$mode" == both ]]; then
  for backend in cupy dask-cupy; do
    python -m terragpu.benchmark --backend "$backend" --size 4096 --repeat "$repeat" --warmup 2 --output "$out/ndvi-$backend.json"
  done
fi
python scripts/export_benchmark_metrics.py "$out"
printf 'Results: %s\n' "$out"
