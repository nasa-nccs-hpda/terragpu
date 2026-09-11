#!/usr/bin/env bash
# Run inside an existing salloc session. No partition/account/site paths assumed.
set -euo pipefail
out=${1:-results/prism-$(date -u +%Y%m%dT%H%M%SZ)}
mode=${2:-both}
repeat=${TERRAGPU_REPEAT:-7}
case "$mode" in both) backends=(numpy cupy);; cpu) backends=(numpy);; *) echo 'Mode must be both or cpu' >&2;exit 2;; esac
if [[ -e "$out" ]]; then echo "Use a new output directory: $out" >&2;exit 1;fi
mkdir -p "$out"
stage=initialization
finish() {
  status=$?
  printf '%s\n' "$status" > "$out/run-status.txt"
  if [[ "$status" -ne 0 ]]; then
    printf 'FAILED at stage: %s (exit %s). Logs: %s\n' "$stage" "$status" "$out" >&2
  fi
}
trap finish EXIT
exec > >(tee "$out/run.log") 2>&1
run_logged() {
  stage=$1
  logfile=$2
  shift 2
  printf 'Running: %s\n' "$stage"
  if "$@" > "$out/$logfile" 2>&1; then
    return 0
  else
    status=$?
    printf 'Failed: %s. Last 60 lines of %s:\n' "$stage" "$out/$logfile" >&2
    tail -n 60 "$out/$logfile" >&2
    return "$status"
  fi
}
# A defined single-thread CPU baseline; use a separate experiment for scaling.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
run_logged 'Python environment' pip-freeze.txt python -m pip freeze
run_logged 'CPU hardware' lscpu.txt lscpu
run_logged 'Source revision' git-commit.txt git rev-parse HEAD
run_logged 'Source cleanliness' git-status.txt git status --short
if [[ -s "$out/git-status.txt" ]]; then echo 'Commit or stash source changes before a benchmark run.' >&2;exit 1;fi
if [[ "$mode" == both ]]; then
  run_logged 'NVIDIA driver' nvidia-smi.txt nvidia-smi
  run_logged 'CUDA computation' cupy-config.txt python -c 'import cupy as cp; cp.show_config(); assert cp.cuda.runtime.getDeviceCount()>0; print(cp.arange(10).sum().get())'
fi
run_logged 'Correctness tests' tests.txt python -m pytest tests -q
stage='Public-data benchmark suite'
printf 'Running: %s\n' "$stage"
python -m terragpu.paper_benchmark --backends "${backends[@]}" --repeat "$repeat" --warmup 2 --output "$out/suite.json"
# Retain Dask as an explicit comparison, with scheduler/scope in each JSON.
for backend in numpy dask; do
  stage="NDVI benchmark: $backend"
  python -m terragpu.benchmark --backend "$backend" --size 4096 --repeat "$repeat" --warmup 2 --output "$out/ndvi-$backend.json"
done
if [[ "$mode" == both ]]; then
  for backend in cupy dask-cupy; do
    stage="NDVI benchmark: $backend"
    python -m terragpu.benchmark --backend "$backend" --size 4096 --repeat "$repeat" --warmup 2 --output "$out/ndvi-$backend.json"
  done
fi
stage='Metric export'
python scripts/export_benchmark_metrics.py "$out"
printf 'Results: %s\n' "$out"
