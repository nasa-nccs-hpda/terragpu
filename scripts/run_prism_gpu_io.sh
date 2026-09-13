#!/usr/bin/env bash
# Existing Slurm allocation, repository root, activated .venv-io-$arch.
set -euo pipefail
out=${1:-results/prism-gpu-io-$(date -u +%Y%m%dT%H%M%SZ)}
work=${TERRAGPU_IO_ROOT:-data/gpu-io}
data_root=${TERRAGPU_DATA_ROOT:-data}
repeat=${TERRAGPU_REPEAT:-5}
if [[ -e "$out" ]]; then echo "Use a fresh results directory: $out" >&2; exit 1; fi
mkdir -p "$out" "$work"
stage=initialization
finish() {
 status=$?
 printf '%s\n' "$status" > "$out/run-status.txt"
 if [[ "$status" -ne 0 ]]; then printf 'FAILED at %s; see %s\n' "$stage" "$out" >&2; fi
}
trap finish EXIT
exec > >(tee "$out/run.log") 2>&1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
stage='environment and source'
python -m pip freeze > "$out/pip-freeze.txt"
git rev-parse HEAD > "$out/git-commit.txt"
git status --short > "$out/git-status.txt"
if [[ -s "$out/git-status.txt" ]]; then echo 'Commit source changes before benchmarking.' >&2; exit 1; fi
lscpu > "$out/lscpu.txt"
nvidia-smi > "$out/nvidia-smi.txt"
python -c 'import cupy as cp; cp.show_config(); assert float(cp.arange(32).sum().get())==496' > "$out/cupy-config.txt" 2>&1
# Mount identity is necessary: the GPU model does not establish GDS support.
if command -v findmnt >/dev/null; then findmnt -T "$work" > "$out/mount.txt" || true; fi
if command -v df >/dev/null; then df -h "$work" > "$out/storage-space.txt" || true; fi
bash scripts/check_prism_gds.sh "$work" > "$out/gdscheck.txt" 2>&1 || true
stage='correctness tests'
if ! python -m pytest tests/test_gpu_io.py tests/test_gpu_io_benchmark.py -q --tb=short --disable-warnings > "$out/tests.txt" 2>&1; then
 tail -60 "$out/tests.txt";exit 1
fi
stage='GPU I/O experiment'
# All scratch payloads are created on work, removed after verification. Results are small.
python -m terragpu.benchmark_gpu_io --work-root "$work" --data-root "$data_root" --tiles 512 1024 --sizes 15 31 --repeat "$repeat" --warmup 1 --output "$out/gpu-io.json"
echo "Results: $out. Inspect skipped_modes and gds_verified; cuFile requested does not establish GDS."
