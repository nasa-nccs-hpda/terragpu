#!/usr/bin/env bash
# One invocation per allocated GPU job; run from repository root, environment active.
set -euo pipefail
out=${1:-results/prism-publication-$(date -u +%Y%m%dT%H%M%SZ)}
if [[ $# -gt 0 ]]; then shift;fi
work=${TERRAGPU_IO_ROOT:-data/publication}
data_root=${TERRAGPU_DATA_ROOT:-data}
if [[ -e "$out" ]]; then echo "Use a fresh results directory: $out" >&2;exit 1;fi
mkdir -p "$out" "$work"
stage=initialization
finish() {
 status=$?
 printf '%s\n' "$status" > "$out/run-status.txt"
 if [[ "$status" -ne 0 ]]; then printf 'FAILED at %s; see %s\n' "$stage" "$out" >&2;fi
}
trap finish EXIT
exec > >(tee "$out/run.log") 2>&1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
stage=preflight
git rev-parse HEAD > "$out/git-commit.txt"
git status --short > "$out/git-status.txt"
if [[ -s "$out/git-status.txt" ]]; then echo 'Commit source changes before benchmarking.' >&2;exit 1;fi
python -m pip freeze > "$out/pip-freeze.txt"
lscpu > "$out/lscpu.txt"
nvidia-smi > "$out/nvidia-smi.txt"
findmnt -T "$work" > "$out/mount.txt" || true
df -h "$work" > "$out/storage-space.txt"
python -c 'import cupy as cp; cp.show_config(); assert float(cp.arange(32).sum().get())==496' > "$out/cupy-config.txt" 2>&1
stage=correctness
if ! python -m pytest tests/test_publication_benchmark.py -q --tb=short --disable-warnings > "$out/tests.txt" 2>&1;then
 tail -60 "$out/tests.txt";exit 1
fi
stage='native publication benchmark'
python -m terragpu.publication_benchmark --output "$out/publication.json" --work-root "$work" --data-root "$data_root" --repeat "${TERRAGPU_REPEAT:-5}" --seed "${TERRAGPU_SEED:-731}" --storage-label "${TERRAGPU_STORAGE_LABEL:-unspecified}" "$@"
stage=figures
python -m terragpu.publication_figures "$out/publication.json" --output "$out/figures"
echo "Completed: $out. Archive this directory; scratch payloads were removed."
