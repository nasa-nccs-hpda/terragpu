#!/usr/bin/env bash
# Run from the repository root. CUDA mode uses pip-provided CUDA 12 components.
set -euo pipefail
mode=${1:-gpu}
envdir=${2:-.venv-prism}
case "$mode" in cpu|gpu) ;; *) echo 'Usage: setup_prism_uv.sh [cpu|gpu] [environment-dir]' >&2; exit 2;; esac
command -v uv >/dev/null || { echo 'Install uv first: https://docs.astral.sh/uv/getting-started/installation/' >&2; exit 1; }
uv venv --python 3.12 --seed "$envdir"
uv pip install --python "$envdir/bin/python" -e '.[benchmark,test,parallel,pace,viirs]' 'numpy>=2,<2.5' 'scipy>=1.16,<1.18'
if [[ "$mode" == gpu ]]; then
  uv pip install --python "$envdir/bin/python" 'cupy-cuda12x[ctk]==14.2.0'
  "$envdir/bin/python" -c 'import cupy as cp; cp.show_config(); x=cp.arange(32,dtype=cp.float32); assert float((x*x).sum().get())==10416.; print("CUDA compute preflight passed")'
fi
uv pip freeze --python "$envdir/bin/python" > "$envdir/requirements-resolved.txt"
printf 'Activate with: source %s/bin/activate\n' "$envdir"
