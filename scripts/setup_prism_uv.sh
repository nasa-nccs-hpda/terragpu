#!/usr/bin/env bash
# Run from the repository root. CUDA mode uses pip-provided CUDA 12 components.
set -euo pipefail
mode=${1:-gpu}
arch=$(uname -m)
case "$arch" in x86_64|aarch64) ;; arm64) arch=aarch64;; *) echo "Unsupported architecture: $arch" >&2; exit 1;; esac
envdir=${2:-.venv-prism-$arch}
case "$mode" in cpu|gpu) ;; *) echo 'Usage: setup_prism_uv.sh [cpu|gpu] [environment-dir]' >&2; exit 2;; esac
# Slurm can inherit a login-node UV_BIN and exported uv() wrapper.
# Resolve the executable on the node that will actually run Python and CUDA.
uv_cmd=${UV_BIN:-$(type -P uv || true)}
case "$uv_cmd" in
  */uv-x86_64/uv|*/uv-aarch64/uv)
    uv_cmd="$(dirname "$(dirname "$uv_cmd")")/uv-$arch/uv" ;;
esac
if [[ -z "$uv_cmd" || ! -x "$uv_cmd" ]]; then
  echo "Set UV_BIN to an executable uv binary for $arch on this node." >&2
  exit 1
fi
if ! "$uv_cmd" --version; then
  echo "uv cannot run on $arch. Check the architecture of UV_BIN on this compute node." >&2
  exit 1
fi
printf 'Node architecture: %s; environment: %s\n' "$arch" "$envdir"
"$uv_cmd" venv --python 3.12 --seed "$envdir"
"$uv_cmd" pip install --python "$envdir/bin/python" -e '.[benchmark,test,parallel,pace,viirs]' 'numpy>=2,<2.5' 'scipy>=1.16,<1.18'
if [[ "$mode" == gpu ]]; then
  "$uv_cmd" pip install --python "$envdir/bin/python" 'cupy-cuda12x[ctk]==14.2.0'
  "$envdir/bin/python" -c 'import cupy as cp; cp.show_config(); x=cp.arange(32,dtype=cp.float32); assert float((x*x).sum().get())==10416.; print("CUDA compute preflight passed")'
fi
"$uv_cmd" pip freeze --python "$envdir/bin/python" > "$envdir/requirements-resolved.txt"
printf 'Activate with: source %s/bin/activate\n' "$envdir"
