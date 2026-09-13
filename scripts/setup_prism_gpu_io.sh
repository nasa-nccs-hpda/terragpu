#!/usr/bin/env bash
# Run on the allocated node; keep this experiment separate from the paper env.
set -euo pipefail
arch=$(uname -m)
[[ "$arch" != arm64 ]] || arch=aarch64
envdir=${1:-.venv-io-$arch}
bash scripts/setup_prism_uv.sh gpu "$envdir"
uv_cmd=${UV_BIN:-$(type -P uv || true)}
case "$uv_cmd" in
  */uv-x86_64/uv|*/uv-aarch64/uv) uv_cmd="$(dirname "$(dirname "$uv_cmd")")/uv-$arch/uv";;
esac
"$uv_cmd" pip install --python "$envdir/bin/python" -e '.[gpu-io]' 'cupy-cuda12x[ctk]==14.2.0' 'numpy>=2,<2.5' 'scipy>=1.16,<1.18'
"$envdir/bin/python" -c 'import cupy,kvikio; print("CuPy",cupy.__version__,"KvikIO",kvikio.__version__); assert float(cupy.arange(32).sum().get())==496'
"$uv_cmd" pip freeze --python "$envdir/bin/python" > "$envdir/requirements-resolved.txt"
printf 'Activate inside srun: source %s/bin/activate\n' "$envdir"
