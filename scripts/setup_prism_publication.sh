#!/usr/bin/env bash
set -euo pipefail
arch=$(uname -m)
[[ "$arch" != arm64 ]] || arch=aarch64
envdir=${1:-.venv-publication-$arch}
bash scripts/setup_prism_uv.sh gpu "$envdir"
uv_cmd=${UV_BIN:-$(type -P uv || true)}
case "$uv_cmd" in
 */uv-x86_64/uv|*/uv-aarch64/uv) uv_cmd="$(dirname "$(dirname "$uv_cmd")")/uv-$arch/uv";;
esac
"$uv_cmd" pip install --python "$envdir/bin/python" -e '.[publication]' 'cupy-cuda12x[ctk]==14.2.0' 'numpy>=2,<2.5' 'scipy>=1.16,<1.18'
"$uv_cmd" pip check --python "$envdir/bin/python"
"$uv_cmd" pip freeze --python "$envdir/bin/python" > "$envdir/requirements-resolved.txt"
printf 'Activate: source %s/bin/activate\n' "$envdir"
