#!/usr/bin/env bash
# Read-only GDS platform checks on the allocated PRISM node; no benchmark rerun.
set -euo pipefail
work=${1:-data/gpu-io}
uname -srmo
if command -v findmnt >/dev/null; then findmnt -T "$work" || true;fi
if [[ -r /proc/modules ]]; then
 echo 'Relevant loaded kernel modules:'
 awk '$1 ~ /^(nvidia|nvme|ib_|mlx5|rdma|panfs)/ {print $1, $2}' /proc/modules
fi
# Tool installations use both names; CUDA may live outside the default symlink.
gdscheck=$(command -v gdscheck || command -v gdscheck.py || true)
if [[ -z "$gdscheck" ]]; then
 for candidate in "${CUDA_HOME:-/usr/local/cuda}"/gds/tools/gdscheck{,.py} /usr/local/cuda*/gds/tools/gdscheck{,.py}; do
  if [[ -f "$candidate" ]]; then gdscheck="$candidate";break;fi
 done
fi
if [[ -n "$gdscheck" ]]; then
 printf 'Diagnostic tool: %s\n' "$gdscheck"
 if [[ "$gdscheck" == *.py ]]; then python "$gdscheck" -p 2>&1 || true
 else "$gdscheck" -p 2>&1 || true;fi
else printf 'gdscheck/gdscheck.py not found; cuFile/GDS qualification remains unverified.\n';fi
echo 'A platform check does not prove direct I/O on this mount.'
echo 'Use mount-specific read/write testing and cuFile telemetry to qualify GDS.'
