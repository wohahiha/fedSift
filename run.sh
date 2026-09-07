#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME="${FEDSIFT_RUNTIME:-${XDG_CACHE_HOME:-$HOME/.cache}/FedSift/runtime}"
EXPECTED="$(cut -d ' ' -f 1 "$PROJECT_ROOT/environment/runtime_archive.sha256")"
if [[ ! -f "$RUNTIME/.runtime-ready" ]]; then
  if [[ -d "$RUNTIME" && -n "$(ls -A "$RUNTIME")" ]]; then
    echo 'Runtime directory is not empty. Set FEDSIFT_RUNTIME to an empty directory.' >&2
    exit 2
  fi
  (cd "$PROJECT_ROOT/environment" && sha256sum --check runtime_parts.sha256)
  PARTS=("$PROJECT_ROOT"/environment/runtime.part*)
  ACTUAL="$(cat "${PARTS[@]}" | sha256sum | cut -d ' ' -f 1)"
  [[ "$ACTUAL" == "$EXPECTED" ]] || { echo 'Runtime archive checksum differs.' >&2; exit 2; }
  mkdir -p "$RUNTIME"
  cat "${PARTS[@]}" | tar -xz -C "$RUNTIME"
  "$RUNTIME/bin/python" -B "$RUNTIME/bin/conda-unpack"
  printf '%s\n' "$EXPECTED" > "$RUNTIME/.runtime-ready"
fi
[[ "$(cat "$RUNTIME/.runtime-ready")" == "$EXPECTED" ]] || { echo 'Runtime identity differs.' >&2; exit 2; }
export FEDSIFT_RUNTIME_PREFIX="$RUNTIME"
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE CUDA_VISIBLE_DEVICES=''
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/scripts:$PROJECT_ROOT"
cd "$PROJECT_ROOT"
exec "$RUNTIME/bin/python" -B "$PROJECT_ROOT/run.py" "$@"
