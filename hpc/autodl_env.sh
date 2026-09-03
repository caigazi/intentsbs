#!/usr/bin/env bash

# Source this file before any direct JAX command on AutoDL.  The base image may
# put an older system ptxas ahead of the CUDA 12 tool installed by pip.
AUTODL_PTXAS_BIN="$(python - <<'PY'
from importlib.util import find_spec
from pathlib import Path

spec = find_spec("nvidia.cuda_nvcc")
if spec is None or not spec.submodule_search_locations:
    raise SystemExit("nvidia-cuda-nvcc-cu12 is not installed")
package_dir = Path(next(iter(spec.submodule_search_locations))).resolve()
binary_dir = package_dir / "bin"
if not (binary_dir / "ptxas").is_file():
    raise SystemExit(f"ptxas not found under {binary_dir}")
print(binary_dir)
PY
)"
export PATH="${AUTODL_PTXAS_BIN}:${PATH}"
export JAX_ENABLE_X64=True
hash -r
