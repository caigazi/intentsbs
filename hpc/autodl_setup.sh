#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip
python -m pip install -r requirements-autodl.txt

source hpc/autodl_env.sh

echo "ptxas: $(command -v ptxas)"
ptxas --version

python - <<'PY'
import jax
print("backend:", jax.default_backend())
print("devices:", jax.devices())
print("x64:", jax.config.x64_enabled)
if jax.default_backend() != "gpu":
    raise SystemExit("JAX did not find the AutoDL GPU")
PY
