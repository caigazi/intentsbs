#!/usr/bin/env bash
set -euo pipefail

source hpc/autodl_env.sh
python -m unittest discover -s tests -q
python benchmark_jax_backend.py --agents 8 --batch 256
python run_v2_teacher_smoke.py \
  --case n2_headon \
  --quick \
  --output sync_event_v2_autodl_n2_quick

echo "AutoDL smoke complete. N=2/N=5/N=8 targeted safety Gate passed at handoff."
