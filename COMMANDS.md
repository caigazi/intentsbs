# Commands and low-cost operating procedure

## Local tests

```powershell
python -m unittest discover -s tests -q
```

If `python` is unavailable, use the configured environment or the bundled
Python path explicitly.

## AutoDL environment

```bash
cd /root/autodl-tmp/sbs
source hpc/autodl_env.sh
which ptxas
python -c "import jax; print(jax.default_backend(), jax.devices())"
```

Always source `hpc/autodl_env.sh` in every new SSH shell before running JAX.
Otherwise the base image may select an old CUDA 11.8 `ptxas` and fail with
`Unsupported .version 8.3; current version is 7.8`.

## Targeted smoke command

```bash
python run_v2_teacher_smoke.py \
  --case n2_headon \
  --quick \
  --output smoke_n2
```

Cases are `n2_headon`, `n5_partial`, and `n8_two_stream`.

## Teacher distillability development audit

This writes diagnostics only and never emits a formal training dataset:

```bash
python run_v2_teacher_distillability.py \
  --case n5_partial \
  --teacher-seeds 0,1,2,3,4 \
  --candidate-count 32 \
  --output audit_n5_high
```

The default is the historical strong/high `12 x 3 x 40` Teacher budget. It has
demonstrated control ability but is not automatically authoritative for
distillation. Gate 1A transforms the same fixed candidates and checks exact
objective cost.
Gate 1B re-evaluates every cross-seed label under that same exact objective.
Run it in the background because the exact candidate audit may exceed one
minute. Inspect the report only after the user asks for status.

`--debug-quick` is only for code-path debugging or coarse candidate proposals.
Its report is marked non-authoritative and must never decide Gate 1, become a
formal label, or support a paper claim.

An authoritative distillation Teacher is defined by stable low regret across
restarts plus exact shortlist reranking, not by a fixed budget name.

## Background rule for runs over one minute

```bash
source hpc/autodl_env.sh
nohup python run_v2_teacher_smoke.py \
  --case n8_two_stream \
  --quick \
  --output smoke_n8 \
  > smoke_n8.log 2>&1 &
echo $!
```

Return to the user immediately with the PID and estimate.

## Patrol-only status check

```bash
pgrep -af 'run_v2_teacher_smoke.py'
test -f smoke_n8/REPORT.json && \
python -c "import json; d=json.load(open('smoke_n8/REPORT.json')); print({k:d[k] for k in ('termination','success','seconds','wall_seconds','minimum_center_distance','sbs_agent_seconds','out_of_certificate_pair_samples')})"
tail -n 30 smoke_n8.log
```

Do not scan source files during a patrol-only request.
