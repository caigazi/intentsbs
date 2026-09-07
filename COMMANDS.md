# Commands and authoritative operating procedure

## Local tests

```powershell
python -m unittest discover -s tests -q
```

If `python` is unavailable, use the configured environment or the bundled
Python path explicitly.

## AutoDL environment

```bash
cd /root/autodl-tmp/IntentComm-SBS-Clean
source hpc/autodl_env.sh
which ptxas
python -c "import jax; print(jax.default_backend(), jax.devices())"
```

Always source `hpc/autodl_env.sh` in every new SSH shell before running JAX.
Otherwise the base image may select an old CUDA 11.8 `ptxas` and fail with
`Unsupported .version 8.3; current version is 7.8`.

## Current Gate 3 continuation

Do not rerun these jobs merely to rediscover their results. The current
artifacts are:

```text
artifacts/raw/gate3_student_compatible_global_cem_audit_20260906_v2/REPORT.json
artifacts/raw/gate3_d2_student_train_global_compatible_20260907_v2/REPORT.json
artifacts/raw/gate3_d2_global_compatible_fixed_validation_20260907_v2/REPORT.json
```

The latest fixed validation is 8/16 overall and 4/10 for N=8. The next command
should be a targeted read-only diagnostic over the already saved failed
trajectories, not another Teacher run or blind D3 collection. Failed scene IDs
are recorded in `START_HERE.md` and `PROJECT_STATUS.md`.

If a new checkpoint must later be evaluated on the frozen 16-scene set, use
`run_v2_rollout_collection.py` with N=2--7 seeds 8600--8605 and N=8 seeds
8700--8709. Keep that evaluation in the background with its own output folder,
log and PID.

## Targeted smoke command

```bash
python run_v2_teacher_smoke.py \
  --case n2_headon \
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

The default is authoritative: `4096 x 3 x 4 restarts`, 40 steps and
32-substep JAX x64 exact ranking. Gate 1A transforms the same fixed candidates
and checks exact objective cost.
Gate 1B re-evaluates every cross-seed label under that same exact objective.
Run it in the background because the exact candidate audit may exceed one
minute. Inspect the report only after the user asks for status.

`--debug-quick` is only for code-path debugging or coarse candidate proposals.
Its report is marked non-authoritative and must never decide Gate 1, become a
formal label, or support a paper claim.

An authoritative distillation Teacher is defined by stable low regret across
restarts plus exact shortlist reranking, not by a fixed budget name.

## Short authoritative Teacher audit

This runs a fresh four-restart exact Teacher on a short ACTIVE window, records
restart costs and continuity, and writes no dataset:

```bash
python run_v2_authoritative_teacher_gate.py \
  --case n5_partial \
  --active-ticks 12 \
  --output artifacts/raw/gate1_authoritative_n5
```

## Surrogate fidelity sweep

Replay the exact same saved CEM populations at several JAX integration
resolutions without running a new Teacher search:

```bash
python run_v2_teacher_surrogate_sweep.py \
  --source-report artifacts/raw/gate1_search_n5_strong_32sub_20260903/REPORT.json \
  --substeps 4,8,16,32 \
  --output artifacts/raw/gate1_surrogate_sweep_replay_n5
```

## Sampling/restart coverage sweep

After 32-substep ranking fidelity has passed, compare equal or similar candidate
budgets without repeating exact evaluation for every population member:

```bash
python run_v2_teacher_coverage_sweep.py \
  --case n5_partial \
  --trial-seeds 0,1,2,3,4,5,6,7,8,9 \
  --strategies 12x3x1,24x3x1,12x3x2,48x3x1,12x3x4 \
  --output artifacts/raw/gate1_coverage_n5
```

Every branch winner, incumbent, and cross-restart winner is reranked with the
exact NumPy objective. Run this in the background.

## Background rule for runs over one minute

```bash
source hpc/autodl_env.sh
nohup python run_v2_teacher_smoke.py \
  --case n8_two_stream \
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
