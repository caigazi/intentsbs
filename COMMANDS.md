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
