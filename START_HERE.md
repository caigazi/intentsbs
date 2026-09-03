# IntentComm-SBS clean handoff

## Objective

Develop a spatially distributed, time-synchronous coordination layer that
reduces agent-agent safety-induced blocking (A-SBS). Each robot uses local
observations and broadcasts only a two-dimensional velocity Intent. A separate
analytic safety filter enforces collision avoidance.

The contribution is not a new nominal navigator or a learned barrier. It is the
event-gated, low-bandwidth, variable-neighborhood coordination policy and its
training/evaluation framework.

## Current stage

The project is at the end of Gate 0 and still before formal Teacher freeze.

The 33 Hz Eq. (17) braking implementation bug has been fixed and its targeted
safety smoke Gate passed for N=2, N=5, and N=8. The next stage is not large-scale
training. It is Teacher-V2 distillability and quality validation under the
frozen 33 Hz execution protocol. The clean AutoDL clone now passes the complete
31-test suite, and JAX/NumPy Wang-QP parity passed on a V100 GPU.

## Immediate next action

1. Finish the residual audit of the shadow predictor in
   `sbs824/v2/trigger.py` without redesigning the online TTC/CPA trigger.
2. Freeze one protocol manifest shared by dataset, Teacher, Student, and eval.
3. Run the Teacher distillability audit described in `ROADMAP.md`.

Do not generate the 48-scene dataset until those checks pass.

## Authoritative snapshot

The source in this folder was copied directly from the AutoDL working tree
after commit `3be21fa` (`fix Eq17 sampled hybrid braking at 33 Hz`). That commit
exists on AutoDL branch `codex/gpu-teacher-dev` but was not pushed to GitHub
because the server has no GitHub credentials. GitHub/origin was still at
`c7a8e56` when this handoff was created.

## Where to look

- Method definition: `METHOD_SPEC.md`
- Evidence and exact latest metrics: `PROJECT_STATUS.md`
- Ordered research Gates: `ROADMAP.md`
- Reproducible commands: `COMMANDS.md`
- Core implementation: `sbs824/v2/`
- Analytic safety backend: `sbs824/wang_safety.py`
- Regression tests: `tests/test_sync_event_v2.py` and
  `tests/test_wang_safety.py`
