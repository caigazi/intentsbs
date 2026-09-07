# IntentComm-SBS clean handoff

## Objective

Develop a spatially distributed, time-synchronous coordination layer that
reduces agent-agent safety-induced blocking (A-SBS). Each robot uses a shared
local policy, observes every neighbor inside 0.50 m, and broadcasts only a
two-dimensional world-velocity Intent. A separate analytic Wang safety filter
enforces collision avoidance.

The contribution is the event-gated, low-bandwidth, variable-neighborhood
coordination policy and its training/evaluation framework. It is not a new
nominal navigator, learned barrier, or centralized deployed controller.

## Current stage

Gate 0 is complete. Representative Teacher search coverage and the audited
Student-compatible label construction are complete enough for development
training. Tiny/small Student training has reached Gate 3, but Gate 3 has not
passed.

The latest checkpoint is the D2 shared GNN trained from 771 labels after
globally constraining identical legal local observations to receive identical
actions. Its fixed 16-scene development result is 8/16 overall and 4/10 for
N=8. This recovers from D1 (5/16, N=8 3/10) but remains below D0 (9/16, N=8
6/10). All latest validation trajectories remain above the public 0.20 m hard
center distance; the minimum is 0.201231 m.

The active problem is therefore small-data cross-scene regression/coverage in
the shared Student, not CEM search failure and not unresolved exact-input label
contradiction. Do not reopen Teacher architecture or add communication fields
without new evidence.

## Immediate next action

1. Use the existing D0/D1/D2 fixed-validation reports and saved trajectories to
   diagnose the eight D2 failures. Start with persistent failures N=4 seed 8602
   and N=5 seed 8603, then N=8 seeds 8701/8703/8704/8706/8707/8709.
2. Separate ordinary on-policy coverage gaps from cross-scene forgetting or
   loss imbalance. Do not launch another blind DAgger round first.
3. Then run a small, GCBF+-aligned training-pipeline experiment: multi-N
   plumbing coverage, N=8 as the main training anchor, shared-policy replay
   balance, and failure-window collection rather than whole-episode patches.
4. Gate 3 must pass before any formal 48-scene dataset, obstacles, third scalar,
   larger swarm scaling, or paper claim.

## Frozen choices

- control and ACTIVE Intent update: 0.03 s synchronous ticks;
- hard center distance: 0.20 m; internal certificate: 0.2025 m;
- sensing radius: 0.50 m; every sensed neighbor is used; no Top-5;
- communication: only two-dimensional world-velocity Intent;
- shared permutation-invariant local Student; no robot IDs;
- analytic safety remains separate from learned coordination;
- no obstacles until the obstacle-free 2--8 agent learning Gate passes;
- benchmark conventions and comparable settings follow official GCBF+ where
  the method permits.

## Current evidence and artifacts

- latest training report:
  `artifacts/raw/gate3_d2_student_train_global_compatible_20260907_v2/REPORT.json`
- latest fixed validation:
  `artifacts/raw/gate3_d2_global_compatible_fixed_validation_20260907_v2/REPORT.json`
- global Student-compatible CEM audit:
  `artifacts/raw/gate3_student_compatible_global_cem_audit_20260906_v2/REPORT.json`
- D1 and D2 authoritative label summaries:
  `artifacts/raw/gate3_d1_authoritative_labels_20260905_v1/REPORT.json` and
  `artifacts/raw/gate3_d2_authoritative_labels_20260906_v1/REPORT.json`

These large/raw artifacts live on AutoDL and are not necessarily stored in Git.
Git contains the implementation, tests, protocol, and handoff documentation.

## Where to look

- exact latest evidence: `PROJECT_STATUS.md`
- frozen method: `METHOD_SPEC.md`
- ordered Gates: `ROADMAP.md`
- reproducible commands: `COMMANDS.md`
- core implementation: `sbs824/v2/`
- regression tests: `tests/`
