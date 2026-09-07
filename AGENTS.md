# Codex working contract

This directory is the clean continuation point for the IntentComm-SBS project.

Before acting, read these files in order:

1. `START_HERE.md`
2. `PROJECT_STATUS.md`
3. `METHOD_SPEC.md`
4. `ROADMAP.md`

Do not scan `../sbs824`, old conversation TXT files, legacy result folders, or
archived scripts unless the user explicitly asks for historical evidence.
Treat this directory as the only implementation source.

Non-negotiable current choices:

- double-integrator control step is 0.03 s (33.3 Hz);
- Intent is updated on the same synchronous step while an agent is ACTIVE;
- do not reintroduce the rejected 0.21 s or 100 Hz experiments;
- public hard center distance is 0.20 m;
- sensing radius is 0.50 m and all sensed neighbors are used;
- communication payload is only the two-dimensional world-velocity Intent;
- do not introduce Top-5 truncation; the deployed policy uses every sensed
  neighbor;
- no static obstacles or third communication scalar until the obstacle-free
  2--8 agent learning Gate passes;
- benchmark definitions, scene conventions, metrics and comparable parameters
  should follow official GCBF+ conventions wherever the method permits;
- Teacher and Student must execute through the same runtime and safety backend;
- the analytic safety layer remains separate from the learned coordination
  policy.

The formal `sbs824/v2` path must not import legacy `train_*`, `phase*`,
`rolling_cem_*`, round-specific, or archived experiment code. It may reuse only
explicitly separated infrastructure such as `simulation`, `spatial`, and
`wang_safety`.

Long-run rule:

- Any Teacher/data-generation/training/evaluation job expected to exceed one
  minute must be started in the background with a log and PID.
- Report the estimate and return immediately. Do not poll until the user asks.
- A status-only request may inspect only the PID, the report file, and at most
  the last 30 log lines. Do not rescan the project.

Never train from a dataset whose protocol manifest differs from the runtime
manifest. Never promote a development result to a paper claim.

Current Student-development rule:

- the project is at Gate 3, not Teacher redesign;
- the latest D2 fixed result is 8/16 overall and 4/10 at N=8;
- exact equivalent-input label conflicts above 0.1 have been eliminated by the
  global Student-compatible constrained-CEM audit;
- diagnose the existing failed D2 trajectories before starting D3;
- small-pipeline checks may cover N=2--8, but formal main training is anchored
  at N=8 and should use balanced broad/hard replay;
- do not add IDs, fixed beta templates, history, multimodal heads, obstacles, or
  a third communication scalar without evidence that the current legal local
  observation and shared GNN are insufficient.

Teacher-budget rule:

- `12 samples x 3 iterations x 40 steps` is the historical strong/high budget;
  it proves that good control can be found in some runs but is not automatically
  an authoritative distillation Teacher;
- authoritative status requires stable low regret across restarts and exact
  shortlist reranking;
- formal labels must use `AuthoritativeComponentTeacher`, with at least
  `4096 samples x 3 iterations x 4 independent cold restarts`; the current
  tiny/Distillability Teacher fresh-searches every ACTIVE tick;
- the authoritative path has no light CEM, single-restart mode, approximate
  ranking, quick fallback, NumPy fallback, or reuse;
- authoritative candidate ranking uses 32 sampled-safety integration substeps;
  the rejected 4-substep surrogate must not determine a Teacher label;
- quick CEM is only a debugging or coarse-proposal accelerator and must never
  decide Teacher stability, enter a formal dataset, or support a paper claim.
