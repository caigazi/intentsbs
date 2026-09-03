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
- sensing radius is 0.50 m and all sensed neighbors are used for now;
- communication payload is only the two-dimensional world-velocity Intent;
- no static obstacles, Top-5 truncation, or third communication scalar until
  the obstacle-free 2--8 agent learning Gate passes;
- Teacher and Student must execute through the same runtime and safety backend;
- the analytic safety layer remains separate from the learned coordination
  policy.

Long-run rule:

- Any Teacher/data-generation/training/evaluation job expected to exceed one
  minute must be started in the background with a log and PID.
- Report the estimate and return immediately. Do not poll until the user asks.
- A status-only request may inspect only the PID, the report file, and at most
  the last 30 log lines. Do not rescan the project.

Never train from a dataset whose protocol manifest differs from the runtime
manifest. Never promote a development result to a paper claim.
