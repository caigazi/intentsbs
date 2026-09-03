# Prompt for the new Codex task

Use this folder as the only workspace for the IntentComm-SBS project. First read
`AGENTS.md`, `START_HERE.md`, `PROJECT_STATUS.md`, `METHOD_SPEC.md`, and
`ROADMAP.md`. Do not scan the old `../sbs824` folder or historical conversation
TXT files unless I explicitly request it.

Continue from Gate 0: verify that every safety/trigger audit path uses the
repaired 33 Hz hybrid Eq. (17) dynamics, then perform the Teacher-V2
distillability audit. Do not generate the 48-scene dataset or train the broad
Student until that Gate passes. Keep the method at 33.3 Hz, public hard distance
0.20 m, sensing radius 0.50 m, all sensed neighbors, two-float world-velocity
Intent, no obstacles and no third communication scalar.

For any Teacher generation, training, or evaluation expected to exceed one
minute, start it in the background, report PID/log/estimated duration, and end
the turn without polling. For status-only requests, inspect only PID, compact
metrics, and at most the last 30 log lines.
