# Prompt for the new Codex task

Use this folder as the only implementation workspace for IntentComm-SBS. First
read `AGENTS.md`, `START_HERE.md`, `PROJECT_STATUS.md`, `METHOD_SPEC.md`, and
`ROADMAP.md` in that order. The AutoDL checkout is the latest experimental
source; do not use legacy folders or old conversation TXT files.

We are at Gate 3 small generalization, not Teacher design. The latest globally
Student-compatible D2 checkpoint uses 771 labels and scores 8/16 overall and
4/10 at N=8 on the frozen development set. D1 scored 5/16 and D0 scored 9/16.
Exact equivalent-input label conflicts above 0.1 have been eliminated; the
remaining blocker is small-data cross-scene regression/coverage in the shared
GNN Student.

Continue by diagnosing the existing eight failed D2 trajectories, especially
the persistent N=4 seed 8602 and N=5 seed 8603 failures. Distinguish on-policy
coverage gaps from replay/loss imbalance before collecting more labels. Do not
restart broad Teacher work, invent fixed beta templates, add robot IDs/history,
add a third communication scalar, add obstacles, or blindly launch D3.

Keep 0.03 s synchronous control, 0.20 m public hard distance, 0.50 m sensing,
all sensed neighbors, a two-float world-velocity Intent, one shared
permutation-invariant local policy, and a separate analytic safety layer.
Follow GCBF+ conventions where compatible; small-pipeline validation may cover
multiple N, while formal main training remains anchored at N=8.

For any Teacher generation, training, or evaluation expected to exceed one
minute, start it in the background with a log and PID, report the estimate, and
return without polling.
