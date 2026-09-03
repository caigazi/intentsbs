# Ordered project Gates

Do not skip a Gate because later experiments are more visually interesting.

## Gate 0 — implementation consistency (closeout)

- Full unit suite passes.
- Dataset, Teacher, Student and evaluation manifests must match exactly.
- One runtime path determines trigger, modes, masks, Intent tracking and safety.
- Remove or repair any audit helper that still uses obsolete safety dynamics.
- Reproduce N=2/N=5/N=8 targeted smoke results with zero certificate violation.

Pass condition: the 33 Hz implementation is internally consistent and the
protocol can be changed from development-only to a versioned frozen protocol.

Current status: the full 31-test suite and N=2/N=5/N=8 safety smoke Gate pass.
Only protocol freeze and residual audit-path cleanup remain; do not reopen the
trigger, safety rate, or two-dimensional Intent design.

## Gate 1 — authoritative Teacher and distillability (next formal Gate)

For selected difficult snapshots:

- first pass Gate 1-Search: build an authoritative offline Teacher whose exact
  cost has stable low regret across restarts;
- treat `12 x 3 x 40` as the historical strong/high budget, not automatic proof
  of authoritative convergence;
- use exact shortlist reranking and audit approximate-versus-exact candidate
  ranks before increasing the sampling budget;
- quick CEM is debugging/proposal infrastructure and cannot determine this Gate;
- repeat Teacher searches with multiple random seeds;
- compare equivalent local observations under rotation/permutation/reflection;
- check that equivalent local observations do not receive contradictory labels;
- quantify branch multimodality and chirality consistency;
- verify high-budget Teacher decisively improves Base on tail A-SBS cases.

Pass condition: Gate 1-Search first establishes a reliable oracle; then Gate
1-Distill shows its labels are locally inferable and stable enough for a shared
distributed Student. If not, repair the relevant stage before data generation.

## Gate 2 — tiny closed-loop overfit

Train only on two or a few strong Teacher trajectories, including N=5
offset-radial and N=8 two-stream tails.

Interpretation:

- high action loss: representation or conflicting-label problem;
- low action loss but failed rollout: covariate shift; use DAgger;
- successful closed loop: Student expressivity is adequate.

## Gate 3 — small generalization

Use roughly 10--20 mixed 2--8-agent scenes with varied topology, seed, speed,
offset and permutation. Keep obstacles absent.

Pass condition: unseen cases retain high success and Student recovers a large
fraction of the Teacher improvement without regressing simple head-on/merge/
chain cases.

## Gate 4 — clean development dataset

Only after Gates 0--3, generate approximately:

- 28 targeted difficult scenes across 2--8 agents;
- 20 random N=8 scenes;
- high-budget labels only near difficult decision windows;
- lower-budget broad coverage elsewhere.

Reject unresolved, unsafe, protocol-mismatched or worse-than-Base Teacher labels.

## Gate 5 — broad Student and failure-driven DAgger

Train with broad data dominant and hard replay limited to roughly 10--20%.
Evaluate first, then relabel only windows around the first causal divergence of
failed Student rollouts. Do not repeatedly replay whole failed episodes.

## Gate 6 — development benchmark

Evaluate old36 and fresh16 separately as a 52-scene development benchmark, not
as a final test set. Compare at minimum:

- Base LQR + identical analytic safety backend;
- IntentComm Student;
- official GCBF+ as an external baseline under clearly disclosed native
  settings.

Report success, mean/P95 completion time, mean/P95 SBS, persistent SBS, minimum
distance and safety violations. Do not claim fair safety-space equivalence if
baseline safety definitions differ.

After checkpoint selection and ablation decisions are complete, generate a new
held-out benchmark for final paper numbers.

## Gate 7 — extensions

Only after the obstacle-free 2--8-agent Gate is solid:

1. scale at fixed density to 16/32/64/128;
2. profile local-neighbor and policy latency;
3. add GCBF+-style non-pathological static obstacle fields;
4. test whether a third nonnegative congestion scalar is needed;
5. progress toward quadrotor tracking and real experiments.
