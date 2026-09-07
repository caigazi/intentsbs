# Project status at handoff

## What is established

- A continuous per-agent two-dimensional Intent `(alpha, beta)` can resolve
  representative A-SBS cases when a sufficiently strong rolling Teacher finds
  the actions.
- A shared local policy with variable-neighbor aggregation is the intended
  Student; network parameters do not grow with swarm size.
- Teacher and Student now use a fixed-rate synchronous 0.03 s update protocol.
- Sensing, event evidence, ACTIVE/RELEASE/BYPASS state, and public-tick updates
  are separate concepts in the implementation.
- The analytic Wang safety backend remains outside the learned policy.
- The frozen `sync_event_v2` manifest is the single protocol source for runtime,
  Teacher, diagnostics, and future datasets/checkpoints. Manifest mismatch is
  rejected explicitly.
- The shadow progress diagnostic uses the shared sampled Wang executor and all
  neighbors inside the 0.50 m sensing disk; TTC/CPA only selects candidate egos.
- The clean AutoDL continuation passes all 53 tests. Its V100 JAX backend has
  x64 enabled and matches the NumPy Wang-QP reference to numerical precision.
- Gate 0 and representative frozen-snapshot Gate 1-Search coverage are complete.
- Tiny/small shared-Student training is now at Gate 3. The current checkpoint
  has improved over D1 but has not passed the fixed development benchmark.

## Current Student-training result

The development sequence is intentionally small and is not a formal dataset or
paper result:

| Round | Training labels | Fixed success | N=8 success | Fit RMSE | Beta sign |
|---|---:|---:|---:|---:|---:|
| D0 | 142 | 9/16 | 6/10 | 0.0049 | not recorded here |
| D1 | 426 | 5/16 | 3/10 | 0.0587 | 96.95% |
| D2 global-compatible | 771 | 8/16 | 4/10 | 0.04382 | 98.83% |

D1 exposed a real label-realizability defect: the centralized component
Teacher could assign different actions to agents whose complete legal local
Student inputs were identical. Fifty exact-equivalence conflict pairs were
found, with maximum action separation 0.763. Simple label averaging was
rejected because only 5/19 conflicting states remained inside the strict 1%
quality band.

The replacement searches the exact 32-substep objective directly under the
shared-policy constraint. The first implementation incorrectly imposed the
constraint only inside each Teacher connected component. The corrected global
audit binds identical local observations anywhere in the swarm. It passed all
19/19 audited states (N=6: 6/6; N=8: 13/13), with worst
improvement-normalized regret -0.002965 and a hard zero-gap assertion for equal
inputs.

After those compatible labels were used, the 771-label D2 dataset contained no
exact-equivalence class with action separation above 0.1. Three small classes
remained above 0.03, with maximum gap 0.0601. Training improved substantially
over D1, but a few non-equivalence outliers still had large continuous-action
error, and closed-loop performance did not recover the D0 result.

The latest fixed validation passed N=2, N=3, N=6, N=7 and four N=8 scenes. It
failed N=4 seed 8602 and N=5 seed 8603, plus N=8 seeds 8701, 8703, 8704, 8706,
8707 and 8709. Seven failures ended at a certified-brake/safety-invalid event;
seed 8707 reached the step limit. The public 0.20 m hard distance was preserved
in all 16 cases; the minimum was 0.201231 m.

Interpretation: CEM coverage and exact-input contradictory labels are no longer
the active blocker. The remaining blocker is small-data cross-scene
regression/coverage in one shared Student. Diagnose the saved failed rollouts
before another DAgger collection round. Do not infer that more labels alone
will fix the result.

## Latest safety bug and repair

The old sampled Eq. (17) code reduced braking force so that a low-speed robot
would coast to zero exactly at the end of a 30 ms tick. Wang Eq. (17) instead
requires maximum braking until zero velocity, followed by zero force. The old
implementation lengthened the stopping segment and invalidated the certificate.

Captured N=8 failure before the repair:

- initial minimum center distance: 0.203990 m;
- initial minimum Wang barrier: +0.0001707;
- first barrier crossing: 5.625 ms into the tick;
- end distance: 0.201526 m;
- out-of-certificate samples: 27.

The identical state after exact hybrid braking repair:

- all eight agents stop before the tick ends;
- minimum center distance: 0.203079 m;
- final minimum barrier: +0.0002349;
- out-of-certificate samples: 0.

Regression result after Gate 0 closeout: all 39 unit tests passed on the clean
AutoDL clone.

## 33 Hz targeted Teacher smoke results

All use quick/light CEM, so they validate execution and safety, not optimal
Teacher performance.

| Case | Result | Sim time | Min distance | SBS agent-s | Brake agent-steps | Certificate violations | Wall time |
|---|---:|---:|---:|---:|---:|---:|---:|
| N=2 head-on | success | 7.08 s | 0.25544 m | 1.23 | 2 | 0 | 20.27 s |
| N=5 partial activation | success | 7.41 s | 0.25098 m | 1.05 | 41 | 0 | 51.62 s |
| N=8 two-stream | success | 13.56 s | 0.23241 m | 2.64 | 186 | 0 | 342.34 s |

Interpretation: the specific sampled braking defect is fixed and the targeted
33 Hz safety smoke Gate passes. N=8 taking 13.56 s must not be presented as a
quality result because this was the quick Teacher budget.

Quick CEM remains useful for execution smoke and coarse proposals, but it is
not authoritative evidence about Teacher distillability. The historical
`12 x 3 x 40` strong/high budget proves that good control can be found in some
runs, but the first N=5 fixed-snapshot audit showed that it does not yet have
stable low regret across seeds. Authoritative status must be earned by
convergence and exact-reranking evidence rather than a budget label.

## Candidate-ranking fidelity result

On the saved 360-candidate N=5 population audit, the 4-substep JAX surrogate
had mean rank correlation 0.469 and selected the exact population winner in
only 10/30 populations. At 32 substeps, rank correlation was 1.0, all 30/30
population winners matched the NumPy exact reference, median winner regret was
zero, and V100 steady time for all 360 candidates was 0.033 s versus 0.0155 s
at four substeps. Formal strong-Teacher candidate ranking therefore uses 32
substeps; four-substep ranking is rejected.

Repeating the same five-seed `12 x 3 x 40` search with 32-substep ranking
eliminated ranking loss completely: all 30/30 population winners matched exact
evaluation, and every seed selected its own best sampled candidate. Exact
selected costs were 4634.63, 5853.64, 4606.99, 4623.58, and 4231.72. Four of
five seeds finished within 9.6% of the global sampled best, while seed 1
remained 38.3% worse. The remaining failure is therefore sampling/
initialization coverage, not surrogate ranking.

Subsequent exact-ranking coverage established the authoritative search floor:
N=2 passed 20/20 seeds with `4096 x 3`; N=5 passed 50/50 seeds with
`4096 x 3`; and the two-component N=8 snapshot passed 50/50 fresh seeds with
`4096 x 3 x 2 restarts`. The worst N=8 improvement-normalized regret was
0.00534. This closes representative frozen-snapshot search coverage, not the
whole-trajectory or Distillability Gates.

## Online Teacher scheduling integration result

A 16-ACTIVE-tick N=5 audit exposed the stale runtime scheduler. The historical
`6 x 1` light update failed at 4/5 search ticks. Raising light to `4096 x 3`
still failed 2/5 because the old CEM distribution was trapped in a stale basin.
Cold periodic/component-change searches reduced this to 1/5; two cold restarts
passed 5/5 with maximum regret 0.00526 and reduced maximum adjacent action jump
from 1.086 to 0.392.

The same development strategy had no substantive quality failure in the
16-tick N=2 and N=8 audits. N=8 nevertheless showed two chirality flips during
a component change. A later stratified whole-trajectory attempt showed that
fixed three-tick reuse still degraded labels and was stopped before N=8.

The tiny/Distillability `AuthoritativeComponentTeacher` is therefore deliberately
simple: every ACTIVE tick performs a fresh `4096 x 3 x 4` search with 32-substep
exact ranking. Within the 1% exact quality band it chooses the actual restart
action closest to previous Student-visible Intent. It has no light search,
reuse, stale distribution, adaptive scheduling or quick fallback. The next
check is a short N=2/N=5/N=8 quality/continuity audit, not another exhaustive
Teacher-vs-Teacher trajectory.

## Still unproven

- Gate 3 small generalization has not passed; the latest result is 8/16 overall
  and 4/10 for N=8.
- The remaining failed-rollout cause has not yet been separated into on-policy
  coverage, replay/loss imbalance, or limited small-model continuous-action fit.
- The current 771-label run is a development pipeline result, not a formal
  dataset, broad checkpoint, or paper claim.
- Full permutation/rotation/reflection claims still require a clean formal
  evaluation even though the deployed architecture is permutation invariant.
- Generalization, tail-SBS quality, obstacles, larger swarms, and real-robot
  behavior remain pending.
- `sbs824/v2/trigger.py` contains audit-only shadow rollout helpers. They remain
  diagnostic/ablation code and are not a second online trigger/runtime.

The online TTC/CPA predictor is early interaction evidence, not the offline SBS
definition. A successful intervention may therefore have predictive evidence
without a subsequently measured SBS event.

## Discarded evidence

Old per-agent timer runs, incorrect due-mask datasets, 0.21 s intent runs,
100 Hz safety experiments, and datasets generated before sensing/trigger/release
were separated are diagnostic history only. They are not training data or paper
evidence.

Reports produced under the earlier `sync_event_v2_dev` manifest remain
diagnostic evidence only. They must not be relabeled as frozen-protocol training
data even where the numerical dynamics are unchanged.
