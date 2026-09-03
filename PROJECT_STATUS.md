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
- The clean GitHub/AutoDL clone passes all 31 tests. Its V100 JAX backend has
  x64 enabled and matches the NumPy Wang-QP reference to numerical precision.
- Gate 0 is in protocol-freeze and residual-audit closeout, not broad redesign.

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

Regression result: all 31 unit tests passed on the clean AutoDL clone; the 18
V2/Wang-specific tests are included in that total.

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
initialization coverage, not surrogate ranking. Gate 1-Search is substantially
improved but does not yet pass its cross-restart low-regret requirement.

## Still unproven

- The trigger definition is still marked development-only.
- Teacher labels have not yet passed the full local-observability,
  equivariance, and distillability audit.
- A newly aligned broad Student has not been trained.
- Generalization, tail-SBS quality, obstacles, larger swarms, and real-robot
  behavior remain pending.
- `sbs824/v2/trigger.py` contains audit-only shadow rollout helpers. They must
  remain diagnostic/ablation code and share the repaired sampled safety
  executor rather than silently becoming a second online runtime.

The online TTC/CPA predictor is early interaction evidence, not the offline SBS
definition. A successful intervention may therefore have predictive evidence
without a subsequently measured SBS event.

## Discarded evidence

Old per-agent timer runs, incorrect due-mask datasets, 0.21 s intent runs,
100 Hz safety experiments, and datasets generated before sensing/trigger/release
were separated are diagnostic history only. They are not training data or paper
evidence.
