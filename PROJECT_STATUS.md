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

Regression result: 18 relevant unit tests passed on AutoDL.

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

## Still unproven

- The trigger definition is still marked development-only.
- Teacher labels have not yet passed the full local-observability,
  equivariance, and distillability audit.
- A newly aligned broad Student has not been trained.
- Generalization, tail-SBS quality, obstacles, larger swarms, and real-robot
  behavior remain pending.
- `sbs824/v2/trigger.py` contains audit-only shadow rollout helpers that still
  use a simple Euler step; they must not silently become part of the frozen
  trigger without matching the repaired hybrid safety execution.

## Discarded evidence

Old per-agent timer runs, incorrect due-mask datasets, 0.21 s intent runs,
100 Hz safety experiments, and datasets generated before sensing/trigger/release
were separated are diagnostic history only. They are not training data or paper
evidence.
