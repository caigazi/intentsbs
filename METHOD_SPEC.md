# Frozen method direction (development specification)

## Dynamics and layers

For robot `i`:

\[
\dot p_i=v_i,\qquad \dot v_i=u_i/m.
\]

The controller is deliberately separated into four layers:

\[
\text{nominal navigation}
\rightarrow
\text{2-D Intent coordination}
\rightarrow
\text{Intent tracking}
\rightarrow
\text{analytic safety filter}.
\]

The nominal controller is currently LQR but is not the contribution and can in
principle be replaced by PID or another local go-to-goal navigator.

## Two-dimensional Intent

Define the ego goal frame

\[
e_i^\parallel={g_i-p_i\over\|g_i-p_i\|},\qquad
e_i^\perp=R_{90^\circ}e_i^\parallel.
\]

The learned coordination output is continuous:

\[
(\alpha_i,\beta_i)\in[-1,1]^2.
\]

It decodes to a world-frame velocity target

\[
v_i^{\rm target}
=\alpha_i\|v_i^{\rm base}\|e_i^\parallel
+\beta_i v_{\rm lat}e_i^\perp.
\]

Thus `alpha<0` permits retreat, while the sign and magnitude of `beta` encode
the lateral tendency. `(1,0)` is the identity/no-coordination command.

The communicated message is the resulting two-float world velocity, not raw
network features, GNN embeddings, CBF multipliers, or a third congestion scalar.

## Rate limit and tracking

The world Intent is rate limited:

\[
v_i^{I,+}=v_i^I+
\operatorname{ClipBall}(v_i^{\rm target}-v_i^I,
a_{I,\max}\Delta t).
\]

The current LQR tracker follows a short local waypoint

\[
p_i^{\rm wp}=p_i+T_{\rm lookahead}v_i^{I,+}.
\]

This keeps the coordination interface at velocity-intent level rather than
making the learned policy output force or acceleration.

## Local observation and shared policy

Every robot observes only itself and robots inside the 0.50 m sensing disk.
All currently sensed neighbors participate; Top-5 truncation is postponed.
The Student will use shared parameters and permutation-invariant attention or
message aggregation:

\[
q_{ij}=\psi_e(z_{ij}),\quad
w_{ij}=\operatorname{softmax}_j\psi_a(q_{ij}),\quad
q_i=\sum_{j\in\mathcal N_i}w_{ij}\psi_v(q_{ij}),
\]

\[
(\alpha_i,\beta_i)=\tanh\psi_o(z_i,q_i).
\]

The input may include ego goal/velocity/base Intent, relative position and
velocity, neighbor world Intent, distance, closing rate and TTC. It must not
include global maps, global robot IDs, or centralized component labels that are
unavailable at deployment.

## Event-gated synchronous execution

The three concepts are distinct:

1. sensing: who is inside 0.50 m;
2. predictive conflict evidence: TTC/CPA/closing conditions;
3. state: BYPASS, ACTIVE, or RELEASE.

Evidence drives a hysteretic state machine. It is not identical to ACTIVE.
Every ACTIVE robot refreshes Intent on the same 0.03 s control tick. RELEASE
returns the world Intent smoothly to the Base velocity. BYPASS uses Base.

This is spatially distributed but time-synchronous, matching the discrete-time
deployment assumption used by the external GCBF+ baseline.

## Safety backend

The safety backend is Wang--Ames--Egerstedt feasible barrier filtering with
independent local QPs and hybrid Eq. (17) braking. The public evaluation line is
0.20 m; an internal 0.2025 m certificate radius supplies a small sampled margin.

If a local normal QP is infeasible, the emergency braking flag propagates over
the currently sensed connected component because the pairwise braking lemma
requires both endpoints to brake. Eq. (17) is integrated as maximum braking to
zero speed and then zero force, even if stopping occurs inside a 30 ms tick.

The safety layer is infrastructure, not the claimed novelty.

## Scope of SBS

The target is agent-agent safety-induced blocking (A-SBS): robot interactions
cause the safety layer to suppress task progress. Pure static-obstacle local
minima (O-SBS) are outside the first-stage contribution. Mixed cases are delayed
until the obstacle-free A-SBS method is established.

Primary metrics are success, completion-time distribution, cumulative and
per-agent SBS duration, persistent-SBS probability, minimum separation,
certificate violations, braking frequency, communication rate and runtime.
