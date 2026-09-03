"""Batched JAX approximation of the obstacle-free Wang safety step.

This backend is used only to rank offline Teacher candidates.  A selected
candidate is always re-evaluated by :mod:`sbs824.v2.sampled_safety`, which is
the reference NumPy implementation and the deployed hard-safety backend.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


def _boundary_pairs(row_count: int) -> tuple[jax.Array, jax.Array]:
    left, right = np.triu_indices(row_count, k=1)
    return jnp.asarray(left, dtype=jnp.int32), jnp.asarray(
        right, dtype=jnp.int32)


def _batched_polygon_projection(u_ref: jax.Array, rows: jax.Array,
                                bounds: jax.Array, valid: jax.Array,
                                max_force: float) -> tuple[jax.Array, jax.Array]:
    """Project every ``[batch, agent]`` reference onto a 2-D polygon."""
    row_count = rows.shape[-2]
    left, right = _boundary_pairs(row_count)
    clipped = jnp.clip(u_ref, -max_force, max_force)[..., None, :]

    denominator = jnp.sum(rows * rows, axis=-1)
    residual = bounds - jnp.sum(rows * u_ref[..., None, :], axis=-1)
    projection = (u_ref[..., None, :]
                  + residual[..., None] * rows
                  / jnp.maximum(denominator[..., None], 1e-20))
    projection_valid = valid & (denominator > 1e-14)

    row_l, row_r = rows[..., left, :], rows[..., right, :]
    bound_l, bound_r = bounds[..., left], bounds[..., right]
    determinant = (row_l[..., 0] * row_r[..., 1]
                   - row_l[..., 1] * row_r[..., 0])
    safe_det = jnp.where(jnp.abs(determinant) > 1e-12,
                         determinant, 1.0)
    vertex = jnp.stack((
        (bound_l * row_r[..., 1] - row_l[..., 1] * bound_r) / safe_det,
        (row_l[..., 0] * bound_r - bound_l * row_r[..., 0]) / safe_det,
    ), axis=-1)
    vertex_valid = (valid[..., left] & valid[..., right]
                    & (jnp.abs(determinant) > 1e-12))

    candidates = jnp.concatenate((clipped, projection, vertex), axis=-2)
    candidate_valid = jnp.concatenate((
        jnp.ones(valid.shape[:-1] + (1,), dtype=bool),
        projection_valid, vertex_valid), axis=-1)
    values = jnp.einsum("...kd,...md->...mk", rows, candidates)
    feasible = jnp.all(
        (~valid[..., None, :]) | (values >= bounds[..., None, :] - 1e-8),
        axis=-1)
    finite = jnp.all(jnp.isfinite(candidates), axis=-1)
    candidate_valid &= feasible & finite
    objective = jnp.sum(
        (candidates - u_ref[..., None, :]) ** 2, axis=-1)
    objective = jnp.where(candidate_valid, objective, jnp.inf)
    selected_id = jnp.argmin(objective, axis=-1)
    selected = jnp.take_along_axis(
        candidates, selected_id[..., None, None], axis=-2)[..., 0, :]
    hard_feasible = jnp.any(candidate_valid, axis=-1)
    return selected, hard_feasible


@partial(jax.jit, static_argnames=("n_agents",))
def batched_wang_qp(state: jax.Array, u_ref: jax.Array,
                    braking_latch: jax.Array, *, n_agents: int,
                    sense_radius: float, mass: float, max_force: float,
                    dt: float, safe_distance: float, wang_gamma: float
                    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Solve a batch of obstacle-free local Wang QPs.

    Args use a leading candidate dimension: state ``[B,N,4]``, reference force
    ``[B,N,2]`` and latch ``[B,N]``.  The implementation has fixed dense
    shapes so XLA can vectorize candidate and agent dimensions.
    """
    if state.shape[-2:] != (n_agents, 4):
        raise ValueError("state must have shape [batch,n_agents,4]")
    position, velocity = state[..., :2], state[..., 2:]
    delta_p = position[..., :, None, :] - position[..., None, :, :]
    delta_v = velocity[..., :, None, :] - velocity[..., None, :, :]
    distance_sq = jnp.sum(delta_p * delta_p, axis=-1)
    eye = jnp.eye(n_agents, dtype=bool)
    neighbor = (distance_sq <= sense_radius * sense_radius) & (~eye)

    speed = jnp.linalg.norm(velocity, axis=-1)
    speed_i, speed_j = speed[..., :, None], speed[..., None, :]
    velocity_i = velocity[..., :, None, :]
    velocity_j = velocity[..., None, :, :]
    alpha = max_force / mass
    q_i = position[..., :, None, :] + (
        speed_i[..., None] * velocity_i / (4.0 * alpha))
    q_j = position[..., None, :, :] + (
        speed_j[..., None] * velocity_j / (4.0 * alpha))
    radius = (safe_distance
              + speed_i * speed_i / (4.0 * alpha)
              + speed_j * speed_j / (4.0 * alpha))
    q_delta = q_i - q_j
    barrier = jnp.sum(q_delta * q_delta, axis=-1) - radius * radius

    dot_ij = jnp.sum(velocity_i * velocity_j, axis=-1)
    dot_pi = jnp.sum(delta_p * velocity_i, axis=-1)
    safe_speed_i = jnp.maximum(speed_i, 1e-10)
    a_hat = (
        (speed_i * speed_j / (8.0 * alpha * alpha))[..., None] * velocity_j
        + (dot_ij * speed_j /
           (8.0 * alpha * alpha * safe_speed_i))[..., None] * velocity_i
        - (dot_pi / (2.0 * alpha * safe_speed_i))[..., None] * velocity_i
        - (speed_i / (2.0 * alpha))[..., None] * delta_p
        + (safe_distance / alpha) * velocity_i
        + (speed_j * speed_j / (4.0 * alpha * alpha))[..., None]
        * velocity_i)
    a_hat = jnp.where((speed_i <= 1e-10)[..., None], 0.0, a_hat)
    gamma = wang_gamma / safe_distance ** 4
    b_hat = (jnp.sum(delta_p * delta_v, axis=-1)
             + speed_i / (2.0 * alpha)
             * jnp.sum(delta_v * velocity_i, axis=-1)
             + 0.5 * gamma * barrier ** 3)
    pair_rows = -a_hat / mass
    pair_bounds = -b_hat

    box_rows = jnp.asarray(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
        dtype=state.dtype)
    box_rows = jnp.broadcast_to(
        box_rows, state.shape[:-2] + (n_agents, 4, 2))
    box_bounds = jnp.full(
        state.shape[:-2] + (n_agents, 4), -max_force,
        dtype=state.dtype)
    rows = jnp.concatenate((pair_rows, box_rows), axis=-2)
    bounds = jnp.concatenate((pair_bounds, box_bounds), axis=-1)
    valid = jnp.concatenate((neighbor, jnp.ones_like(box_bounds, dtype=bool)),
                            axis=-1)
    control, feasible = _batched_polygon_projection(
        u_ref, rows, bounds, valid, max_force)

    must_brake = ((braking_latch & (speed > 1e-8)) | (~feasible))
    # Eq. (17) is propagated over each currently sensed component.
    for _ in range(n_agents):
        must_brake |= jnp.any(neighbor & must_brake[..., None, :], axis=-1)
    brake_force = (-max_force * velocity
                   / jnp.maximum(speed[..., None], 1e-10))
    brake_force = jnp.where((speed > 1e-10)[..., None], brake_force, 0.0)
    action = jnp.where(must_brake[..., None], brake_force, control)
    next_latch = must_brake & (speed > 1e-8)
    outside_pairs = jnp.sum(
        neighbor & (barrier < -1e-9), axis=(-2, -1)) // 2
    return action, next_latch, must_brake, outside_pairs
