"""Compiled, batched candidate rollouts for the offline CEM Teacher.

This is a ranking backend and has no authority to emit a training label: the
selected branch is replayed through the NumPy reference runtime before it can
be accepted. The authoritative configuration uses the same 32 integration
substeps as that reference replay.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

# The Wang feasibility decision is discontinuous: a float32 round-off can turn
# one nearly-active half-space from feasible into certified braking for the
# remainder of the horizon.  Teacher ranking therefore uses x64.  The selected
# label is still replayed by the NumPy reference backend.
jax.config.update("jax_enable_x64", True)

from .jax_safety import batched_wang_qp


def _minimum_center(state: jax.Array, n_agents: int) -> jax.Array:
    delta = state[..., :, None, :2] - state[..., None, :, :2]
    distance = jnp.linalg.norm(delta, axis=-1)
    distance += jnp.eye(n_agents, dtype=state.dtype) * 1e6
    return jnp.min(distance, axis=(-2, -1))


@partial(jax.jit, static_argnames=(
    "n_agents", "horizon_steps", "integration_substeps"))
def batched_candidate_costs(
        state0: jax.Array, world0: jax.Array, latch0: jax.Array,
        previous_force0: jax.Array, parameters: jax.Array,
        previous_parameters: jax.Array, goals: jax.Array, gain: jax.Array,
        active_mask: jax.Array, release_mask: jax.Array,
        focus_mask: jax.Array, *, n_agents: int, horizon_steps: int,
        integration_substeps: int, runtime_step: int, dt: float, mass: float,
        sense_radius: float, max_force: float, max_speed: float,
        safe_distance: float, hard_distance: float, wang_gamma: float,
        lateral_speed: float, max_intent_accel: float,
        intent_lookahead: float, action_smooth_weight: float) -> jax.Array:
    """Return one approximate objective value per candidate."""
    batch = parameters.shape[0]
    state = jnp.broadcast_to(state0, (batch, n_agents, 4))
    world = jnp.broadcast_to(world0, (batch, n_agents, 2))
    latch = jnp.broadcast_to(latch0, (batch, n_agents))
    previous_force = jnp.broadcast_to(
        previous_force0, (batch, n_agents, 2))
    goals_b = jnp.broadcast_to(goals, (batch, n_agents, 2))
    controlled = active_mask | release_mask
    controlled_b = jnp.broadcast_to(controlled, (batch, n_agents))
    active_b = jnp.broadcast_to(active_mask, (batch, n_agents))
    focus = jnp.asarray(focus_mask, dtype=state.dtype)
    initial_distance = jnp.linalg.norm(
        state[..., :2] - goals_b, axis=-1)
    initial_distance = jnp.sum(initial_distance * focus, axis=-1)
    initial_minimum = _minimum_center(state, n_agents)

    def body(carry, horizon_index):
        (state, world, latch, previous_force, distance_integral, brake,
         outside, minimum) = carry
        goal_state = jnp.concatenate((
            goals_b, jnp.zeros((batch, n_agents, 2), dtype=state.dtype)),
            axis=-1)
        error = goal_state - state
        error_norm = jnp.linalg.norm(error, axis=-1, keepdims=True)
        error_max = jnp.abs(
            error / jnp.maximum(error_norm, 1e-9) * sense_radius)
        base = jnp.clip(
            jnp.einsum("bnk,dk->bnd", jnp.clip(error, -error_max, error_max),
                       gain),
            -max_force, max_force)
        base_velocity = jnp.clip(
            state[..., 2:] + base / mass * dt, -max_speed, max_speed)

        goal_delta = goals_b - state[..., :2]
        forward = goal_delta / jnp.maximum(
            jnp.linalg.norm(goal_delta, axis=-1, keepdims=True), 1e-9)
        side = jnp.stack((-forward[..., 1], forward[..., 0]), axis=-1)
        base_speed = jnp.linalg.norm(base_velocity, axis=-1, keepdims=True)
        decoded = (parameters[..., :1] * base_speed * forward
                   + parameters[..., 1:] * lateral_speed * side)
        target = jnp.where(active_b[..., None], decoded, base_velocity)
        world = jnp.where(controlled_b[..., None], world, base_velocity)
        delta = target - world
        delta_norm = jnp.linalg.norm(delta, axis=-1, keepdims=True)
        world_limited = world + delta * jnp.minimum(
            1.0, (max_intent_accel * dt)
            / jnp.maximum(delta_norm, 1e-12))
        world_speed = jnp.linalg.norm(
            world_limited, axis=-1, keepdims=True)
        world_limited *= jnp.minimum(
            1.0, max_speed / jnp.maximum(world_speed, 1e-12))
        world = jnp.where(
            controlled_b[..., None], world_limited, base_velocity)

        waypoint = state[..., :2] + intent_lookahead * world
        target_state = jnp.concatenate((waypoint, world), axis=-1)
        tracking_error = target_state - state
        tracking_norm = jnp.linalg.norm(
            tracking_error, axis=-1, keepdims=True)
        tracking_max = jnp.abs(
            tracking_error / jnp.maximum(tracking_norm, 1e-9)
            * sense_radius)
        tracking = jnp.clip(
            jnp.einsum("bnk,dk->bnd",
                       jnp.clip(tracking_error, -tracking_max, tracking_max),
                       gain),
            -max_force, max_force)
        reference_force = jnp.where(
            controlled_b[..., None], tracking, base)
        smooth = (runtime_step > 0) | (horizon_index > 0)
        smoothed = ((reference_force
                     + action_smooth_weight * previous_force)
                    / (1.0 + action_smooth_weight))
        reference_force = jnp.where(smooth, smoothed, reference_force)

        action, latch, brake_mask, outside_now = batched_wang_qp(
            state, reference_force, latch, n_agents=n_agents,
            sense_radius=sense_radius, mass=mass, max_force=max_force,
            dt=dt, safe_distance=safe_distance, wang_gamma=wang_gamma)
        acceleration = action / mass
        velocity_before = state[..., 2:]
        sub_dt = dt / integration_substeps
        alpha = max_force / mass

        def integrate(_, integration_carry):
            physical, local_minimum = integration_carry
            velocity = physical[..., 2:]
            normal_position = (physical[..., :2] + velocity * sub_dt
                               + 0.5 * acceleration * sub_dt * sub_dt)
            normal_velocity = jnp.clip(
                velocity + acceleration * sub_dt, -max_speed, max_speed)

            speed = jnp.linalg.norm(velocity, axis=-1)
            direction = velocity / jnp.maximum(speed[..., None], 1e-12)
            brake_acceleration = -alpha * direction
            active_time = jnp.minimum(sub_dt, speed / alpha)
            brake_position = (
                physical[..., :2] + velocity * active_time[..., None]
                + 0.5 * brake_acceleration * active_time[..., None] ** 2)
            brake_velocity = (
                velocity + brake_acceleration * active_time[..., None])
            brake_velocity = jnp.where(
                (speed <= alpha * sub_dt + 1e-12)[..., None],
                0.0, brake_velocity)
            position = jnp.where(
                brake_mask[..., None], brake_position, normal_position)
            velocity = jnp.where(
                brake_mask[..., None], brake_velocity, normal_velocity)
            physical = jnp.concatenate((position, velocity), axis=-1)
            local_minimum = jnp.minimum(
                local_minimum, _minimum_center(physical, n_agents))
            return physical, local_minimum

        state, minimum = jax.lax.fori_loop(
            0, integration_substeps, integrate, (state, minimum))
        effective_force = mass * (state[..., 2:] - velocity_before) / dt
        distance = jnp.linalg.norm(state[..., :2] - goals_b, axis=-1)
        distance_integral += jnp.sum(distance * focus, axis=-1) * dt
        brake += jnp.sum(brake_mask, axis=-1)
        outside += outside_now
        return ((state, world, latch, effective_force, distance_integral, brake,
                 outside, minimum), None)

    zeros = jnp.zeros((batch,), dtype=state.dtype)
    carry = (state, world, latch, previous_force, zeros, zeros, zeros,
             initial_minimum)
    (state, _world, _latch, _force, distance_integral, brake, outside,
     minimum), _ = jax.lax.scan(
         body, carry, jnp.arange(horizon_steps, dtype=jnp.int32))
    final_distance_each = jnp.linalg.norm(
        state[..., :2] - goals_b, axis=-1)
    final_distance = jnp.sum(final_distance_each * focus, axis=-1)
    progress = initial_distance - final_distance
    effort_each = ((parameters[..., 0] - 1.0) ** 2
                   + 0.15 * parameters[..., 1] ** 2)
    effort = jnp.sum(effort_each * focus, axis=-1)
    change = jnp.sum(
        jnp.sum((parameters - previous_parameters) ** 2, axis=-1)
        * focus, axis=-1)
    public_violation = jnp.maximum(0.0, hard_distance - minimum)
    return (1e9 * (public_violation > 0.0).astype(state.dtype)
            + 2e7 * outside
            + 400.0 * brake
            + 120.0 * final_distance
            + 3.0 * distance_integral
            - 180.0 * progress
            + 0.05 * effort
            + 0.5 * change
            + 1e8 * public_violation)
