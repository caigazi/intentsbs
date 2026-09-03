"""Small CPU/GPU parity and throughput benchmark for the JAX Wang kernel."""

from __future__ import annotations

import argparse
import json
import os
from time import perf_counter

os.environ.setdefault("JAX_ENABLE_X64", "True")

import jax
import jax.numpy as jnp
import numpy as np

from sbs824.v2.jax_safety import batched_wang_qp
from sbs824.v2.protocol import make_v2_config
from sbs824.wang_safety import solve_wang_braking_qp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", type=int, default=8)
    parser.add_argument("--batch", type=int, default=256)
    args = parser.parse_args()
    cfg = make_v2_config(n_agents=args.agents, n_obstacles=0)
    rng = np.random.default_rng(931)
    angles = np.linspace(0.0, 2.0 * np.pi, args.agents, endpoint=False)
    position = np.column_stack((np.cos(angles), np.sin(angles))) * 0.25 + 2.0
    template = np.column_stack((position, np.zeros((args.agents, 2))))
    states = np.repeat(template[None], args.batch, axis=0)
    states[:, :, 2:] = rng.uniform(
        -0.2, 0.2, size=(args.batch, args.agents, 2))
    references = rng.uniform(
        -1.0, 1.0, size=(args.batch, args.agents, 2))
    latch = np.zeros((args.batch, args.agents), dtype=bool)
    kwargs = dict(
        n_agents=args.agents, sense_radius=cfg.sense_radius,
        mass=cfg.mass, max_force=cfg.max_force, dt=cfg.dt,
        safe_distance=cfg.wang_pair_safe_radius_factor * cfg.car_radius,
        wang_gamma=cfg.wang_gamma)

    def compiled():
        return batched_wang_qp(
            jnp.asarray(states), jnp.asarray(references),
            jnp.asarray(latch), **kwargs)[0]

    started = perf_counter()
    compiled().block_until_ready()
    compile_and_first = perf_counter() - started
    started = perf_counter()
    actual = compiled()
    actual.block_until_ready()
    jax_seconds = perf_counter() - started
    started = perf_counter()
    expected = np.asarray([
        solve_wang_braking_qp(
            states[index], references[index], [], cfg,
            braking_latch=latch[index].copy()).action
        for index in range(args.batch)
    ])
    numpy_seconds = perf_counter() - started
    report = {
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(value) for value in jax.devices()],
        "x64_enabled": bool(jax.config.x64_enabled),
        "agents": args.agents,
        "batch": args.batch,
        "compile_and_first_seconds": compile_and_first,
        "jax_steady_seconds": jax_seconds,
        "numpy_loop_seconds": numpy_seconds,
        "steady_speedup": numpy_seconds / max(jax_seconds, 1e-12),
        "maximum_action_error": float(
            np.max(np.abs(np.asarray(actual) - expected))),
    }
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
