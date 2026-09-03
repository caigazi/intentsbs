"""Single-source protocol contract for clean IntentComm experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum

import numpy as np


class IntentMode(IntEnum):
    BYPASS = 0
    ACTIVE = 1
    RELEASE = 2


@dataclass(frozen=True)
class SyncEventProtocol:
    """Spatially local, fixed-rate synchronous Intent update protocol.

    ``development=True`` prevents this still-audited trigger definition from
    being mistaken for a frozen paper/data protocol.
    """

    version: str = "sync_event_v2_dev"
    development: bool = True
    control_dt: float = 0.03
    intent_period_steps: int = 1
    sensing_radius: float = 0.50
    hard_center_distance: float = 0.20
    # The proof/controller currently audits a 2.5-mm sampled-data margin while
    # evaluation retains the declared 0.20-m hard line.
    certificate_center_distance: float = 0.2025
    integration_substeps: int = 32
    safety_update_rule: str = "one_qp_per_control_tick_zero_order_hold"
    trigger_kind: str = "local_ttc_cpa_v1"
    trigger_distance: float = 0.28
    prediction_horizon: float = 1.0
    minimum_closing_speed: float = 1e-3
    clear_confirm_steps: int = 3
    release_speed_tolerance: float = 0.01
    max_intent_accel: float = 1.0
    intent_lookahead: float = 0.6
    lateral_speed: float = 0.35
    action_smooth_weight: float = 0.25
    self_feature_schema: str = "goal,velocity,base,message,own_active,own_release"
    edge_feature_schema: str = "rel_pos,rel_vel,neighbor_world_intent,distance,closing,ttc"
    communication_payload: str = "world_velocity_float32x2"
    state_machine: str = "BAR_v2"
    safety_backend: str = "wang"

    def validate(self, cfg) -> None:
        if self.version != "sync_event_v2_dev":
            raise ValueError(f"unsupported protocol version: {self.version}")
        if self.intent_period_steps < 1:
            raise ValueError("intent_period_steps must be positive")
        if self.trigger_kind != "local_ttc_cpa_v1":
            raise ValueError(f"unknown trigger_kind: {self.trigger_kind}")
        if self.clear_confirm_steps < 1:
            raise ValueError("clear_confirm_steps must be positive")
        if self.max_intent_accel <= 0.0 or self.intent_lookahead <= 0.0:
            raise ValueError("intent dynamics must be positive")
        actual = {
            "control_dt": float(cfg.dt),
            "sensing_radius": float(cfg.sense_radius),
            "certificate_center_distance": float(
                cfg.wang_pair_safe_radius_factor * cfg.car_radius),
            "trigger_distance": float(cfg.intent_trigger_distance),
            "prediction_horizon": float(cfg.intent_prediction_horizon),
            "minimum_closing_speed": float(cfg.intent_min_closing_speed),
            "integration_substeps": int(cfg.wang_safety_substeps),
        }
        expected = {
            "control_dt": self.control_dt,
            "sensing_radius": self.sensing_radius,
            "certificate_center_distance": self.certificate_center_distance,
            "trigger_distance": self.trigger_distance,
            "prediction_horizon": self.prediction_horizon,
            "minimum_closing_speed": self.minimum_closing_speed,
            "integration_substeps": self.integration_substeps,
        }
        for name, target in expected.items():
            if not np.isclose(actual[name], target, atol=1e-12, rtol=0.0):
                raise ValueError(
                    f"{self.version} requires {name}={target}, got "
                    f"{actual[name]}")

    def assert_data_ready(self) -> None:
        if self.development:
            raise RuntimeError(
                "sync_event_v2 trigger is not frozen; formal datasets are "
                "disabled until the trigger audit passes")

    def manifest(self) -> dict:
        return asdict(self)


SYNC_EVENT_V2_DEV = SyncEventProtocol()


def make_v2_config(**overrides):
    """Build the only accepted double-integrator configuration for v2-dev."""
    from ..simulation import Config

    fixed = {
        "dt": 0.03,
        "mass": 0.1,
        "car_radius": 0.05,
        "sense_radius": 0.5,
        "max_force": 1.0,
        "max_speed": 0.5,
        "wang_pair_safe_radius_factor": 4.05,
        "wang_safety_substeps": 32,
        "intent_trigger_distance": 0.28,
        "intent_prediction_horizon": 1.0,
        "intent_min_closing_speed": 1e-3,
        "influence_top_k": 0,
    }
    conflicts = {
        key: value for key, value in overrides.items()
        if key in fixed and not np.isclose(value, fixed[key])
    }
    if conflicts:
        raise ValueError(f"v2 fixed-parameter override rejected: {conflicts}")
    fixed.update(overrides)
    cfg = Config(**fixed)
    SYNC_EVENT_V2_DEV.validate(cfg)
    return cfg


def transition_modes(previous: np.ndarray, clear_counts: np.ndarray,
                     evidence: np.ndarray, release_complete: np.ndarray,
                     protocol: SyncEventProtocol
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Advance BAR state without ever equating evidence with ACTIVE mode."""
    previous = np.asarray(previous, dtype=np.int8)
    clear = np.asarray(clear_counts, dtype=int).copy()
    evidence = np.asarray(evidence, dtype=bool)
    release_complete = np.asarray(release_complete, dtype=bool)
    if not (previous.shape == clear.shape == evidence.shape
            == release_complete.shape):
        raise ValueError("BAR arrays must have identical shapes")

    mode = previous.copy()
    bypass = previous == IntentMode.BYPASS
    active = previous == IntentMode.ACTIVE
    release = previous == IntentMode.RELEASE

    mode[bypass & evidence] = IntentMode.ACTIVE
    clear[bypass] = 0
    clear[active & evidence] = 0
    clear[active & ~evidence] += 1
    to_release = active & ~evidence & (
        clear >= protocol.clear_confirm_steps)
    mode[to_release] = IntentMode.RELEASE
    clear[to_release] = 0
    mode[release & evidence] = IntentMode.ACTIVE
    mode[release & ~evidence & release_complete] = IntentMode.BYPASS
    clear[mode != IntentMode.ACTIVE] = 0
    return mode, clear


def update_mask(mode: np.ndarray, step: int,
                protocol: SyncEventProtocol) -> np.ndarray:
    """Every ACTIVE robot refreshes on the same fixed-rate tick."""
    return ((np.asarray(mode) == IntentMode.ACTIVE)
            & (step % protocol.intent_period_steps == 0))
