"""Controller-independent local SBS monitoring and intent gating."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum

import numpy as np


@dataclass(frozen=True)
class LocalStep:
    """Quantities available to one robot from direct local observation."""
    nominal_goal_speed: float
    actual_goal_speed: float
    active_agent_interaction: bool
    safe: bool
    reached: bool


@dataclass(frozen=True)
class SBSEvidence:
    ready: bool
    online_sbs: bool
    nominal_progress: float
    progress_ratio: float
    active_fraction: float
    safe: bool
    reached: bool


@dataclass(frozen=True)
class Intent:
    alpha: float = 1.0
    beta: float = 0.0

    def clipped(self) -> "Intent":
        return Intent(float(np.clip(self.alpha, -0.5, 1.0)),
                      float(np.clip(self.beta, -1.0, 1.0)))


class GateMode(str, Enum):
    BYPASS = "bypass"
    COORDINATE = "coordinate"


class LocalSBSMonitor:
    """One-second sliding-window detector matching the official audit."""

    def __init__(self, *, dt: float = 0.03, window_steps: int = 33,
                 nominal_epsilon: float = 0.05, ratio_threshold: float = 0.2,
                 active_fraction_threshold: float = 0.8):
        self.dt = dt
        self.window_steps = window_steps
        self.nominal_epsilon = nominal_epsilon
        self.ratio_threshold = ratio_threshold
        self.active_fraction_threshold = active_fraction_threshold
        self._history: deque[LocalStep] = deque(maxlen=window_steps)

    def reset(self) -> None:
        self._history.clear()

    def update(self, step: LocalStep) -> SBSEvidence:
        self._history.append(step)
        ready = len(self._history) == self.window_steps
        if not ready:
            return SBSEvidence(False, False, 0.0, 1.0, 0.0, step.safe, step.reached)
        nominal = sum(max(0.0, item.nominal_goal_speed) for item in self._history) * self.dt
        actual = sum(max(0.0, item.actual_goal_speed) for item in self._history) * self.dt
        ratio = actual / max(nominal, 1e-9)
        active_fraction = sum(item.active_agent_interaction for item in self._history) / self.window_steps
        safe = all(item.safe for item in self._history)
        reached = any(item.reached for item in self._history)
        online = (nominal > self.nominal_epsilon
                  and ratio < self.ratio_threshold
                  and active_fraction >= self.active_fraction_threshold
                  and safe and not reached)
        return SBSEvidence(True, online, nominal, ratio, active_fraction, safe, reached)


class CoordinationGate:
    """Hysteresis around the detector so transient GCBF+ congestion is bypassed."""

    def __init__(self, *, entry_confirm_steps: int = 17, release_confirm_steps: int = 33,
                 release_ratio: float = 0.6, release_active_fraction: float = 0.2):
        self.entry_confirm_steps = entry_confirm_steps
        self.release_confirm_steps = release_confirm_steps
        self.release_ratio = release_ratio
        self.release_active_fraction = release_active_fraction
        self.mode = GateMode.BYPASS
        self._entry_count = 0
        self._release_count = 0

    def reset(self) -> None:
        self.mode = GateMode.BYPASS
        self._entry_count = 0
        self._release_count = 0

    def update(self, evidence: SBSEvidence) -> GateMode:
        if self.mode == GateMode.BYPASS:
            self._entry_count = self._entry_count + 1 if evidence.online_sbs else 0
            if self._entry_count >= self.entry_confirm_steps:
                self.mode = GateMode.COORDINATE
                self._entry_count = 0
        else:
            recovered = (evidence.reached
                         or (evidence.ready
                             and (evidence.progress_ratio >= self.release_ratio
                                  or evidence.active_fraction <= self.release_active_fraction)))
            self._release_count = self._release_count + 1 if recovered else 0
            if self._release_count >= self.release_confirm_steps:
                self.mode = GateMode.BYPASS
                self._release_count = 0
        return self.mode


def apply_velocity_intent(preferred_velocity: np.ndarray, intent: Intent,
                          side_speed: float = 0.25, max_speed: float = 0.5) -> np.ndarray:
    """Bottom-controller-independent (alpha, beta) preferred-velocity interface."""
    preferred = np.asarray(preferred_velocity, dtype=float)
    if preferred.shape != (2,):
        raise ValueError("preferred_velocity must have shape (2,)")
    intent = intent.clipped()
    norm = float(np.linalg.norm(preferred))
    if norm < 1e-9:
        return np.zeros(2)
    forward = preferred / norm
    side = np.array([-forward[1], forward[0]])
    coordinated = intent.alpha * preferred + intent.beta * side_speed * side
    coordinated_norm = float(np.linalg.norm(coordinated))
    if coordinated_norm > max_speed:
        coordinated *= max_speed / coordinated_norm
    return coordinated
