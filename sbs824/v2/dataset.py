"""Auditable v2 decision records and strict protocol lineage."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

import numpy as np

from .protocol import SyncEventProtocol


@dataclass
class DecisionDataset:
    self_features: list[np.ndarray] = field(default_factory=list)
    edge_features: list[np.ndarray] = field(default_factory=list)
    edge_mask: list[np.ndarray] = field(default_factory=list)
    labels: list[np.ndarray] = field(default_factory=list)
    evidence_mask: list[np.ndarray] = field(default_factory=list)
    mode: list[np.ndarray] = field(default_factory=list)
    active_mask: list[np.ndarray] = field(default_factory=list)
    release_mask: list[np.ndarray] = field(default_factory=list)
    component_id: list[np.ndarray] = field(default_factory=list)
    optimized_mask: list[np.ndarray] = field(default_factory=list)
    label_mask: list[np.ndarray] = field(default_factory=list)
    applied_mask: list[np.ndarray] = field(default_factory=list)
    world_intent_before: list[np.ndarray] = field(default_factory=list)
    world_intent_after: list[np.ndarray] = field(default_factory=list)
    teacher_source: list[str] = field(default_factory=list)
    teacher_cost: list[float] = field(default_factory=list)
    incumbent_cost: list[float] = field(default_factory=list)
    validated: list[bool] = field(default_factory=list)

    def append(self, *, teacher_source: str, teacher_cost: float,
               incumbent_cost: float, validated: bool, **arrays) -> None:
        required = {
            "self_features", "edge_features", "edge_mask", "labels",
            "evidence_mask", "mode", "active_mask", "release_mask",
            "component_id", "optimized_mask", "label_mask", "applied_mask",
            "world_intent_before", "world_intent_after",
        }
        if set(arrays) != required:
            raise ValueError(
                f"dataset record keys differ: missing={required-set(arrays)}, "
                f"extra={set(arrays)-required}")
        masks = [np.asarray(arrays[name], dtype=bool) for name in
                 ("optimized_mask", "label_mask", "applied_mask")]
        if not all(np.array_equal(masks[0], value) for value in masks[1:]):
            raise ValueError("optimized_mask, label_mask and applied_mask must match")
        for name, value in arrays.items():
            getattr(self, name).append(np.asarray(value).copy())
        self.teacher_source.append(str(teacher_source))
        self.teacher_cost.append(float(teacher_cost))
        self.incumbent_cost.append(float(incumbent_cost))
        self.validated.append(bool(validated))

    def save(self, directory: Path, protocol: SyncEventProtocol,
             extra_manifest: dict | None = None) -> None:
        protocol.assert_data_ready()
        directory.mkdir(parents=True, exist_ok=False)
        arrays = {
            name: np.asarray(value) for name, value in vars(self).items()
            if name not in {"teacher_source"}
        }
        arrays["teacher_source"] = np.asarray(self.teacher_source, dtype="U16")
        np.savez_compressed(directory / "decisions.npz", **arrays)
        manifest = {
            "protocol": protocol.manifest(),
            "records": len(self.labels),
            "fields": sorted(vars(self)),
        }
        if extra_manifest:
            manifest["extra"] = extra_manifest
        (directory / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8")
