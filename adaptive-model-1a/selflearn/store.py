"""Cycle workspace helpers for self-learning."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class CyclePaths:
    cycle_id: str
    root: str
    manifest_path: str
    failures_path: str
    candidates_path: str
    accepted_path: str
    benchmark_before_path: str
    benchmark_after_path: str
    train_config_snapshot_path: str
    artifacts_dir: str


def create_cycle(root_dir: str, cycle_id: str) -> CyclePaths:
    root = os.path.join(root_dir, cycle_id)
    os.makedirs(root, exist_ok=True)
    artifacts_dir = os.path.join(root, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    return CyclePaths(
        cycle_id=cycle_id,
        root=root,
        manifest_path=os.path.join(root, "manifest.json"),
        failures_path=os.path.join(root, "failures.jsonl"),
        candidates_path=os.path.join(root, "candidates.jsonl"),
        accepted_path=os.path.join(root, "accepted.jsonl"),
        benchmark_before_path=os.path.join(root, "benchmark_before.json"),
        benchmark_after_path=os.path.join(root, "benchmark_after.json"),
        train_config_snapshot_path=os.path.join(root, "train_config_snapshot.json"),
        artifacts_dir=artifacts_dir,
    )


def write_manifest(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def read_manifest(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
