"""Result logging helpers for autoresearch and selflearn cycles."""

from __future__ import annotations

import csv
import os
from typing import Dict, Iterable, List


RESULT_COLUMNS: List[str] = [
    "commit",
    "backend",
    "profile",
    "val_loss",
    "benchmark_score",
    "status",
    "description",
]


CYCLE_COLUMNS: List[str] = [
    "cycle_id",
    "student_checkpoint",
    "accepted_count",
    "before_score",
    "after_score",
    "status",
    "description",
]


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def append_tsv(path: str, columns: Iterable[str], row: Dict[str, object]) -> None:
    """Append a TSV row, writing the header on first write."""
    _ensure_parent(path)
    columns = list(columns)
    exists = os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, delimiter="\t")
        if not exists:
            writer.writeheader()
        writer.writerow({col: row.get(col, "") for col in columns})
