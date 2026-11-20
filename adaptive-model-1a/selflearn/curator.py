"""Deduplication and accepted-shard writing for self-learning."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Set

from selflearn.verifier import contains_training_artifact, verify_item


def _sample_hash(payload: Dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _content_signature(payload: Dict[str, Any]) -> str:
    basis = {
        "context": payload.get("context", ""),
        "instruction": payload.get("instruction", payload.get("prompt", "")),
        "response": payload.get("response", payload.get("target", "")),
    }
    canonical = json.dumps(basis, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_existing_hashes(selflearn_dir: str, benchmark_items: Iterable[Dict[str, Any]]) -> Set[str]:
    hashes: Set[str] = set()
    root = os.path.expanduser(selflearn_dir)
    if os.path.isdir(root):
        for dirpath, _dirs, filenames in os.walk(root):
            for fname in filenames:
                if not fname.endswith(".jsonl"):
                    continue
                path = os.path.join(dirpath, fname)
                with open(path, "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        entry = json.loads(line)
                        hashes.add(_content_signature(entry))
    for item in benchmark_items:
        hashes.add(_content_signature(item))
    return hashes


def curate_candidates(
    candidates: List[Dict[str, Any]],
    cycle_id: str,
    student_checkpoint: str,
    teacher_model: str,
    existing_hashes: Set[str],
) -> List[Dict[str, Any]]:
    """Select verified non-duplicate candidates as training samples."""
    accepted: List[Dict[str, Any]] = []
    for candidate in candidates:
        chosen_response = str(candidate.get("teacher_output") or "")
        if not chosen_response:
            for output in candidate.get("student_outputs", []):
                passed, _score, _verifier = verify_item(candidate, output)
                if passed:
                    chosen_response = output
                    break
        if not chosen_response or contains_training_artifact(chosen_response):
            continue

        passed, score, verifier_type = verify_item(candidate, chosen_response)
        if not passed:
            continue

        sample = {
            "system": "You are a helpful assistant.",
            "context": candidate.get("context", ""),
            "instruction": candidate.get("prompt", ""),
            "reasoning": candidate.get("reasoning", ""),
            "response": chosen_response,
            "task_type": candidate.get("task_type", "instruction"),
            "source": "selflearn",
            "cycle_id": cycle_id,
            "student_checkpoint": student_checkpoint,
            "teacher_model": teacher_model,
            "verifier_type": verifier_type,
            "verifier_passed": True,
            "verifier_score": score,
            "benchmark_bucket": candidate.get("benchmark_bucket", candidate.get("task_type", "instruction")),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        sample["sample_hash"] = _sample_hash(sample)
        content_signature = _content_signature(sample)
        if content_signature in existing_hashes:
            continue
        existing_hashes.add(content_signature)
        accepted.append(sample)
    return accepted


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=True) + "\n")
