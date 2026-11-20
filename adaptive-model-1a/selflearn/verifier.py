"""Verification helpers for self-learning tasks and benchmarks."""

from __future__ import annotations

import json
import math
import re
from typing import Dict, List, Tuple


ARTIFACT_PATTERNS = [
    re.compile(r"List\[int\]"),
    re.compile(r"seen\.add\(n\)"),
    re.compile(r"(.)\1{10,}"),
    re.compile(r"\b0(?:\s+0){5,}\b"),
]


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def contains_training_artifact(text: str) -> bool:
    return any(pattern.search(text) for pattern in ARTIFACT_PATTERNS)


def extract_code(text: str) -> str:
    if "```" not in text:
        return text.strip()
    parts = text.split("```")
    if len(parts) >= 3:
        code = parts[1]
        if code.startswith("python"):
            code = code[len("python"):].lstrip()
        return code.strip()
    return text.strip()


def _verify_code(response: str, tests: List[str]) -> Tuple[bool, float]:
    namespace: Dict[str, object] = {}
    code = extract_code(response)
    try:
        exec(code, namespace, namespace)
        for test in tests:
            exec(test, namespace, namespace)
    except Exception:
        return False, 0.0
    return True, 1.0


def _verify_math(response: str, target: str) -> Tuple[bool, float]:
    left = normalize_text(response)
    right = normalize_text(target)
    return left == right, 1.0 if left == right else 0.0


def _verify_grounded_qa(response: str, target: str, context: str) -> Tuple[bool, float]:
    norm_resp = normalize_text(response)
    norm_target = normalize_text(target)
    norm_context = normalize_text(context)
    passed = norm_resp == norm_target and norm_target in norm_context
    return passed, 1.0 if passed else 0.0


def _verify_instruction(response: str, target: str) -> Tuple[bool, float]:
    norm_resp = normalize_text(response)
    norm_target = normalize_text(target)
    return norm_resp == norm_target, 1.0 if norm_resp == norm_target else 0.0


def verify_item(item: Dict[str, object], response: str) -> Tuple[bool, float, str]:
    """Verify a candidate or benchmark response against its task metadata."""
    if contains_training_artifact(response):
        return False, 0.0, "artifact_filter"

    task_type = str(item.get("task_type", "instruction"))
    if task_type == "code":
        passed, score = _verify_code(response, list(item.get("tests", [])))
        return passed, score, "unit_test"
    if task_type == "math":
        passed, score = _verify_math(response, str(item.get("target", "")))
        return passed, score, "exact_match"
    if task_type == "grounded_qa":
        passed, score = _verify_grounded_qa(
            response,
            str(item.get("target", "")),
            str(item.get("context", "")),
        )
        return passed, score, "citation"
    passed, score = _verify_instruction(response, str(item.get("target", "")))
    return passed, score, "judge_ensemble"
