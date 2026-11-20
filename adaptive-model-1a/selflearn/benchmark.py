"""Benchmark loading and scoring for self-learning and promotion gates."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from selflearn.verifier import verify_item


TASK_WEIGHTS = {
    "code": 0.35,
    "math": 0.30,
    "grounded_qa": 0.25,
    "instruction": 0.10,
}


def load_benchmark_items(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def render_prompt(item: Dict[str, Any]) -> str:
    parts: List[str] = []
    if item.get("context"):
        parts.append(f"Context:\n{item['context']}")
    parts.append(str(item["prompt"]))
    return "\n\n".join(parts)


def score_items(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    per_task: Dict[str, List[float]] = {}
    for result in results:
        per_task.setdefault(result["task_type"], []).append(result["score"])

    bucket_scores = {
        task_type: (sum(scores) / len(scores) * 100.0) if scores else 0.0
        for task_type, scores in per_task.items()
    }

    composite = 0.0
    for task_type, weight in TASK_WEIGHTS.items():
        composite += bucket_scores.get(task_type, 0.0) * weight

    return {
        "composite_score": composite,
        "bucket_scores": bucket_scores,
        "results": results,
    }


def evaluate_benchmark(
    checkpoint_dir: str,
    benchmark_path: str,
    tokenizer_path: str | None,
    backend,
) -> Dict[str, Any]:
    """Run a benchmark JSONL file through a backend's inference method."""
    items = load_benchmark_items(benchmark_path)
    results: List[Dict[str, Any]] = []
    for item in items:
        response = backend.run_inference(
            checkpoint_dir=checkpoint_dir,
            prompt=render_prompt(item),
            tokenizer_path=tokenizer_path,
            max_tokens=int(item.get("max_tokens", 128)),
            temperature=float(item.get("temperature", 0.0)),
            top_k=int(item.get("top_k", 40)),
            top_p=float(item.get("top_p", 0.92)),
            repetition_penalty=float(item.get("repetition_penalty", 1.0)),
        )
        passed, score, verifier_type = verify_item(item, response)
        results.append(
            {
                "id": item.get("id"),
                "bucket": item.get("bucket", item.get("task_type")),
                "task_type": item.get("task_type"),
                "prompt": item.get("prompt"),
                "response": response,
                "target": item.get("target"),
                "passed": passed,
                "score": score,
                "verifier_type": verifier_type,
            }
        )
    summary = score_items(results)
    summary["benchmark_path"] = benchmark_path
    summary["checkpoint_dir"] = checkpoint_dir
    return summary
