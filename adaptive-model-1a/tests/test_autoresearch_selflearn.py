"""
Tests for autoresearch/selflearn scaffolding.

These tests cover the new outer-loop interfaces without launching full training.
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from autoresearch.backends.ane_backend import ANEBackend
from autoresearch.backends.mlx_backend import MLXBackend
from autoresearch.promotion import evaluate_promotion_gate
from autoresearch.workspace import (
    is_allowed_path,
    is_protected_path,
    validate_program_md,
)
from mlx_data import JSONLStream, SelfLearnJSONLStream
from selflearn.benchmark import evaluate_benchmark
from selflearn.curator import curate_candidates, load_existing_hashes


class _StubBackend:
    def __init__(self, mapping):
        self.mapping = mapping

    def run_inference(self, checkpoint_dir, prompt, **kwargs):
        return self.mapping.get(prompt, "")


def test_program_md_has_required_sections():
    missing = validate_program_md(os.path.join(os.path.dirname(__file__), "..", "program.md"))
    assert missing == [], f"program.md missing sections: {missing}"


def test_allowlist_and_protected_paths():
    assert is_allowed_path("mlx_train.py")
    assert is_allowed_path("autoresearch/orchestrator.py")
    assert not is_allowed_path("README.md")
    assert is_protected_path("program.md")
    assert is_protected_path("data/benchmarks/selflearn_v1.jsonl")


def test_backend_capabilities():
    mlx = MLXBackend(os.path.join(os.path.dirname(__file__), ".."))
    ane = ANEBackend(os.path.join(os.path.dirname(__file__), ".."))
    assert mlx.supports_training() is True
    assert mlx.supports_inference() is True
    assert ane.supports_training() is False


def test_jsonl_format_entry_supports_system_and_context():
    text = JSONLStream.format_entry(
        {
            "system": "You are a helper.",
            "context": "Facts go here.",
            "instruction": "Answer the question.",
            "reasoning": "Think briefly.",
            "response": "Done.",
        }
    )
    assert "<system> You are a helper." in text
    assert "<context> Facts go here." in text
    assert "<user> Answer the question." in text
    assert "<think>Think briefly.</think>" in text
    assert "<assistant> Done." in text


def test_selflearn_stream_dedupes_by_sample_hash():
    with tempfile.TemporaryDirectory() as tmpdir:
        shard = os.path.join(tmpdir, "accepted.jsonl")
        rows = [
            {
                "instruction": "Hello?",
                "response": "World",
                "sample_hash": "same",
            },
            {
                "instruction": "Hello?",
                "response": "World",
                "sample_hash": "same",
            },
        ]
        with open(shard, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        stream = SelfLearnJSONLStream(tmpdir, val_fraction=0.0, split="train", max_shards=10)
        items = list(stream)
        assert len(items) == 1
        assert "<user> Hello?" in items[0]


def test_benchmark_contamination_rejected():
    benchmark_items = [
        {
            "task_type": "instruction",
            "prompt": "Reply with exactly this word in uppercase: cedar",
            "target": "CEDAR",
            "context": "",
        }
    ]
    existing = load_existing_hashes("__none__", benchmark_items)
    accepted = curate_candidates(
        candidates=[
            {
                "task_type": "instruction",
                "prompt": "Reply with exactly this word in uppercase: cedar",
                "gold_response": "CEDAR",
                "teacher_output": "CEDAR",
                "benchmark_bucket": "format_following",
            }
        ],
        cycle_id="cycle_0001",
        student_checkpoint="runs/example/best",
        teacher_model="teacher",
        existing_hashes=existing,
    )
    assert accepted == []


def test_promotion_gate_rules():
    passed = evaluate_promotion_gate(
        baseline_score=50.0,
        candidate_score=53.5,
        baseline_buckets={"code": 50.0},
        candidate_buckets={"code": 50.2},
        baseline_val_loss=5.10,
        candidate_val_loss=5.12,
    )
    assert passed.passed is True

    failed = evaluate_promotion_gate(
        baseline_score=50.0,
        candidate_score=51.0,
        baseline_buckets={"code": 50.0},
        candidate_buckets={"code": 48.0},
        baseline_val_loss=5.10,
        candidate_val_loss=5.20,
    )
    assert failed.passed is False


def test_benchmark_scoring_with_stub_backend():
    benchmark_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "benchmarks", "research_mini_v1.jsonl"
    )
    mapping = {}
    with open(benchmark_path, "r", encoding="utf-8") as fh:
        for line in fh:
            item = json.loads(line)
            prompt = item["prompt"] if not item.get("context") else f"Context:\n{item['context']}\n\n{item['prompt']}"
            mapping[prompt] = item["target"] if item["task_type"] != "code" else "def solution(xs):\n    return list(reversed(xs))\n"
    result = evaluate_benchmark(
        checkpoint_dir="unused",
        benchmark_path=benchmark_path,
        tokenizer_path=None,
        backend=_StubBackend(mapping),
    )
    assert result["composite_score"] > 0.0
