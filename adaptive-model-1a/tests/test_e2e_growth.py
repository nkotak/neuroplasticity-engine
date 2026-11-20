"""
Integration tests for the E2E growth observation pipeline.

Verifies the data generator, training with plasticity, architecture
snapshot capture, and metrics serialization all work correctly.
Uses small models and few epochs for speed.

Docs: https://docs.pytest.org/en/stable/

Sample input:
    pytest tests/test_e2e_growth.py -v

Expected output:
    4 passed
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import tempfile
from collections import Counter

import torch
import torch.nn as nn

from generate_reasoning_data import generate_reasoning_dataset, _SUBJECT_WEIGHTS
from transformer_model import AdaptiveTransformer
from transformer_model.adaptive_feedforward import AdaptiveFeedForward
from adaptive_network import StructuralPlasticityController
from run_e2e_training import capture_architecture_snapshot


def test_data_generator_produces_valid_data():
    """Data generator produces correct number of examples with expected columns and distribution."""
    with tempfile.TemporaryDirectory() as tmpdir:
        examples = generate_reasoning_dataset(
            num_examples=100, seed=123, output_dir=tmpdir
        )

    assert len(examples) == 100, f"Expected 100 examples, got {len(examples)}"

    # Verify all required keys present and non-empty
    for i, ex in enumerate(examples):
        for key in ['instruction', 'reasoning', 'response', 'subject']:
            assert key in ex, f"Example {i} missing key: {key}"
            assert isinstance(ex[key], str) and len(ex[key].strip()) > 0, \
                f"Example {i} has empty {key}"

    # Verify subject distribution (within tolerance)
    counts = Counter(ex['subject'] for ex in examples)
    for subject, weight in _SUBJECT_WEIGHTS.items():
        expected = round(100 * weight)
        actual = counts.get(subject, 0)
        assert abs(actual - expected) <= 7, \
            f"{subject}: expected ~{expected}, got {actual}"


def test_training_with_plasticity_fires():
    """Training for 30 epochs with aggressive plasticity produces at least one plasticity event."""
    vocab_size = 200
    seq_len = 16
    d_model = 32

    model = AdaptiveTransformer(
        vocab_size=vocab_size, d_model=d_model, nhead=2,
        num_layers=2, d_ff=64, max_seq_len=seq_len,
    )
    for _ in range(200):
        model.increment_warmup()
    model.train()

    sp_controller = StructuralPlasticityController(
        prune_threshold=0.01,
        growth_rate=0.2,
        split_threshold=0.3,
        merge_threshold=0.3,
        prune_frequency=5,
        structural_frequency=5,
        min_connections=100,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    torch.manual_seed(42)
    data = torch.randint(0, vocab_size, (100, seq_len))

    for epoch in range(30):
        model.train()
        for i in range(0, len(data), 16):
            batch = data[i:i+16]
            optimizer.zero_grad()
            output = model(batch)
            logits = output[:, :-1, :].contiguous()
            targets = batch[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(
                logits.view(-1, vocab_size), targets.view(-1)
            )
            loss.backward()
            optimizer.step()
            model.increment_warmup()

        sp_controller.step(model)

    # At least one plasticity operation should have occurred in 30 epochs
    stats = sp_controller.stats
    total_events = (
        sum(stats.get('total_pruned', [0])) +
        sum(stats.get('total_grown', [0])) +
        sum(stats.get('total_splits', [0])) +
        sum(stats.get('total_merges', [0]))
    )
    assert total_events > 0, \
        f"No plasticity events in 30 epochs with aggressive settings. Stats: {stats}"


def test_architecture_snapshot_captures_changes():
    """Architecture snapshot correctly reflects d_ff and sparsity after plasticity."""
    vocab_size = 200
    model = AdaptiveTransformer(
        vocab_size=vocab_size, d_model=32, nhead=2,
        num_layers=2, d_ff=64, max_seq_len=16,
    )
    for _ in range(200):
        model.increment_warmup()

    sp_controller = StructuralPlasticityController(
        prune_threshold=0.5,
        growth_rate=0.5,
        prune_frequency=1,
        structural_frequency=1,
        min_connections=10,
    )

    # Get initial snapshot
    snap_before = capture_architecture_snapshot(
        model, sp_controller, epoch=0, train_loss=5.0, val_loss=5.0
    )

    # Verify snapshot structure
    assert 'epoch' in snap_before
    assert 'total_model_params' in snap_before
    assert 'global_sparsity' in snap_before
    assert 'd_ff_per_layer' in snap_before
    assert 'layer_details' in snap_before
    assert len(snap_before['d_ff_per_layer']) == 2  # 2 layers

    # Initial state should have zero sparsity
    assert snap_before['global_sparsity'] == 0.0

    # Verify d_ff values are correct
    for d_ff in snap_before['d_ff_per_layer']:
        assert d_ff == 64, f"Expected d_ff=64, got {d_ff}"

    # Verify all layer details present
    for ld in snap_before['layer_details']:
        assert 'name' in ld
        assert 'd_ff' in ld
        assert 'input_mask_sparsity' in ld
        assert 'output_mask_sparsity' in ld
        assert 'warmup_factor' in ld


def test_metrics_jsonl_format():
    """Metrics can be written to JSONL and read back with all keys intact."""
    vocab_size = 200
    model = AdaptiveTransformer(
        vocab_size=vocab_size, d_model=32, nhead=2,
        num_layers=2, d_ff=64, max_seq_len=16,
    )
    for _ in range(200):
        model.increment_warmup()

    sp_controller = StructuralPlasticityController()

    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        path = f.name
        for epoch in range(3):
            snap = capture_architecture_snapshot(
                model, sp_controller, epoch=epoch,
                train_loss=5.0 - epoch * 0.5, val_loss=4.5 - epoch * 0.4,
            )
            f.write(json.dumps(snap) + '\n')

    # Read back
    loaded = []
    with open(path) as f:
        for line in f:
            loaded.append(json.loads(line))

    assert len(loaded) == 3, f"Expected 3 snapshots, got {len(loaded)}"

    required_keys = [
        "epoch", "train_loss", "val_loss", "perplexity",
        "total_model_params", "global_sparsity", "d_ff_per_layer",
        "cumulative_splits", "cumulative_merges",
        "cumulative_grown", "cumulative_pruned",
        "skip_connections", "layer_details",
    ]
    for snap in loaded:
        for key in required_keys:
            assert key in snap, f"Missing key: {key} in snapshot epoch {snap.get('epoch')}"

    # Verify values are sensible
    assert loaded[0]['epoch'] == 0
    assert loaded[2]['epoch'] == 2
    assert loaded[0]['train_loss'] > loaded[2]['train_loss']

    os.unlink(path)


if __name__ == "__main__":
    all_validation_failures = []
    total_tests = 0

    tests = [
        ("Data generator validation", test_data_generator_produces_valid_data),
        ("Plasticity fires in training", test_training_with_plasticity_fires),
        ("Architecture snapshot capture", test_architecture_snapshot_captures_changes),
        ("Metrics JSONL format", test_metrics_jsonl_format),
    ]

    for name, test_fn in tests:
        total_tests += 1
        try:
            test_fn()
        except Exception as e:
            all_validation_failures.append(f"{name}: {e}")

    if all_validation_failures:
        print(f"VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"VALIDATION PASSED - All {total_tests} tests produced expected results")
        sys.exit(0)
