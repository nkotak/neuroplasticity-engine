"""
Test AdaptiveTransformer forward pass produces correct output shapes.

Uses real tensor data, no mocks. Verifies the core transformer
can process token sequences and produce logit distributions.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformer_model import AdaptiveTransformer


def test_forward_pass_shape():
    """Forward pass produces [batch, seq_len, vocab_size] output."""
    vocab_size = 1000
    d_model = 64
    nhead = 4
    num_layers = 2
    d_ff = 256
    max_seq_len = 32
    batch_size = 4
    seq_len = 16

    model = AdaptiveTransformer(
        vocab_size=vocab_size, d_model=d_model, nhead=nhead,
        num_layers=num_layers, d_ff=d_ff, max_seq_len=max_seq_len
    )
    model.eval()

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    with torch.no_grad():
        output = model(input_ids)

    assert output.shape == (batch_size, seq_len, vocab_size), \
        f"Expected ({batch_size}, {seq_len}, {vocab_size}), got {output.shape}"


def test_forward_pass_gradient_flow():
    """Backward pass computes gradients through all layers."""
    model = AdaptiveTransformer(
        vocab_size=500, d_model=32, nhead=2, num_layers=2,
        d_ff=128, max_seq_len=16
    )
    # Warm up layers so warmup_factor > 0 (default is 0.0 which blocks gradients)
    for _ in range(200):
        model.increment_warmup()
    model.train()

    input_ids = torch.randint(0, 500, (2, 8))
    output = model(input_ids)
    loss = output.sum()
    loss.backward()

    params_with_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    total_params = sum(1 for p in model.parameters() if p.requires_grad)

    assert params_with_grad > 0, "No parameters received gradients"
    grad_ratio = params_with_grad / total_params
    assert grad_ratio > 0.3, f"Only {grad_ratio:.0%} of parameters got gradients"


def test_forward_pass_no_nan():
    """Forward pass produces no NaN values."""
    model = AdaptiveTransformer(
        vocab_size=500, d_model=32, nhead=2, num_layers=2,
        d_ff=128, max_seq_len=16
    )
    model.eval()

    input_ids = torch.randint(0, 500, (2, 8))
    with torch.no_grad():
        output = model(input_ids)

    assert not torch.isnan(output).any(), "NaN values in output"
    assert not torch.isinf(output).any(), "Inf values in output"


def test_different_sequence_lengths():
    """Model handles different sequence lengths up to max_seq_len."""
    model = AdaptiveTransformer(
        vocab_size=500, d_model=32, nhead=2, num_layers=2,
        d_ff=128, max_seq_len=32
    )
    model.eval()

    for seq_len in [1, 4, 16, 32]:
        input_ids = torch.randint(0, 500, (1, seq_len))
        with torch.no_grad():
            output = model(input_ids)
        assert output.shape == (1, seq_len, 500), \
            f"Wrong shape for seq_len={seq_len}: {output.shape}"


if __name__ == "__main__":
    all_validation_failures = []
    total_tests = 0

    tests = [
        ("Forward pass shape", test_forward_pass_shape),
        ("Gradient flow", test_forward_pass_gradient_flow),
        ("No NaN values", test_forward_pass_no_nan),
        ("Different sequence lengths", test_different_sequence_lengths),
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
