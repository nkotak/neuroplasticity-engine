"""
Test that a training step completes and loss decreases.

Uses a small AdaptiveTransformer with real tensor data.
Verifies the training loop mechanics work end-to-end.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from transformer_model import AdaptiveTransformer


def _set_inference_mode(model):
    """Put model in inference mode (training=False)."""
    model.training = False
    for module in model.modules():
        module.training = False
    return model


def test_single_training_step():
    """A single training step computes loss and updates parameters."""
    model = AdaptiveTransformer(
        vocab_size=500, d_model=32, nhead=2, num_layers=2,
        d_ff=128, max_seq_len=16
    )
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    input_ids = torch.randint(0, 500, (2, 8))
    target_ids = torch.randint(0, 500, (2, 8))

    # Forward pass
    output = model(input_ids)
    # output shape: [batch, seq_len, vocab_size]
    loss = nn.functional.cross_entropy(
        output.view(-1, 500), target_ids.view(-1)
    )

    # Backward + step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert loss.item() > 0, "Loss should be positive"
    assert not torch.isnan(loss), "Loss should not be NaN"


def test_loss_decreases_over_steps():
    """Loss decreases over multiple training steps on the same batch."""
    model = AdaptiveTransformer(
        vocab_size=200, d_model=32, nhead=2, num_layers=2,
        d_ff=128, max_seq_len=16
    )
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Fixed batch for overfitting test
    torch.manual_seed(42)
    input_ids = torch.randint(0, 200, (4, 8))
    target_ids = torch.randint(0, 200, (4, 8))

    losses = []
    for step in range(20):
        output = model(input_ids)
        loss = nn.functional.cross_entropy(
            output.view(-1, 200), target_ids.view(-1)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # Loss should decrease — compare first 3 avg vs last 3 avg
    early_avg = sum(losses[:3]) / 3
    late_avg = sum(losses[-3:]) / 3
    assert late_avg < early_avg, \
        f"Loss should decrease: early avg {early_avg:.4f} -> late avg {late_avg:.4f}"


def test_gradient_clipping_compatible():
    """Model works with gradient clipping (common training practice)."""
    model = AdaptiveTransformer(
        vocab_size=200, d_model=32, nhead=2, num_layers=2,
        d_ff=128, max_seq_len=16
    )
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    input_ids = torch.randint(0, 200, (2, 8))
    target_ids = torch.randint(0, 200, (2, 8))

    output = model(input_ids)
    loss = nn.functional.cross_entropy(
        output.view(-1, 200), target_ids.view(-1)
    )

    optimizer.zero_grad()
    loss.backward()

    # Clip gradients
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()

    assert grad_norm >= 0, "Gradient norm should be non-negative"
    assert not torch.isnan(torch.tensor(grad_norm)), "Gradient norm should not be NaN"


def test_model_save_load_roundtrip():
    """Model state dict can be saved and restored, producing same output.

    Note: GlobalLayerRegistry uses UUIDs as layer IDs, so state dicts
    can only be loaded back into the same model instance (or one with
    matching UUIDs). This tests the save/restore-to-same-instance path.
    """
    import tempfile

    model = AdaptiveTransformer(
        vocab_size=200, d_model=32, nhead=2, num_layers=2,
        d_ff=128, max_seq_len=16
    )
    _set_inference_mode(model)

    torch.manual_seed(42)
    input_ids = torch.randint(0, 200, (1, 8))

    with torch.no_grad():
        original_output = model(input_ids)

    # Save state dict
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(model.state_dict(), f.name)
        save_path = f.name

    # Perturb model weights to confirm load actually restores them
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.5)

    # Reload state dict into the same model
    model.load_state_dict(torch.load(save_path, weights_only=True))
    _set_inference_mode(model)

    with torch.no_grad():
        restored_output = model(input_ids)

    # Outputs should be identical to original (before perturbation)
    assert torch.allclose(original_output, restored_output, atol=1e-6), \
        "Restored model should produce same output as original"

    # Cleanup
    os.unlink(save_path)


if __name__ == "__main__":
    all_validation_failures = []
    total_tests = 0

    tests = [
        ("Single training step", test_single_training_step),
        ("Loss decreases over steps", test_loss_decreases_over_steps),
        ("Gradient clipping compatible", test_gradient_clipping_compatible),
        ("Save/load roundtrip", test_model_save_load_roundtrip),
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
