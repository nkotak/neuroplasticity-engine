"""
Test multi-modal forward pass, aux_loss wiring, and backward compatibility.

Verifies that AdaptiveTransformer correctly handles multi-modal inputs
(text + image), surfaces aux_loss from DomainExpertController, and
maintains full backward compatibility with text-only mode.

Uses real tensor data, no mocks. Audio modality excluded due to pre-existing
shape bug in the audio embedder (2D vs 3D output).

Sample input:
    modality_inputs = {'text': torch.randint(0, 500, (2, 8)),
                       'image': torch.randn(2, 3, 64, 64)}

Expected output:
    output shape: [2, seq_len, 500]
    model.last_aux_loss: non-None tensor with requires_grad=True
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformer_model import AdaptiveTransformer
from config import MultiModalConfig


def _create_mm_model(enabled_modalities=None):
    """Create a small multi-modal AdaptiveTransformer for testing."""
    if enabled_modalities is None:
        enabled_modalities = ['text', 'image']
    mm_config = MultiModalConfig(
        text_vocab_size=500, text_max_seq_len=16, text_d_model=64,
        image_size=64, image_patch_size=8, image_d_model=64,
        audio_sample_rate=16000, audio_segment_ms=1000, audio_d_model=64,
        enabled_modalities=enabled_modalities,
    )
    model = AdaptiveTransformer(
        vocab_size=500, d_model=64, nhead=4, num_layers=2,
        d_ff=256, max_seq_len=16,
        modality_config=mm_config.to_modality_dict()
    )
    for _ in range(200):
        model.increment_warmup()
    return model


def _set_inference_mode(model):
    """Put model in inference mode (training=False)."""
    model.training = False
    for module in model.modules():
        module.training = False
    return model


def test_multimodal_forward_shape():
    """Multi-modal forward pass produces 3D logit tensor with correct vocab dim."""
    model = _create_mm_model()
    _set_inference_mode(model)

    inputs = {
        'text': torch.randint(0, 500, (2, 8)),
        'image': torch.randn(2, 3, 64, 64),
    }
    with torch.no_grad():
        output = model(None, inputs)

    assert output.dim() == 3, f"Expected 3D output, got {output.dim()}D"
    assert output.shape[0] == 2, f"Expected batch size 2, got {output.shape[0]}"
    assert output.shape[2] == 500, f"Expected vocab_size 500, got {output.shape[2]}"
    assert not torch.isnan(output).any(), "NaN in multi-modal output"


def test_aux_loss_populated_multimodal():
    """aux_loss attribute is set after multi-modal forward pass and has grad."""
    model = _create_mm_model()
    model.train()

    inputs = {
        'text': torch.randint(0, 500, (2, 8)),
        'image': torch.randn(2, 3, 64, 64),
    }
    model(None, inputs)

    assert model.last_aux_loss is not None, \
        "last_aux_loss should be set in multi-modal mode"
    assert model.last_aux_loss.requires_grad, \
        "last_aux_loss must participate in backward pass"


def test_aux_loss_none_text_only():
    """aux_loss attribute is None for text-only forward pass."""
    model = AdaptiveTransformer(
        vocab_size=500, d_model=32, nhead=2, num_layers=2,
        d_ff=128, max_seq_len=16
    )
    _set_inference_mode(model)

    input_ids = torch.randint(0, 500, (2, 8))
    with torch.no_grad():
        model(input_ids)

    assert model.last_aux_loss is None, \
        "last_aux_loss should be None in text-only mode"


def test_aux_loss_backward():
    """Gradients flow through aux_loss into domain expert router parameters."""
    model = _create_mm_model()
    model.train()

    inputs = {
        'text': torch.randint(0, 500, (2, 8)),
        'image': torch.randn(2, 3, 64, 64),
    }
    output = model(None, inputs)
    total_loss = output.sum() + model.last_aux_loss
    total_loss.backward()

    # Verify router got gradients
    router_grad = False
    for p in model.domain_controller.router.parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            router_grad = True
            break
    assert router_grad, "Router should receive gradients through aux_loss"


def test_text_only_backward_compatibility():
    """Text-only model works exactly as before — no regressions."""
    model = AdaptiveTransformer(
        vocab_size=500, d_model=32, nhead=2, num_layers=2,
        d_ff=128, max_seq_len=16
    )
    for _ in range(200):
        model.increment_warmup()
    model.train()

    input_ids = torch.randint(0, 500, (2, 8))
    output = model(input_ids)
    loss = output.sum()
    loss.backward()

    assert output.shape == (2, 8, 500), \
        f"Expected (2, 8, 500), got {output.shape}"
    assert not torch.isnan(output).any(), "NaN in text-only output"
    assert model.last_aux_loss is None, \
        "last_aux_loss should be None for text-only model"


def test_multimodal_text_only_input():
    """Multi-modal model works when given only text in modality_inputs dict."""
    model = _create_mm_model()
    _set_inference_mode(model)

    inputs = {'text': torch.randint(0, 500, (2, 8))}
    with torch.no_grad():
        output = model(None, inputs)

    assert output.dim() == 3, f"Expected 3D output, got {output.dim()}D"
    assert output.shape[0] == 2, f"Expected batch size 2, got {output.shape[0]}"


if __name__ == "__main__":
    all_validation_failures = []
    total_tests = 0

    tests = [
        ("Multi-modal forward shape", test_multimodal_forward_shape),
        ("aux_loss populated (multi-modal)", test_aux_loss_populated_multimodal),
        ("aux_loss None (text-only)", test_aux_loss_none_text_only),
        ("aux_loss backward", test_aux_loss_backward),
        ("Text-only backward compatibility", test_text_only_backward_compatibility),
        ("Multi-modal text-only input", test_multimodal_text_only_input),
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
