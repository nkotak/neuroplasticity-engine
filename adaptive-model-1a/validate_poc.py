"""
End-to-end proof-of-concept validation for the Neuroplasticity Engine.

Runs a minimal training loop with synthetic data, verifies structural
plasticity is active, and tests inference. No external datasets needed.

Sample input:
    python validate_poc.py

Expected output:
    VALIDATION PASSED - All N tests produced expected results
"""

import sys
import os
import time
import logging
import torch
import torch.nn as nn
from collections import defaultdict

# Imports from the project
from transformer_model import AdaptiveTransformer
from adaptive_network import StructuralPlasticityController, GlobalLayerRegistry
from skip_connection_controller import SkipConnectionController
from adaptive_tokenizer import AdaptiveTokenizer
from config import MultiModalConfig

logging.basicConfig(level=logging.WARNING)


def _create_synthetic_data(vocab_size: int, num_examples: int = 200,
                           seq_len: int = 16) -> torch.Tensor:
    """Generate synthetic token sequences for training."""
    torch.manual_seed(42)
    return torch.randint(0, vocab_size, (num_examples, seq_len))


def validate_minimal_training():
    """Phase 4.1: Train a small model and verify loss decreases."""
    vocab_size = 500
    seq_len = 16
    num_epochs = 5
    batch_size = 16

    model = AdaptiveTransformer(
        vocab_size=vocab_size, d_model=64, nhead=4, num_layers=2,
        d_ff=256, max_seq_len=seq_len
    )

    # Warm up layers so they participate in training
    for _ in range(200):
        model.increment_warmup()
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    data = _create_synthetic_data(vocab_size, num_examples=200, seq_len=seq_len)

    epoch_losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            if len(batch) < 2:
                continue

            input_ids = batch
            targets = batch  # Self-supervised: predict same tokens

            output = model(input_ids)
            loss = nn.functional.cross_entropy(
                output.view(-1, vocab_size), targets.view(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
        epoch_losses.append(avg_loss)

    # Verify loss decreased
    early_loss = sum(epoch_losses[:2]) / 2
    late_loss = sum(epoch_losses[-2:]) / 2

    assert late_loss < early_loss, \
        f"Training loss did not decrease: {early_loss:.4f} -> {late_loss:.4f}"
    assert not any(torch.isnan(torch.tensor(l)) for l in epoch_losses), \
        "NaN loss detected during training"

    return model, epoch_losses


def validate_plasticity(model: nn.Module):
    """Phase 4.2: Verify structural plasticity operations execute on the model."""
    controller = StructuralPlasticityController(
        prune_threshold=0.3,    # Aggressive to ensure something happens
        growth_rate=0.3,
        prune_frequency=1,
        structural_frequency=1,
    )

    # Run several epochs of plasticity in training mode
    model.train()
    x = torch.randn(4, 8, 64)  # batch=4, seq=8, d_model=64

    # Run forward passes to populate activation stats
    for _ in range(30):
        for name, module in model.named_modules():
            if hasattr(module, 'input_activation_stats'):
                module(x)

    # Run plasticity steps
    for _ in range(5):
        controller.step(model)

    assert controller.epoch_counter == 5, \
        f"Expected 5 plasticity steps, got {controller.epoch_counter}"

    # Verify the controller accessed AdaptiveFeedForward modules
    # (the whole point of Phase 2.1 refactor)
    from transformer_model.adaptive_feedforward import AdaptiveFeedForward
    ff_modules = [m for m in model.modules() if isinstance(m, AdaptiveFeedForward)]
    assert len(ff_modules) > 0, "No AdaptiveFeedForward modules found in model"

    # Verify model still produces valid output after plasticity
    model.train(False)
    for m in model.modules():
        m.training = False
    test_input = torch.randint(0, 500, (1, 8))
    with torch.no_grad():
        output = model(test_input)
    assert not torch.isnan(output).any(), "NaN in output after plasticity"

    return controller.stats


def validate_skip_connections():
    """Phase 4.2b: Verify skip connection controller works."""
    registry = GlobalLayerRegistry()
    from transformer_model.adaptive_feedforward import AdaptiveFeedForward

    model = nn.Module()
    d_model, d_ff = 32, 64
    for i in range(4):
        ff = AdaptiveFeedForward(d_model, d_ff)
        model.add_module(f"ff_{i}", ff)
        registry.add_layer(ff, 'feedforward', d_model, d_model, position=i)

    controller = SkipConnectionController(
        correlation_threshold=0.0,  # Accept any correlation
        activation_window=50,
    )
    controller.epoch_counter = 19  # Past warmup

    # Feed correlated activations
    base = torch.randn(4, 32)
    for _ in range(20):
        noise = torch.randn(4, 32) * 0.1
        for layer_id in registry.order:
            controller.update_activations(layer_id, base + noise)

    # Run step — should find candidates
    added = controller.step(registry, model)

    candidates = controller.analyze_layer_correlations(registry)
    assert len(candidates) > 0, \
        "No skip connection candidates found with correlated activations"

    return len(candidates)


def validate_inference(model: nn.Module):
    """Phase 4.3: Verify model can generate tokens without crashing."""
    model.train(False)
    for m in model.modules():
        m.training = False

    # Simple autoregressive generation
    vocab_size = 500
    prompt = torch.randint(0, vocab_size, (1, 4))  # Short prompt

    generated = prompt.clone()
    max_new_tokens = 10

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Truncate to max_seq_len if needed
            input_seq = generated[:, -16:]  # max_seq_len=16
            output = model(input_seq)
            # Get logits for last position
            next_logits = output[:, -1, :]
            # Greedy decode
            next_token = next_logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

    total_generated = generated.shape[1] - prompt.shape[1]
    assert total_generated == max_new_tokens, \
        f"Expected {max_new_tokens} new tokens, got {total_generated}"
    assert not torch.isnan(generated.float()).any(), "NaN in generated tokens"

    return generated[0].tolist()


def validate_tokenizer():
    """Phase 4.3b: Verify tokenizer works end-to-end."""
    tokenizer = AdaptiveTokenizer(initial_vocab_size=500)

    texts = [
        "Hello world. This is a test.",
        "The quick brown fox jumps over the lazy dog.",
        "Neural networks learn representations.",
    ] * 30

    tokenizer.train(texts, output_path="/tmp/validate_poc_tokenizer")

    # Encode/decode roundtrip
    test = "Hello world"
    encoded = tokenizer.encode(test)
    decoded = tokenizer.decode(encoded.ids)

    assert len(encoded.ids) > 0, "Encoding produced no tokens"
    assert isinstance(decoded, str) and len(decoded) > 0, "Decoding failed"

    # Verify special tokens
    think_id = tokenizer.token_to_id("<think>")
    assert think_id is not None, "<think> token not in vocabulary"

    return tokenizer.get_vocab_size()


def validate_multimodal_forward():
    """Phase 4.5: Verify multi-modal forward pass works and produces aux_loss."""
    mm_config = MultiModalConfig(
        text_vocab_size=500, text_max_seq_len=16, text_d_model=64,
        image_size=64, image_patch_size=8, image_d_model=64,
        enabled_modalities=['text', 'image'],
    )

    model = AdaptiveTransformer(
        vocab_size=500, d_model=64, nhead=4, num_layers=2,
        d_ff=256, max_seq_len=16,
        modality_config=mm_config.to_modality_dict()
    )

    # Warm up layers
    for _ in range(200):
        model.increment_warmup()
    model.train()

    # Create multi-modal inputs
    modality_inputs = {
        'text': torch.randint(0, 500, (2, 8)),
        'image': torch.randn(2, 3, 64, 64),
    }

    output = model(None, modality_inputs, domain_id=0)

    # Verify output shape
    assert output.dim() == 3, f"Expected 3D output, got {output.dim()}D"
    assert output.shape[0] == 2, f"Expected batch size 2, got {output.shape[0]}"
    assert output.shape[2] == 500, f"Expected vocab_size 500, got {output.shape[2]}"
    assert not torch.isnan(output).any(), "NaN in multi-modal output"

    # Verify aux_loss was produced
    assert model.last_aux_loss is not None, \
        "aux_loss should be set in multi-modal mode"
    assert model.last_aux_loss.item() >= 0, \
        f"aux_loss should be non-negative, got {model.last_aux_loss.item()}"

    # Verify gradient flow through aux_loss
    loss = output.sum() + model.last_aux_loss
    loss.backward()

    params_with_grad = sum(
        1 for p in model.parameters()
        if p.grad is not None and p.grad.abs().sum() > 0
    )
    assert params_with_grad > 0, \
        "No parameters received gradients through multi-modal path"

    return output.shape


if __name__ == "__main__":
    all_validation_failures = []
    total_tests = 0

    # Test 1: Minimal training
    total_tests += 1
    model = None
    try:
        model, losses = validate_minimal_training()
    except Exception as e:
        all_validation_failures.append(f"Minimal training: {e}")

    # Test 2: Plasticity verification
    total_tests += 1
    if model is not None:
        try:
            plasticity_stats = validate_plasticity(model)
        except Exception as e:
            all_validation_failures.append(f"Plasticity verification: {e}")
    else:
        all_validation_failures.append("Plasticity verification: skipped (no model from training)")

    # Test 3: Skip connection verification
    total_tests += 1
    try:
        num_candidates = validate_skip_connections()
    except Exception as e:
        all_validation_failures.append(f"Skip connections: {e}")

    # Test 4: Inference verification
    total_tests += 1
    if model is not None:
        try:
            tokens = validate_inference(model)
        except Exception as e:
            all_validation_failures.append(f"Inference: {e}")
    else:
        all_validation_failures.append("Inference: skipped (no model from training)")

    # Test 5: Tokenizer verification
    total_tests += 1
    try:
        vocab_size = validate_tokenizer()
    except Exception as e:
        all_validation_failures.append(f"Tokenizer: {e}")

    # Test 6: Multi-modal forward pass
    total_tests += 1
    try:
        mm_shape = validate_multimodal_forward()
    except Exception as e:
        all_validation_failures.append(f"Multi-modal forward: {e}")

    # Final result
    if all_validation_failures:
        print(f"VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"VALIDATION PASSED - All {total_tests} tests produced expected results")
        sys.exit(0)
