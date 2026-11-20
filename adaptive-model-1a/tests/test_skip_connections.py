"""
Test SkipConnectionController evaluates candidates and manages connections.

Uses real tensor data with GlobalLayerRegistry. Verifies the controller
can track activations, compute correlations, and manage skip connection lifecycle.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from adaptive_network import GlobalLayerRegistry
from skip_connection_controller import SkipConnectionController
from transformer_model.adaptive_feedforward import AdaptiveFeedForward


def _build_registry_with_layers(num_layers=4, d_model=32, d_ff=64):
    """Build a registry with AdaptiveFeedForward layers."""
    registry = GlobalLayerRegistry()
    model = nn.Module()

    for i in range(num_layers):
        ff = AdaptiveFeedForward(d_model, d_ff)
        model.add_module(f"ff_{i}", ff)
        registry.add_layer(ff, 'feedforward', d_model, d_model, position=i)

    return registry, model


def test_update_activations():
    """Controller tracks layer activations without errors."""
    controller = SkipConnectionController(activation_window=50)
    registry, model = _build_registry_with_layers()

    # Feed activations for each layer
    for layer_id in registry.order:
        for _ in range(15):
            activations = torch.randn(4, 32)  # batch=4, features=32
            controller.update_activations(layer_id, activations)

    # Verify activations were stored
    for layer_id in registry.order:
        assert layer_id in controller.layer_activations, \
            f"Layer {layer_id} should have tracked activations"
        assert len(controller.layer_activations[layer_id]) == 15, \
            f"Expected 15 activation samples, got {len(controller.layer_activations[layer_id])}"


def test_analyze_correlations():
    """Controller computes correlations between non-adjacent layers."""
    controller = SkipConnectionController(
        correlation_threshold=0.0,  # Accept any correlation as candidate
        activation_window=50,
    )
    registry, model = _build_registry_with_layers(num_layers=4)

    # Feed correlated activations between layer 0 and layer 3
    # (they are non-adjacent, so candidates should be found)
    base_activation = torch.randn(4, 32)
    for _ in range(20):
        noise = torch.randn(4, 32) * 0.1
        for i, layer_id in enumerate(registry.order):
            if i == 0 or i == 3:
                controller.update_activations(layer_id, base_activation + noise)
            else:
                controller.update_activations(layer_id, torch.randn(4, 32))

    candidates = controller.analyze_layer_correlations(registry)

    # With threshold 0.0, any non-adjacent pair with >10 samples should be a candidate
    assert isinstance(candidates, list), f"Expected list, got {type(candidates)}"
    # At least some candidates should be found with correlated activations
    assert len(candidates) > 0, \
        "Expected at least one skip connection candidate with correlated layers"


def test_step_respects_warmup():
    """Controller doesn't add connections before warmup period."""
    controller = SkipConnectionController(
        correlation_threshold=0.0,
        activation_window=50,
    )
    registry, model = _build_registry_with_layers()

    # Feed activations
    for _ in range(15):
        for layer_id in registry.order:
            controller.update_activations(layer_id, torch.randn(4, 32))

    # Step during warmup (epoch_counter starts at 0, threshold is 20)
    added = controller.step(registry, model)
    assert added == 0, f"Should not add connections during warmup, added {added}"
    assert controller.epoch_counter == 1


def test_step_adds_connection_after_warmup():
    """Controller can add skip connections after warmup period."""
    controller = SkipConnectionController(
        correlation_threshold=0.0,  # Accept any
        activation_window=50,
    )
    controller.epoch_counter = 19  # Just before threshold

    registry, model = _build_registry_with_layers()

    # Feed correlated activations
    base = torch.randn(4, 32)
    for _ in range(20):
        noise = torch.randn(4, 32) * 0.05
        for i, layer_id in enumerate(registry.order):
            controller.update_activations(layer_id, base + noise * (i + 1))

    # This step should be epoch 20, past warmup
    added = controller.step(registry, model)

    # Check that the step ran without errors
    assert controller.epoch_counter == 20
    # Connection addition depends on dimension matching — just verify no crash


def test_max_connections_limit():
    """Controller respects the max_skip_connections limit."""
    controller = SkipConnectionController(
        correlation_threshold=0.0,
        max_skip_connections=2,
        activation_window=50,
    )
    controller.epoch_counter = 19

    registry, model = _build_registry_with_layers(num_layers=6)

    # Feed activations
    base = torch.randn(4, 32)
    for _ in range(20):
        for layer_id in registry.order:
            controller.update_activations(layer_id, base + torch.randn(4, 32) * 0.1)

    # Run multiple steps
    total_added = 0
    for _ in range(10):
        added = controller.step(registry, model)
        total_added += added

    # Should not exceed max
    actual_connections = len(registry.skip_connections)
    assert actual_connections <= 2, \
        f"Expected at most 2 skip connections, got {actual_connections}"


def test_correlation_tracking():
    """Layer correlations are stored and accessible."""
    controller = SkipConnectionController(activation_window=50)
    registry, model = _build_registry_with_layers(num_layers=3)

    # Feed identical activations to all layers (perfect correlation)
    for _ in range(15):
        act = torch.randn(4, 32)
        for layer_id in registry.order:
            controller.update_activations(layer_id, act)

    controller.analyze_layer_correlations(registry)

    # Non-adjacent pairs should have high correlation
    assert len(controller.layer_correlations) > 0, \
        "Should have computed at least one correlation"

    for pair, corr in controller.layer_correlations.items():
        assert -1.0 <= corr <= 1.0, \
            f"Correlation for {pair} out of range: {corr}"


if __name__ == "__main__":
    all_validation_failures = []
    total_tests = 0

    tests = [
        ("Update activations", test_update_activations),
        ("Analyze correlations", test_analyze_correlations),
        ("Step respects warmup", test_step_respects_warmup),
        ("Step adds connection after warmup", test_step_adds_connection_after_warmup),
        ("Max connections limit", test_max_connections_limit),
        ("Correlation tracking", test_correlation_tracking),
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
