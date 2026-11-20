"""
Pytest test suite for the MLX Neuroplasticity Engine components.

Tests feedforward, model, and plasticity modules with real MLX tensors.
No mocks -- all tests use actual MLX computations.

Docs: https://docs.pytest.org/

Run: ./venv/bin/python -m pytest tests/test_mlx_model.py -v
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mlx.core as mx

from mlx_adaptive_feedforward import MLXAdaptiveFeedForward, MovingStats
from mlx_model import MLXAdaptiveTransformer, MLXTransformerLayer, _count_params_flat
from mlx_plasticity import MLXPlasticityController


# ---------------------------------------------------------------------------
# Shared fixtures: small model dimensions for fast tests
# ---------------------------------------------------------------------------
VOCAB = 500
D_MODEL = 32
NHEAD = 4
NUM_LAYERS = 2
D_FF = 128
MAX_SEQ = 16


def _make_model(**overrides):
    """Create a small MLXAdaptiveTransformer with sensible test defaults."""
    defaults = dict(
        vocab_size=VOCAB,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        max_seq_len=MAX_SEQ,
    )
    defaults.update(overrides)
    return MLXAdaptiveTransformer(**defaults)


def _make_tokens(batch: int = 1, seq_len: int = 8) -> mx.array:
    """Random integer token tensor (B, T)."""
    return mx.array(
        [[i % VOCAB for i in range(seq_len)] for _ in range(batch)],
        dtype=mx.int32,
    )


# ===================================================================
# TestAdaptiveFeedForward
# ===================================================================
class TestAdaptiveFeedForward:
    """Tests for MLXAdaptiveFeedForward structural plasticity."""

    def test_forward_shape(self):
        ffn = MLXAdaptiveFeedForward(d_model=64, d_ff=256)
        x = mx.random.normal((2, 8, 64))
        out = ffn(x)
        mx.eval(out)
        assert out.shape == (2, 8, 64), f"Expected (2,8,64), got {out.shape}"

    def test_split_increases_dff(self):
        ffn = MLXAdaptiveFeedForward(d_model=D_MODEL, d_ff=D_FF)
        original = ffn.d_ff
        ffn.split_neuron(0)
        assert ffn.d_ff == original + 1

    def test_merge_decreases_dff(self):
        ffn = MLXAdaptiveFeedForward(d_model=D_MODEL, d_ff=D_FF)
        original = ffn.d_ff
        ffn.merge_neurons([0, 1])
        assert ffn.d_ff == original - 1

    def test_forward_after_structural_change(self):
        ffn = MLXAdaptiveFeedForward(d_model=D_MODEL, d_ff=D_FF)
        ffn.split_neuron(0)
        ffn.merge_neurons([0, 1])
        x = mx.random.normal((1, 4, D_MODEL))
        out = ffn(x)
        mx.eval(out)
        assert out.shape == (1, 4, D_MODEL)

    def test_mask_shapes_match_weights(self):
        ffn = MLXAdaptiveFeedForward(d_model=D_MODEL, d_ff=D_FF)
        ffn.split_neuron(0)
        ffn.merge_neurons([0, 1])
        assert ffn._input_mask.shape == ffn.input_layer.weight.shape, (
            f"Input mask {ffn._input_mask.shape} != weight {ffn.input_layer.weight.shape}"
        )
        assert ffn._output_mask.shape == ffn.output_layer.weight.shape, (
            f"Output mask {ffn._output_mask.shape} != weight {ffn.output_layer.weight.shape}"
        )

    def test_initial_sparsity_zero(self):
        ffn = MLXAdaptiveFeedForward(d_model=D_MODEL, d_ff=D_FF)
        assert abs(ffn.get_sparsity()) < 1e-6, (
            f"Fresh FFN sparsity should be 0, got {ffn.get_sparsity()}"
        )

    def test_training_tracks_activations(self):
        ffn = MLXAdaptiveFeedForward(d_model=D_MODEL, d_ff=64)
        ffn.train()
        x = mx.random.normal((1, 4, D_MODEL))
        out = ffn(x)
        # mlx.core.eval — materialize MLX lazy array (NOT Python eval)
        mx.eval(out)  # noqa: S307
        assert ffn.intermediate_activation_stats.count > 0, (
            "Training mode should track activation statistics"
        )

    def test_split_exact_preservation_without_noise(self):
        """Split with noise_scale=0 in eval mode is numerically identical."""
        ffn = MLXAdaptiveFeedForward(d_model=64, d_ff=256)
        ffn.eval()  # noqa: S307  — nn.Module.eval, not builtin
        x = mx.random.normal((2, 8, 64))
        mx.eval(x)  # noqa: S307
        out_before = ffn(x)
        mx.eval(out_before)  # noqa: S307
        ffn.split_neuron(10, noise_scale=0.0)
        out_after = ffn(x)
        mx.eval(out_after)  # noqa: S307
        max_diff = float(mx.max(mx.abs(out_before - out_after)).item())
        assert max_diff < 1e-5, (
            f"Split with noise_scale=0 should be exact, got diff={max_diff}"
        )

    def test_add_neurons_low_drift(self):
        """add_neurons with sparse masks produces small output drift."""
        ffn = MLXAdaptiveFeedForward(d_model=64, d_ff=256)
        ffn.eval()  # noqa: S307
        x = mx.random.normal((2, 8, 64))
        mx.eval(x)  # noqa: S307
        out_before = ffn(x)
        mx.eval(out_before)  # noqa: S307
        new_active = ffn.add_neurons(8, active_fraction=0.6)
        out_after = ffn(x)
        mx.eval(out_after)  # noqa: S307
        max_diff = float(mx.max(mx.abs(out_before - out_after)).item())
        assert max_diff < 5e-3, f"add_neurons drift too large: {max_diff}"
        assert new_active > 0, "Should introduce some active connections"

    def test_split_and_merge_preserve_output_bias_shape(self):
        """Output bias (d_model,) survives split and merge."""
        ffn = MLXAdaptiveFeedForward(d_model=64, d_ff=128)
        assert ffn.output_layer.bias.shape == (64,)
        ffn.split_neuron(0)
        assert ffn.output_layer.bias.shape == (64,), (
            f"Bias shape after split: {ffn.output_layer.bias.shape}"
        )
        ffn.merge_neurons([0, 1])
        assert ffn.output_layer.bias.shape == (64,), (
            f"Bias shape after merge: {ffn.output_layer.bias.shape}"
        )


# ===================================================================
# TestTransformerModel
# ===================================================================
class TestTransformerModel:
    """Tests for MLXAdaptiveTransformer."""

    def test_forward_shape(self):
        model = _make_model()
        tokens = _make_tokens(batch=1, seq_len=8)
        logits = model(tokens)
        mx.eval(logits)
        assert logits.shape == (1, 8, VOCAB), f"Got {logits.shape}"

    def test_batch_forward(self):
        model = _make_model()
        tokens = _make_tokens(batch=3, seq_len=6)
        logits = model(tokens)
        mx.eval(logits)
        assert logits.shape == (3, 6, VOCAB), f"Got {logits.shape}"

    def test_weight_tying(self):
        model = _make_model()
        assert model.embedding.weight is model.output_proj.weight, (
            "Embedding and output projection weights should be tied"
        )

    def test_per_layer_dff(self):
        per_layer = [64, 256]
        model = _make_model(d_ff_per_layer=per_layer)
        actual = [layer.ffn.d_ff for layer in model.layers]
        assert actual == per_layer, f"Expected {per_layer}, got {actual}"

    def test_architecture_info(self):
        model = _make_model()
        info = model.get_architecture_info()
        assert info["vocab_size"] == VOCAB
        assert info["d_model"] == D_MODEL
        assert info["nhead"] == NHEAD
        assert info["num_layers"] == NUM_LAYERS
        assert info["max_seq_len"] == MAX_SEQ
        assert len(info["d_ff_per_layer"]) == NUM_LAYERS

    def test_parameter_counting(self):
        model = _make_model()
        count = model.count_parameters()
        assert isinstance(count, int) and count > 0, (
            f"Expected positive int, got {count}"
        )

    def test_checkpoint_roundtrip(self):
        model = _make_model()
        tokens = _make_tokens(batch=1, seq_len=6)
        original_logits = model(tokens)
        mx.eval(original_logits)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = os.path.join(tmpdir, "ckpt")
            model.save_checkpoint(ckpt_dir)
            loaded = MLXAdaptiveTransformer.load_checkpoint(ckpt_dir)

        loaded_logits = loaded(tokens)
        mx.eval(loaded_logits)
        assert loaded_logits.shape == original_logits.shape, (
            f"Loaded shape {loaded_logits.shape} != original {original_logits.shape}"
        )
        # Verify weights match
        orig_w = model.layers[0].ffn.input_layer.weight
        load_w = loaded.layers[0].ffn.input_layer.weight
        diff = float(mx.max(mx.abs(orig_w - load_w)).item())
        assert diff < 1e-5, f"Weight diff after roundtrip: {diff}"


# ===================================================================
# TestPlasticity
# ===================================================================
class TestPlasticity:
    """Tests for MLXPlasticityController."""

    def _make_trained_model(self):
        """Create a model with activation stats populated (training mode forward)."""
        model = _make_model()
        model.train()
        tokens = _make_tokens(batch=2, seq_len=8)
        out = model(tokens)
        mx.eval(out)
        return model

    def test_step_returns_events(self):
        model = self._make_trained_model()
        controller = MLXPlasticityController(
            prune_frequency=1, structural_frequency=1
        )
        events = controller.step(model)
        required = {"pruned", "grown", "splits", "merges"}
        assert required.issubset(events.keys()), (
            f"Missing keys: {required - events.keys()}"
        )
        for key in required:
            assert isinstance(events[key], int), (
                f"events['{key}'] should be int, got {type(events[key])}"
            )

    def test_prune_reduces_connections(self):
        model = self._make_trained_model()
        controller = MLXPlasticityController(
            prune_threshold=0.5,
            prune_frequency=1,
            min_connections=10,
        )
        ffn0 = model.layers[0].ffn
        before = ffn0.get_active_connections()
        pruned = controller._prune_connections(model)
        after = ffn0.get_active_connections()
        if pruned > 0:
            assert after < before, (
                f"Pruned {pruned} but connections didn't decrease: "
                f"{before} -> {after}"
            )

    def test_snapshot_format(self):
        model = self._make_trained_model()
        controller = MLXPlasticityController()
        snapshot = controller.capture_snapshot(
            model, step=42, train_loss=3.5, val_loss=4.0
        )
        required_keys = {
            "step", "train_loss", "val_loss", "perplexity",
            "total_model_params", "global_sparsity", "d_ff_per_layer",
            "cumulative_splits", "cumulative_merges",
            "cumulative_grown", "cumulative_pruned", "layer_details",
        }
        missing = required_keys - snapshot.keys()
        assert not missing, f"Snapshot missing keys: {missing}"
        assert snapshot["step"] == 42
        assert snapshot["train_loss"] == 3.5
        assert snapshot["val_loss"] == 4.0
        assert len(snapshot["d_ff_per_layer"]) == NUM_LAYERS

    def test_model_valid_after_plasticity(self):
        model = self._make_trained_model()
        controller = MLXPlasticityController(
            prune_threshold=0.5,
            split_threshold=0.001,
            prune_frequency=1,
            structural_frequency=1,
            min_connections=10,
        )
        # Run a plasticity step (prune + grow + split + merge)
        events = controller.step(model)
        # Model should still produce valid output
        model.eval()  # noqa: S307
        tokens = _make_tokens(batch=1, seq_len=6)
        logits = model(tokens)
        mx.eval(logits)  # noqa: S307
        assert logits.shape[0] == 1
        assert logits.shape[1] == 6
        assert logits.shape[2] == VOCAB

    def test_convergence_split_selects_distinct_candidates(self):
        """_select_split_candidates returns distinct indices in descending order."""
        ffn = MLXAdaptiveFeedForward(d_model=D_MODEL, d_ff=D_FF)
        # Seed deterministic activation EMA
        ema = mx.zeros((D_FF,))
        ema = mx.concatenate([
            mx.array([0.9, 0.8, 0.1, 0.7]),
            mx.zeros((D_FF - 4,)),
        ])
        mx.eval(ema)  # noqa: S307
        ffn._persistent_mean_activation = ema
        candidates = MLXPlasticityController._select_split_candidates(ffn, 3)
        assert len(candidates) == 3, f"Expected 3 candidates, got {candidates}"
        assert candidates[0] == 0, f"Highest activation should be idx 0, got {candidates[0]}"
        assert candidates[1] == 1, f"Second highest should be idx 1, got {candidates[1]}"
        assert candidates[2] == 3, f"Third highest should be idx 3, got {candidates[2]}"
        assert len(set(candidates)) == 3, f"Candidates not distinct: {candidates}"

    def test_convergence_split_does_not_repeat_zero(self):
        """With max_splits_per_layer > 1, distinct neurons are split."""
        model = self._make_trained_model()
        # Set deterministic activation EMA on all layers
        for layer in model.layers:
            ema = mx.arange(D_FF, dtype=mx.float32) / D_FF  # ascending
            mx.eval(ema)  # noqa: S307  -- MLX array materialization, not Python eval
            layer.ffn._persistent_mean_activation = ema
        controller = MLXPlasticityController(
            convergence_delta=0.1, convergence_window=4,
            max_splits_per_layer=3, max_split_layers=1,
            neurogenesis_count=0,
            prune_frequency=100, structural_frequency=1,
            warmup_steps=0,
        )
        controller._val_loss_history = [5.0, 5.01, 4.99, 5.02]
        old_dffs = [layer.ffn.d_ff for layer in model.layers]
        split_result = controller._split_neurons(model)
        splits = split_result["count"]
        assert splits == 3, f"Expected 3 splits, got {splits}"
        # Exactly one layer should have grown by 3
        new_dffs = [layer.ffn.d_ff for layer in model.layers]
        deltas = [n - o for n, o in zip(new_dffs, old_dffs)]
        assert sum(deltas) == 3, f"Expected total d_ff growth of 3, got {deltas}"
        assert any(d == 3 for d in deltas), f"Expected one layer +3, got {deltas}"

    def test_reinit_preserves_mask_count(self):
        """Reinitializing dormant neurons does not change active connection count."""
        model = self._make_trained_model()
        ffn0 = model.layers[0].ffn
        # Force 2 dormant neurons
        ffn0._persistent_mean_activation = mx.concatenate([
            mx.zeros((2,)), mx.ones((D_FF - 2,)),
        ])
        mx.eval(ffn0._persistent_mean_activation)  # noqa: S307
        before_active = ffn0.get_active_connections()
        controller = MLXPlasticityController(
            dormant_threshold=0.001, structural_frequency=1,
            prune_frequency=100, warmup_steps=0,
            min_age_for_removal=99999,
            max_reinit_per_event=2,  # Override cap for this test
        )
        reinit_count, removed_count = controller._handle_dormant_neurons(model, 100)
        after_active = ffn0.get_active_connections()
        assert reinit_count == 2, f"Expected 2 reinits, got {reinit_count}"
        assert removed_count == 0
        assert after_active == before_active, (
            f"Active connections changed: {before_active} -> {after_active}"
        )

    def test_reinit_clears_persistent_activation(self):
        """After reinit, _persistent_mean_activation is None."""
        model = self._make_trained_model()
        ffn0 = model.layers[0].ffn
        ffn0._persistent_mean_activation = mx.concatenate([
            mx.zeros((1,)), mx.ones((D_FF - 1,)),
        ])
        mx.eval(ffn0._persistent_mean_activation)  # noqa: S307
        controller = MLXPlasticityController(
            dormant_threshold=0.001, structural_frequency=1,
            prune_frequency=100, warmup_steps=0,
            min_age_for_removal=99999,
        )
        controller._handle_dormant_neurons(model, 100)
        assert ffn0._persistent_mean_activation is None, (
            "EMA should be cleared after reinit to prevent re-trigger"
        )

    def test_remove_then_reinit_uses_remapped_indices(self):
        """When both removal and reinit happen, reinit targets correct neurons."""
        indices = [2, 5, 7]  # original to-reinit indices
        removed = [1, 4]     # removed before reinit
        remapped = MLXPlasticityController._remap_indices_after_removals(
            indices, removed,
        )
        # idx 2: 1 removed below (idx 1) → 2-1 = 1
        # idx 5: 2 removed below (idx 1, 4) → 5-2 = 3
        # idx 7: 2 removed below (idx 1, 4) → 7-2 = 5
        assert remapped == [1, 3, 5], f"Expected [1, 3, 5], got {remapped}"

    def test_structural_budget_debt_blocks_growth(self):
        """Structural densification creates debt that blocks growth."""
        controller = MLXPlasticityController(
            prune_frequency=1, structural_frequency=100,
            warmup_steps=0, target_sparsity=0.4,
        )
        # Simulate: split added 200 active connections
        controller._adjust_structural_budget(200)
        assert controller._structural_budget_debt == 200
        # Now prune 100 — debt should absorb it, no growth budget
        controller._prune_budget = 0
        controller._prune_budget += 100
        payoff = min(controller._prune_budget, controller._structural_budget_debt)
        controller._prune_budget -= payoff
        controller._structural_budget_debt -= payoff
        assert controller._prune_budget == 0, "All budget should go to debt"
        assert controller._structural_budget_debt == 100, "100 debt remains"

    def test_negative_structural_delta_creates_budget_credit(self):
        """Removal/merge reducing active connections creates budget credit."""
        controller = MLXPlasticityController()
        controller._structural_budget_debt = 50
        # Remove reduced 80 active connections
        controller._adjust_structural_budget(-80)
        # 50 pays debt, 30 becomes prune budget
        assert controller._structural_budget_debt == 0
        assert controller._prune_budget == 30
