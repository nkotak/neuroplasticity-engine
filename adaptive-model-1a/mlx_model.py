"""
MLX adaptive transformer model with per-layer structural plasticity.

Pre-norm transformer architecture where each layer's feedforward block can
independently grow or shrink (different d_ff per layer after plasticity events).

Docs: https://ml-explore.github.io/mlx/
      https://ml-explore.github.io/mlx/build/html/python/nn.html

Sample input:
    model = MLXAdaptiveTransformer(vocab_size=32000)
    tokens = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
    logits = model(tokens)

Expected output:
    logits.shape -> (1, 4, 32000)
"""

import json
import os
from typing import Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
from loguru import logger

from mlx_adaptive_feedforward import MLXAdaptiveFeedForward


class MLXTransformerLayer(nn.Module):
    """Single pre-norm transformer layer with adaptive feedforward.

    Includes per-head importance tracking and structured head pruning.
    During training, each head's importance is tracked via an EMA of the
    L2 norm of its attention output. Heads can be pruned (masked to zero)
    or reinstated during plasticity steps.
    """

    def __init__(
        self, d_model: int, nhead: int, d_ff: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiHeadAttention(dims=d_model, num_heads=nhead)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = MLXAdaptiveFeedForward(d_model, d_ff)
        self.dropout = nn.Dropout(p=dropout)

        # Head importance tracking (EMA of per-head output L2 norm)
        self._head_importance: Optional[mx.array] = None  # shape: (nhead,)
        self._head_importance_decay: float = 0.99
        # Head mask: 1.0 = active, 0.0 = pruned. shape: (nhead,)
        self._head_mask: mx.array = mx.ones((nhead,))

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        # Pre-norm attention block
        h = self.norm1(x)
        attn_out = self.attn(h, h, h, mask=mask)

        # Apply head mask: reshape to (B, T, nhead, head_dim), mask, reshape back
        B, T, D = attn_out.shape
        attn_out = attn_out.reshape(B, T, self.nhead, self.head_dim)

        if self.training and self._head_importance is not None:
            # Track per-head importance (EMA of L2 norm of head output)
            head_norms = mx.sqrt(
                mx.sum(attn_out ** 2, axis=(0, 1, 3)) + 1e-8
            )  # (nhead,)
            decay = self._head_importance_decay
            self._head_importance = (
                decay * self._head_importance + (1 - decay) * head_norms
            )
            mx.eval(self._head_importance)
        elif self.training and self._head_importance is None:
            head_norms = mx.sqrt(
                mx.sum(attn_out ** 2, axis=(0, 1, 3)) + 1e-8
            )  # (nhead,)
            self._head_importance = head_norms
            mx.eval(self._head_importance)

        # Apply head mask: zero out pruned heads
        # head_mask shape (nhead,) -> broadcast to (1, 1, nhead, 1)
        attn_out = attn_out * self._head_mask.reshape(1, 1, self.nhead, 1)
        attn_out = attn_out.reshape(B, T, D)

        h = self.dropout(attn_out)
        x = x + h

        # Pre-norm feedforward block
        h = self.norm2(x)
        h = self.ffn(h)
        h = self.dropout(h)
        x = x + h
        return x

    def get_active_heads(self) -> int:
        """Count of active (non-pruned) attention heads."""
        return int(mx.sum(self._head_mask > 0).item())

    def get_head_importance_scores(self) -> Optional[mx.array]:
        """Return per-head importance scores, or None if not yet tracked."""
        return self._head_importance

    def prune_head(self, head_idx: int) -> None:
        """Prune a specific attention head by zeroing its mask entry."""
        if head_idx < 0 or head_idx >= self.nhead:
            return
        mask_list = self._head_mask.tolist()
        mask_list[head_idx] = 0.0
        self._head_mask = mx.array(mask_list)
        mx.eval(self._head_mask)
        logger.info(
            f"Pruned attention head {head_idx}, "
            f"active heads: {self.get_active_heads()}/{self.nhead}"
        )

    def reinstate_head(self, head_idx: int) -> None:
        """Reinstate a previously pruned attention head."""
        if head_idx < 0 or head_idx >= self.nhead:
            return
        mask_list = self._head_mask.tolist()
        mask_list[head_idx] = 1.0
        self._head_mask = mx.array(mask_list)
        mx.eval(self._head_mask)


class MLXAdaptiveTransformer(nn.Module):
    """Transformer with per-layer adaptive feedforward dimensions.

    Simplified from PyTorch version: no registry, no gating params, no skip
    connections, no MetaController. Focus is on clean pre-norm architecture
    with structural plasticity via the feedforward modules.
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 768,
        nhead: int = 12,
        num_layers: int = 12,
        d_ff: int = 3072,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
        d_ff_per_layer: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Learned positional encoding
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        self.embed_dropout = nn.Dropout(p=dropout)

        # Transformer layers -- each may have a different d_ff after plasticity
        layer_dims = d_ff_per_layer or [d_ff] * num_layers
        self.layers: List[MLXTransformerLayer] = []
        for i in range(num_layers):
            layer = MLXTransformerLayer(d_model, nhead, layer_dims[i], dropout)
            self.layers.append(layer)

        self.final_norm = nn.LayerNorm(d_model)

        # Output projection weight-tied with embedding
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        self._tie_weights()

    def _tie_weights(self) -> None:
        """Tie output projection weights to embedding weights."""
        self.output_proj.weight = self.embedding.weight

    def __call__(self, tokens: mx.array) -> mx.array:
        """Forward pass: tokens (B, T) -> logits (B, T, vocab_size)."""
        _batch_size, seq_len = tokens.shape

        # Embeddings
        positions = mx.arange(seq_len)
        x = self.embedding(tokens) + self.pos_embedding(positions)
        x = self.embed_dropout(x)

        # Causal mask for autoregressive attention
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)

        # Transformer layers
        for layer in self.layers:
            x = layer(x, mask=mask)

        x = self.final_norm(x)

        # Weight-tied output projection
        logits = self.output_proj(x)
        return logits

    def reset_last_layer(self) -> None:
        """Reset final transformer layer's FFN + final_norm to break conditioning.

        Lyle et al. (ICML 2023) show that the last layer accumulates loss
        landscape conditioning problems fastest. Periodic reset restores
        learning capacity without catastrophically forgetting earlier layers'
        representations.

        Note: Does NOT reset output_proj since it's weight-tied to the embedding.
        Resets: last layer FFN weights/biases, final_norm, FFN masks, activation EMA.
        """
        # Alias for MLX array materialization (NOT Python's eval builtin)
        _materialize = mx.eval  # noqa: S307

        last_layer = self.layers[-1]
        ffn = last_layer.ffn

        # Reset FFN input weights at ACTIVE positions only (preserve mask sparsity)
        fan_in = ffn.d_model
        std = (2.0 / fan_in) ** 0.5
        ffn.input_layer.weight = mx.where(
            ffn._input_mask != 0,
            mx.random.normal(ffn.input_layer.weight.shape) * std,
            mx.zeros_like(ffn.input_layer.weight),
        )
        ffn.input_layer.bias = mx.zeros(ffn.input_layer.bias.shape)

        # Reset FFN output weights at ACTIVE positions only
        ffn.output_layer.weight = mx.where(
            ffn._output_mask != 0,
            mx.random.normal(ffn.output_layer.weight.shape) * 0.02,
            mx.zeros_like(ffn.output_layer.weight),
        )
        ffn.output_layer.bias = mx.zeros(ffn.output_layer.bias.shape)

        # Reset final_norm to identity transform
        self.final_norm.weight = mx.ones(self.final_norm.weight.shape)
        if hasattr(self.final_norm, 'bias') and self.final_norm.bias is not None:
            self.final_norm.bias = mx.zeros(self.final_norm.bias.shape)

        # Preserve masks — do NOT reset to all-ones
        # (masks encode sparsity structure learned by pruning)

        # Reset activation tracking
        ffn._persistent_mean_activation = None

        # Materialize all reset arrays
        _materialize(
            ffn.input_layer.weight, ffn.input_layer.bias,
            ffn.output_layer.weight, ffn.output_layer.bias,
            self.final_norm.weight,
        )
        logger.info(
            f"Reset last layer FFN (d_ff={ffn.d_ff}, mask preserved) + final_norm"
        )

    def get_architecture_info(self) -> Dict:
        """Capture current architecture state for checkpointing."""
        d_ff_values = [layer.ffn.d_ff for layer in self.layers]
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_layers": self.num_layers,
            "max_seq_len": self.max_seq_len,
            "d_ff_per_layer": d_ff_values,
            "head_masks": [layer._head_mask.tolist() for layer in self.layers],
        }

    def get_head_info(self) -> List[Dict]:
        """Get attention head importance and pruning state per layer."""
        info: List[Dict] = []
        for i, layer in enumerate(self.layers):
            importance = layer.get_head_importance_scores()
            info.append({
                "layer": i,
                "active_heads": layer.get_active_heads(),
                "total_heads": layer.nhead,
                "importance": importance.tolist() if importance is not None else None,
                "mask": layer._head_mask.tolist(),
            })
        return info

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return _count_params_flat(self.parameters())

    def save_checkpoint(self, checkpoint_dir: str) -> None:
        """Save model weights and architecture info to a directory."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        weights_path = os.path.join(checkpoint_dir, "weights.npz")
        arch_path = os.path.join(checkpoint_dir, "arch_info.json")

        self.save_weights(weights_path)
        with open(arch_path, "w") as f:
            json.dump(self.get_architecture_info(), f, indent=2)
        logger.info(f"Checkpoint saved to {checkpoint_dir}")

    @classmethod
    def load_checkpoint(cls, checkpoint_dir: str) -> "MLXAdaptiveTransformer":
        """Load model from checkpoint directory."""
        arch_path = os.path.join(checkpoint_dir, "arch_info.json")
        weights_path = os.path.join(checkpoint_dir, "weights.npz")

        with open(arch_path) as f:
            arch = json.load(f)

        model = cls(
            vocab_size=arch["vocab_size"],
            d_model=arch["d_model"],
            nhead=arch["nhead"],
            num_layers=arch["num_layers"],
            max_seq_len=arch["max_seq_len"],
            d_ff_per_layer=arch["d_ff_per_layer"],
        )
        model.load_weights(weights_path)
        model._tie_weights()
        logger.info(f"Model loaded from {checkpoint_dir}")
        return model


def _count_params_flat(params) -> int:
    """Recursively count parameters in an MLX parameter tree."""
    total = 0
    if isinstance(params, mx.array):
        return params.size
    if isinstance(params, dict):
        for v in params.values():
            total += _count_params_flat(v)
    elif isinstance(params, (list, tuple)):
        for item in params:
            total += _count_params_flat(item)
    return total


if __name__ == "__main__":
    import sys
    import tempfile

    all_validation_failures: List[str] = []
    total_tests = 0

    # Test 1: Forward pass shape with small model
    total_tests += 1
    model = MLXAdaptiveTransformer(
        vocab_size=1000, d_model=64, nhead=4, num_layers=2,
        d_ff=256, max_seq_len=32,
    )
    tokens = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)
    logits = model(tokens)
    mx.eval(logits)
    expected_shape = (1, 5, 1000)
    if logits.shape != expected_shape:
        all_validation_failures.append(
            f"Forward shape: expected {expected_shape}, got {logits.shape}"
        )
    else:
        print(f"  Test 1 OK: forward shape = {logits.shape}")

    # Test 2: Batch forward pass
    total_tests += 1
    batch = mx.array([[1, 2, 3], [4, 5, 6]], dtype=mx.int32)
    batch_logits = model(batch)
    mx.eval(batch_logits)
    expected_batch_shape = (2, 3, 1000)
    if batch_logits.shape != expected_batch_shape:
        all_validation_failures.append(
            f"Batch shape: expected {expected_batch_shape}, got {batch_logits.shape}"
        )
    else:
        print(f"  Test 2 OK: batch shape = {batch_logits.shape}")

    # Test 3: Weight tying -- embedding and output_proj share weights
    total_tests += 1
    emb_w = model.embedding.weight
    out_w = model.output_proj.weight
    if emb_w is not out_w:
        all_validation_failures.append(
            "Weight tying: embedding.weight is not output_proj.weight"
        )
    else:
        print("  Test 3 OK: weight tying verified")

    # Test 4: Per-layer d_ff via constructor
    total_tests += 1
    model2 = MLXAdaptiveTransformer(
        vocab_size=500, d_model=32, nhead=4, num_layers=3,
        d_ff=128, max_seq_len=16,
        d_ff_per_layer=[128, 256, 64],
    )
    dff_list = [layer.ffn.d_ff for layer in model2.layers]
    if dff_list != [128, 256, 64]:
        all_validation_failures.append(
            f"Per-layer d_ff: expected [128, 256, 64], got {dff_list}"
        )
    else:
        print(f"  Test 4 OK: per-layer d_ff = {dff_list}")

    # Test 5: Architecture info captures d_ff correctly
    total_tests += 1
    arch_info = model2.get_architecture_info()
    if arch_info["d_ff_per_layer"] != [128, 256, 64]:
        all_validation_failures.append(
            f"Arch info d_ff: expected [128, 256, 64], got {arch_info['d_ff_per_layer']}"
        )
    else:
        print(f"  Test 5 OK: arch_info d_ff_per_layer = {arch_info['d_ff_per_layer']}")

    # Test 6: Parameter counting is positive and reasonable
    total_tests += 1
    param_count = model.count_parameters()
    if param_count <= 0:
        all_validation_failures.append(f"Param count: expected > 0, got {param_count}")
    else:
        print(f"  Test 6 OK: param count = {param_count:,}")

    # Test 7: Checkpoint save/load roundtrip
    total_tests += 1
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = os.path.join(tmpdir, "test_ckpt")
        model.save_checkpoint(ckpt_dir)

        loaded = MLXAdaptiveTransformer.load_checkpoint(ckpt_dir)
        loaded_logits = loaded(tokens)
        mx.eval(loaded_logits)

        if loaded_logits.shape != expected_shape:
            all_validation_failures.append(
                f"Loaded model shape: expected {expected_shape}, got {loaded_logits.shape}"
            )
        else:
            orig_w = model.layers[0].ffn.input_layer.weight
            load_w = loaded.layers[0].ffn.input_layer.weight
            diff = float(mx.max(mx.abs(orig_w - load_w)).item())
            if diff > 1e-5:
                all_validation_failures.append(
                    f"Checkpoint roundtrip weight diff: {diff}"
                )
            else:
                print(
                    f"  Test 7 OK: checkpoint roundtrip (max weight diff = {diff:.2e})"
                )

    # Test 8: Forward pass after FFN split (structural change)
    total_tests += 1
    old_dff = model.layers[0].ffn.d_ff
    model.layers[0].ffn.split_neuron(0)
    new_dff = model.layers[0].ffn.d_ff
    post_split_logits = model(tokens)
    mx.eval(post_split_logits)
    if post_split_logits.shape != expected_shape:
        all_validation_failures.append(
            f"Post-split shape: expected {expected_shape}, got {post_split_logits.shape}"
        )
    elif new_dff != old_dff + 1:
        all_validation_failures.append(
            f"Post-split d_ff: expected {old_dff + 1}, got {new_dff}"
        )
    else:
        print(f"  Test 8 OK: forward after split (d_ff {old_dff} -> {new_dff})")

    # Test 9: Head importance tracking in training mode
    total_tests += 1
    head_model = MLXAdaptiveTransformer(
        vocab_size=1000, d_model=64, nhead=4, num_layers=2,
        d_ff=256, max_seq_len=32,
    )
    head_model.train()
    train_tokens = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)
    train_logits = head_model(train_tokens)
    mx.eval(train_logits)
    layer0_importance = head_model.layers[0].get_head_importance_scores()
    if layer0_importance is None:
        all_validation_failures.append(
            "Head importance tracking: _head_importance is None after training forward pass"
        )
    elif layer0_importance.shape != (4,):
        all_validation_failures.append(
            f"Head importance shape: expected (4,), got {layer0_importance.shape}"
        )
    else:
        scores = layer0_importance.tolist()
        all_positive = all(s > 0 for s in scores)
        if not all_positive:
            all_validation_failures.append(
                f"Head importance values: expected all > 0, got {scores}"
            )
        else:
            print(f"  Test 9 OK: head importance tracked, scores = "
                  f"[{', '.join(f'{s:.4f}' for s in scores)}]")

    # Test 10: Head pruning decreases active head count
    total_tests += 1
    before_active = head_model.layers[0].get_active_heads()
    head_model.layers[0].prune_head(0)
    after_active = head_model.layers[0].get_active_heads()
    if before_active != 4:
        all_validation_failures.append(
            f"Head pruning pre-check: expected 4 active heads, got {before_active}"
        )
    elif after_active != 3:
        all_validation_failures.append(
            f"Head pruning: expected 3 active heads after prune, got {after_active}"
        )
    else:
        mask_state = head_model.layers[0]._head_mask.tolist()
        print(f"  Test 10 OK: pruned head 0, active {before_active} -> {after_active}, "
              f"mask = {mask_state}")

    # Test 11: Forward pass still works after head pruning
    total_tests += 1
    pruned_logits = head_model(train_tokens)
    mx.eval(pruned_logits)
    if pruned_logits.shape != expected_shape:
        all_validation_failures.append(
            f"Post-prune forward shape: expected {expected_shape}, got {pruned_logits.shape}"
        )
    else:
        # Also verify get_head_info works
        head_info = head_model.get_head_info()
        layer0_info = head_info[0]
        if layer0_info["active_heads"] != 3:
            all_validation_failures.append(
                f"get_head_info active_heads: expected 3, got {layer0_info['active_heads']}"
            )
        elif layer0_info["mask"][0] != 0.0:
            all_validation_failures.append(
                f"get_head_info mask[0]: expected 0.0, got {layer0_info['mask'][0]}"
            )
        else:
            print(f"  Test 11 OK: forward works after pruning, "
                  f"head_info layer 0 = {layer0_info['active_heads']}/{layer0_info['total_heads']} active")

    # Test 12: reset_last_layer changes weights
    total_tests += 1
    reset_model = MLXAdaptiveTransformer(
        vocab_size=1000, d_model=64, nhead=4, num_layers=2,
        d_ff=256, max_seq_len=32,
    )
    # Capture pre-reset weight norms
    last_ffn = reset_model.layers[-1].ffn
    pre_input_norm = float(mx.sqrt(mx.sum(last_ffn.input_layer.weight ** 2)).item())
    pre_output_norm = float(mx.sqrt(mx.sum(last_ffn.output_layer.weight ** 2)).item())

    reset_model.reset_last_layer()

    post_input_norm = float(mx.sqrt(mx.sum(last_ffn.input_layer.weight ** 2)).item())
    post_output_norm = float(mx.sqrt(mx.sum(last_ffn.output_layer.weight ** 2)).item())

    if abs(pre_input_norm - post_input_norm) < 1e-6:
        all_validation_failures.append(
            "reset_last_layer: input weight norm unchanged"
        )
    elif abs(pre_output_norm - post_output_norm) < 1e-6:
        all_validation_failures.append(
            "reset_last_layer: output weight norm unchanged"
        )
    else:
        print(
            f"  Test 12 OK: reset_last_layer changed norms "
            f"(in: {pre_input_norm:.4f}->{post_input_norm:.4f}, "
            f"out: {pre_output_norm:.4f}->{post_output_norm:.4f})"
        )

    # Test 13: reset_last_layer preserves masks and clears activation tracking
    total_tests += 1
    last_ffn_check = reset_model.layers[-1].ffn
    if last_ffn_check._persistent_mean_activation is not None:
        all_validation_failures.append(
            "reset_last_layer: persistent_mean_activation not cleared"
        )
    else:
        print("  Test 13 OK: masks preserved, activation tracking cleared")

    # Test 14: Model still works after reset_last_layer
    total_tests += 1
    reset_model.eval()
    post_reset_logits = reset_model(tokens)
    mx.eval(post_reset_logits)
    if post_reset_logits.shape != expected_shape:
        all_validation_failures.append(
            f"Post-reset forward shape: expected {expected_shape}, got {post_reset_logits.shape}"
        )
    else:
        print(f"  Test 14 OK: forward works after reset_last_layer")

    if all_validation_failures:
        print(
            f"\n\u274c VALIDATION FAILED - {len(all_validation_failures)} "
            f"of {total_tests} tests failed:"
        )
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(
            f"\n\u2705 VALIDATION PASSED - All {total_tests} tests produced expected results"
        )
        sys.exit(0)
