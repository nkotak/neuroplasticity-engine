"""
MLX adaptive feedforward module with structural plasticity.

Ports the PyTorch AdaptiveFeedForward to Apple MLX, supporting dynamic
neuron splitting, merging, and connection pruning/growth via weight masks.

Docs: https://ml-explore.github.io/mlx/
      https://ml-explore.github.io/mlx/build/html/python/nn.html

Sample input:
    ffn = MLXAdaptiveFeedForward(d_model=256, d_ff=1024)
    x = mx.random.normal((2, 16, 256))
    out = ffn(x)

Expected output:
    out.shape -> (2, 16, 256)
    ffn.d_ff -> 1024 (changes after split/merge)
"""

from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn
from loguru import logger


class MovingStats:
    """Lightweight running statistics tracker for activation monitoring."""

    def __init__(self, window_size: int = 100) -> None:
        self._values: List[float] = []
        self._window_size = window_size
        self.recent_activations: Optional[mx.array] = None

    @property
    def mean(self) -> float:
        if not self._values:
            return 0.0
        return sum(self._values) / len(self._values)

    @property
    def count(self) -> int:
        return len(self._values)

    def update(self, value: float, activations: Optional[mx.array] = None) -> None:
        self._values.append(value)
        if len(self._values) > self._window_size:
            self._values = self._values[-self._window_size:]
        if activations is not None:
            # Materialize to prevent holding the forward-pass computation graph
            mx.eval(activations)
            self.recent_activations = activations

    def reset(self) -> None:
        self._values.clear()
        self.recent_activations = None


class MLXAdaptiveFeedForward(nn.Module):
    """Adaptive feedforward with dynamic neuron split/merge and weight masking.

    Weight masks (_input_mask, _output_mask) are plain mx.arrays stored outside
    the model parameter tree so they don't interfere with optimizer state or
    model serialization.
    """

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.input_layer = nn.Linear(d_model, d_ff)
        self.output_layer = nn.Linear(d_ff, d_model)
        # Legacy: retained in param tree for checkpoint compatibility only.
        # NOT used in forward pass — LayerNorm across d_ff breaks function-
        # preserving split/add semantics because resizing changes the norm
        # denominator and creates fresh gamma/beta for new neurons.
        self.intermediate_norm = nn.LayerNorm(d_ff)

        # Masks live outside the parameter tree
        self._input_mask: mx.array = mx.ones((d_ff, d_model))
        self._output_mask: mx.array = mx.ones((d_model, d_ff))

        # Activation tracking (only used when model.training is True)
        self.input_activation_stats = MovingStats()
        self.intermediate_activation_stats = MovingStats()
        self.output_activation_stats = MovingStats()

        # Neuron age tracking for split decisions
        self._neuron_age: mx.array = mx.zeros((d_ff,))

        # Persistent activation stats that survive train/eval mode switches.
        # Updated during training, read by plasticity controller during eval.
        self._persistent_mean_activation: Optional[mx.array] = None
        self._activation_ema_decay: float = 0.99

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass with masked weights."""
        if self.training:
            self.input_activation_stats.update(
                float(mx.mean(x).item()), x
            )

        # Input projection with mask
        masked_w = mx.multiply(self.input_layer.weight, self._input_mask)
        h = x @ masked_w.T + self.input_layer.bias
        h = nn.gelu(h)

        if self.training:
            self.intermediate_activation_stats.update(
                float(mx.mean(h).item()), h
            )
            self._neuron_age = self._neuron_age + 1
            mx.eval(self._neuron_age)  # Materialize to break graph chain

            # Update persistent EMA of per-neuron mean |activation|
            # Shape: (d_ff,) — survives train/eval mode switches
            batch_mean = mx.mean(mx.abs(h.reshape(-1, h.shape[-1])), axis=0)
            if self._persistent_mean_activation is None:
                self._persistent_mean_activation = batch_mean
            else:
                decay = self._activation_ema_decay
                self._persistent_mean_activation = (
                    decay * self._persistent_mean_activation
                    + (1.0 - decay) * batch_mean
                )
            mx.eval(self._persistent_mean_activation)  # Materialize EMA

        # Output projection with mask
        masked_w_out = mx.multiply(self.output_layer.weight, self._output_mask)
        out = h @ masked_w_out.T + self.output_layer.bias

        if self.training:
            self.output_activation_stats.update(
                float(mx.mean(out).item()), out
            )

        return out

    # ------------------------------------------------------------------
    # Structural plasticity operations
    # ------------------------------------------------------------------

    def split_neuron(self, neuron_idx: int, noise_scale: float = 0.1) -> None:
        """Split an intermediate neuron, increasing d_ff by 1.

        Uses Net2Net function-preserving split: output weights are halved
        so parent + child = original contribution. With noise_scale=0 in
        eval mode the output is numerically unchanged.
        """
        new_d_ff = self.d_ff + 1

        # --- Input layer: (d_ff, d_model) -> (d_ff+1, d_model) ---
        old_iw = self.input_layer.weight
        old_ib = self.input_layer.bias
        noise_w = mx.random.normal(old_iw[neuron_idx: neuron_idx + 1].shape) * noise_scale
        new_row = old_iw[neuron_idx: neuron_idx + 1] + noise_w
        new_iw = mx.concatenate([old_iw, new_row], axis=0)

        noise_b = mx.random.normal((1,)) * noise_scale
        new_ib = mx.concatenate([old_ib, old_ib[neuron_idx: neuron_idx + 1] + noise_b])

        self.input_layer = nn.Linear(self.d_model, new_d_ff)
        self.input_layer.weight = new_iw
        self.input_layer.bias = new_ib

        # Copy parent's input mask for child (not all-ones)
        child_in_mask = self._input_mask[neuron_idx: neuron_idx + 1]
        self._input_mask = mx.concatenate([self._input_mask, child_in_mask], axis=0)

        # --- Output layer: (d_model, d_ff) -> (d_model, d_ff+1) ---
        # Net2Net function-preserving: halve output weights for both parent and child
        old_ow = self.output_layer.weight
        old_ob = self.output_layer.bias
        halved_col = old_ow[:, neuron_idx: neuron_idx + 1] / 2.0

        # Set parent column to halved value
        new_ow = mx.concatenate([
            old_ow[:, :neuron_idx],
            halved_col,
            old_ow[:, neuron_idx + 1:],
        ], axis=1)
        # Append child column (also halved)
        new_col = halved_col + mx.random.normal(halved_col.shape) * noise_scale * 0.01
        new_ow = mx.concatenate([new_ow, new_col], axis=1)

        self.output_layer = nn.Linear(new_d_ff, self.d_model)
        self.output_layer.weight = new_ow
        self.output_layer.bias = old_ob  # preserve output bias

        # Copy parent's output mask for child (not all-ones)
        child_out_mask = self._output_mask[:, neuron_idx: neuron_idx + 1]
        self._output_mask = mx.concatenate([self._output_mask, child_out_mask], axis=1)

        # Resize legacy norm for checkpoint compat (not used in forward)
        self.intermediate_norm = nn.LayerNorm(new_d_ff)
        self._neuron_age = mx.concatenate([self._neuron_age, mx.zeros((1,))])
        self.d_ff = new_d_ff
        self._persistent_mean_activation = None
        mx.eval(mx.array(0))  # noqa: S307
        logger.debug(f"Split neuron {neuron_idx}, new d_ff={self.d_ff}")

    def merge_neurons(self, neuron_indices: List[int]) -> None:
        """Merge multiple intermediate neurons into one, decreasing d_ff."""
        if len(neuron_indices) < 2:
            return

        num_to_merge = len(neuron_indices)
        new_d_ff = self.d_ff - num_to_merge + 1
        keep = [i for i in range(self.d_ff) if i not in neuron_indices]

        # --- Input layer ---
        old_iw = self.input_layer.weight
        old_ib = self.input_layer.bias
        kept_w = old_iw[keep]
        merged_w = mx.mean(old_iw[neuron_indices], axis=0, keepdims=True)
        new_iw = mx.concatenate([kept_w, merged_w], axis=0)

        kept_b = old_ib[keep]
        merged_b = mx.mean(old_ib[neuron_indices], keepdims=True)
        new_ib = mx.concatenate([kept_b, merged_b])

        self.input_layer = nn.Linear(self.d_model, new_d_ff)
        self.input_layer.weight = new_iw
        self.input_layer.bias = new_ib

        # Input mask
        kept_mask = self._input_mask[keep]
        merged_mask = mx.max(self._input_mask[neuron_indices], axis=0, keepdims=True)
        self._input_mask = mx.concatenate([kept_mask, merged_mask], axis=0)

        # --- Output layer ---
        old_ow = self.output_layer.weight
        old_ob = self.output_layer.bias
        kept_ow = old_ow[:, keep]
        merged_ow = mx.mean(old_ow[:, neuron_indices], axis=1, keepdims=True)
        new_ow = mx.concatenate([kept_ow, merged_ow], axis=1)

        self.output_layer = nn.Linear(new_d_ff, self.d_model)
        self.output_layer.weight = new_ow
        self.output_layer.bias = old_ob  # preserve output bias

        # Output mask
        kept_om = self._output_mask[:, keep]
        merged_om = mx.max(self._output_mask[:, neuron_indices], axis=1, keepdims=True)
        self._output_mask = mx.concatenate([kept_om, merged_om], axis=1)

        # Resize legacy norm for checkpoint compat (not used in forward)
        self.intermediate_norm = nn.LayerNorm(new_d_ff)
        kept_ages = [float(self._neuron_age[i].item()) for i in range(self.d_ff)
                     if i not in neuron_indices]
        self._neuron_age = mx.array(kept_ages + [0.0])
        self.d_ff = new_d_ff
        # Reset persistent activation buffer — shape changed
        self._persistent_mean_activation = None
        mx.eval(mx.array(0))
        logger.debug(f"Merged neurons {neuron_indices}, new d_ff={self.d_ff}")

    def add_neurons(
        self,
        count: int,
        noise_scale: float = 0.02,
        active_fraction: float = 0.6,
        min_seed_inputs: int = 16,
        min_seed_outputs: int = 4,
    ) -> int:
        """Add fresh neurons, increasing d_ff by count.

        Unlike split_neuron (which duplicates an existing neuron), these are
        entirely new neurons with Kaiming-initialized input weights and
        near-zero output weights. Masks are sparse (active_fraction) so new
        neurons don't create a dense connection spike.

        Returns the number of new active mask entries introduced.
        """
        new_d_ff = self.d_ff + count
        fan_in = self.d_model
        std = (2.0 / fan_in) ** 0.5

        # Build sparse input masks for new neurons
        n_in_active = max(min_seed_inputs, int(self.d_model * active_fraction))
        n_in_active = min(n_in_active, self.d_model)
        in_mask_new = mx.zeros((count, self.d_model))
        for i in range(count):
            perm = mx.random.uniform(shape=(self.d_model,))
            threshold = mx.sort(perm)[self.d_model - n_in_active]
            in_mask_new = mx.concatenate([
                in_mask_new[:i],
                (perm >= threshold).astype(mx.float32).reshape(1, -1),
                in_mask_new[i + 1:],
            ], axis=0)

        # Build sparse output masks for new neurons
        n_out_active = max(min_seed_outputs, int(self.d_model * active_fraction))
        n_out_active = min(n_out_active, self.d_model)
        out_mask_new = mx.zeros((self.d_model, count))
        for i in range(count):
            perm = mx.random.uniform(shape=(self.d_model,))
            threshold = mx.sort(perm)[self.d_model - n_out_active]
            col = (perm >= threshold).astype(mx.float32).reshape(-1, 1)
            out_mask_new = mx.concatenate([
                out_mask_new[:, :i], col, out_mask_new[:, i + 1:],
            ], axis=1)

        mx.eval(in_mask_new, out_mask_new)  # noqa: S307  — mlx.core.eval

        # Input: Kaiming init at active positions, zero elsewhere
        new_rows = mx.random.normal((count, self.d_model)) * std
        new_rows = new_rows * in_mask_new  # zero inactive
        new_bias = mx.zeros((count,))

        old_iw, old_ib = self.input_layer.weight, self.input_layer.bias
        self.input_layer = nn.Linear(self.d_model, new_d_ff)
        self.input_layer.weight = mx.concatenate([old_iw, new_rows], axis=0)
        self.input_layer.bias = mx.concatenate([old_ib, new_bias])

        # Output: near-zero at active positions, zero elsewhere
        new_cols = mx.random.normal((self.d_model, count)) * noise_scale * 0.01
        new_cols = new_cols * out_mask_new  # zero inactive
        old_ow, old_ob = self.output_layer.weight, self.output_layer.bias
        self.output_layer = nn.Linear(new_d_ff, self.d_model)
        self.output_layer.weight = mx.concatenate([old_ow, new_cols], axis=1)
        self.output_layer.bias = old_ob  # bias is (d_model,), unchanged

        # Extend masks (sparse, not dense)
        self._input_mask = mx.concatenate(
            [self._input_mask, in_mask_new], axis=0
        )
        self._output_mask = mx.concatenate(
            [self._output_mask, out_mask_new], axis=1
        )

        # Count new active connections introduced
        new_active = int(mx.sum(in_mask_new != 0).item()) + int(
            mx.sum(out_mask_new != 0).item()
        )

        # Resize legacy norm for checkpoint compat (not used in forward)
        self.intermediate_norm = nn.LayerNorm(new_d_ff)
        self._neuron_age = mx.concatenate([self._neuron_age, mx.zeros((count,))])
        self.d_ff = new_d_ff
        self._persistent_mean_activation = None
        mx.eval(mx.array(0))  # noqa: S307  — mlx.core.eval
        logger.debug(
            f"Added {count} neurons (sparse, {new_active} active conns), "
            f"new d_ff={self.d_ff}"
        )
        return new_active

    def remove_neuron(self, neuron_idx: int, min_d_ff: int = 1) -> bool:
        """Remove a neuron entirely, decreasing d_ff by 1.

        Args:
            neuron_idx: Index of neuron to remove.
            min_d_ff: Safety floor — the controller owns the shrink policy.

        Returns True if removal succeeded, False if at safety floor.
        """
        if self.d_ff <= min_d_ff:
            return False
        new_d_ff = self.d_ff - 1
        keep = [i for i in range(self.d_ff) if i != neuron_idx]

        old_iw, old_ib = self.input_layer.weight, self.input_layer.bias
        self.input_layer = nn.Linear(self.d_model, new_d_ff)
        self.input_layer.weight = old_iw[keep]
        self.input_layer.bias = old_ib[keep]
        self._input_mask = self._input_mask[keep]

        old_ow = self.output_layer.weight
        old_ob = self.output_layer.bias
        self.output_layer = nn.Linear(new_d_ff, self.d_model)
        self.output_layer.weight = old_ow[:, keep]
        self.output_layer.bias = old_ob  # (d_model,), unchanged
        self._output_mask = self._output_mask[:, keep]

        # Resize legacy norm for checkpoint compat (not used in forward)
        self.intermediate_norm = nn.LayerNorm(new_d_ff)
        self._neuron_age = mx.array(
            [float(self._neuron_age[i].item()) for i in keep]
        )
        self.d_ff = new_d_ff
        self._persistent_mean_activation = None
        mx.eval(mx.array(0))  # noqa: S307  — mlx.core.eval
        logger.debug(f"Removed neuron {neuron_idx}, new d_ff={self.d_ff}")
        return True

    def get_sparsity(self) -> float:
        """Return fraction of zeroed-out mask entries across both layers."""
        total = self._input_mask.size + self._output_mask.size
        active = float(mx.sum(self._input_mask != 0).item()) + float(
            mx.sum(self._output_mask != 0).item()
        )
        return 1.0 - (active / total) if total > 0 else 0.0

    def get_active_connections(self) -> int:
        """Count of active (non-zero) mask entries."""
        return int(mx.sum(self._input_mask != 0).item()) + int(
            mx.sum(self._output_mask != 0).item()
        )


if __name__ == "__main__":
    import sys

    all_validation_failures: List[str] = []
    total_tests = 0

    # Test 1: Forward pass shape
    total_tests += 1
    ffn = MLXAdaptiveFeedForward(d_model=64, d_ff=256)
    x = mx.random.normal((2, 8, 64))
    out = ffn(x)
    mx.eval(out)
    if out.shape != (2, 8, 64):
        all_validation_failures.append(
            f"Forward shape: expected (2, 8, 64), got {out.shape}"
        )
    else:
        print(f"  Test 1 OK: forward shape = {out.shape}")

    # Test 2: Split increases d_ff by 1
    total_tests += 1
    old_dff = ffn.d_ff
    ffn.split_neuron(0)
    if ffn.d_ff != old_dff + 1:
        all_validation_failures.append(
            f"Split d_ff: expected {old_dff + 1}, got {ffn.d_ff}"
        )
    else:
        print(f"  Test 2 OK: split d_ff {old_dff} -> {ffn.d_ff}")

    # Test 3: Function-preserving split (Net2Net property)
    total_tests += 1
    ffn_fp = MLXAdaptiveFeedForward(d_model=64, d_ff=256)
    ffn_fp.eval()  # Ensure deterministic (no activation tracking)
    x_fp = mx.random.normal((2, 8, 64))
    mx.eval(x_fp)
    out_before = ffn_fp(x_fp)
    mx.eval(out_before)

    # Split with very small noise so the function-preserving property holds tightly
    ffn_fp.split_neuron(10, noise_scale=0.0)
    out_after = ffn_fp(x_fp)
    mx.eval(out_after)

    max_diff = float(mx.max(mx.abs(out_before - out_after)).item())
    # With noise_scale=0.0 and no intermediate_norm in forward, split is exact.
    fp_tolerance = 1e-5
    if max_diff > fp_tolerance:
        all_validation_failures.append(
            f"Function-preserving split: max output diff = {max_diff:.6f}, "
            f"exceeds tolerance {fp_tolerance}"
        )
    else:
        print(f"  Test 3 OK: function-preserving split max_diff = {max_diff:.6f} "
              f"(tolerance {fp_tolerance})")

    # Test 4: Forward still works after split
    total_tests += 1
    out2 = ffn(x)
    mx.eval(out2)
    if out2.shape != (2, 8, 64):
        all_validation_failures.append(
            f"Post-split forward shape: expected (2, 8, 64), got {out2.shape}"
        )
    else:
        print(f"  Test 4 OK: post-split forward shape = {out2.shape}")

    # Test 5: Merge decreases d_ff
    total_tests += 1
    old_dff = ffn.d_ff
    ffn.merge_neurons([0, 1])
    expected_dff = old_dff - 1
    if ffn.d_ff != expected_dff:
        all_validation_failures.append(
            f"Merge d_ff: expected {expected_dff}, got {ffn.d_ff}"
        )
    else:
        print(f"  Test 5 OK: merge d_ff {old_dff} -> {ffn.d_ff}")

    # Test 6: Forward works after merge
    total_tests += 1
    out3 = ffn(x)
    mx.eval(out3)
    if out3.shape != (2, 8, 64):
        all_validation_failures.append(
            f"Post-merge forward shape: expected (2, 8, 64), got {out3.shape}"
        )
    else:
        print(f"  Test 6 OK: post-merge forward shape = {out3.shape}")

    # Test 7: Mask dimensions match layer dimensions
    total_tests += 1
    mask_ok = True
    if ffn._input_mask.shape != ffn.input_layer.weight.shape:
        all_validation_failures.append(
            f"Input mask shape {ffn._input_mask.shape} != weight {ffn.input_layer.weight.shape}"
        )
        mask_ok = False
    if ffn._output_mask.shape != ffn.output_layer.weight.shape:
        all_validation_failures.append(
            f"Output mask shape {ffn._output_mask.shape} != weight {ffn.output_layer.weight.shape}"
        )
        mask_ok = False
    if mask_ok:
        print("  Test 7 OK: masks match weight shapes")

    # Test 8: Sparsity starts at 0 (all connections active)
    total_tests += 1
    fresh = MLXAdaptiveFeedForward(d_model=32, d_ff=128)
    sparsity = fresh.get_sparsity()
    if abs(sparsity) > 1e-6:
        all_validation_failures.append(
            f"Initial sparsity: expected 0.0, got {sparsity}"
        )
    else:
        print(f"  Test 8 OK: initial sparsity = {sparsity}")

    # Test 9: add_neurons increases d_ff and preserves forward pass
    total_tests += 1
    add_ffn = MLXAdaptiveFeedForward(d_model=64, d_ff=256)
    add_ffn.eval()
    x_add = mx.random.normal((2, 8, 64))
    mx.eval(x_add)
    out_before_add = add_ffn(x_add)
    mx.eval(out_before_add)
    new_active = add_ffn.add_neurons(8, active_fraction=0.6)
    if add_ffn.d_ff != 264:
        all_validation_failures.append(
            f"add_neurons d_ff: expected 264, got {add_ffn.d_ff}"
        )
    else:
        out_after_add = add_ffn(x_add)
        mx.eval(out_after_add)
        add_diff = float(mx.max(mx.abs(out_before_add - out_after_add)).item())
        if add_diff > 5e-3:
            all_validation_failures.append(
                f"add_neurons drift too large: max diff = {add_diff:.6f}"
            )
        else:
            print(
                f"  Test 9 OK: add_neurons d_ff=264, "
                f"max diff = {add_diff:.6f}, new_active={new_active}"
            )

    # Test 10: remove_neuron decreases d_ff and shapes stay consistent
    total_tests += 1
    rm_ffn = MLXAdaptiveFeedForward(d_model=64, d_ff=256)
    result = rm_ffn.remove_neuron(0)
    if not result or rm_ffn.d_ff != 255:
        all_validation_failures.append(
            f"remove_neuron: expected d_ff=255, got {rm_ffn.d_ff}, result={result}"
        )
    else:
        rm_out = rm_ffn(mx.random.normal((2, 8, 64)))
        mx.eval(rm_out)
        if rm_out.shape != (2, 8, 64):
            all_validation_failures.append(
                f"remove_neuron forward shape: expected (2, 8, 64), got {rm_out.shape}"
            )
        else:
            print(f"  Test 10 OK: remove_neuron d_ff=255, forward works")

    # Test 11: remove_neuron respects safety floor (caller sets min_d_ff)
    total_tests += 1
    tiny_ffn = MLXAdaptiveFeedForward(d_model=64, d_ff=16)
    floor_result = tiny_ffn.remove_neuron(0, min_d_ff=16)
    if floor_result:
        all_validation_failures.append("remove_neuron: should have hit safety floor")
    else:
        print("  Test 11 OK: remove_neuron respects caller-provided safety floor")

    # Test 12: Training mode tracks activations
    total_tests += 1
    ffn2 = MLXAdaptiveFeedForward(d_model=32, d_ff=64)
    ffn2.train()
    _ = ffn2(mx.random.normal((1, 4, 32)))
    mx.eval(_)
    if ffn2.intermediate_activation_stats.count == 0:
        all_validation_failures.append("Training mode: no activations tracked")
    else:
        print(f"  Test 12 OK: activations tracked (count={ffn2.intermediate_activation_stats.count})")

    if all_validation_failures:
        print(f"\n❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"\n✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        sys.exit(0)
