"""
Diagnostics module for the Neuroplasticity Engine.

Provides effective rank computation, per-layer diagnostics, gradient diversity
tracking, and LRP (Layer-wise Relevance Propagation) score computation for
informed pruning decisions.

Based on:
- Roy & Vetterli (2007): effective rank via SVD entropy
- Lyle et al. (ICML 2023): gradient covariance degeneracy as plasticity loss signal
- Li et al. (2025, arxiv 2503.21419): LRP-based pruning scores

Docs: https://ml-explore.github.io/mlx/

Sample input:
    from mlx_diagnostics import compute_effective_rank, compute_layer_diagnostics
    rank = compute_effective_rank(mx.random.normal((256, 1024)))
    diagnostics = compute_layer_diagnostics(model)

Expected output:
    rank -> ~256.0 for full-rank random matrix
    diagnostics -> [{"layer": 0, "effective_rank": 245.3, ...}, ...]
"""

import math
from typing import Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
from loguru import logger

# Alias for MLX array materialization (NOT Python's eval builtin)
_mx_materialize = mx.eval  # noqa: S307


def compute_effective_rank(weight_matrix: mx.array) -> float:
    """Compute effective rank of a weight matrix via SVD entropy.

    Formula (Roy & Vetterli, 2007):
        sigma = singular_values(W)
        p = sigma / sum(sigma)
        H = -sum(p * log(p))
        effective_rank = exp(H)

    Returns the number of "effective dimensions" the matrix uses.
    A full-rank random 256x1024 matrix has effective_rank near 256.
    A low-rank matrix (many near-zero singular values) has much lower rank.

    Args:
        weight_matrix: 2D weight matrix (any shape).

    Returns:
        Effective rank as a float. Returns 1.0 for degenerate matrices.
    """
    if weight_matrix.ndim != 2:
        return 1.0

    # SVD -- MLX returns (U, S, Vt); we only need S
    try:
        _, sigma, _ = mx.linalg.svd(weight_matrix, stream=mx.cpu)
        _mx_materialize(sigma)
    except Exception:
        return 1.0

    # Filter near-zero singular values to avoid log(0)
    sigma_np = sigma.tolist()
    sigma_positive = [s for s in sigma_np if s > 1e-10]
    if not sigma_positive:
        return 1.0

    # Normalize to probability distribution
    total = sum(sigma_positive)
    if total < 1e-10:
        return 1.0
    probs = [s / total for s in sigma_positive]

    # Shannon entropy
    entropy = -sum(p * math.log(p) for p in probs if p > 0)

    return math.exp(entropy)


def compute_layer_diagnostics(model) -> List[Dict]:
    """Compute per-layer diagnostics for logging and monitoring.

    Returns a list of dicts with:
        - layer: layer index
        - effective_rank: SVD-based effective dimensionality of FFN input weights
        - weight_norm: Frobenius norm of FFN input weights
        - sparsity: fraction of pruned (masked-out) connections
        - dead_units: neurons with near-zero persistent activation
        - d_ff: current FFN width

    Args:
        model: MLXAdaptiveTransformer instance.

    Returns:
        List of per-layer diagnostic dicts.
    """
    diagnostics: List[Dict] = []

    for i, layer in enumerate(model.layers):
        ffn = layer.ffn

        # Effective rank of input weight matrix
        eff_rank = compute_effective_rank(ffn.input_layer.weight)

        # Weight norm
        w = ffn.input_layer.weight
        _mx_materialize(w)
        w_norm = float(mx.sqrt(mx.sum(w * w)).item())

        # Sparsity
        sparsity = ffn.get_sparsity()

        # Dead units (persistent activation < threshold)
        dead_count = 0
        if ffn._persistent_mean_activation is not None:
            mean_act = ffn._persistent_mean_activation
            _mx_materialize(mean_act)
            dead_count = int(mx.sum(mean_act[:ffn.d_ff] < 0.001).item())

        diagnostics.append({
            "layer": i,
            "effective_rank": round(eff_rank, 2),
            "weight_norm": round(w_norm, 4),
            "sparsity": round(sparsity, 4),
            "dead_units": dead_count,
            "d_ff": ffn.d_ff,
        })

    return diagnostics


class GradientDiversityTracker:
    """Tracks gradient covariance diversity via random projection.

    Lyle et al. identify gradient covariance degeneracy as the most reliable
    predictor of plasticity loss. When gradients across different inputs become
    collinear, the network can no longer discriminate between inputs.

    Uses Johnson-Lindenstrauss random projection to reduce gradient dimensionality,
    then tracks an EMA of the projected gradient outer product. The effective rank
    of this covariance matrix indicates gradient diversity.

    Args:
        projection_dim: Dimensionality to project gradients down to (default 64).
        decay: EMA decay rate for covariance accumulation (default 0.99).
    """

    def __init__(self, projection_dim: int = 64, decay: float = 0.99) -> None:
        self.projection_dim = projection_dim
        self.decay = decay
        self._projection_matrix: Optional[mx.array] = None
        self._covariance: Optional[mx.array] = None
        self._update_count: int = 0

    def _flatten_grads(self, grads) -> mx.array:
        """Flatten gradient tree into a single 1D vector."""
        flat_parts: List[mx.array] = []

        def _collect(g):
            if isinstance(g, mx.array):
                flat_parts.append(g.reshape(-1))
            elif isinstance(g, dict):
                for v in g.values():
                    _collect(v)
            elif isinstance(g, (list, tuple)):
                for item in g:
                    _collect(item)

        _collect(grads)
        if not flat_parts:
            return mx.zeros((1,))
        return mx.concatenate(flat_parts)

    def update(self, grads) -> None:
        """Project gradients and accumulate into covariance EMA.

        Args:
            grads: Gradient tree from loss_and_grad.
        """
        flat = self._flatten_grads(grads)
        n = flat.shape[0]

        # Initialize projection matrix on first call (fixed random)
        if self._projection_matrix is None or self._projection_matrix.shape[1] != n:
            mx.random.seed(42)  # Fixed seed for reproducibility
            self._projection_matrix = mx.random.normal(
                (self.projection_dim, n)
            ) / math.sqrt(self.projection_dim)
            _mx_materialize(self._projection_matrix)
            self._covariance = mx.zeros(
                (self.projection_dim, self.projection_dim)
            )

        # Project: R^n -> R^projection_dim
        g_proj = self._projection_matrix @ flat  # (projection_dim,)
        _mx_materialize(g_proj)

        # Outer product and EMA update
        outer = g_proj.reshape(-1, 1) @ g_proj.reshape(1, -1)
        self._covariance = self.decay * self._covariance + (1 - self.decay) * outer
        _mx_materialize(self._covariance)
        self._update_count += 1

    def get_diversity_score(self) -> float:
        """Compute effective rank of the gradient covariance matrix.

        Higher score = more diverse gradients = healthier plasticity.
        Returns 0.0 if not enough updates have been collected.
        """
        if self._covariance is None or self._update_count < 2:
            return 0.0
        return compute_effective_rank(self._covariance)


def compute_lrp_scores(
    ffn,
    sample_input: mx.array,
) -> Dict[str, mx.array]:
    """Compute LRP (Layer-wise Relevance Propagation) scores for FFN weights.

    LRP decomposes the output relevance backward through the network to assign
    importance scores to each weight. Unlike magnitude pruning, LRP considers
    both weight values AND activation patterns -- a large weight receiving zero
    activations has zero relevance.

    Uses the LRP-epsilon rule for linear layers:
        R_i = sum_j [ (a_i * w_ji) / (z_j + epsilon*sign(z_j)) * R_j ]

    For our FFN (input_proj -> GELU -> output_proj):
    1. Forward pass to get activations at each stage
    2. Start with output activations as initial relevance
    3. Propagate back through output_proj then input_proj

    Args:
        ffn: MLXAdaptiveFeedForward instance.
        sample_input: Input tensor of shape (B, T, d_model).

    Returns:
        Dict with "input_scores" and "output_scores" -- per-weight relevance
        arrays matching the weight shapes. Higher = more important.
    """
    eps = 1e-6

    # Forward pass collecting intermediate activations
    # Stage 1: Input projection
    masked_w_in = mx.multiply(ffn.input_layer.weight, ffn._input_mask)
    z_hidden = sample_input @ masked_w_in.T + ffn.input_layer.bias  # (B, T, d_ff)

    # GELU directly — intermediate_norm removed from live forward path
    h_gelu = nn.gelu(z_hidden)
    _mx_materialize(h_gelu)

    # Stage 2: Output projection
    masked_w_out = mx.multiply(ffn.output_layer.weight, ffn._output_mask)
    z_output = h_gelu @ masked_w_out.T + ffn.output_layer.bias  # (B, T, d_model)
    _mx_materialize(z_output)

    # ---- LRP backward pass ----

    # Initial relevance = absolute output values (what the FFN produces)
    R_output = mx.abs(z_output)  # (B, T, d_model)

    # Step 1: Propagate through output layer (d_model, d_ff)
    z_denom = z_output + eps * mx.sign(z_output)  # (B, T, d_model)
    z_denom = mx.where(mx.abs(z_denom) < eps, mx.full(z_denom.shape, eps), z_denom)

    # Per-hidden-unit relevance via LRP-epsilon rule
    r_over_z = R_output / z_denom  # (B, T, d_model)
    R_hidden = mx.abs(h_gelu) * (r_over_z @ masked_w_out)  # (B, T, d_ff)
    _mx_materialize(R_hidden)

    # Per-weight output scores: mean contribution across batch and time
    h_mean = mx.mean(mx.abs(h_gelu), axis=(0, 1))  # (d_ff,)
    rz_mean = mx.mean(mx.abs(r_over_z), axis=(0, 1))  # (d_model,)
    output_scores = rz_mean.reshape(-1, 1) * h_mean.reshape(1, -1) * mx.abs(masked_w_out)
    _mx_materialize(output_scores)

    # Step 2: Propagate through input layer (d_ff, d_model)
    z_hidden_denom = z_hidden + eps * mx.sign(z_hidden)
    z_hidden_denom = mx.where(
        mx.abs(z_hidden_denom) < eps, mx.full(z_hidden_denom.shape, eps), z_hidden_denom
    )

    rh_over_z = R_hidden / z_hidden_denom  # (B, T, d_ff)
    x_mean = mx.mean(mx.abs(sample_input), axis=(0, 1))  # (d_model,)
    rhz_mean = mx.mean(mx.abs(rh_over_z), axis=(0, 1))  # (d_ff,)
    input_scores = rhz_mean.reshape(-1, 1) * x_mean.reshape(1, -1) * mx.abs(masked_w_in)
    _mx_materialize(input_scores)

    return {
        "input_scores": input_scores,   # shape: (d_ff, d_model)
        "output_scores": output_scores,  # shape: (d_model, d_ff)
    }


if __name__ == "__main__":
    import sys

    from mlx_model import MLXAdaptiveTransformer

    all_validation_failures: List[str] = []
    total_tests = 0

    # Test 1: Effective rank of full-rank random matrix near min(rows, cols)
    total_tests += 1
    full_rank = mx.random.normal((256, 1024))
    _mx_materialize(full_rank)
    rank = compute_effective_rank(full_rank)
    # Full-rank 256x1024 matrix should have effective rank near 256
    if rank < 200 or rank > 260:
        all_validation_failures.append(
            f"Full-rank effective rank: expected ~256, got {rank:.2f}"
        )
    else:
        print(f"  Test 1 OK: full-rank matrix effective_rank = {rank:.2f} (expected ~256)")

    # Test 2: Low-rank matrix has much lower effective rank
    total_tests += 1
    # Create rank-5 matrix: 256x5 @ 5x1024
    low_rank = mx.random.normal((256, 5)) @ mx.random.normal((5, 1024))
    _mx_materialize(low_rank)
    lr_rank = compute_effective_rank(low_rank)
    if lr_rank > 10:
        all_validation_failures.append(
            f"Low-rank effective rank: expected <10, got {lr_rank:.2f}"
        )
    else:
        print(f"  Test 2 OK: rank-5 matrix effective_rank = {lr_rank:.2f} (expected ~5)")

    # Test 3: compute_layer_diagnostics returns correct structure
    total_tests += 1
    model = MLXAdaptiveTransformer(
        vocab_size=500, d_model=32, nhead=4, num_layers=2,
        d_ff=128, max_seq_len=16,
    )
    model.train()
    tokens = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
    _ = model(tokens)
    _mx_materialize(_)
    diag = compute_layer_diagnostics(model)
    if len(diag) != 2:
        all_validation_failures.append(
            f"Layer diagnostics count: expected 2, got {len(diag)}"
        )
    else:
        required_keys = {"layer", "effective_rank", "weight_norm", "sparsity", "dead_units", "d_ff"}
        missing = required_keys - diag[0].keys()
        if missing:
            all_validation_failures.append(f"Layer diagnostics missing keys: {missing}")
        elif diag[0]["d_ff"] != 128:
            all_validation_failures.append(
                f"Layer diagnostics d_ff: expected 128, got {diag[0]['d_ff']}"
            )
        else:
            print(f"  Test 3 OK: layer diagnostics = {diag}")

    # Test 4: GradientDiversityTracker with diverse gradients
    total_tests += 1
    tracker = GradientDiversityTracker(projection_dim=16, decay=0.9)
    # Feed diverse gradients (different random vectors)
    for _ in range(10):
        fake_grads = {"w": mx.random.normal((100,))}
        _mx_materialize(fake_grads["w"])
        tracker.update(fake_grads)
    diverse_score = tracker.get_diversity_score()
    if diverse_score <= 0:
        all_validation_failures.append(
            f"Diverse gradient score: expected > 0, got {diverse_score:.4f}"
        )
    else:
        print(f"  Test 4 OK: diverse gradient score = {diverse_score:.4f}")

    # Test 5: GradientDiversityTracker with identical gradients (low diversity)
    total_tests += 1
    tracker2 = GradientDiversityTracker(projection_dim=16, decay=0.9)
    fixed_grad = mx.random.normal((100,))
    _mx_materialize(fixed_grad)
    for _ in range(10):
        tracker2.update({"w": fixed_grad})
    identical_score = tracker2.get_diversity_score()
    # Identical gradients should produce lower diversity than diverse gradients
    if identical_score >= diverse_score:
        all_validation_failures.append(
            f"Identical gradient score ({identical_score:.4f}) should be < "
            f"diverse score ({diverse_score:.4f})"
        )
    else:
        print(
            f"  Test 5 OK: identical score ({identical_score:.4f}) < "
            f"diverse score ({diverse_score:.4f})"
        )

    # Test 6: LRP scores have correct shapes
    total_tests += 1
    from mlx_adaptive_feedforward import MLXAdaptiveFeedForward
    ffn = MLXAdaptiveFeedForward(d_model=32, d_ff=128)
    ffn.eval()
    sample = mx.random.normal((2, 4, 32))
    _mx_materialize(sample)
    lrp = compute_lrp_scores(ffn, sample)
    if "input_scores" not in lrp or "output_scores" not in lrp:
        all_validation_failures.append(f"LRP scores missing keys: {lrp.keys()}")
    elif lrp["input_scores"].shape != (128, 32):
        all_validation_failures.append(
            f"LRP input_scores shape: expected (128, 32), got {lrp['input_scores'].shape}"
        )
    elif lrp["output_scores"].shape != (32, 128):
        all_validation_failures.append(
            f"LRP output_scores shape: expected (32, 128), got {lrp['output_scores'].shape}"
        )
    else:
        in_min = float(mx.min(lrp["input_scores"]).item())
        in_max = float(mx.max(lrp["input_scores"]).item())
        print(
            f"  Test 6 OK: LRP shapes correct, input_scores range "
            f"[{in_min:.6f}, {in_max:.6f}]"
        )

    # Test 7: LRP scores are non-negative (relevance is always positive)
    total_tests += 1
    if float(mx.min(lrp["input_scores"]).item()) < -1e-6:
        all_validation_failures.append("LRP input_scores has negative values")
    elif float(mx.min(lrp["output_scores"]).item()) < -1e-6:
        all_validation_failures.append("LRP output_scores has negative values")
    else:
        print("  Test 7 OK: LRP scores are all non-negative")

    # Test 8: Degenerate input handling
    total_tests += 1
    zero_matrix = mx.zeros((10, 10))
    zero_rank = compute_effective_rank(zero_matrix)
    if zero_rank != 1.0:
        all_validation_failures.append(
            f"Zero matrix effective rank: expected 1.0, got {zero_rank}"
        )
    else:
        print(f"  Test 8 OK: zero matrix effective_rank = {zero_rank}")

    if all_validation_failures:
        print(
            f"\n❌ VALIDATION FAILED - {len(all_validation_failures)} "
            f"of {total_tests} tests failed:"
        )
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(
            f"\n✅ VALIDATION PASSED - All {total_tests} tests produced "
            f"expected results"
        )
        sys.exit(0)
