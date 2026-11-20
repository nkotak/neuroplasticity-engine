"""
MLX training loop for the Neuroplasticity Engine.

Production training with gradient accumulation, cosine LR scheduling,
structural plasticity, matplotlib visualization, and metrics.jsonl logging.
Includes RigL dense gradient computation, train/val split, and optimizer
state preservation across structural changes.

Docs: https://ml-explore.github.io/mlx/
      https://matplotlib.org/stable/

Sample input:
    python mlx_train.py              # Full training run
    python mlx_train.py --quick      # 50-step smoke test

Expected output:
    runs/{run_id}/metrics.jsonl      # Per-step architecture snapshots
    runs/{run_id}/plots/*.png        # Loss curves, plasticity timelines, etc.
    runs/{run_id}/checkpoints/       # Model weights + arch_info.json
"""

import gc
import json
import math
import os
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils
import typer
from loguru import logger

from config import MLXModelConfig, MLXTrainingConfig, MLXDataConfig, PlasticityConfig
from mlx_adaptive_feedforward import MLXAdaptiveFeedForward
from mlx_data import (
    BookTextStream, JSONLStream, MixedDataStream, TokenizedBatchIterator,
    build_train_val_pipelines,
)
from mlx_diagnostics import GradientDiversityTracker, compute_layer_diagnostics
from mlx_model import MLXAdaptiveTransformer
from mlx_plasticity import MLXPlasticityController
from mlx_tokenizer import MLXTokenizer


# Alias for MLX array materialization (NOT Python's eval builtin)
_mx_materialize = mx.eval  # noqa: S307


# ---------------------------------------------------------------------------
# Signal handling for graceful shutdown
# ---------------------------------------------------------------------------

_shutdown_requested = False


def _handle_signal(signum: int, frame) -> None:
    """Set shutdown flag so training loop can save a checkpoint before exiting."""
    global _shutdown_requested
    _shutdown_requested = True
    sig_name = signal.Signals(signum).name
    logger.warning(f"Received {sig_name}, will save checkpoint and exit after current step")


# ---------------------------------------------------------------------------
# Part A: MetricsLogger
# ---------------------------------------------------------------------------

@dataclass
class MetricsLogger:
    """Collects time series data for training visualization and logging."""

    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    eval_steps: List[int] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    tokens_per_sec: List[float] = field(default_factory=list)
    param_counts: List[int] = field(default_factory=list)
    d_ff_history: List[List[int]] = field(default_factory=list)
    sparsity_history: List[List[float]] = field(default_factory=list)
    plasticity_events: List[Dict[str, int]] = field(default_factory=list)
    steps: List[int] = field(default_factory=list)

    def log_step(
        self,
        step: int,
        train_loss: float,
        lr: float,
        tps: float,
        param_count: int,
        d_ff_per_layer: List[int],
        sparsity_per_layer: List[float],
    ) -> None:
        self.steps.append(step)
        self.train_losses.append(train_loss)
        self.learning_rates.append(lr)
        self.tokens_per_sec.append(tps)
        self.param_counts.append(param_count)
        self.d_ff_history.append(d_ff_per_layer)
        self.sparsity_history.append(sparsity_per_layer)

    def log_eval(self, val_loss: float, step: int) -> None:
        self.val_losses.append(val_loss)
        self.eval_steps.append(step)

    def log_plasticity(self, events: Dict[str, int]) -> None:
        self.plasticity_events.append(events)


# ---------------------------------------------------------------------------
# Part B: Visualization
# ---------------------------------------------------------------------------

def generate_plots(metrics: MetricsLogger, plot_dir: str) -> None:
    """Generate all training visualization plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(plot_dir, exist_ok=True)

    # 1. Loss curves
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(metrics.steps, metrics.train_losses, label="Train", alpha=0.7)
    if metrics.val_losses:
        ax.plot(
            metrics.eval_steps[: len(metrics.val_losses)],
            metrics.val_losses,
            label="Val",
            alpha=0.9,
            linewidth=2,
        )
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(plot_dir, "loss_curves.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. LR schedule
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(metrics.steps, metrics.learning_rates)
    ax.set_xlabel("Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.grid(True, alpha=0.3)
    fig.savefig(
        os.path.join(plot_dir, "lr_schedule.png"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    # 3. Plasticity timeline — 2 subplots to avoid scale mismatch
    if metrics.plasticity_events:
        fig, (ax_conn, ax_neuron) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        x_pos = list(range(len(metrics.plasticity_events)))

        # 3a. Connection-level events (large scale)
        for etype in ["pruned", "grown"]:
            vals = [e.get(etype, 0) for e in metrics.plasticity_events]
            ax_conn.bar(x_pos, vals, label=etype, alpha=0.7,
                        bottom=[0] * len(x_pos) if etype == "pruned" else
                        [e.get("pruned", 0) for e in metrics.plasticity_events])
        ax_conn.set_ylabel("Connection Events")
        ax_conn.set_title("Connection-Level: Pruned & Grown")
        ax_conn.legend()

        # 3b. Neuron-level events (small scale)
        neuron_types = ["splits", "merges", "neurogenesis", "removed", "reinit"]
        bottom_n = [0] * len(metrics.plasticity_events)
        for etype in neuron_types:
            vals = [e.get(etype, 0) for e in metrics.plasticity_events]
            ax_neuron.bar(x_pos, vals, bottom=bottom_n, label=etype, alpha=0.8)
            bottom_n = [b + v for b, v in zip(bottom_n, vals)]
        ax_neuron.set_xlabel("Eval Checkpoint")
        ax_neuron.set_ylabel("Neuron Events")
        ax_neuron.set_title("Neuron-Level: Splits, Merges, Neurogenesis, Removed, Reinit")
        ax_neuron.legend()

        fig.tight_layout()
        fig.savefig(
            os.path.join(plot_dir, "plasticity_timeline.png"),
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig)

    # 4. d_ff evolution
    if metrics.d_ff_history:
        fig, ax = plt.subplots(figsize=(10, 6))
        num_layers = len(metrics.d_ff_history[0])
        for layer_i in range(num_layers):
            vals = [
                h[layer_i] if layer_i < len(h) else h[-1]
                for h in metrics.d_ff_history
            ]
            ax.plot(metrics.steps[: len(vals)], vals, label=f"Layer {layer_i}")
        ax.set_xlabel("Step")
        ax.set_ylabel("d_ff")
        ax.set_title("FFN Width Evolution Per Layer")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(
            os.path.join(plot_dir, "dff_evolution.png"), dpi=150, bbox_inches="tight"
        )
        plt.close(fig)

    # 5. Sparsity heatmap
    if metrics.sparsity_history and len(metrics.sparsity_history) > 1:
        fig, ax = plt.subplots(figsize=(12, 4))
        import numpy as np_viz
        arr = np_viz.array(metrics.sparsity_history).T
        im = ax.imshow(arr, aspect="auto", cmap="YlOrRd", interpolation="nearest")
        ax.set_xlabel("Step Index")
        ax.set_ylabel("Layer")
        ax.set_title("Sparsity Heatmap (per layer over time)")
        fig.colorbar(im, ax=ax)
        fig.savefig(
            os.path.join(plot_dir, "sparsity_heatmap.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

    # 6. Parameter count
    if metrics.param_counts:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(metrics.steps[: len(metrics.param_counts)], metrics.param_counts)
        ax.set_xlabel("Step")
        ax.set_ylabel("Total Parameters")
        ax.set_title("Model Size Over Time")
        ax.grid(True, alpha=0.3)
        fig.savefig(
            os.path.join(plot_dir, "parameter_count.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

    logger.info(f"Plots saved to {plot_dir}")


# ---------------------------------------------------------------------------
# Part C: Helper functions
# ---------------------------------------------------------------------------

def loss_fn(
    model: MLXAdaptiveTransformer,
    tokens: mx.array,
    label_smoothing: float = 0.1,
) -> mx.array:
    """Autoregressive cross-entropy loss with label smoothing."""
    logits = model(tokens)
    # Shift: predict next token
    logits = logits[:, :-1, :]
    targets = tokens[:, 1:]

    return nn.losses.cross_entropy(
        logits, targets, label_smoothing=label_smoothing
    ).mean()


def evaluate_model(
    model: MLXAdaptiveTransformer,
    data_iter,
    num_batches: int = 10,
) -> float:
    """Run evaluation on held-out data, return average loss."""
    model.eval()
    total_loss = 0.0
    count = 0
    for batch in data_iter:
        if count >= num_batches:
            break
        loss = loss_fn(model, batch, label_smoothing=0.0)
        _mx_materialize(loss)
        total_loss += loss.item()
        count += 1
    return total_loss / max(count, 1)


def _rebuild_data_iter(cfg_snapshot: Dict, split: str, tokenizer, seq_len: int, batch_size: int):
    """Rebuild a data iterator from snapshot config."""
    from mlx_data import build_data_pipeline
    return build_data_pipeline(
        tokenizer=tokenizer,
        book_dir=cfg_snapshot["book_dir"],
        hf_datasets=cfg_snapshot["hf_datasets"],
        jsonl_path=cfg_snapshot["jsonl_path"],
        selflearn_dir=cfg_snapshot.get("selflearn_dir"),
        selflearn_max_shards=cfg_snapshot.get("selflearn_max_shards", 20),
        mixing_weights=cfg_snapshot["mixing_weights"],
        seq_len=seq_len,
        batch_size=batch_size,
        split=split,
        val_fraction=cfg_snapshot["val_fraction"],
    )


def _restore_optimizer_state(
    old_state: Dict,
    new_state: Dict,
    skip_prefixes: Optional[List[tuple]] = None,
    _path: tuple = (),
) -> Dict:
    """Recursively copy optimizer state arrays with matching shapes.

    For params that kept the same shape (non-FFN layers, unchanged FFN layers),
    preserves momentum and velocity. For resized params (FFN layers that changed
    d_ff via split/merge), uses the fresh zero-initialized state.

    skip_prefixes: list of path tuples that should always use fresh state
    (e.g., [("layers", 7, "ffn"), ("final_norm",)] to reset last-layer FFN).
    """
    if skip_prefixes is None:
        skip_prefixes = []

    # Check if current path matches any skip prefix
    for prefix in skip_prefixes:
        if _path[:len(prefix)] == prefix:
            return new_state  # Always use fresh zero-initialized state

    if isinstance(old_state, mx.array) and isinstance(new_state, mx.array):
        if old_state.shape == new_state.shape:
            return old_state
        return new_state  # Shape changed — use fresh zeros
    if isinstance(old_state, dict) and isinstance(new_state, dict):
        result = {}
        for k in new_state:
            if k in old_state:
                result[k] = _restore_optimizer_state(
                    old_state[k], new_state[k], skip_prefixes, _path + (k,),
                )
            else:
                result[k] = new_state[k]
        return result
    if isinstance(old_state, list) and isinstance(new_state, list):
        result = []
        for i, new_item in enumerate(new_state):
            if i < len(old_state):
                result.append(_restore_optimizer_state(
                    old_state[i], new_item, skip_prefixes, _path + (i,),
                ))
            else:
                result.append(new_item)
        return result
    return new_state


def _compute_dense_gradients(
    model: MLXAdaptiveTransformer,
    loss_and_grad_fn,
    train_iter,
    label_smoothing: float,
    cfg_snapshot: Dict,
    tokenizer,
    seq_len: int,
    batch_size: int,
) -> tuple:
    """Compute gradients with masks disabled for RigL-style growth.

    Temporarily sets all FFN masks to ones so the backward pass produces
    gradients at inactive (masked-out) positions. These indicate where
    the model would most benefit from new connections.

    Returns (dense_grads, possibly_new_train_iter).
    """
    model.train()
    try:
        dense_batch = next(train_iter)
    except StopIteration:
        train_iter = _rebuild_data_iter(cfg_snapshot, "train", tokenizer, seq_len, batch_size)
        dense_batch = next(train_iter)

    # Save current masks and replace with all-ones
    saved_masks = []
    for layer in model.layers:
        ffn = layer.ffn
        saved_masks.append((ffn._input_mask, ffn._output_mask))
        ffn._input_mask = mx.ones_like(ffn._input_mask)
        ffn._output_mask = mx.ones_like(ffn._output_mask)

    _, dense_grads = loss_and_grad_fn(model, dense_batch, label_smoothing)
    _mx_materialize(dense_grads)

    # Restore original masks
    for layer, (in_mask, out_mask) in zip(model.layers, saved_masks):
        layer.ffn._input_mask = in_mask
        layer.ffn._output_mask = out_mask

    del dense_batch
    return dense_grads, train_iter


# ---------------------------------------------------------------------------
# Part D: Training Loop
# ---------------------------------------------------------------------------

def train(
    model_cfg: MLXModelConfig,
    train_cfg: MLXTrainingConfig,
    data_cfg: MLXDataConfig,
    plasticity_cfg: PlasticityConfig,
    run_dir: str,
    tokenizer_path: Optional[str] = None,
    quick: bool = False,
    resume: Optional[str] = None,
    init_checkpoint: Optional[str] = None,
) -> str:
    """Main training function. Returns run directory."""
    global _shutdown_requested
    _shutdown_requested = False
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)

    if resume and init_checkpoint:
        raise ValueError("Use either resume or init_checkpoint, not both")

    # Override for quick mode
    if quick:
        model_cfg = MLXModelConfig(
            d_model=64, nhead=4, num_layers=2, d_ff=256,
            vocab_size=1000, max_seq_len=64, dropout=0.1,
        )
        train_cfg = MLXTrainingConfig(
            lr=1e-3, warmup_steps=10, total_steps=50,
            batch_size=2, grad_accum_steps=3,  # Misaligns with eval_every=10 to exercise deferral
            eval_every=10, log_every=5, checkpoint_every=25,
            grad_clip_norm=1.0, weight_decay=0.01, label_smoothing=0.1,
        )
        data_cfg = MLXDataConfig(
            book_dir="__none__", hf_datasets=[], seq_len=64,
            reasoning_jsonl_path="__none__",
            selflearn_dir="__none__",
            mixing_weights=[1.0],
        )
        plasticity_cfg = PlasticityConfig(
            prune_frequency=10, structural_frequency=20,
        )

    # Save config
    config_dict = {
        "model": vars(model_cfg),
        "training": vars(train_cfg),
        "data": vars(data_cfg),
        "quick_mode": quick,
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    # Tokenizer
    if tokenizer_path and os.path.exists(tokenizer_path):
        tokenizer = MLXTokenizer.load(tokenizer_path)
        model_cfg.vocab_size = tokenizer.vocab_size
        logger.info(f"Loaded tokenizer from {tokenizer_path}")
    else:
        logger.info("Training new tokenizer on reasoning data...")
        tokenizer = MLXTokenizer(vocab_size=model_cfg.vocab_size)
        sample_texts = []
        jsonl_path = os.path.expanduser(data_cfg.reasoning_jsonl_path)
        if os.path.exists(jsonl_path):
            with open(jsonl_path) as f:
                for i, line in enumerate(f):
                    if i >= 5000:
                        break
                    ex = json.loads(line)
                    sample_texts.append(
                        f"{ex.get('instruction', '')} {ex.get('reasoning', '')} "
                        f"{ex.get('response', '')}"
                    )
        else:
            sample_texts = [
                "The quick brown fox jumps over the lazy dog.",
                "Machine learning on Apple Silicon.",
            ] * 500
        tok_path = os.path.join(run_dir, "tokenizer.json")
        tokenizer.train(sample_texts, output_path=tok_path)
        model_cfg.vocab_size = tokenizer.vocab_size
        logger.info(f"Trained tokenizer: vocab_size={tokenizer.vocab_size}")

    # Model
    start_step = 0
    if resume and os.path.isdir(resume):
        model = MLXAdaptiveTransformer.load_checkpoint(resume)
        _mx_materialize(model.parameters())
        # Parse step from checkpoint dir name (e.g. "step_1200" or "interrupted")
        ckpt_name = os.path.basename(resume.rstrip("/"))
        if ckpt_name.startswith("step_"):
            start_step = int(ckpt_name.split("_")[1])
        else:
            # Check for step in metrics.jsonl from the same run directory
            parent_run = os.path.dirname(resume)
            mpath = os.path.join(parent_run, "metrics.jsonl")
            if os.path.exists(mpath):
                with open(mpath) as f:
                    lines = f.readlines()
                if lines:
                    start_step = json.loads(lines[-1]).get("step", 0)
        logger.info(f"Resumed from checkpoint {resume} at step {start_step}")
    elif init_checkpoint and os.path.isdir(init_checkpoint):
        model = MLXAdaptiveTransformer.load_checkpoint(init_checkpoint)
        _mx_materialize(model.parameters())
        logger.info(
            f"Initialized model weights from checkpoint {init_checkpoint} "
            "with fresh optimizer/training state"
        )
    else:
        model = MLXAdaptiveTransformer(
            vocab_size=model_cfg.vocab_size,
            d_model=model_cfg.d_model,
            nhead=model_cfg.nhead,
            num_layers=model_cfg.num_layers,
            d_ff=model_cfg.d_ff,
            max_seq_len=model_cfg.max_seq_len,
            dropout=model_cfg.dropout,
        )
        _mx_materialize(model.parameters())
    total_params = model.count_parameters()
    logger.info(f"Model initialized: {total_params:,} parameters")

    # Optimizer with warmup + cosine schedule
    # MLX schedules count optimizer.update() calls, not training steps.
    # With grad_accum_steps=N, update() is called every N steps, so divide.
    effective_warmup = train_cfg.warmup_steps // train_cfg.grad_accum_steps
    effective_total = train_cfg.total_steps // train_cfg.grad_accum_steps
    warmup_sched = optim.linear_schedule(0.0, train_cfg.lr, effective_warmup)
    cosine_sched = optim.cosine_decay(
        train_cfg.lr, effective_total - effective_warmup
    )
    lr_schedule = optim.join_schedules(
        [warmup_sched, cosine_sched], [effective_warmup]
    )
    optimizer = optim.AdamW(
        learning_rate=lr_schedule,
        weight_decay=train_cfg.weight_decay,
        betas=train_cfg.adam_betas,
        eps=train_cfg.adam_eps,
    )
    logger.info(
        f"Optimizer: AdamW betas={train_cfg.adam_betas}, "
        f"eps={train_cfg.adam_eps}, wd={train_cfg.weight_decay}"
    )

    # Restore optimizer step counter when resuming
    if start_step > 0:
        restored_opt_step = start_step // train_cfg.grad_accum_steps
        optimizer.state["step"] = mx.array(restored_opt_step)
        _mx_materialize(optimizer.state)
        logger.info(f"Optimizer step restored to {restored_opt_step}")

    # Plasticity controller
    plasticity = MLXPlasticityController(
        prune_threshold=plasticity_cfg.prune_threshold,
        growth_rate=plasticity_cfg.growth_rate,
        split_threshold=plasticity_cfg.split_threshold,
        merge_similarity_threshold=plasticity_cfg.merge_similarity_threshold,
        prune_frequency=plasticity_cfg.prune_frequency,
        structural_frequency=plasticity_cfg.structural_frequency,
        min_connections=plasticity_cfg.min_connections,
        target_sparsity=plasticity_cfg.target_sparsity,
        dormant_threshold=plasticity_cfg.dormant_threshold,
        warmup_steps=plasticity_cfg.warmup_steps,
        head_prune_threshold=plasticity_cfg.head_prune_threshold,
        head_prune_max_per_layer=plasticity_cfg.head_prune_max_per_layer,
        convergence_delta=plasticity_cfg.convergence_delta,
        convergence_window=plasticity_cfg.convergence_window,
        max_splits_per_layer=plasticity_cfg.max_splits_per_layer,
        max_split_layers=plasticity_cfg.max_split_layers,
        neurogenesis_count=plasticity_cfg.neurogenesis_count,
        lrp_pruning=plasticity_cfg.lrp_pruning,
        min_age_for_removal=plasticity_cfg.min_age_for_removal,
        min_d_ff=plasticity_cfg.min_d_ff,
        weight_norm_clip=plasticity_cfg.weight_norm_clip,
        neurogenesis_interval=plasticity_cfg.neurogenesis_interval,
        total_steps=train_cfg.total_steps,
        structural_freeze_fraction=plasticity_cfg.structural_freeze_fraction,
        structural_freeze_lr=plasticity_cfg.structural_freeze_lr,
        min_age_for_merge=plasticity_cfg.min_age_for_merge,
        max_merges_per_event=plasticity_cfg.max_merges_per_event,
        max_removed_per_event=plasticity_cfg.max_removed_per_event,
        max_reinit_per_event=plasticity_cfg.max_reinit_per_event,
        neurogenesis_skip_slope_window=plasticity_cfg.neurogenesis_skip_slope_window,
        neurogenesis_skip_slope_threshold=plasticity_cfg.neurogenesis_skip_slope_threshold,
        structural_backoff_steps=plasticity_cfg.structural_backoff_steps,
        structural_backoff_recovery_evals=plasticity_cfg.structural_backoff_recovery_evals,
        structural_backoff_recovery_tolerance=plasticity_cfg.structural_backoff_recovery_tolerance,
        structural_only_hold_window=plasticity_cfg.structural_only_hold_window,
    )

    # Gradient diversity tracker (Lyle et al. 2023)
    grad_diversity_tracker = GradientDiversityTracker(
        projection_dim=64, decay=0.99,
    )

    # Data pipeline — separate train/val streams (Change 8)
    seq_len = data_cfg.seq_len if not quick else 64
    book_dir = os.path.expanduser(data_cfg.book_dir)
    jsonl_path = os.path.expanduser(data_cfg.reasoning_jsonl_path)
    selflearn_dir = os.path.expanduser(data_cfg.selflearn_dir)
    hf_datasets = data_cfg.hf_datasets if data_cfg.hf_datasets else None
    mixing_weights = list(data_cfg.mixing_weights)
    if len(mixing_weights) >= 4:
        selflearn_weight = max(0.0, min(data_cfg.selflearn_weight, 0.10))
        base_sum = sum(mixing_weights[:3]) or 1.0
        scale = (1.0 - selflearn_weight) / base_sum
        mixing_weights = [w * scale for w in mixing_weights[:3]] + [selflearn_weight]

    has_books = os.path.isdir(book_dir)
    has_jsonl = os.path.exists(jsonl_path)
    has_hf = hf_datasets and len(hf_datasets) > 0
    has_selflearn = os.path.isdir(selflearn_dir)

    if has_books or has_jsonl or has_hf or has_selflearn:
        # Build split-aware train and val iterators
        train_iter, val_iter = build_train_val_pipelines(
            tokenizer=tokenizer,
            book_dir=book_dir if has_books else "__none__",
            hf_datasets=hf_datasets,
            jsonl_path=jsonl_path if has_jsonl else None,
            selflearn_dir=selflearn_dir if has_selflearn else None,
            selflearn_max_shards=data_cfg.selflearn_max_shards,
            mixing_weights=mixing_weights,
            seq_len=seq_len,
            batch_size=train_cfg.batch_size,
            val_fraction=data_cfg.val_fraction,
        )
        logger.info(
            f"Data pipeline: books={'yes' if has_books else 'no'}, "
            f"hf={'yes' if has_hf else 'no'}, jsonl={'yes' if has_jsonl else 'no'}, "
            f"selflearn={'yes' if has_selflearn else 'no'}, "
            f"val_fraction={data_cfg.val_fraction}"
        )
    else:
        logger.warning("No data sources found, using synthetic data")

        class _SyntheticStream:
            def __iter__(self):
                while True:
                    yield "The quick brown fox jumps over the lazy dog. " * 20

        syn_train = MixedDataStream([_SyntheticStream()], [1.0])
        train_iter = TokenizedBatchIterator(
            syn_train, tokenizer, seq_len=seq_len,
            batch_size=train_cfg.batch_size,
        )
        syn_val = MixedDataStream([_SyntheticStream()], [1.0])
        val_iter = TokenizedBatchIterator(
            syn_val, tokenizer, seq_len=seq_len,
            batch_size=train_cfg.batch_size,
        )

    # Keep references for recreating iterators on exhaustion
    _train_cfg_snapshot = {
        "book_dir": book_dir if has_books else "__none__",
        "hf_datasets": hf_datasets,
        "jsonl_path": jsonl_path if has_jsonl else None,
        "selflearn_dir": selflearn_dir if has_selflearn else None,
        "selflearn_max_shards": data_cfg.selflearn_max_shards,
        "mixing_weights": mixing_weights,
        "seq_len": seq_len,
        "batch_size": train_cfg.batch_size,
        "val_fraction": data_cfg.val_fraction,
    }

    # Metrics
    metrics_logger = MetricsLogger()
    metrics_path = os.path.join(run_dir, "metrics.jsonl")

    # Compile loss + grad function
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # Training state
    accumulated_grads = None
    accum_loss = 0.0
    accum_count = 0
    avg_loss = float("inf")
    best_val_loss = float("inf")
    _last_reset_val_loss: Optional[float] = None
    training_start = time.time()
    total_tokens = 0
    pending_eval_step: Optional[int] = None
    pending_checkpoint_step: Optional[int] = None

    logger.info(f"Starting training: {train_cfg.total_steps} steps")
    logger.info(
        f"Effective batch: {train_cfg.batch_size} x {train_cfg.grad_accum_steps}"
    )

    # Write initial snapshot (only for fresh runs)
    if start_step == 0:
        initial_snap = plasticity.capture_snapshot(
            model, step=0, train_loss=float("inf"), val_loss=float("inf")
        )
        with open(metrics_path, "w") as f:
            f.write(json.dumps(initial_snap) + "\n")

    model.train()

    for step in range(start_step + 1, train_cfg.total_steps + 1):
        step_start = time.time()

        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = _rebuild_data_iter(
                _train_cfg_snapshot, "train", tokenizer, seq_len, train_cfg.batch_size,
            )
            batch = next(train_iter)

        loss, grads = loss_and_grad(model, batch, train_cfg.label_smoothing)
        _mx_materialize(loss, grads)

        accum_loss += loss.item()
        accum_count += 1
        total_tokens += batch.size

        # Update gradient diversity tracker (Lyle et al. 2023)
        grad_diversity_tracker.update(grads)

        del loss, batch  # Release graph references

        # Accumulate gradients
        if accumulated_grads is None:
            accumulated_grads = grads
        else:
            accumulated_grads = mlx.utils.tree_map(
                lambda a, b: a + b, accumulated_grads, grads
            )
            _mx_materialize(accumulated_grads)  # Prevent graph buildup
        del grads  # Release reference to this step's gradient tree

        # Apply gradients after accumulation
        if accum_count >= train_cfg.grad_accum_steps:
            avg_grads = mlx.utils.tree_map(
                lambda g: g / accum_count, accumulated_grads
            )
            avg_grads, _ = optim.clip_grad_norm(
                avg_grads, train_cfg.grad_clip_norm
            )
            optimizer.update(model, avg_grads)
            _mx_materialize(model.parameters(), optimizer.state)

            avg_loss = accum_loss / accum_count
            current_lr = optimizer.learning_rate.item()

            step_time = time.time() - step_start
            tps = total_tokens / (time.time() - training_start)

            d_ff_vals = [layer.ffn.d_ff for layer in model.layers]
            sparsity_vals = [layer.ffn.get_sparsity() for layer in model.layers]

            metrics_logger.log_step(
                step, avg_loss, current_lr, tps,
                model.count_parameters(), d_ff_vals, sparsity_vals,
            )

            if step % train_cfg.log_every == 0:
                active_mb = mx.metal.get_active_memory() / 1024**2
                logger.info(
                    f"Step {step}/{train_cfg.total_steps} | "
                    f"Loss: {avg_loss:.4f} | LR: {current_lr:.2e} | "
                    f"Tok/s: {tps:.0f} | Params: {model.count_parameters():,} | "
                    f"Metal: {active_mb:.0f}MB"
                )

            # Sever computation graph chain: delete intermediate arrays that
            # hold references to old parameter/gradient graph nodes, then
            # reclaim Metal GPU memory. Without this, each optimizer step
            # extends an unbounded graph chain (old_params -> new_params -> ...)
            # that grows ~128MB/step and exhausts memory around step 1200.
            del avg_grads
            accumulated_grads = None
            accum_loss = 0.0
            accum_count = 0
            gc.collect()
            mx.metal.clear_cache()

        # Mark eval/checkpoint as pending — deferred to optimizer boundary
        if step % train_cfg.eval_every == 0 and pending_eval_step is None:
            pending_eval_step = step
        if step % train_cfg.checkpoint_every == 0 and pending_checkpoint_step is None:
            pending_checkpoint_step = step

        # Execute pending eval/plasticity ONLY on optimizer boundaries
        # (accumulated_grads is None and accum_count == 0 means we just
        # completed an optimizer update or haven't started accumulating yet)
        if pending_eval_step is not None and accumulated_grads is None and accum_count == 0:
            scheduled_step = pending_eval_step
            executed_step = step
            eval_delay = executed_step - scheduled_step
            pending_eval_step = None

            if eval_delay > 0:
                logger.info(
                    f"Eval deferred: scheduled={scheduled_step}, "
                    f"executed={executed_step}, delay={eval_delay} steps"
                )

            # Hard safety guard: no live gradient accumulation state
            assert accumulated_grads is None and accum_count == 0, (
                "Plasticity step attempted with live gradient accumulation state; "
                "eval/plasticity must run on optimizer boundaries only"
            )

            model.eval()

            # Evaluate on held-out val data (not training sources)
            try:
                val_loss = evaluate_model(model, val_iter, num_batches=5)
            except StopIteration:
                val_iter = _rebuild_data_iter(
                    _train_cfg_snapshot, "val", tokenizer, seq_len, train_cfg.batch_size,
                )
                val_loss = evaluate_model(model, val_iter, num_batches=5)
            metrics_logger.log_eval(val_loss, step=executed_step)

            # Compute dense gradients for RigL growth.
            # Only after warmup — no pruning/growth happens before that anyway.
            dense_grads = None
            if executed_step >= plasticity_cfg.warmup_steps:
                dense_grads, train_iter = _compute_dense_gradients(
                    model, loss_and_grad, train_iter,
                    train_cfg.label_smoothing, _train_cfg_snapshot,
                    tokenizer, seq_len, train_cfg.batch_size,
                )
                model.eval()

            # Capture a batch for LRP scoring — embed token IDs into (B, T, d_model)
            # because compute_lrp_scores expects embedded input, not raw tokens
            try:
                lrp_tokens = next(train_iter)
            except StopIteration:
                train_iter = _rebuild_data_iter(
                    _train_cfg_snapshot, "train", tokenizer, seq_len, train_cfg.batch_size,
                )
                lrp_tokens = next(train_iter)
            positions = mx.arange(lrp_tokens.shape[1])
            lrp_batch = model.embedding(lrp_tokens) + model.pos_embedding(positions)
            _mx_materialize(lrp_batch)
            del lrp_tokens

            # Run plasticity with val_loss + LRP batch for convergence/LRP
            # Extract current LR for structural freeze decision
            current_lr_value = optimizer.learning_rate.item()
            events = plasticity.step(
                model, current_step=executed_step, dense_grads=dense_grads,
                val_loss=val_loss, lrp_batch=lrp_batch,
                current_lr=current_lr_value,
            )
            del lrp_batch
            metrics_logger.log_plasticity(events)
            del dense_grads

            # Reclaim Metal GPU memory after plasticity creates many temp arrays
            gc.collect()
            mx.metal.clear_cache()

            # Preserve optimizer state across structural changes.
            # Only reset states for layers that changed shape, keep all other
            # layers' momentum/velocity intact.
            if (events["splits"] > 0 or events["merges"] > 0
                    or events.get("removed", 0) > 0 or events.get("neurogenesis", 0) > 0):
                old_state = optimizer.state
                old_opt_step = old_state.get("step", mx.array(0))

                optimizer = optim.AdamW(
                    learning_rate=lr_schedule,
                    weight_decay=train_cfg.weight_decay,
                    betas=train_cfg.adam_betas,
                    eps=train_cfg.adam_eps,
                )

                # Do a dummy update so MLX creates state entries for all params
                zero_grads = mlx.utils.tree_map(mx.zeros_like, model.parameters())
                optimizer.update(model, zero_grads)
                _mx_materialize(optimizer.state)
                del zero_grads

                # Copy momentum/velocity from old optimizer for params that
                # kept the same shape (non-FFN layers, unchanged FFN layers)
                restored = _restore_optimizer_state(old_state, optimizer.state)
                restored["step"] = old_opt_step
                optimizer.state = restored
                _mx_materialize(optimizer.state)
                del old_state

                changed_count = (
                    events["splits"] + events["merges"]
                    + events.get("neurogenesis", 0) + events.get("removed", 0)
                )
                logger.info(
                    f"Optimizer state preserved across structural change "
                    f"({changed_count} structural events, "
                    f"opt step {old_opt_step.item():.0f})"
                )

            # Last layer periodic reset (Lyle et al. 2023 S4.3)
            # Guards: (1) don't reset in final 20% (no recovery time)
            #         (2) don't reset if still recovering from previous reset
            steps_remaining = train_cfg.total_steps - executed_step
            min_recovery = int(train_cfg.total_steps * 0.2)
            recovery_ok = (
                _last_reset_val_loss is None
                or val_loss <= _last_reset_val_loss * 1.05
            )
            if (plasticity_cfg.last_layer_reset_interval > 0
                    and executed_step > 0
                    and executed_step % plasticity_cfg.last_layer_reset_interval == 0
                    and steps_remaining >= min_recovery
                    and recovery_ok):
                model.reset_last_layer()
                _last_reset_val_loss = val_loss

                # Reset optimizer state for the freshly randomized last-layer
                # FFN and final_norm — stale Adam moments would cause instability
                last_layer_idx = model_cfg.num_layers - 1
                old_state = optimizer.state
                old_opt_step = old_state.get("step", mx.array(0))

                optimizer = optim.AdamW(
                    learning_rate=lr_schedule,
                    weight_decay=train_cfg.weight_decay,
                    betas=train_cfg.adam_betas,
                    eps=train_cfg.adam_eps,
                )
                zero_grads = mlx.utils.tree_map(mx.zeros_like, model.parameters())
                optimizer.update(model, zero_grads)
                _mx_materialize(optimizer.state)
                del zero_grads

                # Skip prefixes: reset-layer FFN and final_norm get fresh state
                skip = [
                    ("layers", last_layer_idx, "ffn"),
                    ("final_norm",),
                ]
                restored = _restore_optimizer_state(
                    old_state, optimizer.state, skip_prefixes=skip,
                )
                restored["step"] = old_opt_step
                optimizer.state = restored
                _mx_materialize(optimizer.state)
                del old_state

                logger.info(
                    f"Step {executed_step}: Reset last layer FFN (mask preserved) + "
                    f"optimizer state for layer {last_layer_idx} FFN + final_norm"
                )

            # Diagnostics logging (effective rank, gradient diversity)
            layer_diag = compute_layer_diagnostics(model)
            grad_diversity = grad_diversity_tracker.get_diversity_score()

            snapshot = plasticity.capture_snapshot(
                model, executed_step,
                avg_loss if metrics_logger.train_losses else 0.0,
                val_loss,
            )
            # Augment snapshot with paper-informed diagnostics
            snapshot["layer_diagnostics"] = layer_diag
            snapshot["gradient_diversity"] = round(grad_diversity, 4)
            snapshot["structural_mode"] = events.get("_mode", plasticity._last_structural_mode)
            snapshot["structural_backoff_active"] = executed_step < plasticity._backoff_until_step
            snapshot["scheduled_eval_step"] = scheduled_step
            snapshot["executed_eval_step"] = executed_step
            snapshot["eval_delay_steps"] = eval_delay
            with open(metrics_path, "a") as f:
                f.write(json.dumps(snapshot) + "\n")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model.save_checkpoint(os.path.join(run_dir, "best"))
                logger.info(f"  New best val_loss: {val_loss:.4f}")

            generate_plots(metrics_logger, os.path.join(run_dir, "plots"))

            # Reclaim memory after matplotlib + checkpoint + plasticity
            gc.collect()
            mx.metal.clear_cache()

            active_mb = mx.metal.get_active_memory() / 1024**2
            peak_mb = mx.metal.get_peak_memory() / 1024**2
            cache_mb = mx.metal.get_cache_memory() / 1024**2
            logger.info(
                f"  Eval step {executed_step} (scheduled {scheduled_step}): "
                f"val_loss={val_loss:.4f} | Events: {events} | "
                f"Grad diversity: {grad_diversity:.2f} | "
                f"Metal: {active_mb:.0f}MB active, {peak_mb:.0f}MB peak, {cache_mb:.0f}MB cache"
            )

            model.train()

        # Execute pending checkpoint ONLY on optimizer boundaries
        if pending_checkpoint_step is not None and accumulated_grads is None and accum_count == 0:
            ckpt_scheduled = pending_checkpoint_step
            pending_checkpoint_step = None
            ckpt_dir = os.path.join(run_dir, "checkpoints", f"step_{step}")
            model.save_checkpoint(ckpt_dir)
            if step != ckpt_scheduled:
                logger.info(
                    f"Checkpoint due at {ckpt_scheduled} executed at {step}"
                )

        # Graceful shutdown on signal
        if _shutdown_requested:
            logger.info(f"Graceful shutdown at step {step}")
            # Execute any pending eval before saving
            if pending_eval_step is not None:
                logger.info(f"Executing pending eval (scheduled {pending_eval_step}) before shutdown")
                model.eval()
                try:
                    val_loss = evaluate_model(model, val_iter, num_batches=5)
                except StopIteration:
                    val_iter = _rebuild_data_iter(
                        _train_cfg_snapshot, "val", tokenizer, seq_len, train_cfg.batch_size,
                    )
                    val_loss = evaluate_model(model, val_iter, num_batches=5)
                metrics_logger.log_eval(val_loss, step=step)
                snapshot = plasticity.capture_snapshot(
                    model, step,
                    avg_loss if metrics_logger.train_losses else 0.0,
                    val_loss,
                )
                snapshot["scheduled_eval_step"] = pending_eval_step
                snapshot["executed_eval_step"] = step
                snapshot["eval_delay_steps"] = step - pending_eval_step
                with open(metrics_path, "a") as f:
                    f.write(json.dumps(snapshot) + "\n")
                pending_eval_step = None
                model.train()
            model.save_checkpoint(os.path.join(run_dir, "interrupted"))
            generate_plots(metrics_logger, os.path.join(run_dir, "plots"))
            logger.info(f"Emergency checkpoint saved to {run_dir}/interrupted")
            break

    # Execute any pending eval before final save
    if pending_eval_step is not None:
        logger.info(f"Executing pending eval (scheduled {pending_eval_step}) at end of training")
        model.eval()
        try:
            val_loss = evaluate_model(model, val_iter, num_batches=5)
        except StopIteration:
            val_iter = _rebuild_data_iter(
                _train_cfg_snapshot, "val", tokenizer, seq_len, train_cfg.batch_size,
            )
            val_loss = evaluate_model(model, val_iter, num_batches=5)
        metrics_logger.log_eval(val_loss, step=step)
        snapshot = plasticity.capture_snapshot(
            model, step,
            avg_loss if metrics_logger.train_losses else 0.0,
            val_loss,
        )
        snapshot["scheduled_eval_step"] = pending_eval_step
        snapshot["executed_eval_step"] = step
        snapshot["eval_delay_steps"] = step - pending_eval_step
        with open(metrics_path, "a") as f:
            f.write(json.dumps(snapshot) + "\n")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_checkpoint(os.path.join(run_dir, "best"))
            logger.info(f"  New best val_loss: {val_loss:.4f}")
        pending_eval_step = None
        model.train()

    # Final save
    model.save_checkpoint(os.path.join(run_dir, "final"))
    if not os.path.isdir(os.path.join(run_dir, "best")):
        model.save_checkpoint(os.path.join(run_dir, "best"))
        logger.info("No prior best checkpoint existed; saved final weights as best")
    generate_plots(metrics_logger, os.path.join(run_dir, "plots"))

    elapsed = time.time() - training_start
    logger.info(f"Training complete: {elapsed / 60:.1f} minutes")
    logger.info(f"Best val loss: {best_val_loss:.4f}")
    logger.info(f"Run directory: {run_dir}")

    return run_dir


# ---------------------------------------------------------------------------
# Part E: CLI
# ---------------------------------------------------------------------------

app = typer.Typer()


@app.command()
def main(
    quick: bool = typer.Option(False, help="50-step smoke test with tiny model"),
    total_steps: Optional[int] = typer.Option(
        None, help="Override total training steps"
    ),
    tokenizer_path: Optional[str] = typer.Option(
        None, help="Path to tokenizer.json"
    ),
    resume: Optional[str] = typer.Option(
        None, help="Path to checkpoint dir to resume from (e.g. runs/20250101_120000/interrupted)"
    ),
    init_checkpoint: Optional[str] = typer.Option(
        None, help="Initialize from checkpoint weights but start a fresh run"
    ),
) -> None:
    """Train the MLX Neuroplasticity Engine."""
    # When resuming, reuse the original run directory
    if resume:
        run_dir = os.path.dirname(resume.rstrip("/"))
        # Handle checkpoints/step_N paths — go up one more level
        if os.path.basename(run_dir) == "checkpoints":
            run_dir = os.path.dirname(run_dir)
        logger.info(f"Resuming into run dir: {run_dir}")
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join("runs", run_id)

    model_cfg = MLXModelConfig()
    train_cfg = MLXTrainingConfig()
    data_cfg = MLXDataConfig()
    plasticity_cfg = PlasticityConfig()

    if total_steps is not None:
        train_cfg.total_steps = total_steps

    train(
        model_cfg, train_cfg, data_cfg, plasticity_cfg, run_dir,
        tokenizer_path=tokenizer_path, quick=quick, resume=resume,
        init_checkpoint=init_checkpoint,
    )


if __name__ == "__main__":
    import sys

    # If CLI args are passed (e.g., --tokenizer-path, --total-steps, --quick),
    # run the typer CLI for actual training. Otherwise run validation tests.
    if len(sys.argv) > 1:
        app()
        sys.exit(0)

    all_validation_failures: List[str] = []
    total_tests = 0

    # Test: Quick mode smoke test
    total_tests += 1
    try:
        run_id = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir = os.path.join("runs", run_id)

        result_dir = train(
            model_cfg=MLXModelConfig(),
            train_cfg=MLXTrainingConfig(),
            data_cfg=MLXDataConfig(),
            plasticity_cfg=PlasticityConfig(),
            run_dir=run_dir,
            quick=True,
        )

        # Verify outputs
        if not os.path.exists(os.path.join(result_dir, "metrics.jsonl")):
            all_validation_failures.append("metrics.jsonl not created")

        if not os.path.exists(os.path.join(result_dir, "plots", "loss_curves.png")):
            all_validation_failures.append("loss_curves.png not created")

        if not os.path.exists(os.path.join(result_dir, "final", "arch_info.json")):
            all_validation_failures.append("final checkpoint not created")

        # Verify metrics.jsonl has entries
        with open(os.path.join(result_dir, "metrics.jsonl")) as f:
            lines = f.readlines()
        if len(lines) < 2:
            all_validation_failures.append(
                f"metrics.jsonl too few entries: {len(lines)}"
            )

        # Verify loss decreased
        first = json.loads(lines[0])
        last = json.loads(lines[-1])
        if (
            last.get("train_loss", float("inf")) >= first.get("train_loss", 0)
            and first.get("train_loss", 0) != float("inf")
        ):
            logger.warning(
                f"Loss did not decrease: {first.get('train_loss')} -> "
                f"{last.get('train_loss')}"
            )

        if not all_validation_failures:
            print(f"  Test 1 OK: quick training completed, run_dir={result_dir}")
            print(f"    metrics.jsonl: {len(lines)} entries")
            plot_files = os.listdir(os.path.join(result_dir, "plots"))
            print(f"    plots: {plot_files}")
    except Exception as e:
        all_validation_failures.append(f"Quick training failed: {e}")
        import traceback
        traceback.print_exc()

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
            f"\n\u2705 VALIDATION PASSED - All {total_tests} tests produced "
            f"expected results"
        )
        sys.exit(0)
