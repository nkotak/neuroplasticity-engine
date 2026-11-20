"""
MLX structural plasticity controller for the Neuroplasticity Engine.

Implements RigL-style gradient-informed growth, target sparsity maintenance,
dormant neuron reinitialization, plasticity warmup, structured attention head
pruning, convergence-triggered splits, LRP-informed pruning, and
function-preserving neuron splits/merges.

Docs: https://ml-explore.github.io/mlx/
      Evci et al. (2020) "RigL" — gradient-informed growth
      Dohare et al. (2024) "Loss of Plasticity" — dormant neuron reinit
      Li et al. (2025) "Dropin & Out Learning" — convergence-based splits, LRP pruning
      Lyle et al. (ICML 2023) — effective rank diagnostics

Sample input:
    controller = MLXPlasticityController(target_sparsity=0.4, warmup_steps=5000)
    events = controller.step(model, current_step=1000, dense_grads=grads, val_loss=5.0)

Expected output:
    events -> {"pruned": 100, "grown": 20, "splits": 1, "merges": 0,
               "reinit": 2, "heads_pruned": 1}
"""

import math
from collections import defaultdict
from dataclasses import dataclass, field as dc_field
from typing import Dict, List, Optional

import mlx.core as mx
import numpy as np
from loguru import logger

from mlx_adaptive_feedforward import MLXAdaptiveFeedForward
from mlx_model import MLXAdaptiveTransformer


@dataclass
class _PendingRecovery:
    """Tracks a structural event awaiting recovery confirmation."""
    event_step: int
    pre_event_val_loss: float
    structural_mode: str
    events: Dict[str, int]
    evals_since: int = 0
    recovered: Optional[bool] = None


class MLXPlasticityController:
    """Controls structural plasticity with research-backed improvements.

    Key changes from original:
    - RigL-style gradient-informed growth (not random)
    - Target sparsity maintenance (~40%) for regularization
    - Dormant neuron reinitialization (Dohare et al. 2024)
    - Plasticity warmup (no pruning during early training)
    - Structured attention head pruning
    - Convergence-triggered splits (Li et al. 2025 "Dropin")
    - LRP-informed pruning (Li et al. 2025)
    - Effective rank for split layer selection (Lyle et al. 2023)
    """

    def __init__(
        self,
        prune_threshold: float = 0.005,
        growth_rate: float = 0.2,
        split_threshold: float = 0.5,
        merge_similarity_threshold: float = 0.6,
        prune_frequency: int = 10,
        structural_frequency: int = 20,
        min_connections: int = 1000,
        growth_noise_scale: float = 0.01,
        split_noise_scale: float = 0.1,
        target_sparsity: float = 0.4,
        dormant_threshold: float = 0.001,
        warmup_steps: int = 5000,
        head_prune_threshold: float = 0.1,
        head_prune_max_per_layer: int = 1,
        convergence_delta: float = 0.03,
        convergence_window: int = 6,
        max_splits_per_layer: int = 4,
        max_split_layers: int = 3,
        neurogenesis_count: int = 4,
        lrp_pruning: bool = True,
        min_age_for_removal: int = 5000,
        min_d_ff: int = 64,
        weight_norm_clip: float = 30.0,
        # Fix 1: Scheduled neurogenesis
        neurogenesis_interval: int = 2000,
        # Fix 2: Structural freeze
        total_steps: int = 50000,
        structural_freeze_fraction: float = 0.15,
        structural_freeze_lr: float = 1e-5,
        # Fix 3: Merge cooldown
        min_age_for_merge: int = 3000,
        # Event intensity caps
        max_merges_per_event: int = 2,
        max_removed_per_event: int = 1,
        max_reinit_per_event: int = 1,
        # Neurogenesis trend gate
        neurogenesis_skip_slope_window: int = 4,
        neurogenesis_skip_slope_threshold: float = -0.05,
        # Structural backoff
        structural_backoff_steps: int = 3000,
        structural_backoff_recovery_evals: int = 3,
        structural_backoff_recovery_tolerance: float = 0.05,
        # Budget hold
        structural_only_hold_window: int = 3,
    ) -> None:
        self.prune_threshold = prune_threshold
        self.growth_rate = growth_rate
        self.split_threshold = split_threshold
        self.merge_similarity_threshold = merge_similarity_threshold
        self.prune_frequency = prune_frequency
        self.structural_frequency = structural_frequency
        self.min_connections = min_connections
        self.growth_noise_scale = growth_noise_scale
        self.split_noise_scale = split_noise_scale
        self.target_sparsity = target_sparsity
        self.dormant_threshold = dormant_threshold
        self.warmup_steps = warmup_steps
        self.head_prune_threshold = head_prune_threshold
        self.head_prune_max_per_layer = head_prune_max_per_layer
        self.convergence_delta = convergence_delta
        self.convergence_window = convergence_window
        self.max_splits_per_layer = max_splits_per_layer
        self.max_split_layers = max_split_layers
        self.neurogenesis_count = neurogenesis_count
        self.lrp_pruning = lrp_pruning
        self.min_age_for_removal = min_age_for_removal
        self.min_d_ff = min_d_ff
        self.weight_norm_clip = weight_norm_clip
        self.neurogenesis_interval = neurogenesis_interval
        self.total_steps = total_steps
        self.structural_freeze_fraction = structural_freeze_fraction
        self.structural_freeze_lr = structural_freeze_lr
        self.min_age_for_merge = min_age_for_merge
        self.max_merges_per_event = max_merges_per_event
        self.max_removed_per_event = max_removed_per_event
        self.max_reinit_per_event = max_reinit_per_event
        self.neurogenesis_skip_slope_window = neurogenesis_skip_slope_window
        self.neurogenesis_skip_slope_threshold = neurogenesis_skip_slope_threshold
        self.structural_backoff_steps = structural_backoff_steps
        self.structural_backoff_recovery_evals = structural_backoff_recovery_evals
        self.structural_backoff_recovery_tolerance = structural_backoff_recovery_tolerance
        self.structural_only_hold_window = structural_only_hold_window

        self.epoch_counter = 0
        self.stats: Dict[str, List[int]] = defaultdict(list)
        # Budget-coupled growth: can only grow what was pruned
        self._prune_budget: int = 0
        # Structural events inject active connections outside prune/grow;
        # this debt must be repaid from future prune budget before growth.
        self._structural_budget_debt: int = 0
        # Convergence detection state
        self._val_loss_history: List[float] = []
        # Cache for LRP sample input (set by training loop)
        self._lrp_batch: Optional[mx.array] = None
        # Track whether freeze was already logged
        self._freeze_logged: bool = False

        # Scheduled neurogenesis: next step to fire (>= check, not modulo)
        if self.neurogenesis_interval > 0:
            first = self.neurogenesis_interval
            while first <= self.warmup_steps:
                first += self.neurogenesis_interval
            self.next_neurogenesis_step = first
        else:
            self.next_neurogenesis_step = float('inf')
        # Structural mode round-robin counter
        self._structural_cycle: int = 0
        # Pending recovery records for backoff system
        self._pending_recoveries: List[_PendingRecovery] = []
        # Backoff: skip structural edits until this step
        self._backoff_until_step: int = 0
        # Per-layer neurogenesis tracking
        self._neurogenesis_per_layer: Dict[int, int] = defaultdict(int)
        self._last_neurogenesis_step_per_layer: Dict[int, int] = defaultdict(int)
        # Last structural mode for snapshot
        self._last_structural_mode: str = "none"
        # Budget hold: recent (grown, pruned, debt) snapshots for stall detection
        self._budget_hold_history: List[tuple] = []  # (cum_grown, cum_pruned, debt)
        self._in_budget_hold: bool = False

    def step(
        self,
        model: MLXAdaptiveTransformer,
        current_step: int = 0,
        dense_grads: Optional[Dict] = None,
        val_loss: Optional[float] = None,
        lrp_batch: Optional[mx.array] = None,
        current_lr: float = 0.0,
    ) -> Dict[str, int]:
        """Run plasticity operations at configured intervals.

        Implements mutually exclusive structural modes with priority order:
        frozen > backoff > neurogenesis > split > merge > maintenance > none

        Args:
            model: The transformer model.
            current_step: Current training step (for warmup gating).
            dense_grads: Gradients from unmasked forward pass for RigL growth.
            val_loss: Current validation loss (for convergence-based splits).
            lrp_batch: Sample input batch for LRP score computation.
            current_lr: Current learning rate (for structural freeze).

        Returns dict of event counts for this step.
        """
        self.epoch_counter += 1
        events: Dict[str, int] = {
            "pruned": 0, "grown": 0, "splits": 0, "merges": 0,
            "reinit": 0, "removed": 0, "heads_pruned": 0,
            "neurogenesis": 0,
        }

        if val_loss is not None:
            self._val_loss_history.append(val_loss)
        if lrp_batch is not None:
            self._lrp_batch = lrp_batch

        # Update pending recovery records
        self._update_pending_recoveries(val_loss, current_step)

        # Select structural mode (mutually exclusive)
        mode = self._select_structural_mode(current_step, current_lr)
        self._last_structural_mode = mode

        # Prune/grow: independent of mode, continues during backoff and freeze
        is_disruptive = mode not in ("none", "frozen", "backoff")
        if self.epoch_counter % self.prune_frequency == 0:
            if current_step >= self.warmup_steps:
                pruned = self._prune_connections(model)
                events["pruned"] = pruned
                self._prune_budget += pruned
                if self._structural_budget_debt > 0:
                    payoff = min(self._prune_budget, self._structural_budget_debt)
                    self._prune_budget -= payoff
                    self._structural_budget_debt -= payoff
                if not is_disruptive:
                    grown = self._grow_connections(
                        model, dense_grads, max_grow=self._prune_budget,
                    )
                    events["grown"] = grown
                    self._prune_budget = max(0, self._prune_budget - grown)

        # Execute the selected structural mode
        merges_skipped_age = 0
        split_info = None
        if mode == "neurogenesis":
            ng = self._scheduled_neurogenesis(model, current_step)
            events["neurogenesis"] = ng
            self.next_neurogenesis_step += self.neurogenesis_interval
        elif mode == "split":
            split_info = self._split_neurons(model, current_step)
            events["splits"] = split_info["count"]
        elif mode == "merge":
            merge_count, merges_skipped_age = self._merge_neurons(model)
            events["merges"] = merge_count
        elif mode == "maintenance":
            reinit, removed = self._handle_dormant_neurons(model, current_step)
            events["reinit"] = reinit
            events["removed"] = removed
            if current_step >= self.warmup_steps:
                events["heads_pruned"] = self._prune_attention_heads(model)

        # Record pending recovery for disruptive modes
        if is_disruptive and val_loss is not None:
            self._pending_recoveries.append(_PendingRecovery(
                event_step=current_step,
                pre_event_val_loss=val_loss,
                structural_mode=mode,
                events=dict(events),
            ))

        # Update budget hold detection
        self._update_budget_hold()

        # Weight norm clipping (unconditional)
        self._clip_weight_norms(model, max_norm=self.weight_norm_clip)

        # Structured logging
        if mode != "none":
            conv_slope, conv_range = self._convergence_stats()
            branch = split_info["branch"] if split_info else mode
            backoff_active = current_step < self._backoff_until_step
            logger.info(
                f"Structural step {current_step}: mode={mode}, branch={branch}, "
                f"splits={events['splits']}, merges={events['merges']}, "
                f"merges_skipped_age={merges_skipped_age}, "
                f"neurogenesis={events['neurogenesis']}, "
                f"reinit={events['reinit']}, removed={events['removed']}, "
                f"backoff_active={backoff_active}, "
                f"convergence_slope={conv_slope:.4f}, "
                f"convergence_range={conv_range:.4f}"
            )

        return events

    def _select_structural_mode(
        self, current_step: int, current_lr: float,
    ) -> str:
        """Select one mutually-exclusive structural mode per step.

        Priority: frozen > backoff > neurogenesis > split > merge > maintenance > none
        """
        # Freeze check
        freeze_step = int(self.total_steps * (1 - self.structural_freeze_fraction))
        in_freeze = (
            (current_step > freeze_step)
            or (current_lr < self.structural_freeze_lr
                and current_step > self.warmup_steps
                and current_lr > 0)
        )
        if in_freeze:
            if not self._freeze_logged:
                logger.info(
                    f"Structural freeze active: step {current_step}, "
                    f"lr={current_lr:.2e}, freeze_step={freeze_step}"
                )
                self._freeze_logged = True
            return "frozen"

        # Backoff check
        if current_step < self._backoff_until_step:
            return "backoff"

        # Budget hold: pause structural edits when debt rising but prune/grow stalled
        if self._in_budget_hold:
            return "budget_hold"

        # Neurogenesis check (own schedule, pre-empts structural rotation)
        neurogenesis_due = (
            current_step >= self.next_neurogenesis_step
            and self.neurogenesis_count > 0
            and current_step > self.warmup_steps
            and self.neurogenesis_interval > 0
        )
        if neurogenesis_due:
            if self._neurogenesis_trend_gate_skip():
                # Skip but advance schedule
                self.next_neurogenesis_step += self.neurogenesis_interval
                # Fall through to structural check
            else:
                return "neurogenesis"

        # Structural frequency check (round-robin: split -> merge -> maintenance)
        structural_due = (
            self.epoch_counter % self.structural_frequency == 0
            and current_step > self.warmup_steps
        )
        if structural_due:
            cycle = self._structural_cycle % 3
            self._structural_cycle += 1
            if cycle == 0:
                return "split"
            elif cycle == 1:
                return "merge"
            else:
                return "maintenance"

        return "none"

    def _neurogenesis_trend_gate_skip(self) -> bool:
        """Check if validation is improving too strongly to justify neurogenesis.

        Returns True if neurogenesis should be skipped.
        """
        window = self.neurogenesis_skip_slope_window
        if len(self._val_loss_history) < window:
            return False
        recent = self._val_loss_history[-window:]
        n = len(recent)
        x_mean = (n - 1) / 2.0
        y_mean = sum(recent) / n
        numer = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(recent))
        denom = sum((i - x_mean) ** 2 for i in range(n))
        slope = numer / denom if denom > 1e-10 else 0.0

        if slope <= self.neurogenesis_skip_slope_threshold:
            logger.info(
                f"Neurogenesis trend-gate skip: slope={slope:.4f} "
                f"<= threshold={self.neurogenesis_skip_slope_threshold}"
            )
            return True
        return False

    def _update_pending_recoveries(
        self, val_loss: Optional[float], current_step: int,
    ) -> None:
        """Check pending recovery records and activate backoff if needed."""
        if val_loss is None:
            return
        for pr in self._pending_recoveries:
            if pr.recovered is not None:
                continue
            if pr.event_step == current_step:
                continue  # Same step as the event, don't count
            pr.evals_since += 1
            tolerance = self.structural_backoff_recovery_tolerance
            if val_loss <= pr.pre_event_val_loss + tolerance:
                pr.recovered = True
            elif pr.evals_since >= self.structural_backoff_recovery_evals:
                pr.recovered = False
                self._backoff_until_step = current_step + self.structural_backoff_steps
                logger.info(
                    f"Structural backoff activated until step "
                    f"{self._backoff_until_step}: event at step {pr.event_step} "
                    f"({pr.structural_mode}) unrecovered after "
                    f"{pr.evals_since} evals "
                    f"(pre={pr.pre_event_val_loss:.4f}, "
                    f"current={val_loss:.4f})"
                )

    def _update_budget_hold(self) -> None:
        """Check for structural-only regime and enter/exit budget_hold mode.

        Enters budget_hold when over the last N structural checkpoints:
        - cumulative_grown did not change
        - cumulative_pruned did not change
        - structural_budget_debt increased
        Exits when any of those conditions break.
        """
        cum_grown = sum(self.stats.get("total_grown", [0]))
        cum_pruned = sum(self.stats.get("total_pruned", [0]))
        debt = self._structural_budget_debt
        self._budget_hold_history.append((cum_grown, cum_pruned, debt))

        w = self.structural_only_hold_window
        if len(self._budget_hold_history) < w:
            return

        recent = self._budget_hold_history[-w:]
        grown_flat = recent[-1][0] == recent[0][0]
        pruned_flat = recent[-1][1] == recent[0][1]
        debt_rising = (
            all(recent[i][2] >= recent[i - 1][2] for i in range(1, len(recent)))
            and recent[-1][2] > recent[0][2]
        )

        should_hold = grown_flat and pruned_flat and debt_rising

        if should_hold and not self._in_budget_hold:
            self._in_budget_hold = True
            logger.info(
                f"Budget hold activated: debt {recent[0][2]}->{recent[-1][2]}, "
                f"grown flat at {recent[0][0]}, pruned flat at {recent[0][1]}"
            )
        elif not should_hold and self._in_budget_hold:
            self._in_budget_hold = False
            logger.info(
                f"Budget hold released: grown={recent[-1][0]}, "
                f"pruned={recent[-1][1]}, debt={recent[-1][2]}"
            )

    def _convergence_stats(self) -> tuple:
        """Compute current convergence slope and range for logging."""
        if len(self._val_loss_history) < self.convergence_window:
            return 0.0, 0.0
        recent = self._val_loss_history[-self.convergence_window:]
        n = len(recent)
        x_mean = (n - 1) / 2.0
        y_mean = sum(recent) / n
        numer = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(recent))
        denom = sum((i - x_mean) ** 2 for i in range(n))
        slope = numer / denom if denom > 1e-10 else 0.0
        val_range = max(recent) - min(recent)
        return slope, val_range

    def _is_converged(self) -> bool:
        """Detect loss plateau via linear regression slope + range check.

        Li et al. (2025) "Dropin" Algorithm 3: splits should trigger when the
        model has converged (loss plateau). Uses slope-based detection rather
        than consecutive-delta to avoid false triggers from V-shapes.
        """
        if len(self._val_loss_history) < self.convergence_window:
            return False
        recent = self._val_loss_history[-self.convergence_window:]

        # Linear regression slope
        n = len(recent)
        x_mean = (n - 1) / 2.0
        y_mean = sum(recent) / n
        numer = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(recent))
        denom = sum((i - x_mean) ** 2 for i in range(n))
        slope = numer / denom if denom > 1e-10 else 0.0

        # Also check range (reject V-shapes where slope ~0 but variance high)
        val_range = max(recent) - min(recent)
        return abs(slope) < self.convergence_delta and val_range < self.convergence_delta * 4

    def _get_ffn_modules(
        self, model: MLXAdaptiveTransformer
    ) -> List[MLXAdaptiveFeedForward]:
        """Collect all AdaptiveFeedForward modules from the model."""
        return [layer.ffn for layer in model.layers]

    @staticmethod
    def _select_split_candidates(
        ffn: MLXAdaptiveFeedForward, max_candidates: int,
    ) -> List[int]:
        """Select distinct neuron indices for splitting before any splits.

        If persistent activation EMA exists, rank by descending activation.
        Otherwise fall back to descending output-column norm.
        """
        _materialize = mx.eval  # noqa: S307

        mean_act = ffn._persistent_mean_activation
        if mean_act is not None:
            _materialize(mean_act)
            scores = mean_act[:ffn.d_ff]
        else:
            # Fallback: output column norms
            ow = ffn.output_layer.weight  # (d_model, d_ff)
            scores = mx.sqrt(mx.sum(ow * ow, axis=0))  # (d_ff,)
            _materialize(scores)

        # Descending sort → top candidates
        sorted_idx = mx.argsort(-scores)
        _materialize(sorted_idx)
        candidates = []
        for i in range(min(int(sorted_idx.shape[0]), ffn.d_ff)):
            idx = int(sorted_idx[i].item())
            candidates.append(idx)
            if len(candidates) >= max_candidates:
                break
        return candidates

    @staticmethod
    def _remap_indices_after_removals(
        indices: List[int], removed_indices: List[int],
    ) -> List[int]:
        """Remap original indices after neurons at removed_indices were deleted.

        For each original index, compute:
            new_idx = old_idx - count(removed_idx < old_idx)
        """
        removed_sorted = sorted(removed_indices)
        result = []
        for idx in indices:
            shift = sum(1 for r in removed_sorted if r < idx)
            result.append(idx - shift)
        return result

    def _adjust_structural_budget(self, delta_active: int) -> None:
        """Account for active-connection changes from structural events.

        Positive delta (densification) → add to structural debt.
        Negative delta (sparsification) → pay down debt, then credit budget.
        """
        if delta_active > 0:
            self._structural_budget_debt += delta_active
        elif delta_active < 0:
            credit = -delta_active
            if self._structural_budget_debt > 0:
                payoff = min(credit, self._structural_budget_debt)
                self._structural_budget_debt -= payoff
                credit -= payoff
            self._prune_budget += credit

    def _scheduled_neurogenesis(
        self, model: MLXAdaptiveTransformer, current_step: int,
    ) -> int:
        """Add 1 fresh neuron to each of the most bottlenecked layers.

        Distributes neurogenesis across layers rather than dumping all into one.
        Selects neurogenesis_count worst layers by effective rank, breaking ties
        by oldest last_neurogenesis_step so no layer hoards all growth.
        """
        _materialize = mx.eval  # noqa: S307
        from mlx_diagnostics import compute_effective_rank

        ffn_modules = self._get_ffn_modules(model)
        layer_ranks = []
        for layer_idx, ffn in enumerate(ffn_modules):
            eff_rank = compute_effective_rank(ffn.input_layer.weight)
            last_ng = self._last_neurogenesis_step_per_layer.get(layer_idx, 0)
            layer_ranks.append((layer_idx, eff_rank, last_ng))

        if not layer_ranks:
            return 0

        # Sort by: lowest rank first, then oldest neurogenesis (ascending last_ng)
        layer_ranks.sort(key=lambda x: (x[1], x[2]))

        num_target = min(self.neurogenesis_count, len(layer_ranks))
        selected = layer_ranks[:num_target]
        total_added = 0

        for layer_idx, eff_rank, _ in selected:
            ffn = ffn_modules[layer_idx]
            new_active = ffn.add_neurons(
                1, active_fraction=1.0 - self.target_sparsity,
            )
            self._adjust_structural_budget(new_active)
            _materialize(
                ffn.input_layer.weight, ffn.input_layer.bias,
                ffn.output_layer.weight,
                ffn._input_mask, ffn._output_mask, ffn._neuron_age,
            )
            self._neurogenesis_per_layer[layer_idx] += 1
            self._last_neurogenesis_step_per_layer[layer_idx] = current_step
            total_added += 1
            logger.info(
                f"Neurogenesis: 1 fresh neuron in layer {layer_idx} "
                f"(eff_rank={eff_rank:.1f}, d_ff={ffn.d_ff}, "
                f"total_ng_this_layer={self._neurogenesis_per_layer[layer_idx]})"
            )

        self.stats["total_neurogenesis"].append(total_added)
        return total_added

    def _prune_connections(self, model: MLXAdaptiveTransformer) -> int:
        """Prune connections adaptively to maintain target sparsity.

        Uses LRP-informed pruning when enabled and a sample batch is available
        (Li et al. 2025). LRP scores capture weight x activation contributions,
        so a large weight receiving zero activations gets zero relevance.
        Falls back to magnitude-based pruning otherwise.
        """
        # Alias for MLX array materialization (NOT Python's eval builtin)
        _materialize = mx.eval  # noqa: S307

        total_pruned = 0

        # Compute LRP scores if enabled and sample batch available
        lrp_scores_cache: Dict[int, Dict[str, mx.array]] = {}
        if self.lrp_pruning and self._lrp_batch is not None:
            from mlx_diagnostics import compute_lrp_scores
            for layer_idx, ffn in enumerate(self._get_ffn_modules(model)):
                try:
                    lrp_scores_cache[layer_idx] = compute_lrp_scores(
                        ffn, self._lrp_batch
                    )
                except Exception as e:
                    logger.debug(f"LRP scores failed for layer {layer_idx}: {e}")

        for layer_idx, ffn in enumerate(self._get_ffn_modules(model)):
            current_sparsity = ffn.get_sparsity()

            for name, layer_obj, mask in [
                ("input", ffn.input_layer, ffn._input_mask),
                ("output", ffn.output_layer, ffn._output_mask),
            ]:
                weight = layer_obj.weight
                current_active = int(mx.sum(mask != 0).item())

                if current_active <= self.min_connections:
                    continue

                # Adaptive pruning rate based on sparsity gap
                sparsity_gap = self.target_sparsity - current_sparsity
                if sparsity_gap <= 0:
                    continue

                adaptive_rate = min(0.2, sparsity_gap * 3)
                num_to_prune = max(1, int(current_active * adaptive_rate))

                max_prune = current_active - self.min_connections
                num_to_prune = min(num_to_prune, max_prune)
                if num_to_prune <= 0:
                    continue

                flat_active = (mask != 0).reshape(-1)

                # Choose scoring method: LRP or magnitude
                lrp_key = "input_scores" if name == "input" else "output_scores"
                lrp_data = lrp_scores_cache.get(layer_idx, {}).get(lrp_key)

                if lrp_data is not None:
                    # LRP-informed pruning: prune lowest-relevance weights
                    flat_scores = lrp_data.reshape(-1)
                    prune_mode = "lrp"
                else:
                    # Fallback: magnitude-based pruning
                    flat_scores = (mx.abs(weight) * (mask != 0).astype(mx.float32)).reshape(-1)
                    prune_mode = "magnitude"

                # Set inactive to large value so they sort last
                sort_vals = mx.where(
                    flat_active, flat_scores, mx.ones_like(flat_scores) * 1e9
                )
                sorted_idx = mx.argsort(sort_vals)
                prune_indices = sorted_idx[:num_to_prune]

                prune_flat = mx.zeros(flat_scores.shape)
                prune_flat[prune_indices] = 1.0
                _materialize(prune_flat)
                prune_2d = prune_flat.reshape(mask.shape)
                new_mask = mx.where(prune_2d > 0, mx.zeros_like(mask), mask)

                if name == "input":
                    ffn._input_mask = new_mask
                else:
                    ffn._output_mask = new_mask

                layer_obj.weight = mx.where(
                    new_mask != 0, weight, mx.zeros_like(weight)
                )
                total_pruned += num_to_prune

                logger.debug(
                    f"Layer {layer_idx} {name}: pruned {num_to_prune} "
                    f"({prune_mode}, sparsity gap: {sparsity_gap:.3f})"
                )

        if total_pruned > 0:
            for ffn in self._get_ffn_modules(model):
                _materialize(ffn._input_mask, ffn._output_mask,
                             ffn.input_layer.weight, ffn.output_layer.weight)

        self.stats["total_pruned"].append(total_pruned)
        if total_pruned > 0:
            logger.info(f"Pruned {total_pruned} connections total ({prune_mode})")
        return total_pruned

    def _grow_connections(
        self,
        model: MLXAdaptiveTransformer,
        dense_grads: Optional[Dict] = None,
        max_grow: int = 0,
    ) -> int:
        """Grow connections using RigL-style gradient-informed selection.

        Budget-coupled: only grows up to `max_grow` connections (earned by
        pruning). When dense_grads is provided, uses gradient magnitude at
        inactive positions for selection (RigL-style).
        """
        # Alias for MLX array materialization (NOT Python's eval builtin)
        _materialize = mx.eval  # noqa: S307

        if max_grow <= 0:
            return 0

        total_grown = 0
        remaining_budget = max_grow

        # Distribute budget proportionally across layers
        ffn_modules = self._get_ffn_modules(model)
        num_layers = len(ffn_modules)
        per_layer_budget = max(1, remaining_budget // max(num_layers, 1))

        for layer_idx, ffn in enumerate(ffn_modules):
            if remaining_budget <= 0:
                break

            layer_budget = min(per_layer_budget, remaining_budget)

            for name, layer_obj, mask in [
                ("input", ffn.input_layer, ffn._input_mask),
                ("output", ffn.output_layer, ffn._output_mask),
            ]:
                if remaining_budget <= 0:
                    break

                flat_mask = mask.reshape(-1)
                inactive = flat_mask == 0
                num_inactive = int(mx.sum(inactive).item())
                if num_inactive == 0:
                    continue

                # Budget per sub-layer (input/output split)
                sub_budget = min(
                    layer_budget // 2 + 1, remaining_budget, num_inactive,
                )
                if sub_budget <= 0:
                    continue

                # Try gradient-informed selection (RigL-style)
                grad_at_layer = _extract_ffn_grad(
                    dense_grads, layer_idx, name
                ) if dense_grads is not None else None

                if grad_at_layer is not None:
                    flat_grad = mx.abs(grad_at_layer).reshape(-1)
                    grad_priority = mx.where(
                        inactive, flat_grad,
                        mx.full(flat_grad.shape, -1.0),
                    )
                    sorted_idx = mx.argsort(grad_priority)[::-1]
                    grow_indices = sorted_idx[:sub_budget]
                else:
                    rand_priorities = mx.where(
                        inactive,
                        mx.random.uniform(shape=flat_mask.shape),
                        mx.full(flat_mask.shape, float("inf")),
                    )
                    sorted_idx = mx.argsort(rand_priorities)
                    grow_indices = sorted_idx[:sub_budget]

                grow_mask = mx.zeros(flat_mask.shape)
                grow_mask[grow_indices] = 1.0
                _materialize(grow_mask)

                new_flat_mask = mx.where(
                    grow_mask > 0, mx.ones_like(flat_mask), flat_mask
                )

                flat_weight = layer_obj.weight.reshape(-1)
                if grad_at_layer is not None:
                    flat_grad_sign = mx.sign(grad_at_layer.reshape(-1))
                    noise = (
                        mx.abs(mx.random.normal(flat_weight.shape))
                        * self.growth_noise_scale
                    )
                    init_vals = flat_grad_sign * noise
                else:
                    init_vals = (
                        mx.random.normal(flat_weight.shape) * self.growth_noise_scale
                    )
                new_flat_weight = mx.where(grow_mask > 0, init_vals, flat_weight)

                new_mask = new_flat_mask.reshape(mask.shape)
                new_weight = new_flat_weight.reshape(layer_obj.weight.shape)
                _materialize(new_mask, new_weight)

                if name == "input":
                    ffn._input_mask = new_mask
                else:
                    ffn._output_mask = new_mask
                layer_obj.weight = new_weight

                total_grown += sub_budget
                remaining_budget -= sub_budget
                growth_mode = "gradient" if grad_at_layer is not None else "random"
                logger.debug(
                    f"Layer {layer_idx} {name}: grew {sub_budget} "
                    f"({growth_mode}, budget remaining={remaining_budget})"
                )

        if total_grown > 0:
            for ffn in self._get_ffn_modules(model):
                _materialize(
                    ffn._input_mask, ffn._output_mask,
                    ffn.input_layer.weight, ffn.output_layer.weight,
                )

        self.stats["total_grown"].append(total_grown)
        if total_grown > 0:
            logger.info(f"Grew {total_grown} connections (budget-coupled)")
        return total_grown

    def _handle_dormant_neurons(
        self,
        model: MLXAdaptiveTransformer,
        current_step: int,
    ) -> tuple:
        """Handle dormant neurons: reinitialize young ones, remove old ones.

        Dohare et al. (2024) + Li et al. (2025): Neurons with near-zero
        persistent mean activation are "dead". Young dormant neurons get
        reinitialized (second chance). Old dormant neurons (age > threshold)
        are permanently removed (neuroapoptosis).

        Key correctness properties:
        - Reinit preserves existing masks (no sparsity change → delta_active=0).
        - Indices for reinit are remapped after removals.
        - _persistent_mean_activation is cleared after any removal/reinit to
          prevent stale EMA from retriggering on the next structural step.

        Returns (reinit_count, removed_count).
        """
        # Alias for MLX array materialization (NOT Python's eval builtin)
        _materialize = mx.eval  # noqa: S307

        total_reinit = 0
        total_removed = 0

        for layer_idx, ffn in enumerate(self._get_ffn_modules(model)):
            if (total_removed >= self.max_removed_per_event
                    and total_reinit >= self.max_reinit_per_event):
                break

            mean_act = ffn._persistent_mean_activation
            if mean_act is None:
                continue

            # Find dormant neurons
            dormant = mean_act[:ffn.d_ff] < self.dormant_threshold
            _materialize(dormant)
            dormant_indices = []
            for i in range(min(mean_act.shape[0], ffn.d_ff)):
                if bool(dormant[i].item()):
                    dormant_indices.append(i)
                    if len(dormant_indices) >= 8:
                        break

            if not dormant_indices:
                continue

            # Separate into remove vs reinit based on neuron age
            to_remove = []
            to_reinit_orig = []  # original indices (before removals)
            max_removals = ffn.d_ff // 32

            for idx in dormant_indices:
                age = float(ffn._neuron_age[idx].item())
                if (age > self.min_age_for_removal
                        and ffn.d_ff - len(to_remove) > self.min_d_ff
                        and len(to_remove) < max_removals):
                    to_remove.append(idx)
                else:
                    to_reinit_orig.append(idx)

            # Global caps
            remaining_remove = self.max_removed_per_event - total_removed
            to_remove = to_remove[:remaining_remove]
            remaining_reinit = self.max_reinit_per_event - total_reinit
            to_reinit_orig = to_reinit_orig[:remaining_reinit]

            # Neuroapoptosis: remove old dormant neurons (reverse order)
            # Track active-connection delta for budget accounting
            pre_active = ffn.get_active_connections()
            for idx in sorted(to_remove, reverse=True):
                if ffn.remove_neuron(idx, min_d_ff=self.min_d_ff):
                    total_removed += 1
            if to_remove:
                post_active = ffn.get_active_connections()
                self._adjust_structural_budget(post_active - pre_active)
                _materialize(
                    ffn.input_layer.weight, ffn.input_layer.bias,
                    ffn.output_layer.weight,
                    ffn._input_mask, ffn._output_mask, ffn._neuron_age,
                )
                logger.info(
                    f"Layer {layer_idx}: removed {len(to_remove)} dormant "
                    f"neurons (neuroapoptosis), d_ff={ffn.d_ff}"
                )

            # Remap reinit indices after removals
            to_reinit = self._remap_indices_after_removals(
                to_reinit_orig, to_remove,
            )

            # Reinitialize young dormant neurons (preserve masks)
            for idx in to_reinit:
                if idx < 0 or idx >= ffn.d_ff:
                    continue

                # Reinit weights only at active mask positions
                fan_in = ffn.d_model
                std = (2.0 / fan_in) ** 0.5

                # Input row: Kaiming init at active positions, zero elsewhere
                in_mask_row = ffn._input_mask[idx]  # (d_model,)
                new_input_row = mx.random.normal((ffn.d_model,)) * std
                new_input_row = new_input_row * in_mask_row
                new_bias = mx.array([0.0])

                iw = ffn.input_layer.weight
                new_iw = mx.concatenate([
                    iw[:idx], new_input_row.reshape(1, -1), iw[idx + 1:]
                ], axis=0)
                ffn.input_layer.weight = new_iw
                ib = ffn.input_layer.bias
                new_ib = mx.concatenate([ib[:idx], new_bias, ib[idx + 1:]])
                ffn.input_layer.bias = new_ib

                # Output column: near-zero at active positions, zero elsewhere
                out_mask_col = ffn._output_mask[:, idx]  # (d_model,)
                new_out_col = mx.random.normal((ffn.d_model,)) * 0.001
                new_out_col = new_out_col * out_mask_col
                ow = ffn.output_layer.weight
                new_ow = mx.concatenate([
                    ow[:, :idx], new_out_col.reshape(-1, 1), ow[:, idx + 1:]
                ], axis=1)
                ffn.output_layer.weight = new_ow

                # Masks are PRESERVED (no change → no sparsity change)

                # Reset neuron age
                age = ffn._neuron_age
                ffn._neuron_age = mx.concatenate([
                    age[:idx], mx.zeros((1,)), age[idx + 1:]
                ])

                total_reinit += 1

            if to_reinit:
                _materialize(
                    ffn.input_layer.weight, ffn.input_layer.bias,
                    ffn.output_layer.weight, ffn._neuron_age,
                )
                logger.info(
                    f"Layer {layer_idx}: reinit {len(to_reinit)} dormant "
                    f"neurons (masks preserved)"
                )

            # Clear stale EMA to prevent immediate re-trigger
            if to_remove or to_reinit:
                ffn._persistent_mean_activation = None

        self.stats["total_reinit"].append(total_reinit)
        self.stats["total_removed"].append(total_removed)
        return total_reinit, total_removed

    def _prune_attention_heads(self, model: MLXAdaptiveTransformer) -> int:
        """Prune attention heads with low importance scores.

        Change 3C: Structured head pruning. Heads with importance below
        head_prune_threshold are pruned (masked to zero). At most
        head_prune_max_per_layer heads are pruned per layer per step.

        Minimum 2 active heads per layer are always preserved.
        """
        total_pruned = 0

        for layer_idx, layer in enumerate(model.layers):
            importance = layer.get_head_importance_scores()
            if importance is None:
                continue

            active_heads = layer.get_active_heads()
            # Keep at least 2 heads per layer
            if active_heads <= 2:
                continue

            mx.eval(importance)  # noqa: S307
            pruned_this_layer = 0
            for head_idx in range(layer.nhead):
                if pruned_this_layer >= self.head_prune_max_per_layer:
                    break
                if active_heads - pruned_this_layer <= 2:
                    break
                # Only prune active heads that are below threshold
                mask_val = float(layer._head_mask[head_idx].item())
                if mask_val == 0.0:
                    continue
                score = float(importance[head_idx].item())
                if score < self.head_prune_threshold:
                    layer.prune_head(head_idx)
                    pruned_this_layer += 1
                    total_pruned += 1
                    logger.info(
                        f"Layer {layer_idx}: pruned head {head_idx} "
                        f"(importance={score:.4f})"
                    )

        self.stats["total_heads_pruned"].append(total_pruned)
        return total_pruned

    def _split_neurons(
        self,
        model: MLXAdaptiveTransformer,
        current_step: int = 0,
    ) -> Dict:
        """Split neurons based on convergence detection + effective rank.

        Li et al. (2025) "Dropin" Algorithm 3: splits trigger on convergence
        (loss plateau), not activation magnitude. When converged:
        1. Compute effective rank for all layers
        2. Sort by rank (ascending = most bottlenecked)
        3. Split max_splits_per_layer neurons in each of max_split_layers

        Neurogenesis (fresh neurons) is now handled separately by
        _scheduled_neurogenesis() on its own interval.

        Falls back to activation-based splits if convergence detection has
        insufficient history (e.g., early training).

        Returns dict with keys: count, branch, convergence_slope, convergence_range.
        """
        # Alias for MLX array materialization (NOT Python's eval builtin)
        _materialize = mx.eval  # noqa: S307

        total_splits = 0
        conv_slope, conv_range = self._convergence_stats()

        # Check convergence — if converged, use effective rank for layer selection
        if self._is_converged():
            from mlx_diagnostics import compute_effective_rank

            # Compute effective rank for each layer's FFN weight matrix
            layer_ranks: List[tuple] = []  # (layer_idx, effective_rank)
            ffn_modules = self._get_ffn_modules(model)
            for layer_idx, ffn in enumerate(ffn_modules):
                eff_rank = compute_effective_rank(ffn.input_layer.weight)
                layer_ranks.append((layer_idx, eff_rank))

            # Sort by effective rank ascending — lowest rank = biggest bottleneck
            layer_ranks.sort(key=lambda x: x[1])

            # Split max_splits_per_layer neurons in each of max_split_layers
            layers_processed = 0
            for layer_idx, eff_rank in layer_ranks:
                if layers_processed >= self.max_split_layers:
                    break

                ffn = ffn_modules[layer_idx]

                # Select candidates ONCE before any splits in this layer
                candidates = self._select_split_candidates(
                    ffn, self.max_splits_per_layer,
                )

                pre_active = ffn.get_active_connections()
                for split_idx in candidates:
                    ffn.split_neuron(split_idx, noise_scale=self.split_noise_scale)
                    total_splits += 1
                post_active = ffn.get_active_connections()
                self._adjust_structural_budget(post_active - pre_active)

                logger.info(
                    f"Convergence split: layer {layer_idx} "
                    f"{len(candidates)} splits on neurons {candidates} "
                    f"(eff_rank={eff_rank:.1f}), new d_ff={ffn.d_ff}"
                )
                layers_processed += 1

            if total_splits > 0:
                # Keep baseline val_loss for slope detection (don't clear all)
                self._val_loss_history = self._val_loss_history[-2:]

            branch = "converged_split"
        else:
            # Fallback: activation-based splits (original behavior)
            for layer_idx, ffn in enumerate(self._get_ffn_modules(model)):
                mean_act = ffn._persistent_mean_activation
                if mean_act is None:
                    stats = ffn.intermediate_activation_stats
                    if stats.recent_activations is None:
                        continue
                    activations = stats.recent_activations
                    if activations.ndim == 3:
                        activations = activations.reshape(-1, activations.shape[-1])
                    elif activations.ndim != 2:
                        continue
                    mean_act = mx.mean(mx.abs(activations), axis=0)

                _materialize(mean_act)

                overload_mask = mean_act[:ffn.d_ff] > self.split_threshold
                _materialize(overload_mask)
                overload_count = int(mx.sum(overload_mask).item())
                if overload_count == 0:
                    continue

                overload_indices = []
                for i in range(min(mean_act.shape[0], ffn.d_ff)):
                    if bool(overload_mask[i].item()):
                        overload_indices.append(i)
                        if len(overload_indices) >= 2:
                            break

                pre_active = ffn.get_active_connections()
                for idx in overload_indices:
                    ffn.split_neuron(idx, noise_scale=self.split_noise_scale)
                    total_splits += 1
                    logger.info(
                        f"Layer {layer_idx}: split neuron {idx}, "
                        f"new d_ff={ffn.d_ff}"
                    )
                post_active = ffn.get_active_connections()
                self._adjust_structural_budget(post_active - pre_active)

            branch = "fallback_split"

        self.stats["total_splits"].append(total_splits)
        return {
            "count": total_splits,
            "branch": branch,
            "convergence_slope": conv_slope,
            "convergence_range": conv_range,
        }

    def _merge_neurons(self, model: MLXAdaptiveTransformer) -> tuple:
        """Merge similar neurons based on weight cosine similarity.

        Fix 3: Neurons younger than min_age_for_merge are protected from
        merging. This prevents newly-split neurons (which start with sim~1.0
        to their parent) from being immediately re-merged.

        Returns (total_merges, merges_skipped_age).
        """
        total_merges = 0
        merges_skipped_age = 0

        for layer_idx, ffn in enumerate(self._get_ffn_modules(model)):
            if total_merges >= self.max_merges_per_event:
                break
            if ffn.d_ff < 4:
                continue

            weights = np.array(ffn.input_layer.weight.tolist())
            norms = np.linalg.norm(weights, axis=1)

            if np.any(norms < 1e-10):
                continue

            normalized = weights / norms[:, np.newaxis]
            similarity = normalized @ normalized.T

            candidates = []
            for i in range(similarity.shape[0]):
                for j in range(i + 1, similarity.shape[1]):
                    if similarity[i, j] > self.merge_similarity_threshold:
                        candidates.append((i, j, similarity[i, j]))

            candidates.sort(key=lambda x: x[2], reverse=True)
            merged_set: set = set()

            for i, j, sim in candidates:
                if total_merges >= self.max_merges_per_event:
                    break
                if i not in merged_set and j not in merged_set:
                    # Fix 3: Merge cooldown — protect young neurons
                    age_i = float(ffn._neuron_age[i].item())
                    age_j = float(ffn._neuron_age[j].item())
                    if min(age_i, age_j) < self.min_age_for_merge:
                        merges_skipped_age += 1
                        logger.debug(
                            f"Merge skip L{layer_idx} neurons {i},{j}: "
                            f"min_age={min(age_i, age_j):.0f} < {self.min_age_for_merge}"
                        )
                        continue

                    pre_active = ffn.get_active_connections()
                    ffn.merge_neurons([i, j])
                    post_active = ffn.get_active_connections()
                    self._adjust_structural_budget(post_active - pre_active)
                    merged_set.update([i, j])
                    total_merges += 1
                    logger.info(
                        f"Layer {layer_idx}: merged neurons {i},{j} "
                        f"(sim={sim:.3f}), new d_ff={ffn.d_ff}"
                    )
                    break

        self.stats["total_merges"].append(total_merges)
        return total_merges, merges_skipped_age

    def _clip_weight_norms(
        self, model: MLXAdaptiveTransformer, max_norm: float = 30.0,
    ) -> int:
        """Clip per-layer FFN weight matrix norms to prevent divergence.

        Layer 7 showed 2.3x higher norm than other layers in the 50k run,
        caused by reset+regrowth oscillation. Clipping prevents this.
        """
        # Alias for MLX array materialization (NOT Python's eval builtin)
        _materialize = mx.eval  # noqa: S307

        clipped = 0
        for layer_idx, ffn in enumerate(self._get_ffn_modules(model)):
            for layer_obj in [ffn.input_layer, ffn.output_layer]:
                w = layer_obj.weight
                norm = float(mx.sqrt(mx.sum(w * w)).item())
                if norm > max_norm:
                    layer_obj.weight = w * (max_norm / norm)
                    _materialize(layer_obj.weight)
                    clipped += 1
        return clipped

    def capture_snapshot(
        self,
        model: MLXAdaptiveTransformer,
        step: int,
        train_loss: float,
        val_loss: float,
    ) -> Dict:
        """Capture architecture snapshot compatible with metrics.jsonl format."""
        d_ff_values = []
        layer_details = []
        total_active = 0
        total_mask_params = 0

        for i, layer in enumerate(model.layers):
            ffn = layer.ffn
            d_ff_values.append(ffn.d_ff)

            in_mask = ffn._input_mask
            out_mask = ffn._output_mask
            in_active = int(mx.sum(in_mask != 0).item())
            out_active = int(mx.sum(out_mask != 0).item())
            in_total = in_mask.size
            out_total = out_mask.size

            in_sparsity = 1.0 - (in_active / in_total) if in_total > 0 else 0.0
            out_sparsity = 1.0 - (out_active / out_total) if out_total > 0 else 0.0

            total_active += in_active + out_active
            total_mask_params += in_total + out_total

            layer_details.append({
                "name": f"layer_{i}.ffn",
                "d_ff": ffn.d_ff,
                "input_mask_sparsity": round(in_sparsity, 4),
                "output_mask_sparsity": round(out_sparsity, 4),
                "active_heads": layer.get_active_heads(),
                "total_heads": layer.nhead,
            })

        global_sparsity = (
            1.0 - (total_active / total_mask_params)
            if total_mask_params > 0
            else 0.0
        )
        total_params = model.count_parameters()

        try:
            perplexity = math.exp(val_loss) if val_loss < 20 else 9999.99
        except OverflowError:
            perplexity = 9999.99

        cum_splits = sum(self.stats.get("total_splits", [0]))
        cum_merges = sum(self.stats.get("total_merges", [0]))
        cum_grown = sum(self.stats.get("total_grown", [0]))
        cum_pruned = sum(self.stats.get("total_pruned", [0]))
        cum_reinit = sum(self.stats.get("total_reinit", [0]))
        cum_removed = sum(self.stats.get("total_removed", [0]))
        cum_heads = sum(self.stats.get("total_heads_pruned", [0]))
        cum_neurogenesis = sum(self.stats.get("total_neurogenesis", [0]))

        return {
            "step": step,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "perplexity": round(perplexity, 2),
            "total_model_params": total_params,
            "masked_active_params": total_active,
            "masked_total_params": total_mask_params,
            "global_sparsity": round(global_sparsity, 6),
            "target_sparsity": self.target_sparsity,
            "d_ff_per_layer": d_ff_values,
            "cumulative_splits": cum_splits,
            "cumulative_merges": cum_merges,
            "cumulative_grown": cum_grown,
            "cumulative_pruned": cum_pruned,
            "cumulative_reinit": cum_reinit,
            "cumulative_removed": cum_removed,
            "cumulative_heads_pruned": cum_heads,
            "cumulative_neurogenesis": cum_neurogenesis,
            "neurogenesis_per_layer": dict(self._neurogenesis_per_layer),
            "structural_mode": self._last_structural_mode,
            "structural_backoff_active": step < self._backoff_until_step,
            "prune_budget": self._prune_budget,
            "structural_budget_debt": self._structural_budget_debt,
            "layer_details": layer_details,
        }


def _extract_ffn_grad(
    dense_grads: Dict, layer_idx: int, weight_name: str
) -> Optional[mx.array]:
    """Extract FFN gradient for a specific layer from the gradient tree.

    The gradient tree mirrors the model parameter tree. We need to navigate:
    model.layers[layer_idx].ffn.{input_layer,output_layer}.weight
    """
    try:
        layers = dense_grads.get("layers", dense_grads)
        if isinstance(layers, list):
            layer_grads = layers[layer_idx]
        elif isinstance(layers, dict):
            layer_grads = layers.get(str(layer_idx), layers.get(layer_idx))
        else:
            return None

        ffn_grads = layer_grads.get("ffn", layer_grads)
        target = "input_layer" if weight_name == "input" else "output_layer"
        layer_g = ffn_grads.get(target, {})
        grad = layer_g.get("weight")
        if isinstance(grad, mx.array):
            return grad
    except (AttributeError, KeyError, IndexError, TypeError):
        pass
    return None


if __name__ == "__main__":
    import sys

    all_validation_failures: List[str] = []
    total_tests = 0

    # Create a small model for testing
    model = MLXAdaptiveTransformer(
        vocab_size=500, d_model=32, nhead=4, num_layers=2,
        d_ff=128, max_seq_len=16,
    )

    # Run a forward pass in training mode to populate activation stats
    model.train()
    tokens = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
    out = model(tokens)
    mx.eval(out)  # noqa: S307

    controller = MLXPlasticityController(
        prune_threshold=0.5,
        growth_rate=0.5,
        split_threshold=0.001,
        merge_similarity_threshold=0.99,
        prune_frequency=1,
        structural_frequency=1,
        min_connections=10,
        target_sparsity=0.4,
        dormant_threshold=0.001,
        warmup_steps=0,  # No warmup for tests
        head_prune_threshold=0.1,
    )

    # Test 1: Prune reduces active connections
    total_tests += 1
    ffn0 = model.layers[0].ffn
    before_active = ffn0.get_active_connections()
    pruned = controller._prune_connections(model)
    after_active = ffn0.get_active_connections()
    if pruned > 0 and after_active >= before_active:
        all_validation_failures.append(
            f"Prune: active before={before_active}, after={after_active}, "
            f"pruned={pruned}"
        )
    else:
        print(
            f"  Test 1 OK: pruned {pruned} connections "
            f"({before_active} -> {after_active})"
        )

    # Test 2: Budget-coupled grow uses prune budget
    total_tests += 1
    before_active = ffn0.get_active_connections()
    # With max_grow=0, nothing should grow
    grown_zero = controller._grow_connections(model, max_grow=0)
    if grown_zero != 0:
        all_validation_failures.append(
            f"Budget-coupled grow: expected 0 with max_grow=0, got {grown_zero}"
        )
    else:
        # Now give it some budget
        grown = controller._grow_connections(model, max_grow=pruned)
        print(f"  Test 2 OK: budget-coupled grow={grown} (budget={pruned})")

    # Test 3: Split increases d_ff (now returns dict)
    total_tests += 1
    old_dff = ffn0.d_ff
    split_result = controller._split_neurons(model)
    splits = split_result["count"]
    new_dff = ffn0.d_ff
    if splits > 0 and new_dff <= old_dff:
        all_validation_failures.append(
            f"Split: d_ff before={old_dff}, after={new_dff}, splits={splits}"
        )
    elif "branch" not in split_result:
        all_validation_failures.append(
            f"Split: result missing 'branch' key: {split_result}"
        )
    else:
        print(
            f"  Test 3 OK: {splits} splits (d_ff {old_dff} -> {new_dff}, "
            f"branch={split_result['branch']})"
        )

    # Test 4: Full step returns event dict with new keys (incl. neurogenesis)
    total_tests += 1
    events = controller.step(model, current_step=10000)
    required_keys = {"pruned", "grown", "splits", "merges", "reinit", "removed", "heads_pruned", "neurogenesis"}
    if not required_keys.issubset(events.keys()):
        all_validation_failures.append(
            f"Step events missing keys: {required_keys - events.keys()}"
        )
    else:
        print(f"  Test 4 OK: step events = {events}")

    # Test 5: Snapshot has correct format (including new fields)
    total_tests += 1
    snapshot = controller.capture_snapshot(
        model, step=10, train_loss=5.0, val_loss=5.5
    )
    required_snap_keys = {
        "step", "train_loss", "val_loss", "perplexity",
        "total_model_params", "global_sparsity", "d_ff_per_layer",
        "cumulative_neurogenesis",
        "cumulative_splits", "cumulative_merges",
        "cumulative_grown", "cumulative_pruned",
        "cumulative_reinit", "cumulative_removed",
        "cumulative_heads_pruned",
        "target_sparsity", "layer_details",
    }
    missing = required_snap_keys - snapshot.keys()
    if missing:
        all_validation_failures.append(f"Snapshot missing keys: {missing}")
    elif snapshot["step"] != 10:
        all_validation_failures.append(
            f"Snapshot step: expected 10, got {snapshot['step']}"
        )
    elif snapshot["target_sparsity"] != 0.4:
        all_validation_failures.append(
            f"Snapshot target_sparsity: expected 0.4, got {snapshot['target_sparsity']}"
        )
    else:
        print(f"  Test 5 OK: snapshot keys present, step={snapshot['step']}")

    # Test 6: Model still produces valid output after all plasticity
    total_tests += 1
    post_logits = model(tokens)
    mx.eval(post_logits)  # noqa: S307
    if post_logits.shape[0] != 1 or post_logits.shape[1] != 4:
        all_validation_failures.append(
            f"Post-plasticity shape: expected (1, 4, ?), got {post_logits.shape}"
        )
    else:
        print(
            f"  Test 6 OK: post-plasticity forward shape = {post_logits.shape}"
        )

    # Test 7: Stats accumulate correctly
    total_tests += 1
    cum_pruned = sum(controller.stats.get("total_pruned", [0]))
    if cum_pruned < 0:
        all_validation_failures.append(f"Cumulative pruned: {cum_pruned} < 0")
    else:
        print(f"  Test 7 OK: cumulative pruned = {cum_pruned}")

    # Test 8: Warmup prevents pruning
    total_tests += 1
    warmup_ctrl = MLXPlasticityController(
        prune_frequency=1, structural_frequency=100,
        warmup_steps=5000, target_sparsity=0.4,
    )
    warmup_model = MLXAdaptiveTransformer(
        vocab_size=500, d_model=32, nhead=4, num_layers=2,
        d_ff=128, max_seq_len=16,
    )
    warmup_model.train()
    _ = warmup_model(tokens)
    mx.eval(_)  # noqa: S307
    events_warmup = warmup_ctrl.step(warmup_model, current_step=100)
    if events_warmup["pruned"] != 0:
        all_validation_failures.append(
            f"Warmup: expected 0 pruned during warmup, got {events_warmup['pruned']}"
        )
    else:
        print(f"  Test 8 OK: warmup prevented pruning (events={events_warmup})")

    # Test 9: Dormant neuron handling (reinit young, remove old)
    total_tests += 1
    reinit_model = MLXAdaptiveTransformer(
        vocab_size=500, d_model=32, nhead=4, num_layers=2,
        d_ff=128, max_seq_len=16,
    )
    reinit_model.train()
    _ = reinit_model(tokens)
    mx.eval(_)  # noqa: S307
    # Force some neurons to be "dormant" by setting their activation EMA to 0
    reinit_model.layers[0].ffn._persistent_mean_activation = mx.concatenate([
        mx.zeros((4,)),  # 4 dormant neurons
        mx.ones((124,)),  # rest are active
    ])
    mx.eval(reinit_model.layers[0].ffn._persistent_mean_activation)  # noqa: S307
    reinit_ctrl = MLXPlasticityController(
        dormant_threshold=0.001, structural_frequency=1,
        prune_frequency=100, warmup_steps=0,
        min_age_for_removal=5000,  # All neurons are young (age=0), so reinit
        max_reinit_per_event=8,  # Raise cap so all 4 dormant neurons reinit
    )
    reinit_count, removed_count = reinit_ctrl._handle_dormant_neurons(
        reinit_model, current_step=100,
    )
    if reinit_count != 4:
        all_validation_failures.append(
            f"Reinit: expected 4 dormant neurons reinit, got {reinit_count}"
        )
    elif removed_count != 0:
        all_validation_failures.append(
            f"Removed: expected 0 (all young), got {removed_count}"
        )
    else:
        print(f"  Test 9 OK: reinit {reinit_count}, removed {removed_count}")

    # Test 10: Snapshot includes head info
    total_tests += 1
    layer_detail = snapshot["layer_details"][0]
    if "active_heads" not in layer_detail:
        all_validation_failures.append("Snapshot layer_details missing active_heads")
    elif "total_heads" not in layer_detail:
        all_validation_failures.append("Snapshot layer_details missing total_heads")
    else:
        print(
            f"  Test 10 OK: layer_details has head info "
            f"({layer_detail['active_heads']}/{layer_detail['total_heads']})"
        )

    # Test 11: _extract_ffn_grad handles various input formats
    total_tests += 1
    fake_grads = {
        "layers": [
            {"ffn": {"input_layer": {"weight": mx.ones((128, 32))},
                     "output_layer": {"weight": mx.ones((32, 128))}}},
            {"ffn": {"input_layer": {"weight": mx.ones((128, 32))},
                     "output_layer": {"weight": mx.ones((32, 128))}}},
        ]
    }
    g = _extract_ffn_grad(fake_grads, 0, "input")
    if g is None:
        all_validation_failures.append("_extract_ffn_grad returned None for valid input")
    elif g.shape != (128, 32):
        all_validation_failures.append(
            f"_extract_ffn_grad shape: expected (128, 32), got {g.shape}"
        )
    else:
        print(f"  Test 11 OK: _extract_ffn_grad shape = {g.shape}")

    # MLX materialization alias (NOT Python's eval builtin)
    _materialize = mx.eval  # noqa: S307

    # Test 12: Slope-based convergence detection
    total_tests += 1
    conv_ctrl = MLXPlasticityController(
        convergence_delta=0.05, convergence_window=4,
        prune_frequency=100, structural_frequency=100,
    )
    # Not converged: not enough history
    if conv_ctrl._is_converged():
        all_validation_failures.append("Convergence: detected with no history")
    else:
        # Feed a plateau: slope ~0, range small
        conv_ctrl._val_loss_history = [5.0, 5.01, 4.98, 5.02]
        if not conv_ctrl._is_converged():
            all_validation_failures.append(
                "Convergence: should detect plateau [5.0, 5.01, 4.98, 5.02]"
            )
        else:
            # Feed a non-plateau: clearly decreasing loss
            conv_ctrl._val_loss_history = [5.0, 4.5, 4.0, 3.5]
            if conv_ctrl._is_converged():
                all_validation_failures.append(
                    "Convergence: falsely detected for decreasing loss"
                )
            else:
                print("  Test 12 OK: slope-based convergence detection correct")

    # Test 13: Convergence-based splits (neurogenesis decoupled)
    total_tests += 1
    conv_split_model = MLXAdaptiveTransformer(
        vocab_size=500, d_model=32, nhead=4, num_layers=2,
        d_ff=128, max_seq_len=16,
    )
    conv_split_model.train()
    _ = conv_split_model(tokens)
    _materialize(_)
    conv_split_ctrl = MLXPlasticityController(
        convergence_delta=0.1, convergence_window=4,
        max_splits_per_layer=2,
        max_split_layers=2,
        neurogenesis_count=2,
        prune_frequency=100, structural_frequency=1,
        warmup_steps=0,
    )
    # Simulate convergence (slope ~0, range small)
    conv_split_ctrl._val_loss_history = [5.0, 5.01, 4.99, 5.02]
    old_dff_0 = conv_split_model.layers[0].ffn.d_ff
    old_dff_1 = conv_split_model.layers[1].ffn.d_ff
    split_result = conv_split_ctrl._split_neurons(conv_split_model)
    split_count = split_result["count"]
    new_dff_0 = conv_split_model.layers[0].ffn.d_ff
    new_dff_1 = conv_split_model.layers[1].ffn.d_ff
    total_dff_change = (new_dff_0 - old_dff_0) + (new_dff_1 - old_dff_1)
    # Expected: 2 splits per layer * 2 layers = 4 (no neurogenesis in splits now)
    if split_count == 0:
        all_validation_failures.append("Convergence splits: no splits triggered")
    elif total_dff_change != split_count:
        all_validation_failures.append(
            f"Convergence splits: expected d_ff change of {split_count}, "
            f"got {total_dff_change}"
        )
    elif split_result["branch"] != "converged_split":
        all_validation_failures.append(
            f"Convergence splits: expected branch=converged_split, "
            f"got {split_result['branch']}"
        )
    else:
        print(
            f"  Test 13 OK: convergence splits = {split_count} "
            f"(d_ff {old_dff_0},{old_dff_1} -> {new_dff_0},{new_dff_1})"
        )

    # Test 14: step() accepts val_loss and lrp_batch
    total_tests += 1
    step_ctrl = MLXPlasticityController(
        prune_frequency=1, structural_frequency=1,
        warmup_steps=0, target_sparsity=0.4,
    )
    step_model = MLXAdaptiveTransformer(
        vocab_size=500, d_model=32, nhead=4, num_layers=2,
        d_ff=128, max_seq_len=16,
    )
    step_model.train()
    _ = step_model(tokens)
    _materialize(_)
    lrp_batch = mx.random.normal((1, 4, 32))
    _materialize(lrp_batch)
    events_with_vl = step_ctrl.step(
        step_model, current_step=10000,
        val_loss=5.5, lrp_batch=lrp_batch,
    )
    if not isinstance(events_with_vl, dict):
        all_validation_failures.append(
            f"step() with val_loss: expected dict, got {type(events_with_vl)}"
        )
    elif len(step_ctrl._val_loss_history) != 1:
        all_validation_failures.append(
            f"step() val_loss tracking: expected 1 entry, "
            f"got {len(step_ctrl._val_loss_history)}"
        )
    else:
        print(f"  Test 14 OK: step() with val_loss/lrp_batch = {events_with_vl}")

    # Test 15: LRP pruning mode works when batch available
    total_tests += 1
    lrp_ctrl = MLXPlasticityController(
        prune_frequency=1, structural_frequency=100,
        warmup_steps=0, target_sparsity=0.4,
        lrp_pruning=True,
    )
    lrp_model = MLXAdaptiveTransformer(
        vocab_size=500, d_model=32, nhead=4, num_layers=2,
        d_ff=128, max_seq_len=16,
    )
    lrp_model.train()
    _ = lrp_model(tokens)
    _materialize(_)
    lrp_ctrl._lrp_batch = mx.random.normal((1, 4, 32))
    _materialize(lrp_ctrl._lrp_batch)
    pruned_lrp = lrp_ctrl._prune_connections(lrp_model)
    # Should work without errors (LRP or magnitude fallback)
    if pruned_lrp < 0:
        all_validation_failures.append(f"LRP pruning: negative count {pruned_lrp}")
    else:
        print(f"  Test 15 OK: LRP pruning ran, pruned {pruned_lrp}")

    # Test 16: Merge cooldown — newly-split neurons skip merge
    total_tests += 1
    merge_model = MLXAdaptiveTransformer(
        vocab_size=500, d_model=32, nhead=4, num_layers=2,
        d_ff=128, max_seq_len=16,
    )
    merge_model.train()
    _ = merge_model(tokens)
    _materialize(_)
    # Split a neuron (creates young neuron with age=0)
    merge_model.layers[0].ffn.split_neuron(0, noise_scale=0.01)
    _materialize(
        merge_model.layers[0].ffn.input_layer.weight,
        merge_model.layers[0].ffn.output_layer.weight,
        merge_model.layers[0].ffn._input_mask,
        merge_model.layers[0].ffn._output_mask,
        merge_model.layers[0].ffn._neuron_age,
    )
    merge_ctrl = MLXPlasticityController(
        merge_similarity_threshold=0.5,  # Very permissive
        min_age_for_merge=3000,
    )
    merges, skipped = merge_ctrl._merge_neurons(merge_model)
    if skipped == 0:
        all_validation_failures.append(
            f"Merge cooldown: expected skipped > 0 for young neurons, "
            f"got skipped={skipped}, merges={merges}"
        )
    else:
        print(f"  Test 16 OK: merge cooldown skipped={skipped}, merges={merges}")

    # Test 17: Merge proceeds when neurons are old enough
    total_tests += 1
    # Artificially age ALL neurons in ALL layers past threshold
    for layer in merge_model.layers:
        ffn_m = layer.ffn
        ffn_m._neuron_age = mx.ones((ffn_m.d_ff,)) * 5000.0
        _materialize(ffn_m._neuron_age)
    merge_ctrl2 = MLXPlasticityController(
        merge_similarity_threshold=0.5,
        min_age_for_merge=3000,
    )
    merges2, skipped2 = merge_ctrl2._merge_neurons(merge_model)
    # Old neurons should be eligible for merge (skipped_age = 0)
    if skipped2 > 0:
        all_validation_failures.append(
            f"Merge old neurons: expected skipped=0, got {skipped2}"
        )
    else:
        print(f"  Test 17 OK: old neurons merge ok (merges={merges2}, skipped={skipped2})")

    # Test 18: Structural freeze blocks structural edits
    total_tests += 1
    freeze_model = MLXAdaptiveTransformer(
        vocab_size=500, d_model=32, nhead=4, num_layers=2,
        d_ff=128, max_seq_len=16,
    )
    freeze_model.train()
    _ = freeze_model(tokens)
    _materialize(_)
    freeze_ctrl = MLXPlasticityController(
        prune_frequency=1, structural_frequency=1,
        warmup_steps=0, total_steps=100,
        structural_freeze_fraction=0.3,  # Freeze after step 70
        split_threshold=0.001,
    )
    # Step at 90 (past freeze threshold) — should produce no structural edits
    events_frozen = freeze_ctrl.step(freeze_model, current_step=90, current_lr=1e-6)
    if events_frozen["splits"] > 0 or events_frozen["merges"] > 0:
        all_validation_failures.append(
            f"Structural freeze: expected 0 splits/merges at step 90/100, "
            f"got splits={events_frozen['splits']}, merges={events_frozen['merges']}"
        )
    else:
        print(f"  Test 18 OK: structural freeze blocks edits at step 90/100")

    # Test 19: Distributed neurogenesis across 2 layers
    total_tests += 1
    ng_model = MLXAdaptiveTransformer(
        vocab_size=500, d_model=32, nhead=4, num_layers=4,
        d_ff=128, max_seq_len=16,
    )
    ng_model.train()
    _ = ng_model(tokens)
    _materialize(_)
    ng_ctrl = MLXPlasticityController(
        neurogenesis_count=2, warmup_steps=0, total_steps=1000,
    )
    ng_count = ng_ctrl._scheduled_neurogenesis(ng_model, current_step=10)
    if ng_count != 2:
        all_validation_failures.append(f"Distributed neurogenesis: expected 2, got {ng_count}")
    else:
        # Check that 2 different layers got neurons
        layers_hit = [idx for idx, count in ng_ctrl._neurogenesis_per_layer.items() if count > 0]
        if len(layers_hit) != 2:
            all_validation_failures.append(
                f"Distributed neurogenesis: expected 2 layers, got {len(layers_hit)}: {dict(ng_ctrl._neurogenesis_per_layer)}"
            )
        else:
            print(f"  Test 19 OK: neurogenesis distributed to layers {layers_hit}")

    # Test 20: Neurogenesis never co-occurs with splits/merges/maintenance
    total_tests += 1
    excl_model = MLXAdaptiveTransformer(
        vocab_size=500, d_model=32, nhead=4, num_layers=2,
        d_ff=128, max_seq_len=16,
    )
    excl_model.train()
    _ = excl_model(tokens)
    _materialize(_)
    excl_ctrl = MLXPlasticityController(
        neurogenesis_interval=10, neurogenesis_count=2,
        prune_frequency=100, structural_frequency=1,
        warmup_steps=0, total_steps=1000,
    )
    # Step at neurogenesis interval — neurogenesis should pre-empt split/merge/maintenance
    events_excl = excl_ctrl.step(excl_model, current_step=10, val_loss=5.0)
    has_ng = events_excl["neurogenesis"] > 0
    has_split = events_excl["splits"] > 0
    has_merge = events_excl["merges"] > 0
    has_maint = events_excl["reinit"] > 0 or events_excl["removed"] > 0
    if has_ng and (has_split or has_merge or has_maint):
        all_validation_failures.append(
            f"Mode exclusivity: neurogenesis co-occurred with other structural: {events_excl}"
        )
    elif not has_ng:
        all_validation_failures.append(
            f"Mode exclusivity: neurogenesis didn't fire at step 10: {events_excl}"
        )
    else:
        print(f"  Test 20 OK: neurogenesis exclusive, no co-occurrence")

    # Test 21: Split event never exceeds max_splits_per_layer * max_split_layers = 2 total
    total_tests += 1
    cap_model = MLXAdaptiveTransformer(
        vocab_size=500, d_model=32, nhead=4, num_layers=4,
        d_ff=128, max_seq_len=16,
    )
    cap_model.train()
    _ = cap_model(tokens)
    _materialize(_)
    cap_ctrl = MLXPlasticityController(
        convergence_delta=0.1, convergence_window=4,
        max_splits_per_layer=1, max_split_layers=2,
        prune_frequency=100, structural_frequency=1, warmup_steps=0,
    )
    cap_ctrl._val_loss_history = [5.0, 5.01, 4.99, 5.02]
    sr = cap_ctrl._split_neurons(cap_model)
    if sr["count"] > 2:
        all_validation_failures.append(f"Split cap: {sr['count']} > 2 max")
    else:
        print(f"  Test 21 OK: split capped at {sr['count']} <= 2")

    # Test 22: Merge event never exceeds max_merges_per_event = 2
    total_tests += 1
    mcap_model = MLXAdaptiveTransformer(
        vocab_size=500, d_model=32, nhead=4, num_layers=4,
        d_ff=128, max_seq_len=16,
    )
    mcap_model.train()
    _ = mcap_model(tokens)
    _materialize(_)
    # Age all neurons so merge cooldown doesn't interfere
    for layer in mcap_model.layers:
        layer.ffn._neuron_age = mx.ones((layer.ffn.d_ff,)) * 10000.0
        _materialize(layer.ffn._neuron_age)
    mcap_ctrl = MLXPlasticityController(
        merge_similarity_threshold=0.3,
        min_age_for_merge=100,
        max_merges_per_event=2,
    )
    mcount, _ = mcap_ctrl._merge_neurons(mcap_model)
    if mcount > 2:
        all_validation_failures.append(f"Merge cap: {mcount} > 2 max")
    else:
        print(f"  Test 22 OK: merges capped at {mcount} <= 2")

    # Test 23: Maintenance never exceeds max_removed=1 and max_reinit=1
    total_tests += 1
    maint_model = MLXAdaptiveTransformer(
        vocab_size=500, d_model=32, nhead=4, num_layers=2,
        d_ff=128, max_seq_len=16,
    )
    maint_model.train()
    _ = maint_model(tokens)
    _materialize(_)
    # Force many dormant neurons
    for layer in maint_model.layers:
        layer.ffn._persistent_mean_activation = mx.concatenate([
            mx.zeros((8,)), mx.ones((120,)),
        ])
        _materialize(layer.ffn._persistent_mean_activation)
    maint_ctrl = MLXPlasticityController(
        dormant_threshold=0.001, warmup_steps=0,
        min_age_for_removal=5000,
        max_removed_per_event=1, max_reinit_per_event=1,
    )
    reinit_c, removed_c = maint_ctrl._handle_dormant_neurons(maint_model, current_step=100)
    if reinit_c > 1:
        all_validation_failures.append(f"Maintenance reinit cap: {reinit_c} > 1")
    elif removed_c > 1:
        all_validation_failures.append(f"Maintenance remove cap: {removed_c} > 1")
    else:
        print(f"  Test 23 OK: maintenance capped (reinit={reinit_c}, removed={removed_c})")

    # Test 24: Neurogenesis trend gate skips when slope <= -0.05
    total_tests += 1
    tg_ctrl = MLXPlasticityController(
        neurogenesis_skip_slope_window=4,
        neurogenesis_skip_slope_threshold=-0.05,
    )
    # Strongly improving: slope ~ -0.5
    tg_ctrl._val_loss_history = [6.0, 5.5, 5.0, 4.5]
    if not tg_ctrl._neurogenesis_trend_gate_skip():
        all_validation_failures.append("Trend gate: should skip for strongly improving loss")
    else:
        # Plateau: slope ~ 0
        tg_ctrl._val_loss_history = [5.0, 5.01, 4.99, 5.02]
        if tg_ctrl._neurogenesis_trend_gate_skip():
            all_validation_failures.append("Trend gate: should NOT skip for plateau")
        else:
            print("  Test 24 OK: trend gate correct")

    # Test 25: Structural backoff activates after 3 unrecovered evals
    total_tests += 1
    bo_ctrl = MLXPlasticityController(
        structural_backoff_recovery_evals=3,
        structural_backoff_recovery_tolerance=0.05,
        structural_backoff_steps=3000,
        prune_frequency=100, structural_frequency=100,
        warmup_steps=0,
    )
    # Simulate a structural event at step 1000 with pre_loss=5.0
    bo_ctrl._pending_recoveries.append(_PendingRecovery(
        event_step=1000, pre_event_val_loss=5.0,
        structural_mode="split", events={},
    ))
    # 3 evals that don't recover (all above 5.0 + 0.05 = 5.05)
    bo_ctrl._update_pending_recoveries(5.2, current_step=1500)
    bo_ctrl._update_pending_recoveries(5.3, current_step=2000)
    bo_ctrl._update_pending_recoveries(5.4, current_step=2500)
    if bo_ctrl._backoff_until_step == 0:
        all_validation_failures.append("Backoff: should be active after 3 unrecovered evals")
    elif bo_ctrl._backoff_until_step != 2500 + 3000:
        all_validation_failures.append(
            f"Backoff: expected until step {2500 + 3000}, got {bo_ctrl._backoff_until_step}"
        )
    else:
        print(f"  Test 25 OK: backoff until step {bo_ctrl._backoff_until_step}")

    # Test 26: Snapshot includes structural_mode and neurogenesis_per_layer
    total_tests += 1
    snap_ctrl = MLXPlasticityController(warmup_steps=0, total_steps=1000)
    snap_model = MLXAdaptiveTransformer(
        vocab_size=500, d_model=32, nhead=4, num_layers=2,
        d_ff=128, max_seq_len=16,
    )
    snap = snap_ctrl.capture_snapshot(snap_model, step=100, train_loss=5.0, val_loss=5.5)
    for key in ["structural_mode", "neurogenesis_per_layer", "structural_backoff_active"]:
        if key not in snap:
            all_validation_failures.append(f"Snapshot missing '{key}'")
            break
    else:
        print(f"  Test 26 OK: snapshot has structural_mode, neurogenesis_per_layer, backoff")

    # Test 27: Budget hold activates when debt rises and prune/grow stall
    total_tests += 1
    bh_ctrl = MLXPlasticityController(
        structural_only_hold_window=3,
        prune_frequency=100, structural_frequency=100,
        warmup_steps=0,
    )
    # Simulate 3 checkpoints: grown/pruned flat, debt rising
    bh_ctrl._budget_hold_history = [(0, 0, 100), (0, 0, 200)]
    bh_ctrl._structural_budget_debt = 300
    bh_ctrl.stats["total_grown"] = [0]
    bh_ctrl.stats["total_pruned"] = [0]
    bh_ctrl._update_budget_hold()
    if not bh_ctrl._in_budget_hold:
        all_validation_failures.append("Budget hold: should activate with flat prune/grow + rising debt")
    else:
        # Release when grown changes
        bh_ctrl._budget_hold_history.append((0, 0, 300))  # still stalled but need new update
        bh_ctrl.stats["total_grown"] = [100]  # grown changed
        bh_ctrl._update_budget_hold()
        if bh_ctrl._in_budget_hold:
            all_validation_failures.append("Budget hold: should release when grown changes")
        else:
            print("  Test 27 OK: budget hold activates and releases correctly")

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
