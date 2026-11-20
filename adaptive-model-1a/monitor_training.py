"""
Training health monitor for the Neuroplasticity Engine.

Reads metrics.jsonl snapshots and checks eleven invariants against
the live training run. Designed to be run periodically (e.g., every
5 minutes via cron or launchd) and prints a concise health report.

Docs: https://docs.python.org/3/library/json.html

Sample input:
    python monitor_training.py                    # Auto-find latest run
    python monitor_training.py --run-dir runs/20260316_151435
    python monitor_training.py --watch 300        # Re-check every 5 min

Expected output:
    [step 6500] HEALTH REPORT — 7 OK, 2 WARN, 0 FAIL
    ...per-check status lines...
"""

import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from glob import glob
from typing import Dict, List, Optional, Tuple

from loguru import logger

# ---------------------------------------------------------------------------
# Configuration: thresholds for each health check
# ---------------------------------------------------------------------------

@dataclass
class MonitorConfig:
    """Thresholds for training health checks."""
    # Check 1: sparsity convergence
    sparsity_check_step: int = 6000
    min_sparsity_at_check: float = 0.10  # at least 25% of target (0.4)

    # Check 2: budget debt
    budget_check_step: int = 6000

    # Check 3: d_ff oscillation detection
    dff_oscillation_window: int = 6  # consecutive snapshots to check
    dff_churn_ratio: float = 0.3  # oscillating if |net_change| / range < this

    # Check 4: split-rank alignment
    rank_mismatch_tolerance: int = 2  # allow top-N+tolerance

    # Check 5: dead unit growth
    dead_unit_growth_threshold: int = 5  # warn if dead units rise by > this

    # Check 6: structural churn ratio
    merge_split_churn_ratio: float = 0.75  # warn if merges/splits > this

    # Check 7: val_loss degradation after structural events
    val_loss_degradation_evals: int = 3  # check this many evals after event
    val_loss_max_sustained_rise: float = 0.15  # warn if loss rises > this

    # Check 8: last layer reset (step 10000)
    reset_step: int = 10000
    reset_recovery_evals: int = 3
    reset_max_degradation: float = 0.20

    # Check 9: event cadence
    warmup_steps: int = 5000


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def find_latest_run(base: str = "runs") -> Optional[str]:
    """Find the most recent run directory containing metrics.jsonl."""
    candidates = sorted(glob(os.path.join(base, "20*")), reverse=True)
    for d in candidates:
        if os.path.isfile(os.path.join(d, "metrics.jsonl")):
            return d
    return None


def load_snapshots(metrics_path: str) -> List[Dict]:
    """Load all JSON snapshots from metrics.jsonl."""
    snapshots = []
    with open(metrics_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                snapshots.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed line {line_num}: {e}")
    return snapshots


# ---------------------------------------------------------------------------
# Individual health checks
# ---------------------------------------------------------------------------

# Status codes
OK = "OK"
WARN = "WARN"
FAIL = "FAIL"
SKIP = "SKIP"


def check_sparsity_convergence(
    snapshots: List[Dict], cfg: MonitorConfig,
) -> Tuple[str, str]:
    """Check 1: Is global_sparsity moving toward target by step N?"""
    target = snapshots[-1].get("target_sparsity", 0.4) if snapshots else 0.4
    latest_step = snapshots[-1]["step"] if snapshots else 0

    if latest_step < cfg.sparsity_check_step:
        return SKIP, (
            f"Only at step {latest_step}/{cfg.sparsity_check_step}, "
            f"current sparsity={snapshots[-1].get('global_sparsity', 0):.4f}"
        )

    # Use the latest snapshot for current reading
    snap = snapshots[-1]
    sparsity = snap.get("global_sparsity", 0)
    ratio = sparsity / target if target > 0 else 0

    if ratio >= 0.50:
        return OK, (
            f"step {snap['step']}: sparsity={sparsity:.4f} "
            f"({ratio:.0%} of target {target})"
        )
    elif ratio >= 0.25:
        return WARN, (
            f"step {snap['step']}: sparsity={sparsity:.4f} — only "
            f"{ratio:.0%} of target {target}, prune may be too permissive"
        )
    else:
        return FAIL, (
            f"step {snap['step']}: sparsity={sparsity:.4f} — only "
            f"{ratio:.0%} of target {target}, pruning appears ineffective"
        )


def check_budget_debt(
    snapshots: List[Dict], cfg: MonitorConfig,
) -> Tuple[str, str]:
    """Check 2: Is structural budget debt being tracked and paid down?"""
    latest_step = snapshots[-1]["step"] if snapshots else 0

    if latest_step < cfg.budget_check_step:
        snap = snapshots[-1] if snapshots else {}
        budget = snap.get("prune_budget", "?")
        debt = snap.get("structural_budget_debt", "?")
        return SKIP, (
            f"Step {latest_step}/{cfg.budget_check_step}, "
            f"budget={budget}, debt={debt}"
        )

    post_check = [s for s in snapshots if s["step"] >= cfg.budget_check_step]
    if len(post_check) < 2:
        snap = post_check[0] if post_check else snapshots[-1]
        return SKIP, (
            f"Need ≥2 snapshots after step {cfg.budget_check_step}, "
            f"have {len(post_check)}"
        )

    # Check that debt exists or was recently positive
    any_debt = any(s.get("structural_budget_debt", 0) > 0 for s in post_check)
    # Check debt paydown pattern: debt should decrease between some consecutive pairs
    debt_decreased = False
    for i in range(1, len(post_check)):
        prev_debt = post_check[i - 1].get("structural_budget_debt", 0)
        curr_debt = post_check[i].get("structural_budget_debt", 0)
        if curr_debt < prev_debt:
            debt_decreased = True
            break

    latest = post_check[-1]
    details = (
        f"budget={latest.get('prune_budget', 0)}, "
        f"debt={latest.get('structural_budget_debt', 0)}, "
        f"any_debt_seen={any_debt}, debt_paydown={debt_decreased}"
    )

    if any_debt and debt_decreased:
        return OK, details
    elif any_debt:
        return WARN, f"Debt exists but no paydown observed yet — {details}"
    else:
        return WARN, f"No structural debt seen post-warmup — {details}"


def check_dff_oscillation(
    snapshots: List[Dict], cfg: MonitorConfig,
) -> Tuple[str, str]:
    """Check 3: Are d_ff values showing real net growth vs churn?"""
    if len(snapshots) < cfg.dff_oscillation_window:
        return SKIP, f"Need ≥{cfg.dff_oscillation_window} snapshots"

    recent = snapshots[-cfg.dff_oscillation_window:]
    num_layers = len(recent[0].get("d_ff_per_layer", []))
    if num_layers == 0:
        return SKIP, "No d_ff_per_layer in snapshots"

    oscillating_layers = []
    growing_layers = []
    for layer_idx in range(num_layers):
        values = [s["d_ff_per_layer"][layer_idx] for s in recent
                  if layer_idx < len(s.get("d_ff_per_layer", []))]
        if not values:
            continue
        val_range = max(values) - min(values)
        net_change = values[-1] - values[0]

        # Churn = lots of activity (high range) but little net progress
        churn_ratio = abs(net_change) / val_range if val_range > 0 else 1.0
        if val_range > 0 and churn_ratio < cfg.dff_churn_ratio:
            oscillating_layers.append(
                f"L{layer_idx}({values[0]}→{values[-1]}, range={val_range}, net={net_change})"
            )
        elif net_change > 0:
            growing_layers.append(
                f"L{layer_idx}({values[0]}→{values[-1]}, +{net_change})"
            )

    current_dffs = snapshots[-1].get("d_ff_per_layer", [])
    summary = f"d_ff={current_dffs}"

    if not oscillating_layers:
        return OK, f"No churn detected. Growing: {growing_layers or 'none'}. {summary}"
    elif len(oscillating_layers) > num_layers // 2:
        return FAIL, (
            f"Majority of layers oscillating: {oscillating_layers}. {summary}"
        )
    else:
        return WARN, (
            f"Some oscillation: {oscillating_layers}. "
            f"Growing: {growing_layers or 'none'}. {summary}"
        )


def check_split_rank_alignment(
    snapshots: List[Dict], cfg: MonitorConfig,
) -> Tuple[str, str]:
    """Check 4: Are splits targeting the lowest-rank layers?"""
    if len(snapshots) < 2:
        return SKIP, "Need ≥2 snapshots"

    # Find snapshots where splits occurred (cumulative_splits increased)
    split_events = []
    for i in range(1, len(snapshots)):
        prev_splits = snapshots[i - 1].get("cumulative_splits", 0)
        curr_splits = snapshots[i].get("cumulative_splits", 0)
        if curr_splits > prev_splits:
            split_events.append(snapshots[i])

    if not split_events:
        return SKIP, "No split events observed yet"

    mismatches = []
    for snap in split_events[-5:]:  # Check last 5 split events
        diag = snap.get("layer_diagnostics", [])
        dff = snap.get("d_ff_per_layer", [])
        prev_snap_idx = snapshots.index(snap) - 1
        if prev_snap_idx < 0:
            continue
        prev_dff = snapshots[prev_snap_idx].get("d_ff_per_layer", [])

        if not diag or not dff or not prev_dff:
            continue

        # Which layers grew?
        split_layers = [
            i for i in range(min(len(dff), len(prev_dff)))
            if dff[i] > prev_dff[i]
        ]
        # Rank all layers by effective rank (ascending = lowest rank first)
        ranked = sorted(
            enumerate(diag),
            key=lambda x: x[1].get("effective_rank", float("inf")),
        )
        lowest_rank_layers = {
            idx for idx, _ in ranked[:len(split_layers) + cfg.rank_mismatch_tolerance]
        }

        for sl in split_layers:
            if sl not in lowest_rank_layers:
                ranks_str = ", ".join(
                    f"L{idx}={d.get('effective_rank', '?'):.1f}"
                    for idx, d in ranked[:4]
                )
                mismatches.append(
                    f"step {snap['step']}: split L{sl} but lowest ranks are [{ranks_str}]"
                )

    if not mismatches:
        return OK, f"All {len(split_events)} split events targeted low-rank layers"
    elif len(mismatches) <= 1:
        return WARN, f"Minor misalignment: {mismatches[0]}"
    else:
        return FAIL, f"{len(mismatches)} rank mismatches: {'; '.join(mismatches[:3])}"


def check_dead_units(
    snapshots: List[Dict], cfg: MonitorConfig,
) -> Tuple[str, str]:
    """Check 5: Are dead units rising after structural events?"""
    if len(snapshots) < 2:
        return SKIP, "Need ≥2 snapshots"

    # Track dead unit counts per snapshot
    dead_series: List[Tuple[int, int]] = []  # (step, total_dead)
    for snap in snapshots:
        diag = snap.get("layer_diagnostics", [])
        total_dead = sum(d.get("dead_units", 0) for d in diag)
        dead_series.append((snap["step"], total_dead))

    if not dead_series:
        return SKIP, "No layer_diagnostics available"

    # Check for sustained rise
    max_rise = 0
    for i in range(1, len(dead_series)):
        rise = dead_series[i][1] - dead_series[i - 1][1]
        max_rise = max(max_rise, rise)

    latest_dead = dead_series[-1][1]
    initial_dead = dead_series[0][1]
    net_change = latest_dead - initial_dead

    details = (
        f"dead_units: {initial_dead}→{latest_dead} (net {net_change:+d}), "
        f"max single-step rise={max_rise}"
    )

    if max_rise > cfg.dead_unit_growth_threshold * 3:
        return FAIL, f"Dead units exploding: {details}"
    elif max_rise > cfg.dead_unit_growth_threshold:
        return WARN, f"Dead units rising: {details}"
    else:
        return OK, details


def check_structural_churn(
    snapshots: List[Dict], cfg: MonitorConfig,
) -> Tuple[str, str]:
    """Check 6: Are merges closely tracking splits (structural cost for no gain)?"""
    if not snapshots:
        return SKIP, "No snapshots"

    latest = snapshots[-1]
    splits = latest.get("cumulative_splits", 0)
    merges = latest.get("cumulative_merges", 0)
    removed = latest.get("cumulative_removed", 0)
    reinit = latest.get("cumulative_reinit", 0)
    grown = latest.get("cumulative_grown", 0)
    pruned = latest.get("cumulative_pruned", 0)
    neurogenesis = latest.get("cumulative_neurogenesis", 0)

    details = (
        f"splits={splits}, merges={merges}, removed={removed}, "
        f"neurogenesis={neurogenesis}, reinit={reinit}, grown={grown}, pruned={pruned}"
    )

    if splits == 0 and neurogenesis == 0:
        return SKIP, f"No splits or neurogenesis yet. {details}"

    total_growth_events = splits + neurogenesis
    ratio = merges / total_growth_events if total_growth_events > 0 else 0
    net_structural = total_growth_events - merges - removed

    if ratio > cfg.merge_split_churn_ratio and net_structural <= 0:
        return FAIL, (
            f"Merges nearly cancel growth: ratio={ratio:.2f}, "
            f"net structural change={net_structural}. {details}"
        )
    elif ratio > cfg.merge_split_churn_ratio:
        return WARN, (
            f"High merge/growth ratio={ratio:.2f} but net change={net_structural}. "
            f"{details}"
        )
    else:
        return OK, f"Merge/growth ratio={ratio:.2f}, net change=+{net_structural}. {details}"


def _describe_structural_delta(prev: Dict, curr: Dict) -> str:
    """Summarize what structural edits happened between two snapshots."""
    parts = []
    ds = curr.get("cumulative_splits", 0) - prev.get("cumulative_splits", 0)
    dm = curr.get("cumulative_merges", 0) - prev.get("cumulative_merges", 0)
    dr = curr.get("cumulative_reinit", 0) - prev.get("cumulative_reinit", 0)
    dd = curr.get("cumulative_removed", 0) - prev.get("cumulative_removed", 0)
    dn = curr.get("cumulative_neurogenesis", 0) - prev.get("cumulative_neurogenesis", 0)
    if ds: parts.append(f"{ds} splits")
    if dm: parts.append(f"{dm} merges")
    if dn: parts.append(f"{dn} neurogenesis")
    if dr: parts.append(f"{dr} reinit")
    if dd: parts.append(f"{dd} removed")
    return ", ".join(parts) if parts else "no edits"


def check_val_loss_around_structural(
    snapshots: List[Dict], cfg: MonitorConfig,
) -> Tuple[str, str]:
    """Check 7: Does val_loss fail to recover after structural events?

    The val_loss on a structural step is measured BEFORE the structural edit.
    Recovery is judged only from subsequent evals. An event is recovered if
    any of the next 3 evals reaches <= pre_event_val_loss + 0.05.
    """
    if len(snapshots) < 4:
        return SKIP, "Need >=4 snapshots"

    # Find structural event snapshots (splits, merges, neurogenesis, removed, or reinit)
    structural_steps = []
    for i in range(1, len(snapshots)):
        prev = snapshots[i - 1]
        curr = snapshots[i]
        if (curr.get("cumulative_splits", 0) > prev.get("cumulative_splits", 0)
                or curr.get("cumulative_merges", 0) > prev.get("cumulative_merges", 0)
                or curr.get("cumulative_neurogenesis", 0) > prev.get("cumulative_neurogenesis", 0)
                or curr.get("cumulative_removed", 0) > prev.get("cumulative_removed", 0)
                or curr.get("cumulative_reinit", 0) > prev.get("cumulative_reinit", 0)):
            structural_steps.append(i)

    if not structural_steps:
        return SKIP, "No structural events observed"

    degradations = []
    recovered_events = 0
    pending_events = 0
    recovery_tolerance = 0.05

    for idx in structural_steps[-5:]:  # Check last 5 events
        # The val_loss at idx is PRE-EDIT — this is our baseline
        pre_loss = snapshots[idx].get("val_loss", 0)
        if pre_loss == 0:
            continue

        edits = _describe_structural_delta(snapshots[idx - 1], snapshots[idx])
        mode = snapshots[idx].get("structural_mode", "?")
        step = snapshots[idx]["step"]

        # Recovery is judged from post-event evals only (idx+1, idx+2, idx+3)
        recovery_window = min(idx + 1 + cfg.val_loss_degradation_evals, len(snapshots))
        post_losses = [
            snapshots[j].get("val_loss", 0)
            for j in range(idx + 1, recovery_window)
        ]

        if not post_losses:
            pending_events += 1
            continue

        best_post = min(post_losses)
        recovered = best_post <= pre_loss + recovery_tolerance

        if recovered:
            recovered_events += 1
        else:
            # Also check ALL later evals
            all_later = [
                snapshots[j].get("val_loss", 0)
                for j in range(idx + 1, len(snapshots))
            ]
            best_later = min(all_later) if all_later else float("inf")
            degradations.append(
                f"step {step} [mode={mode}, {edits}]: "
                f"pre={pre_loss:.4f}, "
                f"post_window=[{', '.join(f'{l:.4f}' for l in post_losses)}], "
                f"best_later={best_later:.4f} "
                f"(+{best_later - pre_loss:.4f} vs pre)"
            )

    total_checked = recovered_events + len(degradations)
    pending_str = f", {pending_events} pending" if pending_events else ""

    if not degradations:
        return OK, (
            f"{recovered_events}/{total_checked} recovered"
            f"{pending_str} across {len(structural_steps)} structural events"
        )
    elif len(degradations) == 1:
        return WARN, (
            f"1 unrecovered ({recovered_events} recovered{pending_str}): "
            f"{degradations[0]}"
        )
    else:
        return FAIL, (
            f"{len(degradations)} unrecovered ({recovered_events} recovered"
            f"{pending_str}): {'; '.join(degradations[:3])}"
        )


def check_last_layer_reset(
    snapshots: List[Dict], cfg: MonitorConfig,
) -> Tuple[str, str]:
    """Check 8: Did reset_last_layer fire at step 10k, and did loss recover?"""
    latest_step = snapshots[-1]["step"] if snapshots else 0

    if latest_step < cfg.reset_step:
        return SKIP, f"Only at step {latest_step}/{cfg.reset_step}"

    # Find the snapshot at/near the reset step
    reset_snaps = [s for s in snapshots if s["step"] >= cfg.reset_step]
    if not reset_snaps:
        return SKIP, "No snapshots at reset step"

    reset_snap = reset_snaps[0]
    reset_idx = snapshots.index(reset_snap)

    # The reset should appear as a val_loss bump
    pre_loss = snapshots[reset_idx - 1].get("val_loss", 0) if reset_idx > 0 else 0
    reset_loss = reset_snap.get("val_loss", 0)

    # Check recovery in next N evals
    recovery_snaps = snapshots[reset_idx + 1:reset_idx + 1 + cfg.reset_recovery_evals]
    if not recovery_snaps:
        return SKIP, f"No post-reset snapshots yet (step {reset_snap['step']})"

    post_losses = [s.get("val_loss", 0) for s in recovery_snaps]
    recovered = any(l <= pre_loss * 1.05 for l in post_losses)
    max_degradation = max(post_losses) - pre_loss if post_losses else 0

    details = (
        f"pre={pre_loss:.4f}, at_reset={reset_loss:.4f}, "
        f"post={[round(l, 4) for l in post_losses]}"
    )

    if recovered:
        return OK, f"Reset fired and recovered. {details}"
    elif max_degradation <= cfg.reset_max_degradation:
        return WARN, f"Reset fired, modest degradation. {details}"
    else:
        return FAIL, f"Reset caused prolonged degradation. {details}"


def check_event_cadence(
    snapshots: List[Dict], cfg: MonitorConfig,
) -> Tuple[str, str]:
    """Check 9: Warmup respected? Growth=0 on structural+prune coincidence?"""
    issues = []

    # 9a: No pruning/growth before warmup
    for snap in snapshots:
        if snap["step"] < cfg.warmup_steps:
            if snap.get("cumulative_pruned", 0) > 0 or snap.get("cumulative_grown", 0) > 0:
                issues.append(
                    f"Prune/grow before warmup at step {snap['step']}: "
                    f"pruned={snap.get('cumulative_pruned', 0)}, "
                    f"grown={snap.get('cumulative_grown', 0)}"
                )
                break

    # 9b: Check for grown=0 on structural steps (from training.log plasticity events)
    # We can approximate this from metrics.jsonl: if cumulative_splits increased AND
    # cumulative_grown didn't change between two snapshots, the guard is working
    structural_growth_coincidences = 0
    structural_growth_violations = 0
    for i in range(1, len(snapshots)):
        prev = snapshots[i - 1]
        curr = snapshots[i]
        had_structural = (
            curr.get("cumulative_splits", 0) > prev.get("cumulative_splits", 0)
            or curr.get("cumulative_merges", 0) > prev.get("cumulative_merges", 0)
            or curr.get("cumulative_neurogenesis", 0) > prev.get("cumulative_neurogenesis", 0)
            or curr.get("cumulative_removed", 0) > prev.get("cumulative_removed", 0)
            or curr.get("cumulative_reinit", 0) > prev.get("cumulative_reinit", 0)
        )
        had_growth = (
            curr.get("cumulative_grown", 0) > prev.get("cumulative_grown", 0)
        )
        if had_structural:
            if had_growth:
                structural_growth_violations += 1
            else:
                structural_growth_coincidences += 1

    if structural_growth_violations > 0:
        issues.append(
            f"Growth happened on {structural_growth_violations} structural steps "
            f"(expected grown=0 when structural edits fire)"
        )

    cadence_detail = (
        f"structural_steps_no_growth={structural_growth_coincidences}, "
        f"violations={structural_growth_violations}"
    )

    if not issues:
        return OK, f"Warmup respected, cadence correct. {cadence_detail}"
    elif all("before warmup" not in i for i in issues):
        return WARN, "; ".join(issues)
    else:
        return FAIL, "; ".join(issues)


def check_mode_exclusivity(
    snapshots: List[Dict], cfg: MonitorConfig,
) -> Tuple[str, str]:
    """Check 10: No single structural step should contain multiple disruptive families."""
    if len(snapshots) < 2:
        return SKIP, "Need >=2 snapshots"

    violations = []
    for i in range(1, len(snapshots)):
        prev = snapshots[i - 1]
        curr = snapshots[i]
        ds = curr.get("cumulative_splits", 0) - prev.get("cumulative_splits", 0)
        dm = curr.get("cumulative_merges", 0) - prev.get("cumulative_merges", 0)
        dn = curr.get("cumulative_neurogenesis", 0) - prev.get("cumulative_neurogenesis", 0)
        dr = curr.get("cumulative_removed", 0) - prev.get("cumulative_removed", 0)
        di = curr.get("cumulative_reinit", 0) - prev.get("cumulative_reinit", 0)

        families = []
        if ds > 0: families.append(f"splits={ds}")
        if dm > 0: families.append(f"merges={dm}")
        if dn > 0: families.append(f"neurogenesis={dn}")
        if dr > 0 or di > 0: families.append(f"maintenance(rm={dr},ri={di})")

        if len(families) > 1:
            violations.append(f"step {curr['step']}: {', '.join(families)}")

    if not violations:
        return OK, "All structural steps have at most one disruptive family"
    elif len(violations) <= 2:
        return WARN, f"{len(violations)} multi-family steps: {'; '.join(violations[:2])}"
    else:
        return FAIL, f"{len(violations)} multi-family steps: {'; '.join(violations[:3])}"


def check_structural_only_regime(
    snapshots: List[Dict], cfg: MonitorConfig,
) -> Tuple[str, str]:
    """Check 11: Warn if structural debt grows while prune/grow stay flat."""
    if len(snapshots) < 6:
        return SKIP, "Need >=6 snapshots"

    recent = snapshots[-6:]
    debts = [s.get("structural_budget_debt", 0) for s in recent]
    growns = [s.get("cumulative_grown", 0) for s in recent]
    pruneds = [s.get("cumulative_pruned", 0) for s in recent]

    debt_increasing = all(debts[i] >= debts[i-1] for i in range(1, len(debts))) and debts[-1] > debts[0]
    grown_flat = (growns[-1] - growns[0]) == 0
    pruned_flat = (pruneds[-1] - pruneds[0]) == 0

    details = (
        f"debt: {debts[0]}->{debts[-1]}, "
        f"grown: {growns[0]}->{growns[-1]}, "
        f"pruned: {pruneds[0]}->{pruneds[-1]}"
    )

    if debt_increasing and grown_flat and pruned_flat:
        return WARN, f"Structural-only regime: debt rising while prune/grow stalled. {details}"
    else:
        return OK, f"Normal regime. {details}"


# ---------------------------------------------------------------------------
# Main report
# ---------------------------------------------------------------------------

CHECK_NAMES = [
    ("1. Sparsity->target", check_sparsity_convergence),
    ("2. Budget & debt", check_budget_debt),
    ("3. d_ff oscillation", check_dff_oscillation),
    ("4. Split->rank align", check_split_rank_alignment),
    ("5. Dead units", check_dead_units),
    ("6. Structural churn", check_structural_churn),
    ("7. Val loss stability", check_val_loss_around_structural),
    ("8. Last layer reset", check_last_layer_reset),
    ("9. Event cadence", check_event_cadence),
    ("10. Mode exclusivity", check_mode_exclusivity),
    ("11. Structural regime", check_structural_only_regime),
]

STATUS_SYMBOLS = {OK: "✅", WARN: "⚠️ ", FAIL: "❌", SKIP: "⏭️ "}
STATUS_COLORS = {
    OK: "\033[32m", WARN: "\033[33m", FAIL: "\033[31m", SKIP: "\033[90m",
}
RESET_COLOR = "\033[0m"


def run_report(run_dir: str, cfg: MonitorConfig) -> int:
    """Run all checks and print report. Returns exit code (0=ok, 1=fail)."""
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    if not os.path.isfile(metrics_path):
        logger.error(f"No metrics.jsonl in {run_dir}")
        return 1

    snapshots = load_snapshots(metrics_path)
    if not snapshots:
        logger.error(f"No valid snapshots in {metrics_path}")
        return 1

    latest = snapshots[-1]
    step = latest["step"]

    # Print header
    print(f"\n{'='*70}")
    print(f"  TRAINING HEALTH MONITOR -- step {step:,}")
    print(f"  Run: {run_dir}")
    print(f"  Snapshots: {len(snapshots)} | "
          f"Loss: {latest.get('train_loss', '?'):.4f} train, "
          f"{latest.get('val_loss', '?'):.4f} val | "
          f"PPL: {latest.get('perplexity', '?')}")
    print(f"{'='*70}")

    counts = {OK: 0, WARN: 0, FAIL: 0, SKIP: 0}
    results = []

    for name, check_fn in CHECK_NAMES:
        try:
            status, detail = check_fn(snapshots, cfg)
        except Exception as e:
            status, detail = FAIL, f"Check crashed: {e}"

        counts[status] += 1
        results.append((name, status, detail))

        color = STATUS_COLORS[status]
        symbol = STATUS_SYMBOLS[status]
        print(f"\n{color}{symbol} {name}: {status}{RESET_COLOR}")
        print(f"   {detail}")

    # Summary
    print(f"\n{'─'*70}")
    summary_parts = []
    for s in [OK, WARN, FAIL, SKIP]:
        if counts[s] > 0:
            summary_parts.append(f"{counts[s]} {s}")
    print(f"  SUMMARY: {', '.join(summary_parts)}")
    print(f"{'─'*70}\n")

    return 1 if counts[FAIL] > 0 else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import typer

    app = typer.Typer()

    @app.command()
    def main(
        run_dir: Optional[str] = typer.Option(
            None, help="Path to run directory (auto-detects latest if omitted)",
        ),
        watch: Optional[int] = typer.Option(
            None, help="Re-run every N seconds (0 = run once)",
        ),
    ) -> None:
        """Monitor training health by checking metrics.jsonl invariants."""
        cfg = MonitorConfig()

        if run_dir is None:
            run_dir = find_latest_run()
            if run_dir is None:
                logger.error("No run directory with metrics.jsonl found in runs/")
                raise typer.Exit(code=1)
            logger.info(f"Auto-detected run: {run_dir}")

        if watch is None or watch == 0:
            exit_code = run_report(run_dir, cfg)
            raise typer.Exit(code=exit_code)
        else:
            logger.info(f"Watching every {watch}s (Ctrl+C to stop)")
            try:
                while True:
                    # Clear terminal for fresh output
                    subprocess.run(["clear"], check=False)
                    run_report(run_dir, cfg)
                    time.sleep(watch)
            except KeyboardInterrupt:
                print("\nStopped watching.")

    app()
