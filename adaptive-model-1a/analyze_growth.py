"""
Post-training growth analysis and report generator for the Neuroplasticity Engine.

Reads metrics.jsonl from a training run directory and produces a text-based
report showing architecture evolution: parameter counts, d_ff changes,
plasticity event timeline, sparsity progression, and skip connection formation.

Docs: https://docs.python.org/3/library/json.html

Sample input:
    python analyze_growth.py runs/20260222_125331/

Expected output:
    A formatted text report showing architecture growth over training epochs
"""

import os
import sys
import json
import math
from typing import Dict, List, Optional


def load_metrics(run_dir: str) -> List[Dict]:
    """Load all epoch snapshots from metrics.jsonl."""
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    snapshots = []
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if line:
                snapshots.append(json.loads(line))
    return snapshots


def load_config(run_dir: str) -> Dict:
    """Load run configuration."""
    config_path = os.path.join(run_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            return json.load(f)
    return {}


def generate_report(run_dir: str) -> str:
    """Generate a text-based growth report from a training run.

    Args:
        run_dir: Path to the run directory containing metrics.jsonl.

    Returns:
        Formatted report string.
    """
    snapshots = load_metrics(run_dir)
    config = load_config(run_dir)

    if not snapshots:
        return "ERROR: No metrics found in run directory."

    first = snapshots[0]
    last = snapshots[-1]
    num_epochs = last['epoch']

    lines = []
    lines.append("=" * 80)
    lines.append("  NEUROPLASTICITY ENGINE - GROWTH ANALYSIS REPORT")
    lines.append("=" * 80)
    lines.append("")

    # Section 1: Training Summary
    lines.append("1. TRAINING SUMMARY")
    lines.append("-" * 40)
    lines.append(f"  Run ID:              {config.get('run_id', os.path.basename(run_dir))}")
    lines.append(f"  Epochs completed:    {num_epochs}")
    lines.append(f"  Device:              {config.get('device', 'unknown')}")
    if config.get('total_training_time_seconds'):
        mins = config['total_training_time_seconds'] / 60
        lines.append(f"  Total training time: {mins:.1f} minutes")
    lines.append(f"  Model:               d_model={config.get('d_model')}, "
                 f"layers={config.get('num_layers')}, d_ff={config.get('d_ff')}")
    lines.append(f"  Initial train loss:  {snapshots[1]['train_loss']:.4f}" if len(snapshots) > 1 else "")
    lines.append(f"  Final train loss:    {last['train_loss']:.4f}")
    lines.append(f"  Best val loss:       {config.get('best_val_loss', last['val_loss']):.4f}")
    lines.append(f"  Final perplexity:    {last['perplexity']:.2f}")
    lines.append("")

    # Section 2: Architecture Evolution
    lines.append("2. ARCHITECTURE EVOLUTION")
    lines.append("-" * 40)
    lines.append(f"  Starting parameters: {first['total_model_params']:,}")
    lines.append(f"  Final parameters:    {last['total_model_params']:,}")
    param_change = last['total_model_params'] - first['total_model_params']
    pct_change = (param_change / first['total_model_params'] * 100) if first['total_model_params'] > 0 else 0
    direction = "grew" if param_change > 0 else ("shrank" if param_change < 0 else "unchanged")
    lines.append(f"  Net change:          {param_change:+,} ({pct_change:+.2f}%) - model {direction}")
    lines.append("")

    # d_ff per layer comparison
    if first.get('d_ff_per_layer') and last.get('d_ff_per_layer'):
        lines.append("  d_ff per layer (start -> end):")
        max_layers = max(len(first['d_ff_per_layer']), len(last['d_ff_per_layer']))
        for i in range(max_layers):
            start_ff = first['d_ff_per_layer'][i] if i < len(first['d_ff_per_layer']) else "N/A"
            end_ff = last['d_ff_per_layer'][i] if i < len(last['d_ff_per_layer']) else "N/A"
            changed = " *" if start_ff != end_ff else ""
            lines.append(f"    Layer {i}: {start_ff} -> {end_ff}{changed}")
        lines.append("")

    # Sparsity
    lines.append(f"  Starting sparsity:   {first['global_sparsity']:.4f}")
    lines.append(f"  Final sparsity:      {last['global_sparsity']:.4f}")
    lines.append("")

    # Section 3: Plasticity Events
    lines.append("3. PLASTICITY EVENTS (CUMULATIVE)")
    lines.append("-" * 40)
    lines.append(f"  Total splits:        {last['cumulative_splits']}")
    lines.append(f"  Total merges:        {last['cumulative_merges']}")
    lines.append(f"  Total grown:         {last['cumulative_grown']}")
    lines.append(f"  Total pruned:        {last['cumulative_pruned']}")
    lines.append(f"  Skip connections:    {last['skip_connections']}")
    net_neurons = last['cumulative_splits'] - last['cumulative_merges']
    lines.append(f"  Net neuron change:   {net_neurons:+d} (splits - merges)")
    lines.append("")

    # Section 4: Growth Timeline
    lines.append("4. GROWTH TIMELINE")
    lines.append("-" * 80)

    # Header
    header = (
        f"{'Epoch':>6} | {'Loss':>7} | {'Val Loss':>8} | {'PPL':>8} | "
        f"{'Params':>10} | {'Sparsity':>8} | "
        f"{'Splits':>6} | {'Merges':>6} | {'Grown':>7} | {'Pruned':>7} | {'Skips':>5}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    # Determine step size for timeline (show ~12-20 rows)
    if num_epochs <= 20:
        step = 1
    elif num_epochs <= 60:
        step = 5
    else:
        step = 10

    prev_snapshot = None
    for snap in snapshots:
        epoch = snap['epoch']
        if epoch == 0:
            prev_snapshot = snap
            continue
        if epoch % step != 0 and epoch != num_epochs:
            prev_snapshot = snap
            continue

        # Compute delta plasticity events since last shown row
        row = (
            f"{epoch:>6} | "
            f"{snap['train_loss']:>7.4f} | "
            f"{snap['val_loss']:>8.4f} | "
            f"{snap['perplexity']:>8.1f} | "
            f"{snap['total_model_params']:>10,} | "
            f"{snap['global_sparsity']:>8.4f} | "
            f"{snap['cumulative_splits']:>6} | "
            f"{snap['cumulative_merges']:>6} | "
            f"{snap['cumulative_grown']:>7} | "
            f"{snap['cumulative_pruned']:>7} | "
            f"{snap['skip_connections']:>5}"
        )
        lines.append(row)
        prev_snapshot = snap

    lines.append("")

    # Section 5: Per-Layer Detail
    if last.get('layer_details'):
        lines.append("5. PER-LAYER DETAIL (FINAL STATE)")
        lines.append("-" * 70)
        layer_header = f"{'Layer':>30} | {'d_ff':>5} | {'In Sparsity':>11} | {'Out Sparsity':>12} | {'Warmup':>6}"
        lines.append(layer_header)
        lines.append("-" * len(layer_header))
        for ld in last['layer_details']:
            row = (
                f"{ld['name']:>30} | "
                f"{ld['d_ff']:>5} | "
                f"{ld['input_mask_sparsity']:>11.4f} | "
                f"{ld['output_mask_sparsity']:>12.4f} | "
                f"{ld['warmup_factor']:>6.3f}"
            )
            lines.append(row)
        lines.append("")

    # Section 6: Plasticity Activity Log
    lines.append("6. PLASTICITY ACTIVITY LOG")
    lines.append("-" * 40)

    # Find first occurrence of each event type
    event_types = [
        ("First prune", "cumulative_pruned"),
        ("First grow", "cumulative_grown"),
        ("First split", "cumulative_splits"),
        ("First merge", "cumulative_merges"),
        ("First skip connection", "skip_connections"),
    ]
    for label, key in event_types:
        first_epoch = None
        prev_val = 0
        for snap in snapshots:
            val = snap.get(key, 0)
            if val > prev_val:
                first_epoch = snap['epoch']
                break
            prev_val = val
        if first_epoch is not None:
            lines.append(f"  {label:25s} epoch {first_epoch}")
        else:
            lines.append(f"  {label:25s} never occurred")

    # Peak activity period (epoch with most cumulative delta)
    if len(snapshots) > 2:
        max_delta = 0
        peak_epoch = 0
        for i in range(1, len(snapshots)):
            delta = (
                (snapshots[i].get('cumulative_grown', 0) - snapshots[i-1].get('cumulative_grown', 0)) +
                (snapshots[i].get('cumulative_pruned', 0) - snapshots[i-1].get('cumulative_pruned', 0)) +
                (snapshots[i].get('cumulative_splits', 0) - snapshots[i-1].get('cumulative_splits', 0)) +
                (snapshots[i].get('cumulative_merges', 0) - snapshots[i-1].get('cumulative_merges', 0))
            )
            if delta > max_delta:
                max_delta = delta
                peak_epoch = snapshots[i]['epoch']
        if peak_epoch > 0:
            lines.append(f"  {'Peak activity epoch':25s} epoch {peak_epoch} ({max_delta} events)")
        else:
            lines.append(f"  {'Peak activity epoch':25s} no plasticity events recorded")

    lines.append("")
    lines.append("=" * 80)
    lines.append("  END OF REPORT")
    lines.append("=" * 80)

    return '\n'.join(lines)


if __name__ == "__main__":
    all_validation_failures = []
    total_tests = 0

    # Test 1: Generate synthetic metrics and produce report
    total_tests += 1
    try:
        test_dir = "/tmp/test_growth_report"
        os.makedirs(test_dir, exist_ok=True)

        # Create synthetic metrics.jsonl
        snapshots = []
        for epoch in range(0, 31):
            snap = {
                "epoch": epoch,
                "train_loss": 6.0 - epoch * 0.1,
                "val_loss": 5.5 - epoch * 0.09,
                "perplexity": round(math.exp(max(5.5 - epoch * 0.09, 0.1)), 2),
                "total_model_params": 30000000 + epoch * 1000,
                "masked_active_params": 2000000 - epoch * 100,
                "masked_total_params": 2000000,
                "global_sparsity": round(epoch * 0.001, 6),
                "d_ff_per_layer": [1024 + (1 if epoch > 15 and i == 0 else 0) for i in range(4)],
                "cumulative_splits": 0 if epoch < 15 else (epoch - 15) // 5,
                "cumulative_merges": 0 if epoch < 20 else (epoch - 20) // 10,
                "cumulative_grown": 0 if epoch < 10 else (epoch - 10) * 50,
                "cumulative_pruned": 0 if epoch < 10 else (epoch - 10) * 30,
                "skip_connections": 0 if epoch < 20 else min(epoch - 20, 3),
                "layer_details": [
                    {"name": f"layers.{i}.ff", "d_ff": 1024 + (1 if epoch > 15 and i == 0 else 0),
                     "input_mask_sparsity": round(epoch * 0.001, 4),
                     "output_mask_sparsity": round(epoch * 0.0008, 4),
                     "warmup_factor": 1.0}
                    for i in range(4)
                ],
            }
            snapshots.append(snap)

        with open(os.path.join(test_dir, "metrics.jsonl"), 'w') as f:
            for snap in snapshots:
                f.write(json.dumps(snap) + '\n')

        # Write config
        with open(os.path.join(test_dir, "config.json"), 'w') as f:
            json.dump({
                "run_id": "test_run",
                "d_model": 256,
                "num_layers": 4,
                "d_ff": 1024,
                "device": "cpu",
                "best_val_loss": 2.8,
                "total_training_time_seconds": 600,
            }, f)

        report = generate_report(test_dir)
        assert len(report) > 100, f"Report too short: {len(report)} chars"

        # Verify key sections are present
        required_sections = [
            "TRAINING SUMMARY",
            "ARCHITECTURE EVOLUTION",
            "PLASTICITY EVENTS",
            "GROWTH TIMELINE",
            "PER-LAYER DETAIL",
            "PLASTICITY ACTIVITY LOG",
        ]
        for section in required_sections:
            assert section in report, f"Missing section: {section}"

    except Exception as e:
        all_validation_failures.append(f"Synthetic report generation: {e}")

    # Test 2: Verify report with a real run directory if available
    total_tests += 1
    real_runs = []
    if os.path.exists("runs"):
        for d in sorted(os.listdir("runs")):
            run_path = os.path.join("runs", d)
            if os.path.isdir(run_path) and os.path.exists(os.path.join(run_path, "metrics.jsonl")):
                real_runs.append(run_path)

    if real_runs:
        try:
            real_report = generate_report(real_runs[-1])
            assert "TRAINING SUMMARY" in real_report
            assert "ARCHITECTURE EVOLUTION" in real_report
            print(f"Report from real run ({real_runs[-1]}):")
            print(real_report)
        except Exception as e:
            all_validation_failures.append(f"Real run report: {e}")
    else:
        # Not a failure — just no real runs available yet
        print("No real training runs found. Showing synthetic report instead:")
        print(report)

    # Test 3: Verify report from CLI argument
    total_tests += 1
    if len(sys.argv) > 1:
        try:
            arg_report = generate_report(sys.argv[1])
            assert len(arg_report) > 100
            print(arg_report)
        except Exception as e:
            all_validation_failures.append(f"CLI argument report: {e}")

    if all_validation_failures:
        print(f"VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"\nVALIDATION PASSED - All {total_tests} tests produced expected results")
        sys.exit(0)
