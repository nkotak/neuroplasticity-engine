"""CLI entrypoint for verified self-learning cycles."""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Optional

import typer

from autoresearch.backends.mlx_backend import MLXBackend
from autoresearch.promotion import evaluate_promotion_gate
from autoresearch.results import CYCLE_COLUMNS, append_tsv
from config import SelfLearnConfig
from selflearn.benchmark import evaluate_benchmark, load_benchmark_items
from selflearn.curator import curate_candidates, load_existing_hashes, write_jsonl
from selflearn.generator import solve_candidate_tasks
from selflearn.store import create_cycle, write_manifest
from selflearn.task_bank import bucket_failures, generate_candidate_tasks, seed_from_cycle_id


app = typer.Typer(help="Verified self-learning loop.")


def _default_cycle_id() -> str:
    return f"cycle_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


@app.command()
def benchmark(
    student_checkpoint: str = typer.Argument(..., help="Checkpoint directory"),
    benchmark_path: str = typer.Option("data/benchmarks/selflearn_v1.jsonl", help="Benchmark JSONL"),
    tokenizer_path: Optional[str] = typer.Option(None, help="Tokenizer path"),
) -> None:
    backend = MLXBackend(os.getcwd())
    result = evaluate_benchmark(student_checkpoint, benchmark_path, tokenizer_path, backend)
    print(json.dumps(result, indent=2))


@app.command()
def cycle(
    student_checkpoint: str = typer.Argument(..., help="Current promoted checkpoint"),
    benchmark_path: str = typer.Option("data/benchmarks/selflearn_v1.jsonl", help="Held-out benchmark"),
    tokenizer_path: Optional[str] = typer.Option(None, help="Tokenizer path"),
    cycle_id: str = typer.Option(None, help="Cycle ID"),
    candidate_tasks_per_cycle: int = typer.Option(2000, help="Number of candidate tasks"),
    teacher_model: str = typer.Option("qwen2.5-14b-instruct", help="Teacher model name"),
    finetune_steps: int = typer.Option(5000, help="Fine-tune steps for accepted data"),
    seed: Optional[int] = typer.Option(None, help="Optional task-generation seed"),
) -> None:
    repo_root = os.getcwd()
    backend = MLXBackend(repo_root)
    cfg = SelfLearnConfig(
        enabled=True,
        teacher_model=teacher_model,
        candidate_tasks_per_cycle=candidate_tasks_per_cycle,
        cycle_finetune_steps=finetune_steps,
    )
    cycle_id = cycle_id or _default_cycle_id()
    cycle_paths = create_cycle(os.path.join(repo_root, "runs", "selflearn"), cycle_id)

    benchmark_before = evaluate_benchmark(student_checkpoint, benchmark_path, tokenizer_path, backend)
    with open(cycle_paths.benchmark_before_path, "w", encoding="utf-8") as fh:
        json.dump(benchmark_before, fh, indent=2)

    failure_buckets = bucket_failures(benchmark_before)
    task_seed = seed if seed is not None else seed_from_cycle_id(cycle_id)
    tasks = generate_candidate_tasks(
        failure_buckets,
        cfg.candidate_tasks_per_cycle,
        seed=task_seed,
    )
    candidates = solve_candidate_tasks(
        tasks=tasks,
        backend=backend,
        checkpoint_dir=student_checkpoint,
        tokenizer_path=tokenizer_path,
        student_samples_per_task=cfg.student_samples_per_task,
        teacher_samples_per_task=cfg.teacher_samples_per_task,
        teacher_model=cfg.teacher_model,
    )
    write_jsonl(cycle_paths.candidates_path, candidates)

    benchmark_items = load_benchmark_items(benchmark_path)
    accepted_root = os.path.join(repo_root, "data", "selflearn", "accepted", cycle_id)
    existing_hashes = load_existing_hashes(
        os.path.join(repo_root, "data", "selflearn", "accepted"),
        benchmark_items,
    )
    accepted = curate_candidates(
        candidates=candidates,
        cycle_id=cycle_id,
        student_checkpoint=student_checkpoint,
        teacher_model=cfg.teacher_model,
        existing_hashes=existing_hashes,
    )
    os.makedirs(accepted_root, exist_ok=True)
    accepted_path = os.path.join(accepted_root, "accepted.jsonl")
    write_jsonl(accepted_path, accepted)
    write_jsonl(cycle_paths.accepted_path, accepted)

    manifest = {
        "cycle_id": cycle_id,
        "student_checkpoint": student_checkpoint,
        "accepted_count": len(accepted),
        "teacher_model": cfg.teacher_model,
        "benchmark_path": benchmark_path,
        "task_seed": task_seed,
    }
    write_manifest(cycle_paths.manifest_path, manifest)
    with open(cycle_paths.train_config_snapshot_path, "w", encoding="utf-8") as fh:
        json.dump(vars(cfg), fh, indent=2)

    status = "discard"
    benchmark_after = {}
    candidate_run_dir = ""
    if accepted:
        candidate_run_dir = os.path.join(repo_root, "runs", "selflearn", cycle_id, "train")
        backend.run_full_train(
            run_dir=candidate_run_dir,
            profile="promotion",
            tokenizer_path=tokenizer_path,
            init_checkpoint=student_checkpoint,
            selflearn_dir=os.path.join(repo_root, "data", "selflearn", "accepted"),
            total_steps=cfg.cycle_finetune_steps,
        )
        candidate_checkpoint = os.path.join(candidate_run_dir, "best")
        if not os.path.isdir(candidate_checkpoint):
            candidate_checkpoint = os.path.join(candidate_run_dir, "final")
        benchmark_after = evaluate_benchmark(candidate_checkpoint, benchmark_path, tokenizer_path, backend)
        with open(cycle_paths.benchmark_after_path, "w", encoding="utf-8") as fh:
            json.dump(benchmark_after, fh, indent=2)
        decision = evaluate_promotion_gate(
            baseline_score=benchmark_before["composite_score"],
            candidate_score=benchmark_after["composite_score"],
            baseline_buckets=benchmark_before["bucket_scores"],
            candidate_buckets=benchmark_after["bucket_scores"],
            min_score_improvement=2.0,
        )
        status = "promote" if decision.passed else "reject"

    append_tsv(
        os.path.join(repo_root, "runs", "selflearn", "cycles.tsv"),
        CYCLE_COLUMNS,
        {
            "cycle_id": cycle_id,
            "student_checkpoint": student_checkpoint,
            "accepted_count": len(accepted),
            "before_score": benchmark_before["composite_score"],
            "after_score": benchmark_after.get("composite_score", ""),
            "status": status,
            "description": f"accepted={len(accepted)} run_dir={candidate_run_dir}",
        },
    )
    print(
        json.dumps(
            {
                "cycle_id": cycle_id,
                "accepted_count": len(accepted),
                "before_score": benchmark_before["composite_score"],
                "after_score": benchmark_after.get("composite_score"),
                "status": status,
                "candidate_run_dir": candidate_run_dir,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    app()
