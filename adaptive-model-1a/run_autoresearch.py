"""CLI entrypoint for bounded code-research experiments."""

from __future__ import annotations

import os
from typing import Optional

import typer

from autoresearch.orchestrator import AutoresearchOrchestrator


app = typer.Typer(help="Autoresearch-style bounded code experimentation.")


@app.command()
def experiment(
    backend: str = typer.Option("mlx", help="Backend name: mlx, mps, ane"),
    profile: str = typer.Option("research", help="Experiment profile"),
    tag: str = typer.Option("baseline", help="Sandbox tag"),
    benchmark_path: str = typer.Option("data/benchmarks/research_mini_v1.jsonl", help="Benchmark JSONL path"),
    description: str = typer.Option("baseline", help="Experiment description"),
    edit_command: Optional[str] = typer.Option(None, help="Optional shell command that edits the sandbox"),
    tokenizer_path: Optional[str] = typer.Option(None, help="Tokenizer path"),
    init_checkpoint: Optional[str] = typer.Option(None, help="Checkpoint to initialize experiment from"),
) -> None:
    """Run one baseline or edited code experiment in an isolated sandbox."""
    orchestrator = AutoresearchOrchestrator(os.getcwd())
    outcome = orchestrator.run_experiment(
        tag=tag,
        backend_name=backend,
        profile=profile,
        description=description,
        benchmark_path=benchmark_path,
        edit_command=edit_command,
        tokenizer_path=tokenizer_path,
        init_checkpoint=init_checkpoint,
    )
    print(
        f"status={outcome.status}\nrun_dir={outcome.run_dir}\n"
        f"val_loss={outcome.val_loss}\nbenchmark_score={outcome.benchmark_score}"
    )


if __name__ == "__main__":
    app()
