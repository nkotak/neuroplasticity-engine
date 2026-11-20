"""Outer-loop orchestration for bounded code research."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from typing import Dict, Optional

from loguru import logger

from autoresearch.backends.ane_backend import ANEBackend
from autoresearch.backends.base import BackendAdapter
from autoresearch.backends.mlx_backend import MLXBackend
from autoresearch.backends.mps_backend import MPSBackend
from autoresearch.promotion import evaluate_promotion_gate
from autoresearch.results import RESULT_COLUMNS, append_tsv
from autoresearch.workspace import WorkspaceManager, validate_program_md


def make_backend(name: str, repo_root: str) -> BackendAdapter:
    if name == "mps":
        return MPSBackend(repo_root)
    if name == "ane":
        return ANEBackend(repo_root)
    return MLXBackend(repo_root)


@dataclass
class ExperimentOutcome:
    status: str
    run_dir: Optional[str]
    val_loss: Optional[float]
    benchmark_score: Optional[float]
    description: str


class AutoresearchOrchestrator:
    """Runs baseline and bounded experiment iterations in isolated workspaces."""

    def __init__(self, repo_root: str) -> None:
        self.repo_root = os.path.abspath(repo_root)
        missing_sections = validate_program_md(os.path.join(self.repo_root, "program.md"))
        if missing_sections:
            raise RuntimeError(f"program.md missing sections: {missing_sections}")
        self.workspace_manager = WorkspaceManager(self.repo_root)

    def run_experiment(
        self,
        tag: str,
        backend_name: str,
        profile: str,
        description: str,
        benchmark_path: str,
        edit_command: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        init_checkpoint: Optional[str] = None,
        baseline_metrics: Optional[Dict[str, float]] = None,
    ) -> ExperimentOutcome:
        workspace = self.workspace_manager.create(tag)
        backend = make_backend(backend_name, workspace.root)
        backend.prepare()

        if edit_command:
            env = os.environ.copy()
            env["AUTORESEARCH_PROGRAM_MD"] = os.path.join(workspace.root, "program.md")
            env["AUTORESEARCH_WORKSPACE"] = workspace.root
            subprocess.run(edit_command, shell=True, cwd=workspace.root, env=env, check=True)

        disallowed = self.workspace_manager.validate_changes(workspace)
        if disallowed:
            raise RuntimeError(f"Experiment touched disallowed paths: {disallowed}")

        run_dir = os.path.join(workspace.root, "runs", f"autoresearch_{tag}")
        try:
            train_result = backend.run_research_experiment(
                run_dir=run_dir,
                profile=profile,
                tokenizer_path=tokenizer_path,
                init_checkpoint=init_checkpoint,
            )
            run_dir = train_result.get("run_dir", run_dir)
            metrics_path = os.path.join(run_dir, "metrics.jsonl")
            val_loss = None
            if os.path.exists(metrics_path):
                with open(metrics_path, "r", encoding="utf-8") as fh:
                    lines = [line for line in fh if line.strip()]
                if lines:
                    val_loss = json.loads(lines[-1]).get("val_loss")
            benchmark_result = backend.run_benchmark(
                checkpoint_dir=os.path.join(run_dir, "best") if os.path.isdir(os.path.join(run_dir, "best")) else os.path.join(run_dir, "final"),
                benchmark_path=benchmark_path,
                tokenizer_path=tokenizer_path,
            )
            benchmark_score = benchmark_result["composite_score"]
            decision = evaluate_promotion_gate(
                baseline_score=None if baseline_metrics is None else baseline_metrics.get("benchmark_score"),
                candidate_score=benchmark_score,
                baseline_buckets=None if baseline_metrics is None else baseline_metrics.get("bucket_scores"),
                candidate_buckets=benchmark_result.get("bucket_scores"),
                baseline_val_loss=None if baseline_metrics is None else baseline_metrics.get("val_loss"),
                candidate_val_loss=val_loss,
            )
            status = decision.status
            append_tsv(
                os.path.join(workspace.root, "results.tsv"),
                RESULT_COLUMNS,
                {
                    "commit": "workspace",
                    "backend": backend_name,
                    "profile": profile,
                    "val_loss": val_loss,
                    "benchmark_score": benchmark_score,
                    "status": status,
                    "description": description if not decision.reasons else f"{description} | {'; '.join(decision.reasons)}",
                },
            )
            return ExperimentOutcome(status, run_dir, val_loss, benchmark_score, description)
        except Exception as exc:
            append_tsv(
                os.path.join(workspace.root, "results.tsv"),
                RESULT_COLUMNS,
                {
                    "commit": "workspace",
                    "backend": backend_name,
                    "profile": profile,
                    "val_loss": "",
                    "benchmark_score": "",
                    "status": "crash",
                    "description": f"{description} | {exc}",
                },
            )
            logger.exception("Autoresearch experiment crashed")
            return ExperimentOutcome("crash", run_dir, None, None, f"{description} | {exc}")
