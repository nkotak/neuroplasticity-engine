"""MPS backend adapter with capability-gated command hooks."""

from __future__ import annotations

import os
import subprocess
from typing import Any, Dict

from autoresearch.backends.base import BackendAdapter, require_capability


class MPSBackend(BackendAdapter):
    """Secondary backend for external MPS-based inference or experiments."""

    name = "mps"

    def __init__(self, repo_root: str) -> None:
        self.repo_root = os.path.abspath(repo_root)
        self.train_cmd = os.environ.get("AUTORESEARCH_MPS_TRAIN_CMD")
        self.infer_cmd = os.environ.get("AUTORESEARCH_MPS_INFER_CMD")

    def prepare(self) -> None:
        return None

    def run_research_experiment(self, **kwargs: Any) -> Dict[str, Any]:
        require_capability(self, "training")
        assert self.train_cmd is not None
        completed = subprocess.run(self.train_cmd, shell=True, cwd=self.repo_root, check=True, capture_output=True, text=True)
        return {"stdout": completed.stdout, "stderr": completed.stderr}

    def run_full_train(self, **kwargs: Any) -> Dict[str, Any]:
        require_capability(self, "training")
        assert self.train_cmd is not None
        completed = subprocess.run(self.train_cmd, shell=True, cwd=self.repo_root, check=True, capture_output=True, text=True)
        return {"stdout": completed.stdout, "stderr": completed.stderr}

    def run_inference(self, checkpoint_dir: str, prompt: str, **kwargs: Any) -> str:
        require_capability(self, "inference")
        assert self.infer_cmd is not None
        env = os.environ.copy()
        env["AUTORESEARCH_CHECKPOINT_DIR"] = checkpoint_dir
        env["AUTORESEARCH_PROMPT"] = prompt
        completed = subprocess.run(self.infer_cmd, shell=True, cwd=self.repo_root, check=True, capture_output=True, text=True, env=env)
        return completed.stdout.strip()

    def run_benchmark(self, checkpoint_dir: str, benchmark_path: str, **kwargs: Any) -> Dict[str, Any]:
        from selflearn.benchmark import evaluate_benchmark

        require_capability(self, "inference")
        return evaluate_benchmark(
            checkpoint_dir=checkpoint_dir,
            benchmark_path=benchmark_path,
            tokenizer_path=kwargs.get("tokenizer_path"),
            backend=self,
        )

    def supports_training(self) -> bool:
        return bool(self.train_cmd)

    def supports_inference(self) -> bool:
        return bool(self.infer_cmd)

    def supports_judging(self) -> bool:
        return self.supports_inference()
