"""MLX backend adapter."""

from __future__ import annotations

import os
from typing import Any, Dict

from autoresearch.backends.base import BackendAdapter


class MLXBackend(BackendAdapter):
    """Primary backend for training, inference, and benchmark runs."""

    name = "mlx"

    def __init__(self, repo_root: str) -> None:
        self.repo_root = os.path.abspath(repo_root)
        self._model_cache: Dict[str, Any] = {}
        self._tokenizer_cache: Dict[str, Any] = {}

    def prepare(self) -> None:
        return None

    @staticmethod
    def _extract_completion(prompt: str, generated: str) -> str:
        """Return only the completion portion from generate() output.

        mlx_inference.generate() returns the full decoded sequence including the
        prompt prefix. Benchmarks and self-learning verifiers expect only the
        generated continuation.
        """
        if generated.startswith(prompt):
            return generated[len(prompt):].lstrip()
        return generated.strip()

    def run_research_experiment(self, **kwargs: Any) -> Dict[str, Any]:
        from autoresearch.experiment_runner import get_profile, run_mlx_training_job

        profile = get_profile(kwargs.get("profile", "research"))
        run_dir = kwargs["run_dir"]
        result_dir = run_mlx_training_job(
            run_dir=run_dir,
            profile=profile,
            tokenizer_path=kwargs.get("tokenizer_path"),
            init_checkpoint=kwargs.get("init_checkpoint"),
            total_steps_override=kwargs.get("total_steps"),
        )
        return {"run_dir": result_dir}

    def run_full_train(self, **kwargs: Any) -> Dict[str, Any]:
        from autoresearch.experiment_runner import get_profile, run_mlx_training_job

        profile = get_profile(kwargs.get("profile", "promotion"))
        run_dir = kwargs["run_dir"]
        result_dir = run_mlx_training_job(
            run_dir=run_dir,
            profile=profile,
            tokenizer_path=kwargs.get("tokenizer_path"),
            init_checkpoint=kwargs.get("init_checkpoint"),
            resume=kwargs.get("resume"),
            selflearn_dir=kwargs.get("selflearn_dir"),
            total_steps_override=kwargs.get("total_steps"),
        )
        return {"run_dir": result_dir}

    def run_inference(self, checkpoint_dir: str, prompt: str, **kwargs: Any) -> str:
        from mlx_inference import generate, load_model
        from mlx_tokenizer import MLXTokenizer

        tok_path = kwargs.get("tokenizer_path") or os.path.join(checkpoint_dir, "tokenizer.json")
        if not os.path.exists(tok_path):
            tok_path = os.path.join(self.repo_root, "runs", "tokenizer_32k.json")

        checkpoint_dir = os.path.abspath(checkpoint_dir)
        tok_path = os.path.abspath(tok_path)

        model = self._model_cache.get(checkpoint_dir)
        if model is None:
            model = load_model(checkpoint_dir)
            self._model_cache[checkpoint_dir] = model

        tokenizer = self._tokenizer_cache.get(tok_path)
        if tokenizer is None:
            tokenizer = MLXTokenizer.load(tok_path)
            self._tokenizer_cache[tok_path] = tokenizer

        generated = generate(
            model,
            tokenizer,
            prompt,
            max_tokens=kwargs.get("max_tokens", 128),
            temperature=kwargs.get("temperature", 0.0),
            top_k=kwargs.get("top_k", 40),
            top_p=kwargs.get("top_p", 0.92),
            repetition_penalty=kwargs.get("repetition_penalty", 1.0),
        )
        return self._extract_completion(prompt, generated)

    def run_benchmark(self, checkpoint_dir: str, benchmark_path: str, **kwargs: Any) -> Dict[str, Any]:
        from selflearn.benchmark import evaluate_benchmark

        tok_path = kwargs.get("tokenizer_path") or os.path.join(checkpoint_dir, "tokenizer.json")
        if not os.path.exists(tok_path):
            tok_path = os.path.join(self.repo_root, "runs", "tokenizer_32k.json")
        return evaluate_benchmark(
            checkpoint_dir=checkpoint_dir,
            benchmark_path=benchmark_path,
            tokenizer_path=tok_path,
            backend=self,
        )

    def supports_training(self) -> bool:
        return True

    def supports_inference(self) -> bool:
        return True

    def supports_judging(self) -> bool:
        return True
