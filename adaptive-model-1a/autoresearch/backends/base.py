"""Backend abstraction for code research and selflearn loops."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BackendAdapter(ABC):
    """Capability-gated backend interface."""

    name: str

    @abstractmethod
    def prepare(self) -> None:
        """Validate backend prerequisites."""

    @abstractmethod
    def run_research_experiment(self, **kwargs: Any) -> Dict[str, Any]:
        """Run a bounded research experiment."""

    @abstractmethod
    def run_full_train(self, **kwargs: Any) -> Dict[str, Any]:
        """Run a full training job."""

    @abstractmethod
    def run_inference(self, checkpoint_dir: str, prompt: str, **kwargs: Any) -> str:
        """Generate text from a checkpoint."""

    @abstractmethod
    def run_benchmark(self, checkpoint_dir: str, benchmark_path: str, **kwargs: Any) -> Dict[str, Any]:
        """Evaluate a checkpoint on a benchmark."""

    @abstractmethod
    def supports_training(self) -> bool:
        """Whether the backend can train in this environment."""

    @abstractmethod
    def supports_inference(self) -> bool:
        """Whether the backend can run inference in this environment."""

    @abstractmethod
    def supports_judging(self) -> bool:
        """Whether the backend can act as a judge/inference worker."""


def require_capability(backend: BackendAdapter, capability: str) -> None:
    lookup = {
        "training": backend.supports_training,
        "inference": backend.supports_inference,
        "judging": backend.supports_judging,
    }
    if capability not in lookup:
        raise ValueError(f"Unknown backend capability: {capability}")
    if not lookup[capability]():
        raise RuntimeError(f"Backend {backend.name} does not support {capability}")
