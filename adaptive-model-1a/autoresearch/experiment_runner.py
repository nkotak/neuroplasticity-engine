"""Training experiment profiles for autoresearch backends."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Optional


@dataclass(frozen=True)
class ExperimentProfile:
    name: str
    total_steps: int
    warmup_steps: int
    eval_every: int
    checkpoint_every: int


RESEARCH_PROFILE = ExperimentProfile(
    name="research",
    total_steps=12000,
    warmup_steps=2000,
    eval_every=250,
    checkpoint_every=2000,
)

PROMOTION_PROFILE = ExperimentProfile(
    name="promotion",
    total_steps=60000,
    warmup_steps=1000,
    eval_every=500,
    checkpoint_every=5000,
)


def get_profile(name: str) -> ExperimentProfile:
    if name == "promotion":
        return PROMOTION_PROFILE
    return RESEARCH_PROFILE


def run_mlx_training_job(
    run_dir: str,
    profile: ExperimentProfile,
    tokenizer_path: Optional[str] = None,
    resume: Optional[str] = None,
    init_checkpoint: Optional[str] = None,
    selflearn_dir: Optional[str] = None,
    total_steps_override: Optional[int] = None,
) -> str:
    """Run one MLX training job with the selected profile."""
    from config import MLXDataConfig, MLXModelConfig, MLXTrainingConfig, PlasticityConfig
    from mlx_train import train

    model_cfg = MLXModelConfig()
    train_cfg = MLXTrainingConfig(
        total_steps=total_steps_override or profile.total_steps,
        warmup_steps=profile.warmup_steps,
        eval_every=profile.eval_every,
        checkpoint_every=profile.checkpoint_every,
    )
    data_cfg = MLXDataConfig()
    if selflearn_dir:
        data_cfg.selflearn_dir = selflearn_dir
    plasticity_cfg = PlasticityConfig()
    os.makedirs(run_dir, exist_ok=True)
    return train(
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        data_cfg=data_cfg,
        plasticity_cfg=plasticity_cfg,
        run_dir=run_dir,
        tokenizer_path=tokenizer_path,
        resume=resume,
        init_checkpoint=init_checkpoint,
    )
