"""
Configuration module for the Neuroplasticity Engine.

Centralizes all model, training, and plasticity configuration into dataclasses
to eliminate magic numbers scattered across the codebase.

Docs: https://docs.python.org/3/library/dataclasses.html

Sample input:
    config = ModelConfig()
    config.d_model  # 256

Expected output:
    ModelConfig(vocab_size=30000, d_model=256, nhead=4, num_layers=4, ...)
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple


@dataclass
class ModelConfig:
    """Transformer model architecture parameters."""
    vocab_size: int = 30000
    d_model: int = 256
    nhead: int = 4
    num_layers: int = 4
    d_ff: int = 1024
    max_seq_len: int = 64
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    batch_size: int = 32
    learning_rate: float = 0.003
    num_epochs: int = 40
    early_stopping_patience: int = 10
    min_epochs_before_stopping: int = 15
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    label_smoothing: float = 0.1


@dataclass
class PlasticityConfig:
    """Structural plasticity controller parameters."""
    prune_threshold: float = 0.005
    growth_rate: float = 0.2
    split_threshold: float = 0.1
    merge_threshold: float = 0.4
    prune_frequency: int = 2
    structural_frequency: int = 4
    min_connections: int = 1000
    activation_growth_ratio: float = 0.7
    growth_noise_scale: float = 0.01
    split_noise_scale: float = 0.1
    merge_similarity_threshold: float = 0.6
    # New: target sparsity for regularization (Change 2)
    target_sparsity: float = 0.4
    # New: dormant neuron reinitialization threshold (Change 5)
    dormant_threshold: float = 0.001
    # New: warmup steps before pruning starts (Change 6)
    warmup_steps: int = 5000
    # New: attention head pruning importance threshold (Change 3C)
    head_prune_threshold: float = 0.1
    head_prune_max_per_layer: int = 1
    # Paper-informed: convergence-triggered splits (Li et al. "Dropin", Algorithm 3)
    convergence_delta: float = 0.03
    convergence_window: int = 6
    max_splits_per_layer: int = 1
    max_split_layers: int = 2
    neurogenesis_count: int = 2
    # Paper-informed: last layer periodic reset (Lyle et al. §4.3)
    last_layer_reset_interval: int = 10000
    # Paper-informed: LRP-based pruning (Li et al. 2025)
    lrp_pruning: bool = True
    # Neuroapoptosis: dormant neuron removal (Li et al. 2025)
    min_age_for_removal: int = 5000
    min_d_ff: int = 64
    # Weight norm clipping
    weight_norm_clip: float = 30.0
    # Scheduled neurogenesis interval (steps between fresh neuron injections)
    neurogenesis_interval: int = 3000
    # Late-training structural freeze (no splits/merges in final phase)
    structural_freeze_fraction: float = 0.15
    structural_freeze_lr: float = 1e-5
    # Merge cooldown: minimum neuron age (training steps) before merge-eligible
    min_age_for_merge: int = 3000
    # Event intensity caps
    max_merges_per_event: int = 2
    max_removed_per_event: int = 1
    max_reinit_per_event: int = 1
    # Neurogenesis trend gate
    neurogenesis_skip_slope_window: int = 4
    neurogenesis_skip_slope_threshold: float = -0.05
    # Structural backoff
    structural_backoff_steps: int = 3000
    structural_backoff_recovery_evals: int = 3
    structural_backoff_recovery_tolerance: float = 0.05
    # Budget hold: pause structural edits when prune/grow stalled but debt rising
    structural_only_hold_window: int = 3


@dataclass
class SkipConnectionConfig:
    """Skip connection controller parameters."""
    correlation_threshold: float = 0.7
    gradient_threshold: float = 0.4
    initial_gate_value: float = 0.01
    warmup_steps: int = 50
    max_skip_connections: int = 6
    activation_window: int = 100
    min_epochs_before_active: int = 20


@dataclass
class MetaLearnerConfig:
    """PPO meta-learner parameters."""
    state_size: int = 10
    action_size: int = 2
    hidden_dim: int = 256
    memory_size: int = 10000
    batch_size: int = 32
    gamma: float = 0.99
    clip_range: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    lambda_adapt: float = 5.0
    update_frequency: int = 10


@dataclass
class DataConfig:
    """Dataset loading configuration."""
    datasets_info: List[Tuple[str, Optional[str]]] = field(default_factory=lambda: [
        ("cognitivecomputations/dolphin-r1", "reasoning-deepseek"),
        ("HuggingFaceH4/ultrachat_200k", "train_sft"),
        ("tatsu-lab/alpaca", None),
        ("databricks/databricks-dolly-15k", None),
    ])
    num_examples_per_dataset: int = 10000
    tokenizer_vocab_size: int = 30000
    tokenizer_max_vocab_size: int = 100000


@dataclass
class InferenceConfig:
    """Inference/generation parameters."""
    temperature: float = 0.7
    top_k: int = 40
    top_p: float = 0.92
    max_length: int = 100
    system_prompt: str = "You are a helpful assistant."


@dataclass
class MultiModalConfig:
    """Multi-modal input configuration for text, image, and audio modalities.

    Produces the Dict[str, Dict] expected by MultiModalEmbedder via to_modality_dict().
    The aux_loss_weight scales the DomainExpertController's load-balancing loss
    (which is already internally scaled by 0.01) before adding to the training loss.
    """
    # Text modality
    text_vocab_size: int = 30000
    text_max_seq_len: int = 512
    text_d_model: int = 256

    # Image modality (Vision Transformer-style patching)
    image_size: int = 256
    image_patch_size: int = 16
    image_d_model: int = 256

    # Audio modality (mel spectrogram processing)
    audio_sample_rate: int = 16000
    audio_segment_ms: int = 1000
    audio_d_model: int = 256

    # Which modalities to enable
    enabled_modalities: List[str] = field(
        default_factory=lambda: ['text', 'image', 'audio']
    )

    # Auxiliary loss coefficient applied in the training loop
    aux_loss_weight: float = 1.0

    def to_modality_dict(self) -> Dict[str, Dict]:
        """Convert to the dict format expected by MultiModalEmbedder."""
        config: Dict[str, Dict] = {}
        if 'text' in self.enabled_modalities:
            config['text'] = {
                'vocab_size': self.text_vocab_size,
                'max_seq_len': self.text_max_seq_len,
                'd_model': self.text_d_model,
            }
        if 'image' in self.enabled_modalities:
            config['image'] = {
                'image_size': self.image_size,
                'patch_size': self.image_patch_size,
                'd_model': self.image_d_model,
            }
        if 'audio' in self.enabled_modalities:
            config['audio'] = {
                'sample_rate': self.audio_sample_rate,
                'segment_ms': self.audio_segment_ms,
                'd_model': self.audio_d_model,
            }
        return config


@dataclass
class MLXModelConfig:
    """MLX transformer model architecture parameters (~14.5M params target).

    Sized for ~25.6M token corpus (1.8 tokens/param ratio).
    d_model=256, nhead=8 gives 32-dim attention heads.
    """
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 8
    d_ff: int = 1024
    vocab_size: int = 32000
    max_seq_len: int = 1024
    dropout: float = 0.1


@dataclass
class MLXTrainingConfig:
    """MLX training hyperparameters with gradient accumulation."""
    lr: float = 3e-4
    warmup_steps: int = 1000
    total_steps: int = 60000
    batch_size: int = 4
    grad_accum_steps: int = 8
    eval_every: int = 500
    log_every: int = 10
    checkpoint_every: int = 5000
    grad_clip_norm: float = 1.0
    weight_decay: float = 0.1
    label_smoothing: float = 0.1
    # Paper-informed: β₂=0.9 adapts to landscape shifts in ~10 steps vs ~1000
    # (Lyle et al. §4.2); ε=1e-6 prevents near-zero denominators after structural changes
    adam_betas: Tuple[float, float] = (0.9, 0.9)
    adam_eps: float = 1e-6


@dataclass
class MLXDataConfig:
    """MLX data pipeline configuration for mixed-source training."""
    book_dir: str = (
        "~/Library/Mobile Documents/com~apple~CloudDocs/Old Files/2016/Books"
    )
    hf_datasets: List[str] = field(default_factory=lambda: [
        "HuggingFaceFW/fineweb-edu",
    ])
    reasoning_jsonl_path: str = "data/reasoning_50k/reasoning_data.jsonl"
    selflearn_dir: str = "data/selflearn/accepted"
    selflearn_max_shards: int = 20
    selflearn_weight: float = 0.10
    mixing_weights: List[float] = field(default_factory=lambda: [
        0.45, 0.25, 0.20, 0.10,  # books, web, reasoning, selflearn
    ])
    seq_len: int = 1024
    # New: fraction of data held out for validation (Change 8)
    val_fraction: float = 0.15
    # New: minimum book chunk size (Change 8b)
    min_book_chars: int = 500
    # New: minimum HuggingFace text length (Change 8c)
    min_hf_chars: int = 200
    # New: minimum FineWeb-edu quality score (Change 8c)
    min_quality_score: float = 3.0


@dataclass
class SelfLearnConfig:
    """Outer-loop verified self-learning configuration."""
    enabled: bool = False
    teacher_backend: str = "openai_compatible"
    teacher_model: str = "qwen2.5-14b-instruct"
    search_backend: str = "none"
    cycle_finetune_steps: int = 5000
    candidate_tasks_per_cycle: int = 2000
    student_samples_per_task: int = 4
    teacher_samples_per_task: int = 1
    accept_min_judge_score: float = 0.85
    max_accept_rate: float = 0.20
    min_accept_rate: float = 0.10
    selflearn_weight: float = 0.10
    parallel_workers: int = 2
    resume: bool = True


if __name__ == "__main__":
    import sys

    all_validation_failures = []
    total_tests = 0

    # Test 1: All configs instantiate with defaults
    total_tests += 1
    try:
        configs = [
            ModelConfig(), TrainingConfig(), PlasticityConfig(),
            SkipConnectionConfig(), MetaLearnerConfig(), DataConfig(),
            InferenceConfig(), MultiModalConfig(),
            MLXModelConfig(), MLXTrainingConfig(), MLXDataConfig(), SelfLearnConfig(),
        ]
        for c in configs:
            assert hasattr(c, '__dataclass_fields__'), f"{type(c).__name__} is not a dataclass"
    except Exception as e:
        all_validation_failures.append(f"Instantiation: {e}")

    # Test 2: ModelConfig has expected defaults
    total_tests += 1
    mc = ModelConfig()
    if mc.d_model != 256 or mc.nhead != 4 or mc.num_layers != 4:
        all_validation_failures.append(
            f"ModelConfig defaults: d_model={mc.d_model}, nhead={mc.nhead}, num_layers={mc.num_layers}"
        )

    # Test 3: PlasticityConfig threshold is reasonable
    total_tests += 1
    pc = PlasticityConfig()
    if pc.prune_threshold >= 1.0 or pc.prune_threshold <= 0.0:
        all_validation_failures.append(f"PlasticityConfig prune_threshold={pc.prune_threshold} out of range")

    # Test 4: Custom overrides work
    total_tests += 1
    custom = ModelConfig(d_model=512, num_layers=8)
    if custom.d_model != 512 or custom.num_layers != 8:
        all_validation_failures.append(f"Custom override failed: d_model={custom.d_model}, num_layers={custom.num_layers}")

    # Test 5: MultiModalConfig produces correct dict
    total_tests += 1
    mm = MultiModalConfig(
        text_vocab_size=500, text_d_model=64,
        image_size=64, image_patch_size=8, image_d_model=64,
        enabled_modalities=['text', 'image']
    )
    mm_dict = mm.to_modality_dict()
    if 'text' not in mm_dict or 'image' not in mm_dict:
        all_validation_failures.append(f"MultiModalConfig dict missing modalities: {list(mm_dict.keys())}")
    if 'audio' in mm_dict:
        all_validation_failures.append("MultiModalConfig dict contains disabled 'audio' modality")
    if mm_dict.get('text', {}).get('vocab_size') != 500:
        all_validation_failures.append(f"MultiModalConfig text vocab_size wrong: {mm_dict.get('text', {}).get('vocab_size')}")

    # Test 6: MLXModelConfig defaults target ~14.5M params
    total_tests += 1
    mlx_mc = MLXModelConfig()
    if mlx_mc.d_model != 256 or mlx_mc.nhead != 8 or mlx_mc.num_layers != 8:
        all_validation_failures.append(
            f"MLXModelConfig defaults: d_model={mlx_mc.d_model}, nhead={mlx_mc.nhead}, "
            f"num_layers={mlx_mc.num_layers}"
        )

    # Test 7: MLXTrainingConfig gradient accumulation effective batch
    total_tests += 1
    mlx_tc = MLXTrainingConfig()
    effective_batch = mlx_tc.batch_size * mlx_tc.grad_accum_steps
    if effective_batch != 32:
        all_validation_failures.append(
            f"MLXTrainingConfig effective batch: expected 32, got {effective_batch}"
        )
    if mlx_tc.total_steps != 60000:
        all_validation_failures.append(
            f"MLXTrainingConfig total_steps: expected 60000, got {mlx_tc.total_steps}"
        )

    # Test 8: MLXDataConfig mixing weights sum to 1.0
    total_tests += 1
    mlx_dc = MLXDataConfig()
    weight_sum = sum(mlx_dc.mixing_weights)
    if abs(weight_sum - 1.0) > 1e-6:
        all_validation_failures.append(
            f"MLXDataConfig mixing weights sum: expected 1.0, got {weight_sum}"
        )

    # Test 9: SelfLearnConfig sane defaults
    total_tests += 1
    slc = SelfLearnConfig()
    if slc.teacher_backend != "openai_compatible" or slc.selflearn_weight <= 0:
        all_validation_failures.append(
            f"SelfLearnConfig defaults invalid: {slc}"
        )

    # Test 10: MLXTrainingConfig Adam tuning fields
    total_tests += 1
    tc = MLXTrainingConfig()
    if tc.adam_betas != (0.9, 0.9):
        all_validation_failures.append(
            f"MLXTrainingConfig adam_betas: expected (0.9, 0.9), got {tc.adam_betas}"
        )
    if tc.adam_eps != 1e-6:
        all_validation_failures.append(
            f"MLXTrainingConfig adam_eps: expected 1e-6, got {tc.adam_eps}"
        )

    # Test 10: PlasticityConfig convergence + last layer + LRP + neuroapoptosis fields
    total_tests += 1
    pc2 = PlasticityConfig()
    if pc2.convergence_delta != 0.03:
        all_validation_failures.append(
            f"PlasticityConfig convergence_delta: expected 0.03, got {pc2.convergence_delta}"
        )
    if pc2.convergence_window != 6:
        all_validation_failures.append(
            f"PlasticityConfig convergence_window: expected 6, got {pc2.convergence_window}"
        )
    if pc2.max_splits_per_layer != 1:
        all_validation_failures.append(
            f"PlasticityConfig max_splits_per_layer: expected 1, got {pc2.max_splits_per_layer}"
        )
    if pc2.last_layer_reset_interval != 10000:
        all_validation_failures.append(
            f"PlasticityConfig last_layer_reset_interval: expected 10000, got {pc2.last_layer_reset_interval}"
        )
    if pc2.lrp_pruning is not True:
        all_validation_failures.append(
            f"PlasticityConfig lrp_pruning: expected True, got {pc2.lrp_pruning}"
        )
    if pc2.min_age_for_removal != 5000:
        all_validation_failures.append(
            f"PlasticityConfig min_age_for_removal: expected 5000, got {pc2.min_age_for_removal}"
        )
    if pc2.weight_norm_clip != 30.0:
        all_validation_failures.append(
            f"PlasticityConfig weight_norm_clip: expected 30.0, got {pc2.weight_norm_clip}"
        )
    if pc2.neurogenesis_interval != 3000:
        all_validation_failures.append(
            f"PlasticityConfig neurogenesis_interval: expected 3000, got {pc2.neurogenesis_interval}"
        )
    if pc2.neurogenesis_count != 2:
        all_validation_failures.append(
            f"PlasticityConfig neurogenesis_count: expected 2, got {pc2.neurogenesis_count}"
        )
    if pc2.structural_freeze_fraction != 0.15:
        all_validation_failures.append(
            f"PlasticityConfig structural_freeze_fraction: expected 0.15, got {pc2.structural_freeze_fraction}"
        )
    if pc2.structural_freeze_lr != 1e-5:
        all_validation_failures.append(
            f"PlasticityConfig structural_freeze_lr: expected 1e-5, got {pc2.structural_freeze_lr}"
        )
    if pc2.min_age_for_merge != 3000:
        all_validation_failures.append(
            f"PlasticityConfig min_age_for_merge: expected 3000, got {pc2.min_age_for_merge}"
        )

    # Test 11: PlasticityConfig new event/backoff/trend fields
    total_tests += 1
    pc3 = PlasticityConfig()
    new_field_checks = [
        ("max_merges_per_event", pc3.max_merges_per_event, 2),
        ("max_removed_per_event", pc3.max_removed_per_event, 1),
        ("max_reinit_per_event", pc3.max_reinit_per_event, 1),
        ("neurogenesis_skip_slope_window", pc3.neurogenesis_skip_slope_window, 4),
        ("neurogenesis_skip_slope_threshold", pc3.neurogenesis_skip_slope_threshold, -0.05),
        ("structural_backoff_steps", pc3.structural_backoff_steps, 3000),
        ("structural_backoff_recovery_evals", pc3.structural_backoff_recovery_evals, 3),
        ("structural_backoff_recovery_tolerance", pc3.structural_backoff_recovery_tolerance, 0.05),
    ]
    for field_name, actual, expected in new_field_checks:
        if actual != expected:
            all_validation_failures.append(
                f"PlasticityConfig {field_name}: expected {expected}, got {actual}"
            )

    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        sys.exit(0)
