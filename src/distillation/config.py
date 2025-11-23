"""
Configuration for knowledge distillation training.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation training."""

    # Distillation approach
    distillation_type: Literal['output', 'logit', 'hybrid'] = 'output'
    """
    - 'output': Train on teacher's generated text (no teacher model needed at training time)
    - 'logit': Train on teacher's logits/probabilities (requires teacher model during training)
    - 'hybrid': Combine both approaches
    """

    # Loss weighting
    alpha: float = 0.5
    """Weight for hard loss (cross-entropy with true labels). soft_weight = 1 - alpha"""

    temperature: float = 2.0
    """Temperature for softening probability distributions. Higher = softer"""

    # Teacher model (only needed for logit distillation)
    teacher_model_name: Optional[str] = None
    """HuggingFace model name or path to teacher model"""

    teacher_model_device: str = 'cuda'
    """Device for teacher model inference"""

    teacher_dtype: str = 'float16'
    """Data type for teacher model (float32, float16, bfloat16)"""

    # Student model
    student_config_path: str = 'configs/model/100m_laptop.json'
    """Path to student model configuration"""

    # Data
    train_data_path: str = 'data/distillation/train.jsonl'
    """Path to training data (JSONL format from generate_distillation_data.py)"""

    val_data_path: Optional[str] = None
    """Optional validation data path"""

    # Training
    batch_size: int = 8
    """Training batch size"""

    gradient_accumulation_steps: int = 4
    """Gradient accumulation for effective larger batch size"""

    learning_rate: float = 1e-4
    """Peak learning rate"""

    warmup_steps: int = 1000
    """Learning rate warmup steps"""

    max_steps: int = 100000
    """Maximum training steps"""

    max_epochs: Optional[int] = None
    """Maximum epochs (if set, overrides max_steps)"""

    weight_decay: float = 0.01
    """Weight decay for AdamW"""

    max_grad_norm: float = 1.0
    """Gradient clipping threshold"""

    # Sequence length
    max_seq_length: int = 2048
    """Maximum sequence length for training"""

    # Optimization
    use_gradient_checkpointing: bool = True
    """Enable gradient checkpointing to save memory"""

    use_mixed_precision: bool = True
    """Use mixed precision training (fp16/bf16)"""

    mixed_precision_dtype: str = 'bf16'
    """Mixed precision dtype (fp16 or bf16)"""

    # Logging
    logging_steps: int = 10
    """Log metrics every N steps"""

    eval_steps: int = 500
    """Evaluate every N steps"""

    save_steps: int = 1000
    """Save checkpoint every N steps"""

    output_dir: str = 'checkpoints/distilled_model'
    """Directory for saving checkpoints"""

    use_wandb: bool = False
    """Enable Weights & Biases logging"""

    wandb_project: Optional[str] = None
    """W&B project name"""

    wandb_run_name: Optional[str] = None
    """W&B run name"""

    # Checkpointing
    save_total_limit: int = 3
    """Maximum number of checkpoints to keep"""

    resume_from_checkpoint: Optional[str] = None
    """Path to checkpoint to resume from"""

    # Hardware
    device: str = 'cuda'
    """Training device"""

    num_workers: int = 4
    """DataLoader workers"""

    seed: int = 42
    """Random seed"""

    def __post_init__(self):
        """Validate configuration."""
        if self.distillation_type in ['logit', 'hybrid'] and self.teacher_model_name is None:
            raise ValueError(
                f"distillation_type='{self.distillation_type}' requires teacher_model_name to be set"
            )

        if self.alpha < 0 or self.alpha > 1:
            raise ValueError(f"alpha must be between 0 and 1, got {self.alpha}")

        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")

    @property
    def effective_batch_size(self) -> int:
        """Effective batch size including gradient accumulation."""
        return self.batch_size * self.gradient_accumulation_steps


@dataclass
class OutputDistillationConfig(DistillationConfig):
    """Preset for output distillation (text-only, no teacher model needed)."""
    distillation_type: Literal['output'] = 'output'
    alpha: float = 1.0  # Only use hard loss
    teacher_model_name: Optional[str] = None


@dataclass
class LogitDistillationConfig(DistillationConfig):
    """Preset for logit distillation (requires teacher model)."""
    distillation_type: Literal['logit'] = 'logit'
    alpha: float = 0.5  # Balance hard and soft loss
    temperature: float = 2.0
    # teacher_model_name must be set by user


@dataclass
class HybridDistillationConfig(DistillationConfig):
    """Preset for hybrid distillation."""
    distillation_type: Literal['hybrid'] = 'hybrid'
    alpha: float = 0.3  # Favor soft loss
    temperature: float = 2.0
    # teacher_model_name must be set by user
