#!/usr/bin/env python3
"""
Train a student model using knowledge distillation.

Supports three approaches:
1. Output distillation: Train on teacher's text responses (no teacher model needed)
2. Logit distillation: Train on teacher's probability distributions (requires teacher model)
3. Hybrid: Combine both approaches
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import argparse
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.distillation.config import (
    DistillationConfig,
    OutputDistillationConfig,
    LogitDistillationConfig,
    HybridDistillationConfig,
)
from src.distillation.trainer import DistillationTrainer
from src.model.model import DemiurgicForCausalLM
from src.model.config import DemiurgicConfig


def load_student_model(config_path: str) -> tuple[DemiurgicForCausalLM, DemiurgicConfig]:
    """Load student model from config."""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    # Remove fields that aren't part of DemiurgicConfig
    # (e.g., model_type is used for HuggingFace compatibility but not needed here)
    config_dict.pop('model_type', None)

    model_config = DemiurgicConfig(**config_dict)
    model = DemiurgicForCausalLM(model_config)

    print(f"\nStudent Model:")
    print(f"  Architecture: {model_config.hidden_size}d, {model_config.num_hidden_layers} layers")
    print(f"  Parameters: {model.num_parameters() / 1e6:.1f}M")
    print(f"  Vocabulary: {model_config.vocab_size}")

    return model, model_config


def load_teacher_model(model_name: str, device: str = 'cuda', dtype: str = 'float16'):
    """Load teacher model from HuggingFace."""
    print(f"\nLoading teacher model: {model_name}")

    torch_dtype = getattr(torch, dtype)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device if torch.cuda.is_available() else 'cpu',
    )

    print(f"  Teacher loaded on {device}")
    return model


def main():
    parser = argparse.ArgumentParser(
        description='Train with knowledge distillation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

1. Output distillation (train on text responses, no teacher model needed):
   python scripts/train_with_distillation.py \\
       --distillation-type output \\
       --train-data data/distillation/train.jsonl \\
       --student-config configs/model/100m_laptop.json \\
       --output-dir checkpoints/distilled_100m

2. Logit distillation (train on soft labels, requires teacher model):
   python scripts/train_with_distillation.py \\
       --distillation-type logit \\
       --train-data data/distillation/train.jsonl \\
       --student-config configs/model/100m_laptop.json \\
       --teacher-model codellama/CodeLlama-7b-hf \\
       --output-dir checkpoints/distilled_100m_soft

3. Hybrid (combine both approaches):
   python scripts/train_with_distillation.py \\
       --distillation-type hybrid \\
       --train-data data/distillation/train.jsonl \\
       --student-config configs/model/350m_laptop.json \\
       --teacher-model codellama/CodeLlama-7b-hf \\
       --alpha 0.3 \\
       --temperature 2.0 \\
       --output-dir checkpoints/distilled_350m_hybrid
        """
    )

    # Distillation type
    parser.add_argument('--distillation-type', type=str, default='output',
                        choices=['output', 'logit', 'hybrid'],
                        help='Type of distillation')

    # Model configs
    parser.add_argument('--student-config', type=str, required=True,
                        help='Path to student model config JSON')
    parser.add_argument('--teacher-model', type=str, default=None,
                        help='HuggingFace model name for teacher (required for logit/hybrid)')

    # Data
    parser.add_argument('--train-data', type=str, required=True,
                        help='Path to training data (JSONL from generate_distillation_data.py)')
    parser.add_argument('--val-data', type=str, default=None,
                        help='Path to validation data')

    # Distillation hyperparameters
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Weight for hard loss (soft_weight = 1 - alpha)')
    parser.add_argument('--temperature', type=float, default=2.0,
                        help='Temperature for soft labels')

    # Training hyperparameters
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Training batch size')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=4,
                        help='Gradient accumulation steps')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--warmup-steps', type=int, default=1000,
                        help='Warmup steps')
    parser.add_argument('--max-steps', type=int, default=100000,
                        help='Maximum training steps')
    parser.add_argument('--max-epochs', type=int, default=None,
                        help='Maximum epochs (overrides max-steps)')
    parser.add_argument('--max-seq-length', type=int, default=2048,
                        help='Maximum sequence length')

    # Optimization
    parser.add_argument('--use-gradient-checkpointing', action='store_true', default=True,
                        help='Enable gradient checkpointing')
    parser.add_argument('--no-gradient-checkpointing', action='store_false', dest='use_gradient_checkpointing',
                        help='Disable gradient checkpointing')
    parser.add_argument('--use-mixed-precision', action='store_true', default=True,
                        help='Enable mixed precision')
    parser.add_argument('--no-mixed-precision', action='store_false', dest='use_mixed_precision',
                        help='Disable mixed precision')
    parser.add_argument('--mixed-precision-dtype', type=str, default='bf16',
                        choices=['fp16', 'bf16'],
                        help='Mixed precision dtype')

    # Logging
    parser.add_argument('--logging-steps', type=int, default=10,
                        help='Log every N steps')
    parser.add_argument('--eval-steps', type=int, default=500,
                        help='Evaluate every N steps')
    parser.add_argument('--save-steps', type=int, default=1000,
                        help='Save checkpoint every N steps')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for checkpoints')

    # W&B logging
    parser.add_argument('--use-wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='demiurgic-distillation',
                        help='W&B project name')
    parser.add_argument('--wandb-run-name', type=str, default=None,
                        help='W&B run name')

    # Hardware
    parser.add_argument('--device', type=str, default='cuda',
                        help='Training device (cuda or cpu)')
    parser.add_argument('--teacher-device', type=str, default='cuda',
                        help='Teacher model device')
    parser.add_argument('--teacher-dtype', type=str, default='float16',
                        help='Teacher model dtype')

    # Checkpointing
    parser.add_argument('--save-total-limit', type=int, default=3,
                        help='Maximum number of checkpoints to keep')
    parser.add_argument('--resume-from-checkpoint', type=str, default=None,
                        help='Resume from checkpoint')

    # Other
    parser.add_argument('--num-workers', type=int, default=4,
                        help='DataLoader workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # Tokenizer
    parser.add_argument('--tokenizer', type=str, default='gpt2',
                        help='Tokenizer name or path (use your trained tokenizer to match the data)')

    args = parser.parse_args()

    # Validate
    if args.distillation_type in ['logit', 'hybrid'] and args.teacher_model is None:
        parser.error(f"--teacher-model is required for distillation-type={args.distillation_type}")

    # Create distillation config
    if args.distillation_type == 'output':
        base_config = OutputDistillationConfig()
    elif args.distillation_type == 'logit':
        base_config = LogitDistillationConfig()
    else:
        base_config = HybridDistillationConfig()

    # Override with command line args
    config = DistillationConfig(
        distillation_type=args.distillation_type,
        alpha=args.alpha,
        temperature=args.temperature,
        teacher_model_name=args.teacher_model,
        teacher_model_device=args.teacher_device,
        teacher_dtype=args.teacher_dtype,
        student_config_path=args.student_config,
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        max_epochs=args.max_epochs,
        max_seq_length=args.max_seq_length,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_mixed_precision=args.use_mixed_precision,
        mixed_precision_dtype=args.mixed_precision_dtype,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        output_dir=args.output_dir,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        device=args.device,
        num_workers=args.num_workers,
        seed=args.seed,
        save_total_limit=args.save_total_limit,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )

    # Set random seed
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # Load tokenizer (path or model name)
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load student model
    print(f"\nLoading student model from {args.student_config}...")
    student_model, student_config = load_student_model(args.student_config)

    # Update vocab size if tokenizer is different
    if student_config.vocab_size != len(tokenizer):
        print(f"  Warning: Config vocab_size ({student_config.vocab_size}) != tokenizer vocab_size ({len(tokenizer)})")
        print(f"  Resizing model embeddings to {len(tokenizer)}")
        student_model.resize_token_embeddings(len(tokenizer))

    # Load teacher model if needed
    teacher_model = None
    if config.distillation_type in ['logit', 'hybrid']:
        teacher_model = load_teacher_model(
            config.teacher_model_name,
            config.teacher_model_device,
            config.teacher_dtype,
        )

    # Create trainer
    print("\nInitializing trainer...")
    trainer = DistillationTrainer(
        config=config,
        student_model=student_model,
        tokenizer=tokenizer,
        teacher_model=teacher_model,
    )

    # Train
    trainer.train()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
