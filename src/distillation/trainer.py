"""
Knowledge distillation trainer with soft label support.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup, AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import json
from typing import Dict, Optional, Tuple
from tqdm import tqdm
import os
import math

from .config import DistillationConfig
from ..model.model import DemiurgicForCausalLM
from ..model.config import DemiurgicConfig


def get_torch_dtype(dtype_str: str):
    """Convert dtype string to torch dtype."""
    dtype_map = {
        'fp16': torch.float16,
        'float16': torch.float16,
        'bf16': torch.bfloat16,
        'bfloat16': torch.bfloat16,
        'fp32': torch.float32,
        'float32': torch.float32,
    }
    return dtype_map.get(dtype_str, torch.float32)


class DistillationDataset(Dataset):
    """Dataset for knowledge distillation."""

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 2048,
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load data
        self.examples = []
        with open(self.data_path, 'r') as f:
            for line in f:
                if line.strip():
                    self.examples.append(json.loads(line))

        print(f"Loaded {len(self.examples)} examples from {data_path}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]

        # Format as instruction-response pair
        prompt = example['prompt']
        response = example['response']

        # Combine into training format
        # Format: <prompt>\n\n<response>
        text = f"{prompt}\n\n{response}"

        # Tokenize
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )

        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)

        # Create labels (shift is done in the model)
        labels = input_ids.clone()

        # Mask padding tokens in labels
        labels[attention_mask == 0] = -100

        # Also mask the prompt tokens (only train on response)
        prompt_encoded = self.tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors='pt',
        )
        prompt_len = prompt_encoded['input_ids'].shape[1]
        labels[:prompt_len + 2] = -100  # +2 for \n\n

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }


class DistillationTrainer:
    """Trainer for knowledge distillation with soft labels."""

    def __init__(
        self,
        config: DistillationConfig,
        student_model: DemiurgicForCausalLM,
        tokenizer,
        teacher_model: Optional[nn.Module] = None,
    ):
        self.config = config
        self.student_model = student_model
        self.tokenizer = tokenizer
        self.teacher_model = teacher_model

        # Setup device
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        self.student_model.to(self.device)

        if self.teacher_model is not None:
            teacher_device = torch.device(
                config.teacher_model_device if torch.cuda.is_available() else 'cpu'
            )
            self.teacher_model.to(teacher_device)
            self.teacher_model.eval()
            # Freeze teacher
            for param in self.teacher_model.parameters():
                param.requires_grad = False

        # Enable gradient checkpointing if requested
        if config.use_gradient_checkpointing:
            self.student_model.gradient_checkpointing_enable()

        # Setup data
        self.train_dataset = DistillationDataset(
            config.train_data_path,
            tokenizer,
            config.max_seq_length,
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
        )

        if config.val_data_path:
            self.val_dataset = DistillationDataset(
                config.val_data_path,
                tokenizer,
                config.max_seq_length,
            )
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
            )
        else:
            self.val_loader = None

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Setup scheduler
        total_steps = config.max_steps
        if config.max_epochs:
            total_steps = len(self.train_loader) * config.max_epochs

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=total_steps,
        )

        # Setup mixed precision
        self.scaler = None
        if config.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()

        # Tracking
        self.global_step = 0
        self.epoch = 0

        # Create output dir
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.use_wandb = config.use_wandb
        if self.use_wandb:
            import wandb
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config=vars(config),
            )

    def compute_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute distillation loss.

        Args:
            student_logits: [batch_size, seq_len, vocab_size]
            teacher_logits: [batch_size, seq_len, vocab_size] (optional, can be None for output distillation)
            labels: [batch_size, seq_len]

        Returns:
            loss: Combined distillation loss
            metrics: Dict with loss components
        """
        metrics = {}

        # Hard loss (cross-entropy with true labels)
        hard_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )
        metrics['hard_loss'] = hard_loss.item()

        # Soft loss (KL divergence with teacher)
        if teacher_logits is not None and self.config.distillation_type in ['logit', 'hybrid']:
            # Temperature scaling
            T = self.config.temperature

            # Compute soft targets from teacher
            teacher_probs = F.softmax(teacher_logits / T, dim=-1)

            # Compute student log probabilities
            student_log_probs = F.log_softmax(student_logits / T, dim=-1)

            # KL divergence loss (only on non-masked positions)
            mask = (labels != -100).unsqueeze(-1)  # [batch, seq, 1]

            # KL(teacher || student) = sum(teacher * log(teacher/student))
            soft_loss = F.kl_div(
                student_log_probs,
                teacher_probs,
                reduction='none',
            )  # [batch, seq, vocab]

            # Mask and reduce
            soft_loss = (soft_loss * mask).sum() / mask.sum()

            # Scale by temperature^2 (standard practice)
            soft_loss = soft_loss * (T ** 2)

            metrics['soft_loss'] = soft_loss.item()

            # Combine losses
            alpha = self.config.alpha
            total_loss = alpha * hard_loss + (1 - alpha) * soft_loss
            metrics['alpha'] = alpha
        else:
            # Output distillation: only hard loss
            total_loss = hard_loss
            metrics['soft_loss'] = 0.0

        metrics['total_loss'] = total_loss.item()
        return total_loss, metrics

    @torch.no_grad()
    def get_teacher_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Get logits from teacher model."""
        if self.teacher_model is None:
            return None

        # Move to teacher device
        teacher_device = next(self.teacher_model.parameters()).device
        input_ids = input_ids.to(teacher_device)
        attention_mask = attention_mask.to(teacher_device)

        # Forward through teacher
        outputs = self.teacher_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        teacher_logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]

        # Move back to student device
        return teacher_logits.to(self.device)

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Single training step."""
        # Move to device
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)

        # Get teacher logits if needed
        teacher_logits = None
        if self.config.distillation_type in ['logit', 'hybrid']:
            teacher_logits = self.get_teacher_logits(input_ids, attention_mask)

        # Forward through student
        if self.config.use_mixed_precision:
            with torch.cuda.amp.autocast(dtype=get_torch_dtype(self.config.mixed_precision_dtype)):
                outputs = self.student_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                # Handle both tuple and object returns
                student_logits = outputs.logits if hasattr(outputs, 'logits') else outputs[1]

                # Compute loss
                loss, metrics = self.compute_distillation_loss(
                    student_logits,
                    teacher_logits,
                    labels,
                )

            # Backward with scaling
            loss = loss / self.config.gradient_accumulation_steps
            self.scaler.scale(loss).backward()
        else:
            outputs = self.student_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            # Handle both tuple and object returns
            student_logits = outputs.logits if hasattr(outputs, 'logits') else outputs[1]

            # Compute loss
            loss, metrics = self.compute_distillation_loss(
                student_logits,
                teacher_logits,
                labels,
            )

            # Backward
            loss = loss / self.config.gradient_accumulation_steps
            loss.backward()

        return metrics

    def train(self):
        """Main training loop."""
        print(f"\n{'='*60}")
        print(f"Starting Distillation Training")
        print(f"{'='*60}")
        print(f"Distillation type: {self.config.distillation_type}")
        print(f"Student model: {self.config.student_config_path}")
        if self.teacher_model:
            print(f"Teacher model: {self.config.teacher_model_name}")
        print(f"Training examples: {len(self.train_dataset)}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Gradient accumulation: {self.config.gradient_accumulation_steps}")
        print(f"Effective batch size: {self.config.effective_batch_size}")
        print(f"Max steps: {self.config.max_steps}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")

        self.student_model.train()

        # Training loop
        step = 0
        progress_bar = tqdm(total=self.config.max_steps, desc="Training")

        while step < self.config.max_steps:
            self.epoch += 1

            for batch_idx, batch in enumerate(self.train_loader):
                # Training step
                metrics = self.train_step(batch)

                # Gradient accumulation
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    # Clip gradients
                    if self.config.use_mixed_precision:
                        self.scaler.unscale_(self.optimizer)

                    torch.nn.utils.clip_grad_norm_(
                        self.student_model.parameters(),
                        self.config.max_grad_norm,
                    )

                    # Optimizer step
                    if self.config.use_mixed_precision:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    step += 1
                    self.global_step = step

                    # Logging
                    if step % self.config.logging_steps == 0:
                        metrics['learning_rate'] = self.scheduler.get_last_lr()[0]
                        metrics['epoch'] = self.epoch
                        metrics['step'] = step

                        progress_bar.set_postfix(metrics)

                        if self.use_wandb:
                            import wandb
                            wandb.log(metrics, step=step)

                    # Evaluation
                    if step % self.config.eval_steps == 0 and self.val_loader:
                        eval_metrics = self.evaluate()
                        print(f"\nEval at step {step}: {eval_metrics}")
                        if self.use_wandb:
                            import wandb
                            wandb.log({f'eval/{k}': v for k, v in eval_metrics.items()}, step=step)
                        self.student_model.train()

                    # Checkpointing
                    if step % self.config.save_steps == 0:
                        self.save_checkpoint(step)

                    progress_bar.update(1)

                    if step >= self.config.max_steps:
                        break

            if step >= self.config.max_steps:
                break

        progress_bar.close()

        # Final checkpoint
        self.save_checkpoint(step, final=True)

        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Final checkpoint saved to: {self.config.output_dir}")
        print(f"{'='*60}\n")

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set."""
        if self.val_loader is None:
            return {}

        self.student_model.eval()

        total_loss = 0.0
        total_hard_loss = 0.0
        total_soft_loss = 0.0
        num_batches = 0

        for batch in tqdm(self.val_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Get teacher logits if needed
            teacher_logits = None
            if self.config.distillation_type in ['logit', 'hybrid']:
                teacher_logits = self.get_teacher_logits(input_ids, attention_mask)

            # Forward
            outputs = self.student_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            # Handle both tuple and object returns
            student_logits = outputs.logits if hasattr(outputs, 'logits') else outputs[1]

            # Compute loss
            _, metrics = self.compute_distillation_loss(
                student_logits,
                teacher_logits,
                labels,
            )

            total_loss += metrics['total_loss']
            total_hard_loss += metrics['hard_loss']
            total_soft_loss += metrics.get('soft_loss', 0.0)
            num_batches += 1

        return {
            'loss': total_loss / num_batches,
            'hard_loss': total_hard_loss / num_batches,
            'soft_loss': total_soft_loss / num_batches,
            'perplexity': math.exp(total_hard_loss / num_batches),
        }

    def save_checkpoint(self, step: int, final: bool = False):
        """Save model checkpoint."""
        if final:
            checkpoint_dir = Path(self.config.output_dir) / 'final'
        else:
            checkpoint_dir = Path(self.config.output_dir) / f'checkpoint-{step}'

        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        self.student_model.save_pretrained(checkpoint_dir)

        # Save tokenizer
        self.tokenizer.save_pretrained(checkpoint_dir)

        # Save training state
        state = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'config': vars(self.config),
        }

        if self.scaler:
            state['scaler_state'] = self.scaler.state_dict()

        torch.save(state, checkpoint_dir / 'training_state.pt')

        print(f"\nCheckpoint saved to {checkpoint_dir}")

        # Cleanup old checkpoints
        if not final and self.config.save_total_limit > 0:
            checkpoints = sorted(
                Path(self.config.output_dir).glob('checkpoint-*'),
                key=lambda p: int(p.name.split('-')[1]),
            )

            if len(checkpoints) > self.config.save_total_limit:
                for checkpoint in checkpoints[:-self.config.save_total_limit]:
                    import shutil
                    shutil.rmtree(checkpoint)
                    print(f"Removed old checkpoint: {checkpoint}")
