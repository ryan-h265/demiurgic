# Training Guide

## Overview

This guide covers infrastructure setup, training procedures, optimization strategies, and budget management for training Demiurgic from scratch.

## Table of Contents

1. [Infrastructure Setup](#infrastructure-setup)
2. [Training Pipeline](#training-pipeline)
3. [Distributed Training](#distributed-training)
4. [Optimization Techniques](#optimization-techniques)
5. [Budget Management](#budget-management)
6. [Monitoring and Debugging](#monitoring-and-debugging)
7. [Checkpointing and Recovery](#checkpointing-and-recovery)

---

## Infrastructure Setup

### Cloud Provider Selection

**AWS (Recommended for flexibility)**
```
Instance Type: p4d.24xlarge (8x A100 80GB)
Cost: ~$32/hour
Storage: S3 for datasets, EFS for checkpoints
Networking: 400 Gbps for multi-node

Best for: Flexible scaling, spot instances

Monthly cost estimate (7B model):
- Compute: $23,040 (30 days * 24 hours * $32)
- Storage: $500-1000 (datasets + checkpoints)
- Data transfer: $200-500
Total: ~$24,000-25,000 for continuous training
```

**GCP (Good for TPU option)**
```
Instance Type: a2-ultragpu-8g (8x A100 80GB)
Cost: ~$30/hour
TPU alternative: TPU v4-128 pods
Storage: Cloud Storage

Best for: TPU training, simpler infrastructure

Monthly cost: ~$22,000-24,000
```

**Azure (Enterprise-friendly)**
```
Instance Type: Standard_ND96asr_v4 (8x A100 80GB)
Cost: ~$28-32/hour

Best for: Enterprise compliance, hybrid setups
```

### Budget-Optimized Setup: Spot Instances

**AWS Spot Instances: 60-70% cost savings**

```bash
# Launch spot instance request
aws ec2 request-spot-instances \
    --instance-count 1 \
    --type "persistent" \
    --spot-price "20.00" \
    --launch-specification file://spec.json

# With proper checkpointing, spot interruptions are manageable
# Expected savings: $32/hr → $12-15/hr
# Monthly savings: ~$15,000-20,000
```

**Spot Instance Strategy:**
1. Checkpoint every 15-30 minutes
2. Use multiple availability zones
3. Automated restart scripts
4. S3 for checkpoint persistence

### Storage Architecture

```
Data Storage Structure:
├── s3://demiurgic-training/
│   ├── datasets/
│   │   ├── raw/                    # Original datasets (1-5TB)
│   │   ├── processed/              # Tokenized data (500GB-2TB)
│   │   └── shards/                 # Training shards (100-500GB)
│   ├── checkpoints/
│   │   ├── step_1000/              # Regular checkpoints
│   │   ├── step_2000/
│   │   └── best/                   # Best validation checkpoints
│   ├── logs/                       # Training logs, metrics
│   └── tokenizer/                  # Tokenizer artifacts

EFS Mount: /mnt/efs/
├── checkpoints/                    # Fast checkpoint writes
└── cache/                          # Temporary data cache
```

### Compute Setup (8x A100 Single Node - 7B Model)

```bash
# Ubuntu 22.04 LTS with CUDA 12.1
# Instance setup script

#!/bin/bash

# Update system
apt-get update && apt-get upgrade -y

# Install CUDA 12.1
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sh cuda_12.1.0_530.30.02_linux.run --silent --toolkit

# Install dependencies
apt-get install -y python3.10 python3-pip git vim tmux htop nvtop

# Python environment
pip3 install --upgrade pip
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Training dependencies
pip3 install transformers accelerate datasets tokenizers
pip3 install deepspeed wandb tensorboard
pip3 install flash-attn --no-build-isolation
pip3 install triton

# AWS CLI for data access
pip3 install awscli boto3

# Configure AWS credentials
aws configure

# Mount EFS
mkdir -p /mnt/efs
mount -t efs fs-xxxxx:/ /mnt/efs
```

---

## Training Pipeline

### Phase 1: Initial Pre-Training (80% of compute)

**Objective:** Learn general code patterns and syntax

```python
# Training configuration for 7B model

{
    "model": {
        "hidden_size": 4096,
        "num_layers": 32,
        "num_heads": 32,
        "context_length": 4096  # Start smaller, expand later
    },

    "training": {
        "total_steps": 70000,        # ~140B tokens
        "batch_size_per_gpu": 4,      # Micro-batch size
        "gradient_accumulation": 8,   # Effective batch = 4 * 8 * 8 = 256
        "learning_rate": 3e-4,
        "warmup_steps": 2000,
        "lr_schedule": "cosine",
        "weight_decay": 0.1,
        "grad_clip": 1.0,

        "checkpoint_interval": 1000,
        "eval_interval": 500,
        "log_interval": 10
    },

    "optimization": {
        "optimizer": "AdamW",
        "beta1": 0.9,
        "beta2": 0.95,
        "epsilon": 1e-8,
        "use_8bit_adam": false,       # Use for >30B models
        "use_fused_adam": true        # 10-20% speedup
    },

    "data": {
        "training_files": "s3://demiurgic-training/datasets/shards/train_*.jsonl",
        "validation_files": "s3://demiurgic-training/datasets/shards/val_*.jsonl",
        "num_workers": 8,
        "prefetch_factor": 2,
        "fim_rate": 0.5,              # 50% of samples use FIM format
        "shuffle_buffer": 10000
    }
}
```

### Phase 2: Extended Context Training (10% of compute)

**Objective:** Extend to 8K context

```python
# After initial training, continue with longer context
{
    "model": {
        "context_length": 8192       # Double context length
    },
    "training": {
        "total_steps": 10000,        # ~20B tokens
        "learning_rate": 1e-4,       # Lower LR for stability
        "batch_size_per_gpu": 2,     # Reduce for memory
        "gradient_accumulation": 16  # Maintain effective batch size
    }
}
```

### Phase 3: Instruction Fine-Tuning (5% of compute)

**Objective:** Align to user instructions and code tasks

```python
{
    "data": {
        "instruction_datasets": [
            "code_instructions.jsonl",    # Code generation tasks
            "code_explanations.jsonl",    # Explanation tasks
            "bug_fixes.jsonl",            # Debugging tasks
            "refactoring.jsonl"           # Code improvement tasks
        ]
    },
    "training": {
        "total_steps": 5000,
        "learning_rate": 1e-5,
        "batch_size_per_gpu": 8
    }
}
```

### Phase 4: Reinforcement Learning (5% of compute - Optional)

**Objective:** Improve code correctness through execution feedback

```python
# RLHF or RLAIF for code correctness
# Reward model based on:
- Syntax correctness (AST parsing)
- Test passage
- Code execution success
- Human preferences
```

---

## Distributed Training

### Single-Node Multi-GPU (7B-13B models)

**DeepSpeed ZeRO Stage 2 (Recommended)**

```python
# deepspeed_config.json

{
    "train_batch_size": 256,
    "train_micro_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": 8,
    "steps_per_print": 10,

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 3e-4,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.1
        }
    },

    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 3e-4,
            "warmup_num_steps": 2000,
            "total_num_steps": 70000
        }
    },

    "fp16": {
        "enabled": false
    },

    "bf16": {
        "enabled": true              # Better for A100s
    },

    "zero_optimization": {
        "stage": 2,                  # Partition optimizer states
        "offload_optimizer": {
            "device": "none"         # Keep on GPU for speed
        },
        "allgather_partitions": true,
        "allgather_bucket_size": 5e8,
        "reduce_scatter": true,
        "reduce_bucket_size": 5e8,
        "overlap_comm": true,
        "contiguous_gradients": true
    },

    "gradient_clipping": 1.0,
    "wall_clock_breakdown": false,

    "flops_profiler": {
        "enabled": false,
        "profile_step": 1,
        "module_depth": -1,
        "top_modules": 3
    }
}
```

**Training Script:**

```python
# train.py

import torch
import deepspeed
from transformers import AutoConfig, AutoModelForCausalLM
from torch.utils.data import DataLoader
import wandb

def main():
    # Initialize distributed
    deepspeed.init_distributed()

    # Load config
    config = AutoConfig.from_pretrained("./configs/7b_model.json")

    # Initialize model
    model = AutoModelForCausalLM.from_config(config)

    # Initialize DeepSpeed
    model_engine, optimizer, train_loader, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        training_data=train_dataset,
        config="deepspeed_config.json"
    )

    # Training loop
    for step, batch in enumerate(train_loader):
        loss = model_engine(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        ).loss

        model_engine.backward(loss)
        model_engine.step()

        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item()}")
            wandb.log({"loss": loss.item(), "step": step})

        if step % 1000 == 0:
            save_checkpoint(model_engine, step)

def save_checkpoint(model_engine, step):
    checkpoint_dir = f"/mnt/efs/checkpoints/step_{step}"
    model_engine.save_checkpoint(checkpoint_dir)
    # Upload to S3
    os.system(f"aws s3 sync {checkpoint_dir} s3://demiurgic-training/checkpoints/step_{step}/")

if __name__ == "__main__":
    main()
```

**Launch Command:**

```bash
deepspeed --num_gpus=8 train.py \
    --deepspeed_config deepspeed_config.json \
    --output_dir /mnt/efs/checkpoints \
    --logging_dir /mnt/efs/logs
```

### Multi-Node Training (30B-70B models)

**DeepSpeed ZeRO Stage 3 + Pipeline Parallelism**

```json
{
    "zero_optimization": {
        "stage": 3,                      // Partition everything
        "offload_optimizer": {
            "device": "cpu",             // Offload to CPU for 70B
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        }
    },

    "pipeline": {
        "pipe_partitioned": true,
        "grad_partitioned": true
    }
}
```

**Multi-Node Launch:**

```bash
# On master node
deepspeed --num_gpus=8 \
    --num_nodes=4 \
    --master_addr=MASTER_IP \
    --master_port=29500 \
    --hostfile=hostfile \
    train.py
```

---

## Optimization Techniques

### Memory Optimization

**1. Gradient Checkpointing**
```python
# Trade compute for memory (30-40% memory savings)
model.gradient_checkpointing_enable()

# For 7B model on 40GB GPUs: Not needed
# For 13B model on 40GB GPUs: Recommended
# For 30B+ models: Essential
```

**2. Mixed Precision Training**
```python
# BF16 on A100 (recommended)
# - Better numerical stability than FP16
# - No loss scaling needed
# - 2x memory savings, 2-3x speedup

from torch.cuda.amp import autocast

with autocast(dtype=torch.bfloat16):
    outputs = model(inputs)
    loss = outputs.loss
```

**3. Flash Attention 2**
```python
# 2-4x faster attention, lower memory
# Must-have for training

from flash_attn import flash_attn_qkvpacked_func

# Automatic in newer transformers:
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    use_flash_attention_2=True
)
```

### Compute Optimization

**1. torch.compile (PyTorch 2.0+)**
```python
# 20-40% speedup for training
import torch

model = torch.compile(
    model,
    mode="max-autotune",
    fullgraph=False
)
```

**2. Fused Kernels**
```python
# Use fused optimizers
from apex.optimizers import FusedAdam

optimizer = FusedAdam(
    model.parameters(),
    lr=3e-4,
    betas=(0.9, 0.95)
)
```

**3. Data Loading**
```python
# Optimize dataloaders
train_loader = DataLoader(
    dataset,
    batch_size=4,
    num_workers=8,          # 1 worker per GPU
    pin_memory=True,        # Faster GPU transfer
    prefetch_factor=2,      # Prefetch batches
    persistent_workers=True # Keep workers alive
)
```

### Training Stability

**1. Learning Rate Warmup**
```python
# Essential for stable training
def get_lr_schedule(step, warmup_steps=2000, max_steps=70000):
    if step < warmup_steps:
        return step / warmup_steps
    else:
        # Cosine decay
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
```

**2. Gradient Clipping**
```python
# Prevent gradient explosions
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**3. Loss Spikes Handling**
```python
# Detect and recover from loss spikes
if loss > prev_loss * 2.0:
    # Load previous checkpoint
    load_checkpoint(f"step_{step - 1000}")
    # Reduce learning rate
    lr *= 0.5
```

---

## Budget Management

### Cost Breakdown (7B Model Example)

```
Total Training Budget: $15,000-20,000

Breakdown:
1. Compute (70% = $10,500-14,000)
   - Training: 720 hours * $15/hr (spot) = $10,800
   - Development/debugging: $1,000-2,000

2. Storage (15% = $2,250-3,000)
   - Dataset storage: $500-1,000
   - Checkpoint storage: $1,000-1,500
   - Logs/artifacts: $200-500

3. Data Transfer (10% = $1,500-2,000)
   - Dataset download: $500-1,000
   - Checkpoint uploads: $500-800
   - Misc: $200-500

4. Miscellaneous (5% = $750-1,000)
   - Monitoring tools: $200-300
   - Validation compute: $300-500
   - Buffer: $250-200
```

### Cost Optimization Strategies

**1. Use Spot Instances (60-70% savings)**
```bash
# Automated spot instance management
while true; do
    # Request spot instance
    INSTANCE_ID=$(aws ec2 request-spot-instances ...)

    # Wait for interruption
    wait_for_interruption

    # Auto-restart from latest checkpoint
    restart_training
done
```

**2. Optimize Checkpoint Strategy**
```python
# Only keep necessary checkpoints
# Delete old checkpoints after validation

def checkpoint_manager(step):
    if step % 1000 == 0:
        save_checkpoint(step)

        # Keep only last 5 checkpoints + best
        cleanup_old_checkpoints(keep_last=5)
```

**3. Progressive Training**
```python
# Start with smaller context, expand later
# Saves 30-40% compute

Phase 1: Train on 2K context (50% of tokens)
Phase 2: Train on 4K context (30% of tokens)
Phase 3: Train on 8K context (20% of tokens)
```

**4. Data Efficiency**
```python
# Use high-quality filtered data
# 100B high-quality tokens > 500B noisy tokens

# Deduplication saves compute
deduplicate_dataset()  # Remove exact/near duplicates
filter_quality()       # Remove low-quality code
```

### Budget Tracking

```python
# Track costs in real-time
import wandb

wandb.init(project="demiurgic-training")

# Log costs per hour
hourly_cost = num_gpus * cost_per_gpu_hour
wandb.log({
    "compute_cost_cumulative": total_hours * hourly_cost,
    "storage_cost": get_s3_costs(),
    "total_cost": compute_cost + storage_cost
})

# Alert if over budget
if total_cost > budget_limit:
    send_alert("Budget exceeded!")
    pause_training()
```

---

## Monitoring and Debugging

### Metrics to Track

**1. Loss Metrics**
```python
metrics = {
    "loss/train": train_loss,
    "loss/validation": val_loss,
    "loss/perplexity": math.exp(val_loss),

    # Per-language losses
    "loss/python": python_loss,
    "loss/javascript": js_loss,
    "loss/rust": rust_loss,
}
```

**2. Training Dynamics**
```python
metrics = {
    "optimization/learning_rate": current_lr,
    "optimization/grad_norm": grad_norm,
    "optimization/param_norm": param_norm,

    "performance/tokens_per_second": tokens_per_sec,
    "performance/mfu": model_flops_utilization,  # Target: 40-50%
    "performance/gpu_memory_allocated": torch.cuda.max_memory_allocated(),
}
```

**3. Data Quality**
```python
metrics = {
    "data/average_sequence_length": avg_seq_len,
    "data/fim_rate": fim_rate,
    "data/language_distribution": lang_dist,
}
```

### Debugging Common Issues

**Issue: OOM (Out of Memory)**
```python
Solutions:
1. Reduce micro-batch size
2. Enable gradient checkpointing
3. Increase gradient accumulation
4. Use ZeRO Stage 3
5. Offload to CPU
```

**Issue: Loss Not Decreasing**
```python
Checklist:
1. Verify data loading (print samples)
2. Check learning rate (may be too low)
3. Verify labels are shifted correctly
4. Check for NaN gradients
5. Validate model initialization
```

**Issue: Loss Spikes**
```python
Solutions:
1. Reduce learning rate
2. Enable gradient clipping (max_norm=1.0)
3. Use BF16 instead of FP16
4. Check for bad data samples
5. Reduce batch size
```

---

## Checkpointing and Recovery

### Checkpoint Strategy

```python
class CheckpointManager:
    def __init__(self, checkpoint_dir, s3_bucket):
        self.checkpoint_dir = checkpoint_dir
        self.s3_bucket = s3_bucket
        self.best_val_loss = float('inf')

    def save_checkpoint(self, model, optimizer, step, val_loss):
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'step': step,
            'val_loss': val_loss,
        }

        # Save regular checkpoint
        if step % 1000 == 0:
            path = f"{self.checkpoint_dir}/step_{step}"
            torch.save(checkpoint, path)
            self.upload_to_s3(path)

        # Save best checkpoint
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_path = f"{self.checkpoint_dir}/best"
            torch.save(checkpoint, best_path)
            self.upload_to_s3(best_path)

    def upload_to_s3(self, path):
        os.system(f"aws s3 sync {path} {self.s3_bucket}/{path}/")

    def load_latest_checkpoint(self):
        # Find latest checkpoint on S3
        # Download and load
        pass
```

### Spot Instance Recovery

```python
# Auto-recovery script
def train_with_recovery():
    while True:
        try:
            # Load latest checkpoint
            checkpoint = load_latest_checkpoint()

            # Resume training
            train(start_step=checkpoint['step'])

        except SpotInterruption:
            print("Spot instance interrupted, restarting...")
            time.sleep(60)  # Wait for new instance
            continue

        except Exception as e:
            print(f"Training error: {e}")
            # Alert and manual intervention
            send_alert(str(e))
            break
```

---

## Training Timeline (7B Model)

```
Week 1: Setup & Validation
- Day 1-2: Infrastructure setup
- Day 3-4: Data pipeline validation
- Day 5-7: Small model (1B) test run

Week 2-4: Main Pre-Training (Phase 1)
- Context length: 4096
- ~100B tokens
- Monitor loss, adjust hyperparameters

Week 5: Extended Context (Phase 2)
- Context length: 8192
- ~20B tokens
- Evaluate on long-context tasks

Week 6: Instruction Fine-Tuning (Phase 3)
- ~20B tokens
- Evaluate on code tasks

Week 7: Final Evaluation & Optimization
- Run benchmarks (HumanEval, MBPP, etc.)
- Quantization, optimization
- CLI tool integration

Total: 6-7 weeks
Cost: $12,000-18,000
```

---

## Next Steps

1. **Setup Infrastructure**: Follow cloud setup instructions
2. **Validate Data Pipeline**: See [data.md](data.md)
3. **Run Small Test**: Train 1B model for 1-2 days
4. **Scale to 7B**: Full training run
5. **Evaluate**: Use benchmarks in [evaluation.md](evaluation.md)

## Additional Resources

- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [HuggingFace Training](https://huggingface.co/docs/transformers/training)
- [PyTorch FSDP](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [Flash Attention](https://github.com/Dao-AILab/flash-attention)
