# Laptop Development Guide

## Yes! You Can Use Your Laptop ✅

Your laptop is **perfect** for:
- ✅ All development and coding
- ✅ Testing the model architecture
- ✅ Running small models (100M-350M params)
- ✅ Validating the training pipeline
- ✅ Experimenting and debugging

## Your System (Test Results)

Based on the test we just ran:

**Hardware:**
- RAM: 30.7 GB total, 15.4 GB available
- CPU: 8 physical cores, 16 logical cores
- GPU: No CUDA (CPU-only mode)

**Verdict:** ✓ Good for development and small model testing

## What Models Can You Run?

### Memory Requirements (Measured)

| Model Size | Parameters | Model Size | Peak RAM | Usable? |
|------------|-----------|------------|----------|---------|
| **100M** | 134M | 512 MB | ~1.1 GB | ✅ Yes, fast |
| **350M** | 369M | 1.4 GB | ~2.0 GB | ✅ Yes, usable |
| **1B** | 1.3B | 5.0 GB | ~5.8 GB | ⚠️ Possible, slow |
| **7B** | ~7B | ~28 GB | ~30 GB+ | ❌ Not on laptop |

### Recommendations by Task

**For Development & Testing:** ✅ Use 100M model
- Fast to load (~1 second)
- Quick forward passes
- Great for debugging
- Fits easily in memory

**For Pipeline Validation:** ✅ Use 350M model
- Still manageable (~2 GB RAM)
- More realistic architecture
- Test full training workflow
- Catch scaling issues early

**For Inference Testing:** ⚠️ Use 1B model (carefully)
- Loads in ~10-15 seconds
- Slow generation
- Good for final validation
- Watch memory usage

## Quick Start on Laptop

### 1. Run the Laptop Test

```bash
source venv/bin/activate
python scripts/test_laptop.py
```

This will:
- Check your system resources
- Test 100M, 350M, and 1B models
- Measure memory usage
- Give specific recommendations

### 2. Try the 100M Model

```bash
source venv/bin/activate
python3
```

```python
from src.model import DemiurgicForCausalLM, get_100m_config
import torch

# Create a laptop-friendly 100M model
config = get_100m_config()
model = DemiurgicForCausalLM(config)

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
# Parameters: 134,105,856

# Test forward pass
input_ids = torch.randint(0, 32000, (1, 32))
outputs = model(input_ids)
print(f"Output shape: {outputs[1].shape}")
# Output shape: torch.Size([1, 32, 32000])

# Generate tokens
generated = model.generate(input_ids[:, :10], max_length=20, do_sample=False)
print(f"Generated {generated.shape[1]} tokens")
# Generated 20 tokens
```

### 3. Test Training Loop (100M Model)

```python
import torch
from src.model import DemiurgicForCausalLM, get_100m_config

# Create model
config = get_100m_config()
model = DemiurgicForCausalLM(config)

# Simple optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# Training step
model.train()
for step in range(10):  # Just 10 steps for testing
    # Random data
    input_ids = torch.randint(0, 32000, (2, 64))  # batch=2, seq_len=64
    labels = input_ids.clone()

    # Forward pass
    outputs = model(input_ids, labels=labels)
    loss = outputs[0]

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Step {step}: Loss = {loss.item():.4f}")

print("Training test complete!")
```

**Expected time on laptop:** ~30-60 seconds for 10 steps

## What You Can Actually Do

### ✅ Development (All Day, Every Day)

**Architecture Development:**
```bash
# Edit model code
vim src/model/attention.py

# Test changes immediately
python scripts/test_laptop.py

# Fast iteration cycle
```

**Testing:**
```bash
# Run tests
pytest tests/test_model.py -v

# Quick validation
python scripts/test_model_basic.py
```

**Experimentation:**
```python
# Try different architectures
config = get_100m_config()
config.num_hidden_layers = 8  # Reduce layers
config.hidden_size = 512       # Smaller hidden size

model = DemiurgicForCausalLM(config)
# Test your idea quickly
```

### ⚠️ Small-Scale Training (Validation Only)

**Training a 100M model on small dataset:**

Feasible but slow:
- Dataset: 1-10 million tokens
- Time: Hours to days
- Purpose: Validate training pipeline
- Not for production models

**Example:**
```python
# This would work but be slow
# Good for testing your pipeline before moving to cloud

epochs = 1
dataset_size = 1000  # Very small dataset
batch_size = 2
steps_per_epoch = dataset_size // batch_size

# Would take ~30 minutes on laptop CPU
# Same code will run 100x faster on cloud GPU
```

### ❌ Production Training (Use Cloud)

**Not practical on laptop:**
- 350M+ models
- Large datasets (100B+ tokens)
- Extended training runs
- 7B+ models

**Solution:** Develop on laptop, train on cloud
1. Code and test locally (fast iteration)
2. Push to GitHub
3. Pull on cloud instance
4. Train with GPUs

## Development Workflow

### Recommended: Laptop-First Development

**Phase 1: Local Development (Your Laptop)**
```bash
# Week 1-2: Architecture & Testing
1. Edit code in your favorite editor
2. Test with 100M-350M models
3. Validate with test suite
4. Debug issues locally
5. Commit to Git
```

**Phase 2: Small Validation (Still Laptop)**
```bash
# Day 1-2: Pipeline Validation
1. Create tiny dataset (1M tokens)
2. Train 100M model for 1 epoch (slow but works)
3. Verify training loop works
4. Test evaluation metrics
5. Ensure everything runs end-to-end
```

**Phase 3: Production Training (Cloud)**
```bash
# Week 3+: Real Training
1. SSH to cloud instance
2. Clone your repo
3. Run same code with 7B model
4. Train on large dataset
5. Download checkpoint to laptop for testing
```

## Optimizations for Laptop

### 1. Reduce Batch Size

```python
# Instead of:
batch_size = 32  # Too big for laptop

# Use:
batch_size = 1   # Or 2, max 4
```

### 2. Reduce Sequence Length

```python
# Instead of:
max_seq_len = 2048  # Full context

# Use:
max_seq_len = 128   # Or 256, 512
```

### 3. Use Smaller Models

```python
# Development: 100M model
config = get_100m_config()

# Testing: 350M model
config = get_350m_config()

# Production: Use cloud for 7B+
```

### 4. Gradient Accumulation (For Training)

```python
# Simulate larger batches without more memory
accumulation_steps = 8
batch_size = 1

for i, batch in enumerate(dataloader):
    loss = model(batch, labels=batch)[0]
    loss = loss / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## Typical Laptop Performance

Based on your system (CPU-only):

### Inference (Forward Pass Only)

| Model | Tokens/sec | Use Case |
|-------|------------|----------|
| 100M | ~50-100 | Development testing |
| 350M | ~20-40 | Pipeline validation |
| 1B | ~5-10 | Final checks |

### Training (Full Forward + Backward)

| Model | Tokens/sec | Practical? |
|-------|------------|------------|
| 100M | ~10-20 | Slow but possible |
| 350M | ~3-5 | Very slow |
| 1B | ~1-2 | Not recommended |

**Note:** With GPU, these would be 100-1000x faster!

## When to Move to Cloud

Move to cloud when you need:

1. **Larger Models:** 7B+ parameters
2. **Real Training:** >10M tokens
3. **Speed:** Hours instead of days
4. **Production:** Actual usable model

## Cloud Options (For Later)

When you're ready to train seriously:

### Free/Cheap Options
- **Google Colab** (Free tier: 12 hours/session)
  - Good for: 1B-7B models, experimentation
  - GPU: T4 (free) or A100 (Colab Pro)
  - Cost: Free or $10/month

- **Kaggle Kernels** (30 hours/week free GPU)
  - Good for: Medium models, competitions
  - GPU: P100 or T4
  - Cost: Free

### Production Options
- **AWS EC2** (p4d.24xlarge: 8x A100)
  - Good for: 7B-70B models
  - Cost: ~$32/hour (~$15/hr with spot)

- **Google Cloud** (Similar pricing)

- **Lambda Labs** (Cheaper GPU cloud)
  - Cost: ~$1-2/hour for single GPU

## Summary

### Your Laptop is Perfect For:
✅ **All development work** - Code, test, debug
✅ **Testing architecture** - Fast iteration
✅ **Small model experiments** - 100M-350M params
✅ **Pipeline validation** - Prove it works before cloud
✅ **Learning** - Understand how models work

### Move to Cloud For:
☁️ **Real training** - 7B+ models
☁️ **Large datasets** - >10M tokens
☁️ **Speed** - 100x faster training
☁️ **Production models** - Usable results

## Quick Commands

```bash
# Test your laptop capabilities
source venv/bin/activate
python scripts/test_laptop.py

# Quick 100M model test
python3 -c "from src.model import get_100m_config, DemiurgicForCausalLM; \
            model = DemiurgicForCausalLM(get_100m_config()); \
            print(f'✓ 100M model loaded successfully')"

# Check memory usage
python3 -c "import psutil; \
            mem = psutil.virtual_memory(); \
            print(f'RAM: {mem.available/1024**3:.1f} GB available')"
```

## Next Steps

1. ✅ **You're set up!** Your laptop is ready for development

2. **Start coding:**
   ```bash
   # Test with 100M model
   python scripts/test_laptop.py

   # Write your first training script
   vim scripts/train_tiny.py
   ```

3. **When ready for serious training:**
   - Set up cloud account (AWS/GCP)
   - Or use Google Colab (free to start)
   - Run same code, just faster!

---

**Bottom line:** Your laptop is **perfect** for everything except final production training. Use it for all development, then move to cloud only when you need serious compute! 🚀
