# Demiurgic Setup Guide

## Quick Setup (Recommended)

### Linux/Mac

```bash
# 1. Run the setup script
./setup_env.sh

# 2. Activate the environment
source venv/bin/activate

# 3. Test the model
python scripts/test_model_basic.py

# Done! 🎉
```

### Windows

```batch
REM 1. Run the setup script
setup_env.bat

REM 2. Activate the environment
venv\Scripts\activate

REM 3. Test the model
python scripts\test_model_basic.py

REM Done! 🎉
```

## Manual Setup

If you prefer to set up manually:

### 1. Create Virtual Environment

```bash
# Linux/Mac
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 2. Upgrade pip

```bash
pip install --upgrade pip setuptools wheel
```

### 3. Install Dependencies

**Option A: Quick Start (Model Architecture Only)**
```bash
pip install -r requirements-core.txt
```

This installs:
- PyTorch (model framework)
- NumPy (numerical computing)
- pytest (testing)

**Option B: Full Setup (Training Ready)**
```bash
pip install -r requirements.txt
```

This installs everything including:
- Transformers, tokenizers
- Datasets, accelerate
- Weights & Biases (experiment tracking)
- Development tools

**Option C: Staged Installation**
```bash
# Start with core
pip install -r requirements-core.txt

# Add training deps when needed
pip install -r requirements-training.txt

# Add Flash Attention when you have CUDA (optional)
pip install flash-attn --no-build-isolation
```

## Verification

After installation, verify everything works:

```bash
# Activate environment
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows

# Run basic test
python scripts/test_model_basic.py

# Run full test suite (if pytest installed)
pytest tests/test_model.py -v
```

Expected output:
```
✓ All component tests passed!
✓ All tests passed!
✓ All tests completed successfully!
```

## Troubleshooting

### Python Version Issues

**Problem:** "Python 3.10 or higher is required"

**Solution:**
```bash
# Check your Python version
python3 --version

# If < 3.10, install newer Python
# Ubuntu/Debian:
sudo apt install python3.11

# Mac (using Homebrew):
brew install python@3.11

# Then use specific version:
python3.11 -m venv venv
```

### PyTorch Installation Issues

**Problem:** PyTorch not installing or wrong version

**Solution:**
```bash
# For CPU-only (lighter, faster install):
pip install torch --index-url https://download.pytorch.org/whl/cpu

# For CUDA 11.8:
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Flash Attention Installation Issues

**Problem:** Flash Attention fails to install

**Solution:** Flash Attention is optional and only works with CUDA GPUs.

```bash
# Skip it for now - model will use standard attention
# You can add it later when training on GPU

# If you have CUDA and want to try:
pip install flash-attn --no-build-isolation

# If it still fails, the model will automatically fall back
# to standard attention with a warning
```

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'torch'`

**Solution:**
```bash
# Make sure virtual environment is activated!
# You should see (venv) in your prompt

# Linux/Mac:
source venv/bin/activate

# Windows:
venv\Scripts\activate

# Then try again
```

### Testing Errors

**Problem:** Tests fail with shape mismatches or errors

**Solution:**
```bash
# Make sure you're in the project root
cd /path/to/demiurgic

# Make sure venv is activated
source venv/bin/activate

# Try importing directly in Python
python3 -c "from src.model import DemiurgicConfig; print('Success!')"

# If that works, run tests again
python scripts/test_model_basic.py
```

## Environment Management

### Activating the Environment

**Every time** you work on the project, activate the environment first:

```bash
# Linux/Mac
source venv/bin/activate

# Windows
venv\Scripts\activate
```

You'll see `(venv)` in your terminal prompt when activated.

### Deactivating the Environment

When done working:

```bash
deactivate
```

### Deleting the Environment

To start fresh:

```bash
# Linux/Mac
rm -rf venv

# Windows
rmdir /s venv

# Then run setup again
./setup_env.sh  # or setup_env.bat
```

## What Gets Installed

### Core Dependencies (requirements-core.txt)

| Package | Version | Purpose |
|---------|---------|---------|
| torch | ≥2.0.0 | Deep learning framework |
| numpy | ≥1.24.0 | Numerical computing |
| pytest | ≥7.4.0 | Testing framework |

**Size:** ~2-3 GB (depending on CUDA)

### Training Dependencies (requirements-training.txt)

| Package | Version | Purpose |
|---------|---------|---------|
| transformers | ≥4.35.0 | HuggingFace transformers library |
| tokenizers | ≥0.14.0 | Fast tokenization |
| accelerate | ≥0.24.0 | Distributed training |
| datasets | ≥2.14.0 | Dataset loading |
| wandb | ≥0.15.0 | Experiment tracking |
| tensorboard | ≥2.14.0 | Visualization |

**Additional size:** ~1-2 GB

### Optional Dependencies

- **flash-attn** (≥2.0.0): 2-4x training speedup, requires CUDA
- **deepspeed** (≥0.12.0): Multi-GPU training, install when needed

## Next Steps

After successful setup:

1. ✅ **Verify installation** - Run test script
2. 📖 **Read documentation** - Check QUICKSTART.md
3. 🧪 **Experiment** - Try creating models in Python
4. 📋 **Plan training** - Decide on knowledge distillation vs from-scratch

## Quick Reference

```bash
# Create environment
./setup_env.sh                    # Core only
./setup_env.sh --full            # Everything

# Activate
source venv/bin/activate         # Linux/Mac
venv\Scripts\activate            # Windows

# Test
python scripts/test_model_basic.py

# Deactivate
deactivate

# Clean up
rm -rf venv                      # Start fresh
```

## System Requirements

### Minimum (for development/testing)
- Python 3.10+
- 8 GB RAM
- 10 GB disk space
- CPU-only (works fine)

### Recommended (for small model training)
- Python 3.10+
- 16 GB RAM
- 50 GB disk space
- 1x GPU with 16 GB VRAM

### For 7B Model Training
- Python 3.10+
- 64 GB RAM
- 500 GB disk space
- 8x A100 80GB GPUs (or cloud compute)

---

**Need help?** Check QUICKSTART.md or IMPLEMENTATION_STATUS.md for more details!
