# Demiurgic - ChatGLM3 Coding Assistant

Fine-tune ChatGLM3-6B into a next-level coding assistant using knowledge distillation with Claude and GPT-4.

## What is This?

Demiurgic takes the open-source ChatGLM3-6B model and fine-tunes it with high-quality training data generated from Claude (Anthropic) and GPT-4 (OpenAI). The result: a powerful coding assistant that can write code, use tools, and help with software development.

### Why ChatGLM3?

- **6B parameters** - Perfect size for cloud training and local deployment
- **Open source** - Full control and customization
- **Already trained** - Start with a capable base model
- **Efficient fine-tuning** - QLoRA allows training with 6-8GB VRAM

### Training Approach

Instead of training from scratch (expensive and time-consuming), we use **knowledge distillation**:

1. **Generate training data** using Claude 3.5 Sonnet or GPT-4-turbo
2. **Fine-tune ChatGLM3** with QLoRA in cloud (Google Colab recommended)
3. **Deploy locally** as GGUF for fast CPU/GPU inference

**Cost:** $250-350 for 10,000 high-quality training examples
**Training Time:** 2-6 hours on free/low-cost cloud GPUs
**Result:** Custom coding assistant deployable on your laptop

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Training Data

Choose your teacher model:

**Option A: Claude 3.5 Sonnet (Recommended)**
```bash
export ANTHROPIC_API_KEY='sk-ant-api03-...'

python scripts/generate_distillation_data.py \
    --provider anthropic \
    --model claude-3-5-sonnet-20241022 \
    --num-examples 5000 \
    --output-dir data/training
```
**Cost:** ~$120-150 for 5,000 examples

**Option B: GPT-4-turbo**
```bash
export OPENAI_API_KEY='sk-...'

python scripts/generate_distillation_data.py \
    --provider openai \
    --model gpt-4-turbo \
    --num-examples 5000 \
    --output-dir data/training
```
**Cost:** ~$120-180 for 5,000 examples

**Option C: Local ChatGLM3 (Free)**
```bash
python scripts/generate_distillation_data.py \
    --provider local \
    --model-path models/chatglm3-6b.Q4_K_M.gguf \
    --num-examples 10000 \
    --output-dir data/training
```
**Cost:** $0 (already downloaded!)

### 3. Train ChatGLM3 (Cloud)

Use Google Colab for free GPU access (coming soon - use existing ChatGLM3Trainer for now).

### 4. Deploy Locally

Convert trained model to GGUF and run locally (coming soon).

## Project Structure

```
demiurgic/
├── data/
│   ├── distillation/         # Generated training data
│   └── humaneval/            # Evaluation benchmarks
├── models/
│   └── chatglm3-6b.Q4_K_M.gguf  # Local GGUF model
├── src/
│   ├── distillation/
│   │   ├── providers/        # API clients (Anthropic, OpenAI, Local)
│   │   ├── trainer.py        # ChatGLM3Trainer with QLoRA
│   │   ├── prompt_generator.py
│   │   └── quality_filters.py
│   ├── evaluation/
│   │   └── humaneval.py      # Code evaluation
│   ├── model/
│   │   └── model.py          # ChatGLM3 utilities
│   ├── data/                 # Data loading utilities
│   └── cli/                  # Chat formatting
├── scripts/
│   ├── generate_distillation_data.py  # Multi-provider data generation
│   ├── run_chatglm3_gguf.py           # Local inference
│   └── download_chatglm3_gguf.py      # Download GGUF model
└── docs/                     # Documentation
```

## Documentation

### Getting Started
- **[Multi-Provider Usage Guide](MULTI_PROVIDER_USAGE_GUIDE.md)** - How to generate training data with Claude/GPT-4
- **[Quick Reference](QUICK_REFERENCE.md)** - Command cheat sheet
- **[Implementation Summary](IMPLEMENTATION_SUMMARY.md)** - Current status and next steps

### Technical Guides
- **[Knowledge Distillation](docs/knowledge_distillation.md)** - Theory and methods
- **[Evaluation](docs/evaluation.md)** - HumanEval and benchmarks
- **[CLI Tools](docs/cli.md)** - Command-line utilities

### Setup Guides
- **[Next Steps](NEXT_STEPS.md)** - ChatGLM3 GGUF workflow
- **[HumanEval Setup](HUMANEVAL_SETUP.md)** - Code evaluation setup

## Key Features

✅ **Multi-Provider Data Generation**
- Claude 3.5 Sonnet, Opus, Haiku
- GPT-4-turbo, GPT-3.5-turbo
- Local GGUF models (free)

✅ **Quality Filtering**
- Automatic removal of refusals, short responses
- Duplicate detection
- Code block requirement

✅ **Cost Optimization**
- Mix providers for best cost/quality ratio
- Budget-friendly options (Claude Haiku, GPT-3.5)
- Free local generation

✅ **Cloud Training Support**
- ChatGLM3Trainer with QLoRA
- 6-8GB VRAM requirement (Google Colab compatible)
- Efficient fine-tuning

✅ **Local Deployment Ready**
- GGUF format for fast inference
- CPU and GPU support
- Already have ChatGLM3 GGUF downloaded

## Current Status

### ✅ Completed
- Multi-provider data generation system (Anthropic, OpenAI, Local)
- Quality filtering and duplicate detection
- ChatGLM3Trainer with QLoRA support
- Local GGUF inference
- HumanEval evaluation

### 🚧 In Progress
- Google Colab training notebook
- GGUF conversion pipeline
- Tool system (code execution, file operations)
- Interactive chat interface

See [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) for detailed status.

## Cost Estimates

### Training Data Generation (10,000 examples)

| Provider | Cost | Quality | Speed |
|----------|------|---------|-------|
| Claude 3.5 Sonnet | $240-300 | ⭐⭐⭐⭐⭐ | Fast |
| Claude 3 Haiku | $25-50 | ⭐⭐⭐⭐ | Very Fast |
| GPT-4-turbo | $240-360 | ⭐⭐⭐⭐⭐ | Fast |
| GPT-3.5-turbo | $50-100 | ⭐⭐⭐ | Very Fast |
| Local ChatGLM3 | $0 | ⭐⭐⭐ | Varies |

**Recommended:** Mix 40% Claude + 40% GPT-4 + 20% Local = $250-350

### Cloud Training

- **Google Colab (Free):** $0 with T4 GPU (slower)
- **Colab Pro:** $10/month with better GPUs
- **RunPod:** ~$0.20/hr for RTX 4090 (~$1-2 total)
- **Lambda Labs:** ~$0.50/hr for A100 (~$2-5 total)

**Total Project Cost:** $250-400 (data + training)

## Hardware Requirements

### Data Generation (Local)
- **CPU only** - Works fine for API-based generation
- **GPU optional** - Not needed unless using local model

### Training (Cloud Recommended)
- **Minimum:** 6-8GB VRAM (Google Colab Free)
- **Recommended:** 12-16GB VRAM (faster training)
- **Not suitable for <4GB laptops** - Use cloud instead

### Deployment (Local)
- **CPU:** 8GB+ RAM for GGUF inference
- **GPU:** 4GB+ VRAM for faster inference
- **Already works:** Your downloaded ChatGLM3 GGUF

## Examples

### Generate Test Data
```bash
# Test with 100 examples (~$2-3)
python scripts/generate_distillation_data.py \
    --provider anthropic \
    --model claude-3-5-sonnet-20241022 \
    --num-examples 100 \
    --output-dir data/test
```

### Review Generated Data
```bash
# View first few examples
head -n 5 data/test/train.jsonl

# Check metadata (cost, tokens, quality stats)
cat data/test/train_metadata.json
```

### Run Local Inference
```bash
# Chat with local ChatGLM3 GGUF
python scripts/run_chatglm3_gguf.py "Write a Python function to check if a number is prime"
```

## Philosophy

This project focuses on **simplicity and pragmatism**:

- ✅ Use existing capable models (ChatGLM3) instead of building from scratch
- ✅ Leverage powerful APIs (Claude, GPT-4) for data generation
- ✅ Cloud training for accessibility (anyone with $250-400 can do this)
- ✅ Local deployment for privacy and speed
- ✅ Keep codebase simple and maintainable

## Contributing

This is a research project. Feedback and improvements welcome!

## License

[To be determined]

## Acknowledgments

- **ChatGLM3** by Tsinghua University (THUDM)
- **Anthropic** for Claude API
- **OpenAI** for GPT-4 API
- **llama.cpp** for GGUF inference
