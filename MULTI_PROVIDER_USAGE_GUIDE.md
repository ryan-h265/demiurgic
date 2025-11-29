# Multi-Provider Training Data Generation - Usage Guide

This guide explains how to use the new multi-provider system to generate high-quality training data using Claude (Anthropic), GPT-4 (OpenAI), and local models.

## Quick Start

### 1. Install Dependencies

```bash
# Install the new API provider dependencies
pip install anthropic>=0.7.0 openai>=1.0.0 tiktoken>=0.5.0 aiohttp>=3.9.0

# Or install all requirements
pip install -r requirements.txt
```

### 2. Set Up API Keys

```bash
# For Claude (Anthropic)
export ANTHROPIC_API_KEY='sk-ant-api03-...'

# For GPT-4 (OpenAI)
export OPENAI_API_KEY='sk-...'
```

### 3. Generate Training Data

#### Option A: Using Claude 3.5 Sonnet (Recommended for Quality)

```bash
python scripts/generate_distillation_data.py \
    --provider anthropic \
    --model claude-3-5-sonnet-20241022 \
    --num-examples 5000 \
    --output-dir data/claude_training
```

**Cost:** ~$120-150 for 5,000 examples
**Quality:** Excellent reasoning, strong code generation
**Speed:** ~10-15 minutes with 5 concurrent requests

#### Option B: Using GPT-4-turbo

```bash
python scripts/generate_distillation_data.py \
    --provider openai \
    --model gpt-4-turbo \
    --num-examples 5000 \
    --output-dir data/gpt4_training
```

**Cost:** ~$120-180 for 5,000 examples
**Quality:** Very good code generation and refactoring
**Speed:** ~8-12 minutes with 5 concurrent requests

#### Option C: Using Local ChatGLM3 (Free, Already Downloaded)

```bash
python scripts/generate_distillation_data.py \
    --provider local \
    --model-path models/chatglm3-6b.Q4_K_M.gguf \
    --num-examples 10000 \
    --output-dir data/local_training
```

**Cost:** $0 (free!)
**Quality:** Good for basic code tasks
**Speed:** Depends on hardware (slower than APIs)

## Cost-Optimized Strategy: Mixing Providers

For the best quality-to-cost ratio, we recommend generating a mix of data from different providers:

### Recommended Mix for 10,000 Examples

```bash
# 4,000 from Claude (40%) - $120-150
python scripts/generate_distillation_data.py \
    --provider anthropic \
    --model claude-3-5-sonnet-20241022 \
    --num-examples 4000 \
    --output-dir data/mixed/claude

# 4,000 from GPT-4 (40%) - $120-180
python scripts/generate_distillation_data.py \
    --provider openai \
    --model gpt-4-turbo \
    --num-examples 4000 \
    --output-dir data/mixed/gpt4

# 2,000 from local (20%) - $0
python scripts/generate_distillation_data.py \
    --provider local \
    --model-path models/chatglm3-6b.Q4_K_M.gguf \
    --num-examples 2000 \
    --output-dir data/mixed/local

# Then combine them
cat data/mixed/*/train.jsonl > data/final_training/train.jsonl
```

**Total Cost:** ~$250-350 for 10,000 high-quality examples
**Benefits:**
- Claude excels at reasoning and multi-step tasks
- GPT-4 is strong at code generation and refactoring
- Local model adds diversity at no cost

## Advanced Configuration

### Adjusting Generation Parameters

```bash
python scripts/generate_distillation_data.py \
    --provider anthropic \
    --model claude-3-5-sonnet-20241022 \
    --num-examples 1000 \
    --temperature 0.8 \           # Higher = more creative (default: 0.7)
    --max-tokens 3000 \            # Longer responses (default: 2048)
    --max-concurrent 10 \          # More parallel requests (default: 5)
    --output-dir data/custom
```

### Budget-Friendly Options

#### Claude 3 Haiku (Cheapest Quality Option)

```bash
python scripts/generate_distillation_data.py \
    --provider anthropic \
    --model claude-3-haiku-20240307 \
    --num-examples 10000
```

**Cost:** ~$25-50 for 10,000 examples (10x cheaper than Sonnet!)
**Quality:** Still very good, just less sophisticated reasoning

#### GPT-3.5-turbo (Budget OpenAI Option)

```bash
python scripts/generate_distillation_data.py \
    --provider openai \
    --model gpt-3.5-turbo \
    --num-examples 10000
```

**Cost:** ~$50-100 for 10,000 examples
**Quality:** Good for simpler code tasks

## Monitoring and Quality Control

### View Progress

The script provides real-time progress updates:
- Number of prompts generated
- Distribution across categories and languages
- API call progress with progress bar
- Token usage and cost tracking
- Quality filtering statistics

### Checkpoints

By default, the script saves checkpoints every 100 examples. If interrupted, you can continue from a checkpoint.

To disable checkpoints:
```bash
python scripts/generate_distillation_data.py \
    --provider anthropic \
    --num-examples 5000 \
    --no-checkpoints
```

### Quality Filtering

The script automatically filters out:
- ✗ Empty or very short responses (< 50 chars)
- ✗ Responses containing refusals ("I cannot...", "As an AI...")
- ✗ Code tasks without code blocks (```)
- ✗ Duplicate or near-duplicate responses

Typical filtering results: **85-95% pass rate**

## Understanding the Output

### File Structure

After generation, you'll have:

```
data/distillation/
├── train.jsonl                    # Final filtered training data
├── train_metadata.json            # Generation statistics
├── checkpoint_100.jsonl           # Checkpoints (if enabled)
├── checkpoint_200.jsonl
└── ...
```

### Data Format

Each line in `train.jsonl` contains:

```json
{
  "prompt": "Write a Python function to check if a number is prime",
  "response": "Here's a function to check if a number is prime:\n\n```python\ndef is_prime(n):\n    if n <= 1:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n```\n\nThis function...",
  "provider": "anthropic",
  "model": "claude-3-5-sonnet-20241022"
}
```

### Metadata

`train_metadata.json` contains:

```json
{
  "num_examples": 4523,
  "total_tokens": 2847392,
  "total_cost": 127.45,
  "providers": [
    {
      "type": "anthropic",
      "model": "claude-3-5-sonnet-20241022",
      "metrics": {
        "total_tokens": 2847392,
        "input_tokens": 452184,
        "output_tokens": 2395208,
        "total_cost": 127.45,
        "num_requests": 4523,
        "num_errors": 0,
        "avg_cost_per_request": 0.028
      }
    }
  ]
}
```

## Next Steps: Training ChatGLM3

Once you have generated training data, proceed to train ChatGLM3:

### Cloud Training (Recommended for <4GB VRAM)

Use Google Colab for free GPU access:

1. Create a Colab notebook
2. Upload your training data to Google Drive
3. Mount Drive and load data
4. Use ChatGLM3Trainer with QLoRA for efficient training

(Full Colab notebook coming in next implementation phase!)

### Local Training (Requires 6-8GB VRAM)

```bash
python -c "
from src.distillation.trainer import ChatGLM3Trainer, SFTConfig

config = SFTConfig(
    model_name_or_path='THUDM/chatglm3-6b',
    dataset_path='data/distillation/train.jsonl',
    use_qlora=True,
    max_steps=500,
    output_dir='checkpoints/chatglm3_finetuned'
)

trainer = ChatGLM3Trainer(config)
trainer.train()
"
```

## Troubleshooting

### API Key Errors

If you see `ANTHROPIC_API_KEY not set` or `OPENAI_API_KEY not set`:

```bash
# Check if keys are set
echo $ANTHROPIC_API_KEY
echo $OPENAI_API_KEY

# Set them if missing
export ANTHROPIC_API_KEY='your-key-here'
export OPENAI_API_KEY='your-key-here'

# Or pass directly to script
python scripts/generate_distillation_data.py \
    --provider anthropic \
    --api-key 'sk-ant-api03-...' \
    --num-examples 100
```

### Rate Limiting

If you hit rate limits, reduce `--max-concurrent`:

```bash
python scripts/generate_distillation_data.py \
    --provider openai \
    --max-concurrent 3 \    # Slower but safer
    --num-examples 5000
```

### Out of Memory (Local Provider)

If the local model runs out of memory:

```bash
# Reduce context window
python scripts/generate_distillation_data.py \
    --provider local \
    --model-path models/chatglm3-6b.Q4_K_M.gguf \
    --max-tokens 1024 \      # Shorter responses
    --num-examples 1000
```

## Cost Comparison Table

| Provider | Model | Cost per 1K | Cost for 10K | Quality | Speed |
|----------|-------|------------|--------------|---------|-------|
| Anthropic | Claude 3.5 Sonnet | $24-30 | $240-300 | ⭐⭐⭐⭐⭐ | Fast |
| Anthropic | Claude 3 Opus | $90-120 | $900-1200 | ⭐⭐⭐⭐⭐ | Medium |
| Anthropic | Claude 3 Haiku | $2.5-5 | $25-50 | ⭐⭐⭐⭐ | Very Fast |
| OpenAI | GPT-4-turbo | $24-36 | $240-360 | ⭐⭐⭐⭐⭐ | Fast |
| OpenAI | GPT-4 | $60-90 | $600-900 | ⭐⭐⭐⭐⭐ | Medium |
| OpenAI | GPT-3.5-turbo | $5-10 | $50-100 | ⭐⭐⭐ | Very Fast |
| Local | ChatGLM3 GGUF | $0 | $0 | ⭐⭐⭐ | Varies |

## Summary

**For Best Results:**
1. Start with a small test (100 examples) to verify quality
2. Use Claude 3.5 Sonnet or GPT-4-turbo for production
3. Mix providers to optimize cost vs quality
4. Monitor quality filtering statistics
5. Review generated data before training

**Recommended Starting Point:**
```bash
# Test with 100 examples first (~$2-3)
python scripts/generate_distillation_data.py \
    --provider anthropic \
    --model claude-3-5-sonnet-20241022 \
    --num-examples 100 \
    --output-dir data/test

# Review the output
head -n 5 data/test/train.jsonl

# If quality looks good, generate production dataset
python scripts/generate_distillation_data.py \
    --provider anthropic \
    --model claude-3-5-sonnet-20241022 \
    --num-examples 5000 \
    --output-dir data/production
```

Happy training! 🚀
