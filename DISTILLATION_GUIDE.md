# Knowledge Distillation Setup Guide

## What is Knowledge Distillation?

Knowledge distillation lets you train a **smaller, faster "student" model** by learning from a **larger, more capable "teacher" model** (like GPT-4 or Claude).

### Why Use It?

✅ **Cheaper**: $3K-9K vs $15K-20K (50-70% cost savings)
✅ **Faster**: 2-3 weeks vs 5-7 weeks training time
✅ **Better Quality**: Learn from high-quality teacher outputs
✅ **Less Data**: 20-50B tokens vs 140B tokens needed

### How It Works

```
Traditional Training:
  Raw Data (GitHub) → Train Model → Hope it learns

Knowledge Distillation:
  Prompts → Teacher (GPT-4) → High-Quality Code → Train Student → Better Model
```

## Quick Start

### 1. Try the Example (No API Key Needed)

```bash
source venv/bin/activate
python scripts/quick_distillation_example.py
```

This shows you how the system works without needing API keys.

### 2. Get an API Key

Choose ONE of these:

**Option A: OpenAI (GPT-4)**
- Sign up: https://platform.openai.com/
- Get API key: https://platform.openai.com/api-keys
- Cost: ~$0.10-0.15 per example
- Best for: General code tasks

**Option B: Anthropic (Claude)**
- Sign up: https://console.anthropic.com/
- Get API key from dashboard
- Cost: ~$0.05-0.10 per example (Claude Sonnet)
- Best for: Detailed explanations, safety

**Option C: Local Model (Free but needs GPU)**
- Use LM Studio, Ollama, or vLLM
- Run CodeLlama-70B or similar
- Cost: Free (electricity only)
- Best for: Unlimited generation, privacy

### 3. Set Up Your API Key

```bash
# For OpenAI
export OPENAI_API_KEY='sk-...'

# For Anthropic
export ANTHROPIC_API_KEY='sk-ant-...'

# Or save in .env file (recommended)
echo "OPENAI_API_KEY='sk-...'" >> .env
```

### 4. Generate Small Test Dataset

**Start with just 10 examples** to test everything works:

```bash
source venv/bin/activate

python scripts/generate_distillation_data.py \
    --provider openai \
    --model gpt-4-turbo \
    --num-examples 10 \
    --output-dir data/distillation/test

# Cost: ~$1-2 for 10 examples
```

Expected output:
```
Generating 10 training examples
Teacher: openai/gpt-4-turbo

1. Generating prompts...
   ✓ Generated 10 prompts

2. Calling teacher API...
   Generating: 100%|████████| 10/10
   ✓ Generated 10 examples
   ✓ Total tokens: 15,234
   ✓ Estimated cost: $1.52

3. Filtering for quality...
   ✓ Kept 10/10 examples

4. Saving to data/distillation/test/train.jsonl...
   ✓ Saved 10 examples

Summary
✓ Generated: 10 examples
✓ Estimated cost: $1.52
✓ Saved to: data/distillation/test
```

### 5. Inspect the Data

```bash
# Look at the first example
head -n 1 data/distillation/test/train.jsonl | python -m json.tool
```

You'll see:
```json
{
  "prompt": "Write a Python function that checks if a string is a palindrome",
  "response": "```python\ndef is_palindrome(s):\n    # Remove spaces and convert to lowercase\n    s = s.replace(' ', '').lower()\n    # Check if string equals its reverse\n    return s == s[::-1]\n```\n\n**Explanation:**\nThis function checks if a string is a palindrome by...",
  "category": "function_implementation",
  "language": "python",
  "tokens_used": 187,
  "cost_estimate": 0.015
}
```

## Generating Production Dataset

Once you've tested with 10 examples, scale up:

### Recommended Sizes

| Dataset Size | Examples | Est. Cost | Use Case |
|--------------|----------|-----------|----------|
| **Tiny** | 100 | $10-15 | Pipeline testing |
| **Small** | 1,000 | $100-150 | Initial training |
| **Medium** | 10,000 | $1,000-1,500 | Good model |
| **Large** | 50,000 | $5,000-7,500 | Production model |

### Generate Medium Dataset (Recommended)

```bash
python scripts/generate_distillation_data.py \
    --provider openai \
    --model gpt-4-turbo \
    --num-examples 10000 \
    --output-dir data/distillation/medium \
    --max-concurrent 10 \
    --rate-limit 0.5

# This will take ~3-6 hours
# Cost: ~$1,000-1,500
```

**Tips:**
- Run overnight or on weekend
- Uses checkpoints - can resume if interrupted
- Saves progress every 100 examples
- Shows real-time cost tracking

## Cost Management

### Estimating Costs

```python
# Quick cost calculator
examples = 10000
cost_per_example = 0.12  # Average for GPT-4-turbo

total_cost = examples * cost_per_example
print(f"Estimated cost: ${total_cost:,}")
# Estimated cost: $1,200
```

### Save Money

**1. Use Cheaper Models**
```bash
# GPT-3.5-Turbo (much cheaper)
--model gpt-3.5-turbo  # ~$0.01 per example (10x cheaper!)

# Claude Haiku (cheapest)
--provider anthropic --model claude-3-haiku-20240307  # ~$0.005 per example
```

**2. Mix Models**
```python
# Use cheap model for simple tasks, expensive for complex
if task_complexity == 'simple':
    model = 'gpt-3.5-turbo'  # Cheap
else:
    model = 'gpt-4-turbo'  # Quality
```

**3. Use Local Models (Free)**
```bash
# Set up local model with vLLM or LM Studio
--provider openai-compatible \
--base-url http://localhost:8000 \
--model codellama-70b

# Cost: $0 (free!)
```

## Advanced Options

### Custom Prompt Categories

Generate specific types of examples:

```python
from src.distillation import generate_prompts_from_categories

# Only algorithms and data structures
prompts = generate_prompts_from_categories(
    ['algorithm', 'data_structure'],
    num_prompts=1000
)
```

### Multiple Languages

Focus on specific programming languages:

```python
from src.distillation import PromptGenerator

generator = PromptGenerator()
prompts = generator.generate_prompts(1000)

# Filter for Python only
python_prompts = [p for p in prompts if p['language'] == 'python']
```

### Custom System Prompt

Customize how the teacher responds:

```python
custom_system = """You are an expert Python programmer.
Focus on:
- Clean, pythonic code
- Comprehensive docstrings
- Type hints
- Error handling
"""

# Use in generation
--system-prompt="$custom_system"
```

## Troubleshooting

### API Key Not Working

**Error:** `Invalid API key`

**Solution:**
```bash
# Check your key is set
echo $OPENAI_API_KEY

# If empty, set it:
export OPENAI_API_KEY='your-key-here'

# Or pass directly:
--api-key 'your-key-here'
```

### Rate Limits

**Error:** `Rate limit exceeded`

**Solution:**
```bash
# Slow down requests
--rate-limit 2.0  # 2 seconds between requests
--max-concurrent 3  # Fewer parallel requests
```

### Out of Memory

**Error:** Process killed / OOM

**Solution:** You're probably generating too much at once on laptop.
```bash
# Generate in smaller batches
--num-examples 100  # Instead of 10000

# Run multiple times to build up dataset
```

### Connection Errors

**Error:** `Connection timeout`

**Solution:**
```bash
# Increase timeout
--timeout 120  # 120 seconds

# Check internet connection
ping api.openai.com
```

## Next Steps

After generating your dataset:

### 1. Verify Data Quality

```bash
# Check a few random examples
shuf -n 3 data/distillation/train.jsonl | python -m json.tool | less
```

### 2. Train Your Model

```bash
# Coming soon: Training script
python scripts/train_with_distillation.py \
    --data data/distillation \
    --model-config configs/model/100m_laptop.json \
    --output-dir checkpoints/distilled_100m
```

### 3. Evaluate Results

```bash
# Coming soon: Evaluation script
python scripts/evaluate_model.py \
    --model checkpoints/distilled_100m \
    --benchmark humaneval
```

## File Structure

After generating data, you'll have:

```
data/distillation/
├── train.jsonl              # Training examples
├── train_metadata.json      # Cost, tokens, model info
├── checkpoint_100.jsonl     # Progress checkpoints
├── checkpoint_200.jsonl
└── ...
```

## API Comparison

| Provider | Model | Cost/Example | Speed | Quality | Best For |
|----------|-------|--------------|-------|---------|----------|
| OpenAI | GPT-4 Turbo | $0.10-0.15 | Fast | Excellent | Production |
| OpenAI | GPT-3.5 Turbo | $0.01 | Very Fast | Good | Testing |
| Anthropic | Claude Sonnet | $0.05-0.10 | Fast | Excellent | Explanations |
| Anthropic | Claude Haiku | $0.005 | Very Fast | Good | Large datasets |
| Local | CodeLlama-70B | Free | Varies | Good | Unlimited use |

## Summary

**Quick Workflow:**

```bash
# 1. Try the example (free)
python scripts/quick_distillation_example.py

# 2. Test with 10 examples ($1-2)
python scripts/generate_distillation_data.py --num-examples 10

# 3. Generate production dataset ($100-1500)
python scripts/generate_distillation_data.py --num-examples 10000

# 4. Train your model (coming soon)
python scripts/train_with_distillation.py

# 5. Enjoy your custom code model!
```

**You now have:**
- ✅ Teacher API integration
- ✅ Prompt generation system
- ✅ Data generation pipeline
- ✅ Quality filtering
- ✅ Cost tracking

**Next:** Train your student model on this data!

---

**Questions?** Check the example script or docs/knowledge_distillation.md for more details.
