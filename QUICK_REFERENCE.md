# Quick Reference - Multi-Provider Data Generation

## Setup (One-Time)

```bash
# Install dependencies
pip install anthropic openai tiktoken aiohttp peft

# Set API keys
export ANTHROPIC_API_KEY='sk-ant-api03-...'
export OPENAI_API_KEY='sk-...'
```

## Common Commands

### Claude 3.5 Sonnet (Best Quality)

```bash
# Small test (100 examples, ~$2-3)
python scripts/generate_distillation_data.py \
    --provider anthropic \
    --model claude-3-5-sonnet-20241022 \
    --num-examples 100

# Production (5000 examples, ~$120-150)
python scripts/generate_distillation_data.py \
    --provider anthropic \
    --model claude-3-5-sonnet-20241022 \
    --num-examples 5000 \
    --output-dir data/claude_production
```

### Claude 3 Haiku (Budget Option)

```bash
# Large dataset (10000 examples, ~$25-50)
python scripts/generate_distillation_data.py \
    --provider anthropic \
    --model claude-3-haiku-20240307 \
    --num-examples 10000 \
    --output-dir data/claude_haiku
```

### GPT-4-turbo

```bash
# Production (5000 examples, ~$120-180)
python scripts/generate_distillation_data.py \
    --provider openai \
    --model gpt-4-turbo \
    --num-examples 5000 \
    --output-dir data/gpt4_production
```

### GPT-3.5-turbo (Budget)

```bash
# Large dataset (10000 examples, ~$50-100)
python scripts/generate_distillation_data.py \
    --provider openai \
    --model gpt-3.5-turbo \
    --num-examples 10000 \
    --output-dir data/gpt35_production
```

### Local ChatGLM3 (Free)

```bash
# Free generation (10000 examples, $0)
python scripts/generate_distillation_data.py \
    --provider local \
    --model-path models/chatglm3-6b.Q4_K_M.gguf \
    --num-examples 10000 \
    --output-dir data/local_production
```

## Customization

### Adjust Temperature

```bash
# More creative/diverse (temperature 0.8-1.0)
--temperature 0.9

# More focused/deterministic (temperature 0.3-0.5)
--temperature 0.4
```

### Longer Responses

```bash
# Allow up to 3000 tokens per response
--max-tokens 3000
```

### More Concurrent Requests

```bash
# 10 parallel requests (faster but may hit rate limits)
--max-concurrent 10

# 3 parallel requests (safer for lower tier API accounts)
--max-concurrent 3
```

### Disable Checkpoints

```bash
# Don't save intermediate checkpoints
--no-checkpoints
```

## Viewing Results

```bash
# View first few examples
head -n 5 data/distillation/train.jsonl

# Count examples
wc -l data/distillation/train.jsonl

# View metadata (cost, tokens, etc.)
cat data/distillation/train_metadata.json

# Pretty print metadata
python -m json.tool data/distillation/train_metadata.json
```

## Cost Estimates (per 1,000 examples)

| Model | Cost | Speed |
|-------|------|-------|
| Claude 3.5 Sonnet | $24-30 | Fast |
| Claude 3 Opus | $90-120 | Medium |
| Claude 3 Haiku | $2.5-5 | Very Fast |
| GPT-4-turbo | $24-36 | Fast |
| GPT-4 | $60-90 | Medium |
| GPT-3.5-turbo | $5-10 | Very Fast |
| Local ChatGLM3 | $0 | Varies |

## Troubleshooting

### API Key Not Found

```bash
# Check if set
echo $ANTHROPIC_API_KEY
echo $OPENAI_API_KEY

# Set if missing
export ANTHROPIC_API_KEY='your-key'

# Or pass directly
python scripts/generate_distillation_data.py \
    --provider anthropic \
    --api-key 'sk-ant-api03-...' \
    --num-examples 100
```

### Rate Limit Errors

```bash
# Reduce concurrent requests
--max-concurrent 3
```

### Out of Memory (Local)

```bash
# Reduce max tokens
--max-tokens 1024
```

## File Locations

- **Generated data:** `data/distillation/train.jsonl`
- **Metadata:** `data/distillation/train_metadata.json`
- **Checkpoints:** `data/distillation/checkpoint_*.jsonl`
- **Provider code:** `src/distillation/providers/`
- **Generation script:** `scripts/generate_distillation_data.py`
- **Filters:** `src/distillation/quality_filters.py`

## Next Steps After Generation

1. Review the data quality
2. Set up cloud training (Google Colab recommended)
3. Train ChatGLM3 with QLoRA
4. Convert trained model to GGUF
5. Deploy locally for inference

## Full Documentation

- **Usage Guide:** `MULTI_PROVIDER_USAGE_GUIDE.md`
- **Implementation Summary:** `IMPLEMENTATION_SUMMARY.md`
- **Original Distillation Guide:** `DISTILLATION_GUIDE.md`
