# Knowledge Distillation - Quick Start

Complete guide to get from zero to trained model in 3 steps.

## Prerequisites

```bash
# Activate environment
source venv/bin/activate

# Verify installation
python scripts/test_distillation_trainer.py
```

## Three Simple Steps

### Step 1: Generate Training Data (~6 hours, $1000-1500)

Get an API key from OpenAI or Anthropic:
- OpenAI: https://platform.openai.com/api-keys
- Anthropic: https://console.anthropic.com/

```bash
# Set your API key
export OPENAI_API_KEY='sk-...'

# Generate 10,000 training examples
python scripts/generate_distillation_data.py \
    --provider openai \
    --model gpt-4-turbo \
    --num-examples 10000 \
    --output-dir data/distillation

# Wait ~3-6 hours
# Output: data/distillation/train.jsonl
```

### Step 2: Train Your Model

**Choose your hardware:**

**A. Laptop (Works on MacBook/CPU)**
```bash
python scripts/train_with_distillation.py \
    --distillation-type output \
    --train-data data/distillation/train.jsonl \
    --student-config configs/model/100m_laptop.json \
    --batch-size 8 \
    --gradient-accumulation-steps 4 \
    --max-steps 50000 \
    --output-dir checkpoints/100m

# Time: ~24-48 hours on laptop
# Result: 100M parameter model
```

**B. Cloud GPU (Better Quality)**
```bash
python scripts/train_with_distillation.py \
    --distillation-type logit \
    --train-data data/distillation/train.jsonl \
    --student-config configs/model/100m_laptop.json \
    --teacher-model codellama/CodeLlama-7b-hf \
    --alpha 0.5 \
    --temperature 2.0 \
    --batch-size 4 \
    --gradient-accumulation-steps 8 \
    --max-steps 50000 \
    --use-wandb \
    --output-dir checkpoints/100m_soft

# Hardware: 16GB GPU (RTX 3090, V100)
# Time: ~24-30 hours
# Result: Higher quality model with soft labels
```

### Step 3: Use Your Model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load your trained model
model = AutoModelForCausalLM.from_pretrained("checkpoints/100m/final")
tokenizer = AutoTokenizer.from_pretrained("checkpoints/100m/final")

# Generate code
prompt = "Write a Python function to calculate fibonacci numbers:"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)

print(tokenizer.decode(outputs[0]))
```

## Quick Test Run

Test everything with a small dataset first:

```bash
# 1. Generate 100 test examples (~$10, 5 minutes)
python scripts/generate_distillation_data.py \
    --num-examples 100 \
    --output-dir data/distillation_test

# 2. Quick training (100 steps, ~10 minutes)
python scripts/train_with_distillation.py \
    --distillation-type output \
    --train-data data/distillation_test/train.jsonl \
    --student-config configs/model/100m_laptop.json \
    --max-steps 100 \
    --output-dir checkpoints/test

# 3. Verify it works
ls checkpoints/test/final/
```

## Cost Summary

**Small Test (100 examples):**
- Data generation: $10-15
- Training: Free (laptop)
- Time: 30 minutes total

**Full Run (10K examples):**
- Data generation: $1,000-1,500
- Training: $0-100 (laptop free, cloud ~$50-100)
- Time: 1-3 days total

**Total: $1,000-1,600** (50-70% cheaper than training from scratch)

## Monitoring Training

**Command line:**
```
Training: 45%|████▌     | 22500/50000 [8:45:23<10:12:45, total_loss=2.34]
```

**Weights & Biases (Cloud):**
Add these flags to any training command:
```bash
--use-wandb \
--wandb-project my-code-model \
--wandb-run-name experiment-1
```

View at: https://wandb.ai/your-username/my-code-model

## Common Issues

**Issue: "Out of memory"**
```bash
# Solution: Reduce batch size
--batch-size 2 --gradient-accumulation-steps 16
```

**Issue: "Loss not decreasing"**
```bash
# Solution: Lower learning rate
--learning-rate 5e-5
```

**Issue: "Training too slow"**
```bash
# Solution: Use cloud GPU
# Or reduce dataset size for laptop
--num-examples 1000
```

## Next Steps After Training

1. **Evaluate:** Test on benchmarks (HumanEval, MBPP)
2. **Iterate:** Identify weak areas, generate more targeted data
3. **Scale up:** Train larger model (350M, 1B) on cloud
4. **Deploy:** Export to ONNX, quantize, deploy to production

## Documentation

- `DISTILLATION_COMPLETE.md` - Full feature documentation
- `DISTILLATION_TRAINING.md` - Detailed training guide
- `DISTILLATION_GUIDE.md` - Data generation guide
- `LAPTOP_GUIDE.md` - Laptop development guide

## Get Help

**Test the system:**
```bash
python scripts/test_distillation_trainer.py
```

**Try the example:**
```bash
python scripts/quick_distillation_example.py
```

**Check documentation:**
```bash
cat DISTILLATION_COMPLETE.md
```

---

**Ready?** Start with Step 1: Generate your training data! 🚀
