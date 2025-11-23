# HumanEval Benchmark Guide

Complete guide to evaluating Demiurgic models using the HumanEval benchmark.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Understanding Pass@K](#understanding-passk)
5. [Running Full Evaluation](#running-full-evaluation)
6. [Interpreting Results](#interpreting-results)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

---

## Overview

### What is HumanEval?

HumanEval is the **industry standard benchmark** for evaluating code generation models:

- **164 hand-written Python programming problems**
- Created by OpenAI for evaluating Codex
- Used to benchmark GPT-4, Claude, CodeLlama, StarCoder, etc.
- Tests correctness with unit tests
- Measures Pass@K metrics (success rate)

### Why HumanEval?

1. **Industry Standard**: Everyone uses it for comparison
2. **Objective**: Pass/fail based on unit tests
3. **Comprehensive**: Covers various programming concepts
4. **Proven**: Correlates well with real-world usefulness
5. **Comparable**: Easy to compare with published results

### Pass@K Metric

**Pass@K** = Probability that at least one of K generated solutions is correct

- **Pass@1**: Generate 1 solution → is it correct? (most important)
- **Pass@10**: Generate 10 solutions → is at least 1 correct?
- **Pass@100**: Generate 100 solutions → is at least 1 correct?

Higher K values show the model's ability to generate diverse correct solutions.

---

## Installation

### 1. Install HumanEval Package

```bash
pip install human-eval
```

Or install evaluation requirements:

```bash
pip install -r requirements-eval.txt
```

### 2. Verify Installation

```bash
python -c "from human_eval.data import read_problems; print(f'✓ Loaded {len(read_problems())} problems')"
```

Should output: `✓ Loaded 164 problems`

---

## Quick Start

### Option 1: Using the Script (Easiest)

```bash
# Quick test (10 samples per task, ~5 minutes)
python scripts/run_humaneval.py --model checkpoints/distilled_10m/final --quick

# Full evaluation (200 samples per task, ~2-4 hours)
python scripts/run_humaneval.py --model checkpoints/7b/final --samples 200
```

### Option 2: Using Python API

```python
from src.model import DemiurgicForCausalLM
from src.evaluation import evaluate_model_on_humaneval

# Load model
model = DemiurgicForCausalLM.from_pretrained("checkpoints/7b/final")
tokenizer = load_tokenizer("tokenizer")  # Your tokenizer

# Run evaluation
results = evaluate_model_on_humaneval(
    model=model,
    tokenizer=tokenizer,
    output_dir="humaneval_results",
    num_samples_per_task=200,
    temperature=0.8,
)

print(f"Pass@1: {results['pass@1']:.2%}")
print(f"Pass@10: {results['pass@10']:.2%}")
```

### Option 3: Quick Demo (No Dataset Required)

```bash
# Test with sample problems (no full HumanEval needed)
python examples/test_humaneval_quick.py
```

---

## Understanding Pass@K

### How It Works

For each of 164 problems:

1. **Generate K solutions** (e.g., K=200)
2. **Test each solution** against unit tests
3. **Count how many pass**
4. **Calculate probability** that at least 1 of K would pass

### Formula

```
Pass@K = E[1 - (n-c choose k) / (n choose k)]

Where:
  n = total samples generated
  c = number that passed tests
  k = K value (1, 10, 100, etc.)
```

### Example

Problem: "Write a function to reverse a string"

- Generate 200 solutions
- 42 solutions pass tests
- Probability at least 1 of 10 random solutions passes?

```python
Pass@10 ≈ 1 - (158 choose 10) / (200 choose 10) ≈ 99.8%
```

### Why Multiple K Values?

- **Pass@1**: How good is a single attempt? (Most important for users)
- **Pass@10**: Can the model generate correct solutions with retries?
- **Pass@100**: What's the model's ceiling with many attempts?

---

## Running Full Evaluation

### Step 1: Prepare Your Model

```bash
# Ensure model is trained and saved
ls checkpoints/7b/final/
# Should contain: pytorch_model.bin, config.json
```

### Step 2: Prepare Tokenizer

You need a trained tokenizer. If you don't have one:

```bash
python scripts/train_tokenizer.py --output tokenizer/
```

### Step 3: Run Evaluation

```bash
python scripts/run_humaneval.py \
  --model checkpoints/7b/final \
  --tokenizer tokenizer/ \
  --output-dir humaneval_results_7b \
  --samples 200 \
  --temperature 0.8 \
  --k-values 1 10 100
```

### Step 4: Monitor Progress

The script will:
1. Load model and tokenizer
2. Generate 200 solutions per task (164 tasks = 32,800 generations)
3. Save incrementally to `samples.jsonl` (resume-able!)
4. Execute and test each solution
5. Calculate Pass@K metrics
6. Save results to `results.json`

**Estimated time**:
- **7B model on A100**: ~2-3 hours
- **7B model on V100**: ~4-6 hours
- **7B model on RTX 3090**: ~6-8 hours

### Step 5: Check Results

```bash
cat humaneval_results_7b/results.json
```

Output:
```json
{
  "pass@1": 0.32,
  "pass@10": 0.56,
  "pass@100": 0.72
}
```

---

## Interpreting Results

### Pass@1 Benchmarks (7-15B models)

| Category | Pass@1 Range | Example Models |
|----------|--------------|----------------|
| **Weak** | 0-20% | Base models, no code training |
| **Basic** | 20-28% | Early code models |
| **Competitive** | 28-35% | CodeGen-Mono-16B (29.3%) |
| **Strong** | 35-45% | StarCoder-15B (33.6%), CodeLlama-13B (36.0%) |
| **Excellent** | 45-50% | GPT-3.5-turbo (48.1%) |
| **SOTA** | 50%+ | WizardCoder-15B (57.3%), GPT-4 (67%) |

### What Pass@1 Score Means

**< 20%**: Model struggles with basic programming
- May not understand problem structure
- Likely undertrained on code
- Consider more pre-training or distillation

**20-28%**: Functional but limited
- Understands basic syntax
- Struggles with edge cases
- Needs more training data or better training

**28-35%**: Competitive 7B model ⭐ **Target for Demiurgic**
- Good understanding of Python
- Handles common patterns
- Useful for code completion
- Ready for fine-tuning/deployment

**35-45%**: Strong model
- Excellent code understanding
- Handles complex logic
- Competes with larger models
- Production-ready for most use cases

**45%+**: Top tier
- Near-human performance on simple tasks
- Very few errors
- Likely has advanced training (RLHF, etc.)

### Pass@10 vs Pass@1

**Ideal ratio**: Pass@10 ≈ 1.7-2.0× Pass@1

```python
Pass@1 = 30%, Pass@10 = 52%  # Good (1.73x) - diverse solutions
Pass@1 = 30%, Pass@10 = 38%  # Bad (1.27x) - not diverse enough
Pass@1 = 30%, Pass@10 = 65%  # Good (2.17x) - very diverse
```

If Pass@10 is much higher than Pass@1:
- ✅ Model can generate correct solutions
- ✅ Good diversity in sampling
- 💡 Consider ranking/filtering strategies

If Pass@10 is only slightly higher:
- ❌ Model struggles even with retries
- ❌ Low diversity in generations
- 💡 Increase temperature or use more diverse training

---

## Best Practices

### 1. Generation Settings

**For Maximum Pass@1** (deployment):
```python
temperature = 0.2-0.4  # More conservative
top_p = 0.95
top_k = 50
repetition_penalty = 1.05
```

**For Maximum Pass@10/Pass@100** (diversity):
```python
temperature = 0.8-1.0  # More diverse
top_p = 0.95
top_k = 50-100
repetition_penalty = 1.1
frequency_penalty = 0.2
```

**Recommendation**: Use `temperature=0.8` (balanced)

### 2. Number of Samples

- **Quick test**: 10 samples per task (Pass@1 only)
- **Standard**: 200 samples per task (Pass@1, Pass@10, Pass@100)
- **Research**: 500+ samples per task (more accurate Pass@100)

More samples = more accurate metrics but slower.

### 3. Resuming Interrupted Runs

The evaluation automatically resumes from `samples.jsonl`:

```bash
# Run gets interrupted
python scripts/run_humaneval.py --model checkpoint --samples 200
# ... completes 100 tasks, then crashes ...

# Just run again - it resumes!
python scripts/run_humaneval.py --model checkpoint --samples 200
# ✓ Loaded 100 tasks, generating 64 more...
```

### 4. Multiple Evaluations

Track improvements over training:

```bash
# Evaluate each checkpoint
for checkpoint in checkpoints/step_*; do
    python scripts/run_humaneval.py \
        --model $checkpoint \
        --output-dir humaneval_$(basename $checkpoint) \
        --samples 200
done

# Compare results
grep "pass@1" humaneval_*/results.json
```

### 5. Temperature Tuning

Test different temperatures to find optimal:

```bash
for temp in 0.2 0.4 0.6 0.8 1.0; do
    python scripts/run_humaneval.py \
        --model checkpoint \
        --temperature $temp \
        --output-dir humaneval_temp_$temp \
        --samples 50  # Quick test
done
```

---

## Troubleshooting

### Issue: "human_eval package not installed"

```bash
pip install human-eval
```

If installation fails:
```bash
pip install --upgrade pip
pip install human-eval
```

### Issue: "Evaluation is very slow"

**Solutions**:
1. Use GPU: `--device cuda`
2. Reduce samples: `--samples 50` (less accurate)
3. Use smaller model for testing
4. Enable optimizations in model (Flash Attention, etc.)

### Issue: "Pass@1 is 0% or very low"

**Possible causes**:
1. **Model not trained** - Train on code data first
2. **Tokenizer mismatch** - Ensure tokenizer matches training
3. **Wrong prompt format** - Check generation is completing code, not repeating
4. **Temperature too high** - Try lower temperature (0.2-0.4)

**Debug**:
```python
# Check what model generates
from src.evaluation.humaneval import HumanEvalBenchmark

benchmark = HumanEvalBenchmark(model, tokenizer)
problems = benchmark.load_problems()

# Test one problem
prompt = problems['HumanEval/0']['prompt']
completion = benchmark.generate_completion(prompt)

print("Prompt:", prompt)
print("Completion:", completion)
```

### Issue: "Out of memory"

**Solutions**:
1. Reduce batch size in generation
2. Use gradient checkpointing
3. Use model quantization (8-bit)
4. Use smaller model for testing
5. Clear CUDA cache between tasks

```python
import torch
torch.cuda.empty_cache()
```

### Issue: "Results seem wrong"

**Verify**:
1. Check `samples.jsonl` - are completions reasonable?
2. Manually test a few samples
3. Compare with documented results for other models
4. Ensure HumanEval package is latest version

---

## Advanced Usage

### Custom Generation Settings

```python
from src.evaluation.humaneval import HumanEvalBenchmark

benchmark = HumanEvalBenchmark(
    model=model,
    tokenizer=tokenizer,
    num_samples_per_task=200,
    temperature=0.8,
    top_p=0.95,
    max_new_tokens=512,
)

# Load problems
problems = benchmark.load_problems()

# Generate samples
samples = benchmark.generate_samples(
    problems=problems,
    output_path="custom_samples.jsonl",
    resume=True,
)

# Evaluate
results = benchmark.evaluate_samples(
    samples_path="custom_samples.jsonl",
    k_values=[1, 5, 10, 25, 50, 100],
    timeout=5.0,
    num_workers=8,
)
```

### Filtering Generations

```python
def filter_completion(completion):
    """Custom filtering logic."""
    # Remove completions that are too short
    if len(completion) < 20:
        return None

    # Remove completions with syntax errors
    try:
        compile(completion, '<string>', 'exec')
    except SyntaxError:
        return None

    return completion

# Apply during generation
# (modify HumanEvalBenchmark.generate_completion)
```

### Analyzing Failures

```python
# Load samples and problems
problems = benchmark.load_problems()

with open("samples.jsonl") as f:
    samples = [json.loads(line) for line in f]

# Group by task
from collections import defaultdict
task_samples = defaultdict(list)
for sample in samples:
    task_samples[sample['task_id']].append(sample)

# Find tasks with 0% pass rate
for task_id, samples in task_samples.items():
    # Test each sample
    problem = problems[task_id]
    passed = 0

    for sample in samples:
        full_code = problem['prompt'] + sample['completion']
        if test_code(full_code, problem['test']):
            passed += 1

    if passed == 0:
        print(f"❌ {task_id}: 0/{len(samples)} passed")
        print(f"   Problem: {problem['prompt'][:100]}...")
```

---

## Comparison with Other Benchmarks

### HumanEval vs MBPP

| Aspect | HumanEval | MBPP |
|--------|-----------|------|
| **Size** | 164 problems | 974 problems |
| **Difficulty** | Medium | Easy-Medium |
| **Focus** | Algorithm quality | Basic programming |
| **Usage** | Primary benchmark | Secondary validation |
| **Speed** | Faster | Slower |

**Recommendation**: Start with HumanEval, add MBPP later

### HumanEval vs Real-World

HumanEval is a **proxy** for real-world performance:

- ✅ Good correlation with usefulness
- ✅ Objective and reproducible
- ❌ Only tests Python
- ❌ Only tests standalone functions
- ❌ Doesn't test repo-level understanding

For comprehensive evaluation:
1. HumanEval (quick quality check)
2. MBPP (validation)
3. Custom tasks (your specific use case)
4. Human evaluation (real-world testing)

---

## Summary

### Quick Reference

```bash
# Quick test (5 minutes)
python scripts/run_humaneval.py --model checkpoint --quick

# Full evaluation (2-4 hours)
python scripts/run_humaneval.py --model checkpoint --samples 200

# Check results
cat humaneval_results/results.json
```

### Target Scores (7B model)

- **Pass@1 > 28%**: ✅ Competitive
- **Pass@1 > 35%**: ✅ Strong
- **Pass@1 > 45%**: ✅ Excellent
- **Pass@10 / Pass@1 ≈ 1.7-2.0**: ✅ Good diversity

### Next Steps After HumanEval

1. **If Pass@1 < 25%**: More training needed
2. **If Pass@1 = 25-35%**: Ready for fine-tuning
3. **If Pass@1 > 35%**: Add more benchmarks (MBPP, MultiPL-E)
4. **Always**: Test on your actual use case!

---

## References

- **Paper**: "Evaluating Large Language Models Trained on Code" (Chen et al., 2021)
- **Dataset**: https://github.com/openai/human-eval
- **Leaderboard**: https://paperswithcode.com/sota/code-generation-on-humaneval

---

## Support

If you encounter issues:

1. Check this guide
2. Review `examples/test_humaneval_quick.py`
3. Open an issue with:
   - Model size and checkpoint
   - Command used
   - Error message
   - Sample generations

Good luck with your evaluation! 🚀
