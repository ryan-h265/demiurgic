# Evaluation and Benchmarking Guide

## Overview

This guide covers evaluation metrics, benchmarks, and testing protocols to measure the performance of your code model throughout training and after deployment.

## Table of Contents

1. [Evaluation Strategy](#evaluation-strategy)
2. [Code Generation Benchmarks](#code-generation-benchmarks)
3. [Code Understanding Benchmarks](#code-understanding-benchmarks)
4. [Custom Evaluation](#custom-evaluation)
5. [Continuous Evaluation](#continuous-evaluation)
6. [Human Evaluation](#human-evaluation)

---

## Evaluation Strategy

### Multi-Dimensional Evaluation

A code model should be evaluated across multiple dimensions:

1. **Correctness**: Does the code work?
2. **Quality**: Is the code well-written?
3. **Understanding**: Can it explain code?
4. **Debugging**: Can it find and fix bugs?
5. **Efficiency**: Is the generated code optimal?

### Evaluation Phases

```
Phase 1: During Training
├── Validation loss every 500 steps
├── Quick benchmarks every 5000 steps
└── Full benchmarks every 10000 steps

Phase 2: Post-Training
├── Comprehensive benchmark suite
├── Multi-language evaluation
├── Human evaluation
└── Real-world task testing

Phase 3: Continuous
├── Monitor production usage
├── A/B testing
└── User feedback collection
```

---

## Code Generation Benchmarks

### 1. HumanEval (Python - Primary Benchmark)

**Description**: 164 hand-written programming problems with unit tests

```python
# Setup
pip install human-eval

# Evaluation script
from human_eval.data import write_jsonl, read_problems
from human_eval.evaluation import evaluate_functional_correctness

def generate_solutions(problems, model, tokenizer):
    """Generate solutions for HumanEval problems"""
    samples = []

    for task_id, problem in problems.items():
        prompt = problem['prompt']

        # Generate completion
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.2,
            top_p=0.95,
            num_return_sequences=1
        )

        completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
        completion = completion[len(prompt):]  # Remove prompt

        samples.append({
            'task_id': task_id,
            'completion': completion
        })

    return samples

# Run evaluation
problems = read_problems()
samples = generate_solutions(problems, model, tokenizer)
write_jsonl("samples.jsonl", samples)

results = evaluate_functional_correctness("samples.jsonl")
print(f"Pass@1: {results['pass@1']:.2%}")
print(f"Pass@10: {results['pass@10']:.2%}")
print(f"Pass@100: {results['pass@100']:.2%}")
```

**Target Scores (7B model):**
- Pass@1: 25-35% (competitive)
- Pass@1: 35-45% (strong)
- Pass@1: 45%+ (state-of-the-art for size)

**Reference Scores:**
- CodeGen-2.5B-mono: 33.4%
- CodeGen-6B-mono: 29.3%
- StarCoder-15B: 33.6%
- GPT-3.5-turbo: 48.1%

### 2. MBPP (Mostly Basic Programming Problems)

**Description**: 974 Python programming problems

```python
# Setup
from datasets import load_dataset

dataset = load_dataset("mbpp")

def evaluate_mbpp(model, tokenizer):
    """Evaluate on MBPP"""
    correct = 0
    total = 0

    for item in dataset['test']:
        prompt = f"# {item['text']}\n{item['code']}"

        # Generate
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=256)
        completion = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Test against test cases
        try:
            exec_globals = {}
            exec(completion, exec_globals)

            # Run test cases
            passed = True
            for test in item['test_list']:
                try:
                    exec(test, exec_globals)
                except:
                    passed = False
                    break

            if passed:
                correct += 1

        except:
            pass

        total += 1

    accuracy = correct / total
    print(f"MBPP Accuracy: {accuracy:.2%}")
    return accuracy

# Run
evaluate_mbpp(model, tokenizer)
```

**Target Scores:**
- 40-50%: Competitive
- 50-60%: Strong
- 60%+: Excellent

### 3. MultiPL-E (Multi-Language Evaluation)

**Description**: HumanEval translated to 18+ languages

```python
# Setup
pip install multipl-e

# Languages to evaluate
languages = [
    'python', 'javascript', 'typescript', 'java',
    'cpp', 'go', 'rust', 'php', 'csharp'
]

def evaluate_multiple(model, tokenizer, languages):
    """Evaluate across multiple languages"""
    results = {}

    for lang in languages:
        print(f"\nEvaluating {lang}...")

        # Load problems for language
        problems = load_multipl_problems(lang)

        # Generate solutions
        samples = generate_solutions(problems, model, tokenizer, lang)

        # Evaluate
        pass_at_k = evaluate_language(samples, lang)

        results[lang] = pass_at_k

    return results

# Run
results = evaluate_multiple(model, tokenizer, languages)

# Display results
print("\n=== MultiPL-E Results ===")
for lang, scores in results.items():
    print(f"{lang:12s}: Pass@1 = {scores['pass@1']:.2%}")
```

### 4. CodeXGLUE Benchmarks

**Description**: Suite of code understanding and generation tasks

```python
# Tasks:
tasks = [
    'code-to-code',      # Translation between languages
    'text-to-code',      # Natural language to code
    'code-to-text',      # Code summarization
    'code-repair',       # Bug fixing
    'clone-detection',   # Detect code clones
    'defect-detection'   # Find bugs
]

# Example: Code Summarization
from datasets import load_dataset

def evaluate_code_summarization(model, tokenizer):
    """Evaluate code summarization (code -> docstring)"""
    dataset = load_dataset("code_x_glue_tc_text_to_code")

    predictions = []
    references = []

    for item in dataset['test']:
        code = item['code']
        reference = item['docstring']

        # Prompt
        prompt = f"# Explain what this code does:\n{code}\n\n# Explanation:"

        # Generate
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=128)
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

        predictions.append(prediction)
        references.append([reference])

    # Calculate BLEU score
    from sacrebleu import corpus_bleu
    bleu = corpus_bleu(predictions, references)

    print(f"Code Summarization BLEU: {bleu.score:.2f}")
    return bleu.score

# Run
evaluate_code_summarization(model, tokenizer)
```

### 5. DS-1000 (Data Science Tasks)

**Description**: 1000 data science coding problems

```python
# Setup
pip install ds1000

def evaluate_ds1000(model, tokenizer):
    """Evaluate on data science tasks"""
    from ds1000 import DS1000Dataset

    dataset = DS1000Dataset()
    results = {}

    libraries = ['Pandas', 'NumPy', 'Matplotlib', 'Scikit-learn']

    for lib in libraries:
        lib_problems = dataset.filter(library=lib)

        correct = 0
        for problem in lib_problems:
            # Generate solution
            completion = generate_completion(model, tokenizer, problem['prompt'])

            # Test
            if test_solution(completion, problem['test']):
                correct += 1

        accuracy = correct / len(lib_problems)
        results[lib] = accuracy

        print(f"{lib}: {accuracy:.2%}")

    return results

# Run
evaluate_ds1000(model, tokenizer)
```

---

## Code Understanding Benchmarks

### 1. Code Explanation Quality

```python
def evaluate_explanation_quality(model, tokenizer, test_samples):
    """
    Evaluate quality of code explanations
    Use GPT-4 as judge
    """
    from openai import OpenAI
    client = OpenAI()

    scores = []

    for sample in test_samples:
        code = sample['code']

        # Generate explanation
        prompt = f"Explain what this code does:\n\n{code}\n\nExplanation:"
        explanation = generate_text(model, tokenizer, prompt)

        # Judge with GPT-4
        judgment = client.chat.completions.create(
            model="gpt-4",
            messages=[{
                "role": "user",
                "content": f"""Rate the following code explanation on a scale of 1-10:

Code:
{code}

Explanation:
{explanation}

Criteria:
- Accuracy (is it correct?)
- Completeness (covers key points?)
- Clarity (easy to understand?)

Provide only a numeric score 1-10."""
            }]
        )

        score = float(judgment.choices[0].message.content.strip())
        scores.append(score)

    avg_score = sum(scores) / len(scores)
    print(f"Average Explanation Quality: {avg_score:.2f}/10")

    return avg_score
```

### 2. Bug Detection

```python
def evaluate_bug_detection(model, tokenizer):
    """Can the model identify bugs?"""

    # Dataset of buggy code with known bugs
    buggy_samples = load_buggy_code_dataset()

    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    for sample in buggy_samples:
        code = sample['code']
        has_bug = sample['has_bug']
        bug_description = sample['bug_description']

        # Ask model to find bugs
        prompt = f"""Review this code for bugs:

{code}

Are there any bugs? If yes, describe them."""

        response = generate_text(model, tokenizer, prompt)

        # Simple heuristic: does response mention a bug?
        model_found_bug = any(word in response.lower() for word in ['bug', 'error', 'issue', 'problem', 'incorrect'])

        if has_bug and model_found_bug:
            true_positives += 1
        elif has_bug and not model_found_bug:
            false_negatives += 1
        elif not has_bug and model_found_bug:
            false_positives += 1
        else:
            true_negatives += 1

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Bug Detection - Precision: {precision:.2%}, Recall: {recall:.2%}, F1: {f1:.2%}")

    return {'precision': precision, 'recall': recall, 'f1': f1}
```

### 3. Code Completion Quality (Real-World)

```python
def evaluate_completion_quality(model, tokenizer):
    """
    Evaluate on real-world completion scenarios
    """

    # Load real code files
    code_files = load_real_code_files()

    metrics = {
        'exact_match': 0,
        'prefix_match': 0,
        'edit_distance': []
    }

    for file in code_files:
        lines = file['content'].split('\n')

        # Test completion at various points
        for i in range(len(lines) - 1):
            context = '\n'.join(lines[:i])
            target = lines[i]

            # Generate completion
            completion = generate_completion(model, tokenizer, context, max_tokens=50)

            # Metrics
            if completion.strip() == target.strip():
                metrics['exact_match'] += 1

            if target.startswith(completion.split('\n')[0]):
                metrics['prefix_match'] += 1

            edit_dist = edit_distance(completion, target)
            metrics['edit_distance'].append(edit_dist)

    # Calculate scores
    total = len(code_files) * avg_lines_per_file
    exact_match_rate = metrics['exact_match'] / total
    prefix_match_rate = metrics['prefix_match'] / total
    avg_edit_distance = sum(metrics['edit_distance']) / len(metrics['edit_distance'])

    print(f"Exact Match: {exact_match_rate:.2%}")
    print(f"Prefix Match: {prefix_match_rate:.2%}")
    print(f"Avg Edit Distance: {avg_edit_distance:.2f}")

    return metrics
```

---

## Custom Evaluation

### Create Custom Test Suite

```python
# custom_eval.py

class CustomCodeEvaluator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def evaluate_all(self):
        """Run all custom evaluations"""

        results = {}

        # 1. Function generation
        results['function_generation'] = self.test_function_generation()

        # 2. Class generation
        results['class_generation'] = self.test_class_generation()

        # 3. Bug fixing
        results['bug_fixing'] = self.test_bug_fixing()

        # 4. Refactoring
        results['refactoring'] = self.test_refactoring()

        # 5. Documentation
        results['documentation'] = self.test_documentation()

        # 6. Test generation
        results['test_generation'] = self.test_test_generation()

        return results

    def test_function_generation(self):
        """Test ability to generate functions"""

        prompts = [
            "def binary_search(arr, target):",
            "def merge_sort(arr):",
            "def fibonacci(n):",
            # ... more prompts
        ]

        passed = 0
        for prompt in prompts:
            code = self.generate(prompt)

            # Test if it works
            if self.test_code(code, prompt):
                passed += 1

        return passed / len(prompts)

    def test_bug_fixing(self):
        """Test ability to fix bugs"""

        buggy_code_samples = [
            {
                'code': "def sum_list(lst):\n    total = 0\n    for i in range(len(lst)):\n        total += i\n    return total",
                'description': "Fix the bug in this function",
                'test': "assert sum_list([1,2,3]) == 6"
            },
            # ... more samples
        ]

        passed = 0
        for sample in buggy_code_samples:
            prompt = f"{sample['description']}\n\n{sample['code']}\n\nFixed code:"
            fixed_code = self.generate(prompt)

            # Test if fixed
            try:
                exec_globals = {}
                exec(fixed_code, exec_globals)
                exec(sample['test'], exec_globals)
                passed += 1
            except:
                pass

        return passed / len(buggy_code_samples)

    def generate(self, prompt, max_tokens=512):
        """Generate completion"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def test_code(self, code, prompt):
        """Test if generated code works"""
        try:
            exec(code)
            return True
        except:
            return False


# Usage
evaluator = CustomCodeEvaluator(model, tokenizer)
results = evaluator.evaluate_all()

for task, score in results.items():
    print(f"{task}: {score:.2%}")
```

---

## Continuous Evaluation

### During Training

```python
# continuous_eval.py

class ContinuousEvaluator:
    def __init__(self, model, tokenizer, eval_interval=5000):
        self.model = model
        self.tokenizer = tokenizer
        self.eval_interval = eval_interval

    def evaluate_checkpoint(self, checkpoint_path, step):
        """Evaluate a training checkpoint"""

        # Load checkpoint
        self.model.load_state_dict(torch.load(checkpoint_path))
        self.model.eval()

        results = {}

        # Quick evaluation (faster benchmarks)
        with torch.no_grad():
            # HumanEval subset (first 50 problems)
            results['humaneval_quick'] = self.eval_humaneval_subset()

            # Validation loss
            results['val_loss'] = self.eval_validation_loss()

            # Per-language perplexity
            results['perplexity'] = self.eval_perplexity_by_language()

        # Log to wandb
        wandb.log({f"eval/{k}": v for k, v in results.items()}, step=step)

        # Save results
        with open(f"eval_results_step_{step}.json", 'w') as f:
            json.dump(results, f, indent=2)

        return results

    def eval_humaneval_subset(self):
        """Quick HumanEval eval on first 50 problems"""
        # Implementation
        pass

    def eval_validation_loss(self):
        """Calculate validation loss"""
        # Implementation
        pass

    def eval_perplexity_by_language(self):
        """Calculate perplexity for each language"""
        perplexities = {}

        for lang in ['python', 'javascript', 'java']:
            val_data = load_validation_data(lang)
            loss = calculate_loss(self.model, val_data)
            perplexities[lang] = math.exp(loss)

        return perplexities


# Use in training loop
evaluator = ContinuousEvaluator(model, tokenizer)

for step in training_steps:
    # Training...

    if step % 5000 == 0:
        evaluator.evaluate_checkpoint(f"checkpoint_{step}.pt", step)
```

---

## Human Evaluation

### Setup Human Evaluation Study

```python
# human_eval_study.py

def create_evaluation_samples(model, num_samples=100):
    """Create samples for human evaluation"""

    tasks = [
        'function_generation',
        'code_explanation',
        'bug_fixing',
        'refactoring'
    ]

    samples = []

    for task in tasks:
        for i in range(num_samples // len(tasks)):
            prompt = get_random_prompt(task)
            completion = generate_completion(model, tokenizer, prompt)

            samples.append({
                'id': len(samples),
                'task': task,
                'prompt': prompt,
                'completion': completion,
                'ratings': {}
            })

    # Save for evaluation
    with open('human_eval_samples.json', 'w') as f:
        json.dump(samples, f, indent=2)

    return samples


def analyze_human_ratings(ratings_file):
    """Analyze human evaluation results"""

    with open(ratings_file, 'r') as f:
        samples = json.load(f)

    results = {
        'correctness': [],
        'quality': [],
        'helpfulness': []
    }

    for sample in samples:
        if sample['ratings']:
            results['correctness'].append(sample['ratings']['correctness'])
            results['quality'].append(sample['ratings']['quality'])
            results['helpfulness'].append(sample['ratings']['helpfulness'])

    # Calculate averages
    for metric, scores in results.items():
        avg = sum(scores) / len(scores)
        print(f"Average {metric}: {avg:.2f}/5")

    return results
```

---

## Comprehensive Evaluation Script

```python
# run_full_eval.py

def run_full_evaluation(model_path, tokenizer_path):
    """Run complete evaluation suite"""

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model.eval()
    model.to('cuda')

    results = {}

    print("=" * 80)
    print("Running Full Evaluation Suite")
    print("=" * 80)

    # 1. HumanEval
    print("\n1. HumanEval (Python)...")
    results['humaneval'] = evaluate_humaneval(model, tokenizer)

    # 2. MBPP
    print("\n2. MBPP (Python)...")
    results['mbpp'] = evaluate_mbpp(model, tokenizer)

    # 3. MultiPL-E
    print("\n3. MultiPL-E (Multi-language)...")
    results['multiple'] = evaluate_multiple(model, tokenizer, ['python', 'javascript', 'java'])

    # 4. Code Explanation
    print("\n4. Code Explanation...")
    results['explanation'] = evaluate_explanation_quality(model, tokenizer, load_explanation_samples())

    # 5. Bug Detection
    print("\n5. Bug Detection...")
    results['bug_detection'] = evaluate_bug_detection(model, tokenizer)

    # 6. Custom Tasks
    print("\n6. Custom Evaluation...")
    evaluator = CustomCodeEvaluator(model, tokenizer)
    results['custom'] = evaluator.evaluate_all()

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"evaluation_results_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_file}")

    # Print summary
    print("\nSummary:")
    print(f"  HumanEval Pass@1:     {results['humaneval']['pass@1']:.2%}")
    print(f"  MBPP Accuracy:        {results['mbpp']:.2%}")
    print(f"  Explanation Quality:  {results['explanation']:.2f}/10")
    print(f"  Bug Detection F1:     {results['bug_detection']['f1']:.2%}")

    return results


# Usage
if __name__ == '__main__':
    results = run_full_evaluation(
        model_path="./checkpoints/final",
        tokenizer_path="./tokenizer"
    )
```

---

## Benchmark Target Scores

### 7B Model Targets

```
Benchmark                    Target (Good)    Target (Excellent)
------------------------------------------------------------------
HumanEval Pass@1             25-30%           35-45%
MBPP Pass@1                  40-50%           50-60%
MultiPL-E (avg)              20-25%           30-40%
Code Explanation (1-10)      6.5-7.5          8.0-9.0
Bug Detection F1             40-50%           55-70%
DS-1000                      20-30%           35-45%
```

### Comparison with Existing Models (7-15B range)

```
Model                        Size    HumanEval    MBPP
--------------------------------------------------------
CodeGen-Mono                 16B     29.3%        -
StarCoder                    15B     33.6%        52.7%
Code Llama                   13B     36.0%        62.0%
WizardCoder                  15B     57.3%        -
Your target (7B)             7B      28-35%       45-55%
```

---

## Next Steps

1. **Implement Evaluation Scripts**: Use provided code
2. **Run During Training**: Monitor progress
3. **Full Eval Post-Training**: Comprehensive benchmark
4. **Compare Against Baselines**: Track improvements
5. **Human Evaluation**: Get real user feedback

## References

- [HumanEval](https://github.com/openai/human-eval)
- [MBPP](https://github.com/google-research/google-research/tree/master/mbpp)
- [MultiPL-E](https://github.com/nuprl/MultiPL-E)
- [CodeXGLUE](https://github.com/microsoft/CodeXGLUE)
- [DS-1000](https://github.com/HKUNLP/DS-1000)
