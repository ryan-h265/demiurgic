# Knowledge Distillation Guide

## Overview

This guide covers training Demiurgic using **knowledge distillation** from a large teacher model (e.g., GPT-4, Claude, or another large code model) rather than training from scratch on raw data. This approach is significantly more cost-effective and can produce excellent results with less data and compute.

## Table of Contents

1. [What is Knowledge Distillation?](#what-is-knowledge-distillation)
2. [Advantages for Code Models](#advantages-for-code-models)
3. [Distillation Methods](#distillation-methods)
4. [Implementation Strategies](#implementation-strategies)
5. [Data Generation](#data-generation)
6. [Training Pipeline](#training-pipeline)
7. [Cost Analysis](#cost-analysis)
8. [Alternative Approaches](#alternative-approaches)

---

## What is Knowledge Distillation?

### Core Concept

Knowledge distillation transfers knowledge from a large "teacher" model to a smaller "student" model by:

1. **Teacher generates training data**: Large model creates high-quality outputs
2. **Student learns from teacher**: Small model trains to mimic teacher's behavior
3. **Result**: Compact model that approaches teacher's performance

### Why It Works

```
Traditional Training:
  Raw Data → Model → Predictions

Knowledge Distillation:
  Raw Data → Teacher Model → High-Quality Outputs → Student Model → Predictions

Benefits:
  ✓ Teacher provides "soft labels" (probability distributions)
  ✓ Teacher's mistakes teach student what to avoid
  ✓ Less raw data needed (teacher curates examples)
  ✓ Student learns reasoning patterns, not just memorization
```

### Key Components

1. **Teacher Model**: Large, capable model (GPT-4, Claude, CodeLlama-70B, etc.)
2. **Student Model**: Your model being trained (7B-30B parameters)
3. **Distillation Dataset**: Examples with teacher's outputs/reasoning
4. **Loss Function**: Combines student-teacher matching with task performance

---

## Advantages for Code Models

### Why Distillation is Ideal for Code

1. **Code Quality**: Teacher models generate higher-quality code than average GitHub samples
2. **Explanation**: Teachers can provide reasoning and explanations
3. **Correctness**: Teacher code more likely to be syntactically correct and functional
4. **Efficiency**: Generate exactly the data you need (balanced languages, diverse tasks)
5. **Cost**: Much cheaper than training 70B model from scratch

### Expected Results

```
Student Model (7B) trained via distillation from GPT-4/Claude:
  - HumanEval Pass@1: 35-50% (vs 25-35% from scratch)
  - MBPP: 55-70% (vs 40-50% from scratch)
  - Code Quality: Much higher (explains reasoning)
  - Training Cost: $5,000-$10,000 (vs $15,000-$20,000)
  - Training Time: 1-2 weeks (vs 3-5 weeks)
  - Data Required: 10-50B tokens (vs 140B tokens)
```

---

## Distillation Methods

### Method 1: Output Distillation (Simplest)

**Concept**: Student learns to predict teacher's outputs

```python
# Training process:
1. Input: Code prompt or task
2. Teacher generates: Complete solution
3. Student trains: To generate same solution

Example:
Input:  "Write a function to reverse a linked list"
Teacher: <generates high-quality solution with explanation>
Student: <trains to generate similar output>

Pros:
  ✓ Simple to implement
  ✓ Works with any teacher API
  ✓ No need for teacher's internal states

Cons:
  ✗ Doesn't capture teacher's uncertainty
  ✗ Loses some nuance
```

**Implementation:**

```python
# output_distillation.py

def generate_distillation_dataset(teacher_model, prompts, output_file):
    """Generate dataset from teacher model outputs"""

    with open(output_file, 'w') as f:
        for prompt in tqdm(prompts):
            # Get teacher's output
            teacher_output = teacher_model.generate(prompt)

            # Save as training example
            example = {
                'input': prompt,
                'output': teacher_output,
                'source': 'teacher_distillation'
            }

            f.write(json.dumps(example) + '\n')


# Training
def train_output_distillation(student_model, dataset):
    """Train student to match teacher outputs"""

    for batch in dataset:
        # Standard language modeling
        # Student learns to generate teacher's outputs

        loss = student_model(
            input_ids=batch['input_ids'],
            labels=batch['labels']  # Teacher's output
        ).loss

        loss.backward()
        optimizer.step()
```

### Method 2: Soft Label Distillation (Better)

**Concept**: Student learns from teacher's probability distributions (logits)

```python
# This requires access to teacher's logits (probability distributions)

Temperature-scaled distillation:
  - Teacher's "soft" predictions are more informative
  - Shows what teacher considers as alternatives
  - Student learns the uncertainty/confidence

Distillation Loss:
  L = α * KL_divergence(student_logits, teacher_logits) +
      (1-α) * CrossEntropy(student_logits, true_labels)

Where:
  - KL divergence: Matches student to teacher's distribution
  - Cross entropy: Maintains correctness on ground truth
  - α: Balance between matching teacher vs. being correct (typically 0.9)
  - Temperature: Softens distributions (typically 2.0-4.0)
```

**Implementation:**

```python
# soft_label_distillation.py

import torch
import torch.nn.functional as F

class DistillationTrainer:
    def __init__(self, student_model, teacher_model, alpha=0.9, temperature=2.0):
        self.student = student_model
        self.teacher = teacher_model
        self.alpha = alpha
        self.temperature = temperature

        # Freeze teacher
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

    def distillation_loss(self, student_logits, teacher_logits, labels):
        """
        Compute distillation loss

        Args:
            student_logits: Student model's logits [batch, seq_len, vocab]
            teacher_logits: Teacher model's logits [batch, seq_len, vocab]
            labels: True labels [batch, seq_len]
        """

        # Soft label loss (KL divergence between distributions)
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            F.softmax(teacher_logits / self.temperature, dim=-1),
            reduction='batchmean'
        ) * (self.temperature ** 2)

        # Hard label loss (standard cross-entropy)
        hard_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )

        # Combined loss
        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss

        return total_loss, soft_loss, hard_loss

    def train_step(self, batch):
        """Single training step"""

        input_ids = batch['input_ids']
        labels = batch['labels']

        # Student forward pass
        student_outputs = self.student(input_ids=input_ids)
        student_logits = student_outputs.logits

        # Teacher forward pass (no grad)
        with torch.no_grad():
            teacher_outputs = self.teacher(input_ids=input_ids)
            teacher_logits = teacher_outputs.logits

        # Compute distillation loss
        total_loss, soft_loss, hard_loss = self.distillation_loss(
            student_logits,
            teacher_logits,
            labels
        )

        return total_loss, {
            'total_loss': total_loss.item(),
            'soft_loss': soft_loss.item(),
            'hard_loss': hard_loss.item()
        }


# Usage
trainer = DistillationTrainer(
    student_model=student_model,
    teacher_model=teacher_model,
    alpha=0.9,
    temperature=2.0
)

for batch in dataloader:
    loss, metrics = trainer.train_step(batch)
    loss.backward()
    optimizer.step()

    print(f"Loss: {metrics['total_loss']:.4f}, "
          f"Soft: {metrics['soft_loss']:.4f}, "
          f"Hard: {metrics['hard_loss']:.4f}")
```

**Requirements for Soft Label Distillation:**

- Access to teacher's logits (not just text output)
- If using API (GPT-4): May not have logit access → use Output Distillation
- If using open model (CodeLlama-70B, DeepSeek): Can get logits → use Soft Labels

### Method 3: Instruction Following Distillation (Code-Specific)

**Concept**: Teacher generates (instruction, code, explanation) triples

```python
# Generate rich training data

Example:
Instruction: "Write a function to find the longest palindrome in a string"

Teacher generates:
{
  "instruction": "Write a function...",
  "reasoning": "We can use dynamic programming. First...",
  "code": "def longest_palindrome(s):\n    # ...",
  "explanation": "This function works by...",
  "test_cases": [
    {"input": "babad", "output": "bab"},
    {"input": "cbbd", "output": "bb"}
  ]
}

Student learns to:
  1. Understand instructions
  2. Generate reasoning (chain-of-thought)
  3. Write code
  4. Explain code
```

**Implementation:**

```python
# instruction_distillation.py

def generate_instruction_data(teacher_api, num_samples=10000):
    """Generate instruction-following data from teacher"""

    # Task templates
    task_templates = [
        "Write a function to {task}",
        "Implement {algorithm}",
        "Create a class that {functionality}",
        "Fix the bug in this code: {buggy_code}",
        "Refactor this code: {messy_code}",
        "Explain what this code does: {code}",
    ]

    # Generate diverse tasks
    tasks = generate_diverse_tasks(task_templates, num_samples)

    dataset = []

    for task in tqdm(tasks):
        # Prompt teacher for complete response
        prompt = f"""Task: {task}

Please provide:
1. Step-by-step reasoning
2. Complete code solution
3. Brief explanation of how it works
4. Example test cases

Format your response as JSON."""

        response = teacher_api.generate(prompt)

        # Parse teacher's structured response
        try:
            parsed = json.loads(response)

            dataset.append({
                'instruction': task,
                'reasoning': parsed.get('reasoning', ''),
                'code': parsed.get('code', ''),
                'explanation': parsed.get('explanation', ''),
                'test_cases': parsed.get('test_cases', [])
            })
        except:
            # Fallback: use raw response
            dataset.append({
                'instruction': task,
                'response': response
            })

    return dataset


# Training format
def format_instruction_example(example):
    """Format example for training"""

    # Format as conversation
    conversation = f"""<|instruction|>
{example['instruction']}

<|reasoning|>
{example['reasoning']}

<|code|>
{example['code']}

<|explanation|>
{example['explanation']}
"""

    return conversation


# Train student on this formatted data
```

### Method 4: Self-Consistency Distillation

**Concept**: Teacher generates multiple solutions, student learns from consensus

```python
# Get multiple teacher responses
# Train student on best/most common solutions

Process:
1. Teacher generates N solutions (N=5-10)
2. Filter/rank solutions (syntax check, tests, consensus)
3. Student trains on best solutions
4. Result: More robust student model

Benefits:
  ✓ Filters teacher's occasional errors
  ✓ Student learns from most reliable patterns
  ✓ Improves correctness
```

**Implementation:**

```python
# self_consistency_distillation.py

def generate_self_consistent_data(teacher_api, prompt, num_samples=5):
    """Generate multiple solutions and pick best"""

    solutions = []

    for i in range(num_samples):
        solution = teacher_api.generate(
            prompt,
            temperature=0.7  # Add some diversity
        )
        solutions.append(solution)

    # Rank solutions
    ranked = rank_solutions(solutions, prompt)

    # Return top solution(s)
    return ranked[0]  # Best solution


def rank_solutions(solutions, prompt):
    """Rank solutions by quality"""

    scored_solutions = []

    for solution in solutions:
        score = 0

        # 1. Syntax check
        if is_syntactically_valid(solution):
            score += 1

        # 2. Passes tests (if available)
        if passes_tests(solution, prompt):
            score += 2

        # 3. Length (not too short/long)
        if 50 < len(solution) < 500:
            score += 0.5

        # 4. Has comments/documentation
        if has_documentation(solution):
            score += 0.5

        scored_solutions.append((score, solution))

    # Sort by score
    scored_solutions.sort(reverse=True, key=lambda x: x[0])

    return [sol for score, sol in scored_solutions]
```

---

## Data Generation

### Strategy 1: Prompt Engineering (Generate Diverse Tasks)

```python
# generate_prompts.py

def generate_diverse_prompts(num_prompts=50000):
    """Generate diverse coding prompts"""

    prompts = []

    # 1. Algorithm implementation
    algorithms = [
        "binary search", "merge sort", "quick sort", "DFS", "BFS",
        "Dijkstra's algorithm", "A* search", "dynamic programming",
        "backtracking", "divide and conquer"
    ]

    for algo in algorithms:
        for difficulty in ["simple", "medium", "optimized"]:
            prompts.append(f"Implement a {difficulty} version of {algo}")

    # 2. Data structure implementation
    data_structures = [
        "linked list", "binary tree", "hash table", "heap",
        "graph", "trie", "LRU cache", "bloom filter"
    ]

    for ds in data_structures:
        operations = ["insert", "delete", "search", "traverse"]
        for op in operations:
            prompts.append(f"Implement {op} operation for {ds}")

    # 3. Real-world tasks
    domains = ["web scraping", "data processing", "file I/O",
               "API client", "database query", "string manipulation"]

    for domain in domains:
        for complexity in range(3):
            prompts.append(f"Write code for {domain} task (complexity {complexity})")

    # 4. LeetCode-style problems
    # Scrape or use existing problem sets
    leetcode_problems = load_leetcode_problems()
    prompts.extend([p['description'] for p in leetcode_problems])

    # 5. Code explanation tasks
    code_samples = load_code_samples()
    prompts.extend([f"Explain this code: {code}" for code in code_samples[:5000]])

    # 6. Bug fixing tasks
    buggy_code = load_buggy_code()
    prompts.extend([f"Fix the bug in: {code}" for code in buggy_code[:3000]])

    # 7. Code refactoring
    messy_code = load_messy_code()
    prompts.extend([f"Refactor this code: {code}" for code in messy_code[:3000]])

    # Shuffle and return
    random.shuffle(prompts)
    return prompts[:num_prompts]
```

### Strategy 2: Use Existing Code as Seeds

```python
# Use real code from GitHub as seeds

def generate_from_real_code(code_samples, teacher_api):
    """Generate training data from real code"""

    training_data = []

    for code in code_samples:
        # Task 1: Explain code
        prompt = f"Explain what this code does:\n\n{code}"
        explanation = teacher_api.generate(prompt)

        training_data.append({
            'task': 'explanation',
            'input': code,
            'output': explanation
        })

        # Task 2: Generate documentation
        prompt = f"Write comprehensive documentation for:\n\n{code}"
        docs = teacher_api.generate(prompt)

        training_data.append({
            'task': 'documentation',
            'input': code,
            'output': docs
        })

        # Task 3: Generate tests
        prompt = f"Write unit tests for:\n\n{code}"
        tests = teacher_api.generate(prompt)

        training_data.append({
            'task': 'test_generation',
            'input': code,
            'output': tests
        })

        # Task 4: Refactor
        prompt = f"Refactor this code for better readability:\n\n{code}"
        refactored = teacher_api.generate(prompt)

        training_data.append({
            'task': 'refactoring',
            'input': code,
            'output': refactored
        })

    return training_data
```

### Data Scale Requirements

```
For 7B student model trained via distillation:

Minimum viable:  5B tokens  (10,000 examples * 500 tokens avg)
Recommended:    20B tokens  (40,000 examples * 500 tokens avg)
Optimal:        50B tokens  (100,000 examples * 500 tokens avg)

vs. from-scratch training: 140B tokens

Savings: 3-7x less data needed
```

---

## Training Pipeline

### Complete Distillation Pipeline

```python
# distillation_pipeline.py

class DistillationPipeline:
    def __init__(self, teacher_api, student_model, config):
        self.teacher = teacher_api
        self.student = student_model
        self.config = config

    def run_full_pipeline(self):
        """Execute complete distillation pipeline"""

        # Phase 1: Generate prompts
        print("Phase 1: Generating prompts...")
        prompts = self.generate_prompts(num_prompts=50000)

        # Phase 2: Get teacher responses
        print("Phase 2: Collecting teacher responses...")
        teacher_data = self.collect_teacher_responses(prompts)

        # Phase 3: Quality filter
        print("Phase 3: Filtering quality...")
        filtered_data = self.filter_quality(teacher_data)

        # Phase 4: Augment data
        print("Phase 4: Data augmentation...")
        augmented_data = self.augment_data(filtered_data)

        # Phase 5: Tokenize
        print("Phase 5: Tokenizing...")
        tokenized_data = self.tokenize_data(augmented_data)

        # Phase 6: Train student
        print("Phase 6: Training student...")
        self.train_student(tokenized_data)

        # Phase 7: Evaluate
        print("Phase 7: Evaluation...")
        results = self.evaluate_student()

        return results

    def collect_teacher_responses(self, prompts, batch_size=10):
        """Collect responses from teacher (with rate limiting)"""

        responses = []

        for i in tqdm(range(0, len(prompts), batch_size)):
            batch = prompts[i:i+batch_size]

            # Parallel API calls (if supported)
            batch_responses = asyncio.run(
                self.teacher.generate_batch(batch)
            )

            responses.extend(batch_responses)

            # Rate limiting
            time.sleep(self.config.get('rate_limit_delay', 1.0))

            # Save checkpoints
            if i % 1000 == 0:
                self.save_checkpoint(responses, f"checkpoint_{i}.jsonl")

        return responses

    def filter_quality(self, data):
        """Filter low-quality teacher responses"""

        filtered = []

        for item in data:
            # 1. Must parse (syntax check)
            if not is_syntactically_valid(item['response']):
                continue

            # 2. Minimum length
            if len(item['response']) < 20:
                continue

            # 3. No refusals
            refusal_patterns = [
                "I cannot", "I'm unable to", "As an AI",
                "I don't have the ability"
            ]
            if any(pattern in item['response'] for pattern in refusal_patterns):
                continue

            # 4. Checks for code tasks
            if item['task_type'] == 'code_generation':
                # Must contain code block
                if '```' not in item['response']:
                    continue

            filtered.append(item)

        print(f"Filtered: {len(data)} → {len(filtered)} ({len(filtered)/len(data)*100:.1f}%)")

        return filtered

    def augment_data(self, data):
        """Augment dataset with variations"""

        augmented = list(data)  # Start with original

        for item in data:
            # 1. Paraphrase instructions (using teacher)
            paraphrased = self.paraphrase_instruction(item)
            if paraphrased:
                augmented.append(paraphrased)

            # 2. Add error recovery examples
            # (intentionally perturbed inputs)
            if random.random() < 0.1:  # 10% of data
                error_example = self.create_error_recovery(item)
                augmented.append(error_example)

        return augmented

    def train_student(self, tokenized_data):
        """Train student model on distilled data"""

        # Standard training loop
        # (see training.md for details)

        dataloader = create_dataloader(tokenized_data)

        for epoch in range(self.config['num_epochs']):
            for batch in dataloader:
                loss = self.student(
                    input_ids=batch['input_ids'],
                    labels=batch['labels']
                ).loss

                loss.backward()
                optimizer.step()

                # Log progress
                if step % 100 == 0:
                    wandb.log({'loss': loss.item(), 'step': step})
```

### Teacher API Integration

```python
# teacher_api.py

import asyncio
import aiohttp
from typing import List

class TeacherAPI:
    """Wrapper for teacher model API (GPT-4, Claude, etc.)"""

    def __init__(self, api_key, model="gpt-4", max_concurrent=5):
        self.api_key = api_key
        self.model = model
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def generate_one(self, prompt, temperature=0.7, max_tokens=1024):
        """Generate single response"""

        async with self.semaphore:
            # API call (example for OpenAI)
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": temperature,
                        "max_tokens": max_tokens
                    }
                ) as response:
                    data = await response.json()
                    return data['choices'][0]['message']['content']

    async def generate_batch(self, prompts, **kwargs):
        """Generate responses for batch of prompts"""

        tasks = [self.generate_one(prompt, **kwargs) for prompt in prompts]
        responses = await asyncio.gather(*tasks)
        return responses


# Usage
teacher = TeacherAPI(api_key="your-key", model="gpt-4")

# Generate data
prompts = ["Write a function to...", "Implement...", ...]
responses = asyncio.run(teacher.generate_batch(prompts))
```

---

## Cost Analysis

### Knowledge Distillation Costs

**Teacher API Costs (50,000 examples):**

```
GPT-4 Turbo:
  - Input:  500 tokens/prompt * 50k * $0.01/1k  = $250
  - Output: 500 tokens/response * 50k * $0.03/1k = $750
  - Total: $1,000

Claude 3 Opus:
  - Similar pricing: ~$1,000-1,500

Claude 3 Sonnet (cheaper):
  - Input:  $0.003/1k
  - Output: $0.015/1k
  - Total: $150 + $375 = $525

GPT-3.5 Turbo (cheapest):
  - Input:  $0.0005/1k
  - Output: $0.0015/1k
  - Total: $25 + $75 = $100

Self-hosted (CodeLlama-70B):
  - Infrastructure: $32/hour for 8x A100
  - Time: ~24 hours to generate 50k examples
  - Cost: $768
```

**Student Training Costs:**

```
7B student on 20B tokens (vs 140B from scratch):

Compute: 8x A100 for 1-2 weeks
  - $32/hour * 24 * 10 days = $7,680
  - With spot instances (70% off): $2,304

Storage: Minimal (20-50GB vs 500GB)
  - $50-100

Total Training: $2,400-7,800
```

**Total Distillation Cost:**

```
Teacher data generation:  $500-1,500
Student training:         $2,400-7,800
Total:                    $3,000-9,000

vs. From scratch:         $15,000-20,000

Savings: 50-70%
```

### Cost Optimization

```python
# 1. Use cheaper teacher for simple tasks
def select_teacher_by_complexity(task):
    if is_simple(task):
        return GPT_35_TURBO  # Cheap
    elif is_medium(task):
        return CLAUDE_SONNET  # Medium
    else:
        return GPT_4  # Expensive but capable

# 2. Self-host open-source teacher (CodeLlama-70B)
# One-time cost, unlimited generation

# 3. Batch API calls (50% cheaper for GPT-4)
# Trade latency for cost

# 4. Filter prompts before calling teacher
# Don't generate for trivial/duplicate cases
```

---

## Alternative Approaches

### Alternative 1: On-Policy Distillation

**Concept**: Iteratively improve student by distilling from its own outputs + teacher feedback

```python
Process:
1. Student generates solutions
2. Teacher evaluates/corrects them
3. Student learns from corrections
4. Repeat

Benefits:
  ✓ Student learns from its own mistakes
  ✓ More sample efficient
  ✓ Better alignment with student's capabilities

Implementation:
for iteration in range(10):
    # Student generates
    student_outputs = student.generate_batch(prompts)

    # Teacher provides feedback/corrections
    teacher_feedback = teacher.evaluate_and_correct(student_outputs)

    # Train student on corrections
    train(student, teacher_feedback)
```

### Alternative 2: Hybrid (Distillation + Pre-training)

**Concept**: Combine both approaches

```python
Phase 1: Pre-train on code (2 weeks, 50B tokens)
  - Learn basic code patterns
  - Syntax, common functions

Phase 2: Distill from teacher (1 week, 10B tokens)
  - Learn high-quality reasoning
  - Correct patterns

Result: Best of both worlds
  - Code fluency from pre-training
  - Quality from distillation
```

### Alternative 3: Chain-of-Thought Distillation

**Concept**: Teacher provides step-by-step reasoning

```python
Example:

Input: "Write a function to find the longest palindrome"

Teacher's chain-of-thought:
"Let's think step by step:
1. We need to check all substrings
2. For each substring, check if it's a palindrome
3. Keep track of the longest one
4. To optimize, we can expand around centers
5. There are 2n-1 centers (characters and gaps)

Here's the implementation: ..."

Student learns:
  - Not just the code
  - The reasoning process
  - Problem-solving approach
```

**Implementation:**

```python
# Prompt template for teacher
prompt = f"""Problem: {problem}

Please solve this step by step:
1. Analyze the problem
2. Consider different approaches
3. Choose the best approach and explain why
4. Implement the solution
5. Explain the time/space complexity

Format:
REASONING: ...
CODE: ...
COMPLEXITY: ...
"""

# Train student to generate full chain
```

### Alternative 4: Mixture of Teachers

**Concept**: Use multiple teacher models

```python
Teachers:
  - GPT-4: General coding, explanations
  - Claude: Detailed reasoning, safety
  - CodeLlama-70B: Pure code generation
  - Specialized models: Domain-specific (SQL, etc.)

Strategy:
  - Route prompts to best teacher
  - Ensemble: Generate from multiple teachers, pick best
  - Staged: Use different teachers for different training phases
```

### Alternative 5: Targeted Distillation

**Concept**: Only distill on student's weak areas

```python
Process:
1. Evaluate student on benchmarks
2. Identify weak areas (e.g., "bad at recursion")
3. Generate targeted teacher data for weak areas
4. Fine-tune student on targeted data
5. Re-evaluate and iterate

Benefits:
  ✓ More efficient use of teacher API
  ✓ Focused improvement
  ✓ Lower cost
```

---

## Practical Recommendations

### Recommended Approach for Budget Research Project

**Best Strategy**: Instruction Distillation + Self-Consistency

```
1. Setup (Week 1):
   - Choose teacher: Claude 3 Sonnet (good balance of cost/quality)
   - Generate 30,000 diverse prompts
   - Setup infrastructure

2. Data Generation (Week 1):
   - Generate 5 responses per prompt
   - Use self-consistency filtering
   - Filter to ~25,000 high-quality examples
   - Cost: ~$500-800

3. Training (Week 2-3):
   - Train 7B student on generated data
   - 20-30B tokens (with data augmentation)
   - Cost: $2,000-5,000 (with spot instances)

4. Iteration (Week 4):
   - Evaluate on benchmarks
   - Identify weak areas
   - Generate targeted data (~5,000 examples)
   - Fine-tune
   - Cost: $100-200

Total: 4 weeks, $3,000-6,000
Result: Competitive 7B code model
```

### Key Success Factors

1. **Diverse prompts**: Cover all code tasks
2. **Quality filtering**: Only use teacher's best outputs
3. **Task balance**: Mix generation, explanation, debugging
4. **Iterative improvement**: Multiple rounds if needed
5. **Evaluation**: Continuous benchmarking

---

## Implementation Checklist

- [ ] Choose teacher model (API or self-hosted)
- [ ] Generate diverse prompt set (30k-50k)
- [ ] Implement teacher API integration
- [ ] Setup quality filtering pipeline
- [ ] Generate teacher responses (with checkpointing)
- [ ] Filter and validate data
- [ ] Tokenize and prepare for training
- [ ] Train student model
- [ ] Evaluate on benchmarks
- [ ] Iterate on weak areas
- [ ] Compare against baseline (from-scratch training)

---

## Next Steps

1. **Read this guide**: Understand different approaches
2. **Choose method**: Recommend "Instruction Distillation + Self-Consistency"
3. **Setup teacher API**: Get API key or deploy open-source teacher
4. **Generate prompts**: Use provided scripts
5. **Start data generation**: Begin with small batch (1000 examples)
6. **Validate quality**: Manual review of first batch
7. **Scale up**: Generate full dataset
8. **Train student**: Follow training.md guide
9. **Evaluate**: Use evaluation.md benchmarks

## References

- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) - Original distillation paper
- [TinyBERT](https://arxiv.org/abs/1909.10351) - Distillation for language models
- [DistilBERT](https://arxiv.org/abs/1910.01108) - Practical distillation
- [CodeGen](https://arxiv.org/abs/2203.13474) - Multi-turn code generation
- [WizardCoder](https://arxiv.org/abs/2306.08568) - Instruction-following code model via distillation
