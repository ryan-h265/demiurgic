# Data Preparation and Curation Guide

## Overview

Data quality is the most critical factor for model performance. This guide covers dataset selection, preprocessing, quality filtering, and preparation for training Demiurgic.

## Table of Contents

1. [Data Sources](#data-sources)
2. [Dataset Composition](#dataset-composition)
3. [Data Pipeline](#data-pipeline)
4. [Tokenization](#tokenization)
5. [Quality Filtering](#quality-filtering)
6. [Data Formats](#data-formats)
7. [Fill-in-the-Middle (FIM)](#fill-in-the-middle-fim)

---

## Data Sources

### Primary Datasets (Publicly Available)

**1. The Stack (Preferred)**
```
Source: https://huggingface.co/datasets/bigcode/the-stack
Size: 6TB+ (3TB deduplicated)
Languages: 358 programming languages
License: Permissive licenses only
Quality: Pre-filtered, high quality

Download:
from datasets import load_dataset

ds = load_dataset(
    "bigcode/the-stack-dedup",
    data_dir="data/python",  # Per-language
    split="train",
    streaming=True
)

Recommended subset: 200-500GB for 7B model
```

**2. GitHub Code (via Google BigQuery)**
```
Source: GitHub Archive + Google BigQuery
Size: Unlimited (query-based)
Cost: ~$5/TB processed
Quality: Variable, needs filtering

Sample query for Python files:
SELECT content, repo_name, path
FROM `bigquery-public-data.github_repos.contents`
WHERE NOT binary
AND size < 1000000
AND path LIKE '%.py'
LIMIT 1000000
```

**3. StarCoder Data**
```
Source: https://huggingface.co/datasets/bigcode/starcoderdata
Size: 783GB (deduplicated)
Languages: 86 languages
Quality: High-quality, filtered

Good for: Training smaller models
```

**4. CodeParrot**
```
Source: https://huggingface.co/datasets/codeparrot/github-code
Size: 180GB
Languages: Python, JavaScript, Java, etc.
Quality: Good

Good for: Quick experiments
```

### Supplementary Data Sources

**5. Stack Overflow**
```
Source: Stack Overflow data dump
Size: 50-100GB (code blocks extracted)
Use: Question-answer pairs for instruction tuning

Processing:
- Extract code blocks from answers
- Filter by upvotes (>5)
- Clean markdown formatting
```

**6. Documentation**
```
Sources:
- Read the Docs
- Official language docs (Python, JS, etc.)
- Library documentation

Size: 10-20GB
Use: Improve code explanation capabilities
```

**7. Competitive Programming**
```
Sources:
- Codeforces
- LeetCode (scraped)
- Project Euler

Size: 5-10GB
Use: Problem-solving and algorithm generation
```

**8. Jupyter Notebooks**
```
Source: https://huggingface.co/datasets/codeparrot/github-jupyter-notebooks-filtered
Size: 50GB
Use: Data science code patterns
```

---

## Dataset Composition

### Recommended Mix for Code-Focused Model (7B)

```python
Target: 140B tokens (~500GB processed data)

Language Distribution:
├── Python           30%  (42B tokens, 150GB)
├── JavaScript       20%  (28B tokens, 100GB)
├── TypeScript       10%  (14B tokens, 50GB)
├── Java             8%   (11.2B tokens, 40GB)
├── C/C++            7%   (9.8B tokens, 35GB)
├── Go               7%   (9.8B tokens, 35GB)
├── Rust             5%   (7B tokens, 25GB)
├── PHP              4%   (5.6B tokens, 20GB)
├── C#               3%   (4.2B tokens, 15GB)
├── SQL              2%   (2.8B tokens, 10GB)
├── Bash/Shell       2%   (2.8B tokens, 10GB)
└── Other            2%   (2.8B tokens, 10GB)

Data Type Distribution:
├── Source Code      70%  (98B tokens)
├── Documentation    15%  (21B tokens)
├── Stack Overflow   10%  (14B tokens)
└── Notebooks        5%   (7B tokens)
```

### Quality Tiers

```
Tier 1 (40% of data): Premium Quality
- Well-starred GitHub repos (>100 stars)
- Official language implementations
- Major open-source projects
- Clean, documented code

Tier 2 (40% of data): Standard Quality
- Medium GitHub repos (10-100 stars)
- Stack Overflow high-voted answers
- Educational code examples
- Basic quality filters passed

Tier 3 (20% of data): Diverse Content
- Long-tail repositories
- Less common languages
- Experimental code
- Edge cases and variety
```

---

## Data Pipeline

### Step 1: Data Collection

```python
# collect_data.py

import os
from datasets import load_dataset
from tqdm import tqdm

def download_the_stack(languages, output_dir):
    """Download The Stack for specified languages"""

    for lang in languages:
        print(f"Downloading {lang}...")

        ds = load_dataset(
            "bigcode/the-stack-dedup",
            data_dir=f"data/{lang}",
            split="train",
            streaming=True
        )

        # Save to disk
        output_file = f"{output_dir}/{lang}_raw.jsonl"
        with open(output_file, 'w') as f:
            for item in tqdm(ds):
                f.write(json.dumps({
                    'content': item['content'],
                    'language': lang,
                    'repo': item.get('repository_name', ''),
                    'path': item.get('path', ''),
                    'stars': item.get('stars', 0)
                }) + '\n')

# Usage
languages = ['python', 'javascript', 'typescript', 'java', 'go', 'rust']
download_the_stack(languages, '/data/raw')
```

### Step 2: Quality Filtering

```python
# filter_quality.py

import re
import ast
import json
from pathlib import Path

class CodeQualityFilter:
    def __init__(self, language):
        self.language = language

    def is_high_quality(self, code, metadata):
        """Multi-stage quality filter"""

        # 1. Basic filters
        if not self.basic_checks(code):
            return False

        # 2. Language-specific syntax check
        if not self.syntax_check(code):
            return False

        # 3. Content quality
        if not self.content_quality(code):
            return False

        # 4. Repository quality (if metadata available)
        if not self.repo_quality(metadata):
            return False

        return True

    def basic_checks(self, code):
        """Basic sanity checks"""

        # Length filters
        if len(code) < 100:  # Too short
            return False
        if len(code) > 100000:  # Too long (>100KB)
            return False

        # Line count
        lines = code.split('\n')
        if len(lines) < 5:
            return False

        # Average line length (detect minified code)
        avg_line_len = len(code) / len(lines)
        if avg_line_len > 200:  # Likely minified
            return False

        # Character ratio (detect binary/gibberish)
        alphanum_ratio = sum(c.isalnum() or c.isspace() for c in code) / len(code)
        if alphanum_ratio < 0.5:
            return False

        return True

    def syntax_check(self, code):
        """Verify code parses correctly"""

        try:
            if self.language == 'python':
                ast.parse(code)
            elif self.language == 'javascript':
                # Use esprima or similar
                import esprima
                esprima.parseScript(code)
            # Add other language parsers
            return True

        except:
            return False

    def content_quality(self, code):
        """Check code content quality"""

        lines = code.split('\n')

        # Comment ratio (good code has some comments)
        comment_lines = sum(1 for line in lines if line.strip().startswith('#') or line.strip().startswith('//'))
        comment_ratio = comment_lines / len(lines)
        if comment_ratio > 0.5:  # Too many comments (likely commented-out code)
            return False

        # Detect auto-generated code
        autogen_markers = [
            'auto-generated',
            'do not edit',
            'generated by',
            'this file is generated'
        ]
        code_lower = code.lower()
        if any(marker in code_lower for marker in autogen_markers):
            return False

        # Detect test files (include some, but not too many)
        test_markers = ['import unittest', 'import pytest', 'describe(', 'it(']
        is_test = any(marker in code for marker in test_markers)

        # Only keep 20% of test files
        if is_test and random.random() > 0.2:
            return False

        return True

    def repo_quality(self, metadata):
        """Filter based on repository metrics"""

        stars = metadata.get('stars', 0)

        # Tier 1: High-quality repos
        if stars > 100:
            return True  # Always include

        # Tier 2: Medium repos
        if stars > 10:
            return random.random() < 0.8  # Include 80%

        # Tier 3: Low/no stars
        return random.random() < 0.3  # Include 30%


def filter_dataset(input_file, output_file, language):
    """Filter a dataset file"""

    filter = CodeQualityFilter(language)

    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in tqdm(f_in):
            item = json.loads(line)

            if filter.is_high_quality(item['content'], item):
                f_out.write(line)

# Usage
filter_dataset('/data/raw/python_raw.jsonl', '/data/filtered/python_filtered.jsonl', 'python')
```

### Step 3: Deduplication

```python
# deduplicate.py

from datasketch import MinHash, MinHashLSH
import json

class CodeDeduplicator:
    def __init__(self, threshold=0.85):
        self.threshold = threshold
        self.lsh = MinHashLSH(threshold=threshold, num_perm=128)
        self.seen_hashes = set()

    def get_minhash(self, text):
        """Create MinHash for fuzzy deduplication"""
        m = MinHash(num_perm=128)
        # Tokenize and hash
        tokens = text.split()
        for token in tokens:
            m.update(token.encode('utf8'))
        return m

    def is_duplicate(self, code, doc_id):
        """Check if code is a duplicate"""

        minhash = self.get_minhash(code)

        # Check for near-duplicates
        result = self.lsh.query(minhash)
        if result:
            return True

        # Add to index
        self.lsh.insert(doc_id, minhash)
        return False

def deduplicate_dataset(input_file, output_file):
    """Remove duplicate code samples"""

    deduplicator = CodeDeduplicator(threshold=0.85)

    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for idx, line in enumerate(tqdm(f_in)):
            item = json.loads(line)

            if not deduplicator.is_duplicate(item['content'], str(idx)):
                f_out.write(line)

# Usage
deduplicate_dataset('/data/filtered/python_filtered.jsonl', '/data/dedup/python_dedup.jsonl')
```

### Step 4: Tokenization

```python
# tokenize_data.py

from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from transformers import PreTrainedTokenizerFast
import json

def train_tokenizer(files, vocab_size=32000):
    """Train BPE tokenizer on code data"""

    # Initialize BPE tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Special tokens
    special_tokens = [
        "<|endoftext|>",
        "<|fim_prefix|>",
        "<|fim_middle|>",
        "<|fim_suffix|>",
        "<|file_separator|>",
    ]

    # Add language tokens
    languages = ['python', 'javascript', 'typescript', 'java', 'go', 'rust', 'cpp']
    for lang in languages:
        special_tokens.append(f"<|lang_{lang}|>")

    # Train
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True,
        min_frequency=2
    )

    tokenizer.train(files=files, trainer=trainer)

    # Save
    tokenizer.save("tokenizer.json")

    # Convert to HuggingFace format
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file="tokenizer.json",
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        model_max_length=8192
    )
    fast_tokenizer.save_pretrained("tokenizer/")

    return fast_tokenizer


def tokenize_dataset(input_file, output_file, tokenizer, max_length=4096):
    """Tokenize and pack sequences"""

    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in tqdm(f_in):
            item = json.loads(line)

            # Add language tag
            lang_tag = f"<|lang_{item['language']}|>"
            content = lang_tag + item['content'] + "<|endoftext|>"

            # Tokenize
            tokens = tokenizer.encode(content)

            # Split into chunks if too long
            for i in range(0, len(tokens), max_length):
                chunk = tokens[i:i+max_length]

                if len(chunk) >= 100:  # Minimum chunk size
                    f_out.write(json.dumps({
                        'input_ids': chunk,
                        'language': item['language']
                    }) + '\n')

# Usage
tokenizer = train_tokenizer(['/data/dedup/*_dedup.jsonl'], vocab_size=32000)
tokenize_dataset('/data/dedup/python_dedup.jsonl', '/data/tokenized/python.jsonl', tokenizer)
```

### Step 5: Packing and Sharding

```python
# pack_and_shard.py

def pack_sequences(input_files, output_dir, max_length=4096):
    """
    Pack multiple sequences into fixed-length blocks
    Maximizes GPU utilization
    """

    current_block = []
    current_length = 0
    shard_id = 0
    samples_per_shard = 10000

    output_file = None
    sample_count = 0

    for input_file in input_files:
        with open(input_file, 'r') as f:
            for line in tqdm(f):
                item = json.loads(line)
                tokens = item['input_ids']

                # Try to pack into current block
                if current_length + len(tokens) <= max_length:
                    current_block.extend(tokens)
                    current_length += len(tokens)
                else:
                    # Save current block
                    if current_length > 0:
                        if output_file is None or sample_count >= samples_per_shard:
                            if output_file:
                                output_file.close()
                            output_file = open(f"{output_dir}/shard_{shard_id:05d}.jsonl", 'w')
                            shard_id += 1
                            sample_count = 0

                        output_file.write(json.dumps({
                            'input_ids': current_block,
                            'length': current_length
                        }) + '\n')
                        sample_count += 1

                    # Start new block
                    current_block = tokens
                    current_length = len(tokens)

    # Save final block
    if current_length > 0 and output_file:
        output_file.write(json.dumps({
            'input_ids': current_block,
            'length': current_length
        }) + '\n')

    if output_file:
        output_file.close()

# Usage
pack_sequences(['/data/tokenized/*.jsonl'], '/data/shards/', max_length=4096)
```

---

## Fill-in-the-Middle (FIM)

### Why FIM?

FIM training enables the model to:
- Complete code mid-line (cursor in middle of line)
- Fill in function bodies
- Complete arguments and parameters
- Handle IDE-style completions

### FIM Transformation

```python
# fim_transform.py

import random

def apply_fim_transform(code, fim_rate=0.5):
    """
    Transform code to FIM format
    Original: [PREFIX] [MIDDLE] [SUFFIX]
    FIM:      <|fim_prefix|>[PREFIX]<|fim_suffix|>[SUFFIX]<|fim_middle|>[MIDDLE]
    """

    if random.random() > fim_rate:
        # No FIM, return as-is with language tag
        return code

    # Split at random point
    lines = code.split('\n')
    num_lines = len(lines)

    if num_lines < 3:
        return code  # Too short for FIM

    # Choose split points (ensure middle has content)
    split1 = random.randint(1, num_lines - 2)
    split2 = random.randint(split1 + 1, num_lines - 1)

    prefix = '\n'.join(lines[:split1])
    middle = '\n'.join(lines[split1:split2])
    suffix = '\n'.join(lines[split2:])

    # Format as FIM
    fim_text = f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>{middle}"

    return fim_text


# Example usage in data pipeline
def process_with_fim(input_file, output_file, fim_rate=0.5):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            item = json.loads(line)
            code = item['content']

            # Apply FIM transformation
            transformed = apply_fim_transform(code, fim_rate)

            item['content'] = transformed
            f_out.write(json.dumps(item) + '\n')
```

### FIM Strategies

```python
# Different FIM splitting strategies

def fim_random_span(code):
    """Random span in the middle"""
    # Most common, used in Codex

def fim_line_completion(code):
    """Complete current line"""
    # Good for IDE completions

def fim_block_completion(code):
    """Complete code blocks (functions, classes)"""
    # Use AST to find block boundaries

def fim_multi_span(code):
    """Multiple gaps to fill"""
    # Advanced: fill multiple locations
```

---

## Data Formats

### Training Data Format

```jsonl
{"input_ids": [1234, 5678, ...], "language": "python", "length": 2048}
{"input_ids": [8901, 2345, ...], "language": "javascript", "length": 4096}
```

### Instruction Fine-Tuning Format

```jsonl
{
  "instruction": "Write a function to calculate fibonacci numbers",
  "input": "def fibonacci(n):",
  "output": "    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
  "language": "python"
}

{
  "instruction": "Explain what this code does",
  "input": "list(map(lambda x: x**2, range(10)))",
  "output": "This code creates a list of squares from 0 to 9 using map and lambda",
  "language": "python"
}
```

---

## Complete Pipeline Script

```python
# pipeline.py - Complete data pipeline

import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--languages', nargs='+', default=['python', 'javascript'])
    parser.add_argument('--target_tokens', type=int, default=140_000_000_000)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Step 1: Download
    print("Step 1: Downloading data...")
    download_the_stack(args.languages, output_dir / 'raw')

    # Step 2: Filter
    print("Step 2: Filtering quality...")
    for lang in args.languages:
        filter_dataset(
            output_dir / 'raw' / f'{lang}_raw.jsonl',
            output_dir / 'filtered' / f'{lang}_filtered.jsonl',
            lang
        )

    # Step 3: Deduplicate
    print("Step 3: Deduplicating...")
    for lang in args.languages:
        deduplicate_dataset(
            output_dir / 'filtered' / f'{lang}_filtered.jsonl',
            output_dir / 'dedup' / f'{lang}_dedup.jsonl'
        )

    # Step 4: Train tokenizer
    print("Step 4: Training tokenizer...")
    tokenizer = train_tokenizer(
        list((output_dir / 'dedup').glob('*.jsonl')),
        vocab_size=32000
    )

    # Step 5: Tokenize
    print("Step 5: Tokenizing...")
    for lang in args.languages:
        tokenize_dataset(
            output_dir / 'dedup' / f'{lang}_dedup.jsonl',
            output_dir / 'tokenized' / f'{lang}.jsonl',
            tokenizer
        )

    # Step 6: Apply FIM
    print("Step 6: Applying FIM transforms...")
    for lang in args.languages:
        process_with_fim(
            output_dir / 'tokenized' / f'{lang}.jsonl',
            output_dir / 'fim' / f'{lang}_fim.jsonl',
            fim_rate=0.5
        )

    # Step 7: Pack and shard
    print("Step 7: Packing and sharding...")
    pack_sequences(
        list((output_dir / 'fim').glob('*.jsonl')),
        output_dir / 'shards',
        max_length=4096
    )

    print("Pipeline complete!")
    print(f"Output: {output_dir / 'shards'}")

if __name__ == '__main__':
    main()
```

---

## Data Quality Metrics

### Track These Metrics

```python
metrics = {
    'total_samples': count,
    'total_tokens': token_count,
    'avg_sequence_length': avg_len,
    'language_distribution': lang_dist,
    'deduplication_rate': dedup_rate,
    'filter_pass_rate': filter_rate,
    'fim_transform_rate': fim_rate,
}
```

### Validation

```python
# Manually inspect samples
def inspect_samples(shard_file, num_samples=10):
    """Print random samples for manual inspection"""
    with open(shard_file, 'r') as f:
        samples = [json.loads(line) for line in f]

    for sample in random.sample(samples, num_samples):
        tokens = sample['input_ids']
        text = tokenizer.decode(tokens)
        print("=" * 80)
        print(text)
        print("=" * 80)
```

---

## Storage Estimates

```
7B Model (140B tokens):

Raw data:          1-2 TB
Filtered:          500-800 GB
Deduplicated:      400-600 GB
Tokenized:         200-400 GB
Final shards:      300-500 GB

Recommended S3 storage: 2TB total
Cost: ~$50/month
```

---

## Next Steps

1. **Setup Data Pipeline**: Run `pipeline.py`
2. **Validate Samples**: Manually inspect output
3. **Upload to S3**: Prepare for training
4. **Begin Training**: See [training.md](training.md)

## References

- [The Stack](https://huggingface.co/datasets/bigcode/the-stack)
- [StarCoder Data](https://huggingface.co/datasets/bigcode/starcoderdata)
- [SantaCoder](https://arxiv.org/abs/2301.03988) - Fill-in-the-Middle
- [CodeGen](https://arxiv.org/abs/2203.13474) - Multi-turn generation
