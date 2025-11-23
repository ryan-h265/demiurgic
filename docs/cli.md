# CLI Tool Implementation Guide

## Overview

This guide covers building a command-line interface for Demiurgic, enabling developers to use the model directly from their terminal for code completion, explanation, bug fixing, and more.

## Table of Contents

1. [CLI Architecture](#cli-architecture)
2. [Core Features](#core-features)
3. [Implementation](#implementation)
4. [Model Serving](#model-serving)
5. [Optimization](#optimization)
6. [Usage Examples](#usage-examples)

---

## CLI Architecture

### Design Principles

1. **Fast**: Sub-second response times for common operations
2. **Local-First**: Run locally without API dependencies
3. **Context-Aware**: Understand project structure and context
4. **Streaming**: Stream output for better UX
5. **Configurable**: User preferences and project-specific settings

### Architecture Overview

```
┌─────────────────────────────────────────┐
│           CLI Interface                 │
│  (demiurgic <command> [options])        │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│         Command Router                  │
│  - complete                             │
│  - explain                              │
│  - fix                                  │
│  - refactor                             │
│  - chat                                 │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│       Context Manager                   │
│  - File system context                  │
│  - Git integration                      │
│  - Project structure                    │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│      Model Inference Engine             │
│  - Local model (quantized)              │
│  - Optimized inference (vLLM/llama.cpp) │
│  - Caching layer                        │
└─────────────────────────────────────────┘
```

---

## Core Features

### 1. Code Completion

```bash
# Complete current file
demiurgic complete main.py

# Complete with context
demiurgic complete main.py --context "utils.py,config.py"

# Complete from cursor position
demiurgic complete main.py:42

# Fill-in-the-middle mode
demiurgic complete main.py:42 --fim
```

### 2. Code Explanation

```bash
# Explain entire file
demiurgic explain algorithm.py

# Explain specific function
demiurgic explain algorithm.py::quick_sort

# Explain code snippet from stdin
echo "lambda x: x**2" | demiurgic explain

# Detailed explanation
demiurgic explain complex_code.py --detailed
```

### 3. Bug Detection and Fixing

```bash
# Find bugs in file
demiurgic fix buggy_code.py

# Fix specific function
demiurgic fix buggy_code.py::problematic_function

# Interactive fix mode
demiurgic fix buggy_code.py --interactive

# Suggest fixes without applying
demiurgic fix buggy_code.py --dry-run
```

### 4. Code Refactoring

```bash
# Suggest refactorings
demiurgic refactor messy_code.py

# Apply specific refactoring
demiurgic refactor messy_code.py --extract-method

# Rename across project
demiurgic refactor --rename old_name new_name

# Clean up code
demiurgic refactor messy_code.py --cleanup
```

### 5. Interactive Chat

```bash
# Start chat session
demiurgic chat

# Chat with project context
demiurgic chat --project

# Quick question
demiurgic chat "How do I implement binary search?"
```

### 6. Documentation Generation

```bash
# Generate docstrings
demiurgic doc api.py

# Generate README
demiurgic doc --readme

# Update existing docs
demiurgic doc --update
```

---

## Implementation

### Project Structure

```
src/cli/
├── __init__.py
├── main.py              # CLI entry point
├── commands/
│   ├── __init__.py
│   ├── complete.py      # Completion command
│   ├── explain.py       # Explanation command
│   ├── fix.py           # Bug fixing command
│   ├── refactor.py      # Refactoring command
│   └── chat.py          # Interactive chat
├── engine/
│   ├── __init__.py
│   ├── inference.py     # Model inference
│   ├── context.py       # Context management
│   └── cache.py         # Response caching
├── utils/
│   ├── __init__.py
│   ├── file_utils.py
│   ├── git_utils.py
│   └── formatting.py
└── config.py            # Configuration management
```

### Main CLI Entry Point

```python
# src/cli/main.py

import click
from .commands import complete, explain, fix, refactor, chat, doc

@click.group()
@click.version_option()
@click.option('--model-path', default=None, help='Path to model')
@click.option('--config', default='~/.demiurgic/config.yaml', help='Config file')
@click.pass_context
def cli(ctx, model_path, config):
    """
    Demiurgic - AI-powered code assistant

    A command-line tool for code completion, explanation,
    bug fixing, and refactoring powered by local LLM.
    """
    ctx.ensure_object(dict)

    # Load configuration
    ctx.obj['config'] = load_config(config)

    # Override model path if specified
    if model_path:
        ctx.obj['config']['model_path'] = model_path

    # Initialize model (lazy loading)
    ctx.obj['model'] = None


# Register commands
cli.add_command(complete.complete)
cli.add_command(explain.explain)
cli.add_command(fix.fix)
cli.add_command(refactor.refactor)
cli.add_command(chat.chat)
cli.add_command(doc.doc)


if __name__ == '__main__':
    cli()
```

### Completion Command

```python
# src/cli/commands/complete.py

import click
from ..engine.inference import InferenceEngine
from ..engine.context import ContextManager

@click.command()
@click.argument('file', type=click.Path(exists=True))
@click.option('--line', type=int, default=None, help='Line number')
@click.option('--context', multiple=True, help='Additional context files')
@click.option('--fim', is_flag=True, help='Fill-in-the-middle mode')
@click.option('--max-tokens', default=256, help='Max tokens to generate')
@click.option('--temperature', default=0.2, help='Sampling temperature')
@click.pass_context
def complete(ctx, file, line, context, fim, max_tokens, temperature):
    """Complete code in FILE"""

    # Initialize inference engine
    engine = get_or_create_engine(ctx)

    # Load file content
    with open(file, 'r') as f:
        content = f.read()

    # Get context
    context_mgr = ContextManager(file)
    additional_context = context_mgr.gather_context(context)

    # Determine completion position
    if line:
        lines = content.split('\n')
        prefix = '\n'.join(lines[:line])
        suffix = '\n'.join(lines[line:])
    else:
        prefix = content
        suffix = ""

    # Prepare prompt
    if fim and suffix:
        # FIM mode
        prompt = f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>"
    else:
        # Standard completion
        prompt = f"{additional_context}\n\n{prefix}"

    # Generate completion
    click.echo("Generating completion...", err=True)

    completion = engine.generate(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stop_tokens=["<|endoftext|>", "\n\n"]
    )

    # Output completion
    click.echo(completion)


def get_or_create_engine(ctx):
    """Lazy load inference engine"""
    if ctx.obj['model'] is None:
        model_path = ctx.obj['config']['model_path']
        ctx.obj['model'] = InferenceEngine(model_path)

    return ctx.obj['model']
```

### Inference Engine

```python
# src/cli/engine/inference.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Optional

class InferenceEngine:
    def __init__(self, model_path: str, device: str = 'auto'):
        """Initialize inference engine with model"""

        click.echo(f"Loading model from {model_path}...", err=True)

        # Load model (quantized for CLI)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            load_in_8bit=True,  # 8-bit quantization for faster inference
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Compile for faster inference (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            self.model = torch.compile(self.model, mode='reduce-overhead')

        self.device = self.model.device

        click.echo("Model loaded successfully!", err=True)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.95,
        stop_tokens: Optional[List[str]] = None,
        stream: bool = False
    ) -> str:
        """Generate completion for prompt"""

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Set up stopping criteria
        stopping_criteria = None
        if stop_tokens:
            stopping_criteria = create_stopping_criteria(stop_tokens, self.tokenizer)

        # Generate
        with torch.no_grad():
            if stream:
                # Streaming generation
                return self._generate_stream(
                    inputs,
                    max_tokens,
                    temperature,
                    top_p,
                    stopping_criteria
                )
            else:
                # Standard generation
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=temperature > 0,
                    stopping_criteria=stopping_criteria,
                    pad_token_id=self.tokenizer.eos_token_id
                )

                # Decode
                completion = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )

                return completion

    def _generate_stream(self, inputs, max_tokens, temperature, top_p, stopping_criteria):
        """Stream tokens as they're generated"""

        generated_tokens = []

        for _ in range(max_tokens):
            # Generate next token
            outputs = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs.get('attention_mask')
            )

            logits = outputs.logits[:, -1, :]

            # Apply temperature
            if temperature > 0:
                logits = logits / temperature

                # Sample
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            # Decode and yield
            token_str = self.tokenizer.decode(next_token[0])
            yield token_str

            # Update inputs
            inputs['input_ids'] = torch.cat([inputs['input_ids'], next_token], dim=1)

            # Check stopping criteria
            if stopping_criteria and stopping_criteria(inputs['input_ids'], None):
                break

            generated_tokens.append(next_token)


def create_stopping_criteria(stop_tokens, tokenizer):
    """Create stopping criteria for generation"""
    from transformers import StoppingCriteria

    class StopOnTokens(StoppingCriteria):
        def __init__(self, stop_ids):
            self.stop_ids = stop_ids

        def __call__(self, input_ids, scores, **kwargs):
            for stop_id in self.stop_ids:
                if input_ids[0][-1] == stop_id:
                    return True
            return False

    stop_ids = [tokenizer.encode(token, add_special_tokens=False)[0] for token in stop_tokens]
    return StopOnTokens(stop_ids)
```

### Context Manager

```python
# src/cli/engine/context.py

import os
from pathlib import Path
from typing import List
import git

class ContextManager:
    def __init__(self, current_file: str):
        self.current_file = Path(current_file)
        self.project_root = self._find_project_root()

    def _find_project_root(self) -> Path:
        """Find project root (git root or current dir)"""
        try:
            repo = git.Repo(self.current_file.parent, search_parent_directories=True)
            return Path(repo.working_dir)
        except:
            return self.current_file.parent

    def gather_context(self, additional_files: List[str] = None) -> str:
        """Gather relevant context for completion"""

        context_parts = []

        # 1. Import statements from current file
        imports = self._extract_imports(self.current_file)
        if imports:
            context_parts.append("# Imports:\n" + "\n".join(imports))

        # 2. Related files
        related_files = self._find_related_files()
        for file in related_files[:3]:  # Limit to 3 files
            content = self._get_file_summary(file)
            context_parts.append(f"# From {file}:\n{content}")

        # 3. Additional context files
        if additional_files:
            for file in additional_files:
                file_path = Path(file)
                if file_path.exists():
                    content = self._get_file_summary(file_path)
                    context_parts.append(f"# From {file}:\n{content}")

        return "\n\n".join(context_parts)

    def _extract_imports(self, file_path: Path) -> List[str]:
        """Extract import statements"""
        imports = []

        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('import ') or line.startswith('from '):
                        imports.append(line)
                    elif line and not line.startswith('#'):
                        # Stop at first non-import, non-comment line
                        break
        except:
            pass

        return imports

    def _find_related_files(self) -> List[Path]:
        """Find files likely related to current file"""

        related = []

        # Same directory
        for file in self.current_file.parent.glob('*.py'):
            if file != self.current_file:
                related.append(file)

        # Common patterns
        base_name = self.current_file.stem

        # Look for utils, helpers, config
        for pattern in ['utils', 'helpers', 'config', 'base']:
            for file in self.project_root.rglob(f'{pattern}.py'):
                if file not in related:
                    related.append(file)

        return related[:5]  # Limit to 5

    def _get_file_summary(self, file_path: Path, max_lines: int = 50) -> str:
        """Get summary of file (first N lines or function signatures)"""

        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()

            # Extract function/class signatures
            signatures = []
            for line in lines[:max_lines]:
                stripped = line.strip()
                if stripped.startswith('def ') or stripped.startswith('class '):
                    signatures.append(line.rstrip())

            if signatures:
                return "\n".join(signatures)
            else:
                # Return first N lines
                return "".join(lines[:max_lines])

        except:
            return ""
```

### Chat Command (Interactive)

```python
# src/cli/commands/chat.py

import click
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from ..engine.inference import InferenceEngine

@click.command()
@click.option('--project', is_flag=True, help='Include project context')
@click.argument('question', required=False)
@click.pass_context
def chat(ctx, project, question):
    """Interactive chat with code assistant"""

    engine = get_or_create_engine(ctx)

    if question:
        # Single question mode
        response = engine.generate(
            f"Question: {question}\n\nAnswer:",
            max_tokens=512,
            temperature=0.3
        )
        click.echo(response)
    else:
        # Interactive mode
        interactive_chat(engine, project)


def interactive_chat(engine, include_project_context):
    """Run interactive chat session"""

    session = PromptSession(history=FileHistory('.demiurgic_history'))

    click.echo("Demiurgic Chat (type 'exit' to quit)\n")

    conversation_history = []

    while True:
        try:
            # Get user input
            user_input = session.prompt('You: ')

            if user_input.lower() in ['exit', 'quit']:
                break

            # Build prompt with history
            prompt = build_chat_prompt(conversation_history, user_input)

            # Generate response
            click.echo("Assistant: ", nl=False)

            response_parts = []
            for token in engine._generate_stream(
                engine.tokenizer(prompt, return_tensors="pt").to(engine.device),
                max_tokens=512,
                temperature=0.3,
                top_p=0.95,
                stopping_criteria=None
            ):
                click.echo(token, nl=False)
                response_parts.append(token)

            response = "".join(response_parts)
            click.echo()  # Newline

            # Update history
            conversation_history.append({
                'user': user_input,
                'assistant': response
            })

        except KeyboardInterrupt:
            break
        except EOFError:
            break

    click.echo("\nGoodbye!")


def build_chat_prompt(history, current_input):
    """Build chat prompt with conversation history"""

    prompt_parts = ["You are a helpful coding assistant.\n"]

    # Add conversation history (last 5 turns)
    for turn in history[-5:]:
        prompt_parts.append(f"User: {turn['user']}")
        prompt_parts.append(f"Assistant: {turn['assistant']}")

    # Add current input
    prompt_parts.append(f"User: {current_input}")
    prompt_parts.append("Assistant:")

    return "\n\n".join(prompt_parts)
```

---

## Model Serving

### Option 1: Direct PyTorch (Simple)

```python
# Already shown above in InferenceEngine
# Pros: Simple, no dependencies
# Cons: Slower, more memory
```

### Option 2: vLLM (Recommended for Speed)

```python
# src/cli/engine/vllm_inference.py

from vllm import LLM, SamplingParams

class vLLMInferenceEngine:
    def __init__(self, model_path: str):
        """Initialize vLLM engine"""

        # vLLM provides highly optimized inference
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            dtype='auto',
            gpu_memory_utilization=0.9
        )

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.2) -> str:
        """Generate with vLLM"""

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.95
        )

        outputs = self.llm.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text

# Usage in CLI:
# Just replace InferenceEngine with vLLMInferenceEngine
```

### Option 3: llama.cpp (For CPU/Low Memory)

```python
# For running on CPU or limited GPU memory
# Use llama.cpp bindings

from llama_cpp import Llama

class LlamaCppInferenceEngine:
    def __init__(self, model_path: str):
        """Initialize llama.cpp engine"""

        # Load GGUF quantized model
        self.llm = Llama(
            model_path=model_path,  # .gguf file
            n_ctx=4096,
            n_threads=8,
            n_gpu_layers=35  # Offload layers to GPU
        )

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.2) -> str:
        """Generate with llama.cpp"""

        output = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95,
            echo=False
        )

        return output['choices'][0]['text']
```

---

## Optimization

### 1. Model Quantization

```python
# Convert to 8-bit/4-bit for faster inference

# 8-bit quantization (GPT-Q)
from transformers import AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM

model = AutoGPTQForCausalLM.from_pretrained(
    model_path,
    quantize_config={
        'bits': 8,
        'group_size': 128,
        'desc_act': False
    }
)

# 4-bit quantization (GGUF for llama.cpp)
# Use llama.cpp's quantization tools
# Reduces model size by 4-8x with minimal quality loss
```

### 2. Response Caching

```python
# src/cli/engine/cache.py

import hashlib
import json
from pathlib import Path

class ResponseCache:
    def __init__(self, cache_dir: str = "~/.demiurgic/cache"):
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, prompt: str, params: dict) -> str:
        """Get cached response"""
        cache_key = self._get_cache_key(prompt, params)
        cache_file = self.cache_dir / f"{cache_key}.json"

        if cache_file.exists():
            with open(cache_file, 'r') as f:
                data = json.load(f)
                return data['response']

        return None

    def set(self, prompt: str, params: dict, response: str):
        """Cache response"""
        cache_key = self._get_cache_key(prompt, params)
        cache_file = self.cache_dir / f"{cache_key}.json"

        with open(cache_file, 'w') as f:
            json.dump({
                'prompt': prompt,
                'params': params,
                'response': response
            }, f)

    def _get_cache_key(self, prompt: str, params: dict) -> str:
        """Generate cache key"""
        key_data = f"{prompt}{json.dumps(params, sort_keys=True)}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]
```

### 3. Lazy Loading

```python
# Only load model when first needed
# Shown in get_or_create_engine() above
```

---

## Usage Examples

### Installation

```bash
# Install CLI tool
pip install -e .

# Or create executable
pyinstaller --onefile src/cli/main.py --name demiurgic
```

### Configuration

```yaml
# ~/.demiurgic/config.yaml

model_path: ~/models/demiurgic-7b
cache_enabled: true
temperature: 0.2
max_tokens: 512

# Inference engine: 'pytorch', 'vllm', or 'llamacpp'
engine: vllm

# Context settings
max_context_files: 5
include_imports: true
```

### Example Workflows

**1. Complete function while coding:**

```bash
# In editor, save file
# Run completion
demiurgic complete main.py --line 42

# Output is shown, copy to editor
```

**2. Explain complex code:**

```bash
demiurgic explain algorithm.py::dijkstra

# Output:
# The dijkstra function implements Dijkstra's shortest path algorithm.
# It takes a graph and a starting node, then calculates the shortest
# path to all other nodes using a priority queue...
```

**3. Find and fix bugs:**

```bash
demiurgic fix buggy_code.py --interactive

# Interactive mode:
# Found issue: Line 15 - using 'i' instead of 'item' in loop
# Apply fix? [y/n]: y
# Fixed!
```

**4. Chat for help:**

```bash
demiurgic chat "How do I read a CSV file in Python?"

# Output:
# You can read a CSV file using the csv module:
#
# import csv
# with open('file.csv', 'r') as f:
#     reader = csv.reader(f)
#     for row in reader:
#         print(row)
```

---

## Distribution

### Package as PyPI Package

```python
# setup.py

from setuptools import setup, find_packages

setup(
    name='demiurgic',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'click>=8.0',
        'torch>=2.0',
        'transformers>=4.30',
        'prompt_toolkit>=3.0',
        'gitpython>=3.1',
        'pyyaml>=6.0',
    ],
    entry_points={
        'console_scripts': [
            'demiurgic=cli.main:cli',
        ],
    },
    python_requires='>=3.8',
)
```

### Standalone Binary

```bash
# Create standalone executable with PyInstaller
pyinstaller --onefile --name demiurgic src/cli/main.py

# Or use Nuitka for better performance
python -m nuitka --onefile --output-dir=dist src/cli/main.py
```

---

## Next Steps

1. **Implement Core CLI**: Start with basic commands
2. **Add Inference Engine**: Choose serving method
3. **Test with Quantized Model**: Optimize for speed
4. **Add Context Management**: Improve completions
5. **Package and Distribute**: Make it easy to install

## References

- [Click Documentation](https://click.palletsprojects.com/)
- [vLLM](https://github.com/vllm-project/vllm)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [Prompt Toolkit](https://python-prompt-toolkit.readthedocs.io/)
