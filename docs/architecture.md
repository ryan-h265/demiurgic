# Model Architecture Specification

## Overview

This document details the architecture for Demiurgic, a GPT-based model optimized for code understanding and generation.

## Architecture Design

### Base Architecture: GPT-Style Decoder-Only Transformer

We use a decoder-only architecture similar to GPT-3/4 and CodeGen for several reasons:

1. **Autoregressive Generation**: Natural fit for code completion
2. **Proven Scalability**: Well-understood scaling laws
3. **Flexibility**: Single architecture handles all tasks
4. **Strong Transfer**: Pre-training transfers well to code tasks

### Model Configurations

#### Small Research Model (1-3B) - Validation Only
```
Parameters: 1.3B - 2.7B
Layers: 24 - 32
Hidden Size: 2048 - 2560
Attention Heads: 16 - 20
Context Length: 2048 - 4096
Purpose: Validate training pipeline, data quality
Cost: $500 - $2,000
```

#### Medium Scale Model (7-13B) - Recommended Starting Point
```
Parameters: 7B - 13B
Layers: 32 - 40
Hidden Size: 4096 - 5120
Attention Heads: 32 - 40
Head Dimension: 128
Context Length: 4096 - 8192
FFN Multiplier: 4x (SwiGLU: ~2.67x)
Vocabulary Size: 50,257 (GPT-2) or 32,000 (custom)
Max Position Embeddings: 8192
Purpose: Production-capable model
Cost: $8,000 - $20,000
Training Time: 3-5 weeks on 8x A100
```

#### Large Scale Model (30-70B) - Advanced Research
```
Parameters: 30B - 70B
Layers: 60 - 80
Hidden Size: 8192 - 10240
Attention Heads: 64 - 80
Head Dimension: 128
Context Length: 8192 - 16384
FFN Multiplier: 4x (SwiGLU: ~2.67x)
Vocabulary Size: 32,000 - 49,152
Max Position Embeddings: 16384
Purpose: State-of-the-art capabilities
Cost: $50,000 - $150,000
Training Time: 6-10 weeks on 32-64x A100
```

## Architectural Components

### 1. Tokenization

**Recommended: Byte-Pair Encoding (BPE) with Code-Specific Optimizations**

```python
# Key tokenization considerations for code:

Vocabulary Size: 32,000 - 50,000 tokens
Special Tokens:
  - <|endoftext|>     # Sequence separator
  - <|fim_prefix|>    # Fill-in-middle: prefix
  - <|fim_middle|>    # Fill-in-middle: middle
  - <|fim_suffix|>    # Fill-in-middle: suffix
  - <|file_separator|> # Between files in training
  - <|lang_python|>   # Language identifiers
  - <|lang_javascript|>
  - <|lang_rust|>
  # ... other languages

# Code-specific optimizations:
- Preserve whitespace structure (critical for Python)
- Keep common operators as single tokens (==, !=, ->, etc.)
- Preserve camelCase and snake_case boundaries
- Include common code keywords as atomic tokens
```

**Why not use existing tokenizers:**
- GPT-2 tokenizer (50,257 tokens): Acceptable, well-tested
- Custom tokenizer: Better for code-specific patterns
- SentencePiece: Good alternative, used by Llama

**Training the Tokenizer:**
```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# Sample from diverse code corpus
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

trainer = trainers.BpeTrainer(
    vocab_size=32000,
    special_tokens=["<|endoftext|>", "<|fim_prefix|>", ...],
    show_progress=True,
    min_frequency=2
)

# Train on representative code samples (10-100GB)
tokenizer.train(files=["code_samples.txt"], trainer=trainer)
```

### 2. Positional Encoding

**Recommended: Rotary Position Embeddings (RoPE)**

```python
# RoPE advantages for code:
- Extrapolates better to longer sequences
- More parameter-efficient than learned embeddings
- Used successfully in LLaMA, PaLM, GPT-NeoX
- Better for code's hierarchical structure

# Implementation:
from torch import nn
import torch

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=8192, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len_cached = max_position_embeddings

    def forward(self, x, seq_len):
        # Generate rotary embeddings
        # Apply to queries and keys in attention
        pass
```

**Alternative: ALiBi (Attention with Linear Biases)**
- More memory efficient
- Excellent length extrapolation
- Simpler implementation
- Used in BLOOM, MPT

### 3. Attention Mechanism

**Multi-Head Attention with Code-Specific Optimizations**

```python
# Standard configuration:
class CodeAttention(nn.Module):
    def __init__(self, config):
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.hidden_size = config.hidden_size

        # Projections
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # Optimizations
        self.use_flash_attention = True  # 2-4x speedup
        self.use_grouped_query = False   # Optional: GQA for larger models
```

**Flash Attention 2:**
- **Essential** for training efficiency
- 2-4x speedup, lower memory
- Exact attention computation
- No approximation trade-offs

**Grouped-Query Attention (GQA):**
- For 30B+ models to reduce memory
- Shares key/value heads across query heads
- 4-8 query heads per 1 key/value head
- Minimal quality loss, significant memory savings

### 4. Feed-Forward Networks

**SwiGLU Activation (Recommended)**

```python
class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit
    Used in PaLM, LLaMA - superior to ReLU/GELU for LLMs
    """
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)

    def forward(self, x):
        # SwiGLU(x) = (Swish(xW1) ⊙ xW3)W2
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

# Intermediate size calculation:
# For SwiGLU: intermediate_size = hidden_size * 8 / 3 ≈ 2.67x
# For standard FFN: intermediate_size = hidden_size * 4
```

**Why SwiGLU:**
- Better gradient flow
- Improved training dynamics
- State-of-the-art in modern LLMs
- Slightly higher FLOP cost but worth it

### 5. Layer Normalization

**RMSNorm (Recommended)**

```python
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    Simpler and faster than LayerNorm
    """
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states
```

**Pre-Normalization Layout:**
```
Input
  ↓
RMSNorm → Attention → Residual
  ↓                      ↑
RMSNorm → FFN ──────────┘
  ↓
Output
```

### 6. Complete Transformer Block

```python
class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = CausalSelfAttention(config)
        self.feed_forward = SwiGLU(config.hidden_size, config.intermediate_size)
        self.attention_norm = RMSNorm(config.hidden_size)
        self.ffn_norm = RMSNorm(config.hidden_size)

    def forward(self, x, attention_mask=None):
        # Pre-norm attention
        residual = x
        x = self.attention_norm(x)
        x = self.attention(x, attention_mask)
        x = x + residual

        # Pre-norm feed-forward
        residual = x
        x = self.ffn_norm(x)
        x = self.feed_forward(x)
        x = x + residual

        return x
```

## Code-Specific Architectural Features

### 1. Fill-in-the-Middle (FIM) Training

Enable mid-line code completion:

```python
# Transform training samples:
# Original:  [prefix] [middle] [suffix]
# Transformed: <|fim_prefix|>[prefix]<|fim_suffix|>[suffix]<|fim_middle|>[middle]

# Example:
# Original:
def calculate_sum(a, b):
    return a + b

# FIM format (50% of training):
<|fim_prefix|>def calculate_sum(a, b):
    return <|fim_suffix|><|fim_middle|>a + b
```

### 2. Repository-Level Context

**Strategy: Pack multiple files from same repository**

```python
# Training format:
<|file_separator|>
# File: src/utils.py
<|lang_python|>
def helper():
    pass

<|file_separator|>
# File: src/main.py
<|lang_python|>
from utils import helper
# ...
```

### 3. Syntax-Aware Attention Bias (Optional)

For advanced models, consider biasing attention based on AST structure:

```python
# Boost attention to:
- Function definitions when generating function calls
- Import statements when referencing modules
- Variable declarations when using variables

# Implementation: Add learned bias to attention scores
attention_scores = attention_scores + syntax_bias_matrix
```

## Model Configuration Files

### Example: 7B Model Config

```json
{
  "model_type": "demiurgic-gpt",
  "vocab_size": 32000,
  "hidden_size": 4096,
  "intermediate_size": 11008,
  "num_hidden_layers": 32,
  "num_attention_heads": 32,
  "num_key_value_heads": 32,
  "max_position_embeddings": 8192,
  "rms_norm_eps": 1e-6,
  "initializer_range": 0.02,
  "use_cache": true,
  "pad_token_id": 0,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "tie_word_embeddings": false,
  "rope_theta": 10000.0,
  "attention_dropout": 0.0,
  "attention_bias": false,
  "use_flash_attention_2": true,
  "activation_function": "swiglu"
}
```

## Scaling Laws and Compute Budgets

### Chinchilla Scaling Laws

For optimal compute allocation:
- **Tokens** should be ~20x **parameters**
- 7B model → ~140B tokens
- 13B model → ~260B tokens
- 30B model → ~600B tokens
- 70B model → ~1.4T tokens

### Estimated Training Compute

```
7B model:
  - FLOPs: ~1.5e21 (1.5 zettaFLOPs)
  - Time on 8x A100 80GB: ~3-4 weeks
  - Cost: ~$10,000-15,000

13B model:
  - FLOPs: ~2.8e21
  - Time on 16x A100: ~4-5 weeks
  - Cost: ~$20,000-30,000

70B model:
  - FLOPs: ~1.5e22
  - Time on 64x A100: ~8-10 weeks
  - Cost: ~$100,000-150,000
```

## Implementation Framework Recommendations

### PyTorch + HuggingFace Transformers

```python
# Recommended stack:
- PyTorch 2.0+ (with torch.compile)
- HuggingFace Transformers (model architecture)
- HuggingFace Accelerate (distributed training)
- DeepSpeed or FSDP (model parallelism)
- Flash Attention 2 (efficiency)
- Weights & Biases (experiment tracking)
```

### Alternative: JAX + Mesh TensorFlow
- Better for TPU training
- More complex setup
- Excellent scaling

## Next Steps

1. **Decide on model scale**: Start with 7B for budget constraints
2. **Implement base architecture**: Use provided code snippets
3. **Setup tokenizer training**: See [data.md](data.md)
4. **Configure training infrastructure**: See [training.md](training.md)
5. **Validate on small model**: 1-2B parameter proof of concept

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) - GPT-3
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) - LLaMA architecture
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) - Chinchilla scaling laws
- [Flash Attention 2](https://arxiv.org/abs/2307.08691) - Efficient attention
- [CodeGen](https://arxiv.org/abs/2203.13474) - Code-focused LLM
