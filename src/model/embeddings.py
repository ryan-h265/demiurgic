"""
Embedding layers for Demiurgic model.

Implements Rotary Position Embeddings (RoPE) which provides better
extrapolation to longer sequences than learned positional embeddings.
"""

import torch
import torch.nn as nn
from typing import Tuple


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    RoPE encodes positional information by rotating query and key vectors
    in complex space. This provides better length extrapolation and is
    more parameter-efficient than learned embeddings.

    Used in: LLaMA, PaLM, GPT-NeoX, and other modern LLMs.

    Args:
        dim: Dimension of each attention head
        max_position_embeddings: Maximum sequence length
        base: Base for computing inverse frequencies (theta in paper)
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 8192,
        base: float = 10000.0,
        device=None,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Compute inverse frequencies: 1 / (base ^ (2i / dim)) for i in [0, dim/2)
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build cos/sin cache for efficiency
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len: int, device, dtype):
        """Pre-compute cos/sin values for all positions."""
        self.max_seq_len_cached = seq_len

        # Create position indices [0, 1, 2, ..., seq_len-1]
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        # Compute outer product: t * inv_freq
        # Shape: [seq_len, dim/2]
        freqs = torch.outer(t, self.inv_freq)

        # Concatenate to get [seq_len, dim]
        # Different from paper, but works the same with rotary embedding
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get cos and sin values for rotary embedding.

        Args:
            x: Input tensor (used only for device/dtype inference)
            seq_len: Sequence length (if None, uses x.shape[1])

        Returns:
            Tuple of (cos, sin) tensors
        """
        if seq_len is None:
            seq_len = x.shape[1]

        # If sequence is longer than cache, extend it
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate half the hidden dims of the input.

    This is a helper for applying rotary embeddings.
    Splits the last dimension in half and swaps with negation.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to query and key tensors.

    Args:
        q: Query tensor [batch_size, num_heads, seq_len, head_dim]
        k: Key tensor [batch_size, num_heads, seq_len, head_dim]
        cos: Cosine values [seq_len, head_dim]
        sin: Sine values [seq_len, head_dim]
        position_ids: Optional position indices [batch_size, seq_len]

    Returns:
        Tuple of rotated (query, key) tensors
    """
    # Handle position_ids if provided (for cached key/values)
    if position_ids is not None:
        cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        sin = sin[position_ids].unsqueeze(1)
    else:
        # Unsqueeze for broadcasting: [seq_len, dim] -> [1, 1, seq_len, dim]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

    # Apply rotation using the formula:
    # q_rotated = q * cos + rotate_half(q) * sin
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed
