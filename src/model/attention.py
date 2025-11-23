"""
Attention mechanisms for Demiurgic model.

Implements multi-head attention with:
- Rotary Position Embeddings (RoPE)
- Optional Flash Attention 2 for efficiency
- Optional Grouped-Query Attention (GQA) for larger models
- Causal masking for autoregressive generation
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple

from .embeddings import RotaryEmbedding, apply_rotary_pos_emb

# Try to import Flash Attention 2
try:
    from flash_attn import flash_attn_func
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key/value tensors for Grouped-Query Attention.

    This is equivalent to torch.repeat_interleave(x, dim=1, repeats=n_rep).
    Used to expand key/value heads to match query heads in GQA.

    Args:
        hidden_states: [batch, num_kv_heads, seq_len, head_dim]
        n_rep: Number of repetitions

    Returns:
        Repeated tensor [batch, num_kv_heads * n_rep, seq_len, head_dim]
    """
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states

    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


class DemiurgicAttention(nn.Module):
    """
    Multi-head attention with RoPE and optional Flash Attention.

    Supports:
    - Standard Multi-Head Attention (MHA)
    - Grouped-Query Attention (GQA) for larger models
    - Flash Attention 2 for 2-4x speedup during training
    - Causal masking for autoregressive generation

    Args:
        config: Model configuration
        layer_idx: Index of this layer (for caching)
    """

    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        # Check if we're using Grouped-Query Attention
        self.is_gqa = self.num_key_value_heads != self.num_heads

        # Validate dimensions
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads "
                f"(got hidden_size={self.hidden_size} and num_heads={self.num_heads})"
            )

        # Query, Key, Value projections
        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )

        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

        # Dropout
        self.attention_dropout = config.attention_dropout

        # Flash Attention flag
        self.use_flash_attention = (
            config.use_flash_attention_2 and FLASH_ATTENTION_AVAILABLE
        )

        if config.use_flash_attention_2 and not FLASH_ATTENTION_AVAILABLE:
            print(
                "Warning: Flash Attention 2 requested but not available. "
                "Falling back to standard attention. "
                "Install with: pip install flash-attn --no-build-isolation"
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass for attention.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, 1, seq_len, seq_len] or None
            position_ids: [batch_size, seq_len] or None
            past_key_value: Cached (key, value) tuple or None
            output_attentions: Whether to return attention weights
            use_cache: Whether to return key/value for caching

        Returns:
            Tuple of (output, attention_weights, past_key_value)
        """
        bsz, q_len, _ = hidden_states.size()

        # Project to Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape to [batch, seq_len, num_heads, head_dim]
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        # Transpose to [batch, num_heads, seq_len, head_dim]
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # Get sequence length accounting for cache
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        # Apply rotary embeddings
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        # Handle cached key/values
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # Expand key/value heads for Grouped-Query Attention
        if self.is_gqa:
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Choose attention implementation
        if self.use_flash_attention and not output_attentions:
            attn_output = self._flash_attention_forward(
                query_states, key_states, value_states, attention_mask
            )
            attn_weights = None
        else:
            attn_output, attn_weights = self._standard_attention_forward(
                query_states, key_states, value_states, attention_mask
            )

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass using Flash Attention 2.

        Flash Attention requires:
        - Input shape: [batch, seq_len, num_heads, head_dim]
        - Causal flag for autoregressive models
        - Dropout probability
        """
        # Transpose back to [batch, seq_len, num_heads, head_dim] for Flash Attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # Flash Attention expects inputs in BF16 or FP16
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            query_states = query_states.to(torch.float16)
            key_states = key_states.to(torch.float16)
            value_states = value_states.to(torch.float16)

        # Apply Flash Attention
        attn_output = flash_attn_func(
            query_states,
            key_states,
            value_states,
            dropout_p=self.attention_dropout if self.training else 0.0,
            causal=True,  # Autoregressive causal masking
        )

        # Convert back to original dtype
        attn_output = attn_output.to(input_dtype)

        return attn_output

    def _standard_attention_forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Standard scaled dot-product attention.

        Args:
            query_states: [batch, num_heads, seq_len, head_dim]
            key_states: [batch, num_heads, seq_len, head_dim]
            value_states: [batch, num_heads, seq_len, head_dim]
            attention_mask: [batch, 1, seq_len, seq_len] or None

        Returns:
            Tuple of (attention_output, attention_weights)
        """
        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
            self.head_dim
        )

        # Apply attention mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax and dropout
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )

        # Compute output
        attn_output = torch.matmul(attn_weights, value_states)

        return attn_output, attn_weights
