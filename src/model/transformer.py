"""
Transformer block for Demiurgic model.

Implements the complete transformer decoder block with pre-normalization,
combining attention, feed-forward, and residual connections.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .attention import DemiurgicAttention
from .feedforward import MLP
from .normalization import RMSNorm


class DemiurgicDecoderLayer(nn.Module):
    """
    Single transformer decoder layer.

    Uses pre-normalization layout (LayerNorm before attention/FFN):
        x -> RMSNorm -> Attention -> Residual
          -> RMSNorm -> FFN -> Residual

    This is more stable for training than post-normalization and is
    used in modern LLMs like LLaMA, GPT-NeoX, etc.

    Args:
        config: Model configuration
        layer_idx: Index of this layer
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        # Self-attention
        self.self_attn = DemiurgicAttention(config, layer_idx)

        # Feed-forward network
        self.mlp = MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            activation=config.activation_function,
            hidden_dropout=config.hidden_dropout,
        )

        # Pre-normalization layers
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Forward pass for transformer layer.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask
            position_ids: Optional position indices
            past_key_value: Optional cached key/value
            output_attentions: Whether to return attention weights
            use_cache: Whether to cache key/value

        Returns:
            Tuple of (hidden_states, present_key_value)
            If output_attentions, also returns attention weights
        """
        residual = hidden_states

        # Pre-norm attention
        hidden_states = self.input_layernorm(hidden_states)

        # Self-attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        # Residual connection
        hidden_states = residual + hidden_states

        # Pre-norm feed-forward
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        # Residual connection
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
