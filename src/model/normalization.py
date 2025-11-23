"""
Normalization layers for Demiurgic model.

Implements RMSNorm (Root Mean Square Layer Normalization), which is simpler
and more efficient than LayerNorm while providing similar performance.
Used in LLaMA, GPT-NeoX, and other modern LLMs.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    RMSNorm normalizes using only the root mean square statistic,
    eliminating the mean centering step from LayerNorm. This makes
    it more efficient while maintaining similar effectiveness.

    Args:
        hidden_size: Dimension of the input
        eps: Small constant for numerical stability
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm to input tensor.

        Args:
            hidden_states: Input tensor of shape [..., hidden_size]

        Returns:
            Normalized tensor of same shape as input
        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)

        # Compute variance (mean of squares)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)

        # Normalize and scale
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # Apply learnable scale and convert back to input dtype
        return self.weight * hidden_states.to(input_dtype)
