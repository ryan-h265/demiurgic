"""
Feed-forward network layers for Demiurgic model.

Implements SwiGLU activation which has been shown to outperform
standard activations (ReLU, GELU) in large language models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit (SwiGLU) feed-forward network.

    SwiGLU uses the Swish activation (also called SiLU) with gating,
    which has been shown to improve performance in LLMs.

    Used in: PaLM, LLaMA, and other modern LLMs.

    The computation is: SwiGLU(x) = (Swish(xW1) ⊙ xW3)W2
    where ⊙ is element-wise multiplication.

    This requires ~2.67x hidden_size intermediate dimension instead
    of the standard 4x for regular FFN.

    Args:
        hidden_size: Input and output dimension
        intermediate_size: Dimension of the intermediate layer
        hidden_dropout: Dropout probability (default: 0.0)
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # Three linear projections (no bias for efficiency)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

        self.dropout = nn.Dropout(hidden_dropout) if hidden_dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SwiGLU transformation.

        Args:
            x: Input tensor [..., hidden_size]

        Returns:
            Output tensor [..., hidden_size]
        """
        # SwiGLU: (Swish(xW1) ⊙ xW3)W2
        # Swish is same as SiLU in PyTorch
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        intermediate = gate * up
        output = self.down_proj(intermediate)

        if self.dropout is not None:
            output = self.dropout(output)

        return output


class MLP(nn.Module):
    """
    Standard MLP with configurable activation function.

    This is an alternative to SwiGLU for models that want to use
    standard activations (GELU, ReLU, etc.).

    Args:
        hidden_size: Input and output dimension
        intermediate_size: Dimension of the intermediate layer
        activation: Activation function name ('gelu', 'relu', 'swiglu')
        hidden_dropout: Dropout probability
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str = "gelu",
        hidden_dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        if activation == "swiglu":
            # Use SwiGLU instead of standard MLP
            self.is_swiglu = True
            self.swiglu = SwiGLU(hidden_size, intermediate_size, hidden_dropout)
        else:
            self.is_swiglu = False
            self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=False)
            self.dropout = nn.Dropout(hidden_dropout) if hidden_dropout > 0 else None

            # Activation function
            if activation == "gelu":
                self.act = nn.GELU()
            elif activation == "relu":
                self.act = nn.ReLU()
            elif activation == "silu" or activation == "swish":
                self.act = nn.SiLU()
            else:
                raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply MLP transformation.

        Args:
            x: Input tensor [..., hidden_size]

        Returns:
            Output tensor [..., hidden_size]
        """
        if self.is_swiglu:
            return self.swiglu(x)

        # Standard MLP: x -> fc1 -> activation -> fc2 -> dropout
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)

        if self.dropout is not None:
            x = self.dropout(x)

        return x
