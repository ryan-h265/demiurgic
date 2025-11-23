"""
Model configuration for Demiurgic code model.

This module defines the configuration dataclass for the GPT-style transformer model.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DemiurgicConfig:
    """
    Configuration class for Demiurgic model.

    Based on GPT-style decoder-only transformer with code-specific optimizations:
    - RoPE positional embeddings for better length extrapolation
    - RMSNorm for efficiency
    - SwiGLU activation for better performance
    - Optional Flash Attention 2 for training efficiency
    - Optional Grouped-Query Attention for larger models
    """

    # Model architecture
    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008  # For SwiGLU: ~2.67x hidden_size
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: Optional[int] = None  # For GQA; None = MHA
    max_position_embeddings: int = 8192

    # Normalization
    rms_norm_eps: float = 1e-6

    # Initialization
    initializer_range: float = 0.02

    # Attention
    attention_dropout: float = 0.0
    attention_bias: bool = False
    use_flash_attention_2: bool = True

    # RoPE settings
    rope_theta: float = 10000.0
    rope_scaling: Optional[dict] = None

    # Activation function
    activation_function: str = "swiglu"  # swiglu, gelu, relu

    # Dropout
    hidden_dropout: float = 0.0
    embedding_dropout: float = 0.0

    # Special tokens
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    # Training
    use_cache: bool = True
    tie_word_embeddings: bool = False

    # Code-specific settings
    fim_tokens: dict = field(default_factory=lambda: {
        "prefix": 3,   # <|fim_prefix|>
        "middle": 4,   # <|fim_middle|>
        "suffix": 5,   # <|fim_suffix|>
    })

    def __post_init__(self):
        """Validate and compute derived values."""
        # Set num_key_value_heads to num_attention_heads if not specified (standard MHA)
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        # Validate attention heads
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )

        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
                f"num_key_value_heads ({self.num_key_value_heads})"
            )

    @property
    def head_dim(self) -> int:
        """Dimension of each attention head."""
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_dict(cls, config_dict: dict) -> "DemiurgicConfig":
        """Create config from dictionary."""
        return cls(**config_dict)

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }


# Predefined configurations
def get_1b_config() -> DemiurgicConfig:
    """1B parameter model for testing and validation."""
    return DemiurgicConfig(
        vocab_size=32000,
        hidden_size=2048,
        intermediate_size=5504,  # ~2.67x for SwiGLU
        num_hidden_layers=24,
        num_attention_heads=16,
        max_position_embeddings=4096,
    )


def get_7b_config() -> DemiurgicConfig:
    """7B parameter model (recommended starting point)."""
    return DemiurgicConfig(
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        max_position_embeddings=8192,
    )


def get_13b_config() -> DemiurgicConfig:
    """13B parameter model."""
    return DemiurgicConfig(
        vocab_size=32000,
        hidden_size=5120,
        intermediate_size=13824,
        num_hidden_layers=40,
        num_attention_heads=40,
        max_position_embeddings=8192,
    )


def get_70b_config() -> DemiurgicConfig:
    """70B parameter model with Grouped-Query Attention."""
    return DemiurgicConfig(
        vocab_size=32000,
        hidden_size=8192,
        intermediate_size=22016,
        num_hidden_layers=80,
        num_attention_heads=64,
        num_key_value_heads=8,  # GQA: 8 query heads per 1 KV head
        max_position_embeddings=8192,
    )


def get_100m_config() -> DemiurgicConfig:
    """100M parameter model for laptop testing."""
    return DemiurgicConfig(
        vocab_size=32000,
        hidden_size=768,
        intermediate_size=2048,
        num_hidden_layers=12,
        num_attention_heads=12,
        max_position_embeddings=2048,
        use_flash_attention_2=False,  # Disable for CPU
    )


def get_350m_config() -> DemiurgicConfig:
    """350M parameter model for laptop training."""
    return DemiurgicConfig(
        vocab_size=32000,
        hidden_size=1024,
        intermediate_size=2752,
        num_hidden_layers=24,
        num_attention_heads=16,
        max_position_embeddings=2048,
        use_flash_attention_2=False,  # Disable for CPU
    )
