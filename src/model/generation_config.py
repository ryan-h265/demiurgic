"""
Generation configuration for Demiurgic models.

Provides convenient presets and configuration management for text generation.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class GenerationConfig:
    """
    Configuration for text generation.

    Provides sensible defaults and presets for different use cases.
    """

    # Length control
    max_new_tokens: int = 100
    max_length: Optional[int] = None
    min_length: int = 0

    # Sampling parameters
    temperature: float = 0.8
    top_k: Optional[int] = 50
    top_p: Optional[float] = 0.95
    typical_p: Optional[float] = None
    do_sample: bool = True

    # Repetition penalties
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    # Other
    num_return_sequences: int = 1
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None

    @classmethod
    def greedy(cls) -> "GenerationConfig":
        """
        Greedy decoding (deterministic).

        Always picks the highest probability token.
        Best for tasks requiring consistency and factual accuracy.
        """
        return cls(
            temperature=0.0,
            do_sample=False,
            top_k=None,
            top_p=None,
            repetition_penalty=1.0,
        )

    @classmethod
    def balanced_code(cls, max_new_tokens: int = 512) -> "GenerationConfig":
        """
        Balanced sampling for code generation.

        Good default for most code completion tasks.
        Reduces repetition while maintaining quality.
        """
        return cls(
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.1,
            frequency_penalty=0.2,
            presence_penalty=0.0,
        )

    @classmethod
    def creative_code(cls, max_new_tokens: int = 512) -> "GenerationConfig":
        """
        More creative/diverse code generation.

        Use when you want more variety in solutions.
        Good for brainstorming or generating alternatives.
        """
        return cls(
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            top_k=100,
            top_p=0.9,
            repetition_penalty=1.15,
            frequency_penalty=0.3,
            presence_penalty=0.5,
        )

    @classmethod
    def precise_code(cls, max_new_tokens: int = 512) -> "GenerationConfig":
        """
        More conservative/precise code generation.

        Lower temperature for more predictable outputs.
        Good for production code or when correctness is critical.
        """
        return cls(
            max_new_tokens=max_new_tokens,
            temperature=0.6,
            top_k=40,
            top_p=0.95,
            repetition_penalty=1.05,
            frequency_penalty=0.1,
            presence_penalty=0.0,
        )

    @classmethod
    def natural_text(cls, max_new_tokens: int = 256) -> "GenerationConfig":
        """
        Natural text generation (comments, documentation, etc.).

        Higher temperature and presence penalty for more natural variation.
        """
        return cls(
            max_new_tokens=max_new_tokens,
            temperature=0.9,
            top_k=50,
            top_p=0.92,
            repetition_penalty=1.2,
            frequency_penalty=0.0,
            presence_penalty=0.6,
        )

    @classmethod
    def completion_fim(cls, max_new_tokens: int = 256) -> "GenerationConfig":
        """
        Fill-in-the-middle completion.

        Optimized for completing code in the middle of a file.
        Lower max_tokens and higher precision.
        """
        return cls(
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_k=40,
            top_p=0.95,
            repetition_penalty=1.1,
            frequency_penalty=0.15,
            presence_penalty=0.0,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for passing to generate()."""
        return {
            'max_new_tokens': self.max_new_tokens,
            'max_length': self.max_length,
            'min_length': self.min_length,
            'temperature': self.temperature,
            'top_k': self.top_k,
            'top_p': self.top_p,
            'typical_p': self.typical_p,
            'do_sample': self.do_sample,
            'repetition_penalty': self.repetition_penalty,
            'frequency_penalty': self.frequency_penalty,
            'presence_penalty': self.presence_penalty,
            'num_return_sequences': self.num_return_sequences,
            'pad_token_id': self.pad_token_id,
            'eos_token_id': self.eos_token_id,
        }
