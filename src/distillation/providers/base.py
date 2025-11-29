"""Abstract base class for teacher model providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class ProviderType(Enum):
    """Supported provider types."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    LOCAL = "local"


@dataclass
class GenerationMetrics:
    """Metrics for tracking generation costs and performance."""

    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_cost: float = 0.0
    num_requests: int = 0
    num_errors: int = 0

    def add(self, input_tok: int, output_tok: int, cost: float) -> None:
        """Add metrics from a single generation."""
        self.input_tokens += input_tok
        self.output_tokens += output_tok
        self.total_tokens += input_tok + output_tok
        self.total_cost += cost
        self.num_requests += 1

    def add_error(self) -> None:
        """Record an error."""
        self.num_errors += 1

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging."""
        return {
            "total_tokens": self.total_tokens,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_cost": self.total_cost,
            "num_requests": self.num_requests,
            "num_errors": self.num_errors,
            "avg_cost_per_request": self.total_cost / max(1, self.num_requests),
        }


@dataclass
class ProviderConfig:
    """Base configuration for all providers."""

    provider_type: ProviderType
    model_name: str
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 2048
    system_prompt: Optional[str] = None

    # Rate limiting
    max_concurrent: int = 5
    rate_limit_delay: float = 0.0  # seconds between requests

    # Retry configuration
    max_retries: int = 3
    retry_delay: float = 1.0  # initial delay, exponential backoff

    # Cost tracking
    input_token_cost: float = 0.0  # cost per 1M input tokens
    output_token_cost: float = 0.0  # cost per 1M output tokens


class TeacherProvider(ABC):
    """Abstract base class for all teacher model providers."""

    def __init__(self, config: ProviderConfig):
        """Initialize the provider with configuration."""
        self.config = config
        self.metrics = GenerationMetrics()

    @abstractmethod
    async def generate_single(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, any]:
        """
        Generate a single response asynchronously.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt (overrides config)
            **kwargs: Additional generation parameters

        Returns:
            Dict with keys:
                - "response": str - The generated text
                - "input_tokens": int - Number of input tokens
                - "output_tokens": int - Number of output tokens
                - "cost": float - Cost of this generation
        """
        pass

    async def generate_batch(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> List[Dict[str, str]]:
        """
        Generate responses for multiple prompts concurrently.

        Args:
            prompts: List of user prompts
            system_prompt: Optional system prompt (overrides config)
            **kwargs: Additional generation parameters

        Returns:
            List of dicts with keys:
                - "prompt": str - The original prompt
                - "response": str - The generated response
        """
        import asyncio

        # Generate all prompts concurrently with concurrency limit
        semaphore = asyncio.Semaphore(self.config.max_concurrent)

        async def generate_with_limit(prompt: str) -> Dict[str, str]:
            async with semaphore:
                result = await self.generate_single(prompt, system_prompt, **kwargs)

                # Track metrics
                self.metrics.add(
                    result.get("input_tokens", 0),
                    result.get("output_tokens", 0),
                    result.get("cost", 0.0)
                )

                # Rate limiting
                if self.config.rate_limit_delay > 0:
                    await asyncio.sleep(self.config.rate_limit_delay)

                return {
                    "prompt": prompt,
                    "response": result["response"]
                }

        results = await asyncio.gather(
            *[generate_with_limit(p) for p in prompts],
            return_exceptions=True
        )

        # Filter out exceptions and log errors
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.metrics.add_error()
                print(f"Error generating response for prompt {i}: {result}")
            else:
                valid_results.append(result)

        return valid_results

    def get_metrics(self) -> Dict[str, float]:
        """Get current generation metrics."""
        return self.metrics.to_dict()

    def reset_metrics(self) -> None:
        """Reset generation metrics."""
        self.metrics = GenerationMetrics()

    @abstractmethod
    def estimate_cost(self, num_examples: int, avg_prompt_tokens: int = 100, avg_response_tokens: int = 500) -> float:
        """
        Estimate the cost of generating a given number of examples.

        Args:
            num_examples: Number of examples to generate
            avg_prompt_tokens: Average tokens per prompt
            avg_response_tokens: Average tokens per response

        Returns:
            Estimated cost in USD
        """
        pass


__all__ = [
    "TeacherProvider",
    "ProviderConfig",
    "ProviderType",
    "GenerationMetrics",
]
