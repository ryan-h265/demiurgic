"""OpenAI GPT provider for generating training data."""

import asyncio
from dataclasses import dataclass
from typing import Dict, Optional

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

from .base import TeacherProvider, ProviderConfig, ProviderType


@dataclass
class OpenAIConfig(ProviderConfig):
    """Configuration for OpenAI GPT provider."""

    api_key: str = ""

    def __post_init__(self):
        """Set provider type to OpenAI."""
        self.provider_type = ProviderType.OPENAI


class OpenAIProvider(TeacherProvider):
    """Provider for OpenAI GPT models."""

    def __init__(self, config: OpenAIConfig):
        """Initialize the OpenAI provider."""
        if AsyncOpenAI is None:
            raise ImportError(
                "openai package not installed. "
                "Install with: pip install openai>=1.0.0"
            )

        super().__init__(config)
        self.config: OpenAIConfig = config
        self.client = AsyncOpenAI(api_key=config.api_key)

    async def generate_single(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, any]:
        """
        Generate a single response using GPT.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt (overrides config)
            **kwargs: Additional generation parameters

        Returns:
            Dict with response, token counts, and cost
        """
        system = system_prompt or self.config.system_prompt or "You are a helpful coding assistant."

        # Override config params with kwargs
        temperature = kwargs.get("temperature", self.config.temperature)
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)
        top_p = kwargs.get("top_p", self.config.top_p)

        # Build messages
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ]

        # Retry logic with exponential backoff
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                )

                # Extract response text
                response_text = response.choices[0].message.content

                # Calculate token counts and cost
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                cost = self._calculate_cost(input_tokens, output_tokens)

                return {
                    "response": response_text,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cost": cost,
                }

            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    # Exponential backoff
                    delay = self.config.retry_delay * (2 ** attempt)
                    print(f"Error calling OpenAI API (attempt {attempt + 1}/{self.config.max_retries}): {e}")
                    print(f"Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    print(f"Failed after {self.config.max_retries} attempts: {e}")

        # If all retries failed, raise the last error
        raise last_error

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate the cost of a generation in USD."""
        input_cost = (input_tokens / 1_000_000) * self.config.input_token_cost
        output_cost = (output_tokens / 1_000_000) * self.config.output_token_cost
        return input_cost + output_cost

    def estimate_cost(
        self,
        num_examples: int,
        avg_prompt_tokens: int = 100,
        avg_response_tokens: int = 500
    ) -> float:
        """
        Estimate the cost of generating examples.

        Args:
            num_examples: Number of examples to generate
            avg_prompt_tokens: Average tokens per prompt
            avg_response_tokens: Average tokens per response

        Returns:
            Estimated cost in USD
        """
        total_input_tokens = num_examples * avg_prompt_tokens
        total_output_tokens = num_examples * avg_response_tokens

        input_cost = (total_input_tokens / 1_000_000) * self.config.input_token_cost
        output_cost = (total_output_tokens / 1_000_000) * self.config.output_token_cost

        return input_cost + output_cost


__all__ = ["OpenAIProvider", "OpenAIConfig"]
