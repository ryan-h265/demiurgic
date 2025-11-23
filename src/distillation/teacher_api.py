"""
Teacher API integration for knowledge distillation.

Supports multiple providers:
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude)
- OpenAI-compatible APIs (local models, Together, etc.)
"""

import os
import json
import asyncio
import aiohttp
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
import time


@dataclass
class TeacherConfig:
    """Configuration for teacher model API."""
    provider: str  # 'openai', 'anthropic', 'openai-compatible'
    model: str  # e.g., 'gpt-4', 'claude-3-sonnet-20240229'
    api_key: str
    base_url: Optional[str] = None  # For OpenAI-compatible APIs
    max_concurrent: int = 5
    rate_limit_delay: float = 1.0  # Seconds between requests
    max_retries: int = 3
    timeout: int = 60


class TeacherAPI:
    """
    Unified teacher API interface.

    Handles API calls to teacher models for generating training data.
    Supports rate limiting, retries, and concurrent requests.
    """

    def __init__(self, config: TeacherConfig):
        self.config = config
        self.semaphore = asyncio.Semaphore(config.max_concurrent)
        self.last_request_time = 0

    async def generate_one(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Union[str, int]]:
        """
        Generate a single response from the teacher model.

        Args:
            prompt: The user prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            system_prompt: Optional system prompt

        Returns:
            Dict with 'content', 'tokens_used', 'cost_estimate'
        """
        async with self.semaphore:
            # Rate limiting
            await self._rate_limit()

            # Retry logic
            for attempt in range(self.config.max_retries):
                try:
                    if self.config.provider == 'openai':
                        return await self._generate_openai(
                            prompt, temperature, max_tokens, system_prompt
                        )
                    elif self.config.provider == 'anthropic':
                        return await self._generate_anthropic(
                            prompt, temperature, max_tokens, system_prompt
                        )
                    elif self.config.provider == 'openai-compatible':
                        return await self._generate_openai_compatible(
                            prompt, temperature, max_tokens, system_prompt
                        )
                    else:
                        raise ValueError(f"Unsupported provider: {self.config.provider}")

                except Exception as e:
                    if attempt == self.config.max_retries - 1:
                        raise
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff

    async def generate_batch(
        self,
        prompts: List[str],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, Union[str, int]]]:
        """
        Generate responses for a batch of prompts.

        Args:
            prompts: List of prompts
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            system_prompt: Optional system prompt

        Returns:
            List of response dicts
        """
        tasks = [
            self.generate_one(prompt, temperature, max_tokens, system_prompt)
            for prompt in prompts
        ]
        return await asyncio.gather(*tasks)

    async def _rate_limit(self):
        """Apply rate limiting."""
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.config.rate_limit_delay:
            await asyncio.sleep(self.config.rate_limit_delay - elapsed)
        self.last_request_time = time.time()

    async def _generate_openai(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        system_prompt: Optional[str],
    ) -> Dict:
        """Generate using OpenAI API."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.config.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                timeout=aiohttp.ClientTimeout(total=self.config.timeout),
            ) as response:
                response.raise_for_status()
                data = await response.json()

                content = data['choices'][0]['message']['content']
                tokens_used = data['usage']['total_tokens']

                # Estimate cost (approximate, update based on current pricing)
                cost = self._estimate_cost_openai(
                    data['usage']['prompt_tokens'],
                    data['usage']['completion_tokens']
                )

                return {
                    'content': content,
                    'tokens_used': tokens_used,
                    'cost_estimate': cost,
                }

    async def _generate_anthropic(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        system_prompt: Optional[str],
    ) -> Dict:
        """Generate using Anthropic Claude API."""
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": self.config.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
            }

            if system_prompt:
                payload["system"] = system_prompt

            async with session.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.config.api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout),
            ) as response:
                response.raise_for_status()
                data = await response.json()

                content = data['content'][0]['text']
                tokens_used = data['usage']['input_tokens'] + data['usage']['output_tokens']

                # Estimate cost
                cost = self._estimate_cost_anthropic(
                    data['usage']['input_tokens'],
                    data['usage']['output_tokens']
                )

                return {
                    'content': content,
                    'tokens_used': tokens_used,
                    'cost_estimate': cost,
                }

    async def _generate_openai_compatible(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        system_prompt: Optional[str],
    ) -> Dict:
        """Generate using OpenAI-compatible API (local models, etc.)."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        base_url = self.config.base_url or "http://localhost:8000"
        url = f"{base_url}/v1/chat/completions"

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                headers={"Content-Type": "application/json"},
                json={
                    "model": self.config.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                timeout=aiohttp.ClientTimeout(total=self.config.timeout),
            ) as response:
                response.raise_for_status()
                data = await response.json()

                content = data['choices'][0]['message']['content']
                tokens_used = data.get('usage', {}).get('total_tokens', 0)

                return {
                    'content': content,
                    'tokens_used': tokens_used,
                    'cost_estimate': 0.0,  # Free for local models
                }

    def _estimate_cost_openai(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for OpenAI API (approximate pricing)."""
        # Prices as of 2024 (per 1K tokens)
        prices = {
            'gpt-4': {'input': 0.03, 'output': 0.06},
            'gpt-4-turbo': {'input': 0.01, 'output': 0.03},
            'gpt-3.5-turbo': {'input': 0.0005, 'output': 0.0015},
        }

        model_price = prices.get(self.config.model, {'input': 0.01, 'output': 0.03})

        input_cost = (input_tokens / 1000) * model_price['input']
        output_cost = (output_tokens / 1000) * model_price['output']

        return input_cost + output_cost

    def _estimate_cost_anthropic(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for Anthropic API (approximate pricing)."""
        # Prices as of 2024 (per 1K tokens)
        prices = {
            'claude-3-opus-20240229': {'input': 0.015, 'output': 0.075},
            'claude-3-sonnet-20240229': {'input': 0.003, 'output': 0.015},
            'claude-3-haiku-20240307': {'input': 0.00025, 'output': 0.00125},
        }

        model_price = prices.get(self.config.model, {'input': 0.003, 'output': 0.015})

        input_cost = (input_tokens / 1000) * model_price['input']
        output_cost = (output_tokens / 1000) * model_price['output']

        return input_cost + output_cost


def create_teacher_api(
    provider: str = 'openai',
    model: str = 'gpt-4-turbo',
    api_key: Optional[str] = None,
    **kwargs
) -> TeacherAPI:
    """
    Convenient factory function for creating teacher API.

    Args:
        provider: 'openai', 'anthropic', or 'openai-compatible'
        model: Model name
        api_key: API key (or set via environment variable)
        **kwargs: Additional config options

    Returns:
        TeacherAPI instance

    Example:
        >>> teacher = create_teacher_api(
        ...     provider='openai',
        ...     model='gpt-4-turbo',
        ...     api_key=os.getenv('OPENAI_API_KEY')
        ... )
        >>> response = asyncio.run(teacher.generate_one("Write a Python function"))
    """
    # Get API key from environment if not provided
    if api_key is None:
        if provider == 'openai':
            api_key = os.getenv('OPENAI_API_KEY')
        elif provider == 'anthropic':
            api_key = os.getenv('ANTHROPIC_API_KEY')

        if api_key is None:
            raise ValueError(f"API key required. Set {provider.upper()}_API_KEY environment variable.")

    config = TeacherConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        **kwargs
    )

    return TeacherAPI(config)
