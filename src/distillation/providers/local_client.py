"""Local GGUF model provider using llama.cpp."""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

from .base import TeacherProvider, ProviderConfig, ProviderType


@dataclass
class LocalConfig(ProviderConfig):
    """Configuration for local GGUF model provider."""

    model_path: Path = None
    n_ctx: int = 4096
    n_gpu_layers: int = 0  # 0 = CPU only, -1 = all layers on GPU

    def __post_init__(self):
        """Set provider type to Local."""
        self.provider_type = ProviderType.LOCAL
        # Local models are free
        self.input_token_cost = 0.0
        self.output_token_cost = 0.0


class LocalProvider(TeacherProvider):
    """Provider for local GGUF models using llama.cpp."""

    def __init__(self, config: LocalConfig):
        """Initialize the local provider."""
        if Llama is None:
            raise ImportError(
                "llama-cpp-python not installed. "
                "Install with: pip install llama-cpp-python>=0.2.90"
            )

        super().__init__(config)
        self.config: LocalConfig = config

        # Load the model
        print(f"Loading local model from {config.model_path}...")
        self.client = Llama(
            model_path=str(config.model_path),
            n_ctx=config.n_ctx,
            n_gpu_layers=config.n_gpu_layers,
            embedding=False,
        )
        print(f"Model loaded successfully!")

    async def generate_single(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, any]:
        """
        Generate a single response using local GGUF model.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt (overrides config)
            **kwargs: Additional generation parameters

        Returns:
            Dict with response, token counts (0 for local), and cost (0.0)
        """
        system = system_prompt or self.config.system_prompt or "You are a helpful coding assistant."

        # Override config params with kwargs
        temperature = kwargs.get("temperature", self.config.temperature)
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)
        top_p = kwargs.get("top_p", self.config.top_p)

        # Build ChatGLM3-style prompt
        chat_prompt = self._build_chatglm3_prompt(system, prompt)

        # Run generation in thread pool (llama.cpp is synchronous)
        loop = asyncio.get_event_loop()
        output = await loop.run_in_executor(
            None,
            lambda: self.client(
                chat_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        )

        # Extract response text
        response_text = output["choices"][0]["text"].strip()

        # Note: llama.cpp doesn't provide accurate token counts in the same way
        # We could estimate, but for now just return 0
        return {
            "response": response_text,
            "input_tokens": 0,  # llama.cpp doesn't report this accurately
            "output_tokens": 0,  # llama.cpp doesn't report this accurately
            "cost": 0.0,  # Local generation is free
        }

    def _build_chatglm3_prompt(self, system: str, user_message: str) -> str:
        """Build ChatGLM3-formatted prompt."""
        # Import here to avoid circular dependency
        from ...cli import build_prompt

        return build_prompt(
            system,
            [{"role": "user", "content": user_message}]
        )

    def estimate_cost(
        self,
        num_examples: int,
        avg_prompt_tokens: int = 100,
        avg_response_tokens: int = 500
    ) -> float:
        """
        Estimate the cost of generating examples (always $0 for local).

        Args:
            num_examples: Number of examples to generate
            avg_prompt_tokens: Average tokens per prompt (ignored)
            avg_response_tokens: Average tokens per response (ignored)

        Returns:
            0.0 (local generation is free)
        """
        return 0.0


__all__ = ["LocalProvider", "LocalConfig"]
