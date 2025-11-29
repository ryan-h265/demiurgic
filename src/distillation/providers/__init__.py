"""Provider factory for creating teacher model clients."""

from pathlib import Path
from typing import Optional, Union

from .base import (
    TeacherProvider,
    ProviderConfig,
    ProviderType,
    GenerationMetrics,
)


def create_provider(
    provider_type: Union[str, ProviderType],
    model_name: str,
    api_key: Optional[str] = None,
    model_path: Optional[Path] = None,
    **kwargs
) -> TeacherProvider:
    """
    Factory function to create a teacher provider.

    Args:
        provider_type: Type of provider ("anthropic", "openai", or "local")
        model_name: Name of the model to use
        api_key: API key for cloud providers (required for anthropic/openai)
        model_path: Path to local model file (required for local provider)
        **kwargs: Additional configuration parameters

    Returns:
        TeacherProvider instance

    Raises:
        ValueError: If provider_type is invalid or required parameters are missing

    Examples:
        # Anthropic (Claude)
        provider = create_provider(
            "anthropic",
            model_name="claude-3-5-sonnet-20241022",
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )

        # OpenAI (GPT-4)
        provider = create_provider(
            "openai",
            model_name="gpt-4-turbo",
            api_key=os.getenv("OPENAI_API_KEY")
        )

        # Local (GGUF)
        provider = create_provider(
            "local",
            model_name="chatglm3-6b-q4",
            model_path=Path("models/chatglm3-6b.Q4_K_M.gguf")
        )
    """
    # Convert string to enum if needed
    if isinstance(provider_type, str):
        try:
            provider_type = ProviderType(provider_type.lower())
        except ValueError:
            raise ValueError(
                f"Invalid provider type: {provider_type}. "
                f"Must be one of: {[p.value for p in ProviderType]}"
            )

    # Create provider-specific config
    if provider_type == ProviderType.ANTHROPIC:
        if not api_key:
            raise ValueError("api_key is required for Anthropic provider")

        from .anthropic_client import AnthropicProvider, AnthropicConfig

        # Set default costs for Claude models
        input_cost, output_cost = _get_anthropic_costs(model_name)

        config = AnthropicConfig(
            provider_type=provider_type,
            model_name=model_name,
            api_key=api_key,
            input_token_cost=input_cost,
            output_token_cost=output_cost,
            **kwargs
        )
        return AnthropicProvider(config)

    elif provider_type == ProviderType.OPENAI:
        if not api_key:
            raise ValueError("api_key is required for OpenAI provider")

        from .openai_client import OpenAIProvider, OpenAIConfig

        # Set default costs for OpenAI models
        input_cost, output_cost = _get_openai_costs(model_name)

        config = OpenAIConfig(
            provider_type=provider_type,
            model_name=model_name,
            api_key=api_key,
            input_token_cost=input_cost,
            output_token_cost=output_cost,
            **kwargs
        )
        return OpenAIProvider(config)

    elif provider_type == ProviderType.LOCAL:
        if not model_path:
            raise ValueError("model_path is required for local provider")

        from .local_client import LocalProvider, LocalConfig

        config = LocalConfig(
            provider_type=provider_type,
            model_name=model_name,
            model_path=model_path,
            **kwargs
        )
        return LocalProvider(config)

    else:
        raise ValueError(f"Unsupported provider type: {provider_type}")


def _get_anthropic_costs(model_name: str) -> tuple[float, float]:
    """Get input/output costs per 1M tokens for Anthropic models."""
    costs = {
        "claude-3-5-sonnet-20241022": (3.0, 15.0),
        "claude-3-5-sonnet-20240620": (3.0, 15.0),
        "claude-3-opus-20240229": (15.0, 75.0),
        "claude-3-sonnet-20240229": (3.0, 15.0),
        "claude-3-haiku-20240307": (0.25, 1.25),
    }
    return costs.get(model_name, (3.0, 15.0))  # Default to Sonnet pricing


def _get_openai_costs(model_name: str) -> tuple[float, float]:
    """Get input/output costs per 1M tokens for OpenAI models."""
    costs = {
        "gpt-4-turbo": (10.0, 30.0),
        "gpt-4-turbo-preview": (10.0, 30.0),
        "gpt-4-1106-preview": (10.0, 30.0),
        "gpt-4": (30.0, 60.0),
        "gpt-4-0613": (30.0, 60.0),
        "gpt-3.5-turbo": (0.5, 1.5),
        "gpt-3.5-turbo-1106": (1.0, 2.0),
    }
    return costs.get(model_name, (10.0, 30.0))  # Default to GPT-4-turbo pricing


__all__ = [
    "create_provider",
    "TeacherProvider",
    "ProviderConfig",
    "ProviderType",
    "GenerationMetrics",
]
