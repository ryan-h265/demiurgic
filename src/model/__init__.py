"""
Demiurgic model package.

Exports main model classes and configuration.
"""

from .config import (
    DemiurgicConfig,
    get_100m_config,
    get_350m_config,
    get_1b_config,
    get_7b_config,
    get_13b_config,
    get_70b_config,
)
from .model import (
    DemiurgicModel,
    DemiurgicForCausalLM,
    DemiurgicPreTrainedModel,
)
from .generation_config import GenerationConfig

__all__ = [
    "DemiurgicConfig",
    "DemiurgicModel",
    "DemiurgicForCausalLM",
    "DemiurgicPreTrainedModel",
    "GenerationConfig",
    "get_100m_config",
    "get_350m_config",
    "get_1b_config",
    "get_7b_config",
    "get_13b_config",
    "get_70b_config",
]
