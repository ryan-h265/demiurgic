"""
Knowledge distillation module for Demiurgic.

Provides tools for training student models using knowledge from teacher models.
"""

from .teacher_api import (
    TeacherAPI,
    TeacherConfig,
    create_teacher_api,
)
from .prompt_generator import (
    PromptGenerator,
    generate_system_prompt,
    generate_prompts_from_categories,
)
from .config import (
    DistillationConfig,
    OutputDistillationConfig,
    LogitDistillationConfig,
    HybridDistillationConfig,
)
from .trainer import (
    DistillationTrainer,
    DistillationDataset,
)

__all__ = [
    "TeacherAPI",
    "TeacherConfig",
    "create_teacher_api",
    "PromptGenerator",
    "generate_system_prompt",
    "generate_prompts_from_categories",
    "DistillationConfig",
    "OutputDistillationConfig",
    "LogitDistillationConfig",
    "HybridDistillationConfig",
    "DistillationTrainer",
    "DistillationDataset",
]
