"""Configuration management for ChatGLM3 CLI."""

from typing import Optional, List
from dataclasses import dataclass, field, asdict
from pathlib import Path
import yaml


@dataclass
class ModelConfig:
    """Model configuration."""

    model_path: str = "models/chatglm3-6b.Q4_K_M.gguf"
    n_ctx: int = 8192
    n_threads: int = 8
    n_gpu_layers: int = 0
    verbose: bool = False


@dataclass
class GenerationSettings:
    """Generation settings."""

    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    stop_sequences: List[str] = field(default_factory=lambda: ["</s>", "<|user|>", "<|system|>"])


@dataclass
class ConversationSettings:
    """Conversation settings."""

    max_context_length: int = 8192
    max_history_turns: int = 20
    system_prompt: Optional[str] = None


@dataclass
class ToolsConfig:
    """Tools configuration."""

    enabled: bool = True
    enable_code_execution: bool = True
    enable_file_operations: bool = True
    enable_bash_commands: bool = True
    allowed_paths: List[str] = field(default_factory=lambda: ["."])
    code_execution_timeout: int = 30
    bash_timeout: int = 30
    max_file_size: int = 100000
    max_output_size: int = 10000


@dataclass
class CLIConfig:
    """Complete CLI configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationSettings = field(default_factory=GenerationSettings)
    conversation: ConversationSettings = field(default_factory=ConversationSettings)
    tools: ToolsConfig = field(default_factory=ToolsConfig)

    @classmethod
    def from_file(cls, filepath: Path) -> "CLIConfig":
        """Load configuration from YAML file.

        Args:
            filepath: Path to config file

        Returns:
            Loaded CLIConfig instance
        """
        if not filepath.exists():
            # Return default config if file doesn't exist
            return cls()

        with open(filepath) as f:
            data = yaml.safe_load(f)

        return cls(
            model=ModelConfig(**data.get("model", {})),
            generation=GenerationSettings(**data.get("generation", {})),
            conversation=ConversationSettings(**data.get("conversation", {})),
            tools=ToolsConfig(**data.get("tools", {})),
        )

    def to_file(self, filepath: Path) -> None:
        """Save configuration to YAML file.

        Args:
            filepath: Path to save config
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "model": asdict(self.model),
            "generation": asdict(self.generation),
            "conversation": asdict(self.conversation),
            "tools": asdict(self.tools),
        }

        with open(filepath, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def create_default_config(cls, filepath: Path) -> "CLIConfig":
        """Create and save a default configuration file.

        Args:
            filepath: Path to save config

        Returns:
            Default CLIConfig instance
        """
        config = cls()
        config.to_file(filepath)
        return config


def load_config(config_path: Optional[Path] = None) -> CLIConfig:
    """Load configuration from file or create default.

    Args:
        config_path: Optional path to config file (default: ~/.demiurgic/config.yaml)

    Returns:
        Loaded or default CLIConfig
    """
    if config_path is None:
        config_path = Path.home() / ".demiurgic" / "config.yaml"

    if config_path.exists():
        return CLIConfig.from_file(config_path)
    else:
        # Create default config
        return CLIConfig.create_default_config(config_path)


__all__ = [
    "ModelConfig",
    "GenerationSettings",
    "ConversationSettings",
    "ToolsConfig",
    "CLIConfig",
    "load_config",
]
