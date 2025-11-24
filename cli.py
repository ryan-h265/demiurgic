#!/usr/bin/env python3
"""Main CLI entry point for ChatGLM3 interactive assistant."""

import argparse
import sys
from pathlib import Path

from rich.console import Console

from src.cli.config import load_config, CLIConfig
from src.cli.model import ChatGLM3Model, GenerationConfig
from src.cli.conversation import ConversationManager
from src.cli.tools import create_default_registry
from src.cli.interface import ChatInterface
from src.cli.prompts import generate_tool_aware_system_prompt


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="ChatGLM3 Interactive Assistant - A coding assistant with tool capabilities"
    )

    parser.add_argument(
        "--config",
        type=Path,
        help="Path to configuration file (default: ~/.demiurgic/config.yaml)",
    )

    parser.add_argument(
        "--model-path",
        type=Path,
        help="Override model path from config",
    )

    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        help="Number of layers to offload to GPU",
    )

    parser.add_argument(
        "--n-threads",
        type=int,
        help="Number of CPU threads to use",
    )

    parser.add_argument(
        "--system-prompt",
        type=str,
        help="Custom system prompt",
    )

    parser.add_argument(
        "--load-conversation",
        type=Path,
        help="Load conversation from file",
    )

    parser.add_argument(
        "--create-config",
        action="store_true",
        help="Create default config file and exit",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    console = Console()

    # Create default config if requested
    if args.create_config:
        config_path = args.config or Path.home() / ".demiurgic" / "config.yaml"
        config = CLIConfig.create_default_config(config_path)
        console.print(f"[green]✓ Created default config at {config_path}[/green]")
        return 0

    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        console.print(f"[red]Error loading config: {e}[/red]")
        return 1

    # Apply command-line overrides
    if args.model_path:
        config.model.model_path = str(args.model_path)
    if args.n_gpu_layers is not None:
        config.model.n_gpu_layers = args.n_gpu_layers
    if args.n_threads is not None:
        config.model.n_threads = args.n_threads
    if args.verbose:
        config.model.verbose = True
    if args.system_prompt:
        config.conversation.system_prompt = args.system_prompt

    # Initialize model
    console.print("[yellow]Loading model...[/yellow]")
    try:
        model = ChatGLM3Model(
            model_path=Path(config.model.model_path),
            n_ctx=config.model.n_ctx,
            n_threads=config.model.n_threads,
            n_gpu_layers=config.model.n_gpu_layers,
            verbose=config.model.verbose,
        )
        console.print("[green]✓ Model loaded successfully[/green]")
    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("[yellow]Run: python scripts/download_chatglm3_gguf.py[/yellow]")
        return 1
    except Exception as e:
        console.print(f"[red]Error loading model: {e}[/red]")
        return 1

    # Initialize tools first (needed for system prompt)
    if config.tools.enabled:
        allowed_paths = [Path(p) for p in config.tools.allowed_paths]
        tool_registry = create_default_registry(
            allowed_paths=allowed_paths,
            enable_bash=config.tools.enable_bash_commands,
        )
        console.print(f"[green]✓ Loaded {len(tool_registry.tools)} tools[/green]")
    else:
        from src.cli.tools import ToolRegistry
        tool_registry = ToolRegistry()
        console.print("[yellow]⚠ Tools disabled[/yellow]")

    # Generate enhanced system prompt with tool information
    if config.tools.enabled:
        tool_schemas = tool_registry.get_all_schemas()
        enhanced_system_prompt = generate_tool_aware_system_prompt(tool_schemas)
    else:
        enhanced_system_prompt = config.conversation.system_prompt

    # Initialize conversation
    if args.load_conversation:
        try:
            conversation = ConversationManager.load_from_file(args.load_conversation)
            console.print(f"[green]✓ Loaded conversation from {args.load_conversation}[/green]")
        except Exception as e:
            console.print(f"[red]Error loading conversation: {e}[/red]")
            return 1
    else:
        conversation = ConversationManager(
            system_prompt=enhanced_system_prompt,
            max_context_length=config.conversation.max_context_length,
            max_history_turns=config.conversation.max_history_turns,
        )

    # Initialize interface
    interface = ChatInterface(
        model=model,
        conversation=conversation,
        tool_registry=tool_registry,
        console=console,
    )

    # Set generation config
    interface.generation_config = GenerationConfig(
        max_tokens=config.generation.max_tokens,
        temperature=config.generation.temperature,
        top_p=config.generation.top_p,
        top_k=config.generation.top_k,
        repeat_penalty=config.generation.repeat_penalty,
        stop_sequences=config.generation.stop_sequences,
    )

    # Run interactive loop
    try:
        interface.run()
        return 0
    except KeyboardInterrupt:
        console.print("\n[yellow]Goodbye![/yellow]")
        return 0
    except Exception as e:
        console.print(f"\n[red]Fatal error: {e}[/red]")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
