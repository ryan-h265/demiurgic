"""Rich-based TUI interface for interactive chat with ChatGLM3."""

from typing import Optional, List, Dict, Any
from pathlib import Path
import sys

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.table import Table
from rich.live import Live
from rich.spinner import Spinner
from rich.prompt import Prompt, Confirm
from rich.text import Text
from rich.layout import Layout
from rich import box

from .conversation import ConversationManager
from .model import ChatGLM3Model, GenerationConfig
from .tools import ToolRegistry, ToolResult
from .prompts import (
    generate_tool_aware_system_prompt,
    generate_few_shot_examples,
    create_tool_usage_prompt,
)


class ChatInterface:
    """Interactive TUI for chatting with ChatGLM3."""

    def __init__(
        self,
        model: ChatGLM3Model,
        conversation: ConversationManager,
        tool_registry: ToolRegistry,
        console: Optional[Console] = None,
        enable_tool_hints: bool = True,
        use_few_shot: bool = True,
    ):
        """Initialize chat interface.

        Args:
            model: ChatGLM3 model instance
            conversation: Conversation manager
            tool_registry: Tool registry for executing tools
            console: Rich console (creates new if None)
            enable_tool_hints: Add hints to guide tool usage
            use_few_shot: Include few-shot examples in context
        """
        self.model = model
        self.conversation = conversation
        self.tool_registry = tool_registry
        self.console = console or Console()
        self.generation_config = GenerationConfig()
        self.enable_tool_hints = enable_tool_hints
        self.use_few_shot = use_few_shot

        # Inject few-shot examples if enabled
        if self.use_few_shot and len(self.conversation.messages) == 0:
            self._inject_few_shot_examples()

    def _inject_few_shot_examples(self) -> None:
        """Inject few-shot examples to teach model tool usage."""
        examples = generate_few_shot_examples()
        for example in examples:
            self.conversation.add_message(
                role=example["role"],
                content=example["content"],
            )

    def print_welcome(self) -> None:
        """Print welcome message with available commands."""
        welcome_text = """# ChatGLM3 Interactive Assistant

Welcome! I can help you with coding tasks, execute code, read/write files, and more.

## Available Commands:
- `/help` - Show this help message
- `/tools` - List available tools
- `/context` - Show conversation context statistics
- `/clear` - Clear conversation history
- `/save <file>` - Save conversation to file
- `/load <file>` - Load conversation from file
- `/config` - Show/modify generation settings
- `/exit` or `/quit` - Exit the program

## Available Tools:
"""
        welcome_text += self.tool_registry.get_tool_descriptions()
        welcome_text += "\n\nType your message or command to start!"

        self.console.print(Panel(Markdown(welcome_text), title="Welcome", border_style="green"))

    def print_message(self, role: str, content: str, style: Optional[str] = None) -> None:
        """Print a formatted message.

        Args:
            role: Message role (user, assistant, tool, system)
            content: Message content
            style: Optional style override
        """
        # Determine style based on role
        if style is None:
            style_map = {
                "user": "cyan",
                "assistant": "green",
                "tool": "yellow",
                "system": "blue",
                "error": "red",
            }
            style = style_map.get(role, "white")

        # Format role display
        role_display = role.title()

        # Try to render as markdown if it's from assistant
        if role == "assistant":
            try:
                md = Markdown(content)
                self.console.print(Panel(md, title=f"[bold]{role_display}[/bold]", border_style=style))
                return
            except Exception:
                pass  # Fall back to plain text

        # Plain text display
        self.console.print(Panel(content, title=f"[bold]{role_display}[/bold]", border_style=style))

    def print_tool_result(self, tool_name: str, result: ToolResult) -> None:
        """Print tool execution result.

        Args:
            tool_name: Name of executed tool
            result: Tool execution result
        """
        status = "✓ Success" if result.success else "✗ Failed"
        style = "green" if result.success else "red"

        output = result.output if result.success else (result.error or "Unknown error")

        panel = Panel(
            output,
            title=f"[bold]{status}: {tool_name}[/bold]",
            border_style=style,
        )
        self.console.print(panel)

    def get_user_input(self) -> str:
        """Get user input with prompt.

        Returns:
            User input string
        """
        return Prompt.ask("\n[bold cyan]You[/bold cyan]")

    def handle_command(self, command: str) -> bool:
        """Handle special commands.

        Args:
            command: Command string (starting with /)

        Returns:
            True if should continue, False if should exit
        """
        parts = command.split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if cmd in ["/exit", "/quit"]:
            self.console.print("[yellow]Goodbye![/yellow]")
            return False

        elif cmd == "/help":
            self.print_welcome()

        elif cmd == "/tools":
            self._show_tools()

        elif cmd == "/context":
            self._show_context()

        elif cmd == "/clear":
            if Confirm.ask("Clear conversation history?"):
                self.conversation.clear_history(keep_system=True)
                self.console.print("[green]✓ History cleared[/green]")

        elif cmd == "/save":
            if args:
                self._save_conversation(args)
            else:
                self.console.print("[red]Usage: /save <filename>[/red]")

        elif cmd == "/load":
            if args:
                self._load_conversation(args)
            else:
                self.console.print("[red]Usage: /load <filename>[/red]")

        elif cmd == "/config":
            self._show_config()

        else:
            self.console.print(f"[red]Unknown command: {cmd}[/red]")
            self.console.print("[yellow]Type /help for available commands[/yellow]")

        return True

    def _show_tools(self) -> None:
        """Show available tools in a table."""
        table = Table(title="Available Tools", box=box.ROUNDED)
        table.add_column("Tool", style="cyan", no_wrap=True)
        table.add_column("Description", style="white")

        for tool in self.tool_registry.tools.values():
            table.add_row(tool.name, tool.description)

        self.console.print(table)

    def _show_context(self) -> None:
        """Show conversation context statistics."""
        summary = self.conversation.get_context_summary()

        table = Table(title="Conversation Context", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Total Messages", str(summary["total_messages"]))
        table.add_row("User Messages", str(summary["user_messages"]))
        table.add_row("Assistant Messages", str(summary["assistant_messages"]))
        table.add_row("Tool Messages", str(summary["tool_messages"]))
        table.add_row("Estimated Tokens", str(summary["estimated_tokens"]))
        table.add_row("Context Usage", f"{summary['context_usage_percent']:.1f}%")

        self.console.print(table)

    def _save_conversation(self, filename: str) -> None:
        """Save conversation to file."""
        try:
            path = Path(filename)
            self.conversation.save_to_file(path)
            self.console.print(f"[green]✓ Saved to {path}[/green]")
        except Exception as e:
            self.console.print(f"[red]Error saving: {e}[/red]")

    def _load_conversation(self, filename: str) -> None:
        """Load conversation from file."""
        try:
            path = Path(filename)
            self.conversation = ConversationManager.load_from_file(path)
            self.console.print(f"[green]✓ Loaded from {path}[/green]")
        except Exception as e:
            self.console.print(f"[red]Error loading: {e}[/red]")

    def _show_config(self) -> None:
        """Show current generation configuration."""
        table = Table(title="Generation Config", box=box.ROUNDED)
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="white")

        config = self.generation_config
        table.add_row("max_tokens", str(config.max_tokens))
        table.add_row("temperature", str(config.temperature))
        table.add_row("top_p", str(config.top_p))
        table.add_row("top_k", str(config.top_k))
        table.add_row("repeat_penalty", str(config.repeat_penalty))

        self.console.print(table)

    def generate_response(self, user_message: str) -> str:
        """Generate model response with streaming display.

        Args:
            user_message: User's input message

        Returns:
            Complete generated response
        """
        # Add hints if enabled
        if self.enable_tool_hints:
            enhanced_message = create_tool_usage_prompt(user_message)
        else:
            enhanced_message = user_message

        # Add user message to history
        self.conversation.add_message("user", enhanced_message)

        # Get messages for model
        messages = self.conversation.get_messages(include_system=True)

        # Generate with streaming
        response_parts = []

        with Live(
            Panel(
                Spinner("dots", text="Thinking..."),
                title="[bold green]Assistant[/bold green]",
                border_style="green",
            ),
            console=self.console,
            refresh_per_second=10,
        ) as live:
            try:
                # Stream response
                for chunk in self.model.chat(
                    messages,
                    config=self.generation_config,
                    stream=True,
                ):
                    response_parts.append(chunk)
                    current_text = "".join(response_parts)

                    # Update live display with accumulated response
                    try:
                        md = Markdown(current_text)
                        live.update(
                            Panel(
                                md,
                                title="[bold green]Assistant[/bold green]",
                                border_style="green",
                            )
                        )
                    except Exception:
                        # Fallback to plain text if markdown fails
                        live.update(
                            Panel(
                                current_text,
                                title="[bold green]Assistant[/bold green]",
                                border_style="green",
                            )
                        )

            except Exception as e:
                self.console.print(f"[red]Error generating response: {e}[/red]")
                return ""

        complete_response = "".join(response_parts)

        # Add to conversation history
        self.conversation.add_message("assistant", complete_response)

        return complete_response

    def execute_tool_calls(self, response: str) -> List[ToolResult]:
        """Parse and execute tool calls from response.

        Args:
            response: Model response potentially containing tool calls

        Returns:
            List of tool execution results
        """
        tool_calls = self.model.parse_tool_calls(response)

        if not tool_calls:
            return []

        results = []

        for call in tool_calls:
            # Extract tool name and parameters
            tool_name = call.get("tool") or call.get("function") or call.get("name")
            parameters = call.get("parameters") or call.get("args") or {}

            if not tool_name:
                continue

            self.console.print(f"\n[yellow]Executing tool: {tool_name}[/yellow]")

            # Execute tool
            result = self.tool_registry.execute_tool(tool_name, **parameters)
            results.append(result)

            # Display result
            self.print_tool_result(tool_name, result)

            # Add tool result to conversation
            tool_result_text = self.model.format_tool_result(tool_name, result.to_dict())
            self.conversation.add_message("tool", tool_result_text)

        return results

    def run(self) -> None:
        """Run the interactive chat loop."""
        self.print_welcome()

        try:
            while True:
                # Get user input
                user_input = self.get_user_input()

                if not user_input.strip():
                    continue

                # Handle commands
                if user_input.startswith("/"):
                    should_continue = self.handle_command(user_input)
                    if not should_continue:
                        break
                    continue

                # Generate response
                response = self.generate_response(user_input)

                if not response:
                    continue

                # Check for and execute tool calls
                tool_results = self.execute_tool_calls(response)

                # If tools were executed, generate follow-up response
                if tool_results:
                    self.console.print("\n[yellow]Generating follow-up response...[/yellow]")
                    follow_up = self.generate_response(
                        "Based on the tool results above, provide your final answer."
                    )

        except KeyboardInterrupt:
            self.console.print("\n[yellow]Interrupted by user[/yellow]")
        except Exception as e:
            self.console.print(f"\n[red]Fatal error: {e}[/red]")
            raise


__all__ = ["ChatInterface"]
