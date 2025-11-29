"""Tool system for ChatGLM3 agent - code execution, file operations, web search."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import subprocess
import tempfile
from pathlib import Path
import json
import sys
import traceback


@dataclass
class ToolResult:
    """Result from tool execution."""

    success: bool
    output: str
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "metadata": self.metadata,
        }


class Tool(ABC):
    """Base class for all tools."""

    def __init__(self, name: str, description: str):
        """Initialize tool.

        Args:
            name: Tool name (used in tool calls)
            description: Human-readable description
        """
        self.name = name
        self.description = description

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters.

        Args:
            **kwargs: Tool-specific parameters

        Returns:
            ToolResult with execution output
        """
        pass

    def get_schema(self) -> Dict:
        """Get tool schema for model understanding.

        Returns:
            JSON schema describing the tool
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self._get_parameters_schema(),
        }

    @abstractmethod
    def _get_parameters_schema(self) -> Dict:
        """Get JSON schema for tool parameters.

        Returns:
            JSON schema for parameters
        """
        pass


class CodeExecutionTool(Tool):
    """Execute code in various languages with sandboxing."""

    SUPPORTED_LANGUAGES = {
        "python": {"command": ["python3", "-c"], "extension": "py"},
        "javascript": {"command": ["node", "-e"], "extension": "js"},
        "bash": {"command": ["bash", "-c"], "extension": "sh"},
    }

    def __init__(self, timeout: int = 30, max_output_size: int = 10000):
        """Initialize code execution tool.

        Args:
            timeout: Maximum execution time in seconds
            max_output_size: Maximum output size in characters
        """
        super().__init__(
            name="execute_code",
            description="Execute code in Python, JavaScript, or Bash and return the output",
        )
        self.timeout = timeout
        self.max_output_size = max_output_size

    def _get_parameters_schema(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "language": {
                    "type": "string",
                    "enum": list(self.SUPPORTED_LANGUAGES.keys()),
                    "description": "Programming language to execute",
                },
                "code": {
                    "type": "string",
                    "description": "Code to execute",
                },
            },
            "required": ["language", "code"],
        }

    def execute(self, language: str, code: str, **kwargs) -> ToolResult:
        """Execute code in specified language.

        Args:
            language: Programming language (python, javascript, bash)
            code: Code to execute

        Returns:
            ToolResult with execution output
        """
        if language not in self.SUPPORTED_LANGUAGES:
            return ToolResult(
                success=False,
                output="",
                error=f"Unsupported language: {language}. Supported: {list(self.SUPPORTED_LANGUAGES.keys())}",
            )

        try:
            lang_config = self.SUPPORTED_LANGUAGES[language]
            command = lang_config["command"] + [code]

            # Execute with timeout and capture output
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            # Combine stdout and stderr
            output = result.stdout + result.stderr
            output = output[:self.max_output_size]  # Limit output size

            return ToolResult(
                success=result.returncode == 0,
                output=output,
                error=result.stderr if result.returncode != 0 else None,
                metadata={"returncode": result.returncode, "language": language},
            )

        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                output="",
                error=f"Code execution timed out after {self.timeout} seconds",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Execution error: {str(e)}",
            )


class FileReadTool(Tool):
    """Read files from the filesystem."""

    def __init__(self, allowed_paths: Optional[List[Path]] = None, max_file_size: int = 100000):
        """Initialize file read tool.

        Args:
            allowed_paths: List of allowed base paths (None = current directory only)
            max_file_size: Maximum file size to read in bytes
        """
        super().__init__(
            name="read_file",
            description="Read contents of a file from the filesystem",
        )
        self.allowed_paths = allowed_paths or [Path.cwd()]
        self.max_file_size = max_file_size

    def _get_parameters_schema(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "filepath": {
                    "type": "string",
                    "description": "Path to the file to read",
                },
            },
            "required": ["filepath"],
        }

    def _is_path_allowed(self, filepath: Path) -> bool:
        """Check if path is within allowed directories."""
        filepath = filepath.resolve()
        return any(
            str(filepath).startswith(str(allowed.resolve()))
            for allowed in self.allowed_paths
        )

    def execute(self, filepath: str, **kwargs) -> ToolResult:
        """Read file contents.

        Args:
            filepath: Path to file to read

        Returns:
            ToolResult with file contents
        """
        try:
            path = Path(filepath)

            # Security check
            if not self._is_path_allowed(path):
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Access denied: Path outside allowed directories",
                )

            if not path.exists():
                return ToolResult(
                    success=False,
                    output="",
                    error=f"File not found: {filepath}",
                )

            if not path.is_file():
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Not a file: {filepath}",
                )

            # Check file size
            file_size = path.stat().st_size
            if file_size > self.max_file_size:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"File too large: {file_size} bytes (max: {self.max_file_size})",
                )

            # Read file
            content = path.read_text()

            return ToolResult(
                success=True,
                output=content,
                metadata={"filepath": str(path), "size": file_size},
            )

        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Error reading file: {str(e)}",
            )


class FileWriteTool(Tool):
    """Write or append to files on the filesystem."""

    def __init__(self, allowed_paths: Optional[List[Path]] = None, max_file_size: int = 100000):
        """Initialize file write tool.

        Args:
            allowed_paths: List of allowed base paths (None = current directory only)
            max_file_size: Maximum file size to write in bytes
        """
        super().__init__(
            name="write_file",
            description="Write or append content to a file on the filesystem",
        )
        self.allowed_paths = allowed_paths or [Path.cwd()]
        self.max_file_size = max_file_size

    def _get_parameters_schema(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "filepath": {
                    "type": "string",
                    "description": "Path to the file to write",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file",
                },
                "mode": {
                    "type": "string",
                    "enum": ["write", "append"],
                    "description": "Write mode: 'write' (overwrite) or 'append'",
                    "default": "write",
                },
            },
            "required": ["filepath", "content"],
        }

    def _is_path_allowed(self, filepath: Path) -> bool:
        """Check if path is within allowed directories."""
        filepath = filepath.resolve()
        return any(
            str(filepath).startswith(str(allowed.resolve()))
            for allowed in self.allowed_paths
        )

    def execute(self, filepath: str, content: str, mode: str = "write", **kwargs) -> ToolResult:
        """Write content to file.

        Args:
            filepath: Path to file to write
            content: Content to write
            mode: Write mode ('write' or 'append')

        Returns:
            ToolResult with write status
        """
        try:
            path = Path(filepath)

            # Security check
            if not self._is_path_allowed(path):
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Access denied: Path outside allowed directories",
                )

            # Check content size
            if len(content) > self.max_file_size:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Content too large: {len(content)} bytes (max: {self.max_file_size})",
                )

            # Create parent directories if needed
            path.parent.mkdir(parents=True, exist_ok=True)

            # Write file
            if mode == "append":
                with open(path, "a") as f:
                    f.write(content)
                action = "appended to"
            else:
                path.write_text(content)
                action = "written to"

            return ToolResult(
                success=True,
                output=f"Successfully {action} {filepath} ({len(content)} bytes)",
                metadata={"filepath": str(path), "size": len(content), "mode": mode},
            )

        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Error writing file: {str(e)}",
            )


class BashCommandTool(Tool):
    """Execute bash commands with safety restrictions."""

    DANGEROUS_COMMANDS = ["rm -rf", "mkfs", "dd", ">", "format", "del /f"]

    def __init__(self, timeout: int = 30, max_output_size: int = 10000):
        """Initialize bash command tool.

        Args:
            timeout: Maximum execution time in seconds
            max_output_size: Maximum output size in characters
        """
        super().__init__(
            name="bash_command",
            description="Execute bash commands (with safety restrictions)",
        )
        self.timeout = timeout
        self.max_output_size = max_output_size

    def _get_parameters_schema(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Bash command to execute",
                },
            },
            "required": ["command"],
        }

    def _is_command_safe(self, command: str) -> bool:
        """Check if command is safe to execute."""
        command_lower = command.lower()
        return not any(danger in command_lower for danger in self.DANGEROUS_COMMANDS)

    def execute(self, command: str, **kwargs) -> ToolResult:
        """Execute bash command.

        Args:
            command: Bash command to execute

        Returns:
            ToolResult with command output
        """
        # Safety check
        if not self._is_command_safe(command):
            return ToolResult(
                success=False,
                output="",
                error="Command rejected: potentially dangerous operation detected",
            )

        try:
            result = subprocess.run(
                ["bash", "-c", command],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            output = result.stdout + result.stderr
            output = output[:self.max_output_size]

            return ToolResult(
                success=result.returncode == 0,
                output=output,
                error=result.stderr if result.returncode != 0 else None,
                metadata={"returncode": result.returncode},
            )

        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                output="",
                error=f"Command timed out after {self.timeout} seconds",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Execution error: {str(e)}",
            )


class ToolRegistry:
    """Registry for managing available tools."""

    def __init__(self):
        """Initialize tool registry."""
        self.tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool.

        Args:
            tool: Tool instance to register
        """
        self.tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get tool by name.

        Args:
            name: Tool name

        Returns:
            Tool instance or None if not found
        """
        return self.tools.get(name)

    def execute_tool(self, name: str, **kwargs) -> ToolResult:
        """Execute a tool by name.

        Args:
            name: Tool name
            **kwargs: Tool parameters

        Returns:
            ToolResult from tool execution
        """
        tool = self.get_tool(name)
        if not tool:
            return ToolResult(
                success=False,
                output="",
                error=f"Unknown tool: {name}",
            )

        try:
            return tool.execute(**kwargs)
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Tool execution failed: {str(e)}\n{traceback.format_exc()}",
            )

    def get_all_schemas(self) -> List[Dict]:
        """Get schemas for all registered tools.

        Returns:
            List of tool schemas
        """
        return [tool.get_schema() for tool in self.tools.values()]

    def get_tool_descriptions(self) -> str:
        """Get human-readable description of all tools.

        Returns:
            Formatted string describing all tools
        """
        descriptions = []
        for tool in self.tools.values():
            descriptions.append(f"- {tool.name}: {tool.description}")
        return "\n".join(descriptions)


def create_default_registry(
    allowed_paths: Optional[List[Path]] = None,
    enable_bash: bool = True,
) -> ToolRegistry:
    """Create a tool registry with default tools.

    Args:
        allowed_paths: Paths allowed for file operations (None = current dir only)
        enable_bash: Whether to enable bash command execution

    Returns:
        Configured ToolRegistry
    """
    registry = ToolRegistry()

    # Register default tools
    registry.register(CodeExecutionTool())
    registry.register(FileReadTool(allowed_paths=allowed_paths))
    registry.register(FileWriteTool(allowed_paths=allowed_paths))

    if enable_bash:
        registry.register(BashCommandTool())

    return registry


__all__ = [
    "Tool",
    "ToolResult",
    "ToolRegistry",
    "CodeExecutionTool",
    "FileReadTool",
    "FileWriteTool",
    "BashCommandTool",
    "create_default_registry",
]
