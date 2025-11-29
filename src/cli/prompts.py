"""Enhanced system prompts and few-shot examples for tool usage."""

from typing import List, Dict


def generate_tool_aware_system_prompt(available_tools: List[Dict]) -> str:
    """Generate system prompt that teaches the model how to use tools.

    Args:
        available_tools: List of tool schemas

    Returns:
        Enhanced system prompt with tool usage instructions
    """

    tool_list = "\n".join([
        f"- **{tool['name']}**: {tool['description']}"
        for tool in available_tools
    ])

    return f"""You are a helpful coding assistant powered by ChatGLM3. You can write code, debug issues, and use tools to accomplish tasks.

## Available Tools

You have access to these tools:

{tool_list}

## How to Use Tools

When you need to use a tool, output a JSON code block like this:

```json
{{
  "tool": "tool_name",
  "parameters": {{
    "param1": "value1",
    "param2": "value2"
  }}
}}
```

**Important:**
- Use tools when you need to execute code, read/write files, or run commands
- After the tool executes, I'll show you the results
- Then provide your analysis or final answer based on the results
- You can use multiple tools in sequence if needed

## Examples

**Example 1: Execute Code**
User: "Calculate the factorial of 5"
Assistant: Let me write and execute a Python function for that:

```json
{{
  "tool": "execute_code",
  "parameters": {{
    "language": "python",
    "code": "def factorial(n):\\n    if n <= 1:\\n        return 1\\n    return n * factorial(n-1)\\n\\nprint(f'Factorial of 5 = {{factorial(5)}}')"
  }}
}}
```

**Example 2: Read File**
User: "What's in config.yaml?"
Assistant: Let me read that file for you:

```json
{{
  "tool": "read_file",
  "parameters": {{
    "filepath": "config.yaml"
  }}
}}
```

**Example 3: Write File**
User: "Create a hello.py file that prints hello world"
Assistant: I'll create that file:

```json
{{
  "tool": "write_file",
  "parameters": {{
    "filepath": "hello.py",
    "content": "print('Hello, World!')",
    "mode": "write"
  }}
}}
```

**Example 4: Run Command**
User: "List files in the current directory"
Assistant: Let me run that command:

```json
{{
  "tool": "bash_command",
  "parameters": {{
    "command": "ls -la"
  }}
}}
```

## Guidelines

1. **Use tools proactively** - Don't just describe what to do, actually do it
2. **Explain your actions** - Tell the user what you're doing before using a tool
3. **Check results** - After tool execution, verify the output and explain what happened
4. **Chain tools** - Use multiple tools in sequence to solve complex tasks
5. **Handle errors** - If a tool fails, explain why and suggest alternatives

Now, let's help the user with their task!"""


def generate_few_shot_examples() -> List[Dict[str, str]]:
    """Generate few-shot examples demonstrating tool usage.

    Returns:
        List of example conversations
    """
    return [
        {
            "role": "user",
            "content": "Write a Python script to calculate fibonacci numbers and test it with n=10"
        },
        {
            "role": "assistant",
            "content": """I'll write a fibonacci function and test it for you.

```json
{
  "tool": "execute_code",
  "parameters": {
    "language": "python",
    "code": "def fibonacci(n):\\n    if n <= 1:\\n        return n\\n    return fibonacci(n-1) + fibonacci(n-2)\\n\\nresult = fibonacci(10)\\nprint(f'Fibonacci(10) = {result}')"
  }
}
```"""
        },
        {
            "role": "tool",
            "content": """Tool: execute_code
Result:
{
  "success": true,
  "output": "Fibonacci(10) = 55"
}"""
        },
        {
            "role": "assistant",
            "content": "Perfect! The fibonacci function works correctly. Fibonacci(10) = 55, which is the 10th number in the Fibonacci sequence (0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55)."
        },
        {
            "role": "user",
            "content": "Create a file called test.txt with some sample data"
        },
        {
            "role": "assistant",
            "content": """I'll create a test.txt file with sample data.

```json
{
  "tool": "write_file",
  "parameters": {
    "filepath": "test.txt",
    "content": "Sample Data\\nLine 1: Hello\\nLine 2: World\\nLine 3: Test\\n",
    "mode": "write"
  }
}
```"""
        },
        {
            "role": "tool",
            "content": """Tool: write_file
Result:
{
  "success": true,
  "output": "Successfully written to test.txt (42 bytes)"
}"""
        },
        {
            "role": "assistant",
            "content": "Done! I've created test.txt with 3 lines of sample data (42 bytes total)."
        },
    ]


def inject_tool_schemas(system_prompt: str, tool_registry) -> str:
    """Inject detailed tool schemas into system prompt.

    Args:
        system_prompt: Base system prompt
        tool_registry: ToolRegistry instance

    Returns:
        System prompt with tool schemas appended
    """
    schemas = tool_registry.get_all_schemas()

    schema_text = "\n\n## Tool Schemas\n\n"

    for schema in schemas:
        schema_text += f"### {schema['name']}\n"
        schema_text += f"{schema['description']}\n\n"
        schema_text += "**Parameters:**\n```json\n"

        import json
        schema_text += json.dumps(schema['parameters'], indent=2)
        schema_text += "\n```\n\n"

    return system_prompt + schema_text


def create_tool_usage_prompt(user_message: str, tool_hint: str = None) -> str:
    """Create a prompt that encourages tool usage.

    Args:
        user_message: User's message
        tool_hint: Optional hint about which tool to use

    Returns:
        Enhanced prompt that guides tool usage
    """
    if tool_hint:
        return f"""{user_message}

**Hint**: Consider using the {tool_hint} tool to accomplish this task. Remember to output the tool call as a JSON code block."""

    # Analyze message to suggest appropriate tool
    message_lower = user_message.lower()

    if any(word in message_lower for word in ["run", "execute", "test", "try", "calculate"]):
        hint = "execute_code"
    elif any(word in message_lower for word in ["read", "show", "display", "what's in", "content"]):
        hint = "read_file"
    elif any(word in message_lower for word in ["write", "create", "save", "make"]):
        hint = "write_file"
    elif any(word in message_lower for word in ["list", "ls", "files", "directory", "check"]):
        hint = "bash_command"
    else:
        return user_message

    return f"""{user_message}

**Hint**: This task likely needs the {hint} tool. Remember to output the tool call as a JSON code block starting with ```json"""


__all__ = [
    "generate_tool_aware_system_prompt",
    "generate_few_shot_examples",
    "inject_tool_schemas",
    "create_tool_usage_prompt",
]
