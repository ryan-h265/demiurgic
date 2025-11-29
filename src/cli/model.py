"""ChatGLM3 model interface using llama.cpp for GGUF models."""

from typing import Optional, Dict, List, Iterator
from pathlib import Path
from dataclasses import dataclass
import json
import re

from llama_cpp import Llama


@dataclass
class GenerationConfig:
    """Configuration for text generation."""

    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    stop_sequences: List[str] = None

    def __post_init__(self):
        if self.stop_sequences is None:
            self.stop_sequences = ["</s>", "<|user|>", "<|system|>"]


class ChatGLM3Model:
    """ChatGLM3 model wrapper using llama.cpp."""

    def __init__(
        self,
        model_path: Path,
        n_ctx: int = 8192,
        n_threads: int = 8,
        n_gpu_layers: int = 0,
        verbose: bool = False,
    ):
        """Initialize ChatGLM3 model.

        Args:
            model_path: Path to GGUF model file
            n_ctx: Context window size
            n_threads: Number of CPU threads
            n_gpu_layers: Number of layers to offload to GPU
            verbose: Enable verbose logging
        """
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers
        self.verbose = verbose

        # Load model
        self.llm = self._load_model()

    def _load_model(self) -> Llama:
        """Load the GGUF model via llama.cpp."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. "
                "Run scripts/download_chatglm3_gguf.py to download it."
            )

        return Llama(
            model_path=str(self.model_path),
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            n_gpu_layers=self.n_gpu_layers,
            verbose=self.verbose,
        )

    def format_chat_prompt(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
    ) -> str:
        """Format messages into ChatGLM3 prompt format.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            system_prompt: Optional system prompt

        Returns:
            Formatted prompt string
        """
        parts = []

        # Add system prompt
        if system_prompt:
            parts.append(f"<|system|>\n{system_prompt.strip()}\n<|/system|>")

        # Add conversation history
        for msg in messages:
            role = msg["role"]
            content = msg["content"].strip()

            if role == "system":
                parts.append(f"<|system|>\n{content}\n<|/system|>")
            elif role == "user":
                parts.append(f"<|user|>\n{content}\n<|/user|>")
            elif role == "assistant":
                parts.append(f"<|assistant|>\n{content}\n<|/assistant|>")
            elif role == "tool":
                parts.append(f"<|tool|>\n{content}\n<|/tool|>")

        # Add assistant prompt
        parts.append("<|assistant|>")

        return "\n".join(parts)

    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        stream: bool = False,
    ) -> str:
        """Generate text from prompt.

        Args:
            prompt: Input prompt
            config: Generation configuration
            stream: Whether to stream output

        Returns:
            Generated text
        """
        if config is None:
            config = GenerationConfig()

        if stream:
            return self._generate_stream(prompt, config)
        else:
            return self._generate_complete(prompt, config)

    def _generate_complete(self, prompt: str, config: GenerationConfig) -> str:
        """Generate complete response without streaming."""
        response = self.llm.create_completion(
            prompt=prompt,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repeat_penalty=config.repeat_penalty,
            stop=config.stop_sequences,
        )

        return response["choices"][0]["text"].strip()

    def _generate_stream(self, prompt: str, config: GenerationConfig) -> Iterator[str]:
        """Generate response with streaming."""
        stream = self.llm.create_completion(
            prompt=prompt,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repeat_penalty=config.repeat_penalty,
            stop=config.stop_sequences,
            stream=True,
        )

        for chunk in stream:
            text = chunk["choices"][0]["text"]
            if text:
                yield text

    def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
        stream: bool = False,
    ) -> str:
        """Generate chat response from message history.

        Args:
            messages: List of message dictionaries
            system_prompt: Optional system prompt
            config: Generation configuration
            stream: Whether to stream output

        Returns:
            Generated response text
        """
        prompt = self.format_chat_prompt(messages, system_prompt)
        return self.generate(prompt, config, stream)

    def parse_tool_calls(self, response: str) -> List[Dict]:
        """Parse tool calls from model response.

        ChatGLM3 can output tool calls in JSON format. This extracts them.

        Args:
            response: Model response text

        Returns:
            List of tool call dictionaries
        """
        tool_calls = []

        # Look for JSON blocks in response
        json_pattern = r'```json\s*(\{.*?\})\s*```'
        matches = re.findall(json_pattern, response, re.DOTALL)

        for match in matches:
            try:
                data = json.loads(match)
                # Check if it looks like a tool call
                if "tool" in data or "function" in data or "name" in data:
                    tool_calls.append(data)
            except json.JSONDecodeError:
                continue

        # Also try to find plain JSON objects
        if not tool_calls:
            json_obj_pattern = r'\{[^}]*"(?:tool|function|name)"[^}]*\}'
            matches = re.findall(json_obj_pattern, response)
            for match in matches:
                try:
                    data = json.loads(match)
                    tool_calls.append(data)
                except json.JSONDecodeError:
                    continue

        return tool_calls

    def format_tool_result(self, tool_name: str, result: Dict) -> str:
        """Format tool execution result for model consumption.

        Args:
            tool_name: Name of tool that was executed
            result: Tool execution result

        Returns:
            Formatted string for model
        """
        return f"""<|tool|>
Tool: {tool_name}
Result:
{json.dumps(result, indent=2)}
<|/tool|>"""

    def get_model_info(self) -> Dict:
        """Get information about loaded model.

        Returns:
            Dictionary with model information
        """
        return {
            "model_path": str(self.model_path),
            "context_length": self.n_ctx,
            "n_threads": self.n_threads,
            "n_gpu_layers": self.n_gpu_layers,
            "model_exists": self.model_path.exists(),
        }


__all__ = ["ChatGLM3Model", "GenerationConfig"]
