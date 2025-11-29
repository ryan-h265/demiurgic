"""Comprehensive prompt generation for coding assistant knowledge distillation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List
import random
import json


@dataclass
class PromptTemplate:
    category: str
    template: str
    variables: Dict[str, List[str]]

    def fill(self) -> Dict[str, str]:
        values = {key: random.choice(options) for key, options in self.variables.items()}
        filled_prompt = self.template.format(**values)
        return {
            "prompt": filled_prompt,
            "category": self.category,
            "language": values.get("language", "unknown")
        }


class PromptGenerator:
    """
    Comprehensive prompt generator for coding assistant knowledge distillation.

    Combines:
    1. Existing high-quality prompts from /prompts/examples/
    2. New templates for tool use, debugging, test generation, multi-step tasks
    """

    def __init__(self, seed: int = 42, prompts_dir: Path = None):
        random.seed(seed)
        self.prompts_dir = prompts_dir or Path(__file__).parent.parent.parent / "prompts"

        # Load existing prompts
        self.existing_prompts = self._load_existing_prompts()

        # Build new templates for missing categories
        self.templates = self._build_comprehensive_templates()

        print(f"Loaded {len(self.existing_prompts)} existing prompts")
        print(f"Created {len(self.templates)} new templates")

    def _load_existing_prompts(self) -> List[Dict[str, str]]:
        """Load existing high-quality prompts from JSONL file."""
        prompts = []
        examples_file = self.prompts_dir / "examples" / "all_prompts.jsonl"

        if not examples_file.exists():
            print(f"Warning: {examples_file} not found, skipping existing prompts")
            return prompts

        try:
            with open(examples_file, 'r', encoding='utf-8') as f:
                for line in f:
                    prompt_data = json.loads(line.strip())
                    prompts.append(prompt_data)
        except Exception as e:
            print(f"Warning: Could not load existing prompts: {e}")

        return prompts

    def _build_comprehensive_templates(self) -> List[PromptTemplate]:
        """Build comprehensive templates for coding assistant capabilities."""

        languages = ["Python", "JavaScript", "TypeScript", "Java", "Go", "Rust", "C++"]

        templates = []

        # === Tool Usage & Code Execution ===
        templates.extend([
            PromptTemplate(
                category="tool_execution",
                template=(
                    "Write a {language} function to {task}. "
                    "After writing it, explain how you would test it by executing it with sample inputs."
                ),
                variables={
                    "language": languages,
                    "task": [
                        "calculate the factorial of a number",
                        "find the longest common substring between two strings",
                        "implement binary search on a sorted array",
                        "convert a Roman numeral to an integer",
                        "validate an email address using regex",
                        "parse a CSV string into a list of dictionaries",
                        "implement a simple LRU cache",
                        "calculate the nth Fibonacci number recursively and iteratively",
                    ]
                },
            ),
            PromptTemplate(
                category="tool_file_operations",
                template=(
                    "I need to {operation}. Write {language} code that uses file operations safely, "
                    "includes error handling, and explain what tools/permissions are needed."
                ),
                variables={
                    "language": languages,
                    "operation": [
                        "read a JSON config file and validate its schema",
                        "append logs to a file with rotation when it exceeds 10MB",
                        "recursively find all Python files in a directory",
                        "safely create a directory structure with proper error handling",
                        "read a large CSV file in chunks to avoid memory issues",
                        "watch a file for changes and trigger an action",
                    ]
                },
            ),
            PromptTemplate(
                category="tool_api_calls",
                template=(
                    "Write a {language} function to {task}. "
                    "Include proper error handling, retries, and explain what happens if the API fails."
                ),
                variables={
                    "language": languages,
                    "task": [
                        "fetch user data from a REST API with authentication",
                        "make an HTTP request with exponential backoff retry logic",
                        "call multiple APIs concurrently and aggregate results",
                        "handle rate limiting when calling an external API",
                        "parse and validate a JSON response from an API",
                    ]
                },
            ),
        ])

        # === Debugging & Error Handling ===
        templates.extend([
            PromptTemplate(
                category="debugging",
                template=(
                    "This {language} code has a bug:\n```\n{buggy_code}\n```\n"
                    "Find the bug, explain what's wrong, and provide the fixed code."
                ),
                variables={
                    "language": languages,
                    "buggy_code": [
                        "def find_max(nums):\n    max_val = 0\n    for num in nums:\n        if num > max_val:\n            max_val = num\n    return max_val",
                        "function isPalindrome(str) {\n    return str === str.reverse();\n}",
                        "def calculate_average(numbers):\n    total = 0\n    for num in numbers:\n        total += num\n    return total / len(numbers)",
                        "const fetchData = async (url) => {\n    const response = fetch(url);\n    return response.json();\n}",
                    ]
                },
            ),
            PromptTemplate(
                category="debugging_errors",
                template=(
                    "I'm getting this error:\n```\n{error}\n```\n\n"
                    "In this {language} code:\n```\n{code}\n```\n"
                    "Debug the issue step-by-step and fix it."
                ),
                variables={
                    "language": ["Python", "JavaScript", "Java"],
                    "error": [
                        "TypeError: 'NoneType' object is not subscriptable",
                        "IndexError: list index out of range",
                        "KeyError: 'username'",
                        "TypeError: Cannot read property 'map' of undefined",
                        "ReferenceError: variable is not defined",
                    ],
                    "code": [
                        "data = get_user_data()\nusername = data['user']['name']",
                        "result = process_items(items)\nfirst = result[0]",
                        "for i in range(len(arr) + 1):\n    print(arr[i])",
                    ]
                },
            ),
            PromptTemplate(
                category="debugging_trace",
                template=(
                    "Given this stack trace:\n```\n{trace}\n```\n"
                    "Explain what went wrong, where the error originated, and how to fix it."
                ),
                variables={
                    "trace": [
                        "Traceback (most recent call last):\n  File 'main.py', line 45, in <module>\n    result = process(data)\n  File 'processor.py', line 23, in process\n    return transform(item)\n  File 'utils.py', line 10, in transform\n    return item.upper()\nAttributeError: 'int' object has no attribute 'upper'",
                    ]
                },
            ),
        ])

        # === Test Generation ===
        templates.extend([
            PromptTemplate(
                category="test_generation",
                template=(
                    "Write comprehensive unit tests for this {language} function:\n```\n{function}\n```\n"
                    "Include edge cases, error cases, and normal cases."
                ),
                variables={
                    "language": languages,
                    "function": [
                        "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1",
                        "function validateEmail(email) {\n    const re = /^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$/;\n    return re.test(email);\n}",
                    ]
                },
            ),
            PromptTemplate(
                category="test_generation_tdd",
                template=(
                    "Using TDD (Test-Driven Development), write tests FIRST for a {language} function that {task}. "
                    "Then implement the function to make the tests pass."
                ),
                variables={
                    "language": languages,
                    "task": [
                        "validates password strength (8+ chars, uppercase, lowercase, number, special char)",
                        "merges two sorted arrays into one sorted array",
                        "calculates the median of a list of numbers",
                        "converts snake_case to camelCase",
                    ]
                },
            ),
        ])

        # === Multi-Step Tasks ===
        templates.extend([
            PromptTemplate(
                category="multistep_planning",
                template=(
                    "I need to build a {project}. Break this down into steps, "
                    "then implement the first step in {language} with proper structure."
                ),
                variables={
                    "language": languages,
                    "project": [
                        "simple REST API for a todo list with CRUD operations",
                        "command-line tool that analyzes code complexity",
                        "web scraper that extracts product prices from multiple sites",
                        "file converter that transforms CSV to JSON with validation",
                        "log analyzer that finds errors and generates a report",
                    ]
                },
            ),
            PromptTemplate(
                category="multistep_refactor",
                template=(
                    "This {language} code works but needs refactoring:\n```\n{messy_code}\n```\n"
                    "First analyze the issues, then refactor it step-by-step with explanations."
                ),
                variables={
                    "language": languages,
                    "messy_code": [
                        "def process_data(data):\n    result = []\n    for item in data:\n        if item is not None:\n            if isinstance(item, str):\n                if len(item) > 0:\n                    result.append(item.strip().lower())\n    return result",
                    ]
                },
            ),
        ])

        # === Documentation Generation ===
        templates.extend([
            PromptTemplate(
                category="documentation",
                template=(
                    "Write comprehensive documentation for this {language} {code_type}:\n```\n{code}\n```\n"
                    "Include docstrings, parameter descriptions, return values, examples, and edge cases."
                ),
                variables={
                    "language": languages,
                    "code_type": ["function", "class", "module"],
                    "code": [
                        "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return merge(left, right)",
                        "class LRUCache:\n    def __init__(self, capacity):\n        self.cache = {}\n        self.capacity = capacity\n    def get(self, key):\n        return self.cache.get(key, -1)\n    def put(self, key, value):\n        self.cache[key] = value",
                    ]
                },
            ),
        ])

        # === Code Explanation ===
        templates.extend([
            PromptTemplate(
                category="explanation",
                template=(
                    "Explain this {language} code like I'm a {level} developer:\n```\n{code}\n```"
                ),
                variables={
                    "language": languages,
                    "level": ["beginner", "intermediate", "senior"],
                    "code": [
                        "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)",
                        "const debounce = (func, delay) => {\n    let timeout;\n    return (...args) => {\n        clearTimeout(timeout);\n        timeout = setTimeout(() => func(...args), delay);\n    };\n};",
                    ]
                },
            ),
        ])

        # === Code Review & Optimization ===
        templates.extend([
            PromptTemplate(
                category="code_review",
                template=(
                    "Review this {language} code for {aspect}:\n```\n{code}\n```\n"
                    "Provide specific suggestions with examples."
                ),
                variables={
                    "language": languages,
                    "aspect": [
                        "performance issues",
                        "security vulnerabilities",
                        "code readability and maintainability",
                        "potential bugs and edge cases",
                        "best practices and idioms",
                    ],
                    "code": [
                        "def find_users(query):\n    users = []\n    for user in database.all_users:\n        if query in user.name:\n            users.append(user)\n    return users",
                    ]
                },
            ),
            PromptTemplate(
                category="optimization",
                template=(
                    "This {language} code is slow:\n```\n{slow_code}\n```\n"
                    "Analyze the time complexity and optimize it. Show before/after complexity."
                ),
                variables={
                    "language": languages,
                    "slow_code": [
                        "def find_duplicates(arr):\n    dupes = []\n    for i in range(len(arr)):\n        for j in range(i+1, len(arr)):\n            if arr[i] == arr[j] and arr[i] not in dupes:\n                dupes.append(arr[i])\n    return dupes",
                    ]
                },
            ),
        ])

        # === Real-World Scenarios ===
        templates.extend([
            PromptTemplate(
                category="real_world",
                template=(
                    "Build a {component} in {language} that {requirement}. "
                    "Include error handling, logging, and explain deployment considerations."
                ),
                variables={
                    "language": languages,
                    "component": [
                        "rate limiter",
                        "caching layer",
                        "retry mechanism with exponential backoff",
                        "connection pool manager",
                        "circuit breaker",
                    ],
                    "requirement": [
                        "prevents more than 100 requests per minute per user",
                        "stores frequently accessed data with TTL",
                        "handles transient network failures gracefully",
                        "manages database connections efficiently",
                        "stops calling a failing service and recovers automatically",
                    ]
                },
            ),
        ])

        return templates

    def sample(self, count: int, existing_weight: float = 0.5) -> List[Dict[str, str]]:
        """
        Sample prompts with a mix of existing and generated.

        Args:
            count: Total number of prompts to generate
            existing_weight: Proportion of existing prompts (0.0-1.0)
                           0.5 = 50% existing, 50% new templates

        Returns:
            List of prompt dictionaries with 'prompt', 'category', 'language' keys
        """
        prompts = []

        # Calculate how many from each source
        num_existing = int(count * existing_weight)
        num_generated = count - num_existing

        # Sample from existing prompts
        if self.existing_prompts and num_existing > 0:
            existing_sample = random.choices(self.existing_prompts, k=num_existing)
            prompts.extend(existing_sample)

        # Generate from templates
        for _ in range(num_generated):
            template = random.choice(self.templates)
            prompts.append(template.fill())

        # Shuffle to mix existing and generated
        random.shuffle(prompts)

        return prompts


def generate_system_prompt(categories: Iterable[str] = None) -> str:
    """
    Build a comprehensive system prompt for coding assistant training.

    Args:
        categories: Optional list of categories being trained on

    Returns:
        System prompt string optimized for Claude/GPT-4 teachers
    """
    base_prompt = (
        "You are an expert coding assistant helping to train a student model. "
        "Your responses should be:\n"
        "- Clear and well-structured with proper markdown formatting\n"
        "- Include complete, working code examples with explanations\n"
        "- Show step-by-step reasoning for complex problems\n"
        "- Include error handling and edge cases\n"
        "- Demonstrate best practices and idiomatic code\n"
        "- Use code blocks with language tags (```python, ```javascript, etc.)\n"
        "- When showing tool usage, format as JSON with clear descriptions\n"
        "- For debugging, explain the root cause and provide fixes\n"
        "- For multi-step tasks, break down the approach before implementing"
    )

    if categories:
        categories_str = ", ".join(categories)
        base_prompt += f"\n\nFocus areas: {categories_str}"

    return base_prompt


__all__ = ["PromptTemplate", "PromptGenerator", "generate_system_prompt"]
