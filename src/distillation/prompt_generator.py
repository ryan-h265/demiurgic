"""
Prompt generation for knowledge distillation.

Generates diverse coding prompts for creating training data from teacher models.
"""

import random
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class PromptTemplate:
    """Template for generating prompts."""
    category: str
    template: str
    variables: Dict[str, List[str]]


class PromptGenerator:
    """
    Generate diverse coding prompts for knowledge distillation.

    Creates prompts across different:
    - Programming languages
    - Task types (implementation, debugging, explanation, etc.)
    - Difficulty levels
    - Domains (algorithms, web dev, data processing, etc.)
    """

    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.templates = self._create_templates()

    def generate_prompts(self, num_prompts: int = 1000) -> List[Dict[str, str]]:
        """
        Generate diverse coding prompts.

        Args:
            num_prompts: Number of prompts to generate

        Returns:
            List of prompt dicts with 'prompt', 'category', 'language'
        """
        prompts = []

        for _ in range(num_prompts):
            template = random.choice(self.templates)
            prompt_text = self._fill_template(template)

            prompts.append({
                'prompt': prompt_text,
                'category': template.category,
                'language': self._extract_language(template, prompt_text),
            })

        return prompts

    def _create_templates(self) -> List[PromptTemplate]:
        """Create prompt templates for different task types."""
        templates = []

        # 1. Function Implementation
        templates.extend(self._function_templates())

        # 2. Algorithm Implementation
        templates.extend(self._algorithm_templates())

        # 3. Data Structure Implementation
        templates.extend(self._data_structure_templates())

        # 4. Code Explanation
        templates.extend(self._explanation_templates())

        # 5. Bug Fixing
        templates.extend(self._bug_fix_templates())

        # 6. Code Refactoring
        templates.extend(self._refactoring_templates())

        # 7. Real-World Tasks
        templates.extend(self._real_world_templates())

        # 8. Code Review
        templates.extend(self._code_review_templates())

        return templates

    def _function_templates(self) -> List[PromptTemplate]:
        """Templates for function implementation tasks."""
        return [
            PromptTemplate(
                category='function_implementation',
                template='Write a {language} function that {task}',
                variables={
                    'language': ['Python', 'JavaScript', 'TypeScript', 'Java', 'C++', 'Rust', 'Go'],
                    'task': [
                        'calculates the factorial of a number',
                        'checks if a string is a palindrome',
                        'reverses a linked list',
                        'finds the nth Fibonacci number',
                        'sorts an array using quicksort',
                        'validates an email address',
                        'converts a decimal number to binary',
                        'finds the longest common substring',
                        'implements a simple calculator',
                        'parses a JSON string',
                        'generates a URL-friendly slug',
                        'flattens a nested list of numbers',
                        'evaluates a postfix expression',
                        'computes the greatest common divisor (GCD)',
                    ],
                },
            ),
            PromptTemplate(
                category='function_implementation',
                template='Implement a {language} function to {action} given {input}',
                variables={
                    'language': ['Python', 'JavaScript', 'Java', 'C++', 'Go', 'Rust'],
                    'action': [
                        'process',
                        'transform',
                        'validate',
                        'filter',
                        'aggregate',
                        'normalize',
                        'summarize',
                    ],
                    'input': [
                        'a list of numbers',
                        'a string',
                        'an array of objects',
                        'a dictionary',
                        'a stream of log lines',
                        'a matrix',
                    ],
                },
            ),
            PromptTemplate(
                category='function_implementation',
                template='Create a {language} utility that {goal} with proper error handling',
                variables={
                    'language': ['Python', 'JavaScript', 'TypeScript', 'Java', 'C#', 'Go'],
                    'goal': [
                        'parses and validates configuration files',
                        'debounces an async function call',
                        'memoizes expensive computations',
                        'streams large files without loading into memory',
                        'retries HTTP requests with exponential backoff',
                        'generates paginated results from a collection',
                    ],
                },
            ),
        ]

    def _algorithm_templates(self) -> List[PromptTemplate]:
        """Templates for algorithm implementation."""
        return [
            PromptTemplate(
                category='algorithm',
                template='Implement {algorithm} in {language} with detailed comments',
                variables={
                    'algorithm': [
                        'binary search',
                        'merge sort',
                        'quick sort',
                        'depth-first search (DFS)',
                        'breadth-first search (BFS)',
                        'Dijkstra\'s shortest path',
                        'A* pathfinding',
                        'dynamic programming for coin change',
                        'backtracking for N-queens',
                        'Kruskal\'s minimum spanning tree',
                        'topological sort with cycle detection',
                        'edit distance (Levenshtein) using DP',
                        'sliding window maximum',
                        'union-find with path compression',
                        'k-way merge using heaps',
                    ],
                    'language': ['Python', 'Java', 'C++', 'JavaScript', 'Rust', 'Go'],
                },
            ),
            PromptTemplate(
                category='algorithm',
                template='Write {language} code to solve: {problem}',
                variables={
                    'language': ['Python', 'JavaScript', 'Java', 'TypeScript'],
                    'problem': [
                        'Find the kth largest element in an array',
                        'Detect a cycle in a linked list',
                        'Find all permutations of a string',
                        'Implement LRU cache',
                        'Find the median of two sorted arrays',
                        'Longest increasing subsequence',
                        'Word ladder problem',
                        'Trapping rain water',
                        'Maximum subarray sum (Kadane)',
                        'Validate binary search tree',
                        'Rotate a matrix 90 degrees',
                        'Count islands in a grid',
                        'Serialize and deserialize a binary tree',
                    ],
                },
            ),
        ]

    def _data_structure_templates(self) -> List[PromptTemplate]:
        """Templates for data structure implementation."""
        return [
            PromptTemplate(
                category='data_structure',
                template='Implement a {structure} in {language} with {operations}',
                variables={
                    'structure': ['stack', 'queue', 'binary search tree', 'hash table', 'heap', 'trie', 'graph'],
                    'language': ['Python', 'Java', 'C++', 'JavaScript', 'Go', 'Rust'],
                    'operations': [
                        'insert, delete, and search operations',
                        'all basic operations',
                        'efficient lookup and insertion',
                        'O(1) get-min or get-max support',
                        'iterators with lazy traversal',
                    ],
                },
            ),
            PromptTemplate(
                category='data_structure',
                template='Design and implement a {structure} in {language} supporting {operations}',
                variables={
                    'structure': [
                        'disjoint set (union-find)',
                        'LRU cache',
                        'circular buffer',
                        'priority queue',
                        'segment tree',
                    ],
                    'language': ['Python', 'C++', 'Java', 'Rust'],
                    'operations': [
                        'union and find with path compression',
                        'O(1) eviction and retrieval',
                        'bounded capacity with wrap-around semantics',
                        'range queries and point updates',
                        'thread-safe operations',
                    ],
                },
            ),
        ]

    def _explanation_templates(self) -> List[PromptTemplate]:
        """Templates for code explanation tasks."""
        return [
            PromptTemplate(
                category='explanation',
                template='Explain what this {language} code does and how it works:\n{code_snippet}',
                variables={
                    'language': ['Python', 'JavaScript', 'Java'],
                    'code_snippet': [
                        'list(map(lambda x: x**2, filter(lambda x: x % 2 == 0, range(10))))',
                        '[x for x in range(100) if x % 3 == 0 and x % 5 == 0]',
                        'reduce((acc, x) => acc + x, [1, 2, 3, 4], 0)',
                    ],
                },
            ),
            PromptTemplate(
                category='explanation',
                template='Explain the time and space complexity of {algorithm}',
                variables={
                    'algorithm': [
                        'quicksort',
                        'merge sort',
                        'binary search',
                        'DFS and BFS',
                        'Dijkstra\'s algorithm',
                        'topological sort',
                        'two-pointer techniques',
                    ],
                },
            ),
        ]

    def _bug_fix_templates(self) -> List[PromptTemplate]:
        """Templates for bug fixing tasks."""
        return [
            PromptTemplate(
                category='bug_fix',
                template='Find and fix the bug in this {language} code:\n{buggy_code}\nThe code should {expected}',
                variables={
                    'language': ['Python', 'JavaScript', 'Java'],
                    'buggy_code': [
                        'def factorial(n):\n    if n == 1:\n        return 1\n    return n * factorial(n-1)',
                        'for (let i = 0; i <= arr.length; i++) { console.log(arr[i]); }',
                        'if (head.next == null) return false;\nwhile (head) {\n    head = head.next.next;\n}',
                        'for i in range(len(nums)):\n    if nums[i] == target:\n        return i\n    return -1',
                    ],
                    'expected': [
                        'handle n=0 correctly',
                        'not go out of bounds',
                        'avoid null pointer exceptions',
                        'return correct value for missing elements',
                    ],
                },
            ),
        ]

    def _refactoring_templates(self) -> List[PromptTemplate]:
        """Templates for refactoring tasks."""
        return [
            PromptTemplate(
                category='refactoring',
                template='Refactor this {language} code to be more {quality}:\n{code}',
                variables={
                    'language': ['Python', 'JavaScript', 'Java'],
                    'quality': ['readable', 'efficient', 'maintainable', 'pythonic', 'idiomatic'],
                    'code': ['# Code snippet here'],
                },
            ),
        ]

    def _real_world_templates(self) -> List[PromptTemplate]:
        """Templates for real-world programming tasks."""
        return [
            PromptTemplate(
                category='real_world',
                template='Write {language} code to {task}',
                variables={
                    'language': ['Python', 'JavaScript', 'Java'],
                    'task': [
                        'read a CSV file and calculate statistics',
                        'make an HTTP GET request and parse JSON',
                        'create a REST API endpoint',
                        'connect to a database and run a query',
                        'parse command-line arguments',
                        'read and write files',
                        'handle exceptions gracefully',
                        'implement logging',
                        'create a simple web scraper',
                        'validate user input',
                        'implement a CLI tool with subcommands',
                        'schedule recurring background tasks',
                        'stream large uploads with retry support',
                        'consume and publish messages to a queue',
                        'process image uploads and generate thumbnails',
                    ],
                },
            ),
        ]

    def _code_review_templates(self) -> List[PromptTemplate]:
        """Templates for code review tasks."""
        return [
            PromptTemplate(
                category='code_review',
                template='Review this {language} code and suggest improvements for {aspect}:\n{code}',
                variables={
                    'language': ['Python', 'JavaScript', 'Java'],
                    'aspect': ['performance', 'security', 'readability', 'best practices'],
                    'code': ['# Code snippet here'],
                },
            ),
        ]

    def _fill_template(self, template: PromptTemplate) -> str:
        """Fill a template with random values from its variables."""
        prompt = template.template

        for var_name, options in template.variables.items():
            value = random.choice(options)
            prompt = prompt.replace(f'{{{var_name}}}', value)

        return prompt

    def _extract_language(self, template: PromptTemplate, prompt: str) -> str:
        """Extract programming language from the prompt."""
        languages = [
            'Python',
            'JavaScript',
            'TypeScript',
            'Java',
            'C++',
            'Rust',
            'Go',
            'Ruby',
            'PHP',
            'C#',
            'Kotlin',
            'Swift',
        ]

        for lang in languages:
            if lang in prompt:
                return lang.lower()

        return 'unknown'


def generate_system_prompt() -> str:
    """
    Generate system prompt for the teacher model.

    This instructs the model on how to generate high-quality training data.
    """
    return """You are an expert programmer creating training data for a code generation model.

When given a coding task, provide:
1. Clear, well-commented code
2. Explanation of the approach
3. Time and space complexity analysis (if relevant)
4. Example usage
5. Edge cases to consider

Format your response as follows:

```language
# Your code here with comments
```

**Explanation:**
Brief explanation of how the code works.

**Complexity:**
Time: O(...)
Space: O(...)

**Example:**
```
Input: ...
Output: ...
```

**Notes:**
- Any important edge cases
- Potential optimizations
- Common pitfalls to avoid

Keep code clean, well-structured, and following best practices for the language."""


# Convenience function
def generate_prompts_from_categories(
    categories: List[str],
    num_prompts: int = 100
) -> List[Dict[str, str]]:
    """
    Generate prompts from specific categories.

    Args:
        categories: List of categories to include
        num_prompts: Total prompts to generate

    Returns:
        List of prompts

    Example:
        >>> prompts = generate_prompts_from_categories(
        ...     ['function_implementation', 'algorithm'],
        ...     num_prompts=50
        ... )
    """
    generator = PromptGenerator()
    all_prompts = generator.generate_prompts(num_prompts * 2)  # Generate extra

    # Filter by category
    filtered = [p for p in all_prompts if p['category'] in categories]

    # Trim to requested number
    return filtered[:num_prompts]
