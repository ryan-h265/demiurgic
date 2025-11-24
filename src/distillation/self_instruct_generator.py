"""
Self-Instruct and Curriculum-Based prompt generation.

Instead of using templates, this approach lets Claude/GPT-4 generate
both the coding tasks AND solutions, leveraging their full creative capability.
"""

from typing import List, Dict
import random


class SelfInstructGenerator:
    """
    Fully automated prompt generation using self-instruct methodology.

    The teacher model generates diverse coding tasks on its own,
    then solves them - no templates or constraints needed.
    """

    def __init__(self, seed: int = 42):
        random.seed(seed)

    def generate_meta_prompts(self, count: int) -> List[Dict[str, str]]:
        """
        Generate meta-prompts that ask the model to create its own tasks.

        This is the key: we ask Claude/GPT-4 to be creative and generate
        diverse, realistic coding challenges that they then solve.
        """

        meta_templates = [
            # === Open-Ended Generation ===
            {
                "category": "self_instruct_open",
                "prompt": """Generate a unique, practical coding task that would help train a coding assistant to be helpful in real-world development.

The task should:
- Be different from common textbook problems
- Include realistic context and constraints
- Be challenging but solvable
- Cover practical skills (not just algorithms)

After generating the task, provide a complete solution with:
- Clear explanations of your approach
- Working, well-commented code
- Discussion of edge cases and trade-offs
- Testing considerations

Format your response as:
## Task
[Your generated task with full context]

## Solution
[Your complete solution with reasoning and code]"""
            },

            # === Domain-Specific ===
            {
                "category": "self_instruct_domain",
                "prompt": """Create a coding challenge in the domain of {domain} that reflects real-world problems developers face.

Think of a specific, practical scenario and generate:
1. A clear problem statement with context
2. Requirements and constraints
3. A complete, production-quality solution
4. Explanation of why this approach is appropriate

Be creative - don't just rehash common interview questions. Think about what would actually be useful.

Domains to choose from: {domain}"""
            },

            # === Debugging Scenarios ===
            {
                "category": "self_instruct_debugging",
                "prompt": """Generate a realistic debugging scenario where:
1. You write code that has a subtle bug
2. You show the error/unexpected behavior
3. You debug it step-by-step with explanations
4. You provide the fixed version

Make the bug realistic - the kind that happens in real development, not obvious syntax errors.
Show your debugging thought process: how you identified it, why it happened, how to fix it, and how to prevent it."""
            },

            # === System Design ===
            {
                "category": "self_instruct_system",
                "prompt": """Design and implement a production-ready system component that solves a common software engineering problem.

Choose from: caching, rate limiting, retry logic, connection pooling, circuit breakers, queue systems, etc.

Provide:
1. Problem statement and requirements
2. Design decisions and trade-offs
3. Complete implementation with error handling
4. Usage examples and testing approach
5. Deployment considerations

Make it realistic - production code quality, not a toy example."""
            },

            # === Refactoring ===
            {
                "category": "self_instruct_refactor",
                "prompt": """Generate a realistic example of code that works but has quality issues (not obviously broken, but could be better).

Then show how to refactor it step-by-step:
1. Initial working code with issues
2. Analysis of problems (performance, readability, maintainability, etc.)
3. Refactored version with improvements
4. Explanation of why each change makes it better

Focus on real issues developers face, not trivial style fixes."""
            },

            # === Multi-File Projects ===
            {
                "category": "self_instruct_project",
                "prompt": """Design a small multi-file project that demonstrates good software architecture.

Create:
1. Project requirements and goals
2. Architecture decisions (why this structure)
3. Implementation of 2-3 key files with proper separation of concerns
4. How the components interact
5. Testing strategy

Show real-world project organization, not everything in one file."""
            },

            # === Performance Optimization ===
            {
                "category": "self_instruct_performance",
                "prompt": """Create a performance optimization challenge:

1. Write code that solves a problem but is inefficient
2. Profile or analyze why it's slow
3. Show optimized version(s)
4. Compare time/space complexity before and after
5. Explain when the optimization matters and when it doesn't

Choose a realistic scenario where performance actually matters."""
            },

            # === Testing & TDD ===
            {
                "category": "self_instruct_testing",
                "prompt": """Generate a Test-Driven Development (TDD) example:

1. Start with a problem statement
2. Write tests FIRST (with failing tests initially)
3. Implement code to make tests pass
4. Show the iterative process
5. Explain what makes these good tests

Demonstrate real TDD workflow, not just writing tests after the fact."""
            },

            # === API Design ===
            {
                "category": "self_instruct_api",
                "prompt": """Design a well-structured API for a specific use case:

1. Define the use case and requirements
2. Design the API interface (endpoints, methods, data structures)
3. Implement key functionality with error handling
4. Show usage examples
5. Discuss design decisions (RESTful principles, error handling, versioning, etc.)

Focus on good API design principles, not just making it work."""
            },

            # === Error Handling ===
            {
                "category": "self_instruct_errors",
                "prompt": """Create an example demonstrating production-quality error handling:

1. A function/module that interacts with external systems (API, database, files)
2. Multiple potential failure points
3. Proper error handling strategy (retries, fallbacks, logging, user feedback)
4. Examples of what happens when things fail
5. Testing error scenarios

Show enterprise-level error handling, not just try/catch."""
            },

            # === Code Review ===
            {
                "category": "self_instruct_review",
                "prompt": """Generate a code review scenario:

1. Present code that needs review (not broken, but has issues)
2. Conduct a thorough code review covering:
   - Security vulnerabilities
   - Performance concerns
   - Maintainability issues
   - Best practice violations
   - Potential bugs
3. Provide improved version with fixes
4. Explain each issue and why the fix is better

Think like a senior engineer reviewing a pull request."""
            },
        ]

        # Domain options for domain-specific template
        domains = [
            "web development (APIs, authentication, sessions)",
            "data processing (parsing, transformation, validation)",
            "system integration (external APIs, message queues)",
            "file operations (parsing, generation, streaming)",
            "concurrency (async, parallelism, race conditions)",
            "database operations (queries, transactions, migrations)",
            "CLI tools (argument parsing, user interaction)",
            "testing and quality (unit tests, integration tests, mocking)"
        ]

        meta_prompts = []
        for _ in range(count):
            template = random.choice(meta_templates)

            # Fill in domain if needed
            if "{domain}" in template["prompt"]:
                domain = random.choice(domains)
                prompt = template["prompt"].replace("{domain}", domain)
            else:
                prompt = template["prompt"]

            meta_prompts.append({
                "prompt": prompt,
                "category": template["category"],
                "language": "multi",  # Self-instruct can choose language
                "type": "meta",  # Flag this as meta-instruction
            })

        return meta_prompts


class CurriculumGenerator:
    """
    High-level curriculum-based generation.

    Provides learning goals and lets models figure out how to teach them.
    """

    def __init__(self, seed: int = 42):
        random.seed(seed)

    def generate_curriculum_prompts(self, count: int) -> List[Dict[str, str]]:
        """
        Generate high-level curriculum prompts.

        These give the model a learning objective and let it create
        appropriate examples to teach that concept.
        """

        curriculum_areas = [
            # === Core Programming Concepts ===
            {
                "category": "curriculum_algorithms",
                "prompt": "Generate and solve an algorithmic problem that teaches {concept}. Include explanation of the approach, complexity analysis, and when this technique is useful.",
                "concepts": [
                    "divide and conquer",
                    "dynamic programming",
                    "greedy algorithms",
                    "backtracking",
                    "two-pointer technique",
                    "sliding window",
                    "graph traversal",
                    "tree recursion"
                ]
            },

            # === Software Design ===
            {
                "category": "curriculum_design",
                "prompt": "Teach the concept of {pattern} through a practical example. Show why it's useful, when to apply it, and a complete implementation.",
                "concepts": [
                    "separation of concerns",
                    "dependency injection",
                    "strategy pattern",
                    "observer pattern",
                    "factory pattern",
                    "SOLID principles",
                    "composition over inheritance",
                    "interface segregation"
                ]
            },

            # === Best Practices ===
            {
                "category": "curriculum_practices",
                "prompt": "Demonstrate {practice} through a complete example. Show bad practice, why it's problematic, and the correct approach with explanations.",
                "concepts": [
                    "proper exception handling",
                    "defensive programming",
                    "resource management (context managers, RAII)",
                    "avoiding code duplication (DRY)",
                    "meaningful variable names and self-documenting code",
                    "proper logging practices",
                    "input validation and sanitization",
                    "secure coding practices"
                ]
            },

            # === Testing ===
            {
                "category": "curriculum_testing",
                "prompt": "Teach {testing_concept} with a comprehensive example showing the problem, test strategy, implementation, and why it matters.",
                "concepts": [
                    "unit testing with mocks and stubs",
                    "integration testing strategies",
                    "test-driven development workflow",
                    "testing edge cases and error conditions",
                    "property-based testing",
                    "testing asynchronous code",
                    "testing database operations",
                    "testing external API interactions"
                ]
            },

            # === Performance ===
            {
                "category": "curriculum_performance",
                "prompt": "Explain and demonstrate {optimization} through a practical example with measurements and analysis.",
                "concepts": [
                    "time complexity optimization (O(n²) to O(n log n))",
                    "space complexity trade-offs",
                    "caching strategies",
                    "lazy evaluation",
                    "batch processing",
                    "database query optimization",
                    "algorithmic improvements",
                    "data structure selection for performance"
                ]
            },

            # === Concurrency ===
            {
                "category": "curriculum_concurrency",
                "prompt": "Teach {concept} with a practical example showing the problem it solves, implementation, and common pitfalls.",
                "concepts": [
                    "async/await patterns",
                    "parallel processing",
                    "thread safety and race conditions",
                    "deadlock prevention",
                    "concurrent data structures",
                    "producer-consumer pattern",
                    "task queues",
                    "event-driven programming"
                ]
            },
        ]

        prompts = []
        for _ in range(count):
            area = random.choice(curriculum_areas)
            concept = random.choice(area["concepts"])
            prompt = area["prompt"].replace("{concept}", concept).replace("{pattern}", concept).replace("{practice}", concept).replace("{testing_concept}", concept).replace("{optimization}", concept)

            prompts.append({
                "prompt": prompt,
                "category": area["category"],
                "language": "auto",  # Model chooses appropriate language
                "type": "curriculum",
            })

        return prompts


class ChainOfThoughtGenerator:
    """
    Generate prompts focused on reasoning and thought processes.

    Captures HOW models think through problems, not just the final answer.
    """

    def __init__(self, seed: int = 42):
        random.seed(seed)

    def generate_reasoning_prompts(self, count: int) -> List[Dict[str, str]]:
        """Generate prompts that elicit chain-of-thought reasoning."""

        reasoning_templates = [
            {
                "category": "reasoning_analysis",
                "prompt": """You're given a coding problem: {problem}

Think through this step-by-step:
1. Restate the problem in your own words
2. Identify the key challenges
3. Consider 2-3 different approaches
4. Analyze trade-offs of each approach
5. Choose the best one and explain why
6. Implement it with clear explanations
7. Test with examples, including edge cases

Show your complete reasoning process, not just the final solution."""
            },

            {
                "category": "reasoning_debugging",
                "prompt": """Debug this problem by thinking out loud:

Given: {scenario}

Walk through your debugging process:
1. What do I observe?
2. What could cause this?
3. How can I test my hypothesis?
4. What's the root cause?
5. How do I fix it?
6. How do I prevent this in the future?

Show your thought process at each step."""
            },

            {
                "category": "reasoning_design",
                "prompt": """Design a solution for: {requirement}

Think through the design process:
1. Understand the requirements (restate them)
2. Identify constraints and non-functional requirements
3. Consider different architectural approaches
4. Evaluate trade-offs (complexity vs performance, etc.)
5. Make design decisions with justifications
6. Implement key components
7. Discuss scalability and maintenance

Show how you arrive at the design, not just the final design."""
            },
        ]

        # Problem/scenario examples
        problems = [
            "Finding the longest palindromic substring efficiently",
            "Implementing a URL shortener service",
            "Building a rate limiter for an API",
            "Designing a caching system with eviction policy",
            "Processing a large log file efficiently",
        ]

        scenarios = [
            "API sometimes returns 500 errors under load",
            "Memory usage grows unbounded over time",
            "Database queries are slow on large tables",
            "Race condition causing duplicate records",
            "Authentication fails intermittently",
        ]

        requirements = [
            "A job queue system that processes tasks asynchronously",
            "A file watcher that triggers actions on changes",
            "A retry mechanism with exponential backoff",
            "A connection pool for database connections",
            "A distributed cache with consistency guarantees",
        ]

        prompts = []
        for _ in range(count):
            template = random.choice(reasoning_templates)

            # Fill in examples
            prompt = template["prompt"]
            if "{problem}" in prompt:
                prompt = prompt.replace("{problem}", random.choice(problems))
            elif "{scenario}" in prompt:
                prompt = prompt.replace("{scenario}", random.choice(scenarios))
            elif "{requirement}" in prompt:
                prompt = prompt.replace("{requirement}", random.choice(requirements))

            prompts.append({
                "prompt": prompt,
                "category": template["category"],
                "language": "auto",
                "type": "reasoning",
            })

        return prompts


def generate_enhanced_system_prompt() -> str:
    """
    System prompt optimized for self-instruct and curriculum-based generation.

    Encourages the model to be creative, thorough, and educational.
    """
    return """You are an expert software engineer and educator helping to create training data for a coding assistant.

Your role is to:
1. Generate creative, realistic coding challenges (when asked)
2. Provide complete, production-quality solutions
3. Explain your reasoning and thought process
4. Demonstrate best practices and real-world considerations
5. Show how to handle edge cases and errors
6. Make it educational - teach concepts, don't just show code

Guidelines:
- Be creative - avoid cliché interview questions
- Be practical - focus on real development scenarios
- Be thorough - include error handling, testing, documentation
- Be educational - explain WHY, not just WHAT
- Use proper code formatting with markdown
- Include complexity analysis when relevant
- Show multiple approaches when appropriate
- Discuss trade-offs and design decisions

Your goal is to help train a model that can think like an experienced developer, not just memorize solutions."""


__all__ = [
    "SelfInstructGenerator",
    "CurriculumGenerator",
    "ChainOfThoughtGenerator",
    "generate_enhanced_system_prompt",
]
