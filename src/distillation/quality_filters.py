"""Quality filters for training data generation."""

import re
from typing import Dict, List, Tuple


class QualityFilter:
    """Filter for ensuring high-quality training examples."""

    # Patterns that indicate refusals or low-quality responses
    REFUSAL_PATTERNS = [
        r"I cannot",
        r"I can't",
        r"I'm not able to",
        r"I am not able to",
        r"As an AI",
        r"As a language model",
        r"I don't have the ability",
        r"I do not have the ability",
        r"I'm sorry, but I",
        r"I apologize, but I",
        r"I cannot assist with that",
        r"I'm unable to",
        r"I am unable to",
    ]

    def __init__(
        self,
        min_length: int = 50,
        max_length: int = 4000,
        require_code_blocks: bool = True,
        check_refusals: bool = True,
    ):
        """
        Initialize quality filter.

        Args:
            min_length: Minimum response length in characters
            max_length: Maximum response length in characters
            require_code_blocks: Whether to require code blocks (```) for code tasks
            check_refusals: Whether to check for refusal patterns
        """
        self.min_length = min_length
        self.max_length = max_length
        self.require_code_blocks = require_code_blocks
        self.check_refusals = check_refusals

    def filter_batch(
        self,
        examples: List[Dict[str, str]]
    ) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
        """
        Filter a batch of examples and return valid ones with stats.

        Args:
            examples: List of dicts with 'prompt' and 'response' keys

        Returns:
            Tuple of (filtered_examples, stats_dict)
        """
        filtered = []
        stats = {
            "total": len(examples),
            "passed": 0,
            "failed_min_length": 0,
            "failed_max_length": 0,
            "failed_refusal": 0,
            "failed_no_code": 0,
            "failed_empty": 0,
        }

        for example in examples:
            prompt = example.get("prompt", "")
            response = example.get("response", "")

            # Check if response is empty
            if not response or not response.strip():
                stats["failed_empty"] += 1
                continue

            # Check minimum length
            if len(response) < self.min_length:
                stats["failed_min_length"] += 1
                continue

            # Check maximum length
            if len(response) > self.max_length:
                stats["failed_max_length"] += 1
                continue

            # Check for refusals
            if self.check_refusals and self._contains_refusal(response):
                stats["failed_refusal"] += 1
                continue

            # Check for code blocks if this appears to be a code task
            if self.require_code_blocks and self._is_code_task(prompt):
                if not self._contains_code_block(response):
                    stats["failed_no_code"] += 1
                    continue

            # Example passed all filters
            filtered.append(example)
            stats["passed"] += 1

        return filtered, stats

    def _contains_refusal(self, text: str) -> bool:
        """Check if text contains refusal patterns."""
        for pattern in self.REFUSAL_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _is_code_task(self, prompt: str) -> bool:
        """Determine if prompt is asking for code."""
        code_keywords = [
            "write", "implement", "code", "function", "class",
            "program", "script", "algorithm", "refactor",
            "debug", "fix", "create a"
        ]
        prompt_lower = prompt.lower()
        return any(keyword in prompt_lower for keyword in code_keywords)

    def _contains_code_block(self, text: str) -> bool:
        """Check if text contains code blocks (```)."""
        return "```" in text

    def print_stats(self, stats: Dict[str, int]) -> None:
        """Pretty print filtering statistics."""
        total = stats["total"]
        passed = stats["passed"]
        pass_rate = (passed / total * 100) if total > 0 else 0

        print("\n" + "=" * 60)
        print("Quality Filter Statistics")
        print("=" * 60)
        print(f"Total examples: {total}")
        print(f"Passed: {passed} ({pass_rate:.1f}%)")
        print(f"\nFailure breakdown:")
        print(f"  Empty responses: {stats['failed_empty']}")
        print(f"  Too short (< {self.min_length} chars): {stats['failed_min_length']}")
        print(f"  Too long (> {self.max_length} chars): {stats['failed_max_length']}")
        print(f"  Contains refusal: {stats['failed_refusal']}")
        print(f"  Missing code blocks: {stats['failed_no_code']}")
        print("=" * 60 + "\n")


class DuplicateFilter:
    """Filter for removing duplicate or near-duplicate examples."""

    def __init__(self, similarity_threshold: float = 0.9):
        """
        Initialize duplicate filter.

        Args:
            similarity_threshold: Similarity threshold (0-1) for considering duplicates
        """
        self.similarity_threshold = similarity_threshold
        self.seen_responses = []

    def is_duplicate(self, response: str) -> bool:
        """
        Check if response is a duplicate or near-duplicate.

        Args:
            response: Response text to check

        Returns:
            True if duplicate, False otherwise
        """
        # Normalize response
        normalized = self._normalize(response)

        # Check exact duplicates first
        if normalized in self.seen_responses:
            return True

        # Check fuzzy duplicates (simple character-based similarity)
        for seen in self.seen_responses:
            if self._similarity(normalized, seen) >= self.similarity_threshold:
                return True

        # Not a duplicate, add to seen set
        self.seen_responses.append(normalized)
        return False

    def _normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        # Convert to lowercase
        return text.lower()

    def _similarity(self, text1: str, text2: str) -> float:
        """
        Calculate simple similarity between two texts.

        Uses character-based Jaccard similarity for speed.
        For more sophisticated detection, consider using embeddings.
        """
        # Convert to sets of character bigrams
        bigrams1 = set(text1[i:i+2] for i in range(len(text1)-1))
        bigrams2 = set(text2[i:i+2] for i in range(len(text2)-1))

        if not bigrams1 and not bigrams2:
            return 1.0
        if not bigrams1 or not bigrams2:
            return 0.0

        # Jaccard similarity
        intersection = len(bigrams1 & bigrams2)
        union = len(bigrams1 | bigrams2)
        return intersection / union

    def reset(self) -> None:
        """Reset the seen responses."""
        self.seen_responses = []


__all__ = ["QualityFilter", "DuplicateFilter"]
