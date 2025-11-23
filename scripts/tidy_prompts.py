#!/usr/bin/env python3
"""
Tidy up example prompts and answers into clean JSONL format for training.

This script:
1. Reads prompt files and their corresponding answer files
2. Combines them into proper JSONL format
3. Validates and cleans the data
4. Outputs ready-to-use training files
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import re

sys.path.insert(0, str(Path(__file__).parent.parent))


class PromptTidier:
    """Tidy up prompt/answer pairs into training format."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.stats = {
            'total': 0,
            'valid': 0,
            'missing_answer': 0,
            'language_mismatch': 0,
            'fixed': 0,
        }

    def read_prompt_file(self, file_path: Path) -> Tuple[str, str, str]:
        """
        Read a prompt file and extract category, language, and prompt text.

        Returns:
            (category, language, prompt_text)
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        lines = content.strip().split('\n')
        category = lines[0].replace('Category: ', '').strip()
        language = lines[1].replace('Language: ', '').strip()

        # The prompt is after the separator line (============)
        prompt_start = content.find('============')
        if prompt_start >= 0:
            prompt_text = content[prompt_start:].split('\n', 2)[-1].strip()
        else:
            prompt_text = '\n'.join(lines[2:]).strip()

        return category, language, prompt_text

    def read_answer_file(self, file_path: Path) -> str:
        """Read an answer file and return the response text."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()

    def detect_code_language(self, text: str) -> str:
        """
        Detect the programming language from code snippets.

        Returns:
            Detected language or 'unknown'
        """
        # Check for common language indicators
        patterns = {
            'python': [r'def \w+\(', r'import \w+', r'\bfor \w+ in\b', r':\s*$'],
            'javascript': [r'=>', r'function\s+\w+', r'const\s+\w+', r'let\s+\w+', r'var\s+\w+'],
            'java': [r'public\s+class', r'public\s+static', r'void\s+\w+\(', r'System\.out'],
            'rust': [r'fn\s+\w+', r'let\s+mut', r'impl\s+\w+', r'pub\s+fn'],
            'c++': [r'std::', r'#include\s*<', r'cout\s*<<', r'vector<'],
            'go': [r'func\s+\w+', r'package\s+\w+', r'import\s+\(', r':='],
        }

        for lang, lang_patterns in patterns.items():
            if any(re.search(pattern, text, re.MULTILINE) for pattern in lang_patterns):
                return lang

        return 'unknown'

    def validate_and_fix(self, category: str, language: str, prompt: str, response: str) -> Tuple[str, bool]:
        """
        Validate the language label matches the actual code.

        Returns:
            (corrected_language, was_fixed)
        """
        # For explanation tasks, check if the code snippet matches the language
        if category == 'explanation' and language != 'unknown':
            # Extract code from prompt (usually after "code does and how it works:")
            code_match = re.search(r'(?:code does and how it works:|code:)\s*\n(.+)', prompt, re.DOTALL)
            if code_match:
                code_snippet = code_match.group(1).strip()
                detected = self.detect_code_language(code_snippet)

                # If we detected a different language, fix it
                if detected != 'unknown' and detected != language:
                    self.stats['language_mismatch'] += 1
                    self.stats['fixed'] += 1
                    return detected, True

        return language, False

    def process_directory(self, prompts_dir: Path, answers_dir: Path) -> List[Dict]:
        """
        Process a directory of prompts and answers.

        Returns:
            List of training examples
        """
        examples = []

        # Get all prompt files
        prompt_files = sorted(prompts_dir.glob('**/*.txt'))

        for prompt_file in prompt_files:
            self.stats['total'] += 1

            # Find corresponding answer file
            relative_path = prompt_file.relative_to(prompts_dir)
            answer_file = answers_dir / relative_path

            if not answer_file.exists():
                self.stats['missing_answer'] += 1
                print(f"⚠️  Missing answer for: {relative_path}")
                continue

            try:
                # Read files
                category, language, prompt = self.read_prompt_file(prompt_file)
                response = self.read_answer_file(answer_file)

                # Validate and fix if needed
                corrected_language, was_fixed = self.validate_and_fix(
                    category, language, prompt, response
                )

                if was_fixed:
                    print(f"✓ Fixed language: {relative_path} ({language} → {corrected_language})")

                # Create training example
                example = {
                    'prompt': prompt,
                    'response': response,
                    'category': category,
                    'language': corrected_language,
                }

                examples.append(example)
                self.stats['valid'] += 1

            except Exception as e:
                print(f"❌ Error processing {relative_path}: {e}")
                continue

        return examples

    def save_jsonl(self, examples: List[Dict], output_file: Path):
        """Save examples to JSONL file."""
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')

        print(f"\n✓ Saved {len(examples)} examples to {output_file}")

    def print_stats(self):
        """Print processing statistics."""
        print(f"\n{'='*60}")
        print("Processing Statistics")
        print(f"{'='*60}")
        print(f"Total prompts found:    {self.stats['total']}")
        print(f"Valid examples created: {self.stats['valid']}")
        print(f"Missing answers:        {self.stats['missing_answer']}")
        print(f"Language mismatches:    {self.stats['language_mismatch']}")
        print(f"Auto-fixed:             {self.stats['fixed']}")
        print(f"{'='*60}\n")


def main():
    """Main entry point."""
    base_dir = Path(__file__).parent.parent / 'prompts'
    output_dir = Path(__file__).parent.parent / 'data' / 'distillation'

    tidier = PromptTidier(base_dir)

    print("Tidying up prompt/answer pairs for training...\n")

    all_examples = []

    # Process 'examples' directory (original 100 examples)
    if (base_dir / 'examples').exists():
        print("Processing examples/...")
        examples = tidier.process_directory(
            base_dir / 'examples',
            base_dir / 'examples_answers'
        )
        all_examples.extend(examples)
        print(f"  ✓ Processed {len(examples)} examples\n")

    # Process 'generated_examples' directory (500 generated examples)
    if (base_dir / 'generated_examples').exists():
        print("Processing generated_examples/...")
        generated = tidier.process_directory(
            base_dir / 'generated_examples',
            base_dir / 'generated_examples_answers'
        )
        all_examples.extend(generated)
        print(f"  ✓ Processed {len(generated)} examples\n")

    # Print statistics
    tidier.print_stats()

    # Save combined output
    if all_examples:
        output_file = output_dir / 'train.jsonl'
        tidier.save_jsonl(all_examples, output_file)

        # Also create a smaller dev/validation set (10%)
        dev_size = max(10, len(all_examples) // 10)
        dev_examples = all_examples[:dev_size]
        train_examples = all_examples[dev_size:]

        tidier.save_jsonl(train_examples, output_dir / 'train_split.jsonl')
        tidier.save_jsonl(dev_examples, output_dir / 'dev.jsonl')

        print(f"\nCreated splits:")
        print(f"  • train_split.jsonl: {len(train_examples)} examples (for training)")
        print(f"  • dev.jsonl:         {len(dev_examples)} examples (for validation)")
        print(f"  • train.jsonl:       {len(all_examples)} examples (all data)")

        # Show category distribution
        categories = {}
        languages = {}
        for ex in all_examples:
            cat = ex['category']
            lang = ex['language']
            categories[cat] = categories.get(cat, 0) + 1
            languages[lang] = languages.get(lang, 0) + 1

        print(f"\nCategory distribution:")
        for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
            print(f"  • {cat:25s}: {count:4d} ({count/len(all_examples)*100:5.1f}%)")

        print(f"\nLanguage distribution:")
        for lang, count in sorted(languages.items(), key=lambda x: -x[1]):
            print(f"  • {lang:25s}: {count:4d} ({count/len(all_examples)*100:5.1f}%)")
    else:
        print("⚠️  No examples found!")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
