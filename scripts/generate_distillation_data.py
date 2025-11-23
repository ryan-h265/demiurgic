#!/usr/bin/env python3
"""
Generate training data for knowledge distillation.

This script:
1. Generates diverse coding prompts
2. Calls teacher API to get responses
3. Filters for quality
4. Saves to JSONL format for training
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import json
import argparse
from tqdm import tqdm
from typing import List, Dict
import os

from src.distillation.teacher_api import create_teacher_api, TeacherConfig
from src.distillation.prompt_generator import PromptGenerator, generate_system_prompt


class DataGenerator:
    """Generate training data from teacher model."""

    def __init__(
        self,
        teacher_provider: str = 'openai',
        teacher_model: str = 'gpt-4-turbo',
        api_key: str = None,
        output_dir: str = 'data/distillation',
        max_concurrent: int = 5,
        rate_limit_delay: float = 1.0,
    ):
        self.teacher = create_teacher_api(
            provider=teacher_provider,
            model=teacher_model,
            api_key=api_key,
            max_concurrent=max_concurrent,
            rate_limit_delay=rate_limit_delay,
        )
        self.prompt_generator = PromptGenerator()
        self.system_prompt = generate_system_prompt()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.total_cost = 0.0
        self.total_tokens = 0

    async def generate_data(
        self,
        num_examples: int = 1000,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> List[Dict]:
        """
        Generate training examples.

        Args:
            num_examples: Number of examples to generate
            temperature: Sampling temperature
            max_tokens: Max tokens per response

        Returns:
            List of training examples
        """
        print(f"\n{'='*60}")
        print(f"Generating {num_examples} training examples")
        print(f"Teacher: {self.teacher.config.provider}/{self.teacher.config.model}")
        print(f"{'='*60}\n")

        # Generate prompts
        print("1. Generating prompts...")
        prompts = self.prompt_generator.generate_prompts(num_examples)
        print(f"   ✓ Generated {len(prompts)} prompts")

        # Show distribution
        categories = {}
        for p in prompts:
            categories[p['category']] = categories.get(p['category'], 0) + 1

        print(f"\n   Prompt distribution:")
        for cat, count in sorted(categories.items()):
            print(f"     • {cat}: {count}")

        # Generate responses from teacher
        print(f"\n2. Calling teacher API...")
        print(f"   (This may take a while...)")

        examples = []
        batch_size = 10  # Process in smaller batches for progress tracking

        for i in tqdm(range(0, len(prompts), batch_size), desc="   Generating"):
            batch = prompts[i:i+batch_size]
            batch_prompts = [p['prompt'] for p in batch]

            try:
                responses = await self.teacher.generate_batch(
                    batch_prompts,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    system_prompt=self.system_prompt,
                )

                for prompt_info, response in zip(batch, responses):
                    example = {
                        'prompt': prompt_info['prompt'],
                        'response': response['content'],
                        'category': prompt_info['category'],
                        'language': prompt_info['language'],
                        'tokens_used': response['tokens_used'],
                        'cost_estimate': response['cost_estimate'],
                    }
                    examples.append(example)

                    self.total_cost += response['cost_estimate']
                    self.total_tokens += response['tokens_used']

                # Save checkpoint every 100 examples
                if (i + batch_size) % 100 == 0:
                    self._save_checkpoint(examples, i + batch_size)

            except Exception as e:
                print(f"\n   ⚠️  Error in batch {i//batch_size}: {e}")
                continue

        print(f"\n   ✓ Generated {len(examples)} examples")
        print(f"   ✓ Total tokens: {self.total_tokens:,}")
        print(f"   ✓ Estimated cost: ${self.total_cost:.2f}")

        return examples

    def filter_quality(self, examples: List[Dict]) -> List[Dict]:
        """
        Filter examples for quality.

        Removes:
        - Empty or very short responses
        - Responses with error messages
        - Malformed responses
        """
        print(f"\n3. Filtering for quality...")
        filtered = []

        refusal_patterns = [
            "I cannot",
            "I'm unable to",
            "As an AI",
            "I don't have the ability",
            "I apologize, but",
        ]

        for ex in examples:
            response = ex['response']

            # Check minimum length
            if len(response) < 50:
                continue

            # Check for refusals
            if any(pattern in response for pattern in refusal_patterns):
                continue

            # Check for code presence (most tasks should have code)
            if ex['category'] != 'explanation' and '```' not in response:
                continue

            filtered.append(ex)

        removed = len(examples) - len(filtered)
        print(f"   ✓ Kept {len(filtered)}/{len(examples)} examples")
        if removed > 0:
            print(f"   ✓ Removed {removed} low-quality examples ({removed/len(examples)*100:.1f}%)")

        return filtered

    def save_dataset(self, examples: List[Dict], split: str = 'train'):
        """Save dataset to JSONL file."""
        output_file = self.output_dir / f"{split}.jsonl"

        print(f"\n4. Saving to {output_file}...")
        with open(output_file, 'w') as f:
            for ex in examples:
                f.write(json.dumps(ex) + '\n')

        print(f"   ✓ Saved {len(examples)} examples")

        # Save metadata
        meta_file = self.output_dir / f"{split}_metadata.json"
        metadata = {
            'num_examples': len(examples),
            'total_tokens': self.total_tokens,
            'estimated_cost': self.total_cost,
            'teacher_model': self.teacher.config.model,
            'teacher_provider': self.teacher.config.provider,
        }

        with open(meta_file, 'w') as f:
            json.dumps(metadata, f, indent=2)

        print(f"   ✓ Saved metadata to {meta_file}")

    def _save_checkpoint(self, examples: List[Dict], num: int):
        """Save checkpoint during generation."""
        checkpoint_file = self.output_dir / f"checkpoint_{num}.jsonl"
        with open(checkpoint_file, 'w') as f:
            for ex in examples:
                f.write(json.dumps(ex) + '\n')


async def main():
    parser = argparse.ArgumentParser(description='Generate knowledge distillation data')

    # Teacher config
    parser.add_argument('--provider', default='openai', choices=['openai', 'anthropic', 'openai-compatible'],
                        help='Teacher model provider')
    parser.add_argument('--model', default='gpt-4-turbo',
                        help='Teacher model name')
    parser.add_argument('--api-key', default=None,
                        help='API key (or set via environment variable)')

    # Generation config
    parser.add_argument('--num-examples', type=int, default=100,
                        help='Number of examples to generate (default: 100)')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature (default: 0.7)')
    parser.add_argument('--max-tokens', type=int, default=2048,
                        help='Max tokens per response (default: 2048)')

    # Output
    parser.add_argument('--output-dir', default='data/distillation',
                        help='Output directory')

    # Rate limiting
    parser.add_argument('--max-concurrent', type=int, default=5,
                        help='Max concurrent API calls (default: 5)')
    parser.add_argument('--rate-limit', type=float, default=1.0,
                        help='Delay between requests in seconds (default: 1.0)')

    args = parser.parse_args()

    # Confirmation for cost
    if args.num_examples >= 1000:
        print(f"\n⚠️  Warning: Generating {args.num_examples} examples")
        print(f"   Estimated cost: ${args.num_examples * 0.05:.2f} - ${args.num_examples * 0.15:.2f}")
        print(f"   This depends on your teacher model pricing.")
        response = input(f"\n   Continue? (yes/no): ")
        if response.lower() != 'yes':
            print("Cancelled.")
            return

    # Create generator
    generator = DataGenerator(
        teacher_provider=args.provider,
        teacher_model=args.model,
        api_key=args.api_key,
        output_dir=args.output_dir,
        max_concurrent=args.max_concurrent,
        rate_limit_delay=args.rate_limit,
    )

    # Generate data
    examples = await generator.generate_data(
        num_examples=args.num_examples,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    # Filter for quality
    filtered_examples = generator.filter_quality(examples)

    # Save
    generator.save_dataset(filtered_examples, split='train')

    # Summary
    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"✓ Generated: {len(examples)} examples")
    print(f"✓ After filtering: {len(filtered_examples)} examples")
    print(f"✓ Total tokens: {generator.total_tokens:,}")
    print(f"✓ Estimated cost: ${generator.total_cost:.2f}")
    print(f"✓ Saved to: {args.output_dir}")
    print(f"\nNext step: Train your model on this data!")
    print(f"  python scripts/train_with_distillation.py --data {args.output_dir}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
