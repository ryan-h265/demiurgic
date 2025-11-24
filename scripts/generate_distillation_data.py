#!/usr/bin/env python3
"""
Generate training data for knowledge distillation using multi-provider support.

This script:
1. Generates diverse coding prompts
2. Calls teacher API (Claude, GPT-4, or local model) to get responses
3. Filters for quality
4. Saves to JSONL format for training

Examples:
    # Generate 5000 examples with Claude 3.5 Sonnet
    python scripts/generate_distillation_data.py \\
        --provider anthropic \\
        --model claude-3-5-sonnet-20241022 \\
        --num-examples 5000

    # Generate 5000 examples with GPT-4-turbo
    python scripts/generate_distillation_data.py \\
        --provider openai \\
        --model gpt-4-turbo \\
        --num-examples 5000

    # Generate 10000 examples with local ChatGLM3 (free)
    python scripts/generate_distillation_data.py \\
        --provider local \\
        --model-path models/chatglm3-6b.Q4_K_M.gguf \\
        --num-examples 10000

    # Mix providers (40% Claude, 40% GPT-4, 20% local)
    python scripts/generate_distillation_data.py \\
        --providers anthropic,openai,local \\
        --mix-ratio 0.4,0.4,0.2 \\
        --num-examples 10000
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import json
import argparse
import random
from tqdm.asyncio import tqdm
from typing import List, Dict, Optional
import os

from src.distillation.providers import create_provider, ProviderType
from src.distillation.prompt_generator import PromptGenerator, generate_system_prompt
from src.distillation.self_instruct_generator import (
    SelfInstructGenerator,
    CurriculumGenerator,
    ChainOfThoughtGenerator,
    generate_enhanced_system_prompt
)
from src.distillation.quality_filters import QualityFilter, DuplicateFilter


class MultiProviderDataGenerator:
    """Generate training data from multiple teacher models."""

    def __init__(
        self,
        providers: List,
        output_dir: str = 'data/distillation',
        mode: str = 'template',
    ):
        """
        Initialize data generator with multiple providers.

        Args:
            providers: List of TeacherProvider instances
            output_dir: Directory to save generated data
            mode: Generation mode - 'template', 'self-instruct', 'curriculum', 'reasoning', or 'mixed'
        """
        self.providers = providers
        self.mode = mode
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize prompt generators based on mode
        self.template_gen = PromptGenerator() if mode in ['template', 'mixed'] else None
        self.self_instruct_gen = SelfInstructGenerator() if mode in ['self-instruct', 'mixed'] else None
        self.curriculum_gen = CurriculumGenerator() if mode in ['curriculum', 'mixed'] else None
        self.reasoning_gen = ChainOfThoughtGenerator() if mode in ['reasoning', 'mixed'] else None

        # Choose system prompt based on mode
        if mode in ['self-instruct', 'curriculum', 'reasoning', 'mixed']:
            self.system_prompt = generate_enhanced_system_prompt()
        else:
            self.system_prompt = generate_system_prompt()

        # Initialize filters
        self.quality_filter = QualityFilter(
            min_length=50,
            max_length=4000,
            require_code_blocks=True,
            check_refusals=True,
        )
        self.duplicate_filter = DuplicateFilter(similarity_threshold=0.9)

    def _generate_prompts(self, count: int) -> List[Dict[str, str]]:
        """Generate prompts using the selected mode."""
        if self.mode == 'template':
            return self.template_gen.sample(count)

        elif self.mode == 'self-instruct':
            return self.self_instruct_gen.generate_meta_prompts(count)

        elif self.mode == 'curriculum':
            return self.curriculum_gen.generate_curriculum_prompts(count)

        elif self.mode == 'reasoning':
            return self.reasoning_gen.generate_reasoning_prompts(count)

        elif self.mode == 'mixed':
            # Recommended mix: 40% self-instruct, 30% curriculum, 20% template, 10% reasoning
            prompts = []
            prompts.extend(self.self_instruct_gen.generate_meta_prompts(int(count * 0.4)))
            prompts.extend(self.curriculum_gen.generate_curriculum_prompts(int(count * 0.3)))
            prompts.extend(self.template_gen.sample(int(count * 0.2)))
            prompts.extend(self.reasoning_gen.generate_reasoning_prompts(int(count * 0.1)))
            random.shuffle(prompts)
            return prompts

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    async def generate_data(
        self,
        num_examples: int = 1000,
        save_checkpoints: bool = True,
    ) -> List[Dict]:
        """
        Generate training examples using configured providers.

        Args:
            num_examples: Number of examples to generate
            save_checkpoints: Whether to save checkpoints during generation

        Returns:
            List of training examples
        """
        print(f"\n{'='*70}")
        print(f"Generating {num_examples} training examples")
        print(f"{'='*70}")

        # Show provider info
        print(f"\nUsing {len(self.providers)} provider(s):")
        for i, provider in enumerate(self.providers, 1):
            print(f"  {i}. {provider.config.provider_type.value}: {provider.config.model_name}")
            estimated_cost = provider.estimate_cost(
                num_examples=num_examples // len(self.providers)
            )
            print(f"     Estimated cost: ${estimated_cost:.2f}")

        total_estimated_cost = sum(
            p.estimate_cost(num_examples // len(self.providers))
            for p in self.providers
        )
        print(f"\nTotal estimated cost: ${total_estimated_cost:.2f}")
        print()

        # Generate prompts based on mode
        print(f"Step 1: Generating prompts (mode: {self.mode})...")
        prompts_data = self._generate_prompts(num_examples)
        prompts = [p["prompt"] for p in prompts_data]
        print(f"✓ Generated {len(prompts)} prompts\n")

        # Show distribution
        categories = {}
        languages = {}
        for p in prompts_data:
            cat = p.get("category", "unknown")
            lang = p.get("language", "unknown")
            categories[cat] = categories.get(cat, 0) + 1
            languages[lang] = languages.get(lang, 0) + 1

        print("Prompt distribution:")
        print("  Categories:")
        for cat, count in sorted(categories.items(), key=lambda x: -x[1])[:5]:
            print(f"    • {cat}: {count}")
        print("  Languages:")
        for lang, count in sorted(languages.items(), key=lambda x: -x[1])[:5]:
            print(f"    • {lang}: {count}")
        print()

        # Distribute prompts across providers
        provider_batches = self._distribute_prompts(prompts, len(self.providers))

        # Generate responses from all providers concurrently
        print("Step 2: Calling teacher APIs...")
        print("(This may take a while...)\n")

        all_examples = []
        batch_size = 20  # Process 20 prompts at a time from each provider

        # Create progress bar
        total_batches = sum(
            (len(batch) + batch_size - 1) // batch_size
            for batch in provider_batches
        )
        pbar = tqdm(total=total_batches, desc="Generating")

        # Generate from each provider concurrently
        tasks = []
        for provider, prompts_batch in zip(self.providers, provider_batches):
            task = self._generate_from_provider(
                provider,
                prompts_batch,
                batch_size,
                pbar,
                save_checkpoints,
                all_examples,
            )
            tasks.append(task)

        # Wait for all providers to finish
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect results
        for result in results:
            if isinstance(result, Exception):
                print(f"\n⚠️  Provider error: {result}")
            else:
                all_examples.extend(result)

        pbar.close()

        print(f"\n✓ Generated {len(all_examples)} raw examples\n")

        # Show metrics from each provider
        print("Provider metrics:")
        for i, provider in enumerate(self.providers, 1):
            metrics = provider.get_metrics()
            print(f"  {i}. {provider.config.model_name}:")
            print(f"     Requests: {metrics['num_requests']}")
            print(f"     Tokens: {metrics['total_tokens']:,}")
            print(f"     Cost: ${metrics['total_cost']:.2f}")
            if metrics['num_errors'] > 0:
                print(f"     Errors: {metrics['num_errors']}")

        return all_examples

    async def _generate_from_provider(
        self,
        provider,
        prompts: List[str],
        batch_size: int,
        pbar,
        save_checkpoints: bool,
        all_examples: List,
    ) -> List[Dict]:
        """Generate examples from a single provider."""
        examples = []

        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]

            try:
                # Generate responses for this batch
                responses = await provider.generate_batch(
                    batch,
                    system_prompt=self.system_prompt,
                )

                # Add to examples
                for resp in responses:
                    example = {
                        'prompt': resp['prompt'],
                        'response': resp['response'],
                        'provider': provider.config.provider_type.value,
                        'model': provider.config.model_name,
                    }
                    examples.append(example)

                # Update progress
                pbar.update(1)

                # Save checkpoint periodically
                if save_checkpoints and len(all_examples) + len(examples) % 100 < batch_size:
                    self._save_checkpoint(all_examples + examples)

            except Exception as e:
                print(f"\n⚠️  Error in batch: {e}")
                continue

        return examples

    def _distribute_prompts(
        self,
        prompts: List[str],
        num_providers: int,
    ) -> List[List[str]]:
        """Distribute prompts evenly across providers."""
        batches = [[] for _ in range(num_providers)]
        for i, prompt in enumerate(prompts):
            batches[i % num_providers].append(prompt)
        return batches

    def filter_quality(self, examples: List[Dict]) -> List[Dict]:
        """Filter examples for quality and remove duplicates."""
        print("\nStep 3: Filtering for quality...")

        # Quality filtering
        filtered, stats = self.quality_filter.filter_batch(examples)
        self.quality_filter.print_stats(stats)

        # Duplicate filtering
        print("Removing duplicates...")
        deduped = []
        for example in filtered:
            if not self.duplicate_filter.is_duplicate(example['response']):
                deduped.append(example)

        removed_dupes = len(filtered) - len(deduped)
        if removed_dupes > 0:
            print(f"✓ Removed {removed_dupes} duplicate responses\n")

        return deduped

    def save_dataset(
        self,
        examples: List[Dict],
        split: str = 'train',
    ) -> None:
        """Save dataset to JSONL file."""
        output_file = self.output_dir / f"{split}.jsonl"

        print(f"\nStep 4: Saving to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + '\n')

        print(f"✓ Saved {len(examples)} examples")

        # Save metadata
        meta_file = self.output_dir / f"{split}_metadata.json"

        # Aggregate metrics from all providers
        total_cost = sum(p.get_metrics()['total_cost'] for p in self.providers)
        total_tokens = sum(p.get_metrics()['total_tokens'] for p in self.providers)

        metadata = {
            'num_examples': len(examples),
            'total_tokens': total_tokens,
            'total_cost': total_cost,
            'providers': [
                {
                    'type': p.config.provider_type.value,
                    'model': p.config.model_name,
                    'metrics': p.get_metrics(),
                }
                for p in self.providers
            ],
        }

        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ Saved metadata to {meta_file}\n")

    def _save_checkpoint(self, examples: List[Dict]) -> None:
        """Save checkpoint during generation."""
        checkpoint_file = self.output_dir / f"checkpoint_{len(examples)}.jsonl"
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + '\n')


async def main():
    parser = argparse.ArgumentParser(
        description='Generate knowledge distillation data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Provider config
    parser.add_argument(
        '--provider',
        choices=['anthropic', 'openai', 'local'],
        help='Teacher model provider (single provider mode)',
    )
    parser.add_argument(
        '--model',
        help='Model name (for anthropic/openai) or model identifier',
    )
    parser.add_argument(
        '--model-path',
        type=Path,
        help='Path to local GGUF model (required for --provider local)',
    )
    parser.add_argument(
        '--api-key',
        help='API key (or set via ANTHROPIC_API_KEY/OPENAI_API_KEY env vars)',
    )

    # Multi-provider config
    parser.add_argument(
        '--providers',
        help='Comma-separated list of providers for mixing (e.g., "anthropic,openai,local")',
    )
    parser.add_argument(
        '--mix-ratio',
        help='Comma-separated ratios for mixing providers (e.g., "0.4,0.4,0.2")',
    )

    # Generation config
    parser.add_argument(
        '--mode',
        choices=['template', 'self-instruct', 'curriculum', 'reasoning', 'mixed'],
        default='mixed',
        help='Prompt generation mode (default: mixed). '
             'template=use predefined templates, '
             'self-instruct=model generates tasks, '
             'curriculum=high-level concepts, '
             'reasoning=chain-of-thought, '
             'mixed=recommended blend of all'
    )
    parser.add_argument(
        '--num-examples',
        type=int,
        default=100,
        help='Number of examples to generate (default: 100)',
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Sampling temperature (default: 0.7)',
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=2048,
        help='Max tokens per response (default: 2048)',
    )

    # Output
    parser.add_argument(
        '--output-dir',
        default='data/distillation',
        help='Output directory (default: data/distillation)',
    )

    # Advanced
    parser.add_argument(
        '--max-concurrent',
        type=int,
        default=5,
        help='Max concurrent API calls per provider (default: 5)',
    )
    parser.add_argument(
        '--no-checkpoints',
        action='store_true',
        help='Disable checkpoint saving during generation',
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.provider and not args.providers:
        parser.error("Must specify either --provider or --providers")

    # Get API keys from environment if not provided
    anthropic_key = args.api_key or os.getenv('ANTHROPIC_API_KEY')
    openai_key = args.api_key or os.getenv('OPENAI_API_KEY')

    # Create providers
    providers = []

    if args.provider:
        # Single provider mode
        if args.provider == 'anthropic':
            if not anthropic_key:
                parser.error("ANTHROPIC_API_KEY not set")
            if not args.model:
                args.model = 'claude-3-5-sonnet-20241022'

            provider = create_provider(
                'anthropic',
                model_name=args.model,
                api_key=anthropic_key,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                max_concurrent=args.max_concurrent,
            )
            providers.append(provider)

        elif args.provider == 'openai':
            if not openai_key:
                parser.error("OPENAI_API_KEY not set")
            if not args.model:
                args.model = 'gpt-4-turbo'

            provider = create_provider(
                'openai',
                model_name=args.model,
                api_key=openai_key,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                max_concurrent=args.max_concurrent,
            )
            providers.append(provider)

        elif args.provider == 'local':
            if not args.model_path:
                parser.error("--model-path required for local provider")
            if not args.model_path.exists():
                parser.error(f"Model file not found: {args.model_path}")

            provider = create_provider(
                'local',
                model_name=args.model_path.stem,
                model_path=args.model_path,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            providers.append(provider)

    else:
        # Multi-provider mode
        provider_names = args.providers.split(',')

        # TODO: Implement multi-provider mixing
        # For now, just error out
        parser.error("Multi-provider mode (--providers) not yet implemented. Use --provider for single provider.")

    # Cost confirmation
    total_estimated_cost = sum(
        p.estimate_cost(args.num_examples // len(providers))
        for p in providers
    )

    if total_estimated_cost > 10.0:
        print(f"\n⚠️  Warning: Generating {args.num_examples} examples")
        print(f"   Estimated cost: ${total_estimated_cost:.2f}")
        response = input(f"\n   Continue? (yes/no): ")
        if response.lower() != 'yes':
            print("Cancelled.")
            return

    # Create generator
    generator = MultiProviderDataGenerator(
        providers=providers,
        output_dir=args.output_dir,
        mode=args.mode,
    )

    # Generate data
    examples = await generator.generate_data(
        num_examples=args.num_examples,
        save_checkpoints=not args.no_checkpoints,
    )

    # Filter for quality
    filtered_examples = generator.filter_quality(examples)

    # Save
    generator.save_dataset(filtered_examples, split='train')

    # Summary
    total_cost = sum(p.get_metrics()['total_cost'] for p in providers)
    total_tokens = sum(p.get_metrics()['total_tokens'] for p in providers)

    print(f"{'='*70}")
    print(f"Summary")
    print(f"{'='*70}")
    print(f"✓ Generated: {len(examples)} raw examples")
    print(f"✓ After filtering: {len(filtered_examples)} examples")
    print(f"✓ Total tokens: {total_tokens:,}")
    print(f"✓ Actual cost: ${total_cost:.2f}")
    print(f"✓ Saved to: {args.output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Review the data: cat {args.output_dir}/train.jsonl | head")
    print(f"  2. Train ChatGLM3: python scripts/train_chatglm3.py --data {args.output_dir}")
    print(f"{'='*70}\n")


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
