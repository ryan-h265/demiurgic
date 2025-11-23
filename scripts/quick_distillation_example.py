#!/usr/bin/env python3
"""
Quick example of knowledge distillation workflow.

This demonstrates the process without needing API keys - shows how it works.
For actual use, you'll need to set up API keys.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.distillation.prompt_generator import PromptGenerator, generate_system_prompt


def main():
    print("\n" + "="*60)
    print("Knowledge Distillation - Quick Example")
    print("="*60)

    # 1. Generate prompts
    print("\n1. Generating diverse coding prompts...")
    generator = PromptGenerator()
    prompts = generator.generate_prompts(num_prompts=20)

    print(f"   ✓ Generated {len(prompts)} prompts\n")

    # Show some examples
    print("   Example prompts:")
    for i, prompt_info in enumerate(prompts[:5], 1):
        print(f"\n   {i}. Category: {prompt_info['category']}")
        print(f"      Language: {prompt_info['language']}")
        print(f"      Prompt: {prompt_info['prompt'][:100]}...")

    # 2. Show distribution
    print(f"\n2. Prompt distribution:")
    categories = {}
    languages = {}

    for p in prompts:
        categories[p['category']] = categories.get(p['category'], 0) + 1
        languages[p['language']] = languages.get(p['language'], 0) + 1

    print(f"\n   By category:")
    for cat, count in sorted(categories.items()):
        print(f"     • {cat}: {count}")

    print(f"\n   By language:")
    for lang, count in sorted(languages.items()):
        print(f"     • {lang}: {count}")

    # 3. Show system prompt
    print(f"\n3. System prompt for teacher model:")
    print("-" * 60)
    print(generate_system_prompt())
    print("-" * 60)

    # 4. Explain workflow
    print(f"\n4. Knowledge Distillation Workflow:")
    print("""
    Step 1: Generate Prompts ✓ (you just saw this!)
        ├─ Create diverse coding tasks
        ├─ Mix languages, difficulties, categories
        └─ Balance distribution

    Step 2: Call Teacher API (needs API key)
        ├─ Use GPT-4, Claude, or local model
        ├─ Send prompts + system instructions
        ├─ Collect high-quality responses
        └─ Track cost and tokens

    Step 3: Filter Quality
        ├─ Remove short/incomplete responses
        ├─ Check for code presence
        ├─ Remove refusals
        └─ Validate format

    Step 4: Save Dataset
        ├─ JSONL format for training
        ├─ Save metadata (cost, tokens, etc.)
        └─ Ready for training!

    Step 5: Train Student Model
        ├─ Load distilled data
        ├─ Train on teacher's outputs
        ├─ Much cheaper than from-scratch
        └─ Better quality than raw data
    """)

    # 5. Next steps
    print("="*60)
    print("Next Steps:")
    print("="*60)
    print("""
    To actually generate training data, you need:

    1. Get API key:
       • OpenAI: https://platform.openai.com/api-keys
       • Anthropic: https://console.anthropic.com/
       • Or use local model (free but needs GPU)

    2. Set environment variable:
       export OPENAI_API_KEY='your-key-here'
       # or
       export ANTHROPIC_API_KEY='your-key-here'

    3. Generate data (start small!):
       python scripts/generate_distillation_data.py \\
           --provider openai \\
           --model gpt-4-turbo \\
           --num-examples 10 \\
           --output-dir data/distillation

    4. Check cost estimate before large runs:
       10 examples ≈ $0.50 - $1.50
       100 examples ≈ $5 - $15
       1000 examples ≈ $50 - $150
       10000 examples ≈ $500 - $1500

    5. Recommended: Start with 100 examples to test everything!

    See DISTILLATION_GUIDE.md for full details.
    """)


if __name__ == "__main__":
    main()
