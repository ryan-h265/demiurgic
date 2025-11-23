# Demiurgic - Code-Focused GPT Model

A research project to build a GPT-based language model trained from scratch, optimized for code generation and software development assistance.

## Project Overview

**Demiurgic** is a medium-to-large scale transformer model (7B-70B parameters) designed to excel at:

- **Code Completion & Generation**: Context-aware code synthesis across multiple languages
- **Code Explanation**: Natural language descriptions of code functionality
- **Bug Detection & Fixing**: Identifying and correcting code errors
- **Refactoring**: Suggesting and implementing code improvements
- **Multi-Language Support**: Python, JavaScript, TypeScript, Go, Rust, Java, C++, and more

### Training Approach

This project uses **knowledge distillation** from a large teacher model (GPT-4, Claude, or similar) rather than training from scratch. This approach:

- **Reduces costs** by 50-70% ($3,000-9,000 vs $15,000-20,000)
- **Improves quality** through learning from high-quality teacher outputs
- **Decreases data requirements** (20-50B tokens vs 140B tokens)
- **Shortens training time** (2-4 weeks vs 6-7 weeks)

## Key Features

- **Trained from Scratch**: Full control over architecture and training process
- **Code-First Design**: Architecture optimized for code understanding and generation
- **Budget-Conscious**: Training strategy designed for cloud compute with cost optimization
- **CLI-First Interface**: Direct command-line integration for developer workflows
- **Research-Oriented**: Emphasis on experimentation and iteration

## Quick Start

### Prerequisites

- Python 3.10+
- Cloud compute account (AWS/GCP/Azure)
- Access to code datasets (The Stack, GitHub dumps, etc.)

### Repository Structure

```
demiurgic/
├── docs/                    # Detailed documentation
│   ├── architecture.md      # Model architecture specifications
│   ├── training.md          # Training guide and infrastructure
│   ├── data.md              # Data preparation and curation
│   ├── evaluation.md        # Benchmarking and metrics
│   └── cli.md               # CLI tool implementation
├── src/                     # Source code
│   ├── model/               # Model architecture
│   ├── training/            # Training scripts
│   ├── data/                # Data processing pipelines
│   └── cli/                 # CLI interface
├── configs/                 # Training configurations
├── scripts/                 # Utility scripts
└── tests/                   # Test suite
```

## Documentation

### Core Guides
- **[Knowledge Distillation Guide](docs/knowledge_distillation.md)**: **START HERE** - Training using teacher model (recommended approach)
- **[Architecture Guide](docs/architecture.md)**: Model design, tokenization, and scaling strategies
- **[Training Guide](docs/training.md)**: Infrastructure setup, training loops, and optimization
- **[Evaluation Guide](docs/evaluation.md)**: Metrics, benchmarks, and testing protocols
- **[CLI Guide](docs/cli.md)**: Building the command-line interface

### Alternative Approaches
- **[Data Guide](docs/data.md)**: Dataset curation for training from scratch (higher cost alternative)

## Model Specifications

### Architecture Options

**Medium Scale (7-13B parameters)**
- Training Time: 2-4 weeks
- Estimated Cost: $5,000-$15,000
- Hardware: 8x A100 GPUs
- Context Length: 4K-8K tokens

**Large Scale (30-70B parameters)**
- Training Time: 4-8 weeks
- Estimated Cost: $30,000-$100,000
- Hardware: 16-64 A100 GPUs
- Context Length: 8K-16K tokens

### Training Philosophy

1. **Iterative Development**: Start with smaller models, validate approach
2. **Data Quality > Quantity**: Curated, high-quality code datasets
3. **Task-Specific Fine-Tuning**: Multi-stage training for different capabilities
4. **Continuous Evaluation**: Regular benchmarking against code tasks

## Getting Started

### Recommended Path (Knowledge Distillation)

1. **Review Distillation Guide**: Read [docs/knowledge_distillation.md](docs/knowledge_distillation.md) - **START HERE**
2. **Review Architecture**: Read [docs/architecture.md](docs/architecture.md)
3. **Choose Teacher Model**: Select GPT-4, Claude, or open-source alternative
4. **Generate Training Data**: Use teacher to create 30k-50k examples
5. **Setup Infrastructure**: Follow [docs/training.md](docs/training.md)
6. **Train Student Model**: 7B model on distilled data (2-3 weeks)
7. **Evaluate**: Use [docs/evaluation.md](docs/evaluation.md) benchmarks

### Alternative Path (From Scratch)

For those with larger budgets or specific data requirements:

1. **Prepare Data**: Execute [docs/data.md](docs/data.md) pipeline
2. **Train from Scratch**: 4-7 weeks on 140B tokens
3. **Higher Cost**: $15,000-20,000 vs $3,000-9,000

## Research Goals

- Investigate optimal architecture for code understanding
- Explore efficient training techniques for budget constraints
- Develop robust evaluation metrics for code generation
- Study multi-language transfer learning in code domain

## License

[To be determined]

## Contributing

This is a research project. Documentation and implementation feedback welcome.

## Contact

[Your contact information]
