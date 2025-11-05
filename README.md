# OneRuler: Multilingual Long-Context LLM Benchmark

A comprehensive Python implementation of the OneRuler benchmark for evaluating long-context language models across multiple languages.

Based on the paper: **"One ruler to measure them all: Benchmarking multilingual long-context language models"** (Kim et al., COLM 2025)

## 📋 Overview

OneRuler is a multilingual benchmark designed to evaluate long-context language models across 26 languages. It includes seven synthetic tasks that test both retrieval and aggregation capabilities:

### Retrieval Tasks (NIAH variants)
1. **Single-NIAH (S-NIAH)**: Classic needle-in-a-haystack with one magic number
2. **Multi-Key NIAH (MK-NIAH)**: Multiple needles with different keys
3. **Multi-Value NIAH (MV-NIAH)**: Multiple needles with same key, different values
4. **Multi-Query NIAH (MQ-NIAH)**: Multiple queries in one question
5. **None-NIAH**: Recognize when no correct answer exists

### Aggregation Tasks
6. **CWE-Easy**: Find 10 most common words (30x vs 3x frequency)
7. **CWE-Hard**: Find 10 most common words (20x vs 10x frequency)

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
cd oneruler

# Install with your preferred LLM provider
pip install -e .                    # Base installation
pip install -e ".[openai]"          # With OpenAI support
pip install -e ".[anthropic]"       # With Anthropic support
pip install -e ".[google]"          # With Google Gemini support
pip install -e ".[all]"             # With all providers
```

### Set Up API Keys

```bash
# OpenAI
export OPENAI_API_KEY="your-api-key"

# Anthropic
export ANTHROPIC_API_KEY="your-api-key"

# Google Gemini
export GOOGLE_API_KEY="your-api-key"

# Qwen (if using custom endpoint)
export QWEN_API_KEY="your-api-key"
export QWEN_BASE_URL="http://localhost:8000/v1"
```

### Run the Benchmark

```bash
# Run all tasks on GPT-4
oneruler --provider openai --model gpt-4-turbo --all

# Run Single NIAH task only
oneruler --provider anthropic --model claude-3-5-sonnet-20241022 --task s-niah

# Test multiple context lengths
oneruler --provider google --model gemini-1.5-flash \
         --context-lengths 8000 32000 64000 \
         --all

# Run on Chinese
oneruler --provider openai --model gpt-4-turbo \
         --language zh \
         --task s-niah \
         --num-examples 100
```

## 📖 Usage

### Command-Line Interface

```bash
oneruler [options]

Required Arguments:
  --provider {openai,anthropic,google,qwen}
                        LLM provider to use
  --model MODEL         Model name (e.g., 'gpt-4-turbo')
  --all                 Run all tasks
  OR
  --task {s-niah,mk-niah,mv-niah,mq-niah,none-niah,cwe-easy,cwe-hard}
                        Run specific task

Optional Arguments:
  --language LANG       Language code (default: en)
  --context-lengths L1 L2 ...
                        Context lengths to test (default: 8000)
  --num-examples N      Number of examples (default: 50)
  --max-tokens N        Max response tokens (default: 1000)
  --output-dir DIR      Output directory (default: results)
  --api-key KEY         API key
  --base-url URL        Custom API endpoint
```

### Python API

```python
from oneruler import OneRulerBenchmark
from oneruler.providers import get_provider
from oneruler.tasks import SingleNIAH, CWEEasy

# Initialize provider
provider = get_provider(
    provider_name="openai",
    model_name="gpt-4-turbo"
)

# Create benchmark
benchmark = OneRulerBenchmark(
    provider=provider,
    language="en",
    output_dir="results"
)

# Run a specific task
results = benchmark.run_task(
    task_class=SingleNIAH,
    context_length=8000,
    num_examples=50
)

# Run all tasks
all_results = benchmark.run_all_tasks(
    context_lengths=[8000, 32000, 64000],
    num_examples=50
)

# Access results
for result in results:
    print(f"Correct: {result.evaluation['correct']}")
    print(f"Response: {result.response}")
```

### Adding Custom LLM Providers

```python
from oneruler.providers import LLMProvider

class CustomProvider(LLMProvider):
    """Custom LLM provider implementation."""

    def generate(self, prompt, max_tokens=1000, temperature=0.0, **kwargs):
        # Your implementation here
        response = your_llm_api_call(prompt, max_tokens, temperature)

        return {
            "response": response.text,
            "tokens_used": {
                "prompt": response.prompt_tokens,
                "completion": response.completion_tokens,
                "total": response.total_tokens
            },
            "model": self.model_name
        }

    def count_tokens(self, text):
        # Your tokenizer implementation
        return len(text.split()) * 1.3  # Rough estimate

# Use your custom provider
from oneruler import OneRulerBenchmark

provider = CustomProvider(model_name="your-model")
benchmark = OneRulerBenchmark(provider=provider)
```

## 📊 Supported Models

### Out-of-the-Box Support
- **OpenAI**: GPT-4 Turbo, GPT-4, GPT-3.5 Turbo
- **Anthropic**: Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Sonnet
- **Google**: Gemini 1.5 Flash, Gemini 1.5 Pro
- **Qwen**: Qwen 2.5 (via API or local deployment)

### Easy to Add
- Any OpenAI-compatible API
- Custom model endpoints
- Local models via vLLM, Ollama, etc.

## 🌍 Supported Languages

Currently implemented for English (`en`), but designed to support:

Chinese (zh), Czech (cs), Danish (da), Dutch (nl), English (en), Finnish (fi), French (fr), German (de), Hindi (hi), Hungarian (hu), Italian (it), Japanese (ja), Korean (ko), Norwegian (no), Persian (fa), Polish (pl), Portuguese (pt), Russian (ru), Serbian (sr), Sesotho (st), Spanish (es), Swahili (sw), Swedish (sv), Tamil (ta), Ukrainian (uk), Vietnamese (vi)

## 📁 Project Structure

```
oneruler/
├── oneruler/
│   ├── __init__.py          # Package initialization
│   ├── tasks.py             # Task implementations
│   ├── providers.py         # LLM provider abstractions
│   ├── benchmark.py         # Main benchmark orchestration
│   └── cli.py               # Command-line interface
├── results/                 # Benchmark results (created automatically)
├── setup.py                 # Package setup
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## 🔧 Development

### Running Tests

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests (coming soon)
pytest tests/

# Format code
black oneruler/

# Lint code
flake8 oneruler/
```

### Contributing

Contributions are welcome! Areas for improvement:
- Add support for more languages
- Implement multilingual context loading
- Add more LLM providers
- Improve evaluation metrics
- Add visualization tools

## 📝 Citation

If you use OneRuler in your research, please cite the original paper:

```bibtex
@inproceedings{kim2025oneruler,
  title={One ruler to measure them all: Benchmarking multilingual long-context language models},
  author={Kim, Yekyung and Russell, Jenna and Karpinska, Marzena and Iyyer, Mohit},
  booktitle={Conference on Language Modeling (COLM)},
  year={2025}
}
```

## 📄 License

This implementation is provided as-is for research purposes.

## 🔗 Links

- [Original Paper (arXiv)](https://arxiv.org/abs/2503.01996)
- [Original Repository](https://github.com/mungg/OneRuler)
- [COLM 2025](https://colmweb.org/)

## 🙏 Acknowledgments

This implementation is based on the OneRuler benchmark by Kim et al. (2025). The original research was supported by NSF awards IIS-2046248, IIS-2312949, and IIS-2202506.
