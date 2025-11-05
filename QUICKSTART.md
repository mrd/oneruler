# OneRuler Quick Start Guide

Get up and running with the OneRuler benchmark in 5 minutes!

## Installation

```bash
# Navigate to the oneruler directory
cd oneruler

# Install with OpenAI support (recommended for quickstart)
pip install -e ".[openai]"
```

## Set Your API Key

```bash
export OPENAI_API_KEY="sk-your-api-key-here"
```

## Run Your First Benchmark

### Option 1: Command Line (Easiest)

```bash
# Run a quick test with 10 examples
oneruler --provider openai \
         --model gpt-3.5-turbo \
         --task s-niah \
         --num-examples 10

# Check results
ls results/
```

### Option 2: Python Script

Create a file `test.py`:

```python
from oneruler import OneRulerBenchmark
from oneruler.providers import get_provider
from oneruler.tasks import SingleNIAH

# Initialize
provider = get_provider("openai", "gpt-3.5-turbo")
benchmark = OneRulerBenchmark(provider=provider, output_dir="my_results")

# Run
results = benchmark.run_task(
    task_class=SingleNIAH,
    context_length=8000,
    num_examples=10
)

# Check accuracy
correct = sum(1 for r in results if r.evaluation['correct'])
print(f"Accuracy: {correct}/{len(results)}")
```

Run it:
```bash
python test.py
```

## Try Different Tasks

```bash
# Multi-Key NIAH (harder)
oneruler --provider openai --model gpt-3.5-turbo --task mk-niah --num-examples 10

# Common Word Extraction (aggregation task)
oneruler --provider openai --model gpt-3.5-turbo --task cwe-easy --num-examples 10

# Run all tasks (will take longer)
oneruler --provider openai --model gpt-3.5-turbo --all --num-examples 5
```

## Try Longer Contexts

```bash
# Test at 32K tokens
oneruler --provider openai \
         --model gpt-4-turbo \
         --task s-niah \
         --context-lengths 8000 32000 \
         --num-examples 10
```

## Try Other Providers

### Anthropic Claude

```bash
export ANTHROPIC_API_KEY="sk-ant-your-key"
pip install -e ".[anthropic]"

oneruler --provider anthropic \
         --model claude-3-5-sonnet-20241022 \
         --task s-niah \
         --num-examples 10
```

### Google Gemini

```bash
export GOOGLE_API_KEY="your-google-key"
pip install -e ".[google]"

oneruler --provider google \
         --model gemini-1.5-flash \
         --task s-niah \
         --num-examples 10
```

## View Results

Results are saved in the `results/` directory as JSON files:

```bash
# View latest results
cat results/*.json | jq '.[] | {task: .task_name, correct: .evaluation.correct}'

# Or open in Python
python
>>> import json
>>> with open('results/SingleNIAH_en_8000_gpt-3.5-turbo.json') as f:
...     results = json.load(f)
>>> results[0]['evaluation']['correct']
True
```

## Common Issues

### "API key not provided"
Make sure you've exported your API key:
```bash
echo $OPENAI_API_KEY  # Should show your key
```

### "Module not found"
Install the provider dependencies:
```bash
pip install -e ".[openai]"  # or anthropic, google, all
```

### Rate Limits
Add delays between requests or use fewer examples:
```bash
oneruler --provider openai --model gpt-3.5-turbo --task s-niah --num-examples 5
```

## Next Steps

1. **Full Evaluation**: Run all tasks with standard settings (50 examples)
2. **Compare Models**: Test multiple models side-by-side
3. **Custom Context**: Use your own documents as context
4. **Multilingual**: Test on other languages (see main README)

For more details, see the main [README.md](README.md) and [examples/](examples/) directory.

## Cost Estimation

Approximate costs for 50 examples per task (at 8K context):

| Model | Per Task | All 7 Tasks |
|-------|----------|-------------|
| GPT-3.5 Turbo | ~$0.10 | ~$0.70 |
| GPT-4 Turbo | ~$1.00 | ~$7.00 |
| Claude 3.5 Sonnet | ~$0.50 | ~$3.50 |
| Gemini 1.5 Flash | ~$0.05 | ~$0.35 |

*Note: Costs increase significantly with longer contexts (32K, 64K, 128K)*

## Support

- Check the [README](README.md) for detailed documentation
- Look at [examples/](examples/) for code samples
- Open an issue on GitHub for bugs
