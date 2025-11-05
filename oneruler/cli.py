"""
Command-line interface for OneRuler benchmark.
"""

import argparse
import sys
from pathlib import Path

from .benchmark import OneRulerBenchmark
from .providers import get_provider
from .tasks import (
    SingleNIAH, MultiKeyNIAH, MultiValueNIAH,
    MultiQueryNIAH, NoneNIAH, CWEEasy, CWEHard
)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="OneRuler: Multilingual Long-Context LLM Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tasks on GPT-4
  oneruler --provider openai --model gpt-4-turbo --all

  # Run Single NIAH task only
  oneruler --provider anthropic --model claude-3-5-sonnet-20241022 --task s-niah

  # Test multiple context lengths
  oneruler --provider google --model gemini-1.5-flash --context-lengths 8000 32000 64000

  # Run on a specific language
  oneruler --provider openai --model gpt-4-turbo --language zh --all

  # Run with custom number of examples
  oneruler --provider anthropic --model claude-3-5-sonnet-20241022 --task cwe-easy --num-examples 100
        """
    )

    # Provider configuration
    parser.add_argument(
        "--provider",
        required=True,
        choices=["openai", "anthropic", "google", "gemini", "qwen"],
        help="LLM provider to use"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name/ID (e.g., 'gpt-4-turbo', 'claude-3-5-sonnet-20241022', 'gemini-1.5-flash')"
    )
    parser.add_argument(
        "--api-key",
        help="API key (or set via environment variable)"
    )
    parser.add_argument(
        "--base-url",
        help="Base URL for API (for custom endpoints)"
    )

    # Task selection
    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument(
        "--all",
        action="store_true",
        help="Run all OneRuler tasks"
    )
    task_group.add_argument(
        "--task",
        choices=["s-niah", "mk-niah", "mv-niah", "mq-niah", "none-niah", "cwe-easy", "cwe-hard"],
        help="Run a specific task"
    )

    # Benchmark configuration
    parser.add_argument(
        "--language",
        default="en",
        help="Language code (default: en)"
    )
    parser.add_argument(
        "--context-lengths",
        nargs="+",
        type=int,
        default=[8000],
        help="Context lengths to test (default: 8000)"
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=50,
        help="Number of examples per task (default: 50)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1000,
        help="Maximum tokens for model response (default: 1000)"
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Output directory for results (default: results)"
    )

    args = parser.parse_args()

    # Initialize provider
    print(f"\nInitializing {args.provider} provider with model {args.model}...")
    try:
        provider_kwargs = {}
        if args.api_key:
            provider_kwargs["api_key"] = args.api_key
        if args.base_url:
            provider_kwargs["base_url"] = args.base_url

        provider = get_provider(
            provider_name=args.provider,
            model_name=args.model,
            **provider_kwargs
        )
    except Exception as e:
        print(f"Error initializing provider: {e}")
        sys.exit(1)

    # Initialize benchmark
    print(f"Setting up OneRuler benchmark for language: {args.language}")
    benchmark = OneRulerBenchmark(
        provider=provider,
        language=args.language,
        output_dir=args.output_dir
    )

    # Map task names to classes
    task_map = {
        "s-niah": SingleNIAH,
        "mk-niah": MultiKeyNIAH,
        "mv-niah": MultiValueNIAH,
        "mq-niah": MultiQueryNIAH,
        "none-niah": NoneNIAH,
        "cwe-easy": CWEEasy,
        "cwe-hard": CWEHard,
    }

    # Run benchmark
    try:
        if args.all:
            print("\nRunning all OneRuler tasks...")
            results = benchmark.run_all_tasks(
                context_lengths=args.context_lengths,
                num_examples=args.num_examples
            )

            # Print overall summary
            print("\n" + "="*60)
            print("OVERALL SUMMARY")
            print("="*60)
            for key, task_results in results.items():
                if task_results:
                    correct = sum(1 for r in task_results if r.evaluation.get("correct", False))
                    total = len(task_results)
                    accuracy = correct / total if total > 0 else 0.0
                    print(f"{key}: {accuracy:.2%} ({correct}/{total})")
            print("="*60 + "\n")

        else:
            task_class = task_map[args.task]
            print(f"\nRunning {args.task} task...")

            for context_length in args.context_lengths:
                results = benchmark.run_task(
                    task_class=task_class,
                    context_length=context_length,
                    num_examples=args.num_examples,
                    max_tokens=args.max_tokens
                )

        print("\n✅ Benchmark completed successfully!")
        print(f"Results saved to: {Path(args.output_dir).absolute()}\n")

    except KeyboardInterrupt:
        print("\n\n⚠️  Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error running benchmark: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
