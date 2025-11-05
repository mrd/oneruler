"""
Basic usage examples for the OneRuler benchmark.
"""

import os
from oneruler import OneRulerBenchmark
from oneruler.providers import get_provider
from oneruler.tasks import SingleNIAH, MultiKeyNIAH, CWEEasy


def example_single_task():
    """Run a single task with OpenAI."""
    print("Example 1: Running Single NIAH task\n")

    # Initialize OpenAI provider
    provider = get_provider(
        provider_name="openai",
        model_name="gpt-3.5-turbo"  # Use cheaper model for testing
    )

    # Create benchmark
    benchmark = OneRulerBenchmark(
        provider=provider,
        language="en",
        output_dir="results/example1"
    )

    # Run Single NIAH task
    results = benchmark.run_task(
        task_class=SingleNIAH,
        context_length=8000,
        num_examples=5,  # Just 5 examples for demo
        max_tokens=500
    )

    # Print some results
    print("\nSample Results:")
    for i, result in enumerate(results[:3]):
        print(f"\n--- Example {i+1} ---")
        print(f"Correct: {result.evaluation['correct']}")
        print(f"Ground Truth: {result.ground_truth}")
        print(f"Response: {result.response[:200]}...")


def example_multiple_providers():
    """Compare multiple providers on the same task."""
    print("\n" + "="*60)
    print("Example 2: Comparing multiple providers")
    print("="*60 + "\n")

    # Define providers to test
    providers_config = [
        ("openai", "gpt-3.5-turbo"),
        ("anthropic", "claude-3-5-sonnet-20241022"),
        # ("google", "gemini-1.5-flash"),  # Uncomment if you have API key
    ]

    results_by_provider = {}

    for provider_name, model_name in providers_config:
        print(f"\nTesting {provider_name} - {model_name}...")

        try:
            # Initialize provider
            provider = get_provider(
                provider_name=provider_name,
                model_name=model_name
            )

            # Create benchmark
            benchmark = OneRulerBenchmark(
                provider=provider,
                language="en",
                output_dir=f"results/example2/{provider_name}"
            )

            # Run task
            results = benchmark.run_task(
                task_class=MultiKeyNIAH,
                context_length=8000,
                num_examples=10
            )

            # Calculate accuracy
            correct = sum(1 for r in results if r.evaluation.get("correct", False))
            accuracy = correct / len(results)

            results_by_provider[f"{provider_name}/{model_name}"] = accuracy

        except Exception as e:
            print(f"  Error: {e}")
            results_by_provider[f"{provider_name}/{model_name}"] = None

    # Print comparison
    print("\n" + "="*60)
    print("Comparison Results")
    print("="*60)
    for model, accuracy in results_by_provider.items():
        if accuracy is not None:
            print(f"{model}: {accuracy:.2%}")
        else:
            print(f"{model}: FAILED")


def example_all_tasks():
    """Run all tasks on a single model."""
    print("\n" + "="*60)
    print("Example 3: Running all tasks")
    print("="*60 + "\n")

    # Initialize provider
    provider = get_provider(
        provider_name="openai",
        model_name="gpt-4-turbo"
    )

    # Create benchmark
    benchmark = OneRulerBenchmark(
        provider=provider,
        language="en",
        output_dir="results/example3"
    )

    # Run all tasks with shorter contexts for demo
    results = benchmark.run_all_tasks(
        context_lengths=[8000],  # Just one context length
        num_examples=10  # Fewer examples for demo
    )

    # Print summary
    print("\n" + "="*60)
    print("Summary Across All Tasks")
    print("="*60)
    for task_name, task_results in results.items():
        if task_results:
            correct = sum(1 for r in task_results if r.evaluation.get("correct", False))
            total = len(task_results)
            accuracy = correct / total
            print(f"{task_name}: {accuracy:.2%} ({correct}/{total})")


def example_custom_context():
    """Use custom context text for NIAH tasks."""
    print("\n" + "="*60)
    print("Example 4: Using custom context")
    print("="*60 + "\n")

    from oneruler.tasks import SingleNIAH

    # Custom context (e.g., your own documents)
    custom_context = """
    This is a custom document that you want to use as the background context
    for your needle-in-a-haystack evaluation. It could be technical documentation,
    legal text, or any other domain-specific content that you want to test your
    model on. The needle will be injected into this context at a random position.
    """ * 500  # Repeat to make it longer

    # Initialize task directly
    task = SingleNIAH(language="en", context_length=8000)

    # Generate example with custom context
    example = task.generate_example(context_text=custom_context, word="database")

    print(f"Generated prompt length: {len(example.prompt)} characters")
    print(f"Ground truth: {example.answer}")
    print(f"Needle word: {example.metadata['word']}")

    # Now you can send this to any LLM
    provider = get_provider("openai", "gpt-3.5-turbo")
    response = provider.generate(example.prompt, max_tokens=100)

    # Evaluate
    evaluation = task.evaluate(response["response"], example.answer)
    print(f"\nCorrect: {evaluation['correct']}")
    print(f"Response: {response['response']}")


if __name__ == "__main__":
    # Check if API keys are set
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set. Some examples may fail.")
        print("Set it with: export OPENAI_API_KEY='your-key'\n")

    # Run examples
    # Uncomment the examples you want to run:

    example_single_task()
    # example_multiple_providers()  # Requires multiple API keys
    # example_all_tasks()  # More expensive, uses GPT-4
    # example_custom_context()

    print("\n✅ Examples completed!")
