#!/usr/bin/env python3
"""
Verify OneRuler installation and configuration.

Run this script to check if everything is set up correctly.
"""

import sys
import os


def check_python_version():
    """Check Python version."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ✗ Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)")
        return False


def check_core_dependencies():
    """Check core dependencies."""
    print("\nChecking core dependencies...")
    deps = ["tqdm", "numpy"]
    all_ok = True

    for dep in deps:
        try:
            __import__(dep)
            print(f"  ✓ {dep}")
        except ImportError:
            print(f"  ✗ {dep} (install with: pip install {dep})")
            all_ok = False

    return all_ok


def check_provider_dependencies():
    """Check LLM provider dependencies."""
    print("\nChecking LLM provider dependencies...")

    providers = {
        "OpenAI": ["openai", "tiktoken"],
        "Anthropic": ["anthropic"],
        "Google": ["google.generativeai"],
    }

    available_providers = []

    for provider_name, modules in providers.items():
        all_modules_present = True
        for module in modules:
            try:
                __import__(module)
            except ImportError:
                all_modules_present = False
                break

        if all_modules_present:
            print(f"  ✓ {provider_name}")
            available_providers.append(provider_name)
        else:
            print(f"  - {provider_name} (install with: pip install -e \".[{provider_name.lower()}]\")")

    return available_providers


def check_api_keys(available_providers):
    """Check API keys."""
    print("\nChecking API keys...")

    keys = {
        "OpenAI": "OPENAI_API_KEY",
        "Anthropic": "ANTHROPIC_API_KEY",
        "Google": "GOOGLE_API_KEY",
    }

    configured_providers = []

    for provider, env_var in keys.items():
        if provider in available_providers:
            if os.getenv(env_var):
                print(f"  ✓ {provider} API key set")
                configured_providers.append(provider)
            else:
                print(f"  ✗ {provider} API key not set (export {env_var}='your-key')")
        else:
            print(f"  - {provider} (dependencies not installed)")

    return configured_providers


def check_oneruler_installation():
    """Check if oneruler package is properly installed."""
    print("\nChecking OneRuler installation...")

    try:
        import oneruler
        print(f"  ✓ oneruler package version {oneruler.__version__}")

        # Check if components are accessible
        from oneruler import OneRulerBenchmark
        from oneruler.tasks import SingleNIAH
        from oneruler.providers import get_provider

        print("  ✓ Core components accessible")
        return True

    except ImportError as e:
        print(f"  ✗ OneRuler not installed properly: {e}")
        print("     Try: pip install -e .")
        return False


def run_quick_test(provider_name):
    """Run a very quick test."""
    print(f"\nRunning quick test with {provider_name}...")

    try:
        from oneruler import OneRulerBenchmark
        from oneruler.providers import get_provider
        from oneruler.tasks import SingleNIAH

        # Map provider names
        provider_models = {
            "OpenAI": ("openai", "gpt-3.5-turbo"),
            "Anthropic": ("anthropic", "claude-3-5-sonnet-20241022"),
            "Google": ("google", "gemini-1.5-flash"),
        }

        provider_id, model_name = provider_models[provider_name]

        # Initialize
        provider = get_provider(provider_id, model_name)
        benchmark = OneRulerBenchmark(
            provider=provider,
            output_dir="results/verify_test"
        )

        # Run one example
        print(f"  Testing {model_name}...")
        results = benchmark.run_task(
            task_class=SingleNIAH,
            context_length=4000,  # Shorter for quick test
            num_examples=1,
            save_results=False
        )

        if results and results[0].evaluation.get("correct") is not None:
            print(f"  ✓ Test completed successfully!")
            print(f"     Result: {'Correct' if results[0].evaluation['correct'] else 'Incorrect'}")
            return True
        else:
            print(f"  ✗ Test returned unexpected results")
            return False

    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        return False


def main():
    """Run all verification checks."""
    print("="*60)
    print("OneRuler Setup Verification")
    print("="*60)

    # Run checks
    python_ok = check_python_version()
    core_ok = check_core_dependencies()
    available_providers = check_provider_dependencies()
    configured_providers = check_api_keys(available_providers)
    oneruler_ok = check_oneruler_installation()

    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)

    if not (python_ok and core_ok and oneruler_ok):
        print("✗ Basic requirements not met.")
        print("\nTo fix:")
        print("  1. Ensure Python 3.8+ is installed")
        print("  2. Run: pip install -e .")
        sys.exit(1)

    if not configured_providers:
        print("⚠️  No LLM providers configured.")
        print("\nTo add a provider:")
        print("  1. Install: pip install -e \".[openai]\"")
        print("  2. Set API key: export OPENAI_API_KEY='your-key'")
        print("\nAvailable providers: openai, anthropic, google")
        sys.exit(0)

    print(f"✓ Setup complete! {len(configured_providers)} provider(s) ready:")
    for provider in configured_providers:
        print(f"  - {provider}")

    # Offer to run quick test
    print("\n" + "="*60)
    response = input("Run a quick test? (y/N): ").strip().lower()

    if response == 'y':
        # Test with first available provider
        test_provider = configured_providers[0]
        success = run_quick_test(test_provider)

        if success:
            print("\n" + "="*60)
            print("✅ Everything is working! You're ready to use OneRuler.")
            print("\nNext steps:")
            print("  - Read QUICKSTART.md for usage examples")
            print("  - Run: oneruler --help")
            print("  - Try: oneruler --provider openai --model gpt-3.5-turbo --task s-niah --num-examples 5")
        else:
            print("\n⚠️  Test failed. Check your API key and network connection.")
    else:
        print("\n✅ Setup verified! Run 'oneruler --help' to get started.")


if __name__ == "__main__":
    main()
