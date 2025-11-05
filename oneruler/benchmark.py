"""
Main benchmark orchestration for OneRuler.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import random
from tqdm import tqdm

from .tasks import (
    BaseTask, SingleNIAH, MultiKeyNIAH, MultiValueNIAH,
    MultiQueryNIAH, NoneNIAH, CWEEasy, CWEHard
)
from .providers import LLMProvider


@dataclass
class BenchmarkResult:
    """Results from running a benchmark task."""

    task_name: str
    language: str
    context_length: int
    model_name: str
    example_id: int
    prompt: str
    response: str
    ground_truth: Any
    evaluation: Dict[str, Any]
    metadata: Dict[str, Any]
    timestamp: float


class OneRulerBenchmark:
    """Main benchmark class for OneRuler."""

    def __init__(
        self,
        provider: LLMProvider,
        language: str = "en",
        output_dir: str = "results"
    ):
        """
        Initialize OneRuler benchmark.

        Args:
            provider: LLM provider to evaluate
            language: Language code to test
            output_dir: Directory to save results
        """
        self.provider = provider
        self.language = language
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Default context corpus (simple placeholder)
        self.context_corpus = self._load_context_corpus()
        self.word_vocabulary = self._load_word_vocabulary()

    def _load_context_corpus(self) -> str:
        """Load background context text for needle injection."""
        # In a real implementation, this would load language-specific books
        # For now, return a simple placeholder
        return """This is a placeholder context text that would normally be loaded from
        a book in the target language. In the full implementation, this would be replaced
        with actual literary texts from Project Gutenberg or similar sources. The context
        should be long enough to fill the desired context length when repeated. """ * 1000

    def _load_word_vocabulary(self) -> List[str]:
        """Load word vocabulary for CWE tasks."""
        # Simple English vocabulary - should be language-specific
        return [
            "forest", "table", "coffee", "apple", "garden", "queen", "river",
            "plane", "medicine", "newspaper", "paper", "island", "mountain",
            "ocean", "castle", "dragon", "wizard", "knight", "princess", "sword",
            "shield", "armor", "helmet", "boots", "gloves", "ring", "amulet",
            "spell", "potion", "scroll", "wand", "staff", "crown", "throne",
            "village", "city", "kingdom", "empire", "dungeon", "tower", "bridge",
            "road", "path", "trail", "forest", "jungle", "desert", "tundra",
            "volcano", "cave", "mine", "quarry", "farm", "field", "meadow",
            "pond", "lake", "stream", "waterfall", "cliff", "valley", "hill"
        ] * 100  # Expand vocabulary

    def run_task(
        self,
        task_class: type[BaseTask],
        context_length: int = 8000,
        num_examples: int = 50,
        max_tokens: int = 1000,
        save_results: bool = True
    ) -> List[BenchmarkResult]:
        """
        Run a specific task.

        Args:
            task_class: Task class to instantiate and run
            context_length: Target context length in tokens
            num_examples: Number of examples to generate
            max_tokens: Maximum tokens for model response
            save_results: Whether to save results to disk

        Returns:
            List of BenchmarkResult objects
        """
        print(f"\n{'='*60}")
        print(f"Running {task_class.__name__} - {self.language} - {context_length} tokens")
        print(f"{'='*60}\n")

        # Initialize task
        task = task_class(language=self.language, context_length=context_length)

        results = []

        # Generate and evaluate examples
        for i in tqdm(range(num_examples), desc=f"{task_class.__name__}"):
            # Generate example
            if "CWE" in task_class.__name__:
                example = task.generate_example(word_vocabulary=self.word_vocabulary)
            else:
                # Adjust context to target length
                context = self._adjust_context_length(
                    self.context_corpus,
                    context_length
                )
                example = task.generate_example(context_text=context)

            # Get model response
            start_time = time.time()
            response_data = self.provider.generate(
                prompt=example.prompt,
                max_tokens=max_tokens,
                temperature=0.0
            )
            elapsed_time = time.time() - start_time

            response = response_data.get("response", "")

            # Evaluate response
            evaluation = task.evaluate(response, example.answer)
            evaluation["response_time"] = elapsed_time
            evaluation["tokens_used"] = response_data.get("tokens_used", {})

            # Create result
            result = BenchmarkResult(
                task_name=task_class.__name__,
                language=self.language,
                context_length=context_length,
                model_name=self.provider.model_name,
                example_id=i,
                prompt=example.prompt[:500] + "..." if len(example.prompt) > 500 else example.prompt,
                response=response,
                ground_truth=example.answer,
                evaluation=evaluation,
                metadata=example.metadata,
                timestamp=time.time()
            )

            results.append(result)

            # Optional: sleep to avoid rate limits
            time.sleep(0.1)

        # Save results
        if save_results:
            self._save_results(results, task_class.__name__, context_length)

        # Print summary
        self._print_summary(results, task_class.__name__)

        return results

    def run_all_tasks(
        self,
        context_lengths: List[int] = [8000, 32000, 64000, 128000],
        num_examples: int = 50
    ) -> Dict[str, List[BenchmarkResult]]:
        """
        Run all OneRuler tasks.

        Args:
            context_lengths: List of context lengths to test
            num_examples: Number of examples per task/length

        Returns:
            Dictionary mapping task names to results
        """
        all_results = {}

        task_classes = [
            SingleNIAH,
            MultiKeyNIAH,
            MultiValueNIAH,
            MultiQueryNIAH,
            NoneNIAH,
            CWEEasy,
            CWEHard
        ]

        for task_class in task_classes:
            for context_length in context_lengths:
                key = f"{task_class.__name__}_{context_length}"
                results = self.run_task(
                    task_class=task_class,
                    context_length=context_length,
                    num_examples=num_examples
                )
                all_results[key] = results

        # Save aggregate results
        self._save_aggregate_results(all_results)

        return all_results

    def _adjust_context_length(self, text: str, target_length: int) -> str:
        """Adjust context text to approximately match target token length."""
        # Rough estimate: 1 token ≈ 0.75 words
        target_words = int(target_length * 0.75)
        words = text.split()

        if len(words) < target_words:
            # Repeat text to reach target length
            repetitions = (target_words // len(words)) + 1
            words = (words * repetitions)[:target_words]
        else:
            words = words[:target_words]

        return ' '.join(words)

    def _save_results(
        self,
        results: List[BenchmarkResult],
        task_name: str,
        context_length: int
    ):
        """Save results to JSON file."""
        filename = (
            f"{task_name}_{self.language}_{context_length}_"
            f"{self.provider.model_name.replace('/', '_')}.json"
        )
        filepath = self.output_dir / filename

        # Convert results to dictionaries
        results_dict = [asdict(r) for r in results]

        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)

        print(f"\nResults saved to: {filepath}")

    def _save_aggregate_results(self, all_results: Dict[str, List[BenchmarkResult]]):
        """Save aggregate results across all tasks."""
        filename = f"oneruler_results_{self.language}_{self.provider.model_name.replace('/', '_')}.json"
        filepath = self.output_dir / filename

        # Compute aggregate statistics
        aggregate = {
            "model": self.provider.model_name,
            "language": self.language,
            "timestamp": time.time(),
            "summary": {},
            "detailed_results": {}
        }

        for key, results in all_results.items():
            if not results:
                continue

            correct = sum(1 for r in results if r.evaluation.get("correct", False))
            total = len(results)
            accuracy = correct / total if total > 0 else 0.0

            aggregate["summary"][key] = {
                "accuracy": accuracy,
                "correct": correct,
                "total": total
            }

            aggregate["detailed_results"][key] = [asdict(r) for r in results]

        with open(filepath, 'w') as f:
            json.dump(aggregate, f, indent=2)

        print(f"\n{'='*60}")
        print(f"Aggregate results saved to: {filepath}")
        print(f"{'='*60}")

    def _print_summary(self, results: List[BenchmarkResult], task_name: str):
        """Print summary statistics for a task."""
        if not results:
            return

        correct = sum(1 for r in results if r.evaluation.get("correct", False))
        total = len(results)
        accuracy = correct / total if total > 0 else 0.0

        print(f"\n{'='*60}")
        print(f"Summary for {task_name}")
        print(f"{'='*60}")
        print(f"Accuracy: {accuracy:.2%} ({correct}/{total})")

        # Task-specific metrics
        if "CWE" in task_name:
            avg_matches = sum(
                r.evaluation.get("matches", 0) for r in results
            ) / total
            print(f"Average matches: {avg_matches:.2f} / 10")

        print(f"{'='*60}\n")
