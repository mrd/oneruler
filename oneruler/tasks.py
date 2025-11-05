"""
Task implementations for the OneRuler benchmark.

Each task generates prompts with needles embedded in contexts and evaluates responses.
"""

import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path


@dataclass
class TaskExample:
    """A single example for a benchmark task."""

    prompt: str
    context: str
    answer: Any
    metadata: Dict[str, Any]


class BaseTask(ABC):
    """Base class for all OneRuler tasks."""

    def __init__(self, language: str = "en", context_length: int = 8000):
        """
        Initialize a task.

        Args:
            language: Language code (e.g., 'en', 'zh', 'fr')
            context_length: Target context length in tokens
        """
        self.language = language
        self.context_length = context_length
        self.instructions = self._load_instructions()

    @abstractmethod
    def generate_example(self, context_text: str, **kwargs) -> TaskExample:
        """Generate a single example for this task."""
        pass

    @abstractmethod
    def evaluate(self, response: str, ground_truth: Any) -> Dict[str, Any]:
        """Evaluate a model response against ground truth."""
        pass

    @abstractmethod
    def _load_instructions(self) -> str:
        """Load task instructions for the specified language."""
        pass

    def _insert_needle(
        self,
        context: str,
        needle: str,
        position: Optional[float] = None
    ) -> Tuple[str, int]:
        """
        Insert a needle at a specified position in the context.

        Args:
            context: The background context text
            needle: The needle text to insert
            position: Relative position (0.0-1.0), random if None

        Returns:
            Tuple of (modified_context, actual_position)
        """
        if position is None:
            position = random.random()

        # Split context into sentences or paragraphs
        sentences = context.split('. ')
        insert_idx = int(len(sentences) * position)
        insert_idx = max(0, min(insert_idx, len(sentences) - 1))

        # Insert the needle
        sentences.insert(insert_idx, needle)
        modified_context = '. '.join(sentences)

        return modified_context, insert_idx

    def _extract_answer(self, response: str) -> Optional[str]:
        """Extract answer from <answer>...</answer> tags."""
        match = re.search(r'<answer>(.*?)</answer>', response, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return None


class SingleNIAH(BaseTask):
    """Single Needle-in-a-Haystack task."""

    def _load_instructions(self) -> str:
        """Load S-NIAH instructions."""
        # Simplified English template - should load from translated files
        return """Please read and memorize the text below. I will ask you about it later.
<text>
{context}
</text>

<question>
What special magic numbers associated with "{word}" are mentioned in the provided text? Please list all that apply. If no such numbers exist, please answer "none".
</question>

Please provide your answer in the following format:
<answer>List all numbers here</answer>"""

    def generate_example(
        self,
        context_text: str,
        word: Optional[str] = None,
        **kwargs
    ) -> TaskExample:
        """Generate a Single NIAH example."""
        # Generate random word and number if not provided
        if word is None:
            word = self._generate_random_word()

        number = random.randint(1000000, 9999999)
        needle = f'The special magic number for "{word}" is: {number}.'

        # Insert needle at random position
        modified_context, position = self._insert_needle(context_text, needle)

        # Create prompt
        prompt = self.instructions.format(
            context=modified_context,
            word=word
        )

        return TaskExample(
            prompt=prompt,
            context=modified_context,
            answer=str(number),
            metadata={
                "task": "s-niah",
                "word": word,
                "number": number,
                "position": position,
                "context_length": len(modified_context.split())
            }
        )

    def evaluate(self, response: str, ground_truth: Any) -> Dict[str, Any]:
        """Evaluate S-NIAH response."""
        extracted = self._extract_answer(response)

        if extracted is None:
            return {"correct": False, "error": "no_answer_tag"}

        # Check if answer is "none"
        if extracted.lower() == "none":
            return {"correct": False, "error": "wrong_none"}

        # Extract numbers from response
        numbers = re.findall(r'\d+', extracted)
        correct = str(ground_truth) in numbers

        return {
            "correct": correct,
            "extracted": extracted,
            "numbers_found": numbers,
            "ground_truth": str(ground_truth)
        }

    def _generate_random_word(self) -> str:
        """Generate a random word for the needle."""
        words = ["forest", "table", "coffee", "apple", "garden", "queen",
                 "river", "plane", "medicine", "newspaper", "paper", "island"]
        return random.choice(words)


class MultiKeyNIAH(BaseTask):
    """Multi-Key Needle-in-a-Haystack task."""

    def _load_instructions(self) -> str:
        return """Please read and memorize the text below. I will ask you about it later.
<text>
{context}
</text>

<question>
What special magic numbers associated with "{target_word}" are mentioned in the provided text? Please list all that apply. If no such numbers exist, please answer "none".
</question>

Please provide your answer in the following format:
<answer>List all numbers here</answer>"""

    def generate_example(self, context_text: str, **kwargs) -> TaskExample:
        """Generate a Multi-Key NIAH example with 4 needles, 1 target."""
        words = ["forest", "table", "coffee", "apple"]
        target_idx = random.randint(0, 3)
        target_word = words[target_idx]

        # Generate needles
        needles = []
        numbers = []
        for i, word in enumerate(words):
            number = random.randint(1000000, 9999999)
            numbers.append(number)
            needle = f'The special magic number for "{word}" is: {number}.'
            needles.append((needle, word, number))

        # Insert all needles
        modified_context = context_text
        for needle, _, _ in needles:
            modified_context, _ = self._insert_needle(modified_context, needle)

        prompt = self.instructions.format(
            context=modified_context,
            target_word=target_word
        )

        return TaskExample(
            prompt=prompt,
            context=modified_context,
            answer=numbers[target_idx],
            metadata={
                "task": "mk-niah",
                "target_word": target_word,
                "all_words": words,
                "all_numbers": numbers,
                "target_number": numbers[target_idx]
            }
        )

    def evaluate(self, response: str, ground_truth: Any) -> Dict[str, Any]:
        """Evaluate MK-NIAH response."""
        extracted = self._extract_answer(response)

        if extracted is None:
            return {"correct": False, "error": "no_answer_tag"}

        if extracted.lower() == "none":
            return {"correct": False, "error": "wrong_none"}

        numbers = re.findall(r'\d+', extracted)
        correct = str(ground_truth) in numbers and len(numbers) == 1

        return {
            "correct": correct,
            "extracted": extracted,
            "numbers_found": numbers,
            "ground_truth": str(ground_truth)
        }


class MultiValueNIAH(BaseTask):
    """Multi-Value Needle-in-a-Haystack task."""

    def _load_instructions(self) -> str:
        return """Please read and memorize the text below. I will ask you about it later.
<text>
{context}
</text>

<question>
What special magic numbers associated with "{word}" are mentioned in the provided text? Please list all that apply. If no such numbers exist, please answer "none".
</question>

Please provide your answer in the following format:
<answer>List all numbers here</answer>"""

    def generate_example(self, context_text: str, **kwargs) -> TaskExample:
        """Generate a Multi-Value NIAH example with 4 values for same key."""
        word = random.choice(["forest", "table", "coffee", "apple"])

        # Generate 4 different numbers for the same word
        numbers = [random.randint(1000000, 9999999) for _ in range(4)]

        # Insert all needles
        modified_context = context_text
        for number in numbers:
            needle = f'The special magic number for "{word}" is: {number}.'
            modified_context, _ = self._insert_needle(modified_context, needle)

        prompt = self.instructions.format(
            context=modified_context,
            word=word
        )

        return TaskExample(
            prompt=prompt,
            context=modified_context,
            answer=numbers,  # List of all numbers
            metadata={
                "task": "mv-niah",
                "word": word,
                "numbers": numbers
            }
        )

    def evaluate(self, response: str, ground_truth: List[int]) -> Dict[str, Any]:
        """Evaluate MV-NIAH response."""
        extracted = self._extract_answer(response)

        if extracted is None:
            return {"correct": False, "error": "no_answer_tag"}

        if extracted.lower() == "none":
            return {"correct": False, "error": "wrong_none"}

        numbers = re.findall(r'\d+', extracted)
        ground_truth_strs = [str(n) for n in ground_truth]

        # All values must be found
        correct = all(str(gt) in numbers for gt in ground_truth)

        return {
            "correct": correct,
            "extracted": extracted,
            "numbers_found": numbers,
            "ground_truth": ground_truth_strs,
            "recall": sum(1 for gt in ground_truth_strs if gt in numbers) / len(ground_truth)
        }


class MultiQueryNIAH(BaseTask):
    """Multi-Query Needle-in-a-Haystack task."""

    def _load_instructions(self) -> str:
        return """Please read and memorize the text below. I will ask you about it later.
<text>
{context}
</text>

<question>
What special magic numbers associated with "{word1}" and "{word2}" are mentioned in the provided text? Please list all that apply. If no such numbers exist, please answer "none".
</question>

Please provide your answer in the following format:
<answer>List all numbers here</answer>"""

    def generate_example(self, context_text: str, **kwargs) -> TaskExample:
        """Generate a Multi-Query NIAH example."""
        words = random.sample(["forest", "table", "coffee", "apple", "garden"], 2)
        numbers = [random.randint(1000000, 9999999) for _ in range(2)]

        # Insert needles
        modified_context = context_text
        for word, number in zip(words, numbers):
            needle = f'The special magic number for "{word}" is: {number}.'
            modified_context, _ = self._insert_needle(modified_context, needle)

        prompt = self.instructions.format(
            context=modified_context,
            word1=words[0],
            word2=words[1]
        )

        return TaskExample(
            prompt=prompt,
            context=modified_context,
            answer=numbers,
            metadata={
                "task": "mq-niah",
                "words": words,
                "numbers": numbers
            }
        )

    def evaluate(self, response: str, ground_truth: List[int]) -> Dict[str, Any]:
        """Evaluate MQ-NIAH response."""
        extracted = self._extract_answer(response)

        if extracted is None:
            return {"correct": False, "error": "no_answer_tag"}

        if extracted.lower() == "none":
            return {"correct": False, "error": "wrong_none"}

        numbers = re.findall(r'\d+', extracted)
        ground_truth_strs = [str(n) for n in ground_truth]

        correct = all(str(gt) in numbers for gt in ground_truth)

        return {
            "correct": correct,
            "extracted": extracted,
            "numbers_found": numbers,
            "ground_truth": ground_truth_strs
        }


class NoneNIAH(BaseTask):
    """None Needle-in-a-Haystack task - no correct answer exists."""

    def _load_instructions(self) -> str:
        return """Please read and memorize the text below. I will ask you about it later.
<text>
{context}
</text>

<question>
What special magic numbers associated with "{target_word}" are mentioned in the provided text? Please list all that apply. If no such numbers exist, please answer "none".
</question>

Please provide your answer in the following format:
<answer>List all numbers here</answer>"""

    def generate_example(self, context_text: str, **kwargs) -> TaskExample:
        """Generate a None-NIAH example where target word doesn't exist."""
        distractor_words = ["forest", "table", "coffee", "apple"]
        target_word = "mountain"  # Not in distractors

        # Insert distractor needles
        modified_context = context_text
        for word in distractor_words:
            number = random.randint(1000000, 9999999)
            needle = f'The special magic number for "{word}" is: {number}.'
            modified_context, _ = self._insert_needle(modified_context, needle)

        prompt = self.instructions.format(
            context=modified_context,
            target_word=target_word
        )

        return TaskExample(
            prompt=prompt,
            context=modified_context,
            answer="none",
            metadata={
                "task": "none-niah",
                "target_word": target_word,
                "distractor_words": distractor_words
            }
        )

    def evaluate(self, response: str, ground_truth: str) -> Dict[str, Any]:
        """Evaluate None-NIAH response."""
        extracted = self._extract_answer(response)

        if extracted is None:
            return {"correct": False, "error": "no_answer_tag"}

        # Correct answer is "none"
        correct = extracted.lower() == "none"

        return {
            "correct": correct,
            "extracted": extracted,
            "ground_truth": "none"
        }


class CWEEasy(BaseTask):
    """Common Word Extraction - Easy variant."""

    def _load_instructions(self) -> str:
        return """Below is a numbered list of words. In these words, some appear more often than others. Memorize the ones that appear most often.

<list>
{word_list}
</list>

<question>
What are the 10 most common words in the list above?
</question>

Please provide your answer in the following format:
<answer>List the words here</answer>"""

    def generate_example(self, word_vocabulary: List[str], **kwargs) -> TaskExample:
        """
        Generate a CWE-Easy example.

        Args:
            word_vocabulary: List of words to sample from
        """
        # Select 10 common words (appear 30 times each)
        common_words = random.sample(word_vocabulary, 10)

        # Select distractor words (appear 3 times each)
        num_distractors = (self.context_length // 40) - 10  # Rough estimate
        distractor_words = random.sample(
            [w for w in word_vocabulary if w not in common_words],
            num_distractors
        )

        # Build word list
        word_list = []
        word_list.extend(common_words * 30)  # Common words appear 30 times
        word_list.extend(distractor_words * 3)  # Distractors appear 3 times

        random.shuffle(word_list)

        # Create numbered list
        enumerated_list = '\n'.join(
            f"{i+1}. {word}" for i, word in enumerate(word_list)
        )

        prompt = self.instructions.format(word_list=enumerated_list)

        return TaskExample(
            prompt=prompt,
            context=enumerated_list,
            answer=common_words,
            metadata={
                "task": "cwe-easy",
                "common_words": common_words,
                "num_distractors": num_distractors,
                "total_words": len(word_list)
            }
        )

    def evaluate(self, response: str, ground_truth: List[str]) -> Dict[str, Any]:
        """Evaluate CWE response."""
        extracted = self._extract_answer(response)

        if extracted is None:
            return {"correct": False, "error": "no_answer_tag", "accuracy": 0.0}

        # Extract words from response
        response_words = [w.strip().lower() for w in re.split(r'[,\n]', extracted)]
        response_words = [w for w in response_words if w and not w.isdigit()]

        # Check how many ground truth words were found
        ground_truth_lower = [w.lower() for w in ground_truth]
        matches = sum(1 for w in response_words if w in ground_truth_lower)

        accuracy = matches / 10.0  # Out of 10 words
        correct = matches == 10

        return {
            "correct": correct,
            "accuracy": accuracy,
            "matches": matches,
            "extracted": extracted,
            "response_words": response_words[:10],
            "ground_truth": ground_truth
        }


class CWEHard(CWEEasy):
    """Common Word Extraction - Hard variant."""

    def generate_example(self, word_vocabulary: List[str], **kwargs) -> TaskExample:
        """Generate a CWE-Hard example with smaller frequency gap."""
        # Select 10 common words (appear 20 times each)
        common_words = random.sample(word_vocabulary, 10)

        # Select distractor words (appear 10 times each)
        num_distractors = (self.context_length // 30) - 10
        distractor_words = random.sample(
            [w for w in word_vocabulary if w not in common_words],
            num_distractors
        )

        # Build word list
        word_list = []
        word_list.extend(common_words * 20)  # Common words appear 20 times
        word_list.extend(distractor_words * 10)  # Distractors appear 10 times

        random.shuffle(word_list)

        # Create numbered list
        enumerated_list = '\n'.join(
            f"{i+1}. {word}" for i, word in enumerate(word_list)
        )

        prompt = self.instructions.format(word_list=enumerated_list)

        return TaskExample(
            prompt=prompt,
            context=enumerated_list,
            answer=common_words,
            metadata={
                "task": "cwe-hard",
                "common_words": common_words,
                "num_distractors": num_distractors,
                "total_words": len(word_list)
            }
        )
