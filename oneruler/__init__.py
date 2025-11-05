"""
OneRuler: A multilingual benchmark for evaluating long-context language models.

This package implements the OneRuler benchmark described in:
"One ruler to measure them all: Benchmarking multilingual long-context language models"
Kim et al., COLM 2025
"""

__version__ = "0.1.0"
__author__ = "OneRuler Benchmark Team"

from .tasks import (
    SingleNIAH,
    MultiKeyNIAH,
    MultiValueNIAH,
    MultiQueryNIAH,
    NoneNIAH,
    CWEEasy,
    CWEHard,
)
from .benchmark import OneRulerBenchmark
from .providers import LLMProvider, OpenAIProvider, AnthropicProvider, GoogleProvider

__all__ = [
    "SingleNIAH",
    "MultiKeyNIAH",
    "MultiValueNIAH",
    "MultiQueryNIAH",
    "NoneNIAH",
    "CWEEasy",
    "CWEHard",
    "OneRulerBenchmark",
    "LLMProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "GoogleProvider",
]
