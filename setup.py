"""Setup script for OneRuler benchmark."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = ""
if readme_file.exists():
    long_description = readme_file.read_text(encoding="utf-8")

setup(
    name="oneruler",
    version="0.1.0",
    description="OneRuler: A multilingual benchmark for evaluating long-context language models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="OneRuler Benchmark Team",
    author_email="",
    url="https://github.com/mungg/OneRuler",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "tqdm>=4.65.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "openai": ["openai>=1.12.0", "tiktoken>=0.5.0"],
        "anthropic": ["anthropic>=0.18.0"],
        "google": ["google-generativeai>=0.3.0"],
        "all": [
            "openai>=1.12.0",
            "tiktoken>=0.5.0",
            "anthropic>=0.18.0",
            "google-generativeai>=0.3.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "oneruler=oneruler.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="nlp benchmark long-context multilingual llm evaluation",
)
