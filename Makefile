.PHONY: help install install-dev install-all verify test clean format lint run-example

help:
	@echo "OneRuler Benchmark - Available Commands"
	@echo ""
	@echo "  make install         Install package with basic dependencies"
	@echo "  make install-dev     Install with development dependencies"
	@echo "  make install-all     Install with all LLM provider dependencies"
	@echo "  make verify          Verify installation and setup"
	@echo "  make test            Run tests (when implemented)"
	@echo "  make format          Format code with black"
	@echo "  make lint            Lint code with flake8"
	@echo "  make clean           Remove build artifacts"
	@echo "  make run-example     Run basic usage example"
	@echo ""

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-all:
	pip install -e ".[all]"

verify:
	python verify_setup.py

test:
	@echo "Tests not yet implemented"
	# pytest tests/

format:
	black oneruler/ examples/

lint:
	flake8 oneruler/ examples/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

run-example:
	python examples/basic_usage.py
