"""
LLM provider implementations for the OneRuler benchmark.

Supports multiple LLM APIs with a common interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import os
import time


class LLMProvider(ABC):
    """Base class for LLM providers."""

    def __init__(self, model_name: str, **kwargs):
        """
        Initialize LLM provider.

        Args:
            model_name: Name/ID of the model to use
            **kwargs: Provider-specific configuration
        """
        self.model_name = model_name
        self.config = kwargs

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a response from the LLM.

        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional provider-specific parameters

        Returns:
            Dict with 'response', 'tokens_used', and other metadata
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the provider's tokenizer."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""

    def __init__(self, model_name: str = "gpt-4-turbo", api_key: Optional[str] = None, **kwargs):
        """
        Initialize OpenAI provider.

        Args:
            model_name: OpenAI model name (e.g., 'gpt-4-turbo', 'gpt-3.5-turbo')
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            **kwargs: Additional configuration
        """
        super().__init__(model_name, **kwargs)

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. Install with: pip install openai"
            )

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")

        self.client = OpenAI(api_key=self.api_key)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.0,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate response using OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )

            return {
                "response": response.choices[0].message.content,
                "tokens_used": {
                    "prompt": response.usage.prompt_tokens,
                    "completion": response.usage.completion_tokens,
                    "total": response.usage.total_tokens
                },
                "model": response.model,
                "finish_reason": response.choices[0].finish_reason
            }

        except Exception as e:
            return {
                "response": "",
                "error": str(e),
                "tokens_used": {"prompt": 0, "completion": 0, "total": 0}
            }

    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken."""
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model(self.model_name)
            return len(encoding.encode(text))
        except:
            # Rough estimate if tiktoken not available
            return len(text.split()) * 1.3


class AnthropicProvider(LLMProvider):
    """Anthropic (Claude) API provider."""

    def __init__(
        self,
        model_name: str = "claude-3-5-sonnet-20241022",
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Anthropic provider.

        Args:
            model_name: Claude model name
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
            **kwargs: Additional configuration
        """
        super().__init__(model_name, **kwargs)

        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError(
                "Anthropic package not installed. Install with: pip install anthropic"
            )

        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key not provided")

        self.client = Anthropic(api_key=self.api_key)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.0,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate response using Anthropic API."""
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )

            return {
                "response": response.content[0].text,
                "tokens_used": {
                    "prompt": response.usage.input_tokens,
                    "completion": response.usage.output_tokens,
                    "total": response.usage.input_tokens + response.usage.output_tokens
                },
                "model": response.model,
                "stop_reason": response.stop_reason
            }

        except Exception as e:
            return {
                "response": "",
                "error": str(e),
                "tokens_used": {"prompt": 0, "completion": 0, "total": 0}
            }

    def count_tokens(self, text: str) -> int:
        """Count tokens using Anthropic's token counter."""
        try:
            return self.client.count_tokens(text)
        except:
            # Rough estimate
            return len(text.split()) * 1.3


class GoogleProvider(LLMProvider):
    """Google (Gemini) API provider."""

    def __init__(
        self,
        model_name: str = "gemini-1.5-flash",
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Google provider.

        Args:
            model_name: Gemini model name
            api_key: Google API key (or set GOOGLE_API_KEY env var)
            **kwargs: Additional configuration
        """
        super().__init__(model_name, **kwargs)

        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "Google Generative AI package not installed. "
                "Install with: pip install google-generativeai"
            )

        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key not provided")

        genai.configure(api_key=self.api_key)
        self.genai = genai
        self.model = genai.GenerativeModel(self.model_name)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.0,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate response using Google Gemini API."""
        try:
            generation_config = self.genai.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )

            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )

            return {
                "response": response.text,
                "tokens_used": {
                    "prompt": response.usage_metadata.prompt_token_count,
                    "completion": response.usage_metadata.candidates_token_count,
                    "total": response.usage_metadata.total_token_count
                },
                "model": self.model_name,
                "finish_reason": str(response.candidates[0].finish_reason)
            }

        except Exception as e:
            return {
                "response": "",
                "error": str(e),
                "tokens_used": {"prompt": 0, "completion": 0, "total": 0}
            }

    def count_tokens(self, text: str) -> int:
        """Count tokens using Google's tokenizer."""
        try:
            return self.model.count_tokens(text).total_tokens
        except:
            # Rough estimate
            return len(text.split()) * 1.3


class QwenProvider(LLMProvider):
    """Qwen model provider (via API or local)."""

    def __init__(
        self,
        model_name: str = "qwen2.5-72b-instruct",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Qwen provider.

        Args:
            model_name: Qwen model name
            api_key: API key if using hosted service
            base_url: Base URL for API endpoint
            **kwargs: Additional configuration
        """
        super().__init__(model_name, **kwargs)

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI package required for Qwen API. Install with: pip install openai"
            )

        # Qwen models can be accessed via OpenAI-compatible API
        self.api_key = api_key or os.getenv("QWEN_API_KEY", "dummy")
        self.base_url = base_url or os.getenv("QWEN_BASE_URL", "http://localhost:8000/v1")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.0,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate response using Qwen model."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )

            return {
                "response": response.choices[0].message.content,
                "tokens_used": {
                    "prompt": response.usage.prompt_tokens,
                    "completion": response.usage.completion_tokens,
                    "total": response.usage.total_tokens
                },
                "model": response.model,
                "finish_reason": response.choices[0].finish_reason
            }

        except Exception as e:
            return {
                "response": "",
                "error": str(e),
                "tokens_used": {"prompt": 0, "completion": 0, "total": 0}
            }

    def count_tokens(self, text: str) -> int:
        """Rough token count estimate."""
        return len(text.split()) * 1.3


def get_provider(provider_name: str, model_name: str, **kwargs) -> LLMProvider:
    """
    Factory function to get an LLM provider.

    Args:
        provider_name: Name of the provider ('openai', 'anthropic', 'google', 'qwen')
        model_name: Model name/ID
        **kwargs: Provider-specific configuration

    Returns:
        Initialized LLMProvider instance
    """
    providers = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "google": GoogleProvider,
        "gemini": GoogleProvider,
        "qwen": QwenProvider,
    }

    provider_class = providers.get(provider_name.lower())
    if provider_class is None:
        raise ValueError(
            f"Unknown provider: {provider_name}. "
            f"Available: {list(providers.keys())}"
        )

    return provider_class(model_name=model_name, **kwargs)
