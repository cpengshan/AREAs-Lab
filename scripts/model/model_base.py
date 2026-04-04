import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dotenv import load_dotenv

from openai import OpenAI
import google.generativeai as genai
from anthropic import Anthropic

load_dotenv()

# --- Provider Implementation Strategies ---

class LLMProvider(ABC):
    """Abstract interface for LLM backends."""
    @abstractmethod
    def generate(self, model_name: str, prompt: str, **kwargs) -> str:
        pass

class OpenAIProvider(LLMProvider):
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate(self, model_name: str, prompt: str, **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.choices[0].message.content

class GeminiProvider(LLMProvider):
    def __init__(self):
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    def generate(self, model_name: str, prompt: str, **kwargs) -> str:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text

class AnthropicProvider(LLMProvider):
    def __init__(self):
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def generate(self, model_name: str, prompt: str, **kwargs) -> str:
        max_tokens = kwargs.pop("max_tokens", 1024)
        response = self.client.messages.create(
            model=model_name,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.content[0].text

# --- Main Class ---

class LLM:
    """
    Unified LLM interface where the model is defined at instantiation.
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.provider = self._select_provider(model_name)

    def _select_provider(self, model_name: str) -> LLMProvider:
        """
        Determines the correct API provider based on the model string.
        """
        name_lower = model_name.lower()
        
        if "gpt" in name_lower:
            return OpenAIProvider()
        elif "gemini" in name_lower:
            return GeminiProvider()
        elif "claude" in name_lower:
            return AnthropicProvider()
        else:
            raise ValueError(f"Unknown provider for model: {model_name}")

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generates text using the model specified during initialization.
        """
        try:
            return self.provider.generate(self.model_name, prompt, **kwargs)
        except Exception as e:
            return f"Error using {self.model_name}: {str(e)}"

