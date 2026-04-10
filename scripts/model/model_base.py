import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dotenv import load_dotenv

from openai import OpenAI
import google.generativeai as genai
from anthropic import Anthropic

load_dotenv()

# --- Pricing Configuration (USD per 1M tokens) ---
PRICING_DATA = {
    "OpenAI": {
        "gpt-5": {"input": 1.25, "output": 10.00},
        "gpt-5-mini": {"input": 0.25, "output": 2.00},
        "gpt-4.1": {"input": 2.00, "output": 8.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "o3": {"input": 2.00, "output": 8.00}
    },
    "Google": {
        "gemini-2.5-pro-preview-03-25": {"input_std": 1.25, "input_long": 2.50, "output_std": 10.00, "output_long": 15.00},
        "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
        "gemini-2.0-flash-lite": {"input": 0.075, "output": 0.30},
        "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
        "gemini-1.5-pro": {"input_std": 1.25, "input_long": 2.50, "output_std": 5.00, "output_long": 10.00},
        "gemini/gemini-3.1-flash-lite-preview": {"input": 0.075, "output": 0.30},
    },
    "Anthropic": {
        "claude-4.6-opus": {"input": 5.00, "output": 25.00},
        "claude-4.6-sonnet": {"input": 3.00, "output": 15.00},
        "claude-4.5-haiku": {"input": 1.00, "output": 5.00}
    }
}

# --- Provider Implementation Strategies ---

class LLMProvider(ABC):
    """Abstract interface for LLM backends."""
    @abstractmethod
    def generate(self, model_name: str, prompt: str, **kwargs) -> Dict[str, Any]:
        """Returns a dict with 'output' and 'cost'."""
        pass

class OpenAIProvider(LLMProvider):
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate(self, model_name: str, prompt: str, **kwargs) -> Dict[str, Any]:
        response = self.client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        usage = response.usage
        prices = PRICING_DATA["OpenAI"][model_name]
        
        # Calculate cost: (tokens / 1,000,000) * price_per_1M
        cost = (usage.prompt_tokens / 1e6 * prices["input"]) + \
               (usage.completion_tokens / 1e6 * prices["output"])
               
        return {"output": response.choices[0].message.content, "cost": cost}

class GeminiProvider(LLMProvider):
    def __init__(self):
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    def generate(self, model_name: str, prompt: str, **kwargs) -> Dict[str, Any]:
        # Strip "gemini/" prefix if present — the API expects the bare model name
        api_model_name = model_name.removeprefix("gemini/")
        model = genai.GenerativeModel(api_model_name)
        response = model.generate_content(prompt)
        usage = response.usage_metadata
        prices = PRICING_DATA["Google"][model_name]

        # Special tiered logic for Pro models with long-context pricing
        if "input_std" in prices:
            threshold = 200_000
            input_rate = prices["input_long"] if usage.prompt_token_count > threshold else prices["input_std"]
            output_rate = prices["output_long"] if usage.candidates_token_count > threshold else prices["output_std"]
            cost = (usage.prompt_token_count / 1e6 * input_rate) + \
                   (usage.candidates_token_count / 1e6 * output_rate)
        else:
            cost = (usage.prompt_token_count / 1e6 * prices["input"]) + \
                   (usage.candidates_token_count / 1e6 * prices["output"])

        return {"output": response.text, "cost": cost}

class AnthropicProvider(LLMProvider):
    def __init__(self):
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def generate(self, model_name: str, prompt: str, **kwargs) -> Dict[str, Any]:
        max_tokens = kwargs.pop("max_tokens", 1024)
        response = self.client.messages.create(
            model=model_name,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        usage = response.usage
        prices = PRICING_DATA["Anthropic"][model_name]
        
        cost = (usage.input_tokens / 1e6 * prices["input"]) + \
               (usage.output_tokens / 1e6 * prices["output"])

        return {"output": response.content[0].text, "cost": cost}

# --- Main Class ---

class LLM:
    """
    Unified LLM interface with strict model validation and cost tracking.
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.provider = self._select_provider(model_name)

    def _select_provider(self, model_name: str) -> LLMProvider:
        """
        Validates model name and selects the correct provider.
        """
        if model_name in PRICING_DATA["OpenAI"]:
            return OpenAIProvider()
        elif model_name in PRICING_DATA["Google"]:
            return GeminiProvider()
        elif model_name in PRICING_DATA["Anthropic"]:
            return AnthropicProvider()
        else:
            # List valid models for the user if they provide an invalid one
            valid_models = [m for p in PRICING_DATA.values() for m in p.keys()]
            raise ValueError(f"Model '{model_name}' not supported. Valid models: {valid_models}")

    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generates text and returns a dictionary with 'output' and 'cost'.
        """
        try:
            return self.provider.generate(self.model_name, prompt, **kwargs)
        except Exception as e:
            return {
                "output": f"Error using {self.model_name}: {str(e)}",
                "cost": 0.0
            }