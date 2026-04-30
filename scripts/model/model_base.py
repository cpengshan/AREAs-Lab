import os
import concurrent.futures
from functools import partial
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

from openai import OpenAI
from google import genai

from anthropic import Anthropic
from diskcache import Cache

load_dotenv()

# --- Pricing Configuration (USD per 1M tokens) ---
PRICING_DATA = {
    "OpenAI": {
        "gpt-5": {"input": 1.25, "output": 10.00},
        "gpt-5-mini": {"input": 0.25, "output": 2.00},
        "gpt-4.1": {"input": 2.00, "output": 8.00},
        "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
        "o3": {"input": 2.00, "output": 8.00}
    },
    "Google": {
        # Use the -preview suffix for 3.1 models in 2026
        "gemini-3.1-pro-preview": {"input_std": 2.00, "input_long": 4.00, "output_std": 12.00, "output_long": 18.00},
        "gemini-3-flash-preview": {"input": 0.50, "output": 3.00}, 
        "gemini-3.1-flash-lite-preview": {"input": 0.25, "output": 1.50}
    },
    "Anthropic": {
        "claude-4.6-opus": {"input": 5.00, "output": 25.00},
        "claude-4.6-sonnet": {"input": 3.00, "output": 15.00},
        "claude-4.5-haiku": {"input": 0.80, "output": 4.00}
    },
    "DeepSeek": {
        "deepseek-v4-pro": {"input": 1.74, "output": 3.48},
        "deepseek-v4-flash": {"input": 0.14, "output": 0.28}
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
        
        cost = (usage.prompt_tokens / 1e6 * prices["input"]) + \
               (usage.completion_tokens / 1e6 * prices["output"])
               
        return {"output": response.choices[0].message.content, "cost": cost}

class DeepSeekProvider(LLMProvider):
    def __init__(self):
        # DeepSeek uses an OpenAI-compatible API
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )

    def generate(self, model_name: str, prompt: str, **kwargs) -> Dict[str, Any]:
        response = self.client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        usage = response.usage
        prices = PRICING_DATA["DeepSeek"][model_name]
        
        # Note: If DeepSeek v4 implements prompt caching discounts, 
        # you can add 'usage.prompt_cache_hit_tokens' logic here.
        cost = (usage.prompt_tokens / 1e6 * prices["input"]) + \
               (usage.completion_tokens / 1e6 * prices["output"])
               
        return {"output": response.choices[0].message.content, "cost": cost}

class GeminiProvider(LLMProvider):
    def __init__(self):
        # The new SDK uses a Client object
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    def generate(self, model_name: str, prompt: str, **kwargs) -> Dict[str, Any]:
        # Generate content using the new client syntax
        response = self.client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        
        usage = response.usage_metadata
        
        # KEY FIX: In the new SDK, attributes use the '_count' suffix
        # and 'candidates' instead of 'completion'
        input_tokens = usage.prompt_token_count
        output_tokens = usage.candidates_token_count
        
        # Fetch pricing from your global PRICING_DATA
        if model_name not in PRICING_DATA["Google"]:
            raise ValueError(f"Model {model_name} not found in PRICING_DATA")
            
        prices = PRICING_DATA["Google"][model_name]
        
        # Logic for Tiered Pricing (Gemini 3.1 Pro 200k threshold)
        if "input_long" in prices:
            threshold = 200_000
            # Note: Threshold usually applies based on TOTAL context (input)
            is_long = input_tokens > threshold
            
            input_rate = prices["input_long"] if is_long else prices["input_std"]
            output_rate = prices["output_long"] if is_long else prices["output_std"]
            
            cost = (input_tokens / 1e6 * input_rate) + \
                   (output_tokens / 1e6 * output_rate)
        else:
            # Standard calculation for Flash/Lite models
            cost = (input_tokens / 1e6 * prices["input"]) + \
                   (output_tokens / 1e6 * prices["output"])

        return {
            "output": response.text, 
            "cost": cost
        }
     
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
    Unified LLM interface with strict model validation, cost tracking, caching, 
    and concurrent batch processing.
    """
    def __init__(self, model_name: str, cache_dir: str = ".llm_cache"):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.cache = Cache(self.cache_dir)
        self.provider = self._select_provider(model_name)

    def _select_provider(self, model_name: str) -> LLMProvider:
        if model_name in PRICING_DATA["OpenAI"]:
            return OpenAIProvider()
        elif model_name in PRICING_DATA["Google"]:
            return GeminiProvider()
        elif model_name in PRICING_DATA["Anthropic"]:
            return AnthropicProvider()
        elif model_name in PRICING_DATA["DeepSeek"]:
            return DeepSeekProvider()
        else:
            valid_models = [m for p in PRICING_DATA.values() for m in p.keys()]
            raise ValueError(f"Model '{model_name}' not supported. Valid models: {valid_models}")



    def generate(self, prompt: str, use_cache: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Generates text and returns a dictionary with 'output' and 'cost'.
        
        Args:
            prompt: The string to send to the LLM.
            use_cache: If True (default), returns cached results if available. 
                       If False, forces a fresh API call.
            **kwargs: Additional provider arguments (temperature, max_tokens, etc.)
        """
        # Create a unique key for the cache based on model, prompt, and settings
        cache_key = (self.model_name, prompt, str(sorted(kwargs.items())))

        # 1. Check cache ONLY if use_cache is True
        if use_cache and cache_key in self.cache:
            return self.cache[cache_key]
        try:
            result = self.provider.generate(self.model_name, prompt, **kwargs)
            # 2. Store in cache ONLY if use_cache is True
            if use_cache:
                self.cache[cache_key] = result
            return result
        except Exception as e:
            return {
                "output": f"Error using {self.model_name}: {str(e)}",
                "cost": 0.0
            }

    def generate_batch(self, prompts: List[str], use_cache: bool = False, max_workers: int = 6, **kwargs) -> List[Dict[str, Any]]:
        """
        Generates text for a batch of prompts concurrently.
        """
        # Pass use_cache into the partial function so all batch items respect the flag
        func = partial(self.generate, use_cache=use_cache, **kwargs)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(func, prompts))
        return results