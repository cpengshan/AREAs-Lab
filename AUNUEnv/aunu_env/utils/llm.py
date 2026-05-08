"""
LLM utility for AUNUEnv.

Supports OpenAI, Google Gemini, and Anthropic models using the same provider
pattern as scripts/model/model_base.py. Model selection is automatic based on
the model name.

Supported models are listed in PRICING_DATA. Add new entries there to support
additional models without changing any other code.
"""

import os
import time
import logging
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pricing (USD per 1M tokens)
# ---------------------------------------------------------------------------

PRICING_DATA = {
    "OpenAI": {
        "gpt-5":          {"input": 1.25,  "output": 10.00},
        "gpt-5-mini":     {"input": 0.25,  "output":  2.00},
        "gpt-5.4-mini":   {"input": 0.75,  "output":  4.50},
        "gpt-5.4":        {"input": 2.50,  "output": 15.00},
        "gpt-4.1":        {"input": 2.00,  "output":  8.00},
        "gpt-4.1-mini":   {"input": 0.40,  "output":  1.60},
        "o3":             {"input": 2.00,  "output":  8.00},
    },
    "Google": {
        "gemini-2.5-pro-preview-03-25": {"input_std": 1.25, "input_long": 2.50, "output_std": 10.00, "output_long": 15.00},
        "gemini-2.0-flash":             {"input": 0.10,  "output": 0.40},
        "gemini-2.0-flash-lite":        {"input": 0.075, "output": 0.30},
        "gemini-1.5-flash":             {"input": 0.075, "output": 0.30},
        "gemini-1.5-pro":               {"input_std": 1.25, "input_long": 2.50, "output_std": 5.00, "output_long": 10.00},
        "gemini-3.1-pro-preview":       {"input_std": 2.00, "input_long": 4.00, "output_std": 12.00, "output_long": 18.00},
        "gemini-3.1-flash-lite-preview":{"input": 0.075, "output": 0.30},
    },
    "Anthropic": {
        "claude-opus-4-7":           {"input":  5.00, "output": 25.00},
        "claude-4.6-opus":           {"input":  5.00, "output": 25.00},
        "claude-sonnet-4-6":         {"input":  3.00, "output": 15.00},
        "claude-haiku-4-5-20251001": {"input":  1.00, "output":  5.00},
        "claude-4.5-haiku":          {"input":  1.00, "output":  5.00},
    },
}

# OpenAI models that require max_completion_tokens instead of max_tokens
_MAX_COMPLETION_TOKENS_MODELS = {"gpt-5.4", "gpt-5.4-mini", "gpt-5", "gpt-5-mini", "o3"}

# OpenAI models that support the reasoning parameter
_REASONING_MODELS = {"gpt-5.4", "gpt-5.4-mini"}


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------

def _openai_generate(model_name: str, prompt: str, **kwargs) -> dict:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    reasoning_effort = kwargs.pop("reasoning_effort", None)

    # Reasoning models use the Responses API (client.responses.create) with
    # input= and reasoning=, not the Chat Completions API.
    if model_name in _REASONING_MODELS:
        kwargs.pop("max_tokens", None)
        kwargs.pop("max_completion_tokens", None)
        kwargs.pop("temperature", None)
        response = client.responses.create(
            model=model_name,
            reasoning={"effort": reasoning_effort or "low"},
            input=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        usage = response.usage
        try:
            prices = PRICING_DATA["OpenAI"][model_name]
            cost = (usage.input_tokens / 1e6 * prices["input"]) + \
                   (usage.output_tokens / 1e6 * prices["output"])
        except KeyError:
            logger.warning(f"No pricing data for OpenAI model '{model_name}', cost set to 0.0")
            cost = 0.0
        return {
            "output": response.output_text,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cost": cost,
        }

    if model_name in _MAX_COMPLETION_TOKENS_MODELS and "max_tokens" in kwargs:
        kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        **kwargs,
    )
    usage = response.usage
    try:
        prices = PRICING_DATA["OpenAI"][model_name]
        cost = (usage.prompt_tokens / 1e6 * prices["input"]) + \
               (usage.completion_tokens / 1e6 * prices["output"])
    except KeyError:
        logger.warning(f"No pricing data for OpenAI model '{model_name}', cost set to 0.0")
        cost = 0.0
    return {
        "output": response.choices[0].message.content,
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
        "cost": cost,
    }


def _gemini_generate(model_name: str, prompt: str, **kwargs) -> dict:
    from google import genai
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    bare_name = model_name.removeprefix("gemini/")
    response = client.models.generate_content(model=bare_name, contents=prompt)

    usage = response.usage_metadata
    input_tokens = usage.prompt_token_count
    output_tokens = usage.candidates_token_count

    try:
        prices = PRICING_DATA["Google"][model_name]
        if "input_long" in prices:
            is_long = input_tokens > 200_000
            input_rate  = prices["input_long"]  if is_long else prices["input_std"]
            output_rate = prices["output_long"] if is_long else prices["output_std"]
            cost = (input_tokens / 1e6 * input_rate) + (output_tokens / 1e6 * output_rate)
        else:
            cost = (input_tokens / 1e6 * prices["input"]) + (output_tokens / 1e6 * prices["output"])
    except KeyError:
        logger.warning(f"No pricing data for Gemini model '{model_name}', cost set to 0.0")
        cost = 0.0

    return {
        "output": response.text,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost": cost,
    }


def _anthropic_generate(model_name: str, prompt: str, **kwargs) -> dict:
    from anthropic import Anthropic
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    max_tokens = kwargs.pop("max_tokens", 1024)
    kwargs.pop("reasoning_effort", None)
    response = client.messages.create(
        model=model_name,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
        **kwargs,
    )
    usage = response.usage
    try:
        prices = PRICING_DATA["Anthropic"][model_name]
        cost = (usage.input_tokens / 1e6 * prices["input"]) + \
               (usage.output_tokens / 1e6 * prices["output"])
    except KeyError:
        logger.warning(f"No pricing data for Anthropic model '{model_name}', cost set to 0.0")
        cost = 0.0
    return {
        "output": response.content[0].text,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cost": cost,
    }


# ---------------------------------------------------------------------------
# Provider selection
# ---------------------------------------------------------------------------

def _select_provider(model_name: str):
    """Return the generate function for the given model name.

    Checks PRICING_DATA first; falls back to name-prefix matching so models
    not yet listed in PRICING_DATA still work (cost will be reported as 0.0).
    """
    if model_name in PRICING_DATA["OpenAI"]:
        return _openai_generate
    if model_name in PRICING_DATA["Google"]:
        return _gemini_generate
    if model_name in PRICING_DATA["Anthropic"]:
        return _anthropic_generate

    # Fallback: infer provider from name prefix
    name = model_name.lower()
    if name.startswith(("gpt-", "o1", "o3", "o4")):
        logger.warning(f"Model '{model_name}' not in PRICING_DATA; routing to OpenAI (cost=0.0)")
        return _openai_generate
    if name.startswith("gemini"):
        logger.warning(f"Model '{model_name}' not in PRICING_DATA; routing to Google (cost=0.0)")
        return _gemini_generate
    if name.startswith("claude"):
        logger.warning(f"Model '{model_name}' not in PRICING_DATA; routing to Anthropic (cost=0.0)")
        return _anthropic_generate

    valid = [m for provider in PRICING_DATA.values() for m in provider]
    raise ValueError(
        f"Model '{model_name}' is not supported. "
        f"Add it to PRICING_DATA in aunu_env/utils/llm.py, or choose from: {valid}"
    )


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def call_llm(
    model_name: str,
    prompt: str,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    reasoning_effort: str | None = None,
    retries: int = 3,
    retry_delay: float = 2.0,
) -> dict:
    """Call an LLM and return a standardised result dict.

    Automatically selects the provider (OpenAI / Gemini / Anthropic) based on
    the model name. Retries up to `retries` times on failure.

    Args:
        model_name: Model identifier as listed in PRICING_DATA.
        prompt: The full prompt string.
        max_tokens: Maximum output tokens.
        temperature: Sampling temperature.
        reasoning_effort: Reasoning effort for supported OpenAI models
            (e.g. "low", "medium", "high"). Defaults to "low" for GPT-5.4
            models when omitted.
        retries: Number of retry attempts on transient failures.
        retry_delay: Seconds between retries.

    Returns:
        Dict with keys:
            output (str): Generated text.
            input_tokens (int): Prompt token count.
            output_tokens (int): Completion token count.
            cost (float): Estimated USD cost.

    Raises:
        ValueError: If the model name is not in PRICING_DATA.
        RuntimeError: If all retry attempts fail.
    """
    generate = _select_provider(model_name)
    last_error = None

    for attempt in range(retries):
        try:
            result = generate(
                model_name,
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
            )
            return result
        except Exception as e:
            last_error = e
            logger.warning(f"LLM call failed (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(retry_delay)

    raise RuntimeError(f"LLM call failed after {retries} attempts: {last_error}")
