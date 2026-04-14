# AUNU: AI-Assisted User Needs Understanding

AUNU is a specialized framework designed to leverage Large Language Models (LLMs) for deep user needs analysis and processing.

---

## 🛠 Setup & Environment Variables

This framework requires API access to various LLM providers. Before running experiments, create a `.env` file in your root directory and configure your keys as follows:

| Provider | Environment Variable | Key Management Link |
| :--- | :--- | :--- |
| **OpenAI** | `OPENAI_API_KEY` | [OpenAI API Keys](https://platform.openai.com/api-keys) |
| **Google** | `GOOGLE_API_KEY` | [Google AI Studio](https://aistudio.google.com/api-keys) |
| **Anthropic** | `ANTHROPIC_API_KEY` | [Anthropic Console](https://console.anthropic.com/settings/keys) |

> **Note:** The underlying logic for these API calls is managed by `scripts/model/model_base.py`.

---

## 🚀 LLM Usage Example

The `LLM` class provides a unified, stateful interface. You simply define the model name during initialization, and the appropriate provider is resolved automatically.

```python
from scripts.model.model_base import LLM

# 1. Initialize for OpenAI (GPT-4o)
gpt = LLM(model_name="gpt-4o")
print(gpt.generate("Write a haiku about Python."))

# 2. Initialize for Anthropic 
# Note: Use valid model IDs like 'claude-opus-4-6'
claude = LLM("claude-opus-4-6")
print(claude.generate("Who is the first president in the United States?", temperature=0.5))

# 3. Initialize for Google Gemini
gemini = LLM(model_name="gemini-3-flash-preview")
print(gemini.generate("What is the Cappital of France", use_cache=True))

# You can also use generate results in a batch
gpt = LLM(model_name="gpt-4o")
prompts = [
    "What are the two most popular coding languages in 2026?",
    "What is the Capital of France"
]
results = gpt.generate_batch(prompts)