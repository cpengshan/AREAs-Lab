import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv

from openai import OpenAI
import google.generativeai as genai
from anthropic import Anthropic

load_dotenv()


class LLM:
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self._init_clients()

    def _init_clients(self, model_name):
        self.model_name = model_name
        # OpenAI
        if os.getenv("OPENAI_API_KEY"):
            self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # Gemini
        if os.getenv("GOOGLE_API_KEY"):
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        # Claude
        if os.getenv("ANTHROPIC_API_KEY"):
            self.claude_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def generate_gpt(self, prompt, **kwargs):
        response = self.openai_client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.choices[0].message.content
    
    def generate_gemini(self, prompt, **kwargs):
        model = genai.GenerativeModel(self.model_name)
        response = model.generate_content(prompt)
        return response.text
    
    def generate_claude(self, prompt, **kwargs):
        response = self.claude_client.messages.create(
            model=self.model_name,
            max_tokens=kwargs.get("max_tokens", 1024),
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

    def generate(self, model_name: str, prompt: str, **kwargs) -> str:
        """
        :param model_name: 'gpt-4', 'gemini-pro', 'claude-3-opus'
        :param prompt: User input
        """
        model_name = model_name.lower()
        try:
            # 1. OpenAI
            if "gpt" in model_name:
                return self.generate_gpt(prompt, **kwargs)
            # 2. Gemini 
            elif "gemini" in model_name:
                return self.generate_gemini(prompt, **kwargs)
            # 3. Claude
            elif "claude" in model_name:
                return self.generate_claude(prompt, **kwargs)
            else:
                return f"Error: Unsupported model '{model_name}'"
        except Exception as e:
            return f"Error calling {model_name}: {str(e)}"