"""JSON parsing utilities with robust fallback extraction."""

import json
import re
from typing import Any


def parse_json_output(text: str) -> dict:
    """Extract and parse a JSON object from LLM output.

    Attempts (in order):
    1. Direct JSON parse of the stripped text.
    2. Extract from ```json ... ``` fences.
    3. Extract the first {...} block.

    Returns an empty dict on complete failure.
    """
    if not text:
        return {}
    text = text.strip()

    # 1. Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. JSON fence
    m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 3. First {...} block
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    return {}
