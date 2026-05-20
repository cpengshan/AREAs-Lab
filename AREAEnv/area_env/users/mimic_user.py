"""
MimicUser — uses feedback_mimic_user_v3.jinja + responser_habit.json.

The habit dict (passive/neutral/active) is loaded externally and injected
via the constructor — this module does not read responser_habit.json itself.
"""

import logging
import os

from ..dataset.schema import TaskInstance
from ..utils.llm import call_llm
from ..utils.jinja_utils import render_template
from ..utils.json_utils import parse_json_output

logger = logging.getLogger(__name__)

_PROMPT_DIR = os.path.join(os.path.dirname(__file__), "prompts")
_V2_TEMPLATE = os.path.join(_PROMPT_DIR, "feedback_mimic_user_v3.jinja")


class MimicUser:
    """Simulated user driven by a pre-loaded communication habit dict.

    Args:
        model_name: LLM model to use.
        habit: The communication habit dict (one entry from responser_habit.json).
        temperature: Sampling temperature.
        max_tokens: Max output tokens.
    """

    def __init__(
        self,
        model_name: str,
        habit: dict,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ):
        self.model_name = model_name
        self.habit = habit
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.persona_config = {"mode": "v2"}

    def respond(
        self,
        task: TaskInstance,
        chat_history: list,
        agent_message: str,
    ) -> dict:
        """Generate a simulated user response.

        Returns:
            Dict with keys: thought, grounding, response, cost, raw.
        """
        prompt = render_template(
            _V2_TEMPLATE,
            user_profile=task.persona_info,
            communication_habit=self.habit,
            init_user_instruction=task.elevator_pitch,
            gold_task_requirement=task.task_requirement,
            conversation_history=chat_history,
            agent_question=agent_message,
        )
        result = call_llm(
            self.model_name,
            prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        parsed = parse_json_output(result["output"])
        return {
            "thought": parsed.get("thought", ""),
            "grounding": parsed.get("grounding", ""),
            "response": parsed.get("feedback", result["output"]),
            "cost": result.get("cost", 0.0),
            "raw": parsed,
        }
