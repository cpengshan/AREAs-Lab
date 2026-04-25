"""
MimicUserV2 — uses feedback_mimic_user_v2.jinja + responser_habit.json.

Drop-in replacement for MimicUser with the same respond() interface,
but driven by the v2 prompt and habit-based communication styles.

Habit assignment is automatic: cycles through available habits based on
task.persona_id, so no manual --habit_id needed.
"""

import json
import logging
import os

from ..dataset.schema import TaskInstance
from ..utils.llm import call_llm
from ..utils.jinja_utils import render_template
from ..utils.json_utils import parse_json_output

logger = logging.getLogger(__name__)

_PROMPT_DIR = os.path.join(os.path.dirname(__file__), "prompts")
_V2_TEMPLATE = os.path.join(_PROMPT_DIR, "feedback_mimic_user_v2.jinja")
_HABIT_FILE = os.path.join(_PROMPT_DIR, "responser_habit.json")


def _load_all_habits() -> list:
    with open(_HABIT_FILE) as f:
        return json.load(f)


def _get_communication_style(habits: list, persona_id: int) -> str:
    """Cycle through habits based on persona_id (1-indexed)."""
    idx = (persona_id - 1) % len(habits)
    return json.dumps(habits[idx]["communication_style"], indent=2)


class MimicUserV2:
    """Simulated user using feedback_mimic_user_v2.jinja and responser_habit.json.

    Habit is assigned automatically by cycling through responser_habit.json
    based on task.persona_id — no need to set it manually.

    Args:
        model_name: LLM model to use.
        temperature: Sampling temperature.
        max_tokens: Max output tokens.
    """

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.persona_config = {"mode": "v2"}  # required by aunu_env.py
        self._habits = _load_all_habits()

    def respond(
        self,
        task: TaskInstance,
        chat_history: list,
        agent_message: str,
    ) -> dict:
        """Generate a simulated user response using the v2 prompt.

        Args:
            task: Current TaskInstance (elevator_pitch + task_requirement).
            chat_history: List of dicts with keys 'role' and 'content'.
            agent_message: The agent's latest message.

        Returns:
            Dict with keys: thought, grounding, response, cost, raw.
        """
        communication_style = _get_communication_style(self._habits, task.persona_id)
        habit_idx = (task.persona_id - 1) % len(self._habits)
        logger.debug(f"[MimicUserV2] persona_id={task.persona_id} → habit {self._habits[habit_idx]['habit']}")

        prompt = render_template(
            _V2_TEMPLATE,
            user_profile=task.persona_info,
            communication_style_init=communication_style,
            init_user_instruction=task.elevator_pitch,
            gold_task_requirement=task.task_requirement,
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
