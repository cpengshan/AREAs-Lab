"""
MIMIC User simulator for AREAEnv.

Supports two modes controlled by `persona_config`:
  - persona_config=None  → Passive confirmation mode: the user only confirms,
                           rejects, or minimally clarifies the agent's proposals.
  - persona_config=dict  → Persona-conditioned mode: a richer persona (expertise,
                           verbosity, ambiguity tolerance, etc.) shapes HOW the
                           user responds, while the gold task requirement stays
                           the same.

Both modes share the same interface and return a dict with keys:
  thought, response, (grounding — persona mode only), cost
"""

import logging
import os
from typing import Optional

from ..dataset.schema import TaskInstance
from ..utils.llm import call_llm
from ..utils.jinja_utils import render_template
from ..utils.json_utils import parse_json_output

logger = logging.getLogger(__name__)

_PROMPT_DIR = os.path.join(os.path.dirname(__file__), "prompts")
_PASSIVE_TEMPLATE = os.path.join(_PROMPT_DIR, "passive_user.jinja")
_PERSONA_TEMPLATE = os.path.join(_PROMPT_DIR, "persona_user.jinja")


class MimicUser:
    """Simulated user for AREAEnv episodes.

    Args:
        model_name: LLM model to use for response generation.
        persona_config: Optional dict with persona attributes.
            When None, passive confirmation mode is used.
            When set, keys may include:
                expertise_level (str): e.g. "novice", "intermediate", "expert"
                verbosity (str): "low", "moderate", "high"
                ambiguity_tolerance (str): "low", "medium", "high"
                preference_stability (str): "stable", "unstable"
                communication_style (str): e.g. "formal", "casual", "neutral"
        temperature: Sampling temperature for the user simulator.
        max_tokens: Max output tokens.
    """

    def __init__(
        self,
        model_name: str,
        persona_config: Optional[dict] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ):
        self.model_name = model_name
        self.persona_config = persona_config
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._mode = "persona" if persona_config is not None else "passive"

    def respond(
        self,
        task: TaskInstance,
        chat_history: list,
        agent_message: str,
    ) -> dict:
        """Generate a simulated user response.

        Args:
            task: The current TaskInstance (used for elevator_pitch and gold requirement).
            chat_history: List of dicts with keys 'role' and 'content'.
            agent_message: The agent's latest message (question or proposal).

        Returns:
            Dict with keys:
                thought (str): Internal reasoning (not shown to agent).
                response (str): The user's reply to the agent.
                grounding (str): Only present in persona mode.
                cost (float): LLM cost for this call.
                raw (dict): Full parsed LLM output.
        """
        if self._mode == "passive":
            return self._passive_respond(task, chat_history, agent_message)
        else:
            return self._persona_respond(task, chat_history, agent_message)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _passive_respond(self, task: TaskInstance, chat_history: list, agent_message: str) -> dict:
        prompt = render_template(
            _PASSIVE_TEMPLATE,
            task_elevator_pitch=task.elevator_pitch,
            gold_task_requirement=task.task_requirement,
            chat_history=chat_history,
            agent_message=agent_message,
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
            "response": parsed.get("response", result["output"]),
            "cost": result.get("cost", 0.0),
            "raw": parsed,
        }

    def _persona_respond(self, task: TaskInstance, chat_history: list, agent_message: str) -> dict:
        prompt = render_template(
            _PERSONA_TEMPLATE,
            task_elevator_pitch=task.elevator_pitch,
            gold_task_requirement=task.task_requirement,
            persona=task.persona_info,
            persona_config=self.persona_config,
            chat_history=chat_history,
            agent_message=agent_message,
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
            "response": parsed.get("response", result["output"]),
            "cost": result.get("cost", 0.0),
            "raw": parsed,
        }
