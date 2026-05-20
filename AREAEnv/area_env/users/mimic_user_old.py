"""
MIMIC User simulator using the feedback_mimic_user_v2 prompt template.

This variant exposes the same interface as MimicUser (mimic_user.py) but
renders the v2 template, which expects explicit user_profile and
communication_style_init variables derived from the TaskInstance persona.

Both passive and persona modes are supported via persona_config, matching
the MimicUser contract exactly.
"""

import dataclasses
import logging
import os
from typing import Optional

from ..dataset.schema import TaskInstance
from ..utils.llm import call_llm
from ..utils.jinja_utils import render_template
from ..utils.json_utils import parse_json_output

logger = logging.getLogger(__name__)

_PROMPT_DIR = os.path.join(os.path.dirname(__file__), "prompts")
_V2_TEMPLATE = os.path.join(_PROMPT_DIR, "feedback_mimic_user_v2.jinja")

_DEFAULT_COMMUNICATION_STYLE = (
    "Be natural, conversational, and somewhat uncertain. "
    "Keep responses relatively short. Do not sound like an AI assistant."
)


def _persona_info_to_dict(persona_info) -> dict:
    """Convert a PersonaInfo dataclass (or plain dict) to a dict safely."""
    if persona_info is None:
        return {}
    if dataclasses.is_dataclass(persona_info) and not isinstance(persona_info, type):
        return dataclasses.asdict(persona_info)
    if isinstance(persona_info, dict):
        return persona_info
    return {}


def _build_user_profile(task: TaskInstance, persona_config: Optional[dict]) -> str:
    """Compose a user_profile string from TaskInstance persona info and optional overrides."""
    lines = []
    info = _persona_info_to_dict(task.persona_info)
    for k, v in info.items():
        if k != "persona_id":
            lines.append(f"{k}: {v}")
    if persona_config:
        for k, v in persona_config.items():
            lines.append(f"{k}: {v}")
    return "\n".join(lines) if lines else "A general user with no specific background provided."


def _build_communication_style(task: TaskInstance, persona_config: Optional[dict]) -> str:
    """Derive a communication style description from persona info."""
    info = _persona_info_to_dict(task.persona_info)
    overrides = persona_config or {}
    style_parts = []

    # persona_config overrides take priority; fall back to PersonaInfo fields
    expertise = overrides.get("expertise_level") or ""
    verbosity = overrides.get("verbosity") or ""
    comm_style = overrides.get("communication_style") or ""

    # Derive from actual PersonaInfo fields when no overrides provided
    role = info.get("role", "")
    competency = info.get("competency_matrix", {})
    motivation = info.get("business_motivation", "")

    if expertise:
        style_parts.append(f"Expertise level: {expertise}.")
    elif role:
        style_parts.append(f"Role: {role}.")
    if verbosity:
        style_parts.append(f"Verbosity: {verbosity}.")
    if comm_style:
        style_parts.append(f"Communication style: {comm_style}.")
    if motivation:
        style_parts.append(f"Primary motivation: {motivation}.")
    if competency and not expertise:
        style_parts.append(f"Competencies: {competency}.")

    return " ".join(style_parts) if style_parts else _DEFAULT_COMMUNICATION_STYLE


class MimicUserV2:
    """Simulated user for AREAEnv episodes using the v2 feedback prompt.

    Implements the same interface as MimicUser so it can be used as a
    drop-in replacement anywhere MimicUser is accepted.

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

    def respond(
        self,
        task: TaskInstance,
        chat_history: list,
        agent_message: str,
    ) -> dict:
        """Generate a simulated user response using the v2 template.

        Args:
            task: The current TaskInstance (provides elevator_pitch, persona_info,
                  and gold task_requirement).
            chat_history: List of dicts with keys 'role' and 'content'.
            agent_message: The agent's latest message (question or proposal).

        Returns:
            Dict with keys:
                thought (str): Internal reasoning (not shown to agent).
                grounding (str): "supported" | "partially_supported" | "unsupported"
                response (str): The user's reply to the agent.
                cost (float): LLM cost for this call.
                raw (dict): Full parsed LLM output.
        """
        user_profile = _build_user_profile(task, self.persona_config)
        communication_style = _build_communication_style(task, self.persona_config)

        prompt = render_template(
            _V2_TEMPLATE,
            user_profile=user_profile,
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
