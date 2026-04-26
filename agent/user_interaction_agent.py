"""User-interaction agent for AUNUEnv.

Clarifies task requirements through iterative dialogue with the MIMIC user
(hosted inside AUNUEnv), then synthesizes a final requirement.

LangGraph graph:
  [aunu_agent] ──(continue)──▶ [aunu_agent]
               ──(end)──▶ END

The single self-looping node manages the full state machine:
  Phase 0: zero-shot draft
  Phase 1: clarification loop (up to max_turns) → ask_user per turn
  Phase 2: synthesize final → finish

All MIMIC user calls are mediated through env.step(ask_user(...)) — the
agent never instantiates MimicUser directly.
"""

import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from typing import Optional

from langgraph.graph import StateGraph, END

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_AGENT_DIR = os.path.dirname(__file__)
for _p in (_REPO_ROOT, _AGENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent_state import AgentState, Message
from AUNUEnv.aunu_env.config import AUNUEnvConfig
from AUNUEnv.aunu_env.env.actions import ask_user, finish
from AUNUEnv.aunu_env.utils.llm import call_llm
from AUNUEnv.aunu_env.utils.jinja_utils import render_template
from AUNUEnv.aunu_env.utils.json_utils import parse_json_output

logger = logging.getLogger(__name__)

_PROMPTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "prompts")
)
_ZERO_SHOT_TEMPLATE = os.path.join(_PROMPTS_DIR, "zero_shot.jinja")
_USER_INTERACT_TEMPLATE = os.path.join(_PROMPTS_DIR, "user/user_interaction.jinja")
_PREDICTION_TEMPLATE = os.path.join(_PROMPTS_DIR, "user/task_requirement_prediction.jinja")


def _build_chat_history(interaction_history: list) -> list:
    """Convert env interaction_history to the format expected by prompt templates."""
    return [
        {"role": e["role"], "output": e["content"]}
        for e in interaction_history
        if e["role"] in ("agent", "user")
    ]


class UserInteractionAgent:
    """LangGraph-based user-interaction agent that operates inside AUNUEnv.

    The agent first produces a zero-shot draft, then runs up to `max_turns`
    clarification rounds with the MIMIC user via the env's ask_user action,
    and finally synthesizes a complete task requirement.

    Args:
        model_name: LLM model identifier.
        max_turns: Maximum clarification rounds before forced synthesis.
        temperature: Sampling temperature.
        max_tokens: Maximum output tokens for requirement generation.
        question_max_tokens: Maximum output tokens for question generation.
    """

    def __init__(
        self,
        model_name: str,
        max_turns: int = 5,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        question_max_tokens: int = 1024,
    ):
        self.model_name = model_name
        self.max_turns = max_turns
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.question_max_tokens = question_max_tokens
        self._env = None  # set before run()

    @classmethod
    def from_config(cls, config: AUNUEnvConfig, max_turns: int = 5) -> "UserInteractionAgent":
        """Construct a UserInteractionAgent from an AUNUEnvConfig."""
        return cls(
            model_name=config.agent_model,
            max_turns=max_turns,
            temperature=config.effective_agent_temperature,
            max_tokens=config.max_tokens,
        )

    @classmethod
    def from_yaml(cls, yaml_path: str, max_turns: int = 5) -> "UserInteractionAgent":
        """Construct a UserInteractionAgent by loading an AUNUEnvConfig YAML file."""
        config = AUNUEnvConfig.from_yaml(yaml_path)
        return cls.from_config(config, max_turns=max_turns)

    # ------------------------------------------------------------------
    # LangGraph node
    # ------------------------------------------------------------------

    def process(self, state: AgentState) -> dict:
        """LangGraph node: manage one step of the agent state machine."""
        env = self._env
        env_state = env.state
        turn = state["current_turn"]
        user_request = env_state.task.elevator_pitch
        messages: list[Message] = []
        updates = {}

        # ── Phase 0: generate zero-shot draft ──────────────────────────
        if turn == 0:
            start = datetime.now(timezone.utc).isoformat()
            prompt = render_template(_ZERO_SHOT_TEMPLATE, user_instruction=user_request)
            result = call_llm(self.model_name, prompt,
                              max_tokens=self.max_tokens, temperature=self.temperature)
            end = datetime.now(timezone.utc).isoformat()
            draft = result["output"]
            logger.info(f"[UserInteraction] Zero-shot draft ({len(draft)} chars)")

            messages.append(_make_msg(
                start, end, "aunu_agent", "zero_shot_draft",
                prompt, "zero_shot.jinja", "", draft,
                self.model_name, result,
            ))
            updates = {
                "messages": messages,
                "is_complete": False,
                "current_turn": 1,
                "zero_shot_draft": draft,
                "task_requirement_final": draft,
            }
            return updates

        # ── Phase 1: clarification turn ────────────────────────────────
        if turn <= self.max_turns:
            chat_history = _build_chat_history(env_state.interaction_history)

            start = datetime.now(timezone.utc).isoformat()
            prompt = render_template(
                _USER_INTERACT_TEMPLATE,
                chat_history=chat_history if chat_history else None,
                initial_requirement=user_request,
                max_iterations=self.max_turns,
                current_iteration=turn,
            )
            result = call_llm(self.model_name, prompt,
                              max_tokens=self.question_max_tokens, temperature=self.temperature)
            end = datetime.now(timezone.utc).isoformat()

            parsed = parse_json_output(result["output"])
            question = parsed.get("question", result["output"])
            ambiguity = parsed.get("identified_ambiguity", "")
            logger.info(f"[UserInteraction] Turn {turn}: {question[:80]}")

            # Send question to MIMIC user via env
            env_response_time = datetime.now(timezone.utc).isoformat()
            obs, reward, done, info = env.step(ask_user(question))
            env_response_end_time = datetime.now(timezone.utc).isoformat()
            user_response = obs.get("last_response", "")

            messages.append(_make_msg(
                start, end, "aunu_agent", "ask_user",
                prompt, "user/user_interaction.jinja", ambiguity, question,
                self.model_name, result,
            ))
            messages.append({
                "start_time": env_response_time,
                "end_time": env_response_end_time,
                "role": "mimic_user",
                "action": "respond",
                "input": question,
                "prompt_template": "feedback_mimic_user_v3.jinja",
                "identified_ambiguity": "",
                "output": user_response,
                "thought": info.get("user_thought", ""),
                "grounding": info.get("user_grounding", ""),
                "llm": "",
                "input_tokens": 0,
                "output_tokens": 0,
                "cost": info.get("cost", 0.0),
                "env_response": {
                    "obs": obs,
                    "reward": reward,
                    "done": done,
                    "info": info,
                },
            })
            messages.append({
                "start_time": env_response_time,
                "end_time": env_response_end_time,
                "role": "mimic_user_feedback",
                "action": "feedback",
                "input": question,
                "output": user_response,
                "thought": info.get("user_thought", ""),
                "grounding": info.get("user_grounding", ""),
            })

            next_turn = turn + 1
            is_last = (next_turn > self.max_turns) or done
            if not is_last:
                return {
                    "messages": messages,
                    "is_complete": False,
                    "current_turn": next_turn,
                    "zero_shot_draft": state["zero_shot_draft"],
                    "task_requirement_final": state["task_requirement_final"],
                }
            # Last clarification turn — fall through to Phase 2 immediately.
            turn = next_turn

        # ── Phase 2: synthesize final requirement ──────────────────────
        chat_history = _build_chat_history(env_state.interaction_history)

        start = datetime.now(timezone.utc).isoformat()
        prompt = render_template(
            _PREDICTION_TEMPLATE,
            initial_task_requirement=user_request,
            zero_shot_draft=state["zero_shot_draft"],
            chat_history=chat_history if chat_history else None,
        )
        result = call_llm(self.model_name, prompt,
                          max_tokens=self.max_tokens, temperature=self.temperature)
        end = datetime.now(timezone.utc).isoformat()
        final_req = result["output"]
        logger.info(f"[UserInteraction] Final requirement ({len(final_req)} chars)")

        env.step(finish(final_req))

        messages.append(_make_msg(
            start, end, "aunu_agent", "finish",
            prompt, "user/task_requirement_prediction.jinja", "", final_req,
            self.model_name, result,
        ))
        return {
            "messages": messages,
            "is_complete": True,
            "current_turn": turn,
            "zero_shot_draft": state["zero_shot_draft"],
            "task_requirement_final": final_req,
        }

    # ------------------------------------------------------------------
    # Router
    # ------------------------------------------------------------------

    def _router(self, state: AgentState) -> str:
        """Route: continue the loop or terminate."""
        if state["is_complete"]:
            return "end"
        # If we've finished clarification turns, next call synthesizes
        return "continue"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_graph(self):
        """Compile and return the LangGraph StateGraph."""
        workflow = StateGraph(AgentState)
        workflow.add_node("aunu_agent", self.process)
        workflow.set_entry_point("aunu_agent")
        workflow.add_conditional_edges(
            "aunu_agent",
            self._router,
            {"continue": "aunu_agent", "end": END},
        )
        return workflow.compile()

    def run(self, env, task, user) -> dict:
        """Run a complete user-interaction episode and return the trajectory log.

        Args:
            env: AUNUEnv instance (already initialised with evaluator).
            task: TaskInstance to solve.
            user: MimicUser instance (passive or persona-conditioned).

        Returns:
            Trajectory log dict from env.get_trajectory_log(), augmented with
            the full agent message log under the key 'agent_messages'.
        """
        env.reset(task, user)
        self._env = env

        initial_state: AgentState = {
            "messages": [],
            "is_complete": False,
            "task_requirement_final": "",
            "current_turn": 0,
            "zero_shot_draft": "",
        }

        app = self.build_graph()
        all_messages = []
        for output in app.stream(initial_state):
            for _, state_update in output.items():
                if "messages" in state_update:
                    all_messages.extend(state_update["messages"])

        log = env.get_trajectory_log()
        log["agent_messages"] = [dict(m) for m in all_messages]
        return log


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_msg(
    start: str, end: str, role: str, action: str,
    prompt: str, template: str, ambiguity: str, output: str,
    model: str, result: dict,
) -> Message:
    return {
        "start_time": start,
        "end_time": end,
        "role": role,
        "action": action,
        "input": prompt,
        "prompt_template": template,
        "identified_ambiguity": ambiguity,
        "output": output,
        "llm": model,
        "input_tokens": result.get("input_tokens", 0),
        "output_tokens": result.get("output_tokens", 0),
        "cost": result.get("cost", 0.0),
    }
