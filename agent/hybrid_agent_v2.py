"""Hybrid agent v2 for AUNUEnv.

Two-phase workflow:
  Phase 1 (turns 0 .. mid_turn-1): Adaptive routing — same as HybridAgent.
    Router decides data_interaction / user_interaction each turn.
  Mid-synthesis (turn == mid_turn): Synthesise an intermediate task requirement
    from all Phase 1 evidence.
  Phase 2 (turns mid_turn+1 .. max_iterations): User-interaction only.
    Router chooses user_interaction or reflection_synthesis; data_interaction
    is no longer available.
  Final synthesis: Synthesise the definitive task requirement from all evidence.

LangGraph graph:
  [router] ──(data_interaction)──▶ [data_node] ──▶ [router]   (phase 1 only)
           ──(user_interaction)──▶ [user_node] ──▶ [router]
           ──(mid_synthesis)   ──▶ [mid_synthesis_node] ──▶ [router]
           ──(reflection_synthesis)──▶ [synthesis_node] ──▶ END
"""

import logging
import os
import re
import sys
from datetime import datetime, timezone
from typing import Annotated, Sequence, TypedDict
import operator

from langgraph.graph import StateGraph, END

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_AGENT_DIR = os.path.dirname(__file__)
for _p in (_REPO_ROOT, _AGENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent_state import Message
from AUNUEnv.aunu_env.config import AUNUEnvConfig
from AUNUEnv.aunu_env.env.actions import inspect_data, ask_user, finish
from AUNUEnv.aunu_env.utils.llm import call_llm
from AUNUEnv.aunu_env.utils.jinja_utils import render_template
from AUNUEnv.aunu_env.utils.json_utils import parse_json_output

# Re-use unchanged helpers from hybrid_agent
from hybrid_agent import _make_msg, _build_chat_history

logger = logging.getLogger(__name__)

_PROMPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "prompts"))

# Phase 1 — reuse the same templates as HybridAgent (do not modify)
_ROUTER_P1_TEMPLATE          = os.path.join(_PROMPTS_DIR, "hybrid/hybrid_router_v2.jinja")
_DATA_OBSERVATION_TEMPLATE   = os.path.join(_PROMPTS_DIR, "hybrid/hybrid_data_observation.jinja")
_USER_Q_P1_TEMPLATE          = os.path.join(_PROMPTS_DIR, "user/user_interaction_for_hybrid.jinja")
_GUIDELINE_TEMPLATE          = os.path.join(_PROMPTS_DIR, "hybrid/hybrid_guideline_update.jinja")
_FINAL_SYNTHESIS_TEMPLATE    = os.path.join(_PROMPTS_DIR, "hybrid/hybrid_synthesis.jinja")

# Phase 2 — new v2-specific templates
_MID_SYNTHESIS_TEMPLATE      = os.path.join(_PROMPTS_DIR, "hybrid/hybrid_v2_mid_synthesis.jinja")
_ROUTER_P2_TEMPLATE          = os.path.join(_PROMPTS_DIR, "hybrid/hybrid_v2_phase2_router.jinja")
_USER_Q_P2_TEMPLATE          = os.path.join(_PROMPTS_DIR, "hybrid/hybrid_v2_phase2_user.jinja")

_N_SAMPLES = 5


class HybridV2State(TypedDict, total=False):
    messages: Annotated[Sequence[Message], operator.add]
    is_complete: bool
    current_turn: int
    phase: int                      # 1 = data+user exploration, 2 = user refinement only
    task_requirement_final: str     # updated at mid-synthesis and final synthesis
    mid_synthesis: str              # the intermediate synthesised requirement
    next_action: str
    user_interactions: list         # all user Q&A (both phases)
    user_interactions_phase2: list  # phase 2 user Q&A only
    data_interactions: list
    guideline: str
    router_history: list
    format_reflection_history: list


class HybridV2Agent:
    """Two-phase hybrid agent.

    Phase 1 mirrors HybridAgent behaviour exactly (same prompts, same routing).
    After `mid_turn` iterations a mid-synthesis is forced, then Phase 2 proceeds
    with user-interaction-only turns that validate and refine the synthesised draft.

    Args:
        model_name: LLM model identifier.
        max_iterations: Total routing iterations (Phase 1 + Phase 2).
        mid_turn: Turn at which Phase 1 ends and mid-synthesis fires (default 6).
        temperature: Sampling temperature.
        max_tokens: Max output tokens for requirement generation.
        question_max_tokens: Max output tokens for question generation.
        router_max_tokens: Max output tokens for router decisions.
    """

    def __init__(
        self,
        model_name: str,
        max_iterations: int = 20,
        mid_turn: int = 6,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        question_max_tokens: int = 2048,
        router_max_tokens: int = 256,
    ):
        self.model_name = model_name
        self.max_iterations = max_iterations
        self.mid_turn = mid_turn
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.question_max_tokens = question_max_tokens
        self.router_max_tokens = router_max_tokens
        self._env = None

    @classmethod
    def from_config(cls, config: AUNUEnvConfig, max_iterations: int = 20, mid_turn: int = 6) -> "HybridV2Agent":
        return cls(
            model_name=config.agent_model,
            max_iterations=max_iterations,
            mid_turn=mid_turn,
            temperature=config.effective_agent_temperature,
            max_tokens=config.max_tokens,
        )

    @classmethod
    def from_yaml(cls, yaml_path: str, max_iterations: int = 20, mid_turn: int = 6) -> "HybridV2Agent":
        return cls.from_config(AUNUEnvConfig.from_yaml(yaml_path), max_iterations=max_iterations, mid_turn=mid_turn)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _call_guideline_update(self, state: HybridV2State, interaction_type: str,
                                interaction: dict, turn: int, messages: list) -> str:
        env_state = self._env.state
        start = datetime.now(timezone.utc).isoformat()
        prompt = render_template(
            _GUIDELINE_TEMPLATE,
            initial_user_requirement=env_state.task.elevator_pitch,
            current_guideline=state.get("guideline", ""),
            turn=turn,
            interaction_type=interaction_type,
            interaction=interaction,
        )
        result = call_llm(self.model_name, prompt, max_tokens=4096, temperature=self.temperature)
        end = datetime.now(timezone.utc).isoformat()
        messages.append(_make_msg(
            start, end, "aunu_agent", "guideline_update",
            prompt, "hybrid/hybrid_guideline_update.jinja",
            "", result["output"], self.model_name, result,
        ))
        logger.info(f"[HybridV2] Turn {turn}: guideline updated ({len(result['output'])} chars)")
        return result["output"]

    # ------------------------------------------------------------------
    # LangGraph nodes
    # ------------------------------------------------------------------

    def router_node(self, state: HybridV2State) -> dict:
        env_state = self._env.state
        user_requirement = env_state.task.elevator_pitch
        turn = state.get("current_turn", 0)
        phase = state.get("phase", 1)
        messages: list[Message] = []

        # Force final synthesis when max iterations reached
        if turn >= self.max_iterations:
            logger.info("[HybridV2] Max iterations reached — forcing reflection_synthesis")
            return {"messages": messages, "next_action": "reflection_synthesis", "current_turn": turn}

        # Phase 1 → mid_synthesis transition
        if phase == 1 and turn >= self.mid_turn:
            logger.info(f"[HybridV2] Turn {turn}: Phase 1 complete — triggering mid_synthesis")
            return {"messages": messages, "next_action": "mid_synthesis", "current_turn": turn + 1}

        if phase == 1:
            # Phase 1 router — identical logic to HybridAgent.router_node
            start = datetime.now(timezone.utc).isoformat()
            prompt = render_template(
                _ROUTER_P1_TEMPLATE,
                initial_user_requirement=user_requirement,
                current_task_requirement=state.get("task_requirement_final", user_requirement),
                guideline=state.get("guideline", ""),
                current_iteration=turn,
                max_iterations=self.mid_turn,
                data_interaction_count=len(state.get("data_interactions", [])),
            )
            result = call_llm(self.model_name, prompt,
                              max_tokens=self.router_max_tokens, temperature=self.temperature)
            end = datetime.now(timezone.utc).isoformat()

            parsed = parse_json_output(result["output"])
            action = parsed.get("action", "user_interaction")
            reason = parsed.get("reason", "")
            target_gap = parsed.get("target_gap", "")

            valid_p1 = {"data_interaction", "user_interaction"}
            if action not in valid_p1:
                action = "user_interaction"
                reason = f"[OVERRIDDEN] action blocked in Phase 1 (current: {turn})"

            logger.info(f"[HybridV2] Phase 1 Turn {turn}: router → {action} | {reason}")
            messages.append(_make_msg(
                start, end, "aunu_agent", "router",
                prompt, "hybrid/hybrid_router_v2.jinja", target_gap, result["output"],
                self.model_name, result,
            ))

        else:
            # Phase 2 router — user_interaction or reflection_synthesis only
            p2_turns = len(state.get("user_interactions_phase2", []))
            p2_max = self.max_iterations - self.mid_turn - 1  # remaining turns for phase 2

            start = datetime.now(timezone.utc).isoformat()
            prompt = render_template(
                _ROUTER_P2_TEMPLATE,
                initial_user_requirement=user_requirement,
                synthesized_requirement=state.get("mid_synthesis", state.get("task_requirement_final", user_requirement)),
                guideline=state.get("guideline", ""),
                user_interactions_phase2=state.get("user_interactions_phase2", []),
                current_iteration=p2_turns,
                max_iterations=p2_max,
            )
            result = call_llm(self.model_name, prompt,
                              max_tokens=self.router_max_tokens, temperature=self.temperature)
            end = datetime.now(timezone.utc).isoformat()

            parsed = parse_json_output(result["output"])
            action = parsed.get("action", "user_interaction")
            reason = parsed.get("reason", "")
            target_gap = parsed.get("target_gap", "")

            valid_p2 = {"user_interaction", "reflection_synthesis"}
            if action not in valid_p2:
                action = "user_interaction"
                reason = f"[OVERRIDDEN] data_interaction not allowed in Phase 2 (current turn: {turn})"

            logger.info(f"[HybridV2] Phase 2 Turn {turn}: router → {action} | {reason}")
            messages.append(_make_msg(
                start, end, "aunu_agent", "router",
                prompt, "hybrid/hybrid_v2_phase2_router.jinja", target_gap, result["output"],
                self.model_name, result,
            ))

        updated_router_history = list(state.get("router_history", [])) + [{
            "turn": turn,
            "phase": phase,
            "action": action,
            "reason": reason,
            "target_gap": target_gap,
        }]

        return {
            "messages": messages,
            "next_action": action,
            "router_history": updated_router_history,
            "current_turn": turn + 1,
        }

    def data_node(self, state: HybridV2State) -> dict:
        """Phase 1 only — sample data and produce structured observations."""
        env = self._env
        env_state = env.state
        turn = state.get("current_turn", 1)
        current_req = state.get("task_requirement_final", env_state.task.elevator_pitch)
        messages: list[Message] = []

        _, _, _, info = env.step(inspect_data(n_samples=_N_SAMPLES, split="defining_instances"))
        raw_samples = info.get("data_samples", [])
        logger.info(f"[HybridV2] Data turn {turn}: sampled {len(raw_samples)} rows")

        start = datetime.now(timezone.utc).isoformat()
        from hybrid_agent import _DATA_OBSERVATION_TEMPLATE as _DOT
        prompt = render_template(
            _DOT,
            initial_user_requirement=env_state.task.elevator_pitch,
            current_task_requirement=current_req,
            data_samples=raw_samples,
        )
        result = call_llm(self.model_name, prompt, max_tokens=1024, temperature=self.temperature)
        end = datetime.now(timezone.utc).isoformat()

        observation = result["output"]
        messages.append(_make_msg(
            start, end, "aunu_agent", "data_observation",
            prompt, "hybrid/hybrid_data_observation.jinja",
            "", observation, self.model_name, result,
        ))

        new_entry = {"turn": turn, "reflection": observation}
        updated_data = list(state.get("data_interactions", [])) + [new_entry]
        updated_guideline = self._call_guideline_update(state, "data_interaction", new_entry, turn, messages)

        return {
            "messages": messages,
            "data_interactions": updated_data,
            "guideline": updated_guideline,
        }

    def user_node(self, state: HybridV2State) -> dict:
        """User interaction — uses Phase 1 or Phase 2 prompt depending on current phase."""
        env = self._env
        env_state = env.state
        turn = state.get("current_turn", 1)
        phase = state.get("phase", 1)
        messages: list[Message] = []

        chat_history = _build_chat_history(env_state.interaction_history)

        start = datetime.now(timezone.utc).isoformat()

        if phase == 1:
            prompt = render_template(
                _USER_Q_P1_TEMPLATE,
                chat_history=chat_history if chat_history else None,
                initial_requirement=env_state.task.elevator_pitch,
                current_task_requirement=state.get("task_requirement_final", env_state.task.elevator_pitch),
                guideline=state.get("guideline", ""),
                max_iterations=self.mid_turn,
                current_iteration=turn,
            )
            template_name = "user/user_interaction_for_hybrid.jinja"
        else:
            p2_turns = len(state.get("user_interactions_phase2", []))
            p2_max = self.max_iterations - self.mid_turn - 1
            prompt = render_template(
                _USER_Q_P2_TEMPLATE,
                chat_history=chat_history if chat_history else None,
                initial_requirement=env_state.task.elevator_pitch,
                synthesized_requirement=state.get("mid_synthesis", state.get("task_requirement_final", env_state.task.elevator_pitch)),
                guideline=state.get("guideline", ""),
                max_iterations=p2_max,
                current_iteration=p2_turns,
            )
            template_name = "hybrid/hybrid_v2_phase2_user.jinja"

        result = call_llm(self.model_name, prompt,
                          max_tokens=self.question_max_tokens, temperature=self.temperature)
        end = datetime.now(timezone.utc).isoformat()

        parsed = parse_json_output(result["output"])
        question = parsed.get("question") or ""
        if not question:
            m = re.search(r'"question"\s*:\s*"((?:[^"\\]|\\.)*)"', result["output"], re.DOTALL)
            question = m.group(1).replace('\\"', '"') if m else result["output"]
        ambiguity = parsed.get("identified_ambiguity", "")
        logger.info(f"[HybridV2] User turn {turn} (phase {phase}): {question[:80]}")

        messages.append(_make_msg(
            start, end, "aunu_agent", "ask_user",
            prompt, template_name, ambiguity, question,
            self.model_name, result,
        ))

        env_start = datetime.now(timezone.utc).isoformat()
        obs, reward, done, info = env.step(ask_user(question))
        env_end = datetime.now(timezone.utc).isoformat()
        user_response = obs.get("last_response", "")

        messages.append({
            "start_time": env_start, "end_time": env_end,
            "role": "mimic_user", "action": "respond",
            "input": question, "prompt_template": "feedback_mimic_user_v3.jinja",
            "identified_ambiguity": "", "output": user_response,
            "thought": info.get("user_thought", ""), "grounding": info.get("user_grounding", ""),
            "llm": "", "input_tokens": 0, "output_tokens": 0,
            "cost": info.get("cost", 0.0),
            "env_response": {"obs": obs, "reward": reward, "done": done, "info": info},
        })
        messages.append({
            "start_time": env_start, "end_time": env_end,
            "role": "mimic_user_feedback", "action": "feedback",
            "input": question, "output": user_response,
            "thought": info.get("user_thought", ""), "grounding": info.get("user_grounding", ""),
        })

        new_entry = {"turn": turn, "question": question, "answer": user_response, "ambiguity": ambiguity}
        updated_user = list(state.get("user_interactions", [])) + [new_entry]

        updated_guideline = self._call_guideline_update(state, "user_interaction", new_entry, turn, messages)

        updates = {
            "messages": messages,
            "user_interactions": updated_user,
            "guideline": updated_guideline,
        }
        if phase == 2:
            updated_p2 = list(state.get("user_interactions_phase2", [])) + [new_entry]
            updates["user_interactions_phase2"] = updated_p2

        return updates

    def mid_synthesis_node(self, state: HybridV2State) -> dict:
        """Synthesise an intermediate task requirement from Phase 1 evidence."""
        env_state = self._env.state
        messages: list[Message] = []

        start = datetime.now(timezone.utc).isoformat()
        prompt = render_template(
            _MID_SYNTHESIS_TEMPLATE,
            initial_user_requirement=env_state.task.elevator_pitch,
            current_task_requirement=state.get("task_requirement_final", env_state.task.elevator_pitch),
            user_interactions=state.get("user_interactions", []),
            data_interactions=state.get("data_interactions", []),
        )
        result = call_llm(self.model_name, prompt, max_tokens=self.max_tokens, temperature=self.temperature)
        end = datetime.now(timezone.utc).isoformat()
        mid_req = result["output"]
        logger.info(f"[HybridV2] Mid-synthesis complete ({len(mid_req)} chars) — entering Phase 2")

        messages.append(_make_msg(
            start, end, "aunu_agent", "mid_synthesis",
            prompt, "hybrid/hybrid_v2_mid_synthesis.jinja", "", mid_req,
            self.model_name, result,
        ))

        return {
            "messages": messages,
            "mid_synthesis": mid_req,
            "task_requirement_final": mid_req,
            "phase": 2,
            "next_action": "user_interaction",
        }

    def synthesis_node(self, state: HybridV2State) -> dict:
        """Final synthesis — same as HybridAgent.synthesis_node."""
        env = self._env
        env_state = env.state
        messages: list[Message] = []

        start = datetime.now(timezone.utc).isoformat()
        prompt = render_template(
            _FINAL_SYNTHESIS_TEMPLATE,
            initial_user_requirement=env_state.task.elevator_pitch,
            current_task_requirement=state.get("task_requirement_final", env_state.task.elevator_pitch),
            user_interactions=state.get("user_interactions", []),
            data_interactions=state.get("data_interactions", []),
        )
        result = call_llm(self.model_name, prompt, max_tokens=self.max_tokens, temperature=self.temperature)
        end = datetime.now(timezone.utc).isoformat()
        final_req = result["output"]
        logger.info(f"[HybridV2] Final synthesis complete ({len(final_req)} chars)")

        env.step(finish(final_req))

        messages.append(_make_msg(
            start, end, "aunu_agent", "finish",
            prompt, "hybrid/hybrid_synthesis.jinja", "", final_req,
            self.model_name, result,
        ))

        return {
            "messages": messages,
            "is_complete": True,
            "task_requirement_final": final_req,
        }

    # ------------------------------------------------------------------
    # Conditional edge
    # ------------------------------------------------------------------

    def _route(self, state: HybridV2State) -> str:
        if state.get("is_complete"):
            return "end"
        action = state.get("next_action", "reflection_synthesis")
        valid = {"data_interaction", "user_interaction", "mid_synthesis", "reflection_synthesis"}
        if action not in valid:
            return "reflection_synthesis"
        # data_interaction only allowed in phase 1
        if action == "data_interaction" and state.get("phase", 1) != 1:
            return "user_interaction"
        return action

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_graph(self):
        workflow = StateGraph(HybridV2State)
        workflow.add_node("router", self.router_node)
        workflow.add_node("data_interaction", self.data_node)
        workflow.add_node("user_interaction", self.user_node)
        workflow.add_node("mid_synthesis", self.mid_synthesis_node)
        workflow.add_node("reflection_synthesis", self.synthesis_node)

        workflow.set_entry_point("router")

        workflow.add_conditional_edges(
            "router",
            self._route,
            {
                "data_interaction": "data_interaction",
                "user_interaction": "user_interaction",
                "mid_synthesis": "mid_synthesis",
                "reflection_synthesis": "reflection_synthesis",
                "end": END,
            },
        )
        workflow.add_edge("data_interaction", "router")
        workflow.add_edge("user_interaction", "router")
        workflow.add_edge("mid_synthesis", "router")
        workflow.add_edge("reflection_synthesis", END)

        return workflow.compile()

    def run(self, env, task, user, initial_requirement: str | None = None) -> dict:
        """Run a complete hybrid_v2 episode and return the trajectory log."""
        env.reset(task, user)
        self._env = env

        seed = initial_requirement or task.elevator_pitch
        initial_state: HybridV2State = {
            "messages": [],
            "is_complete": False,
            "current_turn": 0,
            "phase": 1,
            "task_requirement_final": seed,
            "mid_synthesis": "",
            "next_action": "",
            "user_interactions": [],
            "user_interactions_phase2": [],
            "data_interactions": [],
            "guideline": "",
            "router_history": [],
            "format_reflection_history": [],
        }

        app = self.build_graph()
        all_messages = []
        final_state: HybridV2State = {}
        for output in app.stream(initial_state):
            for _, state_update in output.items():
                if "messages" in state_update:
                    all_messages.extend(state_update["messages"])
                for key in ("router_history", "user_interactions", "user_interactions_phase2",
                            "data_interactions", "mid_synthesis", "format_reflection_history"):
                    if key in state_update:
                        final_state[key] = state_update[key]

        log = env.get_trajectory_log()
        log["agent_messages"] = [dict(m) for m in all_messages]
        log["mid_synthesis"] = final_state.get("mid_synthesis", "")
        log["router_history"] = final_state.get("router_history", [])
        log["user_interactions"] = final_state.get("user_interactions", [])
        log["user_interactions_phase2"] = final_state.get("user_interactions_phase2", [])
        log["data_interactions"] = final_state.get("data_interactions", [])
        log["format_reflection_history"] = final_state.get("format_reflection_history", [])
        return log
