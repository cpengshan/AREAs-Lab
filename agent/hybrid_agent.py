"""Hybrid agent for AREAEnv.

Adaptive routing workflow that decides at each iteration whether to perform
data interaction, user interaction, or reflection synthesis (final step).

LangGraph graph:
  [router] ──(data_interaction)──▶ [data_node] ──▶ [router]
           ──(user_interaction)──▶ [user_node] ──▶ [router]
           ──(reflection_synthesis)──▶ [synthesis_node] ──▶ END

No zero-shot draft is produced. The agent starts from the initial user
requirement and adaptively gathers evidence before synthesizing.
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
from AREAEnv.area_env.config import AREAEnvConfig
from AREAEnv.area_env.env.actions import inspect_data, ask_user, finish
from AREAEnv.area_env.utils.llm import call_llm
from AREAEnv.area_env.utils.jinja_utils import render_template
from AREAEnv.area_env.utils.json_utils import parse_json_output

logger = logging.getLogger(__name__)


def _build_chat_history(interaction_history: list) -> list:
    return [
        {"role": e["role"], "output": e["content"]}
        for e in interaction_history
        if e["role"] in ("agent", "user")
    ]

_PROMPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "prompts"))
_ZERO_SHOT_TEMPLATE               = os.path.join(_PROMPTS_DIR, "zero_shot.jinja")
_ROUTER_TEMPLATE                  = os.path.join(_PROMPTS_DIR, "hybrid/hybrid_router_v2.jinja")
_DATA_OBSERVATION_TEMPLATE         = os.path.join(_PROMPTS_DIR, "hybrid/hybrid_data_observation.jinja")
_USER_Q_TEMPLATE                  = os.path.join(_PROMPTS_DIR, "user/user_interaction_for_hybrid.jinja")
_SYNTHESIS_TEMPLATE      = os.path.join(_PROMPTS_DIR, "hybrid/hybrid_synthesis.jinja")
_GUIDELINE_TEMPLATE      = os.path.join(_PROMPTS_DIR, "hybrid/hybrid_guideline_update.jinja")

_N_SAMPLES = 5


class HybridState(TypedDict, total=False):
    """LangGraph state for the hybrid agent."""
    messages: Annotated[Sequence[Message], operator.add]
    is_complete: bool
    current_turn: int
    zero_shot_draft: str          # initial zero-shot draft before hybrid refinement
    task_requirement_final: str   # unchanged during data turns; only written at zero-shot seed and synthesis
    next_action: str              # set by router, consumed by conditional edge
    user_interactions: list       # [{turn, question, answer}]   — full history for synthesis
    data_interactions: list       # [{turn, reflection}]          — full history for synthesis
    guideline: str                # running structured summary passed to router each turn
    router_history: list          # [{turn, action, reason, target_gap}]
    format_reflection_history: list  # unused — kept for backwards log compatibility


class HybridAgent:
    """Adaptive hybrid agent combining data and user interaction inside AREAEnv.

    At each iteration a Router LLM chooses one of:
    - "data_interaction": sample data, execute, reflect, update requirement.
    - "user_interaction": generate a clarification question, query MIMIC user.
    - "reflection_synthesis": synthesize all collected evidence into the final
      task requirement and terminate.

    If max_iterations is reached before the router stops, reflection_synthesis
    is triggered automatically.

    Args:
        model_name: LLM model identifier used for all agent LLM calls.
        max_iterations: Maximum routing iterations before forced synthesis.
        temperature: Sampling temperature.
        max_tokens: Max output tokens for requirement generation / rewriting.
        question_max_tokens: Max output tokens for question generation.
        router_max_tokens: Max output tokens for the router decision.
    """

    def __init__(
        self,
        model_name: str,
        max_iterations: int = 6,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        question_max_tokens: int = 2048,
        router_max_tokens: int = 256,
    ):
        self.model_name = model_name
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.question_max_tokens = question_max_tokens
        self.router_max_tokens = router_max_tokens
        self._env = None

    @classmethod
    def from_config(cls, config: AREAEnvConfig, max_iterations: int = 6) -> "HybridAgent":
        return cls(
            model_name=config.agent_model,
            max_iterations=max_iterations,
            temperature=config.effective_agent_temperature,
            max_tokens=config.max_tokens,
        )

    @classmethod
    def from_yaml(cls, yaml_path: str, max_iterations: int = 6) -> "HybridAgent":
        return cls.from_config(AREAEnvConfig.from_yaml(yaml_path), max_iterations=max_iterations)

    # ------------------------------------------------------------------
    # LangGraph nodes
    # ------------------------------------------------------------------

    def zero_shot_node(self, state: HybridState) -> dict:
        """Generate the initial zero-shot draft (mirrors UserInteractionAgent Phase 0)."""
        user_request = self._env.state.task.elevator_pitch
        start = datetime.now(timezone.utc).isoformat()
        prompt = render_template(_ZERO_SHOT_TEMPLATE, user_instruction=user_request)
        result = call_llm(self.model_name, prompt,
                          max_tokens=self.max_tokens, temperature=self.temperature)
        end = datetime.now(timezone.utc).isoformat()
        draft = result["output"]
        logger.info(f"[Hybrid] Zero-shot draft ({len(draft)} chars)")

        messages = [_make_msg(
            start, end, "area_agent", "zero_shot_draft",
            prompt, "zero_shot.jinja", "", draft, self.model_name, result,
        )]
        return {
            "messages": messages,
            "zero_shot_draft": draft,
            "task_requirement_final": draft,
        }

    def router_node(self, state: HybridState) -> dict:
        """Call the Router LLM and set next_action."""
        env_state = self._env.state
        user_requirement = env_state.task.elevator_pitch
        turn = state.get("current_turn", 0)
        messages: list[Message] = []

        # Force synthesis on last iteration
        if turn >= self.max_iterations:
            logger.info("[Hybrid] Max iterations reached — forcing reflection_synthesis")
            return {
                "messages": messages,
                "next_action": "reflection_synthesis",
                "current_turn": turn,
            }

        start = datetime.now(timezone.utc).isoformat()
        prompt = render_template(
            _ROUTER_TEMPLATE,
            initial_user_requirement=user_requirement,
            current_task_requirement=state.get("task_requirement_final", user_requirement),
            guideline=state.get("guideline", ""),
            current_iteration=turn,
            max_iterations=self.max_iterations,
            data_interaction_count=len(state.get("data_interactions", [])),
        )
        result = call_llm(self.model_name, prompt,
                          max_tokens=self.router_max_tokens, temperature=self.temperature)
        end = datetime.now(timezone.utc).isoformat()

        parsed = parse_json_output(result["output"])
        action = parsed.get("action", "reflection_synthesis")
        reason = parsed.get("reason", "")
        target_gap = parsed.get("target_gap", "")

        # Validate action
        valid_actions = {"data_interaction", "user_interaction", "reflection_synthesis"}
        if action not in valid_actions:
            logger.warning(f"[Hybrid] Router returned invalid action '{action}', defaulting to reflection_synthesis")
            action = "reflection_synthesis"

        # Hard-enforce minimum 15 iterations before synthesis
        if action == "reflection_synthesis" and turn < 15:
            logger.warning(f"[Hybrid] Router chose reflection_synthesis at iteration {turn} (<15) — overriding to user_interaction")
            action = "user_interaction"
            reason = f"[OVERRIDDEN] reflection_synthesis blocked before iteration 15 (current: {turn})"

        logger.info(f"[Hybrid] Turn {turn}: router → {action} | {reason}")

        messages.append(_make_msg(
            start, end, "area_agent", "router",
            prompt, "hybrid/hybrid_router_v2.jinja", target_gap, result["output"],
            self.model_name, result,
        ))

        updated_router_history = list(state.get("router_history", [])) + [{
            "turn": turn,
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

    def _call_guideline_update(
        self,
        state: HybridState,
        interaction_type: str,
        interaction: dict,
        turn: int,
        messages: list,
    ) -> str:
        """Call the guideline-update LLM and return the updated guideline string."""
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
        result = call_llm(self.model_name, prompt,
                          max_tokens=4096, temperature=self.temperature)
        end = datetime.now(timezone.utc).isoformat()
        messages.append(_make_msg(
            start, end, "area_agent", "guideline_update",
            prompt, "hybrid/hybrid_guideline_update.jinja",
            "", result["output"], self.model_name, result,
        ))
        logger.info(f"[Hybrid] Turn {turn}: guideline updated ({len(result['output'])} chars)")
        return result["output"]

    def data_node(self, state: HybridState) -> dict:
        """Sample defining_instances data and produce structured observations for guideline update.

        Does NOT rewrite task_requirement_final. Instead calls hybrid_data_observation.jinja
        to surface findings, gaps, and pending signals, then feeds the raw observations into
        the guideline update. The task requirement is only updated at synthesis time.
        """
        env = self._env
        env_state = env.state
        turn = state.get("current_turn", 1)
        current_req = state.get("task_requirement_final", env_state.task.elevator_pitch)
        messages: list[Message] = []

        _, _, _, info = env.step(inspect_data(n_samples=_N_SAMPLES, split="non_defining_instances"))
        raw_samples = info.get("data_samples", [])
        logger.info(f"[Hybrid] Data turn {turn}: sampled {len(raw_samples)} rows from non_defining_instances")

        start = datetime.now(timezone.utc).isoformat()
        prompt = render_template(
            _DATA_OBSERVATION_TEMPLATE,
            initial_user_requirement=env_state.task.elevator_pitch,
            current_task_requirement=current_req,
            data_samples=raw_samples,
        )
        result = call_llm(self.model_name, prompt,
                          max_tokens=1024, temperature=self.temperature)
        end = datetime.now(timezone.utc).isoformat()

        observation = result["output"]
        logger.info(f"[Hybrid] Data turn {turn}: observations ({len(observation)} chars)")

        messages.append(_make_msg(
            start, end, "area_agent", "data_observation",
            prompt, "hybrid/hybrid_data_observation.jinja",
            "", observation, self.model_name, result,
        ))

        new_data_entry = {
            "turn": turn,
            "reflection": observation,
        }
        updated_data_interactions = list(state.get("data_interactions", [])) + [new_data_entry]

        updated_guideline = self._call_guideline_update(
            state, "data_interaction", new_data_entry, turn, messages,
        )

        return {
            "messages": messages,
            "data_interactions": updated_data_interactions,
            "guideline": updated_guideline,
        }

    def user_node(self, state: HybridState) -> dict:
        """Generate a clarification question and query the MIMIC user."""
        env = self._env
        env_state = env.state
        turn = state.get("current_turn", 1)
        messages: list[Message] = []

        # Build chat history from env interaction_history (matches UserInteractionAgent)
        chat_history = _build_chat_history(env_state.interaction_history)

        start = datetime.now(timezone.utc).isoformat()
        prompt = render_template(
            _USER_Q_TEMPLATE,
            chat_history=chat_history if chat_history else None,
            initial_requirement=env_state.task.elevator_pitch,
            current_task_requirement=state.get("task_requirement_final", env_state.task.elevator_pitch),
            guideline=state.get("guideline", ""),
            max_iterations=self.max_iterations,
            current_iteration=turn,
        )
        result = call_llm(self.model_name, prompt,
                          max_tokens=self.question_max_tokens, temperature=self.temperature)
        end = datetime.now(timezone.utc).isoformat()

        parsed = parse_json_output(result["output"])
        question = parsed.get("question") or ""
        if not question:
            # Fallback: try to extract question value from raw text via regex
            m = re.search(r'"question"\s*:\s*"((?:[^"\\]|\\.)*)"', result["output"], re.DOTALL)
            question = m.group(1).replace('\\"', '"') if m else result["output"]
        ambiguity = parsed.get("identified_ambiguity", "")
        logger.info(f"[Hybrid] User turn {turn}: {question[:80]}")

        messages.append(_make_msg(
            start, end, "area_agent", "ask_user",
            prompt, "user/user_interaction.jinja", ambiguity, question,
            self.model_name, result,
        ))

        # Query the MIMIC user via env
        env_start = datetime.now(timezone.utc).isoformat()
        obs, reward, done, info = env.step(ask_user(question))
        env_end = datetime.now(timezone.utc).isoformat()
        user_response = obs.get("last_response", "")

        messages.append({
            "start_time": env_start,
            "end_time": env_end,
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
            "env_response": {"obs": obs, "reward": reward, "done": done, "info": info},
        })
        messages.append({
            "start_time": env_start,
            "end_time": env_end,
            "role": "mimic_user_feedback",
            "action": "feedback",
            "input": question,
            "output": user_response,
            "thought": info.get("user_thought", ""),
            "grounding": info.get("user_grounding", ""),
        })

        new_user_entry = {
            "turn": turn,
            "question": question,
            "answer": user_response,
            "ambiguity": ambiguity,
        }
        updated_user_interactions = list(state.get("user_interactions", [])) + [new_user_entry]

        updated_guideline = self._call_guideline_update(
            state, "user_interaction", new_user_entry, turn, messages,
        )

        return {
            "messages": messages,
            "user_interactions": updated_user_interactions,
            "guideline": updated_guideline,
        }

    def synthesis_node(self, state: HybridState) -> dict:
        """Synthesize all collected evidence into the final task requirement."""
        env = self._env
        env_state = env.state
        messages: list[Message] = []

        start = datetime.now(timezone.utc).isoformat()
        prompt = render_template(
            _SYNTHESIS_TEMPLATE,
            initial_user_requirement=env_state.task.elevator_pitch,
            current_task_requirement=state.get("task_requirement_final", env_state.task.elevator_pitch),
            user_interactions=state.get("user_interactions", []),
            data_interactions=state.get("data_interactions", []),
        )
        result = call_llm(self.model_name, prompt,
                          max_tokens=self.max_tokens, temperature=self.temperature)
        end = datetime.now(timezone.utc).isoformat()
        final_req = result["output"]
        logger.info(f"[Hybrid] Synthesis complete ({len(final_req)} chars)")

        env.step(finish(final_req))

        messages.append(_make_msg(
            start, end, "area_agent", "finish",
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

    def _route(self, state: HybridState) -> str:
        if state.get("is_complete"):
            return "end"
        action = state.get("next_action", "reflection_synthesis")
        if action in ("data_interaction", "user_interaction", "reflection_synthesis"):
            return action
        return "reflection_synthesis"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_graph(self):
        """Compile and return the LangGraph StateGraph."""
        workflow = StateGraph(HybridState)
        workflow.add_node("router", self.router_node)
        workflow.add_node("data_interaction", self.data_node)
        workflow.add_node("user_interaction", self.user_node)
        workflow.add_node("reflection_synthesis", self.synthesis_node)

        workflow.set_entry_point("router")

        workflow.add_conditional_edges(
            "router",
            self._route,
            {
                "data_interaction": "data_interaction",
                "user_interaction": "user_interaction",
                "reflection_synthesis": "reflection_synthesis",
                "end": END,
            },
        )
        workflow.add_edge("data_interaction", "router")
        workflow.add_edge("user_interaction", "router")
        workflow.add_edge("reflection_synthesis", END)

        return workflow.compile()

    def run(self, env, task, user, initial_requirement: str | None = None) -> dict:
        """Run a complete hybrid episode and return the trajectory log.

        Args:
            env: AREAEnv instance (already initialised with evaluator).
            task: TaskInstance to solve.
            user: MimicUser instance (passive or persona-conditioned).
            initial_requirement: Optional pre-seeded task requirement to use as the
                starting current_task_requirement (e.g. from a prior zero-shot run).
                When provided the zero-shot node is skipped and the router sees this
                as its first current requirement instead of the raw elevator pitch.

        Returns:
            Trajectory log dict from env.get_trajectory_log(), augmented with
            the full agent message log under the key 'agent_messages'.
        """
        env.reset(task, user)
        self._env = env

        seed = initial_requirement or task.elevator_pitch
        initial_state: HybridState = {
            "messages": [],
            "is_complete": False,
            "current_turn": 0,
            "zero_shot_draft": seed,
            "task_requirement_final": seed,
            "next_action": "",
            "user_interactions": [],
            "data_interactions": [],
            "guideline": "",
            "router_history": [],
            "format_reflection_history": [],
        }

        app = self.build_graph()
        all_messages = []
        final_state: HybridState = {}
        for output in app.stream(initial_state):
            for _, state_update in output.items():
                if "messages" in state_update:
                    all_messages.extend(state_update["messages"])
                for key in ("format_reflection_history", "router_history",
                            "user_interactions", "data_interactions", "zero_shot_draft"):
                    if key in state_update:
                        final_state[key] = state_update[key]

        log = env.get_trajectory_log()
        log["agent_messages"] = [dict(m) for m in all_messages]
        log["zero_shot_draft"] = final_state.get("zero_shot_draft", "")
        log["format_reflection_history"] = final_state.get("format_reflection_history", [])
        log["router_history"] = final_state.get("router_history", [])
        log["user_interactions"] = final_state.get("user_interactions", [])
        log["data_interactions"] = final_state.get("data_interactions", [])
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
