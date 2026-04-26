"""Hybrid agent for AUNUEnv.

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

logger = logging.getLogger(__name__)

_PROMPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "prompts"))
_ROUTER_TEMPLATE     = os.path.join(_PROMPTS_DIR, "hybrid/hybrid_router_v2.jinja")
_EXECUTE_TEMPLATE    = os.path.join(_PROMPTS_DIR, "data/data_interaction_execute.jinja")
_REFLECT_TEMPLATE    = os.path.join(_PROMPTS_DIR, "data/data_interaction_reflect.jinja")
_USER_Q_TEMPLATE     = os.path.join(_PROMPTS_DIR, "user/user_interaction.jinja")
_SYNTHESIS_TEMPLATE  = os.path.join(_PROMPTS_DIR, "hybrid/hybrid_synthesis.jinja")
_GUIDELINE_TEMPLATE  = os.path.join(_PROMPTS_DIR, "hybrid/hybrid_guideline_update.jinja")

_N_SAMPLES = 3


class HybridState(TypedDict, total=False):
    """LangGraph state for the hybrid agent."""
    messages: Annotated[Sequence[Message], operator.add]
    is_complete: bool
    current_turn: int
    task_requirement_final: str   # evolves through data interaction; frozen after synthesis
    next_action: str              # set by router, consumed by conditional edge
    user_interactions: list       # [{turn, question, answer}]   — full history for synthesis
    data_interactions: list       # [{turn, reflection}]          — full history for synthesis
    guideline: str                # running structured summary passed to router each turn
    router_history: list          # [{turn, action, reason, target_gap}]


class HybridAgent:
    """Adaptive hybrid agent combining data and user interaction inside AUNUEnv.

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
        max_tokens: Max output tokens for requirement generation.
        execute_max_tokens: Max output tokens for per-sample execution.
        reflect_max_tokens: Max output tokens for the reflection step.
        question_max_tokens: Max output tokens for question generation.
        router_max_tokens: Max output tokens for the router decision.
    """

    def __init__(
        self,
        model_name: str,
        max_iterations: int = 6,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        execute_max_tokens: int = 1024,
        reflect_max_tokens: int = 1024,
        question_max_tokens: int = 512,
        router_max_tokens: int = 256,
    ):
        self.model_name = model_name
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.execute_max_tokens = execute_max_tokens
        self.reflect_max_tokens = reflect_max_tokens
        self.question_max_tokens = question_max_tokens
        self.router_max_tokens = router_max_tokens
        self._env = None

    @classmethod
    def from_config(cls, config: AUNUEnvConfig, max_iterations: int = 6) -> "HybridAgent":
        return cls(
            model_name=config.agent_model,
            max_iterations=max_iterations,
            temperature=config.effective_agent_temperature,
            max_tokens=config.max_tokens,
        )

    @classmethod
    def from_yaml(cls, yaml_path: str, max_iterations: int = 6) -> "HybridAgent":
        return cls.from_config(AUNUEnvConfig.from_yaml(yaml_path), max_iterations=max_iterations)

    # ------------------------------------------------------------------
    # LangGraph nodes
    # ------------------------------------------------------------------

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
            guideline=state.get("guideline", ""),
            current_iteration=turn,
            max_iterations=self.max_iterations,
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

        logger.info(f"[Hybrid] Turn {turn}: router → {action} | {reason}")

        messages.append(_make_msg(
            start, end, "aunu_agent", "router",
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
                          max_tokens=512, temperature=self.temperature)
        end = datetime.now(timezone.utc).isoformat()
        messages.append(_make_msg(
            start, end, "aunu_agent", "guideline_update",
            prompt, "hybrid/hybrid_guideline_update.jinja",
            "", result["output"], self.model_name, result,
        ))
        logger.info(f"[Hybrid] Turn {turn}: guideline updated ({len(result['output'])} chars)")
        return result["output"]

    def data_node(self, state: HybridState) -> dict:
        """Sample data, execute current requirement, reflect, update requirement."""
        env = self._env
        env_state = env.state
        turn = state.get("current_turn", 1)
        current_req = state.get("task_requirement_final", env_state.task.elevator_pitch)
        messages: list[Message] = []

        # Sample data
        obs, _, _, info = env.step(inspect_data(n_samples=_N_SAMPLES))
        raw_samples = info.get("data_samples", [])
        input_col = env_state.data_inspections[-1].get("input_col") if env_state.data_inspections else None

        # Execute current requirement on each sample
        sample_data = []
        for sample in raw_samples:
            input_text = sample.get(input_col, str(sample)) if input_col else str(sample)
            start = datetime.now(timezone.utc).isoformat()
            exec_prompt = render_template(
                _EXECUTE_TEMPLATE,
                task_requirement=current_req,
                input=input_text,
            )
            exec_result = call_llm(self.model_name, exec_prompt,
                                   max_tokens=self.execute_max_tokens,
                                   temperature=self.temperature)
            end = datetime.now(timezone.utc).isoformat()
            sample_data.append({
                "input": input_text,
                "predicted_output": exec_result["output"],
                "ground_truth_notes": "",
            })
            messages.append(_make_msg(
                start, end, "aunu_agent", "execute",
                exec_prompt, "data/data_interaction_execute.jinja",
                "", exec_result["output"], self.model_name, exec_result,
            ))

        # Reflect on execution outputs
        previous_reflections = [
            {"output_evaluation": d.get("reflection", ""), "potential_ambiguity": d.get("reflection", "")}
            for d in state.get("data_interactions", [])
        ]
        start = datetime.now(timezone.utc).isoformat()
        reflect_prompt = render_template(
            _REFLECT_TEMPLATE,
            current_task_requirement=current_req,
            sample_data=sample_data,
            previous_reflections=previous_reflections if previous_reflections else None,
        )
        reflect_result = call_llm(self.model_name, reflect_prompt,
                                  max_tokens=self.reflect_max_tokens,
                                  temperature=self.temperature)
        end = datetime.now(timezone.utc).isoformat()

        reflection_parsed = parse_json_output(reflect_result["output"])
        output_evaluation = reflection_parsed.get("output_evaluation", reflect_result["output"])
        potential_ambiguity = reflection_parsed.get("potential_ambiguity", "")
        combined_reflection = f"Output evaluation: {output_evaluation}\nPotential ambiguity: {potential_ambiguity}"

        messages.append(_make_msg(
            start, end, "aunu_agent", "reflect",
            reflect_prompt, "data/data_interaction_reflect.jinja",
            potential_ambiguity, reflect_result["output"], self.model_name, reflect_result,
        ))
        logger.info(f"[Hybrid] Data turn {turn}: reflected | ambiguity={potential_ambiguity[:80]}")

        new_data_entry = {
            "turn": turn,
            "reflection": combined_reflection,
            "output_evaluation": output_evaluation,
            "potential_ambiguity": potential_ambiguity,
        }
        updated_data_interactions = list(state.get("data_interactions", [])) + [new_data_entry]

        updated_guideline = self._call_guideline_update(
            state, "data_interaction", new_data_entry, turn, messages,
        )

        return {
            "messages": messages,
            "data_interactions": updated_data_interactions,
            "task_requirement_final": current_req,
            "guideline": updated_guideline,
        }

    def user_node(self, state: HybridState) -> dict:
        """Generate a clarification question and query the MIMIC user."""
        env = self._env
        env_state = env.state
        turn = state.get("current_turn", 1)
        messages: list[Message] = []

        # Build chat history from accumulated user interactions
        chat_history = [
            {"role": "agent", "output": e["question"]}
            for e in state.get("user_interactions", [])
        ] + [
            {"role": "user", "output": e["answer"]}
            for e in state.get("user_interactions", [])
        ]
        # Interleave properly
        raw_interactions = state.get("user_interactions", [])
        chat_history = []
        for e in raw_interactions:
            chat_history.append({"role": "agent", "output": e["question"]})
            chat_history.append({"role": "user", "output": e["answer"]})

        start = datetime.now(timezone.utc).isoformat()
        prompt = render_template(
            _USER_Q_TEMPLATE,
            chat_history=chat_history if chat_history else None,
            initial_requirement=env_state.task.elevator_pitch,
            max_iterations=self.max_iterations,
            current_iteration=turn,
        )
        result = call_llm(self.model_name, prompt,
                          max_tokens=self.question_max_tokens, temperature=self.temperature)
        end = datetime.now(timezone.utc).isoformat()

        parsed = parse_json_output(result["output"])
        question = parsed.get("question", result["output"])
        ambiguity = parsed.get("identified_ambiguity", "")
        logger.info(f"[Hybrid] User turn {turn}: {question[:80]}")

        messages.append(_make_msg(
            start, end, "aunu_agent", "ask_user",
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
            "prompt_template": "feedback_mimic_user_v2.jinja",
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

    def run(self, env, task, user) -> dict:
        """Run a complete hybrid episode and return the trajectory log.

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

        initial_state: HybridState = {
            "messages": [],
            "is_complete": False,
            "current_turn": 0,
            "task_requirement_final": task.elevator_pitch,
            "next_action": "",
            "user_interactions": [],
            "data_interactions": [],
            "guideline": "",
            "router_history": [],
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
