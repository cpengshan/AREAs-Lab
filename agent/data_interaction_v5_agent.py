"""Data-interaction v5 agent for AUNUEnv.

Differences from v4:
- Each turn applies the zero_shot_with_samples_reason.jinja template using the
  previous task requirement as user_instruction (instead of the 3-step
  extract→reflect→rewrite pipeline used by v4).
- The prompt returns JSON with ``final_task_requirement`` and ``Modifications``
  (list of {Changed_task_requirement, reason, Evidence}).  Each turn the
  ``final_task_requirement`` becomes the active requirement for the next turn.
- Data is sampled from split="non_defining_instances" each turn.
- Phase 0 uses elevator_pitch as the initial task requirement.
- Loop repeats until max_turns is reached; the final output is submitted.

Output format:
  rewrite_history: [{turn, requirement, modifications, scores, counts,
                     subcategory_scores, cost}]
  intermediate_evals: [{turn, scores, counts, subcategory_scores, cost}]
  format_reflection_history: same list (alias for run_agent compat)
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone

from langgraph.graph import StateGraph, END

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_AGENT_DIR = os.path.dirname(__file__)
for _p in (_REPO_ROOT, _AGENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent_state import AgentState, Message
from AUNUEnv.aunu_env.config import AUNUEnvConfig
from AUNUEnv.aunu_env.env.actions import inspect_data, finish
from AUNUEnv.aunu_env.utils.llm import call_llm
from AUNUEnv.aunu_env.utils.jinja_utils import render_template

logger = logging.getLogger(__name__)

_PROMPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "prompts"))
_REWRITE_TEMPLATE = os.path.join(_PROMPTS_DIR, "zero_shot_with_samples_reason.jinja")

_N_SAMPLES = 5
_EVAL_EVERY = 1


class DataInteractionV5Agent:
    """LangGraph-based data-interaction agent (v5).

    Each turn: sample from all instances → call zero_shot_with_data_analysis_and_samples
    template with the previous requirement as user_instruction → the output becomes
    the active requirement for the next turn.

    Args:
        model_name: LLM model identifier.
        max_turns: Total refinement turns before final submission.
        temperature: Sampling temperature.
        max_tokens: Max output tokens per rewrite call.
    """

    def __init__(
        self,
        model_name: str,
        max_turns: int = 9,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ):
        self.model_name = model_name
        self.max_turns = max_turns
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._env = None
        self._evaluator = None
        self._rewrite_history: list = []

    @classmethod
    def from_config(cls, config: AUNUEnvConfig, max_turns: int = 9) -> "DataInteractionV5Agent":
        return cls(
            model_name=config.agent_model,
            max_turns=max_turns,
            temperature=config.effective_agent_temperature,
            max_tokens=config.max_tokens,
        )

    @classmethod
    def from_yaml(cls, yaml_path: str, max_turns: int = 9) -> "DataInteractionV5Agent":
        return cls.from_config(AUNUEnvConfig.from_yaml(yaml_path), max_turns=max_turns)

    def process(self, state: AgentState) -> dict:
        env = self._env
        env_state = env.state
        turn = state["current_turn"]
        messages: list[Message] = []

        # ── Phase 0: use elevator_pitch as initial task requirement ──
        if turn == 0:
            task = env_state.task
            initial_req = task.elevator_pitch
            return {
                "messages": [],
                "is_complete": False,
                "current_turn": 1,
                "zero_shot_draft": initial_req,
                "task_requirement_final": initial_req,
                "intermediate_evals": [],
                "rewrite_history": [],
            }

        # ── Phase 1+: sample non-defining instances → rewrite via zero_shot_with_samples_reason ──
        active_req = state["task_requirement_final"]
        rewrite_history = list(state.get("rewrite_history", []))
        intermediate_evals = list(state.get("intermediate_evals", []))

        task = env_state.task

        # Sample from non-defining instances only
        obs, _, _, info = env.step(inspect_data(n_samples=_N_SAMPLES, split="non_defining_instances"))
        raw_samples = info.get("data_samples", [])
        logger.info(f"[DataInteractionV5] Turn {turn}: sampled {len(raw_samples)} rows (split=non_defining_instances)")

        # Rewrite: use previous requirement as user_instruction
        next_turn = turn + 1
        is_last = next_turn > self.max_turns
        should_eval = (turn % _EVAL_EVERY == 0) or is_last

        start = datetime.now(timezone.utc).isoformat()
        rewrite_prompt = render_template(
            _REWRITE_TEMPLATE,
            user_instruction=active_req,
            data_samples=raw_samples,
        )
        rewrite_result = call_llm(
            self.model_name, rewrite_prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        end = datetime.now(timezone.utc).isoformat()
        raw_output = rewrite_result["output"]

        # Parse JSON output from zero_shot_with_samples_reason.jinja
        rewritten_req = raw_output
        modifications = []
        try:
            text = raw_output.strip()
            if text.startswith("```"):
                text = text.split("```", 2)[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.rsplit("```", 1)[0].strip()
            parsed = json.loads(text)
            rewritten_req = parsed.get("final_task_requirement", raw_output)
            modifications = parsed.get("Modifications", [])
        except (json.JSONDecodeError, ValueError):
            logger.warning(f"[DataInteractionV5] Turn {turn}: failed to parse JSON output; using raw text")

        logger.info(
            f"[DataInteractionV5] Turn {turn}: rewritten ({len(rewritten_req)} chars), "
            f"{len(modifications)} modification(s)"
        )

        messages.append({
            "start_time": start,
            "end_time": end,
            "role": "aunu_agent",
            "action": "rewrite",
            "input": rewrite_prompt,
            "prompt_template": "zero_shot_with_samples_reason.jinja",
            "identified_ambiguity": "",
            "output": raw_output,
            "final_task_requirement": rewritten_req,
            "modifications": modifications,
            "llm": self.model_name,
            "input_tokens": rewrite_result.get("input_tokens", 0),
            "output_tokens": rewrite_result.get("output_tokens", 0),
            "cost": rewrite_result.get("cost", 0.0),
        })

        if should_eval:
            eval_result = self._evaluator.evaluate(
                predicted=rewritten_req,
                gold=task.task_requirement,
                task_id=task.task_id,
            )
            snapshot = {
                "turn": turn,
                "requirement": rewritten_req,
                "modifications": modifications,
                "scores": eval_result.get("scores", {}),
                "counts": eval_result.get("counts", {}),
                "subcategory_scores": eval_result.get("subcategory_scores", {}),
                "cost": eval_result.get("cost", 0.0),
            }
            rewrite_history.append(snapshot)
            intermediate_evals.append({
                "turn": turn,
                "scores": eval_result.get("scores", {}),
                "counts": eval_result.get("counts", {}),
                "subcategory_scores": eval_result.get("subcategory_scores", {}),
                "cost": eval_result.get("cost", 0.0),
            })
            self._rewrite_history.append(snapshot)
            scores = snapshot["scores"]
            logger.info(
                f"[DataInteractionV5] Eval at turn {turn}: "
                f"Precision={scores.get('precision', 'N/A')}  "
                f"Recall={scores.get('recall', 'N/A')}  "
                f"F1={scores.get('f1', 'N/A')}"
            )

        if is_last:
            start = datetime.now(timezone.utc).isoformat()
            env.step(finish(rewritten_req))
            end = datetime.now(timezone.utc).isoformat()
            messages.append({
                "start_time": start,
                "end_time": end,
                "role": "aunu_agent",
                "action": "finish",
                "input": "",
                "prompt_template": "",
                "identified_ambiguity": "",
                "output": rewritten_req,
                "llm": self.model_name,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost": 0.0,
            })
            logger.info(
                f"[DataInteractionV5] Submitted final requirement ({len(rewritten_req)} chars)"
            )
            return {
                "messages": messages,
                "is_complete": True,
                "current_turn": next_turn,
                "zero_shot_draft": state["zero_shot_draft"],
                "task_requirement_final": rewritten_req,
                "intermediate_evals": intermediate_evals,
                "rewrite_history": rewrite_history,
            }

        return {
            "messages": messages,
            "is_complete": False,
            "current_turn": next_turn,
            "zero_shot_draft": state["zero_shot_draft"],
            "task_requirement_final": rewritten_req,
            "intermediate_evals": intermediate_evals,
            "rewrite_history": rewrite_history,
        }

    def _router(self, state: AgentState) -> str:
        return "end" if state["is_complete"] else "continue"

    def build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("aunu_agent", self.process)
        workflow.set_entry_point("aunu_agent")
        workflow.add_conditional_edges(
            "aunu_agent",
            self._router,
            {"continue": "aunu_agent", "end": END},
        )
        return workflow.compile()

    def run(self, env, task) -> dict:
        """Run a complete data-interaction-v5 episode and return the trajectory log."""
        from AUNUEnv.aunu_env.users import MimicUser
        user = MimicUser(model_name=self.model_name)
        env.reset(task, user)
        self._env = env
        self._evaluator = env.evaluator
        self._rewrite_history = []

        initial_state: AgentState = {
            "messages": [],
            "is_complete": False,
            "task_requirement_final": "",
            "current_turn": 0,
            "zero_shot_draft": "",
            "intermediate_evals": [],
            "rewrite_history": [],
        }

        app = self.build_graph()
        all_messages = []
        final_state = initial_state.copy()
        for output in app.stream(initial_state):
            for _, state_update in output.items():
                if "messages" in state_update:
                    all_messages.extend(state_update["messages"])
                for key in ("intermediate_evals", "rewrite_history"):
                    if key in state_update:
                        final_state[key] = state_update[key]

        log = env.get_trajectory_log()
        log["agent_messages"] = [dict(m) for m in all_messages]
        log["format_reflection_history"] = self._rewrite_history
        log["intermediate_evals"] = final_state.get("intermediate_evals", [])
        log["rewrite_history"] = final_state.get("rewrite_history", [])
        return log
