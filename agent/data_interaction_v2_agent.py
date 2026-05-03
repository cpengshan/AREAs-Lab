"""Data-interaction v2 agent for AUNUEnv.

Iteratively refines a task requirement by reflecting on the *format and
structure* of real data samples — no LLM execution on individual rows.

Pipeline per turn:
  1. Random-sample N rows via env.inspect_data()
  2. Reflect on format/schema alignment between the requirement and the data
  3. Rewrite the requirement to fix identified gaps

LangGraph graph:
  [aunu_agent] ──(continue)──▶ [aunu_agent]
               ──(end)──▶ END

State machine phases:
  Phase 0: zero-shot-with-data-analysis draft (from elevator pitch + dataset schema)
  Phase 1: format-reflection loop (up to max_turns)
  Phase 2: submit final requirement
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
from AUNUEnv.aunu_env.utils.json_utils import parse_json_output

logger = logging.getLogger(__name__)

_PROMPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "prompts"))
_ZERO_SHOT_WITH_DA_TEMPLATE = os.path.join(_PROMPTS_DIR, "zero_shot_with_data_analysis.jinja")
_REFLECT_TEMPLATE    = os.path.join(_PROMPTS_DIR, "data/data_format_reflect.jinja")
_REWRITE_TEMPLATE    = os.path.join(_PROMPTS_DIR, "data/data_format_rewrite.jinja")

_DATA_SYNTHESIZED_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "AUNUEnv", "data", "data_synthesized")
)


def _load_data_analysis(dataset_name: str) -> dict:
    path = os.path.join(_DATA_SYNTHESIZED_ROOT, dataset_name, "data_analysis.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"data_analysis.json not found: {path}")
    with open(path) as f:
        return json.load(f)

_N_SAMPLES = 3
_EVAL_EVERY_N_TURNS = 3


class DataInteractionV2Agent:
    """LangGraph-based data-interaction agent (v2) that operates inside AUNUEnv.

    Starts from the zero-shot draft, then runs up to `max_turns` format-
    reflection rounds. Each round samples raw data rows, reflects on how
    well the current requirement aligns with the data's actual structure and
    schema, then rewrites the requirement to close any gaps. No LLM execution
    on individual samples is performed.

    Args:
        model_name: LLM model identifier.
        max_turns: Maximum format-reflection rounds before final submission.
        temperature: Sampling temperature.
        max_tokens: Maximum output tokens for requirement generation.
        reflect_max_tokens: Maximum output tokens for the reflection step.
    """

    def __init__(
        self,
        model_name: str,
        max_turns: int = 9,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        reflect_max_tokens: int = 1024,
    ):
        self.model_name = model_name
        self.max_turns = max_turns
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reflect_max_tokens = reflect_max_tokens
        self._env = None
        self._evaluator = None
        self._reflection_history: list = []  # populated during process(), read in run()

    @classmethod
    def from_config(cls, config: AUNUEnvConfig, max_turns: int = 9) -> "DataInteractionV2Agent":
        return cls(
            model_name=config.agent_model,
            max_turns=max_turns,
            temperature=config.effective_agent_temperature,
            max_tokens=config.max_tokens,
        )

    @classmethod
    def from_yaml(cls, yaml_path: str, max_turns: int = 9) -> "DataInteractionV2Agent":
        return cls.from_config(AUNUEnvConfig.from_yaml(yaml_path), max_turns=max_turns)

    # ------------------------------------------------------------------
    # LangGraph node
    # ------------------------------------------------------------------

    def process(self, state: AgentState) -> dict:
        env = self._env
        env_state = env.state
        turn = state["current_turn"]
        user_request = env_state.task.elevator_pitch
        messages: list[Message] = []

        # ── Phase 0: generate initial draft using data-analysis grounding ─
        if turn == 0:
            dataset_name = env_state.task.dataset_name
            data_analysis = _load_data_analysis(dataset_name)
            start = datetime.now(timezone.utc).isoformat()
            prompt = render_template(
                _ZERO_SHOT_WITH_DA_TEMPLATE,
                user_instruction=user_request,
                data_analysis=data_analysis,
            )
            result = call_llm(self.model_name, prompt,
                              max_tokens=self.max_tokens, temperature=self.temperature)
            end = datetime.now(timezone.utc).isoformat()
            draft = result["output"]
            logger.info(f"[DataInteractionV2] Zero-shot-with-data-analysis draft ({len(draft)} chars)")

            messages.append(_make_msg(
                start, end, "aunu_agent", "zero_shot_with_data_analysis_draft",
                prompt, "zero_shot_with_data_analysis.jinja", "", draft, self.model_name, result,
            ))
            return {
                "messages": messages,
                "is_complete": False,
                "current_turn": 1,
                "zero_shot_draft": draft,
                "task_requirement_final": draft,
            }

        # ── Phase 1: format-reflection turn ────────────────────────────
        current_req = state["task_requirement_final"]
        previous_reflections = state.get("previous_reflections", [])

        # Step 1a: sample raw data rows from the env
        obs, _, _, info = env.step(inspect_data(n_samples=_N_SAMPLES))
        raw_samples = info.get("data_samples", [])
        logger.info(f"[DataInteractionV2] Turn {turn}: sampled {len(raw_samples)} rows")

        # Step 1b: reflect on format/schema alignment
        start = datetime.now(timezone.utc).isoformat()
        reflect_prompt = render_template(
            _REFLECT_TEMPLATE,
            current_task_requirement=current_req,
            data_samples=raw_samples,
            previous_reflections=previous_reflections if previous_reflections else None,
        )
        reflect_result = call_llm(self.model_name, reflect_prompt,
                                  max_tokens=self.reflect_max_tokens,
                                  temperature=self.temperature)
        end = datetime.now(timezone.utc).isoformat()
        reflection = parse_json_output(reflect_result["output"])
        format_gaps = reflection.get("format_gaps", reflect_result["output"])
        alignment_issues = reflection.get("alignment_issues", "")

        messages.append(_make_msg(
            start, end, "aunu_agent", "format_reflect",
            reflect_prompt, "data/data_format_reflect.jinja",
            alignment_issues, reflect_result["output"], self.model_name, reflect_result,
        ))
        logger.info(f"[DataInteractionV2] Turn {turn}: reflected on format")

        next_turn = turn + 1
        is_last = next_turn > self.max_turns

        updated_reflections = list(previous_reflections) + [{
            "format_gaps": format_gaps,
            "alignment_issues": alignment_issues,
        }]

        # Step 1c: rewrite requirement based on reflection
        start = datetime.now(timezone.utc).isoformat()
        rewrite_prompt = render_template(
            _REWRITE_TEMPLATE,
            current_task_requirement=current_req,
            format_gaps=format_gaps,
            alignment_issues=alignment_issues,
            previous_reflections=previous_reflections if previous_reflections else None,
            iterations_remaining=self.max_turns - turn,
        )
        rewrite_result = call_llm(self.model_name, rewrite_prompt,
                                  max_tokens=self.max_tokens, temperature=self.temperature)
        end = datetime.now(timezone.utc).isoformat()
        refined_req = rewrite_result["output"]
        logger.info(f"[DataInteractionV2] Turn {turn}: rewritten ({len(refined_req)} chars)")

        messages.append(_make_msg(
            start, end, "aunu_agent", "rewrite",
            rewrite_prompt, "data/data_format_rewrite.jinja",
            "", refined_req, self.model_name, rewrite_result,
        ))

        self._reflection_history.append({
            "turn": turn,
            "format_gaps": format_gaps,
            "alignment_issues": alignment_issues,
            "rewritten_requirement": refined_req,
        })

        # ── Periodic evaluation every _EVAL_EVERY_N_TURNS turns ────────
        intermediate_evals = list(state.get("intermediate_evals", []))
        if next_turn % _EVAL_EVERY_N_TURNS == 0:
            task = env_state.task
            eval_result = self._evaluator.evaluate(
                predicted=refined_req,
                gold=task.task_requirement,
                task_id=task.task_id,
            )
            snapshot = {
                "turn": next_turn,
                "scores": eval_result.get("scores", {}),
                "counts": eval_result.get("counts", {}),
                "subcategory_scores": eval_result.get("subcategory_scores", {}),
                "cost": eval_result.get("cost", 0.0),
            }
            intermediate_evals.append(snapshot)
            logger.info(
                f"[DataInteractionV2] Eval at turn {next_turn}: "
                f"F1={snapshot['scores'].get('f1', 'N/A')}"
            )

        if not is_last:
            return {
                "messages": messages,
                "is_complete": False,
                "current_turn": next_turn,
                "zero_shot_draft": state["zero_shot_draft"],
                "task_requirement_final": refined_req,
                "previous_reflections": updated_reflections,
                "intermediate_evals": intermediate_evals,
            }

        # ── Phase 2: submit final requirement ──────────────────────────
        start = datetime.now(timezone.utc).isoformat()
        env.step(finish(refined_req))
        end = datetime.now(timezone.utc).isoformat()

        messages.append(_make_msg(
            start, end, "aunu_agent", "finish",
            "", "", "", refined_req, self.model_name, {},
        ))
        logger.info(f"[DataInteractionV2] Submitted final requirement ({len(refined_req)} chars)")

        return {
            "messages": messages,
            "is_complete": True,
            "current_turn": next_turn,
            "zero_shot_draft": state["zero_shot_draft"],
            "task_requirement_final": refined_req,
            "previous_reflections": updated_reflections,
            "intermediate_evals": intermediate_evals,
        }

    # ------------------------------------------------------------------
    # Router
    # ------------------------------------------------------------------

    def _router(self, state: AgentState) -> str:
        return "end" if state["is_complete"] else "continue"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
        """Run a complete data-interaction-v2 episode and return the trajectory log.

        Args:
            env: AUNUEnv instance (already initialised with evaluator).
            task: TaskInstance to solve.

        Returns:
            Trajectory log dict from env.get_trajectory_log(), augmented with
            the full agent message log under the key 'agent_messages', plus
            'format_reflection_history' and 'intermediate_evals'.
        """
        from AUNUEnv.aunu_env.users import MimicUser
        user = MimicUser(model_name=self.model_name)
        env.reset(task, user)
        self._env = env
        self._evaluator = env.evaluator
        self._reflection_history = []

        initial_state: AgentState = {
            "messages": [],
            "is_complete": False,
            "task_requirement_final": "",
            "current_turn": 0,
            "zero_shot_draft": "",
            "previous_reflections": [],
            "intermediate_evals": [],
        }

        app = self.build_graph()
        all_messages = []
        final_state = initial_state.copy()
        for output in app.stream(initial_state):
            for _, state_update in output.items():
                if "messages" in state_update:
                    all_messages.extend(state_update["messages"])
                if "intermediate_evals" in state_update:
                    final_state["intermediate_evals"] = state_update["intermediate_evals"]

        log = env.get_trajectory_log()
        log["agent_messages"] = [dict(m) for m in all_messages]
        log["format_reflection_history"] = self._reflection_history
        log["intermediate_evals"] = final_state.get("intermediate_evals", [])
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
