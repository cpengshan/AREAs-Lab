"""Data-interaction v3 agent for AUNUEnv.

Differences from v2:
- Phase 0 loads the zero-shot output (largest experiment ID) instead of
  generating a fresh zero-shot draft with the LLM.
- Reflection runs every turn, always against the *original* zero-shot
  requirement (the active task requirement is never updated between turns).
  Accumulated reflections are passed into each reflection call.
  No rewrite is run immediately after each reflection.
- Rewrite runs every `_REWRITE_EVERY` turns. The rewritten requirement is
  evaluated and recorded in `rewrite_history`, but the active task requirement
  stays as the zero-shot output — except at the final turn, where the latest
  rewrite becomes the submitted requirement.
- `rewrite_history` stores {turn, requirement, scores, counts} for every
  rewrite step.

Pipeline per episode:
  Phase 0 : load zero-shot output as initial active requirement (no LLM call)
  Phase 1 : for each turn 1…max_turns:
              a. sample _N_SAMPLES rows
              b. reflect (against zero-shot req + accumulated reflections)
              c. if turn % _REWRITE_EVERY == 0 or final turn:
                   rewrite → evaluate → append to rewrite_history
                   if final turn: set active req to rewrite → submit
  Phase 2 : submit (handled in the final iteration of Phase 1)
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
_REFLECT_TEMPLATE = os.path.join(_PROMPTS_DIR, "data/data_format_reflect.jinja")
_REWRITE_TEMPLATE = os.path.join(_PROMPTS_DIR, "data/data_format_rewrite.jinja")

_RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "results"))

_N_SAMPLES = 3
_REWRITE_EVERY = 3  # run rewrite every N reflection turns


def _dataset_dir_name(dataset_name: str) -> str:
    return dataset_name.replace("/", "_")


def _load_zero_shot_requirement(dataset_name: str, persona_id: int, task_num: int) -> str:
    """Return task_requirement_final from the zero-shot experiment with the largest ID.

    Args:
        dataset_name: Dataset identifier (e.g. 'alexfabbri/multi_news').
        persona_id: Integer persona ID (1-based).
        task_num: 1-based task number.

    Returns:
        The task_requirement_final string from zero-shot output.json.

    Raises:
        FileNotFoundError: If no zero-shot results exist for this dataset.
        KeyError: If the persona/task combination is absent from the output.
    """
    ds_dir = _dataset_dir_name(dataset_name)
    zs_dir = os.path.join(_RESULTS_DIR, ds_dir, "zero_shot")
    if not os.path.isdir(zs_dir):
        raise FileNotFoundError(f"No zero_shot results directory for dataset: {ds_dir}")

    experiments = sorted(
        [d for d in os.listdir(zs_dir) if d.startswith("Experiment")],
        key=lambda x: int(x[len("Experiment"):]),
    )
    if not experiments:
        raise FileNotFoundError(f"No zero_shot experiments found in {zs_dir}")

    latest_exp = experiments[-1]
    out_path = os.path.join(zs_dir, latest_exp, "output.json")
    with open(out_path) as f:
        output = json.load(f)

    persona_key = str(persona_id)
    task_key = f"task_{task_num}"
    try:
        req = output[persona_key][task_key]["task_requirement_final"]
    except KeyError:
        raise KeyError(
            f"Zero-shot output {out_path} has no entry for persona={persona_key}, task={task_key}"
        )

    logger.info(
        f"[DataInteractionV3] Loaded zero-shot requirement from {latest_exp} "
        f"(persona={persona_id}, task={task_num}, {len(req)} chars)"
    )
    return req


class DataInteractionV3Agent:
    """LangGraph-based data-interaction agent (v3).

    Starts from the saved zero-shot output, accumulates reflections over
    `max_turns` rounds, and rewrites the requirement every `_REWRITE_EVERY`
    turns without changing the active baseline until the final turn.

    Args:
        model_name: LLM model identifier.
        max_turns: Total reflection turns before final submission.
        temperature: Sampling temperature.
        max_tokens: Max output tokens for rewrite calls.
        reflect_max_tokens: Max output tokens for reflection calls.
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
        self._rewrite_history: list = []  # populated during process(), read in run()

    @classmethod
    def from_config(cls, config: AUNUEnvConfig, max_turns: int = 9) -> "DataInteractionV3Agent":
        return cls(
            model_name=config.agent_model,
            max_turns=max_turns,
            temperature=config.effective_agent_temperature,
            max_tokens=config.max_tokens,
        )

    @classmethod
    def from_yaml(cls, yaml_path: str, max_turns: int = 9) -> "DataInteractionV3Agent":
        return cls.from_config(AUNUEnvConfig.from_yaml(yaml_path), max_turns=max_turns)

    # ------------------------------------------------------------------
    # LangGraph node
    # ------------------------------------------------------------------

    def process(self, state: AgentState) -> dict:
        env = self._env
        env_state = env.state
        turn = state["current_turn"]
        messages: list[Message] = []

        # ── Phase 0: load zero-shot output as initial active requirement ──
        if turn == 0:
            task = env_state.task
            task_num = int(task.task_id.rsplit("_", 1)[-1]) + 1
            zero_shot_req = _load_zero_shot_requirement(
                task.dataset_name, task.persona_id, task_num
            )
            return {
                "messages": [],
                "is_complete": False,
                "current_turn": 1,
                "zero_shot_draft": zero_shot_req,
                # active baseline — never updated except at the final turn
                "task_requirement_final": zero_shot_req,
                "previous_reflections": [],
                "intermediate_evals": [],
                "rewrite_history": [],
            }

        # ── Phase 1: reflection turn ─────────────────────────────────────
        # The active requirement is always the zero-shot output.
        zero_shot_req = state["zero_shot_draft"]
        previous_reflections = state.get("previous_reflections", [])
        rewrite_history = list(state.get("rewrite_history", []))
        intermediate_evals = list(state.get("intermediate_evals", []))

        # Step 1a: sample raw data rows
        obs, _, _, info = env.step(inspect_data(n_samples=_N_SAMPLES))
        raw_samples = info.get("data_samples", [])
        logger.info(f"[DataInteractionV3] Turn {turn}: sampled {len(raw_samples)} rows")

        # Step 1b: reflect against zero-shot baseline + accumulated history
        start = datetime.now(timezone.utc).isoformat()
        reflect_prompt = render_template(
            _REFLECT_TEMPLATE,
            current_task_requirement=zero_shot_req,
            data_samples=raw_samples,
            previous_reflections=previous_reflections if previous_reflections else None,
        )
        reflect_result = call_llm(
            self.model_name, reflect_prompt,
            max_tokens=self.reflect_max_tokens,
            temperature=self.temperature,
        )
        end = datetime.now(timezone.utc).isoformat()
        reflection = parse_json_output(reflect_result["output"])
        format_gaps = reflection.get("format_gaps", reflect_result["output"])
        alignment_issues = reflection.get("alignment_issues", "")

        messages.append(_make_msg(
            start, end, "aunu_agent", "format_reflect",
            reflect_prompt, "data/data_format_reflect.jinja",
            alignment_issues, reflect_result["output"], self.model_name, reflect_result,
        ))
        logger.info(f"[DataInteractionV3] Turn {turn}: reflected on format")

        updated_reflections = list(previous_reflections) + [{
            "format_gaps": format_gaps,
            "alignment_issues": alignment_issues,
        }]

        next_turn = turn + 1
        is_last = next_turn > self.max_turns
        should_rewrite = (turn % _REWRITE_EVERY == 0) or is_last

        # Step 1c (conditional): rewrite + evaluate every _REWRITE_EVERY turns
        if should_rewrite:
            start = datetime.now(timezone.utc).isoformat()
            rewrite_prompt = render_template(
                _REWRITE_TEMPLATE,
                current_task_requirement=zero_shot_req,
                format_gaps=format_gaps,
                alignment_issues=alignment_issues,
                previous_reflections=previous_reflections if previous_reflections else None,
                iterations_remaining=self.max_turns - turn,
            )
            rewrite_result = call_llm(
                self.model_name, rewrite_prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            end = datetime.now(timezone.utc).isoformat()
            rewritten_req = rewrite_result["output"]
            logger.info(f"[DataInteractionV3] Turn {turn}: rewritten ({len(rewritten_req)} chars)")

            messages.append(_make_msg(
                start, end, "aunu_agent", "rewrite",
                rewrite_prompt, "data/data_format_rewrite.jinja",
                "", rewritten_req, self.model_name, rewrite_result,
            ))

            # Evaluate the rewritten requirement
            task = env_state.task
            eval_result = self._evaluator.evaluate(
                predicted=rewritten_req,
                gold=task.task_requirement,
                task_id=task.task_id,
            )
            snapshot = {
                "turn": turn,
                "requirement": rewritten_req,
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
            logger.info(
                f"[DataInteractionV3] Eval at turn {turn}: "
                f"F1={snapshot['scores'].get('f1', 'N/A')}"
            )

            if is_last:
                # Final turn: update active requirement to the latest rewrite and submit
                start = datetime.now(timezone.utc).isoformat()
                env.step(finish(rewritten_req))
                end = datetime.now(timezone.utc).isoformat()
                messages.append(_make_msg(
                    start, end, "aunu_agent", "finish",
                    "", "", "", rewritten_req, self.model_name, {},
                ))
                logger.info(
                    f"[DataInteractionV3] Submitted final requirement ({len(rewritten_req)} chars)"
                )
                return {
                    "messages": messages,
                    "is_complete": True,
                    "current_turn": next_turn,
                    "zero_shot_draft": zero_shot_req,
                    "task_requirement_final": rewritten_req,
                    "previous_reflections": updated_reflections,
                    "intermediate_evals": intermediate_evals,
                    "rewrite_history": rewrite_history,
                }

        # Non-final turn (with or without rewrite): keep zero-shot as active req
        if is_last:
            # Final turn with no rewrite scheduled — submit zero-shot as fallback
            start = datetime.now(timezone.utc).isoformat()
            submit_req = rewrite_history[-1]["requirement"] if rewrite_history else zero_shot_req
            env.step(finish(submit_req))
            end = datetime.now(timezone.utc).isoformat()
            messages.append(_make_msg(
                start, end, "aunu_agent", "finish",
                "", "", "", submit_req, self.model_name, {},
            ))
            logger.info(
                f"[DataInteractionV3] Submitted final requirement ({len(submit_req)} chars)"
            )
            return {
                "messages": messages,
                "is_complete": True,
                "current_turn": next_turn,
                "zero_shot_draft": zero_shot_req,
                "task_requirement_final": submit_req,
                "previous_reflections": updated_reflections,
                "intermediate_evals": intermediate_evals,
                "rewrite_history": rewrite_history,
            }

        return {
            "messages": messages,
            "is_complete": False,
            "current_turn": next_turn,
            "zero_shot_draft": zero_shot_req,
            "task_requirement_final": zero_shot_req,  # baseline unchanged
            "previous_reflections": updated_reflections,
            "intermediate_evals": intermediate_evals,
            "rewrite_history": rewrite_history,
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
        """Run a complete data-interaction-v3 episode and return the trajectory log.

        Args:
            env: AUNUEnv instance (already initialised with evaluator).
            task: TaskInstance to solve.

        Returns:
            Trajectory log dict from env.get_trajectory_log(), augmented with
            'agent_messages', 'format_reflection_history', 'intermediate_evals',
            and 'rewrite_history'.
        """
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
            "previous_reflections": [],
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
        log["format_reflection_history"] = self._rewrite_history  # named for run_agent compat
        log["intermediate_evals"] = final_state.get("intermediate_evals", [])
        log["rewrite_history"] = final_state.get("rewrite_history", [])
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
