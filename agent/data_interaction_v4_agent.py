"""Data-interaction v4 agent for AUNUEnv.

Differences from v3:
- Phase 0 loads the zero-shot output as the initial task requirement (same as v3).
- Each turn:
    a. Sample all defining_instances via inspect_data.
    b. Run sampled_data_analysis.jinja to extract top-5 data features (LLM call).
    c. Run data_feature_reflect.jinja with (task_requirement, extracted_features)
       → JSON {potential_ambiguity, reason}.
    d. Run data_feature_rewrite.jinja with (task_requirement, ambiguity, reason,
       previous_reflections) → updated task requirement.
    e. Evaluate the rewritten requirement.
- The rewritten requirement from the previous turn becomes the active task
  requirement for the next turn (unlike v3 which keeps the zero-shot baseline).
- Loop repeats until max_turns is reached; the final rewrite is submitted.

Output format matches data_interaction_v3:
  rewrite_history: [{turn, requirement, scores, counts, subcategory_scores, cost}]
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
from AUNUEnv.aunu_env.utils.json_utils import parse_json_output

logger = logging.getLogger(__name__)

_PROMPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "prompts"))
_EXTRACT_TEMPLATE = os.path.join(_PROMPTS_DIR, "sampled_data_analysis.jinja")
_REFLECT_TEMPLATE = os.path.join(_PROMPTS_DIR, "data/data_feature_reflect.jinja")
_REWRITE_TEMPLATE = os.path.join(_PROMPTS_DIR, "data/data_feature_rewrite.jinja")

_RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "results"))
_DATA_SYNTHESIZED_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "AUNUEnv", "data", "data_synthesized")
)

# Use a large number so inspect_data returns all defining instances.
_N_SAMPLES = 9999
_EVAL_EVERY = 3  # evaluate every N rewrite turns


def _dataset_dir_name(dataset_name: str) -> str:
    return dataset_name.replace("/", "_")


def _load_data_analysis(dataset_name: str) -> dict:
    """Load data_analysis.json for the given dataset from the synthesized data dir."""
    path = os.path.join(_DATA_SYNTHESIZED_DIR, dataset_name, "data_analysis.json")
    with open(path) as f:
        return json.load(f)


def _load_zero_shot_requirement(dataset_name: str, persona_id: int, task_num: int) -> str:
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

    out_path = os.path.join(zs_dir, experiments[-1], "output.json")
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
        f"[DataInteractionV4] Loaded zero-shot requirement from {experiments[-1]} "
        f"(persona={persona_id}, task={task_num}, {len(req)} chars)"
    )
    return req


class DataInteractionV4Agent:
    """LangGraph-based data-interaction agent (v4).

    Each turn: sample all data → extract features → reflect → rewrite.
    The rewritten requirement becomes the active requirement for the next turn.

    Args:
        model_name: LLM model identifier.
        max_turns: Total refinement turns before final submission.
        temperature: Sampling temperature.
        max_tokens: Max output tokens for rewrite calls.
        reflect_max_tokens: Max output tokens for reflection calls.
        extract_max_tokens: Max output tokens for feature extraction calls.
    """

    def __init__(
        self,
        model_name: str,
        max_turns: int = 9,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        reflect_max_tokens: int = 1024,
        extract_max_tokens: int = 2048,
    ):
        self.model_name = model_name
        self.max_turns = max_turns
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reflect_max_tokens = reflect_max_tokens
        self.extract_max_tokens = extract_max_tokens
        self._env = None
        self._evaluator = None
        self._rewrite_history: list = []

    @classmethod
    def from_config(cls, config: AUNUEnvConfig, max_turns: int = 9) -> "DataInteractionV4Agent":
        return cls(
            model_name=config.agent_model,
            max_turns=max_turns,
            temperature=config.effective_agent_temperature,
            max_tokens=config.max_tokens,
        )

    @classmethod
    def from_yaml(cls, yaml_path: str, max_turns: int = 9) -> "DataInteractionV4Agent":
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
                "task_requirement_final": zero_shot_req,
                "previous_reflections": [],
                "intermediate_evals": [],
                "rewrite_history": [],
            }

        # ── Phase 1: extract → reflect → rewrite ────────────────────────
        active_req = state["task_requirement_final"]
        previous_reflections = state.get("previous_reflections", [])
        rewrite_history = list(state.get("rewrite_history", []))
        intermediate_evals = list(state.get("intermediate_evals", []))

        task = env_state.task

        # Step a: sample all defining instances
        obs, _, _, info = env.step(inspect_data(n_samples=_N_SAMPLES))
        raw_samples = info.get("data_samples", [])
        logger.info(f"[DataInteractionV4] Turn {turn}: sampled {len(raw_samples)} rows")

        # Step b: extract data features
        data_analysis = _load_data_analysis(task.dataset_name)
        start = datetime.now(timezone.utc).isoformat()
        extract_prompt = render_template(
            _EXTRACT_TEMPLATE,
            dataset_name=task.dataset_name,
            data_analysis=data_analysis,
            user_instruction=task.elevator_pitch,
            sampled_data=raw_samples,
        )
        extract_result = call_llm(
            self.model_name, extract_prompt,
            max_tokens=self.extract_max_tokens,
            temperature=self.temperature,
        )
        end = datetime.now(timezone.utc).isoformat()
        extracted_features = parse_json_output(extract_result["output"])
        if not isinstance(extracted_features, list):
            extracted_features = []
        messages.append(_make_msg(
            start, end, "aunu_agent", "extract_features",
            extract_prompt, "sampled_data_analysis.jinja",
            "", extract_result["output"], self.model_name, extract_result,
        ))
        logger.info(
            f"[DataInteractionV4] Turn {turn}: extracted {len(extracted_features)} features"
        )

        # Step c: reflect — identify potential ambiguity
        start = datetime.now(timezone.utc).isoformat()
        reflect_prompt = render_template(
            _REFLECT_TEMPLATE,
            current_task_requirement=active_req,
            extracted_features=extracted_features,
            previous_reflections=previous_reflections if previous_reflections else None,
        )
        reflect_result = call_llm(
            self.model_name, reflect_prompt,
            max_tokens=self.reflect_max_tokens,
            temperature=self.temperature,
        )
        end = datetime.now(timezone.utc).isoformat()
        reflection = parse_json_output(reflect_result["output"])
        potential_ambiguity = reflection.get("potential_ambiguity", reflect_result["output"])
        reason = reflection.get("reason", "")

        messages.append(_make_msg(
            start, end, "aunu_agent", "feature_reflect",
            reflect_prompt, "data/data_feature_reflect.jinja",
            potential_ambiguity, reflect_result["output"], self.model_name, reflect_result,
        ))
        logger.info(f"[DataInteractionV4] Turn {turn}: reflected, ambiguity identified")

        updated_reflections = list(previous_reflections) + [{
            "potential_ambiguity": potential_ambiguity,
            "reason": reason,
        }]

        # Step d: rewrite using identified ambiguity
        next_turn = turn + 1
        is_last = next_turn > self.max_turns
        should_eval = (turn % _EVAL_EVERY == 0) or is_last

        start = datetime.now(timezone.utc).isoformat()
        rewrite_prompt = render_template(
            _REWRITE_TEMPLATE,
            current_task_requirement=active_req,
            potential_ambiguity=potential_ambiguity,
            reason=reason,
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
        logger.info(f"[DataInteractionV4] Turn {turn}: rewritten ({len(rewritten_req)} chars)")

        messages.append(_make_msg(
            start, end, "aunu_agent", "rewrite",
            rewrite_prompt, "data/data_feature_rewrite.jinja",
            "", rewritten_req, self.model_name, rewrite_result,
        ))

        # Step e: evaluate every _EVAL_EVERY turns and at the final turn
        if should_eval:
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
            scores = snapshot["scores"]
            logger.info(
                f"[DataInteractionV4] Eval at turn {turn}: "
                f"Precision={scores.get('precision', 'N/A')}  "
                f"Recall={scores.get('recall', 'N/A')}  "
                f"F1={scores.get('f1', 'N/A')}"
            )

        if is_last:
            start = datetime.now(timezone.utc).isoformat()
            env.step(finish(rewritten_req))
            end = datetime.now(timezone.utc).isoformat()
            messages.append(_make_msg(
                start, end, "aunu_agent", "finish",
                "", "", "", rewritten_req, self.model_name, {},
            ))
            logger.info(
                f"[DataInteractionV4] Submitted final requirement ({len(rewritten_req)} chars)"
            )
            return {
                "messages": messages,
                "is_complete": True,
                "current_turn": next_turn,
                "zero_shot_draft": state["zero_shot_draft"],
                "task_requirement_final": rewritten_req,
                "previous_reflections": updated_reflections,
                "intermediate_evals": intermediate_evals,
                "rewrite_history": rewrite_history,
            }

        # Non-final: rewritten req becomes active for next turn
        return {
            "messages": messages,
            "is_complete": False,
            "current_turn": next_turn,
            "zero_shot_draft": state["zero_shot_draft"],
            "task_requirement_final": rewritten_req,
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
        """Run a complete data-interaction-v4 episode and return the trajectory log."""
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
        log["format_reflection_history"] = self._rewrite_history
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
