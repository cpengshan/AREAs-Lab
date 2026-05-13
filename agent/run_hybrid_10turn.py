#!/usr/bin/env python3
"""Post-process a hybrid experiment (from 'results copy') to produce a
truncated output.json in the same format as the hybrid pipeline.

For each task, the first --n_user_turns user interactions are used, along with
all data interactions that occurred up to (and including) that last user turn.
A final task requirement is synthesised via hybrid/hybrid_synthesis.jinja and
the result is written to results/{dataset_safe}/hybrid_10turn/ExperimentN/output.json.

Usage:
    python agent/run_hybrid_10turn.py \\
        --dataset alexfabbri/multi_news \\
        --source_aunu_model gpt-5.4 \\
        --agent_model claude-sonnet-4-6 \\
        [--n_user_turns 10] \\
        [--output_dir agent/results] \\
        [--exp_id 3]
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (_REPO_ROOT, _AGENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from AUNUEnv.aunu_env.utils.llm import call_llm
from AUNUEnv.aunu_env.utils.jinja_utils import render_template

logger = logging.getLogger(__name__)

_PROMPTS_DIR = os.path.join(_AGENT_DIR, "prompts")
_SYNTHESIS_TEMPLATE = os.path.join(_PROMPTS_DIR, "hybrid/hybrid_synthesis.jinja")

_RESULTS_COPY_DIR = os.path.join(_AGENT_DIR, "results copy")
_RESULTS_DIR = os.path.join(_AGENT_DIR, "results")


# ---------------------------------------------------------------------------
# Source experiment selection
# ---------------------------------------------------------------------------

def _find_source_experiment(dataset_safe: str, aunu_model: str, communication_habit: str = "neutral") -> str:
    """Return the path of the highest-N ExperimentN dir in 'results copy'
    whose output.json has args.aunu_model == *aunu_model* and
    args.communication_habit == *communication_habit*.
    """
    base = os.path.join(_RESULTS_COPY_DIR, dataset_safe, "hybrid")
    if not os.path.isdir(base):
        raise FileNotFoundError(
            f"Source hybrid directory not found: {base}"
        )

    best_id, best_dir = -1, None
    for entry in os.scandir(base):
        if not entry.is_dir() or not entry.name.startswith("Experiment"):
            continue
        try:
            exp_id = int(entry.name[len("Experiment"):])
        except ValueError:
            continue
        output_path = os.path.join(entry.path, "output.json")
        if not os.path.exists(output_path):
            continue
        try:
            with open(output_path) as f:
                data = json.load(f)
            exp_args = data.get("args", {})
            if exp_args.get("aunu_model") != aunu_model:
                continue
            if exp_args.get("communication_habit") != communication_habit:
                continue
        except Exception:
            continue
        if exp_id > best_id:
            best_id, best_dir = exp_id, entry.path

    if best_dir is None:
        raise FileNotFoundError(
            f"No hybrid experiment with aunu_model='{aunu_model}' and "
            f"communication_habit='{communication_habit}' found under {base}"
        )
    logger.info(f"Selected source: {best_dir}")
    return best_dir


# ---------------------------------------------------------------------------
# Interaction extraction
# ---------------------------------------------------------------------------

def _extract_interactions(task_data: dict, n_user_turns: int) -> dict:
    """Slice user_interactions to first *n_user_turns* entries, then include
    all data_interactions whose turn number is <= the last user turn taken.

    Returns:
        {
            "user_interactions":  list of interaction dicts (≤ n_user_turns),
            "data_interactions":  list of interaction dicts (up to last user turn),
            "n_data_turns":       int,
            "actual_user_turns":  int,
            "last_turn":          int,
        }
    """
    all_user = task_data.get("user_interactions", [])
    all_data = task_data.get("data_interactions", [])

    user_slice = all_user[:n_user_turns]
    actual_user_turns = len(user_slice)
    last_turn = user_slice[-1]["turn"] if user_slice else 0

    data_slice = [d for d in all_data if d["turn"] <= last_turn]

    return {
        "user_interactions": user_slice,
        "data_interactions": data_slice,
        "n_data_turns": len(data_slice),
        "actual_user_turns": actual_user_turns,
        "last_turn": last_turn,
    }


# ---------------------------------------------------------------------------
# Elevator pitch loader
# ---------------------------------------------------------------------------

def _load_elevator_pitch(dataset: str, persona_id: int, task_id: int) -> str:
    data_dir = os.path.join(_REPO_ROOT, "AUNUEnv", "data", "data_synthesized", dataset)
    path = os.path.join(data_dir, "synthesized_output_2.6.json")
    if not os.path.exists(path):
        logger.warning(f"Synthesized data not found at {path}; using empty string")
        return ""
    with open(path) as f:
        data = json.load(f)
    for task_raw in data.get(f"user_{persona_id}", {}).get("tasks_info", []):
        if task_raw.get("task_id") == task_id:
            return task_raw.get("elevator_pitch_summary", "")
    return ""


# ---------------------------------------------------------------------------
# Per-task processing
# ---------------------------------------------------------------------------

def _process_task(
    task_data: dict,
    dataset: str,
    persona_id: int,
    task_id: int,
    model_name: str,
    temperature: float,
    max_tokens: int,
    n_user_turns: int,
) -> dict:
    conv = _extract_interactions(task_data, n_user_turns)

    initial_user_requirement = (
        task_data.get("elevator_pitch")
        or _load_elevator_pitch(dataset, persona_id, task_id)
    )
    current_task_requirement = task_data.get("zero_shot_draft", "")

    logger.info(
        f"  persona={persona_id} task={task_id}: "
        f"user_turns={conv['actual_user_turns']} "
        f"data_turns={conv['n_data_turns']}"
    )

    start = datetime.now(timezone.utc).isoformat()
    prompt = render_template(
        _SYNTHESIS_TEMPLATE,
        initial_user_requirement=initial_user_requirement,
        current_task_requirement=current_task_requirement,
        user_interactions=conv["user_interactions"] if conv["user_interactions"] else None,
        data_interactions=conv["data_interactions"] if conv["data_interactions"] else None,
    )
    result = call_llm(model_name, prompt, max_tokens=max_tokens, temperature=temperature)
    end = datetime.now(timezone.utc).isoformat()

    final_req = result["output"]
    strategy_name = f"hybrid_{n_user_turns}turn"

    synthesis_msg = {
        "start_time": start,
        "end_time": end,
        "role": "aunu_agent",
        "action": "finish",
        "input": prompt,
        "prompt_template": "hybrid/hybrid_synthesis.jinja",
        "identified_ambiguity": "",
        "output": final_req,
        "llm": model_name,
        "input_tokens": result.get("input_tokens", 0),
        "output_tokens": result.get("output_tokens", 0),
        "cost": result.get("cost", 0.0),
    }

    return {
        "messages": task_data.get("messages", []) + [synthesis_msg],
        "is_complete": True,
        "task_requirement_final": final_req,
        "strategy": strategy_name,
        "persona": persona_id,
        "task_id": task_id,
        "dataset": dataset,
        "model": model_name,
        "input_type": task_data.get("input_type", "elevator_pitch_summary"),
        "ground_truth": task_data.get("ground_truth", ""),
        "cost": round(result.get("cost", 0.0), 6),
        "elevator_pitch": initial_user_requirement,
        "zero_shot_draft": current_task_requirement,
        "user_interactions": conv["user_interactions"],
        "data_interactions": conv["data_interactions"],
        "actual_user_turns_used": conv["actual_user_turns"],
        "actual_data_turns_used": conv["n_data_turns"],
        "start_time": start,
        "end_time": end,
    }


# ---------------------------------------------------------------------------
# Experiment-level processing
# ---------------------------------------------------------------------------

def _make_exp_dir(base_dir: str, exp_id: int | None) -> str:
    if exp_id is not None:
        exp_dir = os.path.join(base_dir, f"Experiment{exp_id}")
        os.makedirs(exp_dir, exist_ok=True)
        return exp_dir
    next_id = 1
    while os.path.exists(os.path.join(base_dir, f"Experiment{next_id}")):
        next_id += 1
    exp_dir = os.path.join(base_dir, f"Experiment{next_id}")
    os.makedirs(exp_dir)
    return exp_dir


def run_pipeline(
    dataset: str,
    source_aunu_model: str,
    agent_model: str,
    n_user_turns: int,
    temperature: float,
    max_tokens: int,
    output_dir: str,
    exp_id: int | None,
    communication_habit: str = "neutral",
) -> dict:
    dataset_safe = dataset.replace("/", "_")
    source_dir = _find_source_experiment(dataset_safe, source_aunu_model, communication_habit)

    with open(os.path.join(source_dir, "output.json")) as f:
        source_data = json.load(f)

    source_args = source_data.get("args", {})
    strategy_name = f"hybrid_{n_user_turns}turn"

    base_dir = os.path.join(output_dir, dataset_safe, strategy_name)
    exp_dir = _make_exp_dir(base_dir, exp_id)
    out_path = os.path.join(exp_dir, "output.json")

    if os.path.exists(out_path):
        with open(out_path) as f:
            output = json.load(f)
        logger.info(f"Resuming from existing {out_path}")
    else:
        output = {
            "args": {
                "strategy_aunu": strategy_name,
                "aunu_model": agent_model,
                "source_aunu_model": source_aunu_model,
                "source_exp_dir": source_dir,
                "mimic_model": source_args.get("mimic_model", ""),
                "evaluator_model": source_args.get("evaluator_model", ""),
                "persona": source_args.get("persona", []),
                "communication_habit": source_args.get("communication_habit"),
                "dataset": dataset,
                "input_type": source_args.get("input_type", "elevator_pitch_summary"),
                "n_user_turns": n_user_turns,
                "max_turns": source_args.get("max_turns"),
                "user_mode": source_args.get("user_mode", "passive"),
                "total_cost": 0.0,
            }
        }

    total_cost = output["args"].get("total_cost", 0.0)

    for persona_key, persona_val in source_data.items():
        if persona_key == "args" or not isinstance(persona_val, dict):
            continue

        persona_id = int(persona_key)
        if persona_key not in output:
            output[persona_key] = {}

        for habit_key, habit_val in persona_val.items():
            if not isinstance(habit_val, dict):
                continue
            if habit_key not in output[persona_key]:
                output[persona_key][habit_key] = {}

            for task_key, task_val in habit_val.items():
                if not isinstance(task_val, dict):
                    continue

                if task_key in output[persona_key][habit_key]:
                    logger.info(
                        f"Skipping persona={persona_id} {habit_key} {task_key} (already done)"
                    )
                    continue

                task_id = task_val.get("task_id")
                if task_id is None:
                    continue

                logger.info(f"Processing persona={persona_id} {habit_key} {task_key}")
                try:
                    task_result = _process_task(
                        task_data=task_val,
                        dataset=dataset,
                        persona_id=persona_id,
                        task_id=task_id,
                        model_name=agent_model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        n_user_turns=n_user_turns,
                    )
                    total_cost += task_result.get("cost", 0.0)
                    output[persona_key][habit_key][task_key] = task_result
                    logger.info(
                        f"  Done: cost={task_result.get('cost', 0):.4f}, "
                        f"user_turns={task_result.get('actual_user_turns_used')}, "
                        f"data_turns={task_result.get('actual_data_turns_used')}"
                    )
                except Exception as e:
                    logger.error(f"  FAILED: {e}", exc_info=True)
                    output[persona_key][habit_key][task_key] = {"error": str(e)}

                output["args"]["total_cost"] = round(total_cost, 6)
                with open(out_path, "w") as f:
                    json.dump(output, f, indent=2, default=str)

    output["args"]["total_cost"] = round(total_cost, 6)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(f"Saved → {out_path} (total_cost={total_cost:.4f})")
    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Synthesize final task requirements from the first N user turns of a "
            "hybrid experiment (from 'results copy'), including all data interactions "
            "that occurred within those turns."
        )
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Dataset name, e.g. 'alexfabbri/multi_news'",
    )
    parser.add_argument(
        "--source_aunu_model", required=True,
        help="aunu_model value in the source experiment's args (used to select experiment)",
    )
    parser.add_argument(
        "--agent_model", required=True,
        help="LLM model to use for synthesizing the final task requirement",
    )
    parser.add_argument(
        "--n_user_turns", type=int, default=10,
        help="Number of user interaction turns to use (default: 10); "
             "all data interactions up to the last user turn are also included",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.4,
        help="Sampling temperature (default: 0.4)",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=4096,
        help="Max output tokens per requirement (default: 4096)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Base results directory (default: agent/results)",
    )
    parser.add_argument(
        "--exp_id", type=int, default=None,
        help="Resume or target a specific ExperimentN (default: auto-increment)",
    )
    parser.add_argument(
        "--log_file_path", type=str, default=None,
        help="Path to log file (auto-generated if omitted)",
    )
    parser.add_argument(
        "--communication_habit", type=str, default="neutral",
        help="communication_habit value in the source experiment's args (default: neutral)",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = args.output_dir or _RESULTS_DIR
    dataset_safe = args.dataset.replace("/", "_")
    strategy_name = f"hybrid_{args.n_user_turns}turn"

    log_dir = os.path.join(_AGENT_DIR, "logs", strategy_name)
    os.makedirs(log_dir, exist_ok=True)
    log_path = args.log_file_path or os.path.join(
        log_dir, f"{dataset_safe}_{args.agent_model.replace('/', '_')}.log"
    )
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
    )

    logger.info(
        f"Starting {strategy_name} pipeline | dataset={args.dataset} | "
        f"source_model={args.source_aunu_model} | agent_model={args.agent_model}"
    )

    run_pipeline(
        dataset=args.dataset,
        source_aunu_model=args.source_aunu_model,
        agent_model=args.agent_model,
        n_user_turns=args.n_user_turns,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        output_dir=output_dir,
        exp_id=args.exp_id,
        communication_habit=args.communication_habit,
    )

    logger.info("Done.")


if __name__ == "__main__":
    main()
