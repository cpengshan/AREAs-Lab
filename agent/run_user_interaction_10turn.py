#!/usr/bin/env python3
"""Post-process a user_interaction experiment (from 'results copy') to produce a
10-turn-truncated output.json in the same format as the user_interaction pipeline.

For each task in the source experiment, the first --n_turns turns of the
clarification conversation are used to synthesize a final task requirement via
task_requirement_prediction.jinja.  The output is written to a new
ExperimentN directory under results/{dataset_safe}/user_interaction_10turn/.

Usage:
    python agent/run_user_interaction_10turn.py \\
        --dataset alexfabbri/multi_news \\
        --source_aunu_model claude-sonnet-4-6 \\
        --agent_model claude-sonnet-4-6 \\
        [--n_turns 10] \\
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
_PREDICTION_TEMPLATE = os.path.join(_PROMPTS_DIR, "user/task_requirement_prediction.jinja")

# Where source experiments live
_RESULTS_COPY_DIR = os.path.join(_AGENT_DIR, "results copy")
# Where new experiments are written
_RESULTS_DIR = os.path.join(_AGENT_DIR, "results")


# ---------------------------------------------------------------------------
# Source experiment selection
# ---------------------------------------------------------------------------

def _find_source_experiment(dataset_safe: str, aunu_model: str) -> str:
    """Return the path of the ExperimentN dir in 'results copy' whose
    output.json args.aunu_model matches *aunu_model* and
    args.communication_habit == 'neutral'.  Prefers the highest N.
    """
    base = os.path.join(_RESULTS_COPY_DIR, dataset_safe, "user_interaction")
    if not os.path.isdir(base):
        raise FileNotFoundError(
            f"Source user_interaction directory not found: {base}"
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
            if exp_args.get("communication_habit") != "neutral":
                continue
        except Exception:
            continue
        if exp_id > best_id:
            best_id, best_dir = exp_id, entry.path

    if best_dir is None:
        raise FileNotFoundError(
            f"No user_interaction experiment with aunu_model='{aunu_model}' "
            f"and communication_habit='neutral' found under {base}"
        )
    logger.info(f"Selected source: {best_dir}")
    return best_dir


# ---------------------------------------------------------------------------
# Conversation extraction (mirrors extract_turn_requirements.py)
# ---------------------------------------------------------------------------

def _extract_conversation(messages: list, n_turns: int) -> dict:
    """Extract zero_shot_draft and the first *n_turns* agent/user exchanges.

    Returns:
        {
            "zero_shot_draft": str,
            "chat_history": [{"role": "agent"|"user", "output": str}, ...],
            "truncated_messages": list[dict],   # raw messages up to turn n
        }
    """
    zero_shot_draft = ""
    agent_turns: list[str] = []
    user_turns: list[str] = []
    truncated_messages: list[dict] = []

    for msg in messages:
        role = msg.get("role", "")
        action = msg.get("action", "")
        output = msg.get("output", "")

        if role == "aunu_agent" and action == "zero_shot_draft":
            zero_shot_draft = output
            truncated_messages.append(msg)

        elif role == "aunu_agent" and action == "ask_user":
            if len(user_turns) >= n_turns:
                # Already collected enough turns; stop adding agent questions.
                break
            agent_turns.append(output)
            truncated_messages.append(msg)

        elif role == "mimic_user" and action == "respond":
            user_turns.append(output)
            truncated_messages.append(msg)
            # Also include the mimic_user_feedback message that immediately follows
            if len(user_turns) >= n_turns:
                break  # We've reached the requested turn count

        elif role == "mimic_user_feedback" and action == "feedback":
            # Include feedback messages that belong to collected turns
            if len(user_turns) <= n_turns:
                truncated_messages.append(msg)

    chat_history = []
    for i in range(min(len(agent_turns), len(user_turns))):
        chat_history.append({"role": "agent", "output": agent_turns[i]})
        chat_history.append({"role": "user", "output": user_turns[i]})

    return {
        "zero_shot_draft": zero_shot_draft,
        "chat_history": chat_history,
        "truncated_messages": truncated_messages,
        "actual_turns": len(user_turns),
    }


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
    n_turns: int,
) -> dict:
    """Synthesize a final requirement from the first *n_turns* turns.

    Returns a task result dict in the same format as user_interaction output.json.
    """
    messages = task_data.get("messages", [])
    conv = _extract_conversation(messages, n_turns)

    initial_task_requirement = _load_elevator_pitch(dataset, persona_id, task_id)

    logger.info(
        f"  persona={persona_id} task={task_id}: "
        f"generating requirement from {conv['actual_turns']} turns"
    )

    start = datetime.now(timezone.utc).isoformat()
    prompt = render_template(
        _PREDICTION_TEMPLATE,
        initial_task_requirement=initial_task_requirement,
        zero_shot_draft=conv["zero_shot_draft"] if conv["zero_shot_draft"] else None,
        chat_history=conv["chat_history"] if conv["chat_history"] else None,
    )
    result = call_llm(model_name, prompt, max_tokens=max_tokens, temperature=temperature)
    end = datetime.now(timezone.utc).isoformat()

    final_req = result["output"]
    synthesis_msg = {
        "start_time": start,
        "end_time": end,
        "role": "aunu_agent",
        "action": "finish",
        "input": prompt,
        "prompt_template": "user/task_requirement_prediction.jinja",
        "identified_ambiguity": "",
        "output": final_req,
        "llm": model_name,
        "input_tokens": result.get("input_tokens", 0),
        "output_tokens": result.get("output_tokens", 0),
        "cost": result.get("cost", 0.0),
    }

    all_messages = conv["truncated_messages"] + [synthesis_msg]

    return {
        "messages": all_messages,
        "is_complete": True,
        "task_requirement_final": final_req,
        "strategy": f"user_interaction_{n_turns}turn",
        "persona": persona_id,
        "task_id": task_id,
        "dataset": dataset,
        "model": model_name,
        "input_type": task_data.get("input_type", "elevator_pitch_summary"),
        "ground_truth": task_data.get("ground_truth", ""),
        "cost": round(result.get("cost", 0.0), 6),
        "actual_turns_used": conv["actual_turns"],
        "start_time": start,
        "end_time": end,
    }


def _load_elevator_pitch(dataset: str, persona_id: int, task_id: int) -> str:
    """Load elevator_pitch_summary from synthesized_output_2.6.json as fallback."""
    data_dir = os.path.join(_REPO_ROOT, "AUNUEnv", "data", "data_synthesized", dataset)
    path = os.path.join(data_dir, "synthesized_output_2.6.json")
    if not os.path.exists(path):
        logger.warning(f"Synthesized data not found at {path}; using empty string")
        return ""
    with open(path) as f:
        data = json.load(f)
    user_key = f"user_{persona_id}"
    for task_raw in data.get(user_key, {}).get("tasks_info", []):
        if task_raw.get("task_id") == task_id:
            return task_raw.get("elevator_pitch_summary", "")
    return ""


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
    n_turns: int,
    temperature: float,
    max_tokens: int,
    output_dir: str,
    exp_id: int | None,
) -> dict:
    dataset_safe = dataset.replace("/", "_")
    source_dir = _find_source_experiment(dataset_safe, source_aunu_model)

    with open(os.path.join(source_dir, "output.json")) as f:
        source_data = json.load(f)

    source_args = source_data.get("args", {})
    strategy_name = f"user_interaction_{n_turns}turn"

    base_dir = os.path.join(output_dir, dataset_safe, strategy_name)
    exp_dir = _make_exp_dir(base_dir, exp_id)
    out_path = os.path.join(exp_dir, "output.json")

    # Load existing output if resuming
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
                "n_turns": n_turns,
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

                # Skip already completed tasks
                if task_key in output[persona_key][habit_key]:
                    logger.info(
                        f"Skipping persona={persona_id} {habit_key} {task_key} (already done)"
                    )
                    continue

                task_id = task_val.get("task_id")
                if task_id is None:
                    continue

                logger.info(
                    f"Processing persona={persona_id} {habit_key} {task_key}"
                )
                try:
                    task_result = _process_task(
                        task_data=task_val,
                        dataset=dataset,
                        persona_id=persona_id,
                        task_id=task_id,
                        model_name=agent_model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        n_turns=n_turns,
                    )
                    total_cost += task_result.get("cost", 0.0)
                    output[persona_key][habit_key][task_key] = task_result
                    logger.info(
                        f"  Done: cost={task_result.get('cost', 0):.4f}, "
                        f"turns_used={task_result.get('actual_turns_used')}"
                    )
                except Exception as e:
                    logger.error(f"  FAILED: {e}", exc_info=True)
                    output[persona_key][habit_key][task_key] = {"error": str(e)}

                # Save incrementally after each task
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
            "Synthesize final task requirements from the first N turns of a "
            "user_interaction experiment (from 'results copy')."
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
        "--n_turns", type=int, default=10,
        help="Number of clarification turns to use (default: 10)",
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
        help=f"Base results directory (default: agent/results)",
    )
    parser.add_argument(
        "--exp_id", type=int, default=None,
        help="Resume or target a specific ExperimentN (default: auto-increment)",
    )
    parser.add_argument(
        "--log_file_path", type=str, default=None,
        help="Path to log file (auto-generated if omitted)",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = args.output_dir or _RESULTS_DIR
    dataset_safe = args.dataset.replace("/", "_")
    strategy_name = f"user_interaction_{args.n_turns}turn"

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
        n_turns=args.n_turns,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        output_dir=output_dir,
        exp_id=args.exp_id,
    )

    logger.info("Done.")


if __name__ == "__main__":
    main()
