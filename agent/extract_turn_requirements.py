#!/usr/bin/env python3
"""Post-process a user_interaction output.json to generate per-turn task requirements.

For each task in the experiment, rebuilds the conversation history incrementally and
calls task_requirement_prediction.jinja at every turn checkpoint to produce a
task_requirement_final as it would have looked after exactly T turns of clarification.

Usage:
    python agent/extract_turn_requirements.py \\
        --exp_dir agent/results/alexfabbri_multi_news/user_interaction/Experiment5 \\
        --agent_model claude-sonnet-4-6 \\
        [--log_file_path path/to/log]
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
_DATA_SYNTHESIZED_ROOT = os.path.join(_REPO_ROOT, "AUNUEnv", "data", "data_synthesized")


def _load_elevator_pitch(dataset: str, persona_id: int, task_id: int) -> str:
    """Load elevator_pitch_summary from synthesized_output_2.6.json."""
    path = os.path.join(_DATA_SYNTHESIZED_ROOT, dataset, "synthesized_output_2.6.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Synthesized data not found: {path}")
    with open(path) as f:
        data = json.load(f)
    user_key = f"user_{persona_id}"
    user_entry = data.get(user_key)
    if user_entry is None:
        raise ValueError(f"No '{user_key}' in {path}")
    for task_raw in user_entry.get("tasks_info", []):
        if task_raw.get("task_id") == task_id:
            return task_raw.get("elevator_pitch_summary", "")
    raise ValueError(f"task_id={task_id} not found under '{user_key}' in {path}")


def _extract_turn_checkpoints(messages: list) -> list[dict]:
    """Return a list of turn checkpoints from the message log.

    Each checkpoint captures the state just after one agent→user exchange:
      {
        "turn": <int>,                  # 1-based turn number
        "zero_shot_draft": <str>,       # from the zero_shot_draft message
        "chat_history": [               # conversation up to this turn
            {"role": "agent"|"user", "output": <str>}, ...
        ]
      }
    """
    zero_shot_draft = ""
    agent_turns: list[str] = []   # agent questions in order
    user_turns: list[str] = []    # user responses in order
    checkpoints: list[dict] = []

    for msg in messages:
        role = msg.get("role", "")
        action = msg.get("action", "")
        output = msg.get("output", "")

        if role == "aunu_agent" and action == "zero_shot_draft":
            zero_shot_draft = output

        elif role == "aunu_agent" and action == "ask_user":
            agent_turns.append(output)

        elif role == "mimic_user" and action == "respond":
            user_turns.append(output)
            # We have a complete exchange — snapshot a checkpoint
            turn_num = len(user_turns)
            chat_history = []
            for i in range(turn_num):
                chat_history.append({"role": "agent", "output": agent_turns[i]})
                chat_history.append({"role": "user", "output": user_turns[i]})
            checkpoints.append({
                "turn": turn_num,
                "zero_shot_draft": zero_shot_draft,
                "chat_history": chat_history,
            })

    return checkpoints


def process_task(
    task_data: dict,
    dataset: str,
    persona_id: int,
    task_id: int,
    model_name: str,
    temperature: float,
    max_tokens: int,
) -> dict:
    """Generate per-turn requirements for a single task.

    Returns a dict keyed by "turn_N" with each turn's requirement and metadata.
    """
    messages = task_data.get("messages", [])
    checkpoints = _extract_turn_checkpoints(messages)

    if not checkpoints:
        logger.warning(f"  persona={persona_id} task={task_id}: no turn checkpoints found")
        return {}

    elevator_pitch = _load_elevator_pitch(dataset, persona_id, task_id)
    results: dict = {}

    for cp in checkpoints:
        turn = cp["turn"]
        logger.info(f"  persona={persona_id} task={task_id} turn={turn}: generating requirement")

        start = datetime.now(timezone.utc).isoformat()
        prompt = render_template(
            _PREDICTION_TEMPLATE,
            initial_task_requirement=elevator_pitch,
            zero_shot_draft=cp["zero_shot_draft"],
            chat_history=cp["chat_history"] if cp["chat_history"] else None,
        )
        result = call_llm(model_name, prompt, max_tokens=max_tokens, temperature=temperature)
        end = datetime.now(timezone.utc).isoformat()

        results[f"turn_{turn}"] = {
            "task_requirement_final": result["output"],
            "turn": turn,
            "persona": persona_id,
            "task_id": task_id,
            "dataset": dataset,
            "model": model_name,
            "cost": result.get("cost", 0.0),
            "input_tokens": result.get("input_tokens", 0),
            "output_tokens": result.get("output_tokens", 0),
            "start_time": start,
            "end_time": end,
        }

    return results


def process_experiment(exp_dir: str, model_name: str, temperature: float, max_tokens: int) -> dict:
    """Process all tasks in an experiment directory.

    Reads output.json and writes turn_requirements.json.
    """
    out_path = os.path.join(exp_dir, "output.json")
    if not os.path.exists(out_path):
        raise FileNotFoundError(f"output.json not found in {exp_dir}")

    with open(out_path) as f:
        data = json.load(f)

    args_meta = data.get("args", {})
    dataset = args_meta.get("dataset", "")
    total_cost = 0.0

    turn_output: dict = {
        "args": {
            "source_exp_dir": exp_dir,
            "model": model_name,
            "dataset": dataset,
            "strategy": "user_interaction_multi_rounds",
            "total_cost": 0.0,
        }
    }

    for persona_key, persona_val in data.items():
        if persona_key == "args" or not isinstance(persona_val, dict):
            continue

        persona_id = int(persona_key)
        turn_output[persona_key] = {}

        for habit_key, habit_val in persona_val.items():
            if not isinstance(habit_val, dict):
                continue
            turn_output[persona_key][habit_key] = {}

            for task_key, task_val in habit_val.items():
                if not isinstance(task_val, dict):
                    continue

                task_id = task_val.get("task_id")
                if task_id is None:
                    continue

                logger.info(f"Processing persona={persona_id} habit={habit_key} {task_key}")
                try:
                    per_turn = process_task(
                        task_data=task_val,
                        dataset=dataset,
                        persona_id=persona_id,
                        task_id=task_id,
                        model_name=model_name,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    task_cost = sum(v.get("cost", 0.0) for v in per_turn.values())
                    total_cost += task_cost
                    turn_output[persona_key][habit_key][task_key] = per_turn
                    logger.info(
                        f"  Done: {len(per_turn)} turns, cost={task_cost:.4f}"
                    )
                except Exception as e:
                    logger.error(f"  FAILED: {e}", exc_info=True)
                    turn_output[persona_key][habit_key][task_key] = {"error": str(e)}

    turn_output["args"]["total_cost"] = round(total_cost, 6)

    result_path = os.path.join(exp_dir, "turn_requirements.json")
    with open(result_path, "w") as f:
        json.dump(turn_output, f, indent=2, default=str)
    logger.info(f"Saved turn requirements → {result_path} (total_cost={total_cost:.4f})")
    return turn_output


def main():
    parser = argparse.ArgumentParser(
        description="Generate per-turn task requirements from a user_interaction output.json"
    )
    parser.add_argument(
        "--exp_dir", required=True,
        help="Path to user_interaction experiment directory containing output.json",
    )
    parser.add_argument(
        "--agent_model", type=str, default="claude-sonnet-4-6",
        help="LLM model to use for task_requirement_prediction (default: claude-sonnet-4-6)",
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
        "--log_file_path", type=str, default=None,
        help="Path to log file (default: <exp_dir>/turn_requirements.log)",
    )
    args = parser.parse_args()

    log_path = args.log_file_path or os.path.join(args.exp_dir, "turn_requirements.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
    )

    logger.info(f"Processing experiment: {args.exp_dir}")
    process_experiment(
        exp_dir=args.exp_dir,
        model_name=args.agent_model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    logger.info("Done.")


if __name__ == "__main__":
    main()
