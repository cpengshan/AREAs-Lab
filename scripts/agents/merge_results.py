"""
merge_results.py

Standalone script that merges the task_requirement_final from a prior user-interaction
run and a prior data-interaction run into a single output, saved in the same format as
other strategy outputs.

Usage:
    python scripts/agents/merge_results.py \
        --user_result_path results/.../user/ExperimentN/output.json \
        --data_result_path results/.../data/ExperimentN/output.json \
        --aunu_model claude-haiku-4-5-20251001 \
        --dataset alexfabbri/multi_news \
        --input_type elevator_pitch_summary
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone

from jinja2 import Environment, FileSystemLoader

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.model_base import LLM

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "../../results")
PROMPT_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "../../prompts/agents/aunu_agent")
DATA_SYNTHESIZED_DIR = os.path.join(os.path.dirname(__file__), "../../data/data_synthesized")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result file helpers
# ---------------------------------------------------------------------------

def load_result_file(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Result file not found: {path}")
    with open(path) as f:
        return json.load(f)


def get_task_requirement(data: dict, persona_id: int, task_id: int, source_label: str) -> str:
    persona_key = str(persona_id)
    task_key = f"task_{task_id}"
    if persona_key not in data:
        raise ValueError(
            f"{source_label}: persona {persona_id} not found in result file."
        )
    if task_key not in data[persona_key]:
        raise ValueError(
            f"{source_label}: task {task_id} for persona {persona_id} not found in result file."
        )
    req = data[persona_key][task_key].get("task_requirement_final", "")
    if not req:
        raise ValueError(
            f"{source_label}: task_requirement_final is empty for persona {persona_id}, task {task_id}."
        )
    return req


def get_user_instruction(data: dict, persona_id: int, task_id: int) -> str:
    """Pull the original user instruction from an existing result entry."""
    persona_key = str(persona_id)
    task_key = f"task_{task_id}"
    entry = data.get(persona_key, {}).get(task_key, {})
    messages = entry.get("messages", [])
    # The mimic_user init message carries the original instruction
    for m in messages:
        if m.get("role") == "mimic_user" and m.get("action") == "init":
            return m.get("output", "")
    return entry.get("input_type", "")


def load_ground_truth(dataset: str, persona_id: int, task_id: int) -> str:
    output_path = os.path.join(DATA_SYNTHESIZED_DIR, dataset, "synthesized_output.json")
    with open(output_path) as f:
        d = json.load(f)
    user_key = f"user_{persona_id}"
    if user_key not in d:
        raise ValueError(f"Persona {persona_id} not found in synthesized_output.json")
    for task in d[user_key]["tasks_info"]:
        if task["task_id"] == task_id:
            return task["task_requirement"]
    raise ValueError(f"Task {task_id} for persona {persona_id} not found in synthesized_output.json")


# ---------------------------------------------------------------------------
# Merge logic
# ---------------------------------------------------------------------------

def merge_task_requirements(
    llm: LLM,
    env: Environment,
    user_instruction: str,
    user_req: str,
    data_req: str,
    model_name: str,
) -> dict:
    template = env.get_template("merge_user_data.jinja")
    prompt = template.render(
        user_instruction=user_instruction,
        user_interaction_requirement=user_req,
        data_interaction_requirement=data_req,
    )

    start_time = datetime.now(timezone.utc).isoformat()
    response = llm.generate(prompt)
    end_time = datetime.now(timezone.utc).isoformat()

    return {
        "start_time": start_time,
        "end_time": end_time,
        "role": "aunu_agent",
        "action": "merge_user_data",
        "input": prompt,
        "prompt_template": "merge_user_data.jinja",
        "output": response["output"],
        "llm": model_name,
        "input_tokens": response.get("input_tokens", 0),
        "output_tokens": response.get("output_tokens", 0),
        "cost": response.get("cost", 0.0),
    }


# ---------------------------------------------------------------------------
# Output directory helpers (mirrors run_aunu_interaction.py)
# ---------------------------------------------------------------------------

def make_exp_dir(base_dir: str) -> str:
    exp_id = 1
    while os.path.exists(os.path.join(base_dir, f"Experiment{exp_id}")):
        exp_id += 1
    exp_dir = os.path.join(base_dir, f"Experiment{exp_id}")
    os.makedirs(exp_dir)
    return exp_dir


# ---------------------------------------------------------------------------
# Per-task runner
# ---------------------------------------------------------------------------

def run_task(
    args,
    user_data: dict,
    data_data: dict,
    llm: LLM,
    env: Environment,
    persona_id: int,
    task_id: int,
) -> dict:
    user_req = get_task_requirement(user_data, persona_id, task_id, "user result")
    data_req = get_task_requirement(data_data, persona_id, task_id, "data result")

    user_instruction = get_user_instruction(user_data, persona_id, task_id)

    logger.info(f"Merging persona {persona_id}, task {task_id}...")
    message = merge_task_requirements(llm, env, user_instruction, user_req, data_req, args.aunu_model)
    logger.info(f"Done — persona {persona_id}, task {task_id}.")

    ground_truth = load_ground_truth(args.dataset, persona_id, task_id)

    return {
        "messages": [message],
        "is_complete": True,
        "task_requirement_final": message["output"],
        "strategy": "merge",
        "persona": persona_id,
        "task_id": task_id,
        "dataset": args.dataset,
        "model": args.aunu_model,
        "input_type": args.input_type,
        "ground_truth": ground_truth,
        "cost": round(message["cost"], 6),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Merge user-interaction and data-interaction results")
    parser.add_argument("--user_result_path", type=str, required=True,
                        help="Path to output.json from a prior 'user' strategy run")
    parser.add_argument("--data_result_path", type=str, required=True,
                        help="Path to output.json from a prior 'data' strategy run")
    parser.add_argument("--aunu_model", type=str, default="claude-haiku-4-5-20251001",
                        help="Model to use for the merge LLM call")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (e.g. alexfabbri/multi_news)")
    parser.add_argument("--input_type", type=str, default="elevator_pitch_summary",
                        choices=["elevator_pitch_summary", "deep_dive_summary"])
    parser.add_argument("--persona", type=int, nargs="+", required=True,
                        help="One or more persona IDs to process")
    parser.add_argument("--log_file_path", type=str,
                        default=os.path.join(os.path.dirname(__file__), "../loggings/merge/run.log"))
    return parser.parse_args()


def run():
    args = parse_args()

    os.makedirs(os.path.dirname(args.log_file_path), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(args.log_file_path),
            logging.StreamHandler(),
        ],
    )

    user_data = load_result_file(args.user_result_path)
    data_data = load_result_file(args.data_result_path)

    llm = LLM(args.aunu_model)
    env = Environment(loader=FileSystemLoader(PROMPT_TEMPLATE_DIR))

    base_dir = os.path.join(RESULTS_DIR, args.dataset.replace("/", "_"), "merge")
    os.makedirs(base_dir, exist_ok=True)
    exp_dir = make_exp_dir(base_dir)

    all_results = {
        "args": {
            "strategy_aunu": "merge",
            "aunu_model": args.aunu_model,
            "dataset": args.dataset,
            "input_type": args.input_type,
            "persona": args.persona,
            "user_result_path": args.user_result_path,
            "data_result_path": args.data_result_path,
            "total_cost": 0.0,
        }
    }

    for persona_id in args.persona:
        persona_key = str(persona_id)
        all_results.setdefault(persona_key, {})

        for task_id in (1, 2):
            task_key = f"task_{task_id}"
            task_result = run_task(args, user_data, data_data, llm, env, persona_id, task_id)
            all_results[persona_key][task_key] = task_result

            total_cost = sum(
                all_results[pid][tk].get("cost", 0.0)
                for pid in all_results
                if pid != "args"
                for tk in all_results[pid]
            )
            all_results["args"]["total_cost"] = round(total_cost, 6)

            out_path = os.path.join(exp_dir, "output.json")
            with open(out_path, "w") as f:
                json.dump(all_results, f, indent=2)
            logger.info(f"Saved to {out_path}")

    eval_path = os.path.join(exp_dir, "eval_results.json")
    if not os.path.exists(eval_path):
        with open(eval_path, "w") as f:
            json.dump({}, f, indent=2)

    logger.info(f"Merge complete. Results: {exp_dir}")


if __name__ == "__main__":
    run()
