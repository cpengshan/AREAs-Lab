"""
Analyze ground-truth task requirements across all datasets.

For each task requirement:
  1. Decompose into atomic units (LLM_judge_decompose.jinja)
  2. Classify each unit as data_specified or user_specified (LLM_judge_classify.jinja)
  3. Aggregate counts and ratios per task and per dataset.

Usage:
    python scripts/analyze_requirement_categories.py \
        [--model gpt-5.4] \
        [--data-dir data/data_synthesized] \
        [--output results/requirement_categories.json]
"""

import argparse
import json
import logging
import os
import sys

# Allow running from the AUNUEnv root directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aunu_env.utils.llm import call_llm
from aunu_env.utils.jinja_utils import render_template
from aunu_env.utils.json_utils import parse_json_output

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_PROMPT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "aunu_env", "evaluator", "prompt",
)
_DECOMPOSE_TEMPLATE = os.path.join(_PROMPT_DIR, "LLM_judge_decompose.jinja")
_CLASSIFY_TEMPLATE = os.path.join(_PROMPT_DIR, "LLM_judge_classify.jinja")


def decompose(requirement: str, model: str, temperature: float = 0.0) -> tuple[list[str], float]:
    prompt = render_template(_DECOMPOSE_TEMPLATE, task_requirement=requirement)
    result = call_llm(model, prompt, max_tokens=4096, temperature=temperature)
    parsed = parse_json_output(result["output"])
    units = parsed.get("atomic_units", [])
    if not isinstance(units, list):
        units = []
    return units, result.get("cost", 0.0)


def classify(units: list[str], model: str, temperature: float = 0.0) -> tuple[list[dict], float]:
    bullet_list = "\n".join(f"- {u}" for u in units)
    prompt = render_template(_CLASSIFY_TEMPLATE, atomic_units=bullet_list)
    result = call_llm(model, prompt, max_tokens=8192, temperature=temperature)
    parsed = parse_json_output(result["output"])
    classifications = parsed.get("classifications", [])
    if not isinstance(classifications, list):
        classifications = []
    return classifications, result.get("cost", 0.0)


def _ratio(n: int, total: int) -> float:
    return round(n / total, 4) if total > 0 else 0.0


def collect_task_requirements(data_dir: str, explicit_dirs: list[str] | None = None) -> list[dict]:
    """Walk data_dir (or explicit_dirs) and collect all task requirements with metadata."""
    tasks = []
    if explicit_dirs:
        # Use explicit list; dataset_id is derived relative to data_dir when possible
        candidates = [(d, os.path.relpath(d, data_dir) if data_dir else d) for d in explicit_dirs]
    else:
        candidates = []
        for root, dirs, files in os.walk(data_dir):
            if "synthesized_output.json" in files:
                candidates.append((root, os.path.relpath(root, data_dir)))

    for dir_path, dataset_id in candidates:
        path = os.path.join(dir_path, "synthesized_output.json")
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"Could not read {path}: {e}")
            continue

        for key, value in data.items():
            if not key.startswith("user_") or not isinstance(value, dict):
                continue
            tasks_info = value.get("tasks_info", [])
            if not isinstance(tasks_info, list):
                continue
            for task in tasks_info:
                req = task.get("task_requirement")
                if not req:
                    continue
                tasks.append({
                    "dataset_id": dataset_id,
                    "user_key": key,
                    "task_id": task.get("task_id"),
                    "task_name": task.get("task_name", ""),
                    "difficulty": task.get("difficulty", ""),
                    "task_requirement": req,
                })
    return tasks


def analyze(data_dir: str, model: str, output_path: str, explicit_dirs: list[str] | None = None) -> None:
    tasks = collect_task_requirements(data_dir, explicit_dirs)
    logger.info(f"Found {len(tasks)} task requirements across datasets.")

    task_results = []
    total_cost = 0.0

    for i, task in enumerate(tasks):
        label = f"{task['dataset_id']} / {task['user_key']} / task {task['task_id']}"
        logger.info(f"[{i+1}/{len(tasks)}] Processing: {label}")

        try:
            units, cost1 = decompose(task["task_requirement"], model)
            total_cost += cost1
            logger.info(f"  Decomposed into {len(units)} units (cost ${cost1:.5f})")

            if not units:
                task_results.append({**task, "units": [], "classifications": [],
                                      "data_specified_count": 0, "user_specified_count": 0,
                                      "total_count": 0, "data_specified_ratio": 0.0,
                                      "user_specified_ratio": 0.0, "cost": cost1})
                continue

            classifications, cost2 = classify(units, model)
            total_cost += cost2
            logger.info(f"  Classified units (cost ${cost2:.5f})")

            ds_count = sum(1 for c in classifications if c.get("category") == "data_specified")
            us_count = sum(1 for c in classifications if c.get("category") == "user_specified")
            total = len(classifications)

            task_results.append({
                **task,
                "units": units,
                "classifications": classifications,
                "data_specified_count": ds_count,
                "user_specified_count": us_count,
                "total_count": total,
                "data_specified_ratio": _ratio(ds_count, total),
                "user_specified_ratio": _ratio(us_count, total),
                "cost": cost1 + cost2,
            })

        except Exception as e:
            logger.error(f"  Failed: {e}")
            task_results.append({**task, "error": str(e)})

    # Aggregate per dataset
    dataset_agg: dict[str, dict] = {}
    for tr in task_results:
        did = tr["dataset_id"]
        if did not in dataset_agg:
            dataset_agg[did] = {"data_specified_count": 0, "user_specified_count": 0, "total_count": 0, "task_count": 0}
        agg = dataset_agg[did]
        agg["task_count"] += 1
        agg["data_specified_count"] += tr.get("data_specified_count", 0)
        agg["user_specified_count"] += tr.get("user_specified_count", 0)
        agg["total_count"] += tr.get("total_count", 0)

    dataset_results = {}
    for did, agg in dataset_agg.items():
        total = agg["total_count"]
        dataset_results[did] = {
            **agg,
            "data_specified_ratio": _ratio(agg["data_specified_count"], total),
            "user_specified_ratio": _ratio(agg["user_specified_count"], total),
        }

    output = {
        "model": model,
        "total_cost_usd": round(total_cost, 5),
        "task_results": task_results,
        "dataset_results": dataset_results,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results saved to {output_path} (total cost: ${total_cost:.4f})")

    # Print summary table
    print("\n=== Per-Dataset Summary ===")
    print(f"{'Dataset':<50} {'Tasks':>6} {'Total':>7} {'Data%':>7} {'User%':>7}")
    print("-" * 82)
    for did, dr in sorted(dataset_results.items()):
        print(f"{did:<50} {dr['task_count']:>6} {dr['total_count']:>7} "
              f"{dr['data_specified_ratio']*100:>6.1f}% {dr['user_specified_ratio']*100:>6.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Analyze requirement categories.")
    parser.add_argument("--model", default="gpt-5.4", help="LLM model name")
    parser.add_argument("--data-dir", default="data/data_synthesized",
                        help="Root directory containing synthesized_output.json files")
    parser.add_argument("--data-dirs", nargs="+", default=None,
                        help="Explicit list of dataset directories (overrides --data-dir walk)")
    parser.add_argument("--output", default="results/requirement_categories.json",
                        help="Output JSON file path")
    args = parser.parse_args()

    # Resolve paths relative to the AUNUEnv root
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = args.data_dir if os.path.isabs(args.data_dir) else os.path.join(root, args.data_dir)
    output = args.output if os.path.isabs(args.output) else os.path.join(root, args.output)
    explicit_dirs = args.data_dirs  # already absolute paths when passed from shell

    analyze(data_dir, args.model, output, explicit_dirs)


if __name__ == "__main__":
    main()
