#!/usr/bin/env python3
"""Re-run atomic evaluation on an existing output.json experiment file.

Evaluates all tasks in parallel (one thread per task) and writes
eval_results.json in the same directory.

Usage:
    python agent/scripts/run_evaluation.py \
        --input_path agent/results/alexfabbri_multi_news/zero_shot/Experiment17/output.json \
        --evaluator_model gpt-4.1

    # Limit parallelism:
    python agent/scripts/run_evaluation.py \
        --input_path /path/to/output.json \
        --evaluator_model gpt-4.1 \
        --workers 5

    # Resume a partial run:
    python agent/scripts/run_evaluation.py \
        --input_path /path/to/output.json \
        --evaluator_model gpt-4.1 \
        --resume
"""

import argparse
import json
import logging
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from AUNUEnv.aunu_env.evaluator.atomic_evaluator import AtomicEvaluator
from AUNUEnv.aunu_env.evaluator.metrics import aggregate_results

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Re-evaluate experiment results from output.json")
    parser.add_argument("--input_path", required=True,
                        help="Path to existing output.json from a run_agent.py experiment")
    parser.add_argument("--evaluator_model", type=str, default=None,
                        help="LLM model for evaluation (default: read from output.json args)")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Where to save eval_results.json (default: same dir as input)")
    parser.add_argument("--cache_path", type=str, default=None,
                        help="Path to gold unit cache JSON (default: ground_truth_decompose.json "
                             "next to the synthesized dataset)")
    parser.add_argument("--workers", type=int, default=10,
                        help="Max parallel evaluation threads (default: 10, one per task)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip tasks that already have scores in an existing eval_results.json")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _task_cache_id(persona_id: int, task_num: int) -> str:
    """Reproduce the task_id string used by the dataset loader for gold caching."""
    return f"user_{persona_id}_task_{task_num - 1}"


def _format_eval_result(eval_result: dict) -> dict:
    """Flatten AtomicEvaluator output into the canonical eval_results.json format."""
    comparison = eval_result.get("comparison", {})
    return {
        "predicted_units": eval_result.get("predicted_units", []),
        "ground_truth_units": eval_result.get("gold_units", []),
        "matched_pairs": comparison.get("matched_pairs", []),
        "missing_units": comparison.get("missing_units", []),
        "hallucinated_units": comparison.get("hallucinated_units", []),
        "misaligned_units": comparison.get("misaligned_units", []),
        "critical_units": comparison.get("critical_units", []),
        "critical_missing": comparison.get("critical_missing", []),
        "counts": eval_result.get("counts", {}),
        "scores": eval_result.get("scores", {}),
        "gold_categories": eval_result.get("gold_categories", {}),
        "pred_categories": eval_result.get("pred_categories", {}),
        "subcategory_scores": eval_result.get("subcategory_scores", {}),
    }


def _collect_tasks(data: dict, existing: dict, resume: bool) -> list[dict]:
    """Extract all (persona_key, habit_key, task_key, predicted, gold, cache_id) tuples."""
    tasks = []
    for persona_key, persona_data in data.items():
        if persona_key == "args" or not isinstance(persona_data, dict):
            continue

        persona_id = int(persona_key)

        # Detect habit-keyed structure: {habit_key: {task_1: result, ...}}
        # If persona_data's own keys start with "task_", it's flat; otherwise habit-keyed.
        if any(k.startswith("task_") for k in persona_data):
            habit_items = [(None, persona_data)]
        else:
            habit_items = list(persona_data.items())

        for habit_key, task_dict in habit_items:
            for task_key, task_result in task_dict.items():
                if not task_key.startswith("task_") or not isinstance(task_result, dict):
                    continue

                if resume:
                    ex = existing.get(persona_key, {})
                    if habit_key:
                        ex = ex.get(habit_key, {})
                    if task_key in ex and ex[task_key].get("scores"):
                        logger.info(f"[persona={persona_id} {task_key}] already evaluated, skipping")
                        continue

                predicted = task_result.get("task_requirement_final", "")
                gold = task_result.get("ground_truth", "")
                if not predicted or not gold:
                    logger.warning(f"[persona={persona_id} {task_key}] missing predicted or gold, skipping")
                    continue

                task_num = int(task_key.split("_")[-1])
                tasks.append({
                    "persona_key": persona_key,
                    "persona_id": persona_id,
                    "habit_key": habit_key,
                    "task_key": task_key,
                    "predicted": predicted,
                    "gold": gold,
                    "cache_id": _task_cache_id(persona_id, task_num),
                })
    return tasks


def _evaluate_task(task: dict, evaluator: AtomicEvaluator) -> tuple[dict, dict]:
    """Run evaluation for a single task. Returns (task_meta, formatted_result)."""
    label = f"persona={task['persona_id']} {task['task_key']}"
    logger.info(f"[{label}] evaluating...")
    eval_result = evaluator.evaluate(
        predicted=task["predicted"],
        gold=task["gold"],
        task_id=task["cache_id"],
    )
    scores = eval_result.get("scores", {})
    logger.info(
        f"[{label}] F1={scores.get('f1', 0):.3f}  "
        f"P={scores.get('precision', 0):.3f}  "
        f"R={scores.get('recall', 0):.3f}"
    )
    print(
        f"[{label}] F1={scores.get('f1', 0):.3f}  "
        f"P={scores.get('precision', 0):.3f}  "
        f"R={scores.get('recall', 0):.3f}"
    )
    return task, _format_eval_result(eval_result), eval_result.get("cost", 0.0)


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    with open(args.input_path) as f:
        data = json.load(f)

    run_args = data.get("args", {})
    evaluator_model = args.evaluator_model or run_args.get("evaluator_model")
    if not evaluator_model:
        raise ValueError(
            "evaluator_model not found in output.json args; pass --evaluator_model"
        )

    exp_dir = os.path.dirname(os.path.abspath(args.input_path))
    output_path = args.output_path or os.path.join(exp_dir, "eval_results.json")

    cache_path = args.cache_path
    if cache_path is None:
        dataset_name = run_args.get("dataset", "")
        if dataset_name:
            cache_path = os.path.join(
                _REPO_ROOT, "AUNUEnv", "data", "data_synthesized",
                dataset_name, "ground_truth_decompose.json",
            )

    # Single shared evaluator — AtomicEvaluator is thread-safe for the LLM
    # calls; gold cache writes are protected by a lock below.
    evaluator = AtomicEvaluator(
        model_name=evaluator_model,
        cache_gold_units=True,
        cache_path=cache_path,
    )

    eval_output: dict = {}
    if args.resume and os.path.exists(output_path):
        with open(output_path) as f:
            eval_output = json.load(f)
        logger.info(f"Resuming from {output_path}")

    tasks_to_run = _collect_tasks(data, eval_output, args.resume)
    logger.info(f"Running {len(tasks_to_run)} tasks with up to {args.workers} parallel workers")
    print(f"Evaluating {len(tasks_to_run)} tasks in parallel (workers={args.workers})...")

    all_formatted: list[dict] = []
    total_cost = 0.0
    write_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_evaluate_task, task, evaluator): task
            for task in tasks_to_run
        }

        for future in as_completed(futures):
            try:
                task_meta, formatted, cost = future.result()
            except Exception as exc:
                t = futures[future]
                logger.error(
                    f"[persona={t['persona_id']} {t['task_key']}] FAILED: {exc}",
                    exc_info=True,
                )
                continue

            all_formatted.append(formatted)
            total_cost += cost

            with write_lock:
                pk = task_meta["persona_key"]
                hk = task_meta["habit_key"]
                tk = task_meta["task_key"]

                if pk not in eval_output:
                    eval_output[pk] = {}
                if hk:
                    if hk not in eval_output[pk]:
                        eval_output[pk][hk] = {}
                    eval_output[pk][hk][tk] = formatted
                else:
                    eval_output[pk][tk] = formatted

                with open(output_path, "w") as f:
                    json.dump(eval_output, f, indent=2, default=str)

    # Collect already-skipped results from existing eval_output into agg
    for pk, pv in eval_output.items():
        if pk in ("args", "aggregate_scores") or not isinstance(pv, dict):
            continue
        for tk, tv in pv.items():
            if isinstance(tv, dict) and tv.get("scores"):
                if tv not in all_formatted:
                    all_formatted.append(tv)

    agg = aggregate_results(all_formatted)
    print(f"\n=== Evaluation Summary ===")
    print(f"Tasks evaluated: {agg.get('n_tasks', 0)}")
    print(f"Total eval cost: ${total_cost:.6f}")
    for metric in ("f1", "precision", "recall", "alignment", "constraint_preservation"):
        if metric in agg:
            print(f"  {metric}: {agg[metric]:.4f}")

    eval_output["aggregate_scores"] = agg
    with open(output_path, "w") as f:
        json.dump(eval_output, f, indent=2, default=str)

    print(f"\nSaved eval_results.json → {output_path}")


if __name__ == "__main__":
    main()
