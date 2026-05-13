#!/usr/bin/env python3
"""Evaluate per-turn task requirements from turn_requirements.json.

Reads turn_requirements.json produced by extract_turn_requirements.py and
the companion output.json (for ground-truth labels), then evaluates each
turn's task_requirement_final against the gold using AtomicEvaluator.

Results are written to eval_results_multi_rounds.json in the same directory,
keyed by persona → habit → task → turn.

Usage:
    python agent/run_evaluation_multi_rounds.py \\
        --exp_dir agent/results/alexfabbri_multi_news/user_interaction/Experiment3 \\
        --evaluator_model gpt-4.1 \\
        [--workers 10] [--resume] [--reasoning_effort none]
"""

import argparse
import json
import logging
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from AUNUEnv.aunu_env.evaluator.atomic_evaluator import AtomicEvaluator
from AUNUEnv.aunu_env.evaluator.metrics import aggregate_results

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate per-turn requirements from turn_requirements.json"
    )
    parser.add_argument("--exp_dir", required=True,
                        help="Experiment dir containing turn_requirements.json and output.json")
    parser.add_argument("--evaluator_model", type=str, default="gpt-4.1",
                        help="LLM model for AtomicEvaluator (default: gpt-4.1)")
    parser.add_argument("--reasoning_effort", type=str, default="none",
                        help="Reasoning effort: none | low | medium | high (default: none)")
    parser.add_argument("--workers", type=int, default=10,
                        help="Max parallel evaluation threads (default: 10)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip turns already present in eval_results_multi_rounds.json")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Output path (default: <exp_dir>/eval_results_multi_rounds.json)")
    parser.add_argument("--cache_path", type=str, default=None,
                        help="Gold unit cache JSON path")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _task_cache_id(persona_id: int, task_num: int) -> str:
    return f"user_{persona_id}_task_{task_num - 1}"


def _format_eval_result(eval_result: dict) -> dict:
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


def _load_ground_truths(output_json: dict) -> dict:
    """Extract ground_truth per (persona_key, habit_key, task_key) from output.json."""
    gold = {}
    for persona_key, persona_val in output_json.items():
        if persona_key == "args" or not isinstance(persona_val, dict):
            continue
        for habit_key, habit_val in persona_val.items():
            if not isinstance(habit_val, dict):
                continue
            for task_key, task_val in habit_val.items():
                if not isinstance(task_val, dict):
                    continue
                gt = task_val.get("ground_truth", "")
                if gt:
                    gold[(persona_key, habit_key, task_key)] = gt
    return gold


def _collect_tasks(turn_data: dict, ground_truths: dict, existing: dict, resume: bool) -> list[dict]:
    tasks = []
    for persona_key, persona_val in turn_data.items():
        if persona_key == "args" or not isinstance(persona_val, dict):
            continue
        persona_id = int(persona_key)

        for habit_key, habit_val in persona_val.items():
            if not isinstance(habit_val, dict):
                continue

            for task_key, task_val in habit_val.items():
                if not isinstance(task_val, dict):
                    continue

                gold = ground_truths.get((persona_key, habit_key, task_key), "")
                if not gold:
                    logger.warning(
                        f"[persona={persona_id} {habit_key} {task_key}] "
                        "no ground_truth in output.json — skipping"
                    )
                    continue

                task_num = int(task_key.split("_")[-1])
                cache_id = _task_cache_id(persona_id, task_num)

                for turn_key, turn_val in task_val.items():
                    if not (turn_key.startswith("turn_") or turn_key.startswith("data_turn_")) or not isinstance(turn_val, dict):
                        continue

                    if resume:
                        ex_score = (
                            existing
                            .get(persona_key, {})
                            .get(habit_key, {})
                            .get(task_key, {})
                            .get(turn_key, {})
                            .get("scores")
                        )
                        if ex_score:
                            logger.info(
                                f"[persona={persona_id} {habit_key} {task_key} {turn_key}] "
                                "already evaluated, skipping"
                            )
                            continue

                    predicted = turn_val.get("task_requirement_final", "")
                    if not predicted:
                        logger.warning(
                            f"[persona={persona_id} {habit_key} {task_key} {turn_key}] "
                            "no task_requirement_final — skipping"
                        )
                        continue

                    tasks.append({
                        "persona_key": persona_key,
                        "persona_id": persona_id,
                        "habit_key": habit_key,
                        "task_key": task_key,
                        "turn_key": turn_key,
                        "turn": turn_val.get("turn", turn_key),
                        "predicted": predicted,
                        "gold": gold,
                        "cache_id": cache_id,
                    })
    return tasks


def _evaluate_task(task: dict, evaluator: AtomicEvaluator) -> tuple[dict, dict, float]:
    label = f"persona={task['persona_id']} {task['habit_key']} {task['task_key']} {task['turn_key']}"
    logger.info(f"[{label}] evaluating...")
    eval_result = evaluator.evaluate(
        predicted=task["predicted"],
        gold=task["gold"],
        task_id=task["cache_id"],
    )
    scores = eval_result.get("scores", {})
    sub = eval_result.get("subcategory_scores") or {}
    us = sub.get("user_specified", {})
    ds = sub.get("data_specified", {})
    print(
        f"[{label}] "
        f"F1={scores.get('f1', 0):.3f} P={scores.get('precision', 0):.3f} R={scores.get('recall', 0):.3f}  |  "
        f"User-spec F1={us.get('f1', 0):.3f}  |  "
        f"Data-spec F1={ds.get('f1', 0):.3f}"
    )
    return task, _format_eval_result(eval_result), eval_result.get("cost", 0.0)


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    turn_req_path = os.path.join(args.exp_dir, "turn_requirements.json")
    output_json_path = os.path.join(args.exp_dir, "output.json")
    eval_output_path = args.output_path or os.path.join(args.exp_dir, "eval_results_multi_rounds.json")

    if not os.path.exists(turn_req_path):
        raise FileNotFoundError(f"turn_requirements.json not found: {turn_req_path}")
    if not os.path.exists(output_json_path):
        raise FileNotFoundError(f"output.json not found: {output_json_path}")

    with open(turn_req_path) as f:
        turn_data = json.load(f)
    with open(output_json_path) as f:
        output_data = json.load(f)

    run_args = turn_data.get("args", {})
    dataset_name = run_args.get("dataset", output_data.get("args", {}).get("dataset", ""))

    cache_path = args.cache_path
    if cache_path is None and dataset_name:
        slug = dataset_name.replace("/", "_")
        cache_path = os.path.join(
            _REPO_ROOT, "AUNUEnv", "aunu_env", "evaluator", "cache", f"{slug}.json"
        )

    evaluator = AtomicEvaluator(
        model_name=args.evaluator_model,
        cache_gold_units=True,
        cache_path=cache_path,
        reasoning_effort=args.reasoning_effort,
    )

    eval_output: dict = {
        "args": {
            "source_exp_dir": args.exp_dir,
            "evaluator_model": args.evaluator_model,
            "reasoning_effort": args.reasoning_effort,
            "dataset": dataset_name,
            "strategy": run_args.get("strategy", "user_interaction_multi_rounds"),
        }
    }
    if args.resume and os.path.exists(eval_output_path):
        with open(eval_output_path) as f:
            eval_output = json.load(f)
        logger.info(f"Resuming from {eval_output_path}")

    ground_truths = _load_ground_truths(output_data)
    tasks_to_run = _collect_tasks(turn_data, ground_truths, eval_output, args.resume)

    logger.info(f"Running {len(tasks_to_run)} turn evaluations (workers={args.workers})")
    print(f"Evaluating {len(tasks_to_run)} turns in parallel (workers={args.workers})...")

    all_formatted: list[dict] = []
    total_cost = 0.0
    write_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_evaluate_task, task, evaluator): task for task in tasks_to_run}

        for future in as_completed(futures):
            try:
                task_meta, formatted, cost = future.result()
            except Exception as exc:
                t = futures[future]
                logger.error(
                    f"[persona={t['persona_id']} {t['habit_key']} {t['task_key']} {t['turn_key']}] "
                    f"FAILED: {exc}",
                    exc_info=True,
                )
                continue

            all_formatted.append(formatted)
            total_cost += cost

            with write_lock:
                pk = task_meta["persona_key"]
                hk = task_meta["habit_key"]
                tk = task_meta["task_key"]
                trk = task_meta["turn_key"]

                eval_output.setdefault(pk, {}).setdefault(hk, {}).setdefault(tk, {})[trk] = formatted

                with open(eval_output_path, "w") as f:
                    json.dump(eval_output, f, indent=2, default=str)

    # Collect all results (including pre-existing resumed ones)
    all_formatted = []
    for pk, pv in eval_output.items():
        if pk in ("args", "aggregate_scores") or not isinstance(pv, dict):
            continue
        for hk, hv in pv.items():
            if not isinstance(hv, dict):
                continue
            for tk, tv in hv.items():
                if not isinstance(tv, dict):
                    continue
                for trk, trv in tv.items():
                    if isinstance(trv, dict) and trv.get("scores"):
                        all_formatted.append(trv)

    agg = aggregate_results(all_formatted)

    us_vals = [(f.get("subcategory_scores") or {}).get("user_specified", {}) or {} for f in all_formatted]
    ds_vals = [(f.get("subcategory_scores") or {}).get("data_specified", {}) or {} for f in all_formatted]
    def _avg(lst, key): return sum(x.get(key, 0) for x in lst) / len(lst) if lst else 0

    print(f"\n=== Evaluation Summary ===")
    print(f"Turns evaluated: {len(all_formatted)}")
    print(f"Total eval cost: ${total_cost:.6f}")
    for metric in ("f1", "precision", "recall", "alignment", "constraint_preservation"):
        if metric in agg:
            print(f"  {metric}: {agg[metric]:.4f}")
    print(f"User-specified   F1={_avg(us_vals, 'f1'):.4f}  P={_avg(us_vals, 'precision'):.4f}  R={_avg(us_vals, 'recall'):.4f}")
    print(f"Data-specified   F1={_avg(ds_vals, 'f1'):.4f}  P={_avg(ds_vals, 'precision'):.4f}  R={_avg(ds_vals, 'recall'):.4f}")

    eval_output["aggregate_scores"] = agg
    with open(eval_output_path, "w") as f:
        json.dump(eval_output, f, indent=2, default=str)

    print(f"\nSaved → {eval_output_path}")


if __name__ == "__main__":
    main()
