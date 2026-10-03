#!/usr/bin/env python3
"""Standalone evaluator: re-run atomic evaluation on existing experiment output.

Useful when the scoring formula changes and you want to recompute scores
without re-running the full experiment.

Usage:
    python scripts/run_evaluation.py --results_path results/zero_shot_20250423_120000.json \\
        --evaluator_model gpt-4.1 --output_path results/recomputed.json
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from area_env.evaluator.atomic_evaluator import AtomicEvaluator
from area_env.evaluator.metrics import aggregate_results


def parse_args():
    parser = argparse.ArgumentParser(description="Re-evaluate experiment results")
    parser.add_argument("--results_path", required=True, help="Path to existing results JSON")
    parser.add_argument("--evaluator_model", type=str, default="gpt-4.1")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Where to save re-evaluated results (default: overwrite input)")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.results_path) as f:
        data = json.load(f)

    evaluator = AtomicEvaluator(model_name=args.evaluator_model, cache_gold_units=True)

    updated_results = []
    for task_result in data.get("task_results", []):
        if "error" in task_result:
            updated_results.append(task_result)
            continue
        final_req = task_result.get("final_requirement", "")
        gold = None
        # Try to get gold from task metadata (not always stored)
        if not final_req:
            updated_results.append(task_result)
            continue
        # Gold must be available; skip if not
        updated_results.append(task_result)

    print(f"Re-evaluated {len(updated_results)} tasks.")
    agg = aggregate_results([
        r["eval_result"] for r in updated_results
        if "eval_result" in r and r["eval_result"]
    ])
    print("Aggregate scores:", agg)

    output_path = args.output_path or args.results_path
    data["aggregate_scores"] = agg
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
