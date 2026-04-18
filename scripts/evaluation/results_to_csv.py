#!/usr/bin/env python3
"""
Convert experiment results to a single CSV for comparison.

Merges the args block, output.json (per-persona/task task_requirement_final
and ground_truth), and eval_results.json (LLM judge scores and counts) into
one flat CSV with one row per persona × task.

Usage:
    python scripts/evaluation/results_to_csv.py --id <exp_id> [--strategy <strategy>] [--dataset <dataset>]
"""

import argparse
import csv
import json
import sys
from pathlib import Path


SCORE_COLS = [
    "completeness",
    "alignment",
    "completeness_alignment",
    "faithfulness",
    "constraint_preservation",
]

COUNT_COLS = [
    "n_predicted_units",
    "n_ground_truth_units",
    "n_matched_pairs",
    "n_missing_units",
    "n_hallucinated_units",
    "n_misaligned_units",
    "n_critical_units",
    "n_critical_matched",
]


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def build_rows(exp_dir: Path, exp_id: str, strategy: str) -> list:
    """Build rows from a single experiment directory."""
    output_path = exp_dir / "output.json"
    eval_path = exp_dir / "eval_results.json"

    for p in (output_path, eval_path):
        if not p.exists():
            print(f"Warning: required file not found, skipping: {p}", file=sys.stderr)
            return []

    output_data = load_json(output_path)
    eval_data = load_json(eval_path)

    exp_args = output_data.get("args", {})
    if "persona" in exp_args and isinstance(exp_args["persona"], list):
        exp_args["persona_list"] = " ".join(str(p) for p in exp_args["persona"])

    rows = []
    for persona_id in sorted(
        (k for k in output_data if k != "args"), key=lambda x: int(x)
    ):
        tasks = output_data[persona_id]
        persona_eval = eval_data.get(persona_id, {})

        for task_key in sorted(tasks.keys()):
            out_entry = tasks[task_key]
            eval_entry = persona_eval.get(task_key, {})

            row = {}

            row["experiment_id"] = exp_id
            row["strategy"] = strategy
            for col in ("strategy_aunu", "aunu_model", "mimic_model", "dataset", "input_type", "max_turns"):
                row[col] = exp_args.get(col, out_entry.get(col, ""))

            row["persona"] = persona_id
            row["task_key"] = task_key
            row["task_id"] = out_entry.get("task_id", "")
            row["task_requirement_final"] = out_entry.get("task_requirement_final", "")
            row["ground_truth"] = out_entry.get("ground_truth", "")

            scores = eval_entry.get("scores", {})
            for col in SCORE_COLS:
                row[col] = scores.get(col, "")

            counts = eval_entry.get("counts", {})
            for col in COUNT_COLS:
                row[col] = counts.get(col, "")

            row["summary"] = eval_entry.get("summary", "")
            row["eval_error"] = eval_entry.get("error", "")

            rows.append(row)

    return rows


def find_highest_experiment(strategy_dir: Path) -> tuple[Path, str] | None:
    """Return (exp_dir, exp_id) for the highest-numbered Experiment with eval_results.json."""
    candidates = []
    for d in strategy_dir.iterdir():
        if d.is_dir() and d.name.startswith("Experiment"):
            id_str = d.name[len("Experiment"):]
            if id_str.isdigit() and (d / "eval_results.json").exists():
                candidates.append((int(id_str), d, id_str))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    _, exp_dir, exp_id = candidates[0]
    return exp_dir, exp_id


def main():
    parser = argparse.ArgumentParser(
        description="Merge experiment files into a single comparison CSV."
    )
    parser.add_argument("--id", help="Experiment ID (e.g. 12); ignored when --combine is set")
    parser.add_argument("--strategy", default="user", help="Strategy subfolder (default: user); ignored when --combine is set")
    parser.add_argument("--dataset", default="ccdv_patent-classification", help="Dataset folder name")
    parser.add_argument("--combine", action="store_true",
                        help="Combine highest-experiment eval_results.json from all strategies into one CSV under the dataset folder")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    dataset_dir = repo_root / "results" / args.dataset

    if args.combine:
        all_rows = []
        for strategy_dir in sorted(dataset_dir.iterdir()):
            if not strategy_dir.is_dir():
                continue
            result = find_highest_experiment(strategy_dir)
            if result is None:
                print(f"No valid experiment found in {strategy_dir}, skipping.", file=sys.stderr)
                continue
            exp_dir, exp_id = result
            print(f"  {strategy_dir.name}: using Experiment{exp_id}")
            rows = build_rows(exp_dir, exp_id, strategy_dir.name)
            all_rows.extend(rows)

        if not all_rows:
            print("No rows to write.", file=sys.stderr)
            sys.exit(1)

        csv_out_path = dataset_dir / "combined_results.csv"
        fieldnames = list(all_rows[0].keys())
        with open(csv_out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)

        print(f"CSV saved to: {csv_out_path} ({len(all_rows)} rows)")
        return

    # Single-experiment mode (original behavior)
    if not args.id:
        print("Error: --id is required unless --combine is set.", file=sys.stderr)
        sys.exit(1)

    exp_dir = dataset_dir / args.strategy / f"Experiment{args.id}"
    csv_out_path = exp_dir / "results.csv"

    rows = build_rows(exp_dir, args.id, args.strategy)
    if not rows:
        print("No rows to write.", file=sys.stderr)
        sys.exit(1)

    fieldnames = list(rows[0].keys())
    with open(csv_out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV saved to: {csv_out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
