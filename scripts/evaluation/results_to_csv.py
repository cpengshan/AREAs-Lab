#!/usr/bin/env python3
"""
Convert experiment results to a single CSV for comparison.

Merges exp_setting.csv (experiment metadata), output.json (per-persona
task_requirement_final and ground_truth), and eval_results.json (LLM judge
scores and counts) into one flat CSV with one row per persona.

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


def load_exp_setting(path: Path) -> dict:
    """Return the experiment settings from exp_setting.json."""
    with open(path) as f:
        data = json.load(f)
    # Normalise persona list to a space-separated string for CSV readability
    if "persona" in data and isinstance(data["persona"], list):
        data["persona"] = " ".join(str(p) for p in data["persona"])
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Merge experiment files into a single comparison CSV."
    )
    parser.add_argument("--id", required=True, help="Experiment ID (e.g. 8)")
    parser.add_argument("--strategy", default="user", help="Strategy subfolder (default: user)")
    parser.add_argument("--dataset", default="ccdv_patent-classification", help="Dataset folder name")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    exp_dir = repo_root / "results" / args.dataset / args.strategy / f"Experiment{args.id}"

    output_path = exp_dir / "output.json"
    eval_path = exp_dir / "eval_results.json"
    setting_path = exp_dir / "exp_setting.json"
    csv_out_path = exp_dir / "results.csv"

    for p in (output_path, eval_path):
        if not p.exists():
            print(f"Error: required file not found: {p}", file=sys.stderr)
            sys.exit(1)

    output_data = load_json(output_path)
    eval_data = load_json(eval_path)
    setting = load_exp_setting(setting_path) if setting_path.exists() else {}

    # Build one row per persona ID
    rows = []
    for entry_id in sorted(output_data.keys(), key=lambda x: int(x)):
        out_entry = output_data[entry_id]
        eval_entry = eval_data.get(entry_id, {})

        row = {}

        # Experiment metadata (same for all rows)
        row["experiment_id"] = args.id
        for col in ("strategy_aunu", "aunu_model", "mimic_model", "dataset", "input_type", "max_turns"):
            row[col] = setting.get(col, out_entry.get(col, ""))

        # Per-persona fields from output.json
        row["persona"] = out_entry.get("persona", entry_id)
        row["task_requirement_final"] = out_entry.get("task_requirement_final", "")
        row["ground_truth"] = out_entry.get("ground_truth", "")

        # Eval scores
        scores = eval_entry.get("scores", {})
        for col in SCORE_COLS:
            row[col] = scores.get(col, "")

        # Eval counts
        counts = eval_entry.get("counts", {})
        for col in COUNT_COLS:
            row[col] = counts.get(col, "")

        # Summary and error flag
        row["summary"] = eval_entry.get("summary", "")
        row["eval_error"] = eval_entry.get("error", "")

        rows.append(row)

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
