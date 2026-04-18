#!/usr/bin/env python3
"""
Recompute scores from already-evaluated eval_results.json without re-running
the LLM pipeline. Reads the existing comparison data (matched_pairs,
predicted_units, ground_truth_units, etc.) and reapplies compute_scores().

Usage:
    python scripts/evaluation/recompute_scores.py \
        --id 1 --strategy zero_shot --dataset ccdv_mediasum

    # or point directly at a file:
    python scripts/evaluation/recompute_scores.py --file results/.../eval_results.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from user_interaction_judge import compute_scores


def recompute(path: Path, dry_run: bool = False) -> None:
    with open(path) as f:
        eval_results = json.load(f)

    changed = 0
    for persona_id, tasks in eval_results.items():
        if not isinstance(tasks, dict):
            continue
        for task_key, entry in tasks.items():
            if not isinstance(entry, dict) or "error" in entry:
                continue
            if "matched_pairs" not in entry and "ground_truth_units" not in entry:
                print(f"[{persona_id}/{task_key}] No comparison data — skipping.")
                continue

            old_scores = entry.get("scores", {})
            counts, scores = compute_scores(entry)

            if scores != old_scores:
                print(f"[{persona_id}/{task_key}] CHANGED")
                print(f"  old: {old_scores}")
                print(f"  new: {scores}")
                changed += 1
            else:
                print(f"[{persona_id}/{task_key}] unchanged: {scores}")

            entry["counts"] = counts
            entry["scores"] = scores

    if dry_run:
        print(f"\nDry run — {changed} entries would change. File not written.")
        return

    with open(path, "w") as f:
        json.dump(eval_results, f, indent=2)

    print(f"\nDone. {changed} entries updated. Saved to: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Recompute scores in eval_results.json without re-running LLMs."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", help="Direct path to eval_results.json")
    group.add_argument("--id", help="Experiment ID (e.g. 1); also requires --strategy and --dataset")

    parser.add_argument("--strategy", help="Strategy name (e.g. zero_shot, user, data)")
    parser.add_argument("--dataset", default="ccdv_patent-classification", help="Dataset folder name")
    parser.add_argument("--dry-run", action="store_true", help="Print what would change without writing")
    args = parser.parse_args()

    if args.id:
        if not args.strategy:
            print("Error: --strategy is required when using --id", file=sys.stderr)
            sys.exit(1)
        repo_root = Path(__file__).resolve().parents[2]
        path = repo_root / "results" / args.dataset / args.strategy / f"Experiment{args.id}" / "eval_results.json"
    else:
        path = Path(args.file)

    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    recompute(path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
