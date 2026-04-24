#!/usr/bin/env python3
"""
Recompute precision, recall, and F1 for every eval_results.json under results/.

Walks the full results tree, applies the current compute_scores() to every
persona/task entry that has comparison data (matched_pairs / ground_truth_units),
writes the updated scores back to the file, and prints a summary of what changed.

Usage:
    python scripts/evaluation/recompute_all_scores.py
    python scripts/evaluation/recompute_all_scores.py --dry-run
    python scripts/evaluation/recompute_all_scores.py --results-dir path/to/results
"""

import argparse
import json
import sys
from pathlib import Path

# Import only compute_scores — stub out the LLM dependency so the module
# loads even when openai/anthropic packages are not installed.
import importlib, types

def _import_compute_scores():
    # Stub every missing top-level import so user_interaction_judge loads.
    for mod in ("openai", "anthropic", "google", "google.generativeai", "litellm"):
        if mod not in sys.modules:
            sys.modules[mod] = types.ModuleType(mod)
    # Stub model.model_base
    for mod in ("model", "model.model_base"):
        if mod not in sys.modules:
            m = types.ModuleType(mod)
            m.LLM = object
            sys.modules[mod] = m

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    module = importlib.import_module("user_interaction_judge")
    return module.compute_scores

compute_scores = _import_compute_scores()


def recompute_file(path: Path, dry_run: bool) -> dict:
    """
    Recompute scores for every entry in one eval_results.json.
    Returns a summary dict: {changed, skipped, errors, path}.
    """
    with open(path) as f:
        data = json.load(f)

    changed = 0
    skipped = 0
    errors = 0

    for persona_id, tasks in data.items():
        if not isinstance(tasks, dict):
            continue
        for task_key, entry in tasks.items():
            if not isinstance(entry, dict) or "error" in entry:
                skipped += 1
                continue
            if "matched_pairs" not in entry and "ground_truth_units" not in entry:
                skipped += 1
                continue

            try:
                counts, scores = compute_scores(entry)
            except Exception as e:
                print(f"  ERROR [{persona_id}/{task_key}]: {e}")
                errors += 1
                continue

            old_scores = entry.get("scores", {})
            if scores != old_scores:
                changed += 1
                if not dry_run:
                    entry["counts"] = counts
                    entry["scores"] = scores

    if not dry_run and changed > 0:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    return {"path": path, "changed": changed, "skipped": skipped, "errors": errors}


def main():
    parser = argparse.ArgumentParser(
        description="Recompute precision/recall/F1 for all eval_results.json files."
    )
    parser.add_argument(
        "--results-dir",
        default=str(Path(__file__).resolve().parents[2] / "results"),
        help="Root results directory to search (default: repo/results)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without writing any files.",
    )
    args = parser.parse_args()

    results_root = Path(args.results_dir)
    eval_files = sorted(results_root.rglob("eval_results.json"))

    if not eval_files:
        print(f"No eval_results.json files found under {results_root}")
        sys.exit(0)

    print(f"Found {len(eval_files)} eval_results.json files under {results_root}")
    if args.dry_run:
        print("DRY RUN — no files will be written.\n")
    else:
        print()

    total_changed = 0
    total_skipped = 0
    total_errors = 0
    files_changed = 0

    for path in eval_files:
        rel = path.relative_to(results_root)
        summary = recompute_file(path, dry_run=args.dry_run)

        status = "changed" if summary["changed"] > 0 else "unchanged"
        print(f"[{status:9s}] {rel}  "
              f"(changed={summary['changed']}, skipped={summary['skipped']}, errors={summary['errors']})")

        total_changed += summary["changed"]
        total_skipped += summary["skipped"]
        total_errors += summary["errors"]
        if summary["changed"] > 0:
            files_changed += 1

    print()
    print("=" * 60)
    print(f"Files scanned : {len(eval_files)}")
    print(f"Files updated : {files_changed}" + (" (dry run)" if args.dry_run else ""))
    print(f"Entries changed : {total_changed}")
    print(f"Entries skipped : {total_skipped}  (no comparison data or error field)")
    print(f"Entries errored : {total_errors}")


if __name__ == "__main__":
    main()
