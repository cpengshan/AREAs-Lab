#!/usr/bin/env python3
"""Post-process existing eval_results.json files to add subcategory scores.

For each eval entry, classifies ground_truth_units and predicted_units as
user_specified / data_specified using LLM_judge_classify.jinja, then computes
per-subcategory precision, recall, and F1.

Uses experiment5.json unit classifications as a cache for ground-truth units
when an exact match is found; otherwise falls back to calling the LLM.

Usage:
    python agent/add_subcategory_scores.py \\
        --strategy data_interaction_v2 \\
        [--model gpt-4.1] \\
        [--dry_run]
"""

import argparse
import json
import logging
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _REPO_ROOT)

from AUNUEnv.aunu_env.utils.llm import call_llm
from AUNUEnv.aunu_env.utils.jinja_utils import render_template
from AUNUEnv.aunu_env.utils.json_utils import parse_json_output
from AUNUEnv.aunu_env.evaluator.metrics import compute_subcategory_scores

logger = logging.getLogger(__name__)

_CLASSIFY_TEMPLATE = os.path.join(
    _REPO_ROOT, "AUNUEnv", "aunu_env", "evaluator", "prompt", "LLM_judge_classify.jinja"
)
_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
_EXP5_PATH = os.path.join(_RESULTS_DIR, "data_analysis", "experiment5.json")

# Map result dir name → experiment5 dataset_id
_DIR_TO_DATASET = {
    "alexfabbri_multi_news": "alexfabbri/multi_news",
    "santoshtyss_uk_legislation": "santoshtyss/uk_legislation",
    "starmpcc_Asclepius-Synthetic-Clinical-Notes": "starmpcc/Asclepius-Synthetic-Clinical-Notes",
    "thu-coai_esconv": "thu-coai/esconv",
}


def _build_exp5_cache(exp5_path: str) -> dict:
    """Build a lookup: (dataset_id, user_key, task_id) → {unit: category}."""
    cache = {}
    try:
        with open(exp5_path) as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.warning(f"experiment5.json not found at {exp5_path}; no caching.")
        return cache
    for r in data.get("task_results", []):
        key = (r["dataset_id"], r["user_key"], r["task_id"])
        cache[key] = {c["unit"]: c["category"] for c in r.get("classifications", [])}
    return cache


def _classify_units(units: list[str], model: str) -> tuple[dict[str, str], float]:
    """Call LLM_judge_classify.jinja and return {unit: category}, cost."""
    if not units:
        return {}, 0.0
    prompt = render_template(
        _CLASSIFY_TEMPLATE,
        atomic_units="\n".join(f"- {u}" for u in units),
    )
    result = call_llm(model, prompt, max_tokens=8192, temperature=0.0)
    parsed = parse_json_output(result["output"])
    classifications = parsed.get("classifications", [])
    categories = {
        c["unit"]: c["category"]
        for c in classifications
        if "unit" in c and "category" in c
    }
    return categories, result.get("cost", 0.0)


def _process_eval_file(
    eval_path: str,
    dataset_dir: str,
    exp5_cache: dict,
    model: str,
    dry_run: bool,
) -> float:
    """Add subcategory_scores to each entry in eval_path. Returns total LLM cost."""
    dataset_id = _DIR_TO_DATASET.get(dataset_dir)

    with open(eval_path) as f:
        eval_data = json.load(f)

    total_cost = 0.0
    changed = False

    for persona_key, tasks in eval_data.items():
        if not isinstance(tasks, dict):
            continue
        for task_key, entry in tasks.items():
            if not isinstance(entry, dict):
                continue
            if "subcategory_scores" in entry:
                logger.debug(f"  {persona_key}/{task_key}: already has subcategory_scores, skipping.")
                continue

            gt_units = entry.get("ground_truth_units", [])
            pred_units = entry.get("predicted_units", [])

            # Try to get GT classifications from experiment5 cache
            task_num = int(task_key.split("_")[-1])
            user_key = f"user_{persona_key}"
            cache_key = (dataset_id, user_key, task_num) if dataset_id else None
            cached_gt = exp5_cache.get(cache_key, {}) if cache_key else {}

            # Check coverage: only use cache if all GT units are covered
            gt_covered = all(u in cached_gt for u in gt_units)
            if gt_covered and gt_units:
                gold_categories = {u: cached_gt[u] for u in gt_units}
                gt_cost = 0.0
                logger.info(f"  {persona_key}/{task_key}: GT from cache ({len(gt_units)} units)")
            else:
                logger.info(f"  {persona_key}/{task_key}: classifying GT via LLM ({len(gt_units)} units)")
                gold_categories, gt_cost = _classify_units(gt_units, model)
                total_cost += gt_cost

            logger.info(f"  {persona_key}/{task_key}: classifying predicted via LLM ({len(pred_units)} units)")
            pred_categories, pred_cost = _classify_units(pred_units, model)
            total_cost += pred_cost

            # Reconstruct comparison dict needed by compute_subcategory_scores
            comparison = {
                "ground_truth_units": gt_units,
                "predicted_units": pred_units,
                "matched_pairs": entry.get("matched_pairs", []),
            }

            subcategory_scores = compute_subcategory_scores(comparison, gold_categories, pred_categories)

            entry["gold_categories"] = gold_categories
            entry["pred_categories"] = pred_categories
            entry["subcategory_scores"] = subcategory_scores
            changed = True

            logger.info(
                f"  {persona_key}/{task_key}: "
                f"user p={subcategory_scores['user_specified']['precision']} "
                f"r={subcategory_scores['user_specified']['recall']} "
                f"f1={subcategory_scores['user_specified']['f1']}  |  "
                f"data p={subcategory_scores['data_specified']['precision']} "
                f"r={subcategory_scores['data_specified']['recall']} "
                f"f1={subcategory_scores['data_specified']['f1']}"
            )

    if changed and not dry_run:
        with open(eval_path, "w") as f:
            json.dump(eval_data, f, indent=2)
        logger.info(f"  Saved → {eval_path}")
    elif dry_run:
        logger.info(f"  [dry_run] would write → {eval_path}")

    return total_cost


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="data_interaction_v2",
                        help="Strategy sub-directory name (default: data_interaction_v2)")
    parser.add_argument("--model", default="gpt-4.1", help="LLM model for classification")
    parser.add_argument("--dry_run", action="store_true",
                        help="Classify and compute scores but do not write files")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    exp5_cache = _build_exp5_cache(_EXP5_PATH)
    logger.info(f"Loaded experiment5 cache: {len(exp5_cache)} task entries")

    total_cost = 0.0
    for dataset_dir in sorted(os.listdir(_RESULTS_DIR)):
        strategy_dir = os.path.join(_RESULTS_DIR, dataset_dir, args.strategy)
        if not os.path.isdir(strategy_dir):
            continue
        for exp_name in sorted(os.listdir(strategy_dir)):
            eval_path = os.path.join(strategy_dir, exp_name, "eval_results.json")
            if not os.path.exists(eval_path):
                continue
            logger.info(f"Processing {dataset_dir}/{args.strategy}/{exp_name}/eval_results.json")
            cost = _process_eval_file(eval_path, dataset_dir, exp5_cache, args.model, args.dry_run)
            total_cost += cost

    logger.info(f"\nDone. Total classification cost: ${total_cost:.5f}")


if __name__ == "__main__":
    main()
