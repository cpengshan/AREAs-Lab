#!/usr/bin/env python3
"""
LLM-based judge for evaluating user interaction experiment results.

Two-stage pipeline:
  1. Decompose each requirement (predicted and ground truth) into atomic units
     using LLM_judge_decompose.jinja.
  2. Compare the two unit sets using LLM_judge_compare.jinja and compute
     scores in Python from the returned alignment data.
"""

import argparse
import json
import re
import sys
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.model_base import LLM


# ---------------------------------------------------------------------------
# Template helpers
# ---------------------------------------------------------------------------

def load_template(template_path: Path):
    env = Environment(loader=FileSystemLoader(str(template_path.parent)))
    return env.get_template(template_path.name)


def parse_json_response(response: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?\s*", "", response.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return json.loads(cleaned)


def call_llm(llm: LLM, prompt: str) -> tuple[str, float]:
    result = llm.generate(prompt, max_tokens=16384, temperature=0)
    # if not result["output"] or not result["output"].strip():
    print(f"results: {result}")
    return result["output"], result["cost"]


# ---------------------------------------------------------------------------
# Stage 1 – decompose a single requirement into atomic units
# ---------------------------------------------------------------------------

def decompose(llm: LLM, template, requirement: str) -> tuple[list[str], float]:
    prompt = template.render(task_requirement=requirement)
    response, cost = call_llm(llm, prompt)
    try:
        data = parse_json_response(response)
    except Exception as e:
        e.raw_response = f"[decompose] {response}"
        raise
    return data["atomic_units"], cost


# ---------------------------------------------------------------------------
# Stage 2 – compare two unit lists
# ---------------------------------------------------------------------------

def compare(llm: LLM, template, ground_truth_units: list[str], predicted_units: list[str]) -> tuple[dict, float]:
    prompt = template.render(
        ground_truth_units="\n".join(f"- {u}" for u in ground_truth_units),
        predicted_units="\n".join(f"- {u}" for u in predicted_units),
    )
    prompt_chars = len(prompt)
    prompt_tokens_est = prompt_chars // 4
    print(f"  [compare] prompt length: {prompt_chars} chars (~{prompt_tokens_est} tokens)")
    response, cost = call_llm(llm, prompt)
    if not response or not response.strip():
        raise ValueError(
            f"compare stage returned an empty response (prompt was ~{prompt_tokens_est} tokens)\n"
        )
    try:
        data = parse_json_response(response)
    except Exception as e:
        e.raw_response = f"[compare] {response}"
        raise
    return data, cost


# ---------------------------------------------------------------------------
# Score computation (done in Python, not by the LLM)
# ---------------------------------------------------------------------------

def compute_scores(comparison: dict) -> dict:
    n_gt = len(comparison.get("ground_truth_units", []))
    n_pred = len(comparison.get("predicted_units", []))
    n_hallucinated = len(comparison.get("hallucinated_units", []))
    n_misaligned = len(comparison.get("misaligned_units", []))
    n_critical = len(comparison.get("critical_units", []))
    n_critical_missing = len(comparison.get("critical_missing", []))
    n_critical_matched = max(0, n_critical - n_critical_missing)

    # Each entry in matched_pairs has one ground_truth and one or more predicted.
    # Only count a pair as matched when predicted is non-empty (len > 0).
    matched_pairs = comparison.get("matched_pairs", [])
    n_matched_gt = sum(1 for p in matched_pairs if len(p.get("predicted", [])) != 0)
    

    # Derive missing from what is not matched, ignoring the LLM's missing_units list
    # which can overlap with matched_pairs (LLM double-counts).
    # This guarantees: n_matched_gt + n_missing == n_gt.
    matched_gt_set = {p["ground_truth"] for p in matched_pairs if len(p.get("predicted", [])) != 0}
    n_missing = sum(1 for u in comparison.get("ground_truth_units", []) if u not in matched_gt_set)

    # Derive matched predicted units from matched_pairs to avoid LLM double-counting
    # in hallucinated_units (LLM often lists matched units there too).
    matched_predicted = set()
    for p in matched_pairs:
        for u in p.get("predicted", []):
            matched_predicted.add(u)
    all_predicted = comparison.get("predicted_units", [])
    n_hallucinated = sum(1 for u in all_predicted if u not in matched_predicted)

    completeness = n_matched_gt / n_gt if n_gt > 0 else 0.0
    alignment = (
        n_matched_gt / (n_matched_gt + n_misaligned)
        if (n_matched_gt + n_misaligned) > 0
        else 0.0
    )
    faithfulness = (
        1.0 - n_hallucinated / n_pred if n_pred > 0 else 0.0
    )
    constraint_preservation = (
        n_critical_matched / n_critical if n_critical > 0 else 0.0
    )

    counts = {
        "n_predicted_units": n_pred,
        "n_ground_truth_units": n_gt,
        "n_matched_gt_units": n_matched_gt,
        "n_missing_units": n_missing,
        "n_hallucinated_units": n_hallucinated,
        "n_misaligned_units": n_misaligned,
        "n_critical_units": n_critical,
        "n_critical_matched": n_critical_matched,
    }

    scores = {
        "completeness": round(completeness, 4),
        "alignment": round(alignment, 4),
        "faithfulness": round(faithfulness, 4),
        "constraint_preservation": round(constraint_preservation, 4),
    }

    return counts, scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_gt_units_cache(dataset_dir: Path) -> dict:
    """Load pre-computed ground truth atomic units if available."""
    cache_path = dataset_dir / "ground_truth_units.json"
    if cache_path.exists():
        with open(cache_path) as f:
            cache = json.load(f)
        print(f"Loaded pre-computed ground truth units from: {cache_path}")
        return cache
    return {}


def main():
    parser = argparse.ArgumentParser(
        description="LLM judge for user interaction experiment evaluation."
    )
    parser.add_argument("--id", required=True, help="Experiment ID (e.g. 3)")
    parser.add_argument("--model", required=True, help="Model ID (e.g. gpt-4.1-mini)")
    parser.add_argument(
        "--ids",
        nargs="+",
        default=None,
        help="Subset of persona IDs to evaluate. Defaults to all.",
    )
    parser.add_argument(
        "--strategy",
        required=True,
        help="Strategy name (e.g. zero_shot, user, data, hybrid, persona).",
    )
    parser.add_argument(
        "--dataset",
        default="ccdv_patent-classification",
        help="Dataset folder name under results/.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    dataset_dir = repo_root / "results" / args.dataset
    experiment_dir = dataset_dir / args.strategy / f"Experiment{args.id}"
    input_path = experiment_dir / "output.json"
    output_path = experiment_dir / "eval_results.json"
    prompts_dir = repo_root / "prompts" / "Evaluation"

    for p in (input_path, prompts_dir / "LLM_judge_decompose.jinja", prompts_dir / "LLM_judge_compare.jinja"):
        if not p.exists():
            print(f"Error: file not found: {p}", file=sys.stderr)
            sys.exit(1)

    with open(input_path) as f:
        data = json.load(f)

    eval_results = {}
    if output_path.exists():
        with open(output_path) as f:
            eval_results = json.load(f)

    ids_to_run = args.ids if args.ids is not None else [k for k in data if k != "args"]

    gt_units_cache = load_gt_units_cache(dataset_dir)
    decompose_tpl = load_template(prompts_dir / "LLM_judge_decompose.jinja")
    compare_tpl = load_template(prompts_dir / "LLM_judge_compare.jinja")
    llm = LLM(args.model)
    total_cost = 0.0

    for persona_id in ids_to_run:
        if persona_id not in data:
            print(f"[persona {persona_id}] Skipping: ID not found in input file")
            continue

        if persona_id not in eval_results:
            eval_results[persona_id] = {}

        for task_key, entry in data[persona_id].items():
            label = f"persona {persona_id} / {task_key}"

            if task_key in eval_results[persona_id] and "error" not in eval_results[persona_id][task_key]:
                print(f"[{label}] Already evaluated. Skipping.")
                continue

            task_req = entry.get("task_requirement_final", "")
            ground_truth = entry.get("ground_truth", "")

            if not task_req or not ground_truth:
                print(f"[{label}] Skipping: missing task_requirement_final or ground_truth")
                eval_results[persona_id][task_key] = {"error": "missing required fields"}
                continue

            print(f"[{label}] Evaluating with model {args.model} ...")

            try:
                # Stage 1a: load pre-computed GT units or decompose on the fly
                cached_gt = gt_units_cache.get(persona_id, {}).get(task_key, {})
                if cached_gt.get("atomic_units"):
                    gt_units = cached_gt["atomic_units"]
                    cost1 = 0.0
                    print(f"[{label}] Using cached GT units ({len(gt_units)} units).")
                else:
                    print(f"[{label}] No cached GT units found; decomposing ground truth now.")
                    gt_units, cost1 = decompose(llm, decompose_tpl, ground_truth)
                    total_cost += cost1
                    # Persist newly computed GT units to cache
                    gt_units_cache.setdefault(persona_id, {})[task_key] = {
                        "ground_truth": ground_truth,
                        "atomic_units": gt_units,
                    }
                    cache_path = dataset_dir / "ground_truth_units.json"
                    with open(cache_path, "w") as _cf:
                        json.dump(gt_units_cache, _cf, indent=2)

                # Stage 1b: decompose predicted requirement
                pred_units, cost2 = decompose(llm, decompose_tpl, task_req)
                total_cost += cost2

                # Stage 2: compare unit sets
                comparison, cost3 = compare(llm, compare_tpl, gt_units, pred_units)
                total_cost += cost3

                # Stage 3: compute scores in Python
                counts, scores = compute_scores(comparison)

                result = {**comparison, "counts": counts, "scores": scores}
                eval_results[persona_id][task_key] = result
                print(f"[{label}] Scores: {scores} | Cost so far: ${total_cost:.6f}")

            except json.JSONDecodeError as e:
                raw = getattr(e, "raw_response", "<no response captured>")
                print("*" * 80)
                print(f"[{label}] Failed to parse LLM response as JSON: {e}")
                print(f"LLM response:\n{raw}")
                print("*" * 80)
                eval_results[persona_id][task_key] = {"error": "json_parse_error"}
            except Exception as e:
                raw = getattr(e, "raw_response", None)
                print(f"[{label}] Error: {e}")
                if raw:
                    print(f"LLM response:\n{raw}")
                eval_results[persona_id][task_key] = {"error": str(e)}

            finally:
                with open(output_path, "w") as f:
                    json.dump(eval_results, f, indent=2)

    print(f"\nEvaluation results saved to: {output_path}")
    print(f"Total cost: ${total_cost:.6f}")


if __name__ == "__main__":
    main()
