import argparse
import json
import os
import sys
from jinja2 import Environment, FileSystemLoader

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from model.model_base import LLM

PROMPT_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "../../prompts/evaluation")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "../../results")


def parse_args():
    parser = argparse.ArgumentParser(description="LLM Judge for AUNU task requirement evaluation")
    parser.add_argument(
        "--results_path",
        type=str,
        required=True,
        help="Path to the results JSON file (e.g. results/ccdv_patent-classification/zero_shot/gemini-3.1-flash-lite-preview.json)",
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default="gemini-3.1-flash-lite-preview",
        help="Model name to use as the LLM judge",
    )
    return parser.parse_args()


def evaluate_persona(persona_id: str, persona_result: dict, template, llm: LLM) -> dict:
    inferred = persona_result.get("task_requirement_final", "")
    ground_truth_raw = persona_result.get("ground_truth", "")
    ground_truth = ground_truth_raw if isinstance(ground_truth_raw, str) else ground_truth_raw.get("task_requirement", "")

    if not inferred or not ground_truth:
        return {"error": "Missing inferred or ground truth task requirement"}

    prompt = template.render(
        inferred_task_requirement=inferred,
        ground_truth_task_requirement=ground_truth,
    )

    raw = llm.generate(prompt)

    try:
        evaluation = json.loads(raw)
        evaluation["count_metrics"] = compute_count_metrics(evaluation)
    except json.JSONDecodeError:
        evaluation = {"raw_output": raw, "error": "Failed to parse LLM output as JSON"}

    return evaluation


def compute_count_metrics(evaluation: dict) -> dict:
    n_matched = len(evaluation.get("matched", []))
    n_missing = len(evaluation.get("missing", []))
    n_drifted = len(evaluation.get("drifted", []))
    n_hallucinated = len(evaluation.get("hallucinated", []))

    total_gt = n_matched + n_missing + n_drifted
    total_inferred = n_matched + n_hallucinated + n_drifted

    coverage = n_matched / total_gt if total_gt > 0 else 0.0
    precision = n_matched / total_inferred if total_inferred > 0 else 0.0
    f1 = (2 * coverage * precision / (coverage + precision)) if (coverage + precision) > 0 else 0.0

    return {
        "n_matched": n_matched,
        "n_missing": n_missing,
        "n_drifted": n_drifted,
        "n_hallucinated": n_hallucinated,
        "coverage": round(coverage, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
    }


def run_evaluation():
    args = parse_args()

    results_path = args.results_path
    if not os.path.isabs(results_path):
        results_path = os.path.join(os.path.dirname(__file__), "../..", results_path)

    with open(results_path) as f:
        all_results = json.load(f)

    env = Environment(loader=FileSystemLoader(PROMPT_TEMPLATE_DIR))
    template = env.get_template("task_requirement_diff.jinja")
    llm = LLM(args.judge_model)

    out_path = os.path.join(os.path.dirname(results_path), "LLM_Judge.json")

    existing = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)

    # Backfill count_metrics for any existing entries that are missing them
    for persona_id, entry in existing.items():
        evaluation = entry.get("evaluation", {})
        if "count_metrics" not in evaluation and "missing" in evaluation:
            evaluation["count_metrics"] = compute_count_metrics(evaluation)
            print(f"Backfilled count_metrics for persona {persona_id}.")

    for persona_id, persona_result in all_results.items():
        if persona_id in existing:
            print(f"Persona {persona_id} already evaluated. Skipping.")
            continue

        print(f"Evaluating persona {persona_id}...")
        evaluation = evaluate_persona(persona_id, persona_result, template, llm)
        existing[persona_id] = {
            "persona": persona_id,
            "judge_model": args.judge_model,
            "evaluation": evaluation,
        }

        print(f"  Saved evaluation for persona {persona_id}.")

    existing["aggregate"] = compute_aggregate(existing)

    with open(out_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\nAll evaluations saved to {out_path}")


def compute_aggregate(existing: dict) -> dict:
    persona_entries = {k: v for k, v in existing.items() if k != "aggregate"}
    if not persona_entries:
        return {}

    llm_score_keys = ["coverage", "precision", "faithfulness", "overall"]
    count_keys = ["n_matched", "n_missing", "n_drifted", "n_hallucinated", "coverage", "precision", "f1"]

    llm_totals = {k: [] for k in llm_score_keys}
    count_totals = {k: [] for k in count_keys}

    for entry in persona_entries.values():
        evaluation = entry.get("evaluation", {})
        scores = evaluation.get("scores", {})
        for k in llm_score_keys:
            if k in scores:
                llm_totals[k].append(scores[k])
        count_metrics = evaluation.get("count_metrics", {})
        for k in count_keys:
            if k in count_metrics:
                count_totals[k].append(count_metrics[k])

    def mean(lst):
        return round(sum(lst) / len(lst), 4) if lst else None

    return {
        "n_personas": len(persona_entries),
        "llm_scores_mean": {k: mean(v) for k, v in llm_totals.items()},
        "count_metrics_mean": {k: mean(v) for k, v in count_totals.items()},
    }


if __name__ == "__main__":
    run_evaluation()
