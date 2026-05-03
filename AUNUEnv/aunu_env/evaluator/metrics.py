"""Score computation from LLM comparison alignment data.

Reuses the same TP/FP/FN definitions as scripts/evaluation/user_interaction_judge.py.
"""

_SUBCATEGORIES = ("user_specified", "data_specified")


def compute_subcategory_scores(
    comparison: dict,
    gold_categories: dict[str, str],
    pred_categories: dict[str, str],
) -> dict:
    """Compute per-subcategory precision, recall, F1 and unit counts.

    Args:
        comparison: Same dict passed to compute_scores (has matched_pairs,
            ground_truth_units, predicted_units).
        gold_categories: Maps each gold unit string → "user_specified" | "data_specified".
        pred_categories: Maps each predicted unit string → "user_specified" | "data_specified".

    Returns:
        Dict with keys "user_specified" and "data_specified", each containing:
            n_gt, n_pred, tp, precision, recall, f1.
    """
    matched_pairs = comparison.get("matched_pairs", [])
    gold_units = comparison.get("ground_truth_units", [])
    pred_units = comparison.get("predicted_units", [])

    # GT units matched (have at least one predicted)
    matched_gt = {p["ground_truth"] for p in matched_pairs if p.get("predicted")}

    result = {}
    for cat in _SUBCATEGORIES:
        gt_in_cat = [u for u in gold_units if gold_categories.get(u) == cat]
        pred_in_cat = [u for u in pred_units if pred_categories.get(u) == cat]

        n_gt = len(gt_in_cat)
        n_pred = len(pred_in_cat)
        tp = sum(1 for u in gt_in_cat if u in matched_gt)

        precision = tp / n_pred if n_pred > 0 else 0.0
        recall = tp / n_gt if n_gt > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        result[cat] = {
            "n_gt": n_gt,
            "n_pred": n_pred,
            "tp": tp,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
    return result


def compute_scores(comparison: dict) -> tuple[dict, dict]:
    """Compute precision, recall, F1, and legacy scores from comparison alignment.

    Args:
        comparison: Dict returned by the LLM compare stage with keys:
            ground_truth_units, predicted_units, matched_pairs,
            missing_units, hallucinated_units, misaligned_units, critical_units,
            critical_missing.

    Returns:
        (counts, scores) where both are dicts.
    """
    n_gt = len(comparison.get("ground_truth_units", []))
    n_pred = len(comparison.get("predicted_units", []))
    n_misaligned = len(comparison.get("misaligned_units", []))
    n_critical = len(comparison.get("critical_units", []))
    n_critical_missing = len(comparison.get("critical_missing", []))
    n_critical_matched = max(0, n_critical - n_critical_missing)

    matched_pairs = comparison.get("matched_pairs", [])

    # TP: one per covered ground-truth unit
    tp = sum(1 for p in matched_pairs if len(p.get("predicted", [])) != 0)

    matched_gt_set = {p["ground_truth"] for p in matched_pairs if len(p.get("predicted", [])) != 0}
    fn = sum(1 for u in comparison.get("ground_truth_units", []) if u not in matched_gt_set)
    fp = n_pred - tp

    matched_predicted = {u for p in matched_pairs for u in p.get("predicted", [])}
    n_hallucinated = sum(1 for u in comparison.get("predicted_units", []) if u not in matched_predicted)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    alignment = tp / (tp + n_misaligned) if (tp + n_misaligned) > 0 else 0.0
    constraint_preservation = n_critical_matched / n_critical if n_critical > 0 else 0.0

    counts = {
        "n_predicted_units": n_pred,
        "n_ground_truth_units": n_gt,
        "n_matched_gt_units": tp,
        "n_missing_units": fn,
        "n_hallucinated_units": n_hallucinated,
        "n_misaligned_units": n_misaligned,
        "n_critical_units": n_critical,
        "n_critical_matched": n_critical_matched,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }

    scores = {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "alignment": round(alignment, 4),
        "constraint_preservation": round(constraint_preservation, 4),
    }

    return counts, scores


def aggregate_results(eval_results: list[dict]) -> dict:
    """Compute mean scores across a list of per-task evaluation results.

    Args:
        eval_results: List of dicts, each with a 'scores' key.

    Returns:
        Dict with mean of each score key.
    """
    if not eval_results:
        return {}
    score_keys = list(eval_results[0].get("scores", {}).keys())
    aggregated = {}
    for key in score_keys:
        values = [r["scores"][key] for r in eval_results if key in r.get("scores", {})]
        aggregated[key] = round(sum(values) / len(values), 4) if values else 0.0
    aggregated["n_tasks"] = len(eval_results)
    return aggregated
