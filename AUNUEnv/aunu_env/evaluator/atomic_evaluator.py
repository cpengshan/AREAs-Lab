"""
Atomic unit evaluator for AUNUEnv.

Implements the same three-stage LLM pipeline as scripts/evaluation/user_interaction_judge.py:
  1. Decompose gold requirement into atomic units.
  2. Decompose predicted requirement into atomic units.
  3. Compare the two unit sets (LLM alignment).
  4. Compute TP/FP/FN scores in Python.

Gold unit decomposition results are cached by task_id to avoid redundant LLM calls.
"""

import json
import logging
import os
from typing import Optional

from ..utils.llm import call_llm
from ..utils.jinja_utils import render_template
from ..utils.json_utils import parse_json_output
from .metrics import compute_scores, compute_subcategory_scores

logger = logging.getLogger(__name__)

# Locate the project-level evaluation prompt templates.
_EVAL_PROMPT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "prompt")
)
_DECOMPOSE_TEMPLATE = os.path.join(_EVAL_PROMPT_DIR, "LLM_judge_decompose.jinja")
_COMPARE_TEMPLATE = os.path.join(_EVAL_PROMPT_DIR, "LLM_judge_compare.jinja")
_CLASSIFY_TEMPLATE = os.path.join(_EVAL_PROMPT_DIR, "LLM_judge_classify.jinja")

_DEFAULT_CACHE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "gold_cache.json")
)


def _load_persistent_cache(path: str) -> dict:
    """Load cache from *path*.

    Supports two on-disk formats:
    - New format: {task_id: [{unit, category}, ...]}
    - Legacy format: {task_id: {units: [...], categories: {...}}}

    Always returns the internal representation: {task_id: {units, categories}}.
    """
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError):
        logger.warning(f"{path} is corrupted or unreadable; starting fresh.")
        return {}

    result = {}
    for task_id, value in raw.items():
        if isinstance(value, list):
            # New format: [{unit, category}, ...]
            units = [item["unit"] for item in value if "unit" in item]
            categories = {item["unit"]: item["category"] for item in value if "unit" in item and "category" in item}
            result[task_id] = {"units": units, "categories": categories}
        else:
            # Legacy format: {units: [...], categories: {...}}
            result[task_id] = value
    return result


def _save_persistent_cache(cache: dict, path: str) -> None:
    """Save cache to *path* using the new list-of-{unit,category} format."""
    serializable = {}
    for task_id, entry in cache.items():
        units = entry.get("units", [])
        categories = entry.get("categories", {})
        serializable[task_id] = [
            {"unit": u, "category": categories.get(u, "unknown")}
            for u in units
        ]
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)
    except OSError as e:
        logger.warning(f"Could not write {path}: {e}")


class AtomicEvaluator:
    """Evaluates a predicted task requirement against a gold standard.

    Args:
        model_name: LLM model for decompose and compare calls.
        temperature: Sampling temperature (0 recommended for determinism).
        max_tokens: Max tokens for LLM responses.
        cache_gold_units: If True, gold unit decompositions are cached by task_id.
        cache_path: Path to the persistent gold cache file.  Defaults to
            ``gold_cache.json`` next to this module.  Pass
            ``{dataset_dir}/ground_truth_decompose.json`` to store the cache
            alongside the dataset's synthesized outputs.
    """

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: int = 16384,
        cache_gold_units: bool = True,
        cache_path: Optional[str] = None,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._cache_gold = cache_gold_units
        self._cache_path = cache_path or _DEFAULT_CACHE_PATH
        # In-memory cache: task_id → {units, categories}
        self._gold_cache: dict[str, dict] = {}
        # Persistent cache loaded from disk (superset of in-memory at startup)
        self._persistent_cache: dict = _load_persistent_cache(self._cache_path)
        logger.debug(f"Loaded {len(self._persistent_cache)} entries from {self._cache_path}")

    def evaluate(
        self,
        predicted: str,
        gold: str,
        task_id: Optional[str] = None,
    ) -> dict:
        """Evaluate predicted requirement against gold requirement.

        Args:
            predicted: The agent's final predicted task requirement.
            gold: The ground-truth task requirement.
            task_id: Optional task identifier for gold unit caching.

        Returns:
            Dict with keys:
                gold_units (list): Atomic units from gold requirement.
                predicted_units (list): Atomic units from predicted requirement.
                comparison (dict): Raw LLM alignment output.
                counts (dict): TP/FP/FN and unit counts.
                scores (dict): precision, recall, f1, and legacy scores.
                cost (float): Total LLM cost for this evaluation.
        """
        total_cost = 0.0

        # Stage 1: decompose gold + classify gold (with persistent caching)
        cache_key = task_id or gold[:64]
        cached = (
            self._gold_cache.get(cache_key)
            or (self._persistent_cache.get(cache_key) if self._cache_gold else None)
        )
        if cached:
            gold_units = cached["units"]
            gold_categories = cached["categories"]
            logger.debug(f"Using cached gold decomposition for task '{cache_key}'")
        else:
            gold_units, cost = self._decompose(gold)
            total_cost += cost
            gold_categories, cost = self._classify(gold_units)
            total_cost += cost
            if self._cache_gold:
                entry = {"units": gold_units, "categories": gold_categories}
                self._gold_cache[cache_key] = entry
                self._persistent_cache[cache_key] = entry
                _save_persistent_cache(self._persistent_cache, self._cache_path)
                logger.debug(f"Cached gold decomposition for task '{cache_key}'")

        # Stage 2: decompose predicted
        pred_units, cost = self._decompose(predicted)
        total_cost += cost

        # Stage 3: compare
        comparison, cost = self._compare(gold_units, pred_units)
        total_cost += cost

        counts, scores = compute_scores(comparison)

        # Stage 4: classify predicted units
        pred_categories, cost = self._classify(pred_units)
        total_cost += cost

        subcategory_scores = compute_subcategory_scores(comparison, gold_categories, pred_categories)

        return {
            "gold_units": gold_units,
            "predicted_units": pred_units,
            "comparison": comparison,
            "counts": counts,
            "scores": scores,
            "gold_categories": gold_categories,
            "pred_categories": pred_categories,
            "subcategory_scores": subcategory_scores,
            "cost": total_cost,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _decompose(self, requirement: str) -> tuple[list[str], float]:
        prompt = render_template(_DECOMPOSE_TEMPLATE, task_requirement=requirement)
        result = call_llm(
            self.model_name, prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        parsed = parse_json_output(result["output"])
        units = parsed.get("atomic_units", [])
        if not isinstance(units, list):
            units = []
        return units, result.get("cost", 0.0)

    def _classify(self, units: list[str]) -> tuple[dict[str, str], float]:
        """Classify units as user_specified or data_specified.

        Returns:
            (categories, cost) where categories maps unit string → category.
        """
        if not units:
            return {}, 0.0
        prompt = render_template(
            _CLASSIFY_TEMPLATE,
            atomic_units="\n".join(f"- {u}" for u in units),
        )
        result = call_llm(
            self.model_name, prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        parsed = parse_json_output(result["output"])
        classifications = parsed.get("classifications", [])
        categories = {c["unit"]: c["category"] for c in classifications if "unit" in c and "category" in c}
        return categories, result.get("cost", 0.0)

    def _compare(
        self, gold_units: list[str], pred_units: list[str]
    ) -> tuple[dict, float]:
        prompt = render_template(
            _COMPARE_TEMPLATE,
            ground_truth_units="\n".join(f"- {u}" for u in gold_units),
            predicted_units="\n".join(f"- {u}" for u in pred_units),
        )
        result = call_llm(
            self.model_name, prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        parsed = parse_json_output(result["output"])
        return parsed, result.get("cost", 0.0)
