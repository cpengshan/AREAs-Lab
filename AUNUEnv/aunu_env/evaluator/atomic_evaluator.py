"""
Atomic unit evaluator for AUNUEnv.

Implements the same three-stage LLM pipeline as scripts/evaluation/user_interaction_judge.py:
  1. Decompose gold requirement into atomic units.
  2. Decompose predicted requirement into atomic units.
  3. Compare the two unit sets (LLM alignment).
  4. Compute TP/FP/FN scores in Python.

Gold unit decomposition results are cached by task_id to avoid redundant LLM calls.
"""

import logging
import os
from typing import Optional

from ..utils.llm import call_llm
from ..utils.jinja_utils import render_template
from ..utils.json_utils import parse_json_output
from .metrics import compute_scores

logger = logging.getLogger(__name__)

# Locate the project-level evaluation prompt templates.
_EVAL_PROMPT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "prompt")
)
_DECOMPOSE_TEMPLATE = os.path.join(_EVAL_PROMPT_DIR, "LLM_judge_decompose.jinja")
_COMPARE_TEMPLATE = os.path.join(_EVAL_PROMPT_DIR, "LLM_judge_compare.jinja")


class AtomicEvaluator:
    """Evaluates a predicted task requirement against a gold standard.

    Args:
        model_name: LLM model for decompose and compare calls.
        temperature: Sampling temperature (0 recommended for determinism).
        max_tokens: Max tokens for LLM responses.
        cache_gold_units: If True, gold unit decompositions are cached by task_id.
    """

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: int = 16384,
        cache_gold_units: bool = True,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._gold_cache: dict[str, list[str]] = {}  # task_id → gold_units
        self._cache_gold = cache_gold_units

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

        # Stage 1: decompose gold (with caching)
        cache_key = task_id or gold[:64]
        if self._cache_gold and cache_key in self._gold_cache:
            gold_units = self._gold_cache[cache_key]
            logger.debug(f"Using cached gold units for task '{cache_key}'")
        else:
            gold_units, cost = self._decompose(gold)
            total_cost += cost
            if self._cache_gold:
                self._gold_cache[cache_key] = gold_units

        # Stage 2: decompose predicted
        pred_units, cost = self._decompose(predicted)
        total_cost += cost

        # Stage 3: compare
        comparison, cost = self._compare(gold_units, pred_units)
        total_cost += cost

        counts, scores = compute_scores(comparison)

        return {
            "gold_units": gold_units,
            "predicted_units": pred_units,
            "comparison": comparison,
            "counts": counts,
            "scores": scores,
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
