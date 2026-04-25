"""
AUNUEnv — core gymnasium-style environment for AUNU benchmark evaluation.

Interface mirrors gym.Env without requiring the gymnasium package:
  env.reset(task, user)  →  (observation, info)
  env.step(action)       →  (observation, reward, done, info)

Action space (dict-based):
  ask_user(question)
  inspect_data(n_samples, query)
  propose_requirement_update(updated_requirement)
  finish(final_requirement)
"""

import logging
import os
import random
from datetime import datetime, timezone
import pandas as pd
from typing import Optional

# Root of the AUNUEnv package's data directory: AUNUEnv/data/data_raw/
_DATA_RAW_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../data/data_raw")
)

from ..dataset.schema import TaskInstance
from ..users import MimicUser
from ..evaluator.atomic_evaluator import AtomicEvaluator
from .actions import (
    ACTION_ASK_USER, ACTION_INSPECT_DATA,
    ACTION_PROPOSE_UPDATE, ACTION_FINISH,
    validate_action,
)
from .state import EpisodeState

logger = logging.getLogger(__name__)


class AUNUEnv:
    """Interactive benchmark environment for AI-Assisted User Needs Understanding.

    Args:
        evaluator: AtomicEvaluator instance used when the agent submits 'finish'.
        max_steps: Maximum number of steps per episode before forced termination.
    """

    def __init__(
        self,
        evaluator: AtomicEvaluator,
        max_steps: int = 10,
    ):
        self.evaluator = evaluator
        self.max_steps = max_steps
        self._state: Optional[EpisodeState] = None
        self._user: Optional[MimicUser] = None
        self._df: Optional[pd.DataFrame] = None
        self._input_col: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, task: TaskInstance, user: MimicUser) -> tuple[dict, dict]:
        """Start a new episode.

        Args:
            task: The TaskInstance to solve.
            user: MimicUser instance (passive or persona-conditioned).

        Returns:
            (observation, info) where observation is a dict and info contains
            task metadata.
        """
        self._state = EpisodeState(task=task)
        self._user = user
        self._df = None
        self._input_col = None

        obs = self._build_observation(last_response="")
        info = {
            "task_id": task.task_id,
            "dataset": task.dataset_name,
            "persona_id": task.persona_id,
            "mode": "persona" if user.persona_config is not None else "passive",
        }
        return obs, info

    def step(self, action: dict) -> tuple[dict, float, bool, dict]:
        """Execute one agent action.

        Args:
            action: Action dict (see aunu_env/env/actions.py).

        Returns:
            (observation, reward, done, info)
            - reward is 0.0 during episode; final f1 score is in info at 'finish'.
            - done is True when the agent calls 'finish' or max_steps is reached.
        """
        assert self._state is not None, "Call reset() before step()"
        validate_action(action)

        s = self._state
        action_type = action["type"]
        response_text = ""
        step_info: dict = {}
        cost = 0.0

        # --- Dispatch ---
        if action_type == ACTION_ASK_USER:
            response_text, cost, step_info = self._handle_ask_user(action)

        elif action_type == ACTION_INSPECT_DATA:
            response_text, cost, step_info = self._handle_inspect_data(action)

        elif action_type == ACTION_PROPOSE_UPDATE:
            response_text, cost, step_info = self._handle_propose_update(action)

        elif action_type == ACTION_FINISH:
            response_text, cost, step_info = self._handle_finish(action)

        s.total_cost += cost
        s.step_count += 1

        # Log step to trajectory
        trajectory_entry = {
            "step": s.step_count,
            "action": action,
            "response": response_text,
            **step_info,
        }
        s.trajectory.append(trajectory_entry)

        # Termination check
        if action_type == ACTION_FINISH or s.step_count >= self.max_steps:
            s.is_done = True
            if action_type != ACTION_FINISH:
                # Forced termination: use last draft as final requirement
                final_req = s.draft_requirement or s.task.elevator_pitch
                logger.warning(
                    f"Episode {s.task.task_id} terminated by step limit. "
                    f"Using last draft as final requirement."
                )
                eval_result = self.evaluator.evaluate(
                    predicted=final_req,
                    gold=s.task.task_requirement,
                    task_id=s.task.task_id,
                )
                s.total_cost += eval_result.get("cost", 0.0)
                step_info["eval_result"] = eval_result
                s.trajectory[-1]["eval_result"] = eval_result

        obs = self._build_observation(response_text)
        info = {"step": s.step_count, "cost": cost, **step_info}
        reward = step_info.get("eval_result", {}).get("scores", {}).get("f1", 0.0)
        return obs, reward, s.is_done, info

    def get_trajectory_log(self) -> dict:
        """Return a structured log of the completed episode."""
        assert self._state is not None, "No episode has been run yet."
        s = self._state
        return {
            "task_id": s.task.task_id,
            "task_name": s.task.task_name,
            "dataset": s.task.dataset_name,
            "persona_id": s.task.persona_id,
            "user_mode": "persona" if self._user and self._user.persona_config else "passive",
            "persona_config": self._user.persona_config if self._user else None,
            "n_steps": s.step_count,
            "total_cost": round(s.total_cost, 6),
            "conversation_history": s.interaction_history,
            "data_inspection_history": s.data_inspections,
            "trajectory": s.trajectory,
            "final_requirement": s.draft_requirement,
            "eval_result": next(
                (e["eval_result"] for e in reversed(s.trajectory) if "eval_result" in e),
                None,
            ),
        }

    @property
    def state(self) -> Optional[EpisodeState]:
        return self._state

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _handle_ask_user(self, action: dict) -> tuple[str, float, dict]:
        s = self._state
        question = action["question"]

        # Build chat history for the MIMIC user
        history = self._build_chat_history()
        asked_at = datetime.now(timezone.utc).isoformat()
        user_result = self._user.respond(
            task=s.task,
            chat_history=history,
            agent_message=question,
        )
        responded_at = datetime.now(timezone.utc).isoformat()
        response = user_result["response"]
        cost = user_result.get("cost", 0.0)

        turn_idx = len(s.interaction_history) // 2 + 1
        s.interaction_history.append({
            "role": "agent",
            "content": question,
            "turn": turn_idx,
            "timestamp": asked_at,
        })
        s.interaction_history.append({
            "role": "user",
            "content": response,
            "turn": turn_idx,
            "timestamp": responded_at,
            "thought": user_result.get("thought", ""),
            "grounding": user_result.get("grounding", ""),
        })

        return response, cost, {
            "user_thought": user_result.get("thought", ""),
            "user_grounding": user_result.get("grounding", ""),
        }

    def _handle_inspect_data(self, action: dict) -> tuple[str, float, dict]:
        s = self._state
        n_samples = action.get("n_samples", 3)

        if self._df is None:
            csv_path = s.task.data_csv_path or os.path.join(
                _DATA_RAW_ROOT, s.task.dataset_name, "sampled_data.csv"
            )
            if not os.path.exists(csv_path):
                raise FileNotFoundError(
                    f"Dataset CSV not found: {csv_path}. "
                    f"Expected at AUNUEnv/data/data_raw/{s.task.dataset_name}/sampled_data.csv"
                )
            self._df = pd.read_csv(csv_path)
            self._input_col = self._detect_input_col(self._df)

        n_samples = min(n_samples, len(self._df))
        sampled = self._df.sample(n=n_samples, random_state=s.step_count)
        inspected_at = datetime.now(timezone.utc).isoformat()

        samples = []
        for idx, row in sampled.iterrows():
            sample = {"_row_idx": int(idx)}
            sample.update({col: str(row[col])[:500] for col in self._df.columns})
            samples.append(sample)

        summary = self._format_data_samples(samples)
        inspection = {
            "inspection_idx": len(s.data_inspections) + 1,
            "timestamp": inspected_at,
            "step": s.step_count + 1,
            "query": action.get("query", ""),
            "n_samples": n_samples,
            "row_indices": [s["_row_idx"] for s in samples],
            "input_col": self._input_col,
            "samples": samples,
        }
        s.data_inspections.append(inspection)

        return summary, 0.0, {"data_samples": samples, "row_indices": inspection["row_indices"]}

    def _handle_propose_update(self, action: dict) -> tuple[str, float, dict]:
        s = self._state
        s.draft_requirement = action["updated_requirement"]
        s.requirement_updates.append({
            "step": s.step_count + 1,
            "requirement": s.draft_requirement,
        })
        response = "Draft requirement updated."
        return response, 0.0, {}

    def _handle_finish(self, action: dict) -> tuple[str, float, dict]:
        s = self._state
        final_req = action["final_requirement"]
        s.draft_requirement = final_req

        eval_result = self.evaluator.evaluate(
            predicted=final_req,
            gold=s.task.task_requirement,
            task_id=s.task.task_id,
        )
        s.total_cost += eval_result.get("cost", 0.0)

        scores = eval_result.get("scores", {})
        response = (
            f"Episode complete. "
            f"F1={scores.get('f1', 0):.3f}, "
            f"Precision={scores.get('precision', 0):.3f}, "
            f"Recall={scores.get('recall', 0):.3f}"
        )
        return response, eval_result.get("cost", 0.0), {"eval_result": eval_result}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_observation(self, last_response: str) -> dict:
        s = self._state
        return {
            "user_request": s.task.elevator_pitch,
            "draft_requirement": s.draft_requirement,
            "last_response": last_response,
            "step_count": s.step_count,
            "is_done": s.is_done,
        }

    def _build_chat_history(self) -> list:
        """Convert interaction_history to simple {role, content} dicts."""
        return [
            {"role": e["role"], "content": e["content"]}
            for e in self._state.interaction_history
        ]

    def _detect_input_col(self, df: pd.DataFrame) -> str:
        candidates = ["text", "article", "report", "script", "news_text", "document"]
        for c in candidates:
            if c in df.columns:
                return c
        for col in df.columns:
            if df[col].dtype == object:
                return col
        return df.columns[0]

    def _format_data_samples(self, samples: list) -> str:
        lines = []
        for i, s in enumerate(samples, 1):
            lines.append(f"--- Sample {i} (row {s.get('_row_idx', i)}) ---")
            for k, v in s.items():
                if k == "_row_idx":
                    continue
                lines.append(f"{k}: {str(v)[:300]}")
        return "\n".join(lines)
