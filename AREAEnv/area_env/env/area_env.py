"""
AREAEnv — core gymnasium-style environment for AREA benchmark evaluation.

Interface mirrors gym.Env without requiring the gymnasium package:
  env.reset(dataset_name, persona_id, task_id)  →  (observation, info)
  env.step(action)                               →  (observation, reward, done, info)

The environment owns all dataset access. Agents interact using identifiers only:
  dataset_name  — e.g. "alexfabbri/multi_news"
  persona_id    — int, numeric string, "user_N", or "persona_N"
  task_id       — e.g. "user_1_task_0"

Action space (dict-based):
  ask_user(question)
  inspect_data(n_samples, query)
  propose_requirement_update(updated_requirement)
  finish(final_requirement)
"""

import json
import logging
import os
import random
from datetime import datetime, timezone
from typing import Optional

# Default root for sampled data: AREAEnv/data/data_sampled/
_DATA_SAMPLED_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../data/data_sampled")
)

from .actions import VALID_SPLITS
from ..dataset.loader import DatasetRegistry
from ..users import MimicUser
from ..evaluator.atomic_evaluator import AtomicEvaluator
from .actions import (
    ACTION_ASK_USER, ACTION_INSPECT_DATA,
    ACTION_PROPOSE_UPDATE, ACTION_FINISH,
    validate_action,
)
from .state import EpisodeState

logger = logging.getLogger(__name__)


class AREAEnv:
    """Interactive benchmark environment for AI-Assisted User Needs Understanding.

    The environment manages all dataset access. Agents identify episodes by
    (dataset_name, persona_id, task_id) and never receive raw TaskInstance objects.

    Args:
        evaluator: AtomicEvaluator used when the agent submits 'finish'.
        user: MimicUser that responds to ask_user actions.
        registry: DatasetRegistry that resolves task identifiers to TaskInstances.
        max_steps: Maximum steps per episode before forced termination.
        data_sampled_root: Root directory for data_sampled files. Defaults to
            AREAEnv/data/data_sampled/.
        data_sampled_file: Filename within each dataset's data_sampled directory.
    """

    def __init__(
        self,
        evaluator: AtomicEvaluator,
        user: MimicUser,
        registry: DatasetRegistry,
        max_steps: int = 10,
        data_sampled_root: Optional[str] = None,
        data_sampled_file: str = "data_sampled.json",
    ):
        self.evaluator = evaluator
        self.user = user
        self.registry = registry
        self.max_steps = max_steps
        self._data_sampled_root = data_sampled_root or _DATA_SAMPLED_ROOT
        self.data_sampled_file = data_sampled_file
        self._state: Optional[EpisodeState] = None
        self._data: Optional[dict] = None  # {defining_instances: [...], non_defining_instances: [...]}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, dataset_name: str, persona_id, task_id: str) -> tuple[dict, dict]:
        """Start a new episode.

        The environment resolves the task internally; agents do not receive
        the TaskInstance (which contains the ground-truth requirement).

        Args:
            dataset_name: Dataset identifier, e.g. "alexfabbri/multi_news".
            persona_id: Persona identifier — int, numeric str, "user_N", or "persona_N".
            task_id: Task identifier string, e.g. "user_1_task_0".

        Returns:
            (observation, info) where observation exposes only the underspecified
            user request, and info contains episode metadata.
        """
        task = self.registry.get_task(dataset_name, persona_id, task_id)
        self._state = EpisodeState(task=task)
        self._data = None

        obs = self._build_observation(last_response="")
        info = {
            "task_id": task.task_id,
            "dataset": task.dataset_name,
            "persona_id": task.persona_id,
            "mode": "persona" if self.user.persona_config is not None else "passive",
        }
        return obs, info

    def step(self, action: dict) -> tuple[dict, float, bool, dict]:
        """Execute one agent action.

        Args:
            action: Action dict (see area_env/env/actions.py).

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
            "user_mode": "persona" if self.user.persona_config else "passive",
            "persona_config": self.user.persona_config,
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
        user_result = self.user.respond(
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
        split = action.get("split", "all")

        if self._data is None:
            json_path = os.path.join(
                self._data_sampled_root, s.task.dataset_name, self.data_sampled_file
            )
            if not os.path.exists(json_path):
                raise FileNotFoundError(
                    f"Sampled data not found: {json_path}. "
                    f"Expected at AREAEnv/data/data_sampled/{s.task.dataset_name}/{self.data_sampled_file}"
                )
            with open(json_path, "r", encoding="utf-8") as f:
                self._data = json.load(f)

        if split == "all":
            pool = (
                self._data.get("defining_instances", [])
                + self._data.get("non_defining_instances", [])
            )
        else:
            pool = self._data.get(split, [])
            if not pool:
                # Requested split is empty; fall back to all available instances.
                logger.warning(
                    f"Split '{split}' is empty for dataset '{s.task.dataset_name}'; falling back to 'all'."
                )
                pool = (
                    self._data.get("defining_instances", [])
                    + self._data.get("non_defining_instances", [])
                )

        if not pool:
            raise ValueError(f"No instances found for split='{split}' in dataset '{s.task.dataset_name}'")

        rng = random.Random(s.step_count)
        n_samples = min(n_samples, len(pool))
        sampled = rng.sample(pool, n_samples)
        inspected_at = datetime.now(timezone.utc).isoformat()

        samples = []
        for i, item in enumerate(sampled):
            sample = {"_sample_idx": i}
            sample.update({k: str(v)[:500] for k, v in item.items()})
            samples.append(sample)

        summary = self._format_data_samples(samples, split)
        inspection = {
            "inspection_idx": len(s.data_inspections) + 1,
            "timestamp": inspected_at,
            "step": s.step_count + 1,
            "query": action.get("query", ""),
            "split": split,
            "n_samples": n_samples,
            "samples": samples,
        }
        s.data_inspections.append(inspection)

        return summary, 0.0, {"data_samples": samples, "split": split}

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

    def _format_data_samples(self, samples: list, split: str = "all") -> str:
        lines = [f"[split={split}]"]
        for i, s in enumerate(samples, 1):
            lines.append(f"--- Sample {i} ---")
            for k, v in s.items():
                if k == "_sample_idx":
                    continue
                lines.append(f"{k}: {str(v)[:300]}")
        return "\n".join(lines)
