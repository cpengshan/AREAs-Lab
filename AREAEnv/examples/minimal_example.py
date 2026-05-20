#!/usr/bin/env python3
"""
Minimal AREAEnv example — demonstrates the gym-style env API without real LLM calls.

The agent interacts with the environment using only three identifiers:
  dataset_name, persona_id, task_id.
All dataset access is managed inside AREAEnv.

Run from the AREAEnv directory:
    python3 examples/minimal_example.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from area_env.env.area_env import AREAEnv
from area_env.env.actions import finish, propose_requirement_update, ask_user
from area_env.users import MimicUser
from area_env.evaluator.atomic_evaluator import AtomicEvaluator
from area_env.dataset.loader import DatasetRegistry


# ---------------------------------------------------------------------------
# Stubs — no real LLM calls
# ---------------------------------------------------------------------------

class StubEvaluator(AtomicEvaluator):
    def __init__(self):
        self.model_name = "stub"
        self.temperature = 0.0
        self.max_tokens = 0
        self._gold_cache = {}
        self._cache_gold = False

    def evaluate(self, predicted, gold, task_id=None):
        return {
            "gold_units": ["unit A", "unit B"],
            "predicted_units": ["unit A", "unit C"],
            "comparison": {
                "matched_pairs": [{"ground_truth": "unit A", "predicted": ["unit A"]}],
                "missing_units": ["unit B"], "hallucinated_units": ["unit C"],
                "misaligned_units": [], "ground_truth_units": ["unit A", "unit B"],
                "predicted_units": ["unit A", "unit C"], "critical_units": [], "critical_missing": [],
            },
            "counts": {"tp": 1, "fp": 1, "fn": 1, "n_predicted_units": 2, "n_ground_truth_units": 2},
            "scores": {"precision": 0.5, "recall": 0.5, "f1": 0.5,
                       "alignment": 1.0, "constraint_preservation": 0.0},
            "cost": 0.0,
        }


class StubUser(MimicUser):
    def __init__(self):
        self.model_name = "stub"
        self.persona_config = None
        self.temperature = 0.0
        self.max_tokens = 0
        self._mode = "passive"

    def respond(self, task, chat_history, agent_message):
        return {"thought": "Seems reasonable.", "response": "Yes, that sounds right.", "cost": 0.0, "raw": {}}


# ---------------------------------------------------------------------------
# A simple external policy — interacts with the env via reset/step
# ---------------------------------------------------------------------------

class SimpleUserPolicy:
    """Ask one question then finish. Demonstrates the gym-style interaction loop."""

    def run(self, env, dataset_name: str, persona_id, task_id: str) -> dict:
        # Agent identifies the episode by three IDs only — no raw TaskInstance.
        obs, info = env.reset(dataset_name, persona_id, task_id)
        print(f"  Task: {info['task_id']}")
        print(f"  User request: {obs['user_request']}")

        # Step 1: propose an initial draft
        obs, _, done, _ = env.step(propose_requirement_update(
            "### 1. Strategic Intent\nSummarize weekly sales data..."
        ))
        print(f"  Step 1 — propose_requirement_update: draft set")

        # Step 2: ask user a clarifying question
        obs, _, done, _ = env.step(ask_user("Should the summary include revenue trends?"))
        print(f"  Step 2 — ask_user → '{obs['last_response']}'")

        # Step 3: finish with refined requirement
        obs, reward, done, _ = env.step(finish(
            "### 1. Strategic Intent\nSummarize weekly sales data with revenue trends..."
        ))
        print(f"  Step 3 — finish: reward={reward:.3f}")

        return env.get_trajectory_log()


# ---------------------------------------------------------------------------
# Build env and run
# ---------------------------------------------------------------------------

# DatasetRegistry owns all synthesized data. Point it at your data directory.
_DATA_SYNTHESIZED_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../data/data_synthesized")
)
registry = DatasetRegistry(data_synthesized_root=_DATA_SYNTHESIZED_ROOT)

env = AREAEnv(
    evaluator=StubEvaluator(),
    user=StubUser(),
    registry=registry,
    max_steps=10,
)
policy = SimpleUserPolicy()

print("=== AREAEnv Minimal Example ===\n")

# Agent provides only three identifiers — the env resolves everything internally.
# Adjust dataset_name / persona_id / task_id to match your local data.
DATASET = "alexfabbri/multi_news"
PERSONA_ID = 1          # also accepts "user_1" or "persona_1"
TASK_ID = "user_1_task_0"

try:
    log = policy.run(env, DATASET, PERSONA_ID, TASK_ID)
    print(f"\nTrajectory: {len(log['trajectory'])} steps")
    print(f"Eval scores: {log['eval_result']['scores']}")
    print(f"Log keys: {list(log.keys())}")
except KeyError as e:
    print(f"\nTask not found ({e}). Adjust DATASET/PERSONA_ID/TASK_ID above.")
