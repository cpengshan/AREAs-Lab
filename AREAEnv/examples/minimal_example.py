#!/usr/bin/env python3
"""
Minimal AREAEnv example — demonstrates the env API without real LLM calls.

Policies are external to AREAEnv. This example shows how to write a simple
policy that interacts with the env through the standard reset/step interface.

Run from the AREAEnv directory:
    python3 examples/minimal_example.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from area_env.dataset.schema import PersonaInfo, TaskInstance
from area_env.env.area_env import AREAEnv
from area_env.env.actions import finish, propose_requirement_update, ask_user
from area_env.users import MimicUser
from area_env.evaluator.atomic_evaluator import AtomicEvaluator


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
    """Ask one question then finish. Demonstrates the env interaction loop."""

    def __init__(self, user):
        self.user = user

    def run(self, env, task):
        obs, info = env.reset(task, self.user)
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
# Build task and run
# ---------------------------------------------------------------------------

persona = PersonaInfo(
    persona_id=1, role="Data Analyst",
    competency_matrix={"proficiencies": ["SQL"], "limitations": []},
    business_motivation="Automate weekly reports.",
    workflow_friction="Manual formatting.",
)
task = TaskInstance(
    task_id="demo_task_0", persona_id=1, persona_info=persona,
    task_name="Weekly Report Summarizer",
    task_requirement="### 1. Strategic Intent\nSummarize weekly sales data concisely...",
    elevator_pitch="I need a tool that summarizes weekly sales reports automatically.",
    deep_dive="Detailed description.", dataset_name="demo", data_csv_path="",
)

env = AREAEnv(evaluator=StubEvaluator(), max_steps=10)
policy = SimpleUserPolicy(user=StubUser())

print("=== AREAEnv Minimal Example ===\n")
log = policy.run(env, task)

print(f"\nTrajectory: {len(log['trajectory'])} steps")
print(f"Eval scores: {log['eval_result']['scores']}")
print(f"Log keys: {list(log.keys())}")
