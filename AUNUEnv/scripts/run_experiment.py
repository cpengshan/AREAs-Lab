#!/usr/bin/env python3
"""
AUNUEnv experiment runner.

Orchestrates policy runs over one or more tasks, saves structured JSON results,
and prints aggregate scores. Policies and the runner itself are external to
aunu_env — the env package only provides the environment, dataset, user, and
evaluator primitives.

Usage:
    # Run with a built-in example policy
    python scripts/run_experiment.py \
        --config aunu_env/configs/zero_shot.yaml \
        --policy zero_shot

    python scripts/run_experiment.py \
        --config aunu_env/configs/user_only.yaml \
        --policy user_only \
        --max_turns 5

    python scripts/run_experiment.py \
        --config aunu_env/configs/data_only.yaml \
        --policy data_only \
        --max_data_iterations 3 \
        --n_data_samples 3

    # Run a single task
    python scripts/run_experiment.py \
        --config aunu_env/configs/zero_shot.yaml \
        --policy zero_shot \
        --task_id user_1_task_0

    # Use a custom external policy (module.factory_fn(user) -> policy)
    python scripts/run_experiment.py \
        --config aunu_env/configs/user_only.yaml \
        --policy_module my_pkg.policies.my_factory
"""

import argparse
import importlib
import json
import logging
import os
import sys
from datetime import datetime
from typing import Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from aunu_env.config import AUNUEnvConfig
from aunu_env.dataset.loader import load_dataset
from aunu_env.evaluator.atomic_evaluator import AtomicEvaluator
from aunu_env.evaluator.metrics import aggregate_results
from aunu_env.env.aunu_env import AUNUEnv
from aunu_env.users.mimic_user import MimicUser

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_experiment(config: AUNUEnvConfig, policy_factory: Callable) -> dict:
    """Run a benchmark experiment using the supplied policy factory.

    Args:
        config: AUNUEnvConfig specifying data, models, and output settings.
        policy_factory: Callable(user) -> policy with run(env, task) -> dict.

    Returns:
        Dict with keys: config, n_tasks, n_successful, aggregate_scores,
        task_results, output_path.
    """
    tasks = load_dataset(
        synthesized_path=config.dataset_path,
        data_csv_path=config.data_csv_path,
        dataset_name=config.dataset_name,
        task_ids=config.task_ids,
    )
    logger.info(f"Loaded {len(tasks)} tasks")

    user = MimicUser(
        model_name=config.user_model,
        persona_config=config.persona_config if config.user_mode == "persona" else None,
        temperature=config.effective_user_temperature,
    )
    evaluator = AtomicEvaluator(
        model_name=config.evaluator_model,
        temperature=config.effective_evaluator_temperature,
        cache_gold_units=True,
    )
    env = AUNUEnv(evaluator=evaluator, max_steps=config.max_steps)

    task_results = []
    for i, task in enumerate(tasks):
        logger.info(f"[{i+1}/{len(tasks)}] {task.task_id}")
        policy = policy_factory(user)
        try:
            log = policy.run(env, task)
            task_results.append(log)
            f1 = log.get("eval_result", {}).get("scores", {}).get("f1", "N/A")
            logger.info(f"  → F1={f1}")
        except Exception as e:
            logger.error(f"  Task {task.task_id} failed: {e}", exc_info=True)
            task_results.append({"task_id": task.task_id, "error": str(e)})

    valid = [r for r in task_results if "eval_result" in r and r["eval_result"]]
    agg = aggregate_results([r["eval_result"] for r in valid])

    result = {
        "config": config.to_dict(),
        "timestamp": datetime.utcnow().isoformat(),
        "n_tasks": len(tasks),
        "n_successful": len(valid),
        "aggregate_scores": agg,
        "task_results": task_results,
    }

    os.makedirs(config.output_dir, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(config.output_dir, f"results_{ts}.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info(f"Saved → {output_path}")

    result["output_path"] = output_path
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Run an AUNUEnv experiment")
    parser.add_argument("--config", required=True, help="Path to YAML config file")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--policy", choices=["zero_shot", "user_only", "data_only"],
                       help="Built-in example policy (from examples/policies.py)")
    group.add_argument("--policy_module", metavar="MODULE.FACTORY",
                       help="Custom policy factory: dotted path to a function(user)->policy")

    parser.add_argument("--task_id", help="Run a single task by ID")
    parser.add_argument("--output_dir", help="Override output directory from config")

    # Policy-specific overrides
    parser.add_argument("--max_turns", type=int, default=5,
                        help="Max user clarification turns (user_only)")
    parser.add_argument("--max_data_iterations", type=int, default=3,
                        help="Max data inspection cycles (data_only)")
    parser.add_argument("--n_data_samples", type=int, default=3,
                        help="Rows sampled per inspect_data action (data_only)")

    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def load_builtin_policy(name: str, args) -> Callable:
    """Return a factory function for one of the example policies."""
    from examples.policies import ZeroShotPolicy, UserOnlyPolicy, DataOnlyPolicy

    if name == "zero_shot":
        def factory(user):
            return ZeroShotPolicy(user, model_name=_cfg.agent_model,
                                  temperature=_cfg.temperature, max_tokens=_cfg.max_tokens)
    elif name == "user_only":
        def factory(user):
            return UserOnlyPolicy(user, model_name=_cfg.agent_model, max_turns=args.max_turns,
                                  temperature=_cfg.temperature, max_tokens=_cfg.max_tokens)
    elif name == "data_only":
        def factory(user):
            return DataOnlyPolicy(user, model_name=_cfg.agent_model,
                                  max_iterations=args.max_data_iterations,
                                  n_samples=args.n_data_samples,
                                  temperature=_cfg.temperature, max_tokens=_cfg.max_tokens)
    return factory


def load_custom_policy(dotted_path: str) -> Callable:
    module_path, _, fn_name = dotted_path.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, fn_name)


_cfg = None  # module-level ref so nested factory closures can read it


def main():
    global _cfg
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    _cfg = AUNUEnvConfig.from_yaml(args.config)
    if args.task_id:
        _cfg.task_ids = [args.task_id]
    if args.output_dir:
        _cfg.output_dir = args.output_dir
    if args.verbose:
        _cfg.verbose = True

    if args.policy:
        policy_factory = load_builtin_policy(args.policy, args)
    else:
        policy_factory = load_custom_policy(args.policy_module)

    result = run_experiment(_cfg, policy_factory)

    print("\n=== Results ===")
    print(f"Tasks:      {result['n_tasks']}  (successful: {result['n_successful']})")
    print("Scores:")
    for k, v in result.get("aggregate_scores", {}).items():
        print(f"  {k}: {v}")
    print(f"Output:     {result.get('output_path')}")


if __name__ == "__main__":
    main()
