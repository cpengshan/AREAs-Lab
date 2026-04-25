#!/usr/bin/env python3
"""Entry point for running AUNUEnv-based agents.

Mirrors the interface of scripts/agents/run_aunu_interaction.py but routes
all MIMIC-user interaction through AUNUEnv, keeping the agent's strategy as
a policy over the environment's action space.

Usage examples:
    # Zero-shot — no interaction
    python agent/run_agent.py \\
        --strategy zero_shot \\
        --agent_model gpt-4.1 \\
        --persona 1 2 \\
        --dataset alexfabbri/multi_news

    # User-interaction — clarify via MIMIC user
    python agent/run_agent.py \\
        --strategy user_interaction \\
        --agent_model gpt-4.1 \\
        --mimic_model gpt-4.1 \\
        --persona 1 2 \\
        --dataset alexfabbri/multi_news \\
        --max_turns 5

    # Persona-conditioned MIMIC user
    python agent/run_agent.py \\
        --strategy user_interaction \\
        --agent_model gpt-4.1 \\
        --mimic_model gpt-4.1 \\
        --user_mode persona \\
        --persona 1 \\
        --dataset alexfabbri/multi_news
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime

# Ensure repo root and agent dir are on path
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (_REPO_ROOT, _AGENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from AUNUEnv.aunu_env.config import AUNUEnvConfig
from AUNUEnv.aunu_env.dataset.loader import load_dataset
from AUNUEnv.aunu_env.env.aunu_env import AUNUEnv
from AUNUEnv.aunu_env.evaluator.atomic_evaluator import AtomicEvaluator
from AUNUEnv.aunu_env.evaluator.metrics import aggregate_results
from AUNUEnv.aunu_env.users.mimic_user import MimicUser

from zero_shot_agent import ZeroShotAgent
from user_interaction_agent import UserInteractionAgent
from data_interaction_agent import DataInteractionAgent

RESULTS_DIR = os.path.join(_AGENT_DIR, "results")
DATA_SYNTHESIZED_DIR = os.path.join(_REPO_ROOT, "AUNUEnv/data/data_synthesized")


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run AUNUEnv agents (zero-shot or user-interaction)"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to AUNUEnvConfig YAML file. All model/temperature/max_steps/user_mode "
             "settings are read from the config; CLI flags override individual fields.",
    )
    parser.add_argument(
        "--strategy",
        choices=["zero_shot", "user_interaction", "data_interaction"],
        default="zero_shot",
        help="Agent strategy (default: zero_shot)",
    )
    parser.add_argument("--agent_model", type=str, default=None,
                        help="LLM model for the AUNU agent (overrides config)")
    parser.add_argument("--mimic_model", type=str, default=None,
                        help="LLM model for the MIMIC user (overrides config)")
    parser.add_argument("--evaluator_model", type=str, default=None,
                        help="LLM model for the evaluator (overrides config)")
    parser.add_argument(
        "--persona",
        type=int, nargs="+", required=True,
        help="One or more persona IDs (e.g. --persona 1 2 3)",
    )
    parser.add_argument(
        "--dataset",
        type=str, default=None,
        help="Dataset name under data/data_synthesized/ (overrides config)",
    )
    parser.add_argument(
        "--input_type",
        choices=["elevator_pitch_summary", "deep_dive_summary"],
        default="elevator_pitch_summary",
    )
    parser.add_argument("--max_turns", type=int, default=None,
                        help="Max clarification turns (user_interaction only; overrides config)")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Max env steps per episode (overrides config)")
    parser.add_argument(
        "--user_mode",
        choices=["passive", "persona"],
        default=None,
        help="MIMIC user mode: 'passive' or 'persona' (overrides config)",
    )
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory for results")
    parser.add_argument("--log_file_path", type=str, default=None,
                        help="Path to log file (default: auto-generated in results/)")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _resolve_config(args: argparse.Namespace) -> "AUNUEnvConfig":
    """Load config from YAML (if provided) then apply any CLI overrides."""
    if args.config:
        cfg = AUNUEnvConfig.from_yaml(args.config)
    else:
        cfg = AUNUEnvConfig()

    # CLI flags take precedence over YAML values when explicitly provided.
    if args.agent_model is not None:
        cfg.agent_model = args.agent_model
    if args.mimic_model is not None:
        cfg.user_model = args.mimic_model
    if args.evaluator_model is not None:
        cfg.evaluator_model = args.evaluator_model
    if args.dataset is not None:
        cfg.dataset_name = args.dataset
        # Re-derive paths for the new dataset_name (clear stale explicit paths).
        cfg.dataset_path = ""
        cfg.data_csv_path = ""
        cfg.__post_init__()
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
    if args.user_mode is not None:
        cfg.user_mode = args.user_mode

    # Validate required fields.
    if not cfg.agent_model:
        raise ValueError("agent_model must be set via --agent_model or the config YAML")
    if not cfg.dataset_name:
        raise ValueError("dataset must be set via --dataset or the config YAML")
    return cfg


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _synthesized_path(dataset: str) -> str:
    return os.path.join(DATA_SYNTHESIZED_DIR, dataset, "synthesized_output.json")



# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def make_exp_dir(base_dir: str) -> str:
    exp_id = 1
    while os.path.exists(os.path.join(base_dir, f"Experiment{exp_id}")):
        exp_id += 1
    exp_dir = os.path.join(base_dir, f"Experiment{exp_id}")
    os.makedirs(exp_dir)
    return exp_dir


def _task_number(task_id: str) -> int:
    """Extract 1-based task number from a TaskInstance.task_id string.

    TaskInstance.task_id format: 'user_{persona}_task_{idx}' where idx is
    0-based. The existing results format uses 1-based keys (task_1, task_2).
    """
    return int(task_id.rsplit("_", 1)[-1]) + 1


def _format_eval_result(log: dict) -> dict:
    """Flatten an AtomicEvaluator result into the canonical eval_results.json format.

    Matches the schema produced by scripts/evaluation/user_interaction_judge.py:
      {
        "predicted_units": [...],
        "ground_truth_units": [...],
        "matched_pairs": [...],
        "missing_units": [...],
        "hallucinated_units": [...],
        "misaligned_units": [...],
        "critical_units": [...],
        "critical_missing": [...],
        "counts": {...},
        "scores": {...}
      }
    """
    eval_result = log.get("eval_result") or {}
    if not eval_result:
        return {"error": "no eval result"}

    comparison = eval_result.get("comparison", {})
    return {
        "predicted_units": eval_result.get("predicted_units", []),
        "ground_truth_units": eval_result.get("gold_units", []),
        "matched_pairs": comparison.get("matched_pairs", []),
        "missing_units": comparison.get("missing_units", []),
        "hallucinated_units": comparison.get("hallucinated_units", []),
        "misaligned_units": comparison.get("misaligned_units", []),
        "critical_units": comparison.get("critical_units", []),
        "critical_missing": comparison.get("critical_missing", []),
        "counts": eval_result.get("counts", {}),
        "scores": eval_result.get("scores", {}),
    }


def _format_task_result(log: dict, task, args: argparse.Namespace, cfg: "AUNUEnvConfig", task_num: int) -> dict:
    """Convert an agent trajectory log into the canonical output.json task format.

    Matches the schema used by scripts/agents/run_aunu_interaction.py:
      {
        "messages": [...],
        "is_complete": true,
        "task_requirement_final": "...",
        "strategy": "...",
        "persona": <int>,
        "task_id": <int>,
        "dataset": "...",
        "model": "...",
        "input_type": "...",
        "ground_truth": "...",
        "cost": <float>
      }
    """
    result = {
        "messages": log.get("agent_messages", []),
        "is_complete": not log.get("error"),
        "task_requirement_final": log.get("final_requirement", ""),
        "strategy": args.strategy,
        "persona": task.persona_id,
        "task_id": task_num,
        "dataset": cfg.dataset_name,
        "model": cfg.agent_model,
        "input_type": args.input_type,
        "ground_truth": task.task_requirement,
        "cost": round(log.get("total_cost", 0.0), 6),
    }
    if args.strategy == "user_interaction" and log.get("conversation_history"):
        result["conversation_history"] = [
            {"role": e["role"], "turn": e["turn"], "timestamp": e.get("timestamp", ""),
             "content": e["content"], "thought": e.get("thought", "")}
            for e in log["conversation_history"]
        ]
    if args.strategy == "data_interaction" and log.get("data_inspection_history"):
        result["data_inspection_history"] = [
            {"inspection_idx": e["inspection_idx"], "timestamp": e.get("timestamp", ""),
             "step": e["step"], "query": e.get("query", ""),
             "n_samples": e["n_samples"], "row_indices": e["row_indices"],
             "input_col": e.get("input_col", ""), "samples": e["samples"]}
            for e in log["data_inspection_history"]
        ]
    return result


def run_experiment(args: argparse.Namespace, cfg: "AUNUEnvConfig", exp_dir: str) -> dict:
    """Run the benchmark for all requested personas and tasks.

    Saves incremental results to output.json after each task so progress is
    not lost on failure.

    Returns:
        The full results dict (same as written to output.json).
    """
    max_turns = args.max_turns if args.max_turns is not None else 5

    evaluator = AtomicEvaluator(
        model_name=cfg.evaluator_model,
        temperature=cfg.effective_evaluator_temperature,
        cache_gold_units=True,
    )
    env = AUNUEnv(evaluator=evaluator, max_steps=cfg.max_steps)

    persona_config = {"user_mode": "persona"} if cfg.user_mode == "persona" else None

    all_tasks = load_dataset(
        synthesized_path=cfg.dataset_path,
        dataset_name=cfg.dataset_name,
    )
    tasks = [t for t in all_tasks if t.persona_id in args.persona]
    if not tasks:
        raise ValueError(f"No tasks found for personas {args.persona} in {cfg.dataset_path}")
    logger.info(f"Running {len(tasks)} tasks for personas {args.persona}")

    # Top-level output structure matches existing results/*/output.json
    output = {
        "args": {
            "strategy_aunu": args.strategy,
            "aunu_model": cfg.agent_model,
            "mimic_model": cfg.user_model,
            "evaluator_model": cfg.evaluator_model,
            "persona": args.persona,
            "dataset": cfg.dataset_name,
            "input_type": args.input_type,
            "max_turns": max_turns,
            "max_steps": cfg.max_steps,
            "user_mode": cfg.user_mode,
            "total_cost": 0.0,
        }
    }

    out_path = os.path.join(exp_dir, "output.json")
    eval_path = os.path.join(exp_dir, "eval_results.json")

    # Load existing eval_results if resuming
    eval_output: dict = {}
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            eval_output = json.load(f)

    for task in tasks:
        persona_key = str(task.persona_id)
        task_num = _task_number(task.task_id)
        task_key = f"task_{task_num}"

        # Skip already-completed tasks (resume support)
        if persona_key in output and task_key in output[persona_key]:
            logger.info(f"[{task.task_id}] Already done, skipping.")
            continue

        logger.info(f"[{task.task_id}] Starting (persona={task.persona_id}, task={task_num})...")

        user = MimicUser(
            model_name=cfg.user_model,
            persona_config=persona_config,
            temperature=cfg.effective_user_temperature,
        )
        log = None

        try:
            if args.strategy == "zero_shot":
                agent = ZeroShotAgent.from_config(cfg)
                log = agent.run(env, task)
            elif args.strategy == "data_interaction":
                agent = DataInteractionAgent.from_config(cfg, max_turns=max_turns)
                log = agent.run(env, task)
            else:
                agent = UserInteractionAgent.from_config(cfg, max_turns=max_turns)
                log = agent.run(env, task, user)

            task_result = _format_task_result(log, task, args, cfg, task_num)
            f1 = (log.get("eval_result") or {}).get("scores", {}).get("f1", "N/A")
            logger.info(f"[{task.task_id}] F1={f1}")

        except Exception as e:
            logger.error(f"[{task.task_id}] FAILED: {e}", exc_info=True)
            task_result = {
                "messages": [],
                "is_complete": False,
                "task_requirement_final": "",
                "strategy": args.strategy,
                "persona": task.persona_id,
                "task_id": task_num,
                "dataset": cfg.dataset_name,
                "model": cfg.agent_model,
                "input_type": args.input_type,
                "ground_truth": task.task_requirement,
                "cost": 0.0,
                "error": str(e),
            }

        # Update output.json
        if persona_key not in output:
            output[persona_key] = {}
        output[persona_key][task_key] = task_result

        output["args"]["total_cost"] = round(
            sum(
                output[pk][tk].get("cost", 0.0)
                for pk in output if pk != "args"
                for tk in output[pk]
            ),
            6,
        )

        with open(out_path, "w") as f:
            json.dump(output, f, indent=2, default=str)

        # Update eval_results.json
        if log is not None:
            if persona_key not in eval_output:
                eval_output[persona_key] = {}
            eval_output[persona_key][task_key] = _format_eval_result(log)

            with open(eval_path, "w") as f:
                json.dump(eval_output, f, indent=2, default=str)
            logger.info(f"Saved → {out_path}, {eval_path}")
        else:
            logger.info(f"Saved → {out_path}")

    return output


def main():
    args = parse_args()
    cfg = _resolve_config(args)

    base_dir = os.path.join(
        RESULTS_DIR, cfg.dataset_name.replace("/", "_"), args.strategy
    )
    os.makedirs(base_dir, exist_ok=True)
    exp_dir = make_exp_dir(base_dir)

    log_path = args.log_file_path or os.path.join(exp_dir, "run.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
    )

    output = run_experiment(args, cfg, exp_dir)

    out_path = os.path.join(exp_dir, "output.json")
    eval_path = os.path.join(exp_dir, "eval_results.json")
    completed = sum(len(v) for k, v in output.items() if k != "args")

    # Aggregate scores from eval_results.json for the summary
    eval_output = {}
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            eval_output = json.load(f)
    all_scores = [
        eval_output[pk][tk]["scores"]
        for pk in eval_output
        for tk in eval_output[pk]
        if "scores" in eval_output[pk][tk]
    ]

    print(f"\n=== Results ===")
    print(f"Tasks completed: {completed}")
    print(f"Total cost:      ${output['args']['total_cost']:.6f}")
    if all_scores:
        for metric in ("f1", "precision", "recall", "completeness", "faithfulness"):
            mean = round(sum(s.get(metric, 0) for s in all_scores) / len(all_scores), 4)
            print(f"  {metric}: {mean}")
    print(f"Output:          {out_path}")
    print(f"Eval results:    {eval_path}")


if __name__ == "__main__":
    main()
