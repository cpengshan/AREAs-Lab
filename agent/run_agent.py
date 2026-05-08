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


class _NoOpEvaluator:
    """Drop-in replacement for AtomicEvaluator that skips all LLM calls."""

    def evaluate(self, predicted: str, gold: str, task_id=None) -> dict:
        return {"gold_units": [], "predicted_units": [], "comparison": {},
                "counts": {}, "scores": {}, "gold_categories": {},
                "pred_categories": {}, "subcategory_scores": {}, "cost": 0.0}
from AUNUEnv.aunu_env.users.mimic_user import MimicUser
from AUNUEnv.aunu_env.users.mimic_user_v2 import MimicUserV2

from zero_shot_agent import ZeroShotAgent
from user_interaction_agent import UserInteractionAgent
from data_interaction_agent import DataInteractionAgent
from data_interaction_v2_agent import DataInteractionV2Agent
from data_interaction_v3_agent import DataInteractionV3Agent
from data_interaction_v4_agent import DataInteractionV4Agent
from data_interaction_v5_agent import DataInteractionV5Agent
from hybrid_agent import HybridAgent
from hybrid_agent_v2 import HybridV2Agent
from zero_shot_variants_agent import (
    ZeroShotWithDataAnalysisAgent,
    ZeroShotWithSamplesAgent,
    ZeroShotWithSamplesReasonAgent,
    ZeroShotWithDataAnalysisAndSamplesAgent,
    ZeroShotWithDataSummaryAgent,
)

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
        choices=[
            "zero_shot", "zero_shot_with_data_analysis", "zero_shot_with_samples",
            "zero_shot_with_samples_reason",
            "zero_shot_with_data_analysis_and_samples", "zero_shot_with_data_summary",
            "user_interaction", "data_interaction", "data_interaction_v2", "data_interaction_v3", "data_interaction_v4", "data_interaction_v5", "hybrid", "hybrid_v2",
        ],
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
    parser.add_argument("--max_iterations", type=int, default=None,
                        help="Max routing iterations for hybrid strategy (overrides config)")
    parser.add_argument("--mid_turn", type=int, default=6,
                        help="Turn at which hybrid_v2 fires mid-synthesis and enters Phase 2 (default: 6)")
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
    parser.add_argument("--exp_id", type=int, default=None,
                        help="Resume an existing ExperimentN instead of creating a new one")
    parser.add_argument("--log_file_path", type=str, default=None,
                        help="Path to log file (default: auto-generated in results/)")
    parser.add_argument(
        "--split",
        choices=["defining_instances", "non_defining_instances", "all"],
        default="all",
        help="Which subset of data_sampled_2.1.json to sample from (default: all)",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--no_eval", action="store_true",
        help="Skip evaluation at the end of each episode; only save output.json",
    )
    parser.add_argument(
        "--use_v2", action="store_true",
        help="Use MimicUserV2 (feedback_mimic_user_v3.jinja + responser_habit.json)",
    )
    parser.add_argument(
        "--communication_habit", type=str, default=None,
        choices=["passive", "neutral", "active"],
        help="Communication habit for MimicUserV2 (passive/neutral/active). Required with --use_v2.",
    )
    parser.add_argument(
        "--seed_requirement_dir", type=str, default=None,
        help="Path to a prior experiment directory whose output.json task_requirement_final values "
             "are used as the initial current_task_requirement for hybrid runs (e.g. zero_shot/Experiment10).",
    )
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

def make_exp_dir(base_dir: str, exp_id: int | None = None) -> str:
    if exp_id is not None:
        exp_dir = os.path.join(base_dir, f"Experiment{exp_id}")
        if not os.path.exists(exp_dir):
            raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")
        return exp_dir
    next_id = 1
    while os.path.exists(os.path.join(base_dir, f"Experiment{next_id}")):
        next_id += 1
    exp_dir = os.path.join(base_dir, f"Experiment{next_id}")
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
    result = {
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
        "gold_categories": eval_result.get("gold_categories", {}),
        "pred_categories": eval_result.get("pred_categories", {}),
        "subcategory_scores": eval_result.get("subcategory_scores", {}),
    }
    if log.get("intermediate_evals"):
        result["intermediate_evals"] = log["intermediate_evals"]
    return result


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
    if args.strategy in ("data_interaction_v2", "data_interaction_v3", "data_interaction_v4", "data_interaction_v5") and log.get("format_reflection_history"):
        result["format_reflection_history"] = log["format_reflection_history"]
    if args.strategy in ("data_interaction_v2", "data_interaction_v3", "data_interaction_v4", "data_interaction_v5") and log.get("intermediate_evals"):
        result["intermediate_evals"] = log["intermediate_evals"]
    if args.strategy in ("data_interaction_v3", "data_interaction_v4", "data_interaction_v5") and log.get("rewrite_history"):
        result["rewrite_history"] = log["rewrite_history"]
    if args.strategy in ("data_interaction", "data_interaction_v2", "data_interaction_v3", "data_interaction_v4", "data_interaction_v5") and log.get("data_inspection_history"):
        result["data_inspection_history"] = [
            {"inspection_idx": e["inspection_idx"], "timestamp": e.get("timestamp", ""),
             "step": e["step"], "query": e.get("query", ""),
             "n_samples": e["n_samples"], "row_indices": e.get("row_indices", []),
             "input_col": e.get("input_col", ""), "samples": e["samples"]}
            for e in log["data_inspection_history"]
        ]
    if args.strategy == "hybrid":
        result["router_history"] = log.get("router_history", [])
        result["user_interactions"] = log.get("user_interactions", [])
        result["data_interactions"] = log.get("data_interactions", [])
        result["format_reflection_history"] = log.get("format_reflection_history", [])
        result["elevator_pitch"] = getattr(task, "elevator_pitch", "")
        result["zero_shot_draft"] = log.get("zero_shot_draft", "")
    if args.strategy == "hybrid_v2":
        result["router_history"] = log.get("router_history", [])
        result["user_interactions"] = log.get("user_interactions", [])
        result["user_interactions_phase2"] = log.get("user_interactions_phase2", [])
        result["data_interactions"] = log.get("data_interactions", [])
        result["mid_synthesis"] = log.get("mid_synthesis", "")
        result["format_reflection_history"] = log.get("format_reflection_history", [])
        result["elevator_pitch"] = getattr(task, "elevator_pitch", "")
    if args.strategy == "zero_shot_with_samples_reason" and log.get("modifications") is not None:
        result["modifications"] = log["modifications"]
    return result


def _find_best_zero_shot_dir(dataset_name: str, aunu_model: str) -> str | None:
    """Return the zero_shot ExperimentN dir with the largest N whose output.json matches aunu_model."""
    zero_shot_base = os.path.join(RESULTS_DIR, dataset_name.replace("/", "_"), "zero_shot")
    if not os.path.isdir(zero_shot_base):
        return None
    best_id, best_dir = -1, None
    for entry in os.scandir(zero_shot_base):
        if not entry.is_dir() or not entry.name.startswith("Experiment"):
            continue
        try:
            exp_id = int(entry.name[len("Experiment"):])
        except ValueError:
            continue
        output_path = os.path.join(entry.path, "output.json")
        if not os.path.exists(output_path):
            continue
        try:
            with open(output_path) as f:
                data = json.load(f)
            if data.get("args", {}).get("aunu_model") != aunu_model:
                continue
        except Exception:
            continue
        if exp_id > best_id:
            best_id, best_dir = exp_id, entry.path
    return best_dir


def _load_seed_requirements(seed_dir: str) -> dict:
    """Load task_requirement_final values keyed by (persona_id, task_num) from a prior output.json."""
    path = os.path.join(seed_dir, "output.json")
    with open(path) as f:
        data = json.load(f)
    seeds: dict = {}
    for pk, pv in data.items():
        if pk == "args" or not isinstance(pv, dict):
            continue
        for tk, tv in pv.items():
            if not isinstance(tv, dict):
                continue
            # tv is {task_1: result, task_2: result, ...} or a result itself
            first = next(iter(tv.values()), None)
            if isinstance(first, dict) and "task_requirement_final" in first:
                # habit-keyed layer
                for task_result in tv.values():
                    if isinstance(task_result, dict) and "task_requirement_final" in task_result:
                        p = task_result.get("persona", int(pk) if pk.isdigit() else None)
                        t = task_result.get("task_id")
                        if p is not None and t is not None:
                            seeds[(p, t)] = task_result["task_requirement_final"]
            elif isinstance(tv, dict) and "task_requirement_final" in tv:
                p = tv.get("persona", int(pk) if pk.isdigit() else None)
                t = tv.get("task_id")
                if p is not None and t is not None:
                    seeds[(p, t)] = tv["task_requirement_final"]
    return seeds


def _sum_costs(output: dict) -> float:
    """Sum cost across all persona/habit/task entries in output."""
    total = 0.0
    for pk, pv in output.items():
        if pk == "args" or not isinstance(pv, dict):
            continue
        for hk, hv in pv.items():
            if isinstance(hv, dict):
                # hv is either {task_key: result} or a task result itself
                first = next(iter(hv.values()), None)
                if isinstance(first, dict) and "cost" in first:
                    # habit-keyed: {habit_1: {task_1: result}}
                    for tv in hv.values():
                        if isinstance(tv, dict):
                            total += tv.get("cost", 0.0)
                else:
                    # flat: {task_1: result}
                    total += hv.get("cost", 0.0)
    return round(total, 6)


def run_experiment(args: argparse.Namespace, cfg: "AUNUEnvConfig", exp_dir: str) -> dict:
    """Run the benchmark for all requested personas and tasks.

    Saves incremental results to output.json after each task so progress is
    not lost on failure.

    Returns:
        The full results dict (same as written to output.json).
    """
    max_turns = args.max_turns if args.max_turns is not None else 5
    max_iterations = args.max_iterations if args.max_iterations is not None else 6

    if getattr(args, "no_eval", False):
        evaluator = _NoOpEvaluator()
    else:
        _gt_cache_path = None
        if cfg.dataset_path:
            _gt_cache_path = os.path.join(os.path.dirname(cfg.dataset_path), "ground_truth_decompose.json")
        evaluator = AtomicEvaluator(
            model_name=cfg.evaluator_model,
            temperature=cfg.effective_evaluator_temperature,
            cache_gold_units=True,
            cache_path=_gt_cache_path,
        )
    env = AUNUEnv(evaluator=evaluator, max_steps=cfg.max_steps, data_sampled_file=getattr(cfg, "data_sampled_file", "data_sampled_2.6.json"))

    persona_config = {"user_mode": "persona"} if cfg.user_mode == "persona" else None

    all_tasks = load_dataset(
        synthesized_path=cfg.dataset_path,
        dataset_name=cfg.dataset_name,
    )
    tasks = sorted(
        [t for t in all_tasks if t.persona_id in args.persona],
        key=lambda t: (t.persona_id, int(t.task_id.rsplit("_", 1)[-1])),
    )
    if not tasks:
        raise ValueError(f"No tasks found for personas {args.persona} in {cfg.dataset_path}")
    logger.info(f"Running {len(tasks)} tasks for personas {args.persona}")

    # Load communication habit dict if using MimicUserV2
    if args.use_v2:
        if not args.communication_habit:
            raise ValueError("--communication_habit (passive/neutral/active) is required with --use_v2")
        _habit_file = os.path.join(
            _REPO_ROOT, "AUNUEnv", "aunu_env", "users", "prompts", "responser_habit.json"
        )
        with open(_habit_file) as f:
            _all_habits = json.load(f)
        habit_dict = _all_habits[args.communication_habit]
        habit_key = f"habit_{args.communication_habit}"
    else:
        habit_dict = None
        habit_key = None

    seed_requirements: dict = {}
    if getattr(args, "seed_requirement_dir", None):
        seed_dir = args.seed_requirement_dir
        seed_requirements = _load_seed_requirements(seed_dir)
        logger.info(f"Loaded {len(seed_requirements)} seed requirements from {seed_dir}")
    elif args.strategy == "hybrid":
        seed_dir = _find_best_zero_shot_dir(cfg.dataset_name, cfg.agent_model)
        if seed_dir:
            seed_requirements = _load_seed_requirements(seed_dir)
            logger.info(f"Auto-selected zero_shot seed dir: {seed_dir} ({len(seed_requirements)} requirements)")
        else:
            logger.info(f"No matching zero_shot experiment found for dataset={cfg.dataset_name}, model={cfg.agent_model}; will run zero_shot inline per task.")

    out_path = os.path.join(exp_dir, "output.json")
    eval_path = os.path.join(exp_dir, "eval_results.json")

    # Load existing results if resuming, otherwise start fresh
    if os.path.exists(out_path):
        with open(out_path) as f:
            output = json.load(f)
        logger.info(f"Loaded existing output.json from {out_path} — will skip completed tasks")
    else:
        output = {
            "args": {
                "strategy_aunu": args.strategy,
                "aunu_model": cfg.agent_model,
                "mimic_model": cfg.user_model,
                "evaluator_model": cfg.evaluator_model,
                "persona": args.persona,
                "communication_habit": args.communication_habit if args.use_v2 else None,
                "dataset": cfg.dataset_name,
                "input_type": args.input_type,
                "max_turns": max_turns,
                "max_steps": cfg.max_steps,
                "user_mode": cfg.user_mode,
                "total_cost": 0.0,
            }
        }

    eval_output: dict = {}
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            eval_output = json.load(f)

    for task in tasks:
        persona_key = str(task.persona_id)
        task_num = _task_number(task.task_id)
        task_key = f"task_{task_num}"

        # Skip already-completed (resume support)
        if habit_key:
            already_done = (
                persona_key in output
                and habit_key in output.get(persona_key, {})
                and task_key in output[persona_key][habit_key]
            )
        else:
            already_done = persona_key in output and task_key in output.get(persona_key, {})
        if already_done:
            logger.info(f"[{task.task_id}] habit={args.communication_habit} Already done, skipping.")
            continue

        label = f"habit={args.communication_habit} " if habit_key else ""
        logger.info(f"[{task.task_id}] {label}Starting (persona={task.persona_id}, task={task_num})...")

        if args.use_v2:
            user = MimicUserV2(
                model_name=cfg.user_model,
                habit=habit_dict,
                temperature=cfg.effective_user_temperature,
            )
        else:
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
            elif args.strategy == "zero_shot_with_data_analysis":
                agent = ZeroShotWithDataAnalysisAgent.from_config(cfg, split=args.split)
                log = agent.run(env, task)
            elif args.strategy == "zero_shot_with_samples":
                agent = ZeroShotWithSamplesAgent.from_config(cfg, split=args.split)
                log = agent.run(env, task)
            elif args.strategy == "zero_shot_with_samples_reason":
                agent = ZeroShotWithSamplesReasonAgent.from_config(cfg, split=args.split)
                log = agent.run(env, task)
            elif args.strategy == "zero_shot_with_data_analysis_and_samples":
                agent = ZeroShotWithDataAnalysisAndSamplesAgent.from_config(cfg, split=args.split)
                log = agent.run(env, task)
            elif args.strategy == "zero_shot_with_data_summary":
                agent = ZeroShotWithDataSummaryAgent.from_config(cfg, split=args.split)
                log = agent.run(env, task)
            elif args.strategy == "data_interaction":
                agent = DataInteractionAgent.from_config(cfg, max_turns=max_turns)
                log = agent.run(env, task)
            elif args.strategy == "data_interaction_v2":
                agent = DataInteractionV2Agent.from_config(cfg, max_turns=max_turns)
                log = agent.run(env, task)
            elif args.strategy == "data_interaction_v3":
                agent = DataInteractionV3Agent.from_config(cfg, max_turns=max_turns)
                log = agent.run(env, task)
            elif args.strategy == "data_interaction_v4":
                agent = DataInteractionV4Agent.from_config(cfg, max_turns=max_turns)
                log = agent.run(env, task)
            elif args.strategy == "data_interaction_v5":
                agent = DataInteractionV5Agent.from_config(cfg, max_turns=max_turns)
                log = agent.run(env, task)
            elif args.strategy == "hybrid":
                seed_req = seed_requirements.get((task.persona_id, task_num))
                if seed_req is None:
                    logger.info(f"[{task.task_id}] No zero-shot seed found — running zero_shot first.")
                    zs_log = ZeroShotAgent.from_config(cfg).run(env, task)
                    seed_req = zs_log.get("task_requirement_final")
                agent = HybridAgent.from_config(cfg, max_iterations=max_iterations)
                log = agent.run(env, task, user, initial_requirement=seed_req)
            elif args.strategy == "hybrid_v2":
                mid_turn = getattr(args, "mid_turn", 6)
                agent = HybridV2Agent.from_config(cfg, max_iterations=max_iterations, mid_turn=mid_turn)
                seed_req = seed_requirements.get((task.persona_id, task_num))
                log = agent.run(env, task, user, initial_requirement=seed_req)
            else:
                agent = UserInteractionAgent.from_config(cfg, max_turns=max_turns)
                log = agent.run(env, task, user)

            task_result = _format_task_result(log, task, args, cfg, task_num)
            if habit_key:
                task_result["communication_habit"] = args.communication_habit
            scores = (log.get("eval_result") or {}).get("scores", {})
            recall = scores.get("recall", "N/A")
            precision = scores.get("precision", "N/A")
            f1 = scores.get("f1", "N/A")
            logger.info(f"[{task.task_id}] {label}Recall={recall}  Precision={precision}  F1={f1}")
            print(f"[{task.task_id}] Recall={recall}  Precision={precision}  F1={f1}")

        except Exception as e:
            logger.error(f"[{task.task_id}] {label}FAILED: {e}", exc_info=True)
            logger.error(f"[{task.task_id}] elevator_pitch:\n{task.elevator_pitch}")
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
            if habit_key:
                task_result["communication_habit"] = args.communication_habit

        # Update output.json
        if persona_key not in output:
            output[persona_key] = {}
        if habit_key:
            if habit_key not in output[persona_key]:
                output[persona_key][habit_key] = {}
            output[persona_key][habit_key][task_key] = task_result
        else:
            output[persona_key][task_key] = task_result

        output["args"]["total_cost"] = round(
            _sum_costs(output),
            6,
        )

        with open(out_path, "w") as f:
            json.dump(output, f, indent=2, default=str)

        # Update eval_results.json
        if log is not None:
            if persona_key not in eval_output:
                eval_output[persona_key] = {}
            if habit_key:
                if habit_key not in eval_output[persona_key]:
                    eval_output[persona_key][habit_key] = {}
                eval_output[persona_key][habit_key][task_key] = _format_eval_result(log)
            else:
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
    exp_dir = make_exp_dir(base_dir, args.exp_id)

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
