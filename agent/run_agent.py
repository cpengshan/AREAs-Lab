#!/usr/bin/env python3
"""Entry point for running AREAEnv-based agents.

Mirrors the interface of scripts/agents/run_area_interaction.py but routes
all MIMIC-user interaction through AREAEnv, keeping the agent's strategy as
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

from AREAEnv.area_env.config import AREAEnvConfig
from AREAEnv.area_env.dataset.loader import load_dataset, DatasetRegistry
from AREAEnv.area_env.env.area_env import AREAEnv
from AREAEnv.area_env.evaluator.atomic_evaluator import AtomicEvaluator
from AREAEnv.area_env.evaluator.metrics import aggregate_results
from AREAEnv.area_env.utils.llm import call_llm
from AREAEnv.area_env.utils.jinja_utils import render_template


class _NoOpEvaluator:
    """Drop-in replacement for AtomicEvaluator that skips all LLM calls."""

    def evaluate(self, predicted: str, gold: str, task_id=None) -> dict:
        return {"gold_units": [], "predicted_units": [], "comparison": {},
                "counts": {}, "scores": {}, "gold_categories": {},
                "pred_categories": {}, "subcategory_scores": {}, "cost": 0.0}
from AREAEnv.area_env.users.mimic_user import MimicUser

from zero_shot_agent import ZeroShotAgent
from user_interaction_agent import UserInteractionAgent
from hybrid_agent import HybridAgent
from zero_shot_variants_agent import DataInteractionAgent

RESULTS_DIR = os.path.join(_AGENT_DIR, "results")
DATA_SYNTHESIZED_DIR = os.path.join(_REPO_ROOT, "AREAEnv/data/data_synthesized")


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run AREAEnv agents (zero-shot or user-interaction)"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to AREAEnvConfig YAML file. All model/temperature/max_steps/user_mode "
             "settings are read from the config; CLI flags override individual fields.",
    )
    parser.add_argument(
        "--strategy",
        choices=[
            "zero_shot", "data_interaction",
            "user_interaction", "hybrid",
        ],
        default="zero_shot",
        help="Agent strategy (default: zero_shot)",
    )
    parser.add_argument("--agent_model", type=str, default=None,
                        help="LLM model for the AREA agent (overrides config)")
    parser.add_argument("--mimic_model", type=str, default=None,
                        help="LLM model for the MIMIC user (overrides config)")
    parser.add_argument("--evaluator_model", type=str, default=None,
                        help="LLM model for the evaluator (overrides config)")
    parser.add_argument(
        "--persona",
        type=int, nargs="*", default=None,
        help="Persona IDs to run (e.g. --persona 1 2 3). Omit to run all personas.",
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
        "--communication_habit", type=str, default=None,
        choices=["passive", "neutral", "active"],
        help="Communication habit for MimicUser (passive/neutral/active). Required for user_interaction and hybrid strategies.",
    )
    parser.add_argument(
        "--seed_requirement_dir", type=str, default=None,
        help="Path to a prior experiment directory whose output.json task_requirement_final values "
             "are used as the initial current_task_requirement for hybrid runs (e.g. zero_shot/Experiment10).",
    )
    parser.add_argument(
        "--turn_id", type=int, default=1,
        help="Round number for multi-round pipelines (e.g. data_multi_rounds). "
             "Passed to data_interaction so the prompt knows which iteration this is.",
    )
    return parser.parse_args()


def _resolve_config(args: argparse.Namespace) -> "AREAEnvConfig":
    """Load config from YAML (if provided) then apply any CLI overrides."""
    if args.config:
        cfg = AREAEnvConfig.from_yaml(args.config)
    else:
        cfg = AREAEnvConfig()

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


def _format_task_result(log: dict, task, args: argparse.Namespace, cfg: "AREAEnvConfig", task_num: int) -> dict:
    """Convert an agent trajectory log into the canonical output.json task format.

    Matches the schema used by scripts/agents/run_area_interaction.py:
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
    if args.strategy == "hybrid":
        result["router_history"] = log.get("router_history", [])
        result["user_interactions"] = log.get("user_interactions", [])
        result["data_interactions"] = log.get("data_interactions", [])
        result["format_reflection_history"] = log.get("format_reflection_history", [])
        result["elevator_pitch"] = getattr(task, "elevator_pitch", "")
        result["zero_shot_draft"] = log.get("zero_shot_draft", "")
    if args.strategy == "data_interaction" and log.get("modifications") is not None:
        result["modifications"] = log["modifications"]
    return result


def _find_best_zero_shot_dir(dataset_name: str, area_model: str) -> str | None:
    """Return the zero_shot ExperimentN dir with the largest N whose output.json matches area_model."""
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
            if data.get("args", {}).get("area_model") != area_model:
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


def run_experiment(args: argparse.Namespace, cfg: "AREAEnvConfig", exp_dir: str) -> dict:
    """Run the benchmark for all requested personas and tasks.

    Saves incremental results to output.json after each task so progress is
    not lost on failure.

    Returns:
        The full results dict (same as written to output.json).
    """
    max_turns = args.max_turns if args.max_turns is not None else 5
    max_iterations = args.max_iterations if args.max_iterations is not None else 6

    # ── Evaluator ──────────────────────────────────────────────────────────
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

    # ── Dataset registry — env owns all data access ────────────────────────
    registry = DatasetRegistry(
        data_synthesized_root=cfg.data_synthesized_root,
        synthesized_output_file=cfg.synthesized_output_file,
    )

    # ── Enumerate task IDs via the registry (agents receive only identifiers) ──
    all_tasks = load_dataset(
        synthesized_path=cfg.dataset_path,
        dataset_name=cfg.dataset_name,
    )
    if args.persona:
        tasks = [t for t in all_tasks if t.persona_id in args.persona]
    else:
        tasks = list(all_tasks)
    tasks = sorted(tasks, key=lambda t: (t.persona_id, int(t.task_id.rsplit("_", 1)[-1])))
    if not tasks:
        raise ValueError(f"No tasks found for dataset '{cfg.dataset_name}' "
                         f"(persona filter: {args.persona})")
    persona_desc = str(args.persona) if args.persona else "all"
    logger.info(f"Running {len(tasks)} tasks for personas={persona_desc}")

    # ── Communication habit ─────────────────────────────────────────────────
    if args.strategy in ("user_interaction", "hybrid") and not args.communication_habit:
        raise ValueError("--communication_habit (passive/neutral/active) is required for user_interaction and hybrid strategies")
    _habit_file = os.path.join(
        _REPO_ROOT, "AREAEnv", "area_env", "users", "prompts", "responser_habit.json"
    )
    with open(_habit_file) as f:
        _all_habits = json.load(f)
    habit_dict = _all_habits[args.communication_habit] if args.communication_habit else _all_habits["neutral"]
    habit_key = f"habit_{args.communication_habit}" if args.communication_habit else None

    # ── User (created once — stateless across tasks) ───────────────────────
    user = MimicUser(
        model_name=cfg.user_model,
        habit=habit_dict,
        temperature=cfg.effective_user_temperature,
    )

    # ── Environment (owns user + registry) ────────────────────────────────
    env = AREAEnv(
        evaluator=evaluator,
        user=user,
        registry=registry,
        max_steps=cfg.max_steps,
        data_sampled_root=cfg.data_sampled_root,
        data_sampled_file=getattr(cfg, "data_sampled_file", "data_sampled.json"),
    )

    seed_requirements: dict = {}
    if getattr(args, "seed_requirement_dir", None):
        seed_dir = args.seed_requirement_dir
        seed_requirements = _load_seed_requirements(seed_dir)
        logger.info(f"Loaded {len(seed_requirements)} seed requirements from {seed_dir}")

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
                "strategy_area": args.strategy,
                "area_model": cfg.agent_model,
                "mimic_model": cfg.user_model,
                "evaluator_model": cfg.evaluator_model,
                "persona": args.persona,
                "communication_habit": args.communication_habit,
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

        log = None

        try:
            if args.strategy == "zero_shot":
                agent = ZeroShotAgent.from_config(cfg)
                log = agent.run(env, task.dataset_name, task.persona_id, task.task_id)
            elif args.strategy == "data_interaction":
                seed_req = seed_requirements.get((task.persona_id, task_num))
                turn_id = getattr(args, "turn_id", 1)
                agent = DataInteractionAgent.from_config(
                    cfg, split=args.split,
                    initial_requirement=seed_req,
                    turn_id=turn_id,
                )
                log = agent.run(env, task.dataset_name, task.persona_id, task.task_id)
            elif args.strategy == "hybrid":
                logger.info(f"[{task.task_id}] Generating zero_shot seed (no evaluation).")
                _zs_template = os.path.join(os.path.dirname(__file__), "prompts", "zero_shot.jinja")
                _zs_prompt = render_template(_zs_template, user_instruction=task.elevator_pitch)
                _zs_result = call_llm(cfg.agent_model, _zs_prompt,
                                      max_tokens=cfg.max_tokens,
                                      temperature=cfg.effective_agent_temperature)
                seed_req = _zs_result["output"]
                agent = HybridAgent.from_config(cfg, max_iterations=max_iterations)
                log = agent.run(env, task.dataset_name, task.persona_id, task.task_id,
                                initial_requirement=seed_req)
            else:
                agent = UserInteractionAgent.from_config(cfg, max_turns=max_turns)
                log = agent.run(env, task.dataset_name, task.persona_id, task.task_id)

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
