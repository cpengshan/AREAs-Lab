#!/usr/bin/env python3
"""Post-process a hybrid output.json to generate per-turn task requirements.

For each task, replays the hybrid interaction history incrementally and
calls a synthesis template at two types of checkpoints:

  - user_turn_N  (key: "turn_N"):      after each user-interaction turn,
                                        using hybrid_synthesis.jinja.
  - data_turn_N  (key: "data_turn_N"): after each data-interaction turn,
                                        using hybrid_v2_mid_synthesis.jinja.

Each checkpoint includes all interactions (user and data) that occurred up
to that point, so the synthesized requirement reflects the full evidence
available at that moment.

Usage:
    python agent/extract_turn_requirements_hybrid.py \\
        --exp_dir "agent/results copy/alexfabbri_multi_news/hybrid/Experiment3" \\
        --agent_model claude-sonnet-4-6 \\
        [--log_file_path path/to/log]
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (_REPO_ROOT, _AGENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from AUNUEnv.aunu_env.utils.llm import call_llm
from AUNUEnv.aunu_env.utils.jinja_utils import render_template

logger = logging.getLogger(__name__)

_PROMPTS_DIR = os.path.join(_AGENT_DIR, "prompts")
_SYNTHESIS_TEMPLATE = os.path.join(_PROMPTS_DIR, "hybrid", "hybrid_synthesis.jinja")
_MID_SYNTHESIS_TEMPLATE = os.path.join(_PROMPTS_DIR, "hybrid", "hybrid_v2_mid_synthesis.jinja")


def _extract_turn_checkpoints(task_data: dict) -> tuple[list[dict], list[dict]]:
    """Return checkpoints for both user-interaction and data-interaction turns.

    Returns (user_checkpoints, data_checkpoints).

    user_checkpoints — one entry per user-interaction turn, key "turn_N":
      {
        "kind":              "user",
        "turn":              <int>,   # 1-based index over user-interaction turns
        "router_turn":       <int>,
        "elevator_pitch":    <str>,
        "zero_shot_draft":   <str>,
        "current_req":       <str>,
        "user_interactions": [...],  # up to and including this turn
        "data_interactions": [...],  # all data turns before this user turn
      }

    data_checkpoints — one entry per data-interaction turn, key "data_turn_N":
      {
        "kind":              "data",
        "turn":              <int>,   # 1-based index over data-interaction turns
        "router_turn":       <int>,
        "elevator_pitch":    <str>,
        "zero_shot_draft":   <str>,
        "current_req":       <str>,
        "user_interactions": [...],  # all user turns before this data turn
        "data_interactions": [...],  # up to and including this data turn
      }
    """
    elevator_pitch = task_data.get("elevator_pitch", "")
    zero_shot_draft = task_data.get("zero_shot_draft", "")

    all_user = task_data.get("user_interactions", [])
    all_data = task_data.get("data_interactions", [])

    user_sorted = sorted(all_user, key=lambda x: x.get("turn", 0))
    data_sorted = sorted(all_data, key=lambda x: x.get("turn", 0))

    guideline_by_turn: dict[int, str] = {}
    for msg in task_data.get("messages", []):
        if msg.get("action") == "guideline_update":
            guideline_by_turn[len(guideline_by_turn)] = msg.get("output", "")
    guideline_outputs = list(guideline_by_turn.values())

    # ── user-interaction checkpoints ──────────────────────────────────────
    user_checkpoints: list[dict] = []
    for ui_idx, ui in enumerate(user_sorted):
        user_turn_num = ui.get("turn", 0)
        user_so_far = [u for u in user_sorted if u.get("turn", 0) <= user_turn_num]
        data_so_far = [d for d in data_sorted if d.get("turn", 0) <= user_turn_num]
        current_req = guideline_outputs[ui_idx] if ui_idx < len(guideline_outputs) else zero_shot_draft
        if not current_req:
            current_req = zero_shot_draft
        user_checkpoints.append({
            "kind": "user",
            "turn": ui_idx + 1,
            "router_turn": user_turn_num,
            "elevator_pitch": elevator_pitch,
            "zero_shot_draft": zero_shot_draft,
            "current_req": current_req,
            "user_interactions": user_so_far,
            "data_interactions": data_so_far,
        })

    # ── data-interaction checkpoints ──────────────────────────────────────
    data_checkpoints: list[dict] = []
    for di_idx, di in enumerate(data_sorted):
        data_turn_num = di.get("turn", 0)
        data_so_far = [d for d in data_sorted if d.get("turn", 0) <= data_turn_num]
        user_so_far = [u for u in user_sorted if u.get("turn", 0) <= data_turn_num]
        # current_req: use guideline output just before this data turn
        # (number of user turns seen so far determines which guideline to use)
        gi = len(user_so_far)
        current_req = guideline_outputs[gi - 1] if gi > 0 and gi - 1 < len(guideline_outputs) else zero_shot_draft
        if not current_req:
            current_req = zero_shot_draft
        data_checkpoints.append({
            "kind": "data",
            "turn": di_idx + 1,
            "router_turn": data_turn_num,
            "elevator_pitch": elevator_pitch,
            "zero_shot_draft": zero_shot_draft,
            "current_req": current_req,
            "user_interactions": user_so_far,
            "data_interactions": data_so_far,
        })

    return user_checkpoints, data_checkpoints


def process_task(
    task_data: dict,
    dataset: str,
    persona_id: int,
    task_id: int,
    model_name: str,
    temperature: float,
    max_tokens: int,
    existing_task: dict | None = None,
) -> dict:
    """Generate per-turn synthesis requirements for a single hybrid task.

    Returns a dict keyed by "turn_N" (user-interaction turns, hybrid_synthesis.jinja)
    and "data_turn_N" (data-interaction turns, hybrid_v2_mid_synthesis.jinja).

    Resume logic: any checkpoint whose router_turn is already covered by an
    entry in existing_task is skipped — this keeps the router_turn sequence
    continuous without recomputing entries that already exist.
    """
    existing_task = existing_task or {}

    # Build the set of router_turns already covered by any existing entry
    covered_router_turns: set[int] = {
        v["router_turn"]
        for v in existing_task.values()
        if isinstance(v, dict) and "router_turn" in v
    }

    user_checkpoints, data_checkpoints = _extract_turn_checkpoints(task_data)

    if not user_checkpoints and not data_checkpoints:
        logger.warning(f"  persona={persona_id} task={task_id}: no interaction turns found")
        return {}

    results: dict = {}

    # ── user-interaction turns (final synthesis) ──────────────────────────
    for cp in user_checkpoints:
        turn = cp["turn"]
        if cp["router_turn"] in covered_router_turns:
            logger.info(f"  persona={persona_id} task={task_id} user_turn={turn} (router_turn={cp['router_turn']}): already exists, skipping")
            continue
        logger.info(
            f"  persona={persona_id} task={task_id} user_turn={turn} "
            f"(router_turn={cp['router_turn']}): generating synthesis requirement"
        )

        start = datetime.now(timezone.utc).isoformat()
        prompt = render_template(
            _SYNTHESIS_TEMPLATE,
            initial_user_requirement=cp["elevator_pitch"],
            current_task_requirement=cp["current_req"],
            user_interactions=cp["user_interactions"] if cp["user_interactions"] else [],
            data_interactions=cp["data_interactions"] if cp["data_interactions"] else [],
        )
        result = call_llm(model_name, prompt, max_tokens=max_tokens, temperature=temperature)
        end = datetime.now(timezone.utc).isoformat()

        results[f"turn_{turn}"] = {
            "task_requirement_final": result["output"],
            "kind": "user",
            "turn": turn,
            "router_turn": cp["router_turn"],
            "persona": persona_id,
            "task_id": task_id,
            "dataset": dataset,
            "model": model_name,
            "num_user_interactions": len(cp["user_interactions"]),
            "num_data_interactions": len(cp["data_interactions"]),
            "cost": result.get("cost", 0.0),
            "input_tokens": result.get("input_tokens", 0),
            "output_tokens": result.get("output_tokens", 0),
            "start_time": start,
            "end_time": end,
        }

    # ── data-interaction turns (mid synthesis) ────────────────────────────
    for cp in data_checkpoints:
        turn = cp["turn"]
        if cp["router_turn"] in covered_router_turns:
            logger.info(f"  persona={persona_id} task={task_id} data_turn={turn} (router_turn={cp['router_turn']}): already exists, skipping")
            continue
        logger.info(
            f"  persona={persona_id} task={task_id} data_turn={turn} "
            f"(router_turn={cp['router_turn']}): generating mid-synthesis requirement"
        )

        start = datetime.now(timezone.utc).isoformat()
        prompt = render_template(
            _MID_SYNTHESIS_TEMPLATE,
            initial_user_requirement=cp["elevator_pitch"],
            current_task_requirement=cp["current_req"],
            user_interactions=cp["user_interactions"] if cp["user_interactions"] else [],
            data_interactions=cp["data_interactions"] if cp["data_interactions"] else [],
        )
        result = call_llm(model_name, prompt, max_tokens=max_tokens, temperature=temperature)
        end = datetime.now(timezone.utc).isoformat()

        results[f"data_turn_{turn}"] = {
            "task_requirement_final": result["output"],
            "kind": "data",
            "turn": turn,
            "router_turn": cp["router_turn"],
            "persona": persona_id,
            "task_id": task_id,
            "dataset": dataset,
            "model": model_name,
            "num_user_interactions": len(cp["user_interactions"]),
            "num_data_interactions": len(cp["data_interactions"]),
            "cost": result.get("cost", 0.0),
            "input_tokens": result.get("input_tokens", 0),
            "output_tokens": result.get("output_tokens", 0),
            "start_time": start,
            "end_time": end,
        }

    return results


def process_experiment(exp_dir: str, model_name: str, temperature: float, max_tokens: int) -> dict:
    """Process all tasks in a hybrid experiment directory.

    Reads output.json and writes turn_requirements.json.
    Resumes automatically: if turn_requirements.json already exists, any
    turn key already present in it is skipped — only missing keys are computed.
    """
    out_path = os.path.join(exp_dir, "output.json")
    if not os.path.exists(out_path):
        raise FileNotFoundError(f"output.json not found in {exp_dir}")

    with open(out_path) as f:
        data = json.load(f)

    args_meta = data.get("args", {})
    dataset = args_meta.get("dataset", "")
    total_cost = 0.0

    # Load existing turn_requirements.json if present (resume mode)
    result_path = os.path.join(exp_dir, "turn_requirements.json")
    if os.path.exists(result_path):
        with open(result_path) as f:
            turn_output = json.load(f)
        logger.info(f"Resuming from existing {result_path}")
    else:
        turn_output = {
            "args": {
                "source_exp_dir": exp_dir,
                "model": model_name,
                "dataset": dataset,
                "strategy": "hybrid_multi_rounds",
                "source_strategy": args_meta.get("strategy_aunu", "hybrid"),
                "total_cost": 0.0,
            }
        }

    for persona_key, persona_val in data.items():
        if persona_key == "args" or not isinstance(persona_val, dict):
            continue

        persona_id = int(persona_key)
        turn_output.setdefault(persona_key, {})

        for habit_key, habit_val in persona_val.items():
            if not isinstance(habit_val, dict):
                continue
            turn_output[persona_key].setdefault(habit_key, {})

            for task_key, task_val in habit_val.items():
                if not isinstance(task_val, dict):
                    continue

                task_id = task_val.get("task_id")
                if task_id is None:
                    continue

                existing_task = turn_output[persona_key][habit_key].get(task_key, {})

                logger.info(f"Processing persona={persona_id} habit={habit_key} {task_key}")
                try:
                    per_turn = process_task(
                        task_data=task_val,
                        dataset=dataset,
                        persona_id=persona_id,
                        task_id=task_id,
                        model_name=model_name,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        existing_task=existing_task,
                    )
                    task_cost = sum(v.get("cost", 0.0) for v in per_turn.values())
                    total_cost += task_cost
                    # Merge new keys into existing task dict
                    existing_task.update(per_turn)
                    turn_output[persona_key][habit_key][task_key] = existing_task
                    logger.info(
                        f"  Done: {len(per_turn)} new turns computed "
                        f"({len(existing_task)} total), cost={task_cost:.4f}"
                    )
                except Exception as e:
                    logger.error(f"  FAILED: {e}", exc_info=True)
                    turn_output[persona_key][habit_key].setdefault(task_key, {})["error"] = str(e)

                # Save incrementally after each task
                with open(result_path, "w") as f:
                    json.dump(turn_output, f, indent=2, default=str)

    turn_output["args"]["total_cost"] = round(
        turn_output["args"].get("total_cost", 0.0) + total_cost, 6
    )

    with open(result_path, "w") as f:
        json.dump(turn_output, f, indent=2, default=str)
    logger.info(f"Saved turn requirements → {result_path} (new_cost={total_cost:.4f})")
    return turn_output


def main():
    parser = argparse.ArgumentParser(
        description="Generate per-turn task requirements from a hybrid output.json"
    )
    parser.add_argument(
        "--exp_dir", required=True,
        help="Path to hybrid experiment directory containing output.json",
    )
    parser.add_argument(
        "--agent_model", type=str, default="claude-sonnet-4-6",
        help="LLM model to use for synthesis (default: claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.4,
        help="Sampling temperature (default: 0.4)",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=4096,
        help="Max output tokens per requirement (default: 4096)",
    )
    parser.add_argument(
        "--log_file_path", type=str, default=None,
        help="Path to log file (default: <exp_dir>/turn_requirements.log)",
    )
    args = parser.parse_args()

    log_path = args.log_file_path or os.path.join(args.exp_dir, "turn_requirements.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
    )

    logger.info(f"Processing hybrid experiment: {args.exp_dir}")
    process_experiment(
        exp_dir=args.exp_dir,
        model_name=args.agent_model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    logger.info("Done.")


if __name__ == "__main__":
    main()
