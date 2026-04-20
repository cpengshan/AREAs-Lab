#!/usr/bin/env python3
"""
Pre-compute atomic units for ground-truth task requirements.

Reads ground truth from any strategy's output.json, decomposes each
(persona_id, task_key) pair once, and saves the result to:
    results/<dataset>/ground_truth_units.json

Shape of that file:
{
  "<persona_id>": {
    "<task_key>": {
      "ground_truth": "<original text>",
      "atomic_units": ["...", ...]
    },
    ...
  },
  ...
}

Run:
    python scripts/evaluation/precompute_gt_units.py \
        --dataset alexfabbri_multi_news \
        --model gpt-4.1-mini
"""

import argparse
import json
import re
import sys
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.model_base import LLM

STRATEGY_PRIORITY = ["zero_shot", "user", "persona", "data", "hybrid"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pre-compute ground-truth atomic units for a dataset."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset folder name under results/ (e.g. alexfabbri_multi_news)",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model to use for decomposition (e.g. gpt-4.1-mini)",
    )
    parser.add_argument(
        "--ids",
        nargs="+",
        default=None,
        help="Subset of persona IDs to process. Defaults to all.",
    )
    return parser.parse_args()


def load_template(path: Path):
    env = Environment(loader=FileSystemLoader(str(path.parent)))
    return env.get_template(path.name)


def parse_json_response(response: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?\s*", "", response.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return json.loads(cleaned)


def decompose(llm: LLM, template, requirement: str) -> tuple[list[str], float]:
    prompt = template.render(task_requirement=requirement)
    result = llm.generate(prompt, max_tokens=16384, temperature=0)
    print(f"  decompose result: {result}")
    response, cost = result["output"], result["cost"]
    try:
        data = parse_json_response(response)
    except Exception as e:
        e.raw_response = f"[decompose] {response}"
        raise
    return data["atomic_units"], cost


def find_source_output(dataset_dir: Path) -> Path:
    """Return the first output.json found under a known strategy sub-directory."""
    for strategy in STRATEGY_PRIORITY:
        for exp_dir in sorted((dataset_dir / strategy).glob("Experiment*")):
            candidate = exp_dir / "output.json"
            if candidate.exists():
                return candidate
    raise FileNotFoundError(
        f"No output.json found under any strategy in {dataset_dir}"
    )


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    dataset_dir = repo_root / "results" / args.dataset
    prompts_dir = repo_root / "prompts" / "Evaluation"

    decompose_tpl_path = prompts_dir / "LLM_judge_decompose.jinja"
    if not decompose_tpl_path.exists():
        print(f"Error: {decompose_tpl_path} not found", file=sys.stderr)
        sys.exit(1)

    source_path = find_source_output(dataset_dir)
    print(f"Loading ground truths from: {source_path}")
    with open(source_path) as f:
        data = json.load(f)

    output_path = dataset_dir / "ground_truth_units.json"
    gt_units: dict = {}
    if output_path.exists():
        with open(output_path) as f:
            gt_units = json.load(f)

    ids_to_run = args.ids if args.ids is not None else [k for k in data if k != "args"]
    decompose_tpl = load_template(decompose_tpl_path)
    llm = LLM(args.model)
    total_cost = 0.0

    for persona_id in ids_to_run:
        if persona_id not in data:
            print(f"[persona {persona_id}] Not found in source data, skipping.")
            continue

        if persona_id not in gt_units:
            gt_units[persona_id] = {}

        for task_key, entry in data[persona_id].items():
            label = f"persona {persona_id} / {task_key}"

            if task_key in gt_units[persona_id] and "atomic_units" in gt_units[persona_id][task_key]:
                print(f"[{label}] Already decomposed, skipping.")
                continue

            ground_truth = entry.get("ground_truth", "")
            if not ground_truth:
                print(f"[{label}] No ground truth found, skipping.")
                gt_units[persona_id][task_key] = {"error": "missing ground_truth"}
                continue

            print(f"[{label}] Decomposing ground truth ...")
            try:
                units, cost = decompose(llm, decompose_tpl, ground_truth)
                total_cost += cost
                gt_units[persona_id][task_key] = {
                    "ground_truth": ground_truth,
                    "atomic_units": units,
                }
                print(f"[{label}] {len(units)} units | Cost so far: ${total_cost:.6f}")
            except Exception as e:
                raw = getattr(e, "raw_response", None)
                print(f"[{label}] Error: {e}")
                if raw:
                    print(f"LLM response:\n{raw}")
                gt_units[persona_id][task_key] = {"error": str(e)}
            finally:
                with open(output_path, "w") as f:
                    json.dump(gt_units, f, indent=2)

    print(f"\nGround truth units saved to: {output_path}")
    print(f"Total cost: ${total_cost:.6f}")


if __name__ == "__main__":
    main()
