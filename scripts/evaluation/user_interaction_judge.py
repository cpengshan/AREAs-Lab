#!/usr/bin/env python3
"""
LLM-based judge for evaluating user interaction experiment results.

Compares task_requirement_final against ground_truth for each entry
in the experiment output, using the LLM_judge.jinja prompt template.
"""

import argparse
import json
import re
import sys
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.model_base import LLM


def load_template(template_path: str):
    template_dir = str(Path(template_path).parent)
    template_file = Path(template_path).name
    env = Environment(loader=FileSystemLoader(template_dir))
    return env.get_template(template_file)


def render_prompt(template, task_requirement_final: str, ground_truth: str) -> str:
    return template.render(
        task_requirement_final=task_requirement_final,
        ground_truth=ground_truth,
    )


def call_llm(llm: LLM, prompt: str) -> tuple[str, float]:
    result = llm.generate(prompt, max_tokens=4096)
    return result["output"], result["cost"]


def parse_json_response(response: str) -> dict:
    # Strip markdown code fences if present
    cleaned = re.sub(r"^```(?:json)?\s*", "", response.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return json.loads(cleaned)


def main():
    parser = argparse.ArgumentParser(
        description="LLM judge for user interaction experiment evaluation."
    )
    parser.add_argument("--id", required=True, help="Experiment ID (e.g. 3)")
    parser.add_argument(
        "--model",
        required=True,
        help="OpenAI model ID (e.g. gpt-5.4-mini)",
    )
    parser.add_argument(
        "--ids",
        nargs="+",
        default=None,
        help="Subset of entry IDs to evaluate (e.g. --ids 1 3). Defaults to all IDs.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    experiment_dir = (
        repo_root
        / "results"
        / "ccdv_patent-classification"
        / "user"
        / f"Experiment{args.id}"
    )
    input_path = experiment_dir / "output.json"
    output_path = experiment_dir / "eval_results.json"
    template_path = repo_root / "prompts" / "Evaluation" / "LLM_judge.jinja"

    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    if not template_path.exists():
        print(f"Error: template not found: {template_path}", file=sys.stderr)
        sys.exit(1)

    with open(input_path) as f:
        data = json.load(f)

    # Load existing results to merge into (preserves already-successful evaluations)
    eval_results = {}
    if output_path.exists():
        with open(output_path) as f:
            eval_results = json.load(f)

    # Determine which IDs to evaluate
    if args.ids is not None:
        ids_to_run = args.ids
    else:
        ids_to_run = list(data.keys())

    template = load_template(str(template_path))
    llm = LLM(args.model)
    total_cost = 0.0

    for entry_id in ids_to_run:
        if entry_id not in data:
            print(f"[{entry_id}] Skipping: ID not found in input file")
            continue
        entry = data[entry_id]
        task_req = entry.get("task_requirement_final", "")
        ground_truth = entry.get("ground_truth", "")

        if not task_req or not ground_truth:
            print(f"[{entry_id}] Skipping: missing task_requirement_final or ground_truth")
            eval_results[entry_id] = {"error": "missing required fields"}
            continue

        print(f"[{entry_id}] Evaluating with model {args.model} ...")
        prompt = render_prompt(template, task_req, ground_truth)

        try:
            response, cost = call_llm(llm, prompt)
            total_cost += cost
            result = parse_json_response(response)
            eval_results[entry_id] = result
            scores = result.get("scores", {})
            print(f"[{entry_id}] Scores: {scores} | Cost: ${cost:.6f}")
        except json.JSONDecodeError as e:
            print(f"[{entry_id}] Failed to parse LLM response as JSON: {e}", file=sys.stderr)
            print(f"LLM input: \n {prompt}")
            print(f"LLM response: \n {response}")
            eval_results[entry_id] = {"error": "json_parse_error", "raw_response": response}
        except Exception as e:
            print(f"[{entry_id}] Error: {e}", file=sys.stderr)
            eval_results[entry_id] = {"error": str(e)}

    with open(output_path, "w") as f:
        json.dump(eval_results, f, indent=2)

    print(f"\nEvaluation results saved to: {output_path}")
    print(f"Total cost: ${total_cost:.6f}")


if __name__ == "__main__":
    main()
