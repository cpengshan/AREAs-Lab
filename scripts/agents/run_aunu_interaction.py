import logging
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))  # ensure agents/ is on path

from interactive_manager import create_workflow

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "../../results")

MIMIC_REQUIRED_STRATEGIES = {"user", "mix"}

DATA_SYNTHESIZED_DIR = os.path.join(os.path.dirname(__file__), "../../data/data_synthesized")


def _load_synthesized_output(dataset: str) -> dict:
    output_path = os.path.join(DATA_SYNTHESIZED_DIR, dataset, "synthesized_output.json")
    with open(output_path) as f:
        return json.load(f)


def _get_task(data: dict, persona_id: int, task_id: int) -> dict:
    user_key = f"user_{persona_id}"
    if user_key not in data:
        raise ValueError(f"Persona ID {persona_id} not found in synthesized_output.json")
    tasks = data[user_key]["tasks_info"]
    for task in tasks:
        if task["task_id"] == task_id:
            return task
    raise ValueError(f"Task ID {task_id} not found for persona {persona_id}")


def load_persona(dataset: str, persona_id: int) -> dict:
    data = _load_synthesized_output(dataset)
    user_key = f"user_{persona_id}"
    if user_key not in data:
        raise ValueError(f"Persona ID {persona_id} not found in synthesized_output.json")
    return data[user_key]["user_info"]


def load_user_instruction(dataset: str, persona_id: int, input_type: str, task_id: int = 1) -> str:
    data = _load_synthesized_output(dataset)
    task = _get_task(data, persona_id, task_id)
    return task[input_type]


def load_ground_truth(dataset: str, persona_id: int, task_id: int = 1) -> str:
    data = _load_synthesized_output(dataset)
    task = _get_task(data, persona_id, task_id)
    return task["task_requirement"]


def parse_args():
    parser = argparse.ArgumentParser(description="Args Parser for AUNU simulation")
    parser.add_argument("--log_file_path", type=str, default=os.path.join(os.path.dirname(__file__), "../loggings/zero_shot/run.log"), help="Path to store the log file")
    parser.add_argument(
        "--strategy_aunu",
        type=str,
        default="zero_shot",
        choices=["zero_shot", "persona", "user", "data", "mix"],
        help="AUNU strategy (default: zero_shot)",
    )
    parser.add_argument(
        "--strategy_mimic",
        type=str,
        default="zero_shot",
        choices=["zero_shot", "user", "data", "mix"],
        help="MIMIC strategy (default: zero_shot)",
    )
    parser.add_argument(
        "--aunu_model",
        type=str,
        default="gemini/gemini-3.1-flash-lite-preview",
        help="Model name for the AUNU agent",
    )
    parser.add_argument(
        "--mimic_model",
        type=str,
        default=None,
        help="Model name for the MimicUser agent (required when strategy is 'user' or 'mix')",
    )
    parser.add_argument(
        "--persona",
        type=int,
        nargs="+",
        required=True,
        help="One or more persona IDs (e.g. --persona 1 3 5)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ccdv/patent-classification",
        help="Dataset name under data/data_synthesized/ (default: ccdv/patent-classification)",
    )
    parser.add_argument(
        "--input_type",
        type=str,
        default="elevator_pitch_summary",
        choices=["elevator_pitch_summary", "deep_dive_summary"],
        help="Which field from the task file to use as the user instruction (default: elevator_pitch_summary)",
    )
    parser.add_argument(
        "--prompt_template_path",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "../../prompts/agents/mimic_user"),
        help="Directory containing MimicUser Jinja2 templates",
    )
    parser.add_argument(
        "--prompt_template_user_feedback_path",
        type=str,
        default="feedback_mimic_user.jinja",
        help="Filename of the MimicUser feedback template (relative to prompt_template_path)",
    )
    parser.add_argument(
        "--max_turns",
        type=int,
        default=5,
        help="Maximum number of aunu<->mimic_user turns before stopping (default: 5)",
    )
    parser.add_argument(
        "--aunu_template",
        type=str,
        default="user_interaction.jinja",
        help="Filename of the AUNU user-interaction Jinja2 template (relative to prompts/agents/aunu_agent/)",
    )
    parser.add_argument(
        "--mimic_template",
        type=str,
        default="feedback_mimic_user.jinja",
        help="Filename of the MimicUser feedback Jinja2 template (relative to prompts/agents/mimic_user/)",
    )
    args = parser.parse_args()

    if args.strategy_aunu in MIMIC_REQUIRED_STRATEGIES and args.mimic_model is None:
        parser.error(f"--mimic_model is required when strategy is '{args.strategy_aunu}'")

    return args


def make_exp_dir(base_dir: str) -> str:
    """Create and return the next available ExperimentN folder under base_dir."""
    exp_id = 1
    while os.path.exists(os.path.join(base_dir, f"Experiment{exp_id}")):
        exp_id += 1
    exp_dir = os.path.join(base_dir, f"Experiment{exp_id}")
    os.makedirs(exp_dir)
    return exp_dir


def build_args_dict(args) -> dict:
    """Build a serialisable args dict to embed in output.json."""
    settings = {k: v for k, v in vars(args).items()
                if not k.startswith("_") and k not in ("persona_profile", "task_requirement_gold", "user_instruction_init", "task_id")}
    settings["persona"] = [int(p) for p in settings["persona"]]
    return settings


def run_task(args, persona_id: int, task_id: int) -> dict:
    """Run the simulation for a single persona + task and return the result dict."""
    persona = load_persona(args.dataset, persona_id)
    args.user_instruction_init = load_user_instruction(args.dataset, persona_id, args.input_type, task_id)
    args.persona_profile = persona
    args.task_requirement_gold = load_ground_truth(args.dataset, persona_id, task_id)

    app = create_workflow(args)

    initial_state = {
        "messages": [],
        "is_complete": False,
        "task_requirement_final": "",
    }

    logging.info(f"Starting AU-NU Simulation for persona {persona_id}, task {task_id}...")
    all_messages = []
    final_state = {}
    for output in app.stream(initial_state):
        for node_name, state_update in output.items():
            if "messages" in state_update:
                all_messages.extend(state_update["messages"])
            final_state.update(state_update)
    final_state["messages"] = all_messages

    logging.info(f"Simulation Complete for persona {persona_id}, task {task_id}.")

    final_state["strategy"] = args.strategy_aunu
    final_state["persona"] = persona_id
    final_state["task_id"] = task_id
    final_state["dataset"] = args.dataset
    final_state["model"] = args.aunu_model
    final_state["input_type"] = args.input_type
    final_state["ground_truth"] = load_ground_truth(args.dataset, persona_id, task_id)
    final_state["cost"] = round(sum(m.get("cost", 0.0) for m in all_messages), 6)

    return final_state


def run_persona(args, persona_id: int, all_results: dict, exp_dir: str):
    persona_key = str(persona_id)
    already_done = set(all_results.get(persona_key, {}).keys()) - {"args"}

    for task_id in (1, 2):
        task_key = f"task_{task_id}"
        if task_key in already_done:
            logging.info(f"Persona {persona_id} task {task_id} already evaluated. Skipping.")
            continue

        task_result = run_task(args, persona_id, task_id)

        if persona_key not in all_results:
            all_results[persona_key] = {}
        all_results[persona_key][task_key] = task_result

        # Update total cost in args metadata
        total_cost = sum(
            tasks[tk].get("cost", 0.0)
            for pid, tasks in all_results.items()
            if pid not in ("args",)
            for tk in tasks
        )
        all_results["args"]["total_cost"] = round(total_cost, 6)

        out_path = os.path.join(exp_dir, "output.json")
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logging.info(f"Results saved to {out_path}")

    # Create empty eval_results.json placeholder
    eval_path = os.path.join(exp_dir, "eval_results.json")
    if not os.path.exists(eval_path):
        with open(eval_path, "w") as f:
            json.dump({}, f, indent=2)


def run_simulation():
    args = parse_args()

    base_dir = os.path.join(RESULTS_DIR, args.dataset.replace("/", "_"), args.strategy_aunu)
    os.makedirs(base_dir, exist_ok=True)
    exp_dir = make_exp_dir(base_dir)

    # --- Logging Configuration ---
    os.makedirs(os.path.dirname(args.log_file_path), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(args.log_file_path),
            logging.StreamHandler()
        ]
    )

    all_results = {"args": build_args_dict(args)}
    for persona_id in args.persona:
        run_persona(args, persona_id, all_results, exp_dir)

if __name__ == "__main__":
    run_simulation()