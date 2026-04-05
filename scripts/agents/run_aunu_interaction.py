import logging
import argparse
import json
import os
from interactive_manager import create_workflow

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "../../results")

MIMIC_REQUIRED_STRATEGIES = {"user", "mix"}

DATA_SYNTHESIZED_DIR = os.path.join(os.path.dirname(__file__), "../../data/data_synthesized")


def load_persona(dataset: str, persona_id: int) -> dict:
    users_json = os.path.join(DATA_SYNTHESIZED_DIR, dataset, "users.json")
    with open(users_json) as f:
        users = json.load(f)
    for user in users:
        if user["persona_id"] == persona_id:
            return user
    raise ValueError(f"Persona ID {persona_id} not found in {users_json}")


def load_user_instruction(dataset: str, persona_id: int, input_type: str) -> str:
    task_path = os.path.join(DATA_SYNTHESIZED_DIR, dataset, "synthesized_tasks", f"persona_{persona_id}.json")
    with open(task_path) as f:
        data = json.load(f)
    return data[input_type]


def load_ground_truth(dataset: str, persona_id: int) -> str:
    task_path = os.path.join(DATA_SYNTHESIZED_DIR, dataset, "synthesized_tasks", f"persona_{persona_id}.json")
    with open(task_path) as f:
        return json.load(f)["task_requirement"]


def parse_args():
    parser = argparse.ArgumentParser(description="Args Parser for AUNU simulation")
    parser.add_argument("--log_file_path", type=str, default="/local/scratch/zzh2365/AUNU/scripts/loggings/zero_shot/persona.log", help="Path to store the log file")
    parser.add_argument(
        "--strategy",
        type=str,
        default="zero_shot",
        choices=["zero_shot", "user", "data", "mix"],
        help="AUNU strategy (default: zero_shot)",
    )
    parser.add_argument(
        "--aunu_model",
        type=str,
        default="gemini-3.1-flash-lite-preview",
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
    args = parser.parse_args()

    if args.strategy in MIMIC_REQUIRED_STRATEGIES and args.mimic_model is None:
        parser.error(f"--mimic_model is required when strategy is '{args.strategy}'")

    return args


def run_persona(args, persona_id: int, all_results: dict, out_path: str):
    persona_key = str(persona_id)
    if persona_key in all_results:
        print(f"Persona {persona_id} already evaluated. Skipping.")
        return

    persona = load_persona(args.dataset, persona_id)
    user_instruction_init = load_user_instruction(args.dataset, persona_id, args.input_type)
    args.user_instruction_init = user_instruction_init
    args.persona_profile = persona

    app = create_workflow(args)

    initial_message = {
        "start_time": None,
        "end_time": None,
        "role": "user",
        "action": "init",
        "input": "",
        "output": user_instruction_init,
        "llm": None,
    }
    initial_state = {
        "messages": [initial_message],
        "is_complete": False,
        "task_requirement_final": "",
    }

    print(f"🚀 Starting AU-NU Simulation for persona {persona_id}...")
    final_state = {}
    for output in app.stream(initial_state):
        for node_name, state_update in output.items():
            print(f"\n--- Node: {node_name} ---")
            if "messages" in state_update:
                print(f"Message: {state_update['messages'][-1]['output']}")
            final_state.update(state_update)

    print(f"\n✅ Simulation Complete for persona {persona_id}.")

    final_state["strategy"] = args.strategy
    final_state["persona"] = persona_id
    final_state["dataset"] = args.dataset
    final_state["model"] = args.aunu_model
    final_state["input_type"] = args.input_type
    final_state["ground_truth"] = load_ground_truth(args.dataset, persona_id)

    all_results[persona_key] = final_state
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {out_path}")


def run_simulation():
    args = parse_args()

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

    out_dir = os.path.join(RESULTS_DIR, args.dataset.replace("/", "_"), args.strategy)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.aunu_model}.json")

    all_results = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            all_results = json.load(f)

    for persona_id in args.persona:
        run_persona(args, persona_id, all_results, out_path)

if __name__ == "__main__":
    run_simulation()