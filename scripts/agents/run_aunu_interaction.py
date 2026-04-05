import logging
import argparse
from interactive_manager import create_workflow


<<<<<<<< HEAD:scripts/agents/run_aunu_interaction.py

MIMIC_REQUIRED_STRATEGIES = {"user", "mix"}


========
>>>>>>>> data_synthesis:scripts/agents/run_aunu_interactiion.py
def parse_args():
    parser = argparse.ArgumentParser(description="Args Parser for AUNU simulation")
    parser.add_argument("log_file_path", type=str, help="Path to store the log file")
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
    args = parser.parse_args()

    if args.strategy in MIMIC_REQUIRED_STRATEGIES and args.mimic_model is None:
        parser.error(f"--mimic_model is required when strategy is '{args.strategy}'")

    return args


def run_simulation():
    args = parse_args()

    # --- Logging Configuration ---
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(args.log_file_path),
            logging.StreamHandler()
        ]
    )

    app = create_workflow(args)
    # Initial State
    initial_state = {
        "messages": [{"role": "user_feedback", "content": "I need a sales analysis."}],
        "is_complete": False,
        "aunu_strategy": args.strategy,
        "aunu_model": args.aunu_model,
        "mimic_model": args.mimic_model,
    }

    print("🚀 Starting AU-NU Simulation...")
    for output in app.stream(initial_state):
        # LangGraph streams the updates from each node
        for node_name, state_update in output.items():
            print(f"\n--- Node: {node_name} ---")
            if "messages" in state_update:
                print(f"Message: {state_update['messages'][-1]['content']}")
    
    print("\n✅ Simulation Complete.")

if __name__ == "__main__":
    run_simulation()