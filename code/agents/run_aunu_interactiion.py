import logging
import argparse
from interactive_manager import create_workflow



def parse_args():
    parser = argparse.ArgumentParser(description="Args Parser for AUNU similation")
    parser.add_argument("log_file_path", type=str, help="Path to store the log file")
    parser.add_argument("strategy", type=str, help="AUNU strategy")
    # TODO add other args if necessary
    args = parser.parse_args()
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

    app = create_workflow()
    # Initial State
    initial_state = {
        "messages": [{"role": "user_feedback", "content": "I need a sales analysis."}],
        "is_complete": False,
        "aunu_strategy": "user_interaction",
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