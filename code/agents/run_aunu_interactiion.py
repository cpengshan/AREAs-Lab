from interactive_manager import create_workflow

def run_simulation():
    app = create_workflow()
    
    # Initial State
    initial_state = {
        "messages": [{"role": "user_feedback", "content": "I need a sales analysis."}],
        "specification": "",
        "is_satisfied": False,
        "strategy": "active_elicitation",
        "ground_truth": "I want to see Q4 sales for the electronics category.",
        "iteration_count": 0
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