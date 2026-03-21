from langgraph.graph import StateGraph, END
from agent_basic import InteractionState
from aunu_agent import AUNUAgent
from mimic_user import MimicUser

def create_workflow():
    # Initialize agents
    aunu = AUNUAgent()
    user = MimicUser()

    # Define the graph
    workflow = StateGraph(InteractionState)

    # Add nodes
    workflow.add_node("aunu_agent", aunu.process)
    workflow.add_node("mimic_user", user.respond)

    # Entry point
    workflow.set_entry_point("mimic_user")

    # Routing logic
    def router(state: InteractionState):
        if state["is_satisfied"] or state["iteration_count"] > 5:
            return "end"
        return "continue"

    # Add edges
    workflow.add_conditional_edges(
        "aunu_agent",
        router,
        {"continue": "mimic_user", "end": END}
    )
    workflow.add_edge("mimic_user", "aunu_agent")

    return workflow.compile()