from langgraph.graph import StateGraph, END
from agent_basic import InteractionState
from aunu_agent import AUNUAgent
from mimic_user import MimicUser

MIMIC_STRATEGIES = {"user", "mix"}


def create_workflow(args):
    strategy = args.strategy_aunu
    aunu = AUNUAgent(args)

    workflow = StateGraph(InteractionState)
    workflow.add_node("aunu_agent", aunu.process)

    def router(state: InteractionState):
        return "end" if state["is_complete"] else "continue"

    if strategy in MIMIC_STRATEGIES:
        # aunu_agent <-> mimic_user loop
        mimic = MimicUser(args)
        workflow.add_node("mimic_user", mimic.respond)

        workflow.set_entry_point("mimic_user")
        workflow.add_edge("mimic_user", "aunu_agent")
        workflow.add_conditional_edges(
            "aunu_agent",
            router,
            {"continue": "mimic_user", "end": END},
        )
    else:
        # zero_shot / data: aunu_agent runs once then ends
        workflow.set_entry_point("aunu_agent")
        workflow.add_conditional_edges(
            "aunu_agent",
            router,
            {"continue": "aunu_agent", "end": END},
        )

    return workflow.compile()