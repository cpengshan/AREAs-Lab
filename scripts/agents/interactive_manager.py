import os
from langgraph.graph import StateGraph, END
from agent_basic import InteractionState
from aunu_agent import AUNUAgent
from mimic_user import MimicUser

MIMIC_STRATEGIES = {"user", "hybrid"}

DEFAULT_TEMPLATE_PATH = os.path.join(
    os.path.dirname(__file__), "../../prompts/agents/mimic_user/feedback_mimic_user.jinja"
)


def create_workflow(args):
    strategy = args.strategy_aunu
    aunu = AUNUAgent(
        strategy=args.strategy_aunu,
        model=args.aunu_model,
        user_instruction_init=args.user_instruction_init,
        dataset=args.dataset,
        max_turns=getattr(args, "max_turns", 5),
        persona=getattr(args, "persona_profile", None),
        user_interaction_template=getattr(args, "aunu_template", "user_interaction.jinja"),
        zero_shot_seed=getattr(args, "zero_shot_seed", None),
    )

    workflow = StateGraph(InteractionState)
    workflow.add_node("aunu_agent", aunu.process)

    def router(state: InteractionState):
        return "end" if state["is_complete"] else "continue"

    if strategy in MIMIC_STRATEGIES:
        # Flow: aunu_agent receives user_instruction_init from initial state,
        # asks clarifying questions, mimic_user responds, repeat until complete.
        mimic = MimicUser(
            persona=args.persona_profile,
            task_requirement_gold=args.task_requirement_gold,
            user_instruction_init=args.user_instruction_init,
            communication_style_init=getattr(args, "communication_style_init", ""),
            model=args.mimic_model,
            template_path=getattr(args, "mimic_template_path",
                                  os.path.join(os.path.dirname(__file__), "../../prompts/agents/mimic_user",
                                               getattr(args, "mimic_template", "feedback_mimic_user_v2.jinja"))),
        )
        workflow.add_node("mimic_user", mimic.respond)

        # mimic_user is entry: sends user_instruction_init first,
        # then aunu_agent asks clarifying questions, mimic_user responds, repeat.
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