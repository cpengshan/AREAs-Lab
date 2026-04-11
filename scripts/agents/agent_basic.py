from typing import TypedDict, Annotated, Sequence
import operator


class Message(TypedDict):
    start_time: str     # ISO 8601 datetime when input was sent to LLM
    end_time: str       # ISO 8601 datetime when output was received from LLM
    role: str           # "aunu_agent" | "mimic_user"
    action: str         # method name, e.g. "user_interaction", "data_interaction", "revise_task_requirement", "respond"
    input: str          # rendered prompt sent to LLM
    output: str         # raw output from LLM
    llm: str            # model name, e.g. "claude-sonnet-4-6"
    input_tokens: int   # number of input/prompt tokens consumed
    output_tokens: int  # number of output/completion tokens generated
    cost: float         # estimated cost in USD for this LLM call


class InteractionState(TypedDict):
    messages: Annotated[Sequence[Message], operator.add]  # All messages exchanged between aunu_agent and mimic_user
    is_complete: bool                                      # Whether the interaction is complete
    task_requirement_final: str                            # Final task requirement produced by the agent
