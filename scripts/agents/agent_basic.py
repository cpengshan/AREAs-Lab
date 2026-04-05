from typing import TypedDict, Annotated, Sequence
import operator


class Message(TypedDict):
    start_time: float   # Unix timestamp when input was sent to LLM
    end_time: float     # Unix timestamp when output was received from LLM
    role: str           # "aunu_agent" | "mimic_user"
    action: str         # method name, e.g. "user_interaction", "data_interaction", "revise_task_requirement", "respond"
    input: str          # rendered prompt sent to LLM
    output: str         # raw output from LLM
    llm: str            # model name, e.g. "claude-sonnet-4-6"


class InteractionState(TypedDict):
<<<<<<< HEAD:code/agents/agent_basic.py
    messages: Annotated[Sequence[Message], operator.add]  # All messages exchanged between aunu_agent and mimic_user
    is_complete: bool                                      # Whether the interaction is complete
=======
    messages: Annotated[Sequence[dict], operator.add]   # This variable stores all the messages, including questions from aunu agent, experiment results from aunu agent and feedback from mimic user and 
    # Each message should contain the following keys:
    # role: aunu_agent / mimic_user
    # action: The name of the action 
    # input: Input to LLM 
    # output: output of LLM
    # llm: the name of the LLM 
    # Logging already record time so time is no longer necessary
    is_complete: bool   # This variables records if the interaction is completed
>>>>>>> data_synthesis:scripts/agents/agent_basic.py
