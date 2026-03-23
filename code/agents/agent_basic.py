from typing import TypedDict, List, Annotated, Sequence
import operator

class InteractionState(TypedDict):
    messages: Annotated[Sequence[dict], operator.add]   # This variable stores all the messages, including questions from aunu agent, experiment results from aunu agent and feedback from mimic user and 
    # Each message should contain the following keys:
    # start_time: Time the message is received
    # end_time: Time the message is sent / recorded
    # role: aunu_agent / mimic_user
    # action: The name of the action 
    # input: Input to LLM 
    # output: output of LLM
    # llm: the name of the LLM 
    is_complete: bool   # This variables records if the interaction is completed