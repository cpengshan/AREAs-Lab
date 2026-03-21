from typing import TypedDict, List, Annotated, Sequence
import operator

class InteractionState(TypedDict):
    messages: Annotated[Sequence[dict], operator.add]   # This variable stores all the messages, including questions from aunu agent, experiment results from aunu agent and feedback from mimic user and 
    is_complete: bool   # This variables records if the interaction is completed