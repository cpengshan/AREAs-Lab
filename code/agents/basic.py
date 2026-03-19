from typing import TypedDict, List, Annotated, Optional
import operator

# The shared state across the entire LangGraph workflow
class InteractionState(TypedDict):
    # 'operator.add' allows LangGraph to append to the list automatically
    messages: Annotated[List[dict], operator.add] 
    specification: str
    is_satisfied: bool
    strategy: str  # e.g., "zero_shot", "data_driven", "active_elicitation"
    ground_truth: str  # The hidden target requirements
    iteration_count: int