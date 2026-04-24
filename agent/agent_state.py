"""LangGraph state definition for AUNUEnv-based agents.

Extends agent_basic.InteractionState to include env-level tracking fields
that map to AUNUEnv's trajectory log format.
"""

from typing import TypedDict, Annotated, Sequence
import operator


class Message(TypedDict):
    """A single logged message in the agent trajectory."""
    start_time: str
    end_time: str
    role: str              # "aunu_agent" | "mimic_user"
    action: str            # "zero_shot" | "ask_user" | "propose_update" | "finish"
    input: str             # rendered prompt
    prompt_template: str
    identified_ambiguity: str
    output: str            # raw LLM output or env response
    llm: str
    input_tokens: int
    output_tokens: int
    cost: float


class AgentState(TypedDict, total=False):
    """LangGraph state shared across all agent nodes in an episode."""
    messages: Annotated[Sequence[Message], operator.add]
    is_complete: bool
    task_requirement_final: str
    current_turn: int          # current clarification / refinement turn
    zero_shot_draft: str       # zero-shot draft cached for use in prediction
    previous_reflections: list  # accumulated reflect outputs (data_interaction only)
