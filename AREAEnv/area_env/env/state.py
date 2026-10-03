"""Episode state for AREAEnv."""

from dataclasses import dataclass, field
from typing import Optional

from ..dataset.schema import TaskInstance


@dataclass
class EpisodeState:
    """Tracks the state of a single AREAEnv episode.

    Attributes:
        task: The TaskInstance being solved.
        draft_requirement: The agent's current draft task requirement.
        interaction_history: List of user-agent exchange dicts.
        data_inspections: List of data inspection result dicts.
        step_count: Number of steps taken so far.
        is_done: Whether the episode has ended.
        trajectory: Full ordered log of every step taken.
        total_cost: Accumulated LLM cost across all steps.
    """
    task: TaskInstance
    draft_requirement: str = ""
    interaction_history: list = field(default_factory=list)
    data_inspections: list = field(default_factory=list)
    step_count: int = 0
    is_done: bool = False
    trajectory: list = field(default_factory=list)
    total_cost: float = 0.0
