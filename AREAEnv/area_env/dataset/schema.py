"""Data schema for AREA benchmark tasks."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PersonaInfo:
    """Metadata about the synthesized user persona."""
    persona_id: int
    role: str
    competency_matrix: dict
    business_motivation: str
    workflow_friction: str = ""


@dataclass
class TaskInstance:
    """A single benchmark task drawn from a synthesized persona+task pair.

    Attributes:
        task_id: Unique string identifier, e.g. "user_1_task_0".
        persona_id: Which persona (user_N key in synthesized_output.json).
        persona_info: Full persona metadata.
        task_name: Human-readable task name.
        task_requirement: Ground-truth task requirement (used only for evaluation).
        elevator_pitch: Brief, underspecified user request shown to the agent.
        deep_dive: Detailed description (may be used by evaluator or policy).
        dataset_name: Dataset identifier, e.g. "alexfabbri/multi_news".
        data_csv_path: Path to the corresponding sampled_data.csv file.
    """
    task_id: str
    persona_id: int
    persona_info: PersonaInfo
    task_name: str
    task_requirement: str          # ground truth — not shown to agent
    elevator_pitch: str            # underspecified user request shown to agent
    deep_dive: str
    dataset_name: str
    data_csv_path: str = ""
