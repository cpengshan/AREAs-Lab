"""Dataset loader for AUNU synthesized outputs."""

import json
import os
from typing import List, Optional

from .schema import PersonaInfo, TaskInstance


def load_dataset(
    synthesized_path: str,
    data_csv_path: str = "",
    dataset_name: Optional[str] = None,
    task_ids: Optional[List[str]] = None,
) -> List[TaskInstance]:
    """Load TaskInstance objects from a synthesized_output.json file.

    Args:
        synthesized_path: Path to synthesized_output.json.
        data_csv_path: Optional path to sampled_data.csv. If omitted, the env
            resolves it automatically from dataset_name at runtime.
        dataset_name: Optional label (defaults to the parent directory name).
        task_ids: If provided, return only tasks whose task_id is in this list.

    Returns:
        Flat list of TaskInstance objects, one per (persona, task) pair.

    Raises:
        FileNotFoundError: If synthesized_path doesn't exist.
        ValueError: If synthesized_output.json is missing required fields.
    """
    if not os.path.exists(synthesized_path):
        raise FileNotFoundError(f"synthesized_output.json not found: {synthesized_path}")

    with open(synthesized_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if dataset_name is None:
        # Infer from path: …/data_synthesized/<org>/<name>/synthesized_output.json
        dataset_name = os.path.basename(os.path.dirname(synthesized_path))

    tasks: List[TaskInstance] = []

    for key, value in data.items():
        # Keys like "user_1", "user_2", … hold persona data.
        if not key.startswith("user_"):
            continue
        persona_num = int(key.split("_")[1])

        user_info = value.get("user_info", {})
        _validate_user_info(user_info, key)

        persona = PersonaInfo(
            persona_id=user_info.get("persona_id", persona_num),
            role=user_info["role"],
            competency_matrix=user_info.get("competency_matrix", {}),
            business_motivation=user_info.get("business_motivation", ""),
            workflow_friction=user_info.get("workflow_friction", ""),
        )

        for task_raw in value.get("tasks_info", []):
            _validate_task_info(task_raw, key)

            task_idx = task_raw.get("task_id", 0)
            # task_idx from JSON is 1-based; we convert to 0-based index in the ID.
            task_id = f"user_{persona_num}_task_{task_idx - 1}"

            instance = TaskInstance(
                task_id=task_id,
                persona_id=persona_num,
                persona_info=persona,
                task_name=task_raw["task_name"],
                task_requirement=task_raw["task_requirement"],
                elevator_pitch=task_raw["elevator_pitch_summary"],
                deep_dive=task_raw.get("deep_dive_summary", ""),
                dataset_name=dataset_name,
                data_csv_path=data_csv_path,
            )
            tasks.append(instance)

    if task_ids is not None:
        tasks = [t for t in tasks if t.task_id in task_ids]

    return tasks


# ---------------------------------------------------------------------------
# Internal validation helpers
# ---------------------------------------------------------------------------

_REQUIRED_USER_FIELDS = {"role"}
_REQUIRED_TASK_FIELDS = {"task_name", "task_requirement", "elevator_pitch_summary"}


def _validate_user_info(user_info: dict, key: str) -> None:
    missing = _REQUIRED_USER_FIELDS - user_info.keys()
    if missing:
        raise ValueError(f"User entry '{key}' is missing fields: {missing}")


def _validate_task_info(task_raw: dict, user_key: str) -> None:
    missing = _REQUIRED_TASK_FIELDS - task_raw.keys()
    if missing:
        raise ValueError(
            f"Task entry under '{user_key}' is missing fields: {missing}. "
            f"Task data: {list(task_raw.keys())}"
        )
