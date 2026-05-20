"""Dataset loader and registry for AREA synthesized outputs."""

import json
import os
from typing import Dict, List, Optional, Tuple

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


def list_personas(dataset_name: str, data_synthesized_dir: str) -> List[int]:
    """Return sorted persona IDs for a dataset from its synthesized_output.json.

    Args:
        dataset_name: HuggingFace-style dataset id, e.g. 'alexfabbri/multi_news'.
        data_synthesized_dir: Absolute path to the data_synthesized root directory.

    Returns:
        Sorted list of integer persona IDs.
    """
    path = os.path.join(data_synthesized_dir, dataset_name, "synthesized_output.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"synthesized_output.json not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return sorted(int(k.split("_")[1]) for k in data if k.startswith("user_"))


def _normalize_persona_id(persona_id) -> Optional[int]:
    """Normalize persona_id to int. Accepts int, numeric str, 'user_N', 'persona_N'."""
    if persona_id is None:
        return None
    if isinstance(persona_id, int):
        return persona_id
    s = str(persona_id)
    for prefix in ("user_", "persona_"):
        if s.startswith(prefix):
            try:
                return int(s[len(prefix):])
            except ValueError:
                pass
    try:
        return int(s)
    except ValueError:
        return None  # cannot normalize; caller should skip persona_id check


class DatasetRegistry:
    """Manages loading and caching of AREA synthesized datasets.

    The registry owns all dataset I/O; agents interact with the environment
    using identifiers only (dataset_name, persona_id, task_id).

    Args:
        data_synthesized_root: Absolute path to the data_synthesized/ directory.
        synthesized_output_file: Filename of the synthesized JSON inside each
            dataset sub-directory (default: "synthesized_output.json").
    """

    def __init__(
        self,
        data_synthesized_root: str,
        synthesized_output_file: str = "synthesized_output.json",
    ):
        self._root = os.path.abspath(data_synthesized_root)
        self._output_file = synthesized_output_file
        self._cache: Dict[str, List[TaskInstance]] = {}

    def load(self, dataset_name: str) -> List[TaskInstance]:
        """Load (and cache) all TaskInstances for a dataset."""
        if dataset_name not in self._cache:
            path = os.path.join(self._root, dataset_name, self._output_file)
            self._cache[dataset_name] = load_dataset(path, dataset_name=dataset_name)
        return self._cache[dataset_name]

    def list_tasks(self, dataset_name: str) -> List[Tuple[str, int, str]]:
        """Return (dataset_name, persona_id, task_id) for every task in the dataset."""
        return [(t.dataset_name, t.persona_id, t.task_id) for t in self.load(dataset_name)]

    def get_task(self, dataset_name: str, persona_id, task_id: str) -> TaskInstance:
        """Retrieve a single TaskInstance by its three identifiers.

        Args:
            dataset_name: Dataset identifier, e.g. "alexfabbri/multi_news".
            persona_id: Persona identifier — int, numeric string, "user_N", or "persona_N".
            task_id: Exact task_id string, e.g. "user_1_task_0".

        Raises:
            KeyError: If no matching task is found.
        """
        tasks = self.load(dataset_name)
        norm_pid = _normalize_persona_id(persona_id)
        for t in tasks:
            pid_match = (norm_pid is None) or (t.persona_id == norm_pid)
            if pid_match and t.task_id == task_id:
                return t
        raise KeyError(
            f"Task not found: dataset={dataset_name!r}, "
            f"persona_id={persona_id!r}, task_id={task_id!r}"
        )


if __name__ == "__main__":
    import sys as _sys
    # Usage: python -m AREAEnv.area_env.dataset.loader <dataset_name> <data_synthesized_dir>
    if len(_sys.argv) != 3:
        print("Usage: python -m AREAEnv.area_env.dataset.loader <dataset_name> <data_synthesized_dir>",
              file=_sys.stderr)
        _sys.exit(1)
    ids = list_personas(_sys.argv[1], _sys.argv[2])
    print(" ".join(str(i) for i in ids))
