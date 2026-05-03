"""Action definitions for AUNUEnv.

Actions are represented as plain Python dicts for flexibility and easy JSON
serialization.  Each action dict must have a "type" key.

Supported action types:
  ask_user                  — send a question to the MIMIC user
  inspect_data              — sample rows from the task dataset for analysis
  propose_requirement_update — update the current draft task requirement
  finish                    — submit the final requirement and end the episode
"""

from typing import Any


# ---------------------------------------------------------------------------
# Action type constants
# ---------------------------------------------------------------------------

ACTION_ASK_USER = "ask_user"
ACTION_INSPECT_DATA = "inspect_data"
ACTION_PROPOSE_UPDATE = "propose_requirement_update"
ACTION_FINISH = "finish"

VALID_ACTION_TYPES = {
    ACTION_ASK_USER,
    ACTION_INSPECT_DATA,
    ACTION_PROPOSE_UPDATE,
    ACTION_FINISH,
}


# ---------------------------------------------------------------------------
# Constructor helpers
# ---------------------------------------------------------------------------

def ask_user(question: str) -> dict:
    """Create an ask_user action.

    Args:
        question: The question to pose to the MIMIC user.
    """
    return {"type": ACTION_ASK_USER, "question": question}


VALID_SPLITS = {"defining_instances", "non_defining_instances", "all"}


def inspect_data(n_samples: int = 3, query: str = "", split: str = "all") -> dict:
    """Create an inspect_data action.

    Args:
        n_samples: Number of data instances to sample.
        query: Optional description of what to look for (informational only).
        split: Which subset to sample from — 'defining_instances',
            'non_defining_instances', or 'all'.
    """
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {sorted(VALID_SPLITS)}, got '{split}'")
    return {"type": ACTION_INSPECT_DATA, "n_samples": n_samples, "query": query, "split": split}


def propose_requirement_update(updated_requirement: str) -> dict:
    """Create a propose_requirement_update action.

    Args:
        updated_requirement: The revised draft task requirement.
    """
    return {"type": ACTION_PROPOSE_UPDATE, "updated_requirement": updated_requirement}


def finish(final_requirement: str) -> dict:
    """Create a finish action, ending the episode.

    Args:
        final_requirement: The agent's final task requirement submission.
    """
    return {"type": ACTION_FINISH, "final_requirement": final_requirement}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_action(action: Any) -> None:
    """Raise ValueError if the action dict is malformed.

    Args:
        action: The action to validate.

    Raises:
        ValueError: If the action is invalid.
    """
    if not isinstance(action, dict):
        raise ValueError(f"Action must be a dict, got {type(action)}")
    action_type = action.get("type")
    if action_type not in VALID_ACTION_TYPES:
        raise ValueError(
            f"Invalid action type '{action_type}'. "
            f"Must be one of: {sorted(VALID_ACTION_TYPES)}"
        )
    if action_type == ACTION_ASK_USER and "question" not in action:
        raise ValueError("ask_user action requires a 'question' field")
    if action_type == ACTION_INSPECT_DATA:
        split = action.get("split", "all")
        if split not in VALID_SPLITS:
            raise ValueError(f"inspect_data 'split' must be one of {sorted(VALID_SPLITS)}, got '{split}'")
    if action_type == ACTION_PROPOSE_UPDATE and "updated_requirement" not in action:
        raise ValueError("propose_requirement_update action requires 'updated_requirement'")
    if action_type == ACTION_FINISH and "final_requirement" not in action:
        raise ValueError("finish action requires 'final_requirement'")
