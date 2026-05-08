"""Configuration dataclass for AUNUEnv experiments."""

import os
from dataclasses import dataclass, field
from typing import Optional

# Root of AUNUEnv's bundled data directory, resolved relative to this file.
_AUNU_ENV_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_DATA_SYNTHESIZED_ROOT = os.path.join(_AUNU_ENV_ROOT, "data", "data_synthesized")
_DATA_RAW_ROOT = os.path.join(_AUNU_ENV_ROOT, "data", "data_raw")


@dataclass
class AUNUEnvConfig:
    """Experiment configuration for AUNUEnv.

    Attributes:
        agent_model: LLM model hint for external policies.
        agent_model_temperature: Sampling temperature for the agent model.
        user_model: LLM model used by the MIMIC user.
        user_model_temperature: Sampling temperature for the user simulator.
        evaluator_model: LLM model used for atomic unit evaluation.
        evaluator_model_temperature: Sampling temperature for the evaluator.
        max_steps: Maximum environment steps per episode.
        user_mode: "passive" or "persona".
        persona_config: Optional persona attributes dict (required when user_mode="persona").
        dataset_name: Dataset identifier (e.g. "alexfabbri/multi_news"). When set and
            dataset_path / data_csv_path are omitted, paths are resolved automatically
            from the AUNUEnv data directory.
        dataset_path: Explicit path to synthesized_output.json. Inferred from
            dataset_name when blank.
        data_csv_path: Explicit path to sampled_data.csv. Inferred from
            dataset_name when blank.
        task_ids: If provided, run only these task IDs.
        output_dir: Directory for saving results.
        temperature: Fallback sampling temperature (used when per-model temperature
            is not set and as a hint for external policies).
        max_tokens: Default max output tokens.
        verbose: Enable verbose logging.
    """

    # Models
    agent_model: str = "gpt-4.1"
    agent_model_temperature: Optional[float] = None
    user_model: str = "gpt-4.1"
    user_model_temperature: Optional[float] = None
    evaluator_model: str = "gpt-4.1"
    evaluator_model_temperature: Optional[float] = None

    # Environment
    max_steps: int = 15

    # User configuration
    user_mode: str = "passive"
    persona_config: Optional[dict] = None

    # Data
    dataset_name: Optional[str] = None
    dataset_path: str = ""
    data_csv_path: str = ""
    task_ids: Optional[list] = None

    # Output
    output_dir: str = "results"

    # LLM defaults
    temperature: float = 0.0
    max_tokens: int = 4096

    # File names (configurable so experiments can pin to a specific version)
    synthesized_output_file: str = "synthesized_output_2.6.json"
    data_sampled_file: str = "data_sampled_2.6.json"

    # Misc
    verbose: bool = False

    def __post_init__(self):
        valid_user_modes = {"passive", "persona"}
        if self.user_mode not in valid_user_modes:
            raise ValueError(f"user_mode must be one of {valid_user_modes}, got '{self.user_mode}'")
        if self.user_mode == "persona" and self.persona_config is None:
            raise ValueError("persona_config must be set when user_mode='persona'")
        # Auto-resolve dataset paths from dataset_name when not explicitly set.
        if self.dataset_name and not self.dataset_path:
            self.dataset_path = os.path.join(
                _DATA_SYNTHESIZED_ROOT, self.dataset_name, self.synthesized_output_file
            )
        if self.dataset_name and not self.data_csv_path:
            self.data_csv_path = os.path.join(
                _DATA_RAW_ROOT, self.dataset_name, "sampled_data.csv"
            )

    @property
    def effective_agent_temperature(self) -> float:
        return self.agent_model_temperature if self.agent_model_temperature is not None else self.temperature

    @property
    def effective_user_temperature(self) -> float:
        return self.user_model_temperature if self.user_model_temperature is not None else self.temperature

    @property
    def effective_evaluator_temperature(self) -> float:
        return self.evaluator_model_temperature if self.evaluator_model_temperature is not None else self.temperature

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AUNUEnvConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_yaml(cls, path: str) -> "AUNUEnvConfig":
        """Load config from a YAML file, resolving relative paths against the config file's directory."""
        import yaml
        path = os.path.abspath(path)
        config_dir = os.path.dirname(path)
        with open(path, "r") as f:
            d = yaml.safe_load(f)
        # Resolve explicit path fields relative to the config file's location.
        for field_name in ("dataset_path", "data_csv_path", "output_dir"):
            if field_name in d and d[field_name] and not os.path.isabs(d[field_name]):
                d[field_name] = os.path.normpath(os.path.join(config_dir, d[field_name]))
        return cls.from_dict(d)
