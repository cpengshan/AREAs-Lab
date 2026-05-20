"""Zero-shot variant agents for AREAEnv.

Three agents that augment the baseline zero-shot prompt with grounding signals:

  ZeroShotWithDataAnalysisAgent      — adds dataset schema/structural analysis
  ZeroShotWithSamplesAgent           — adds 3 sampled data rows
  ZeroShotWithDataAnalysisAndSamplesAgent — adds both

All share the same single-node LangGraph topology as ZeroShotAgent:
  [area_agent] → END
"""

import json
import logging
import os
import random
import sys
from datetime import datetime, timezone

from langgraph.graph import StateGraph, END

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_AGENT_DIR = os.path.dirname(__file__)
for _p in (_REPO_ROOT, _AGENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent_state import AgentState, Message
from AREAEnv.area_env.config import AREAEnvConfig
from AREAEnv.area_env.env.actions import finish, inspect_data, VALID_SPLITS
from AREAEnv.area_env.utils.llm import call_llm
from AREAEnv.area_env.utils.jinja_utils import render_template

logger = logging.getLogger(__name__)

_PROMPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "prompts"))
_DATA_SYNTHESIZED_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "AREAEnv", "data", "data_synthesized")
)
_DATA_SAMPLED_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "AREAEnv", "data", "data_sampled")
)

_N_SAMPLES = 5


def _sample_data(
    dataset_name: str,
    split: str = "all",
    n: int = _N_SAMPLES,
    random_state: int = 42,
) -> list[dict]:
    """Sample n instances from data_sampled_2.1.json, truncating each field to 500 chars.

    Args:
        dataset_name: HuggingFace dataset id.
        split: 'defining_instances', 'non_defining_instances', or 'all'.
        n: Number of instances to sample.
        random_state: RNG seed for reproducibility.
    """
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {sorted(VALID_SPLITS)}, got '{split}'")

    json_path = os.path.join(_DATA_SAMPLED_ROOT, dataset_name, "data_sampled_2.1.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"data_sampled_2.1.json not found: {json_path}")

    with open(json_path) as f:
        data = json.load(f)

    if split == "all":
        pool = data.get("defining_instances", []) + data.get("non_defining_instances", [])
    else:
        pool = data.get(split, [])

    if not pool:
        raise ValueError(f"No instances found for split='{split}' in {json_path}")

    rng = random.Random(random_state)
    n = min(n, len(pool))
    sampled = rng.sample(pool, n)
    return [{k: str(v)[:500] for k, v in item.items()} for item in sampled]


def _make_msg(
    start: str,
    end: str,
    action: str,
    prompt: str,
    template: str,
    output: str,
    model: str,
    result: dict,
) -> Message:
    return {
        "start_time": start,
        "end_time": end,
        "role": "area_agent",
        "action": action,
        "input": prompt,
        "prompt_template": template,
        "identified_ambiguity": "",
        "output": output,
        "llm": model,
        "input_tokens": result.get("input_tokens", 0),
        "output_tokens": result.get("output_tokens", 0),
        "cost": result.get("cost", 0.0),
    }


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class _BaseZeroShotVariantAgent:
    """Shared infrastructure for zero-shot variant agents."""

    _TEMPLATE_NAME: str  # subclasses must set this
    _ACTION_NAME: str

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        split: str = "all",
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.split = split
        self._env = None

    @classmethod
    def from_config(cls, config: AREAEnvConfig, split: str = "all"):
        return cls(
            model_name=config.agent_model,
            temperature=config.effective_agent_temperature,
            max_tokens=config.max_tokens,
            split=split,
        )

    @classmethod
    def from_yaml(cls, yaml_path: str):
        return cls.from_config(AREAEnvConfig.from_yaml(yaml_path))

    def _build_template_kwargs(self, user_request: str, dataset_name: str, task) -> dict:
        raise NotImplementedError

    def _fetch_samples(self, n: int = _N_SAMPLES) -> list[dict]:
        """Sample data via the env's inspect_data action (defining_instances split)."""
        _, _, _, info = self._env.step(
            inspect_data(n_samples=n, split="defining_instances")
        )
        return info["data_samples"]

    def process(self, state: AgentState) -> dict:
        obs = self._env.state
        task = obs.task
        user_request = task.elevator_pitch
        dataset_name = task.dataset_name

        template_path = os.path.join(_PROMPTS_DIR, self._TEMPLATE_NAME)
        kwargs = self._build_template_kwargs(user_request, dataset_name, task)

        start = datetime.now(timezone.utc).isoformat()
        prompt = render_template(template_path, **kwargs)
        result = call_llm(
            self.model_name, prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        end = datetime.now(timezone.utc).isoformat()

        requirement = result["output"]
        logger.info(f"[{self.__class__.__name__}] Generated requirement ({len(requirement)} chars)")

        self._env.step(finish(requirement))

        msg = _make_msg(start, end, self._ACTION_NAME, prompt,
                        self._TEMPLATE_NAME, requirement, self.model_name, result)
        return {
            "messages": [msg],
            "is_complete": True,
            "task_requirement_final": requirement,
            "current_turn": 0,
            "zero_shot_draft": requirement,
        }

    def build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("area_agent", self.process)
        workflow.set_entry_point("area_agent")
        workflow.add_edge("area_agent", END)
        return workflow.compile()

    def run(self, env, task) -> dict:
        from AREAEnv.area_env.users import MimicUser
        user = MimicUser(model_name=self.model_name)
        env.reset(task, user)
        self._env = env

        initial_state: AgentState = {
            "messages": [],
            "is_complete": False,
            "task_requirement_final": "",
            "current_turn": 0,
            "zero_shot_draft": "",
        }

        app = self.build_graph()
        for _ in app.stream(initial_state):
            pass

        log = env.get_trajectory_log()
        log["agent_messages"] = [dict(m) for m in initial_state.get("messages", [])]
        return log


# ---------------------------------------------------------------------------
# Variant: Zero-shot + Sampled Data + Modifications (reason)
# ---------------------------------------------------------------------------

class DataInteractionAgent(_BaseZeroShotVariantAgent):
    """Like ZeroShotWithSamplesAgent but uses a prompt that outputs JSON with
    ``final_task_requirement`` and a ``Modifications`` list explaining each change."""

    _TEMPLATE_NAME = "data_interaction.jinja"
    _ACTION_NAME = "data_interaction"

    def __init__(self, *args, initial_requirement: str | None = None, turn_id: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self._initial_requirement = initial_requirement
        self._turn_id = turn_id

    @classmethod
    def from_config(cls, config: AREAEnvConfig, split: str = "all",
                    initial_requirement: str | None = None, turn_id: int = 1):
        return cls(
            model_name=config.agent_model,
            temperature=config.effective_agent_temperature,
            max_tokens=config.max_tokens,
            split=split,
            initial_requirement=initial_requirement,
            turn_id=turn_id,
        )

    def _build_template_kwargs(self, user_request: str, dataset_name: str, task) -> dict:
        data_samples = self._fetch_samples()
        return {"user_instruction": user_request, "data_samples": data_samples, "turn_id": self._turn_id}

    def process(self, state: AgentState) -> dict:
        obs = self._env.state
        task = obs.task
        user_request = self._initial_requirement if self._initial_requirement else task.elevator_pitch
        dataset_name = task.dataset_name

        template_path = os.path.join(_PROMPTS_DIR, self._TEMPLATE_NAME)
        kwargs = self._build_template_kwargs(user_request, dataset_name, task)

        start = datetime.now(timezone.utc).isoformat()
        prompt = render_template(template_path, **kwargs)
        result = call_llm(
            self.model_name, prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        end = datetime.now(timezone.utc).isoformat()

        raw_output = result["output"]

        # Parse JSON output produced by the reason prompt
        modifications = []
        requirement = raw_output
        try:
            # Strip markdown code fences if present
            text = raw_output.strip()
            if text.startswith("```"):
                text = text.split("```", 2)[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.rsplit("```", 1)[0].strip()
            parsed = json.loads(text)
            requirement = parsed.get("final_task_requirement", raw_output)
            modifications = parsed.get("Modifications", [])
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning(
                f"[{self.__class__.__name__}] Could not parse JSON output: {exc}. "
                "Storing raw output as requirement."
            )

        logger.info(
            f"[{self.__class__.__name__}] Generated requirement ({len(requirement)} chars), "
            f"{len(modifications)} modification(s)"
        )

        self._env.step(finish(requirement))
        self._modifications = modifications

        msg = _make_msg(start, end, self._ACTION_NAME, prompt,
                        self._TEMPLATE_NAME, requirement, self.model_name, result)
        return {
            "messages": [msg],
            "is_complete": True,
            "task_requirement_final": requirement,
            "current_turn": 0,
            "zero_shot_draft": requirement,
        }

    def run(self, env, task, initial_requirement: str | None = None) -> dict:
        if initial_requirement is not None:
            self._initial_requirement = initial_requirement
        self._modifications = []
        log = super().run(env, task)
        log["modifications"] = self._modifications
        return log


