"""Zero-shot variant agents for AUNUEnv.

Three agents that augment the baseline zero-shot prompt with grounding signals:

  ZeroShotWithDataAnalysisAgent      — adds dataset schema/structural analysis
  ZeroShotWithSamplesAgent           — adds 3 sampled data rows
  ZeroShotWithDataAnalysisAndSamplesAgent — adds both

All share the same single-node LangGraph topology as ZeroShotAgent:
  [aunu_agent] → END
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone

import pandas as pd
from langgraph.graph import StateGraph, END

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_AGENT_DIR = os.path.dirname(__file__)
for _p in (_REPO_ROOT, _AGENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent_state import AgentState, Message
from AUNUEnv.aunu_env.config import AUNUEnvConfig
from AUNUEnv.aunu_env.env.actions import finish
from AUNUEnv.aunu_env.utils.llm import call_llm
from AUNUEnv.aunu_env.utils.jinja_utils import render_template

logger = logging.getLogger(__name__)

_PROMPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "prompts"))
_DATA_SYNTHESIZED_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "AUNUEnv", "data", "data_synthesized")
)
_DATA_RAW_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "AUNUEnv", "data", "data_raw")
)

_N_SAMPLES = 3


def _load_data_analysis(dataset_name: str) -> dict:
    """Load data_analysis.json for the given dataset."""
    path = os.path.join(_DATA_SYNTHESIZED_ROOT, dataset_name, "data_analysis.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"data_analysis.json not found: {path}"
        )
    with open(path) as f:
        return json.load(f)


def _sample_data(dataset_name: str, n: int = _N_SAMPLES, random_state: int = 42) -> list[dict]:
    """Sample n rows from the dataset CSV, truncating each field to 500 chars."""
    csv_path = os.path.join(_DATA_RAW_ROOT, dataset_name, "sampled_data.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"sampled_data.csv not found: {csv_path}"
        )
    df = pd.read_csv(csv_path)
    n = min(n, len(df))
    sampled = df.sample(n=n, random_state=random_state)
    return [
        {col: str(row[col])[:500] for col in df.columns}
        for _, row in sampled.iterrows()
    ]


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
        "role": "aunu_agent",
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

    def __init__(self, model_name: str, temperature: float = 0.0, max_tokens: int = 4096):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._env = None

    @classmethod
    def from_config(cls, config: AUNUEnvConfig):
        return cls(
            model_name=config.agent_model,
            temperature=config.effective_agent_temperature,
            max_tokens=config.max_tokens,
        )

    @classmethod
    def from_yaml(cls, yaml_path: str):
        return cls.from_config(AUNUEnvConfig.from_yaml(yaml_path))

    def _build_template_kwargs(self, user_request: str, dataset_name: str) -> dict:
        raise NotImplementedError

    def process(self, state: AgentState) -> dict:
        obs = self._env.state
        user_request = obs.task.elevator_pitch
        dataset_name = obs.task.dataset_name

        template_path = os.path.join(_PROMPTS_DIR, self._TEMPLATE_NAME)
        kwargs = self._build_template_kwargs(user_request, dataset_name)

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
        workflow.add_node("aunu_agent", self.process)
        workflow.set_entry_point("aunu_agent")
        workflow.add_edge("aunu_agent", END)
        return workflow.compile()

    def run(self, env, task) -> dict:
        from AUNUEnv.aunu_env.users import MimicUser
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
# Variant 1: Zero-shot + Data Analysis
# ---------------------------------------------------------------------------

class ZeroShotWithDataAnalysisAgent(_BaseZeroShotVariantAgent):
    """Augments the zero-shot prompt with dataset schema/structural analysis."""

    _TEMPLATE_NAME = "zero_shot_with_data_analysis.jinja"
    _ACTION_NAME = "zero_shot_with_data_analysis"

    def _build_template_kwargs(self, user_request: str, dataset_name: str) -> dict:
        data_analysis = _load_data_analysis(dataset_name)
        return {"user_instruction": user_request, "data_analysis": data_analysis}


# ---------------------------------------------------------------------------
# Variant 2: Zero-shot + Sampled Data
# ---------------------------------------------------------------------------

class ZeroShotWithSamplesAgent(_BaseZeroShotVariantAgent):
    """Augments the zero-shot prompt with 3 sampled data rows."""

    _TEMPLATE_NAME = "zero_shot_with_samples.jinja"
    _ACTION_NAME = "zero_shot_with_samples"

    def _build_template_kwargs(self, user_request: str, dataset_name: str) -> dict:
        data_samples = _sample_data(dataset_name)
        return {"user_instruction": user_request, "data_samples": data_samples}


# ---------------------------------------------------------------------------
# Variant 3: Zero-shot + Data Analysis + Sampled Data
# ---------------------------------------------------------------------------

class ZeroShotWithDataAnalysisAndSamplesAgent(_BaseZeroShotVariantAgent):
    """Augments the zero-shot prompt with both schema analysis and data samples."""

    _TEMPLATE_NAME = "zero_shot_with_data_analysis_and_samples.jinja"
    _ACTION_NAME = "zero_shot_with_data_analysis_and_samples"

    def _build_template_kwargs(self, user_request: str, dataset_name: str) -> dict:
        data_analysis = _load_data_analysis(dataset_name)
        data_samples = _sample_data(dataset_name)
        return {
            "user_instruction": user_request,
            "data_analysis": data_analysis,
            "data_samples": data_samples,
        }
