"""Zero-shot agent for AREAEnv.

Generates a final task requirement from the elevator pitch alone — no
interaction with the MIMIC user and no data inspection.

LangGraph graph:
  [area_agent] → END

The single node renders the zero_shot.jinja prompt, calls the LLM, and
submits the result to the environment via env.step(finish(...)).
"""

import logging
import os
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
from AREAEnv.area_env.env.actions import finish
from AREAEnv.area_env.utils.llm import call_llm
from AREAEnv.area_env.utils.jinja_utils import render_template

logger = logging.getLogger(__name__)

_PROMPTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "prompts")
)
_ZERO_SHOT_TEMPLATE = os.path.join(_PROMPTS_DIR, "zero_shot.jinja")


class ZeroShotAgent:
    """LangGraph-based zero-shot agent that operates inside AREAEnv.

    The agent does a single LLM call on the elevator pitch and submits the
    result as the final task requirement.

    Args:
        model_name: LLM model identifier (must be in AREAEnv's PRICING_DATA).
        temperature: Sampling temperature.
        max_tokens: Maximum output tokens.
    """

    def __init__(self, model_name: str, temperature: float = 0.0, max_tokens: int = 4096):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._env = None   # set before run()

    @classmethod
    def from_config(cls, config: AREAEnvConfig) -> "ZeroShotAgent":
        """Construct a ZeroShotAgent from an AREAEnvConfig."""
        return cls(
            model_name=config.agent_model,
            temperature=config.effective_agent_temperature,
            max_tokens=config.max_tokens,
        )

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ZeroShotAgent":
        """Construct a ZeroShotAgent by loading an AREAEnvConfig YAML file."""
        return cls.from_config(AREAEnvConfig.from_yaml(yaml_path))

    # ------------------------------------------------------------------
    # LangGraph node
    # ------------------------------------------------------------------

    def process(self, state: AgentState) -> dict:
        """LangGraph node: generate requirement and submit to env."""
        obs = self._env.state
        user_request = obs.task.elevator_pitch

        start_time = datetime.now(timezone.utc).isoformat()
        prompt = render_template(_ZERO_SHOT_TEMPLATE, user_instruction=user_request)
        result = call_llm(
            self.model_name,
            prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        end_time = datetime.now(timezone.utc).isoformat()

        requirement = result["output"]
        logger.info(f"[ZeroShot] Generated requirement ({len(requirement)} chars)")

        # Submit to environment
        self._env.step(finish(requirement))

        msg: Message = {
            "start_time": start_time,
            "end_time": end_time,
            "role": "area_agent",
            "action": "zero_shot",
            "input": prompt,
            "prompt_template": "zero_shot.jinja",
            "identified_ambiguity": "",
            "output": requirement,
            "llm": self.model_name,
            "input_tokens": result.get("input_tokens", 0),
            "output_tokens": result.get("output_tokens", 0),
            "cost": result.get("cost", 0.0),
        }
        return {
            "messages": [msg],
            "is_complete": True,
            "task_requirement_final": requirement,
            "current_turn": 0,
            "zero_shot_draft": requirement,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_graph(self):
        """Compile and return the LangGraph StateGraph."""
        workflow = StateGraph(AgentState)
        workflow.add_node("area_agent", self.process)
        workflow.set_entry_point("area_agent")
        workflow.add_edge("area_agent", END)
        return workflow.compile()

    def run(self, env, task) -> dict:
        """Run a complete zero-shot episode and return the trajectory log.

        Args:
            env: AREAEnv instance (already initialised with evaluator).
            task: TaskInstance to solve.

        Returns:
            Trajectory log dict from env.get_trajectory_log().
        """
        # Reset env with a passive MimicUser (unused in zero-shot but required by API)
        from AREAEnv.area_env.users import MimicUser
        user = MimicUser(model_name=self.model_name)
        obs, _ = env.reset(task, user)
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
