"""Data-interaction agent for AUNUEnv.

Iteratively refines a task requirement by stress-testing it against real data
samples — no interaction with the MIMIC user.

LangGraph graph:
  [aunu_agent] ──(continue)──▶ [aunu_agent]
               ──(end)──▶ END

The single self-looping node manages the full state machine:
  Phase 0: zero-shot draft
  Phase 1: data loop (up to max_turns)
             → inspect_data
             → execute requirement on each sample (LLM)
             → reflect on outputs
             → rewrite requirement
  Phase 2: finish with final requirement
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
from AUNUEnv.aunu_env.config import AUNUEnvConfig
from AUNUEnv.aunu_env.env.actions import inspect_data, finish
from AUNUEnv.aunu_env.utils.llm import call_llm
from AUNUEnv.aunu_env.utils.jinja_utils import render_template
from AUNUEnv.aunu_env.utils.json_utils import parse_json_output

logger = logging.getLogger(__name__)

_PROMPTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "prompts/agents/aunu_agent")
)
_ZERO_SHOT_TEMPLATE  = os.path.join(_PROMPTS_DIR, "zero_shot.jinja")
_EXECUTE_TEMPLATE    = os.path.join(_PROMPTS_DIR, "data/data_interaction_execute.jinja")
_REFLECT_TEMPLATE    = os.path.join(_PROMPTS_DIR, "data/data_interaction_reflect.jinja")
_REWRITE_TEMPLATE    = os.path.join(_PROMPTS_DIR, "data/data_interaction_rewrite.jinja")

# Number of dataset rows to sample per iteration
_N_SAMPLES = 3


class DataInteractionAgent:
    """LangGraph-based data-interaction agent that operates inside AUNUEnv.

    The agent produces a zero-shot draft, then runs up to `max_turns`
    refinement rounds. Each round samples data rows from the env, executes
    the current requirement on each sample, reflects on the outputs, and
    rewrites the requirement. Finally it submits the refined requirement.

    Args:
        model_name: LLM model identifier.
        max_turns: Maximum data-refinement rounds before final submission.
        temperature: Sampling temperature.
        max_tokens: Maximum output tokens for requirement generation.
        execute_max_tokens: Maximum output tokens for per-sample execution.
        reflect_max_tokens: Maximum output tokens for the reflection step.
    """

    def __init__(
        self,
        model_name: str,
        max_turns: int = 3,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        execute_max_tokens: int = 1024,
        reflect_max_tokens: int = 1024,
    ):
        self.model_name = model_name
        self.max_turns = max_turns
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.execute_max_tokens = execute_max_tokens
        self.reflect_max_tokens = reflect_max_tokens
        self._env = None  # set before run()

    @classmethod
    def from_config(cls, config: AUNUEnvConfig, max_turns: int = 3) -> "DataInteractionAgent":
        """Construct a DataInteractionAgent from an AUNUEnvConfig."""
        return cls(
            model_name=config.agent_model,
            max_turns=max_turns,
            temperature=config.effective_agent_temperature,
            max_tokens=config.max_tokens,
        )

    @classmethod
    def from_yaml(cls, yaml_path: str, max_turns: int = 3) -> "DataInteractionAgent":
        """Construct a DataInteractionAgent by loading an AUNUEnvConfig YAML file."""
        return cls.from_config(AUNUEnvConfig.from_yaml(yaml_path), max_turns=max_turns)

    # ------------------------------------------------------------------
    # LangGraph node
    # ------------------------------------------------------------------

    def process(self, state: AgentState) -> dict:
        """LangGraph node: manage one step of the agent state machine."""
        env = self._env
        env_state = env.state
        turn = state["current_turn"]
        user_request = env_state.task.elevator_pitch
        messages: list[Message] = []

        # ── Phase 0: generate zero-shot draft ──────────────────────────
        if turn == 0:
            start = datetime.now(timezone.utc).isoformat()
            prompt = render_template(_ZERO_SHOT_TEMPLATE, user_instruction=user_request)
            result = call_llm(self.model_name, prompt,
                              max_tokens=self.max_tokens, temperature=self.temperature)
            end = datetime.now(timezone.utc).isoformat()
            draft = result["output"]
            logger.info(f"[DataInteraction] Zero-shot draft ({len(draft)} chars)")

            messages.append(_make_msg(
                start, end, "aunu_agent", "zero_shot_draft",
                prompt, "zero_shot.jinja", "", draft, self.model_name, result,
            ))
            return {
                "messages": messages,
                "is_complete": False,
                "current_turn": 1,
                "zero_shot_draft": draft,
                "task_requirement_final": draft,
            }

        # ── Phase 1: data refinement turn ──────────────────────────────
        current_req = state["task_requirement_final"]

        # Step 1a: sample data from the env
        obs, _, _, info = env.step(inspect_data(n_samples=_N_SAMPLES))
        raw_samples = info.get("data_samples", [])

        # Determine which column is the primary input (env detects this)
        input_col = env_state.data_inspections[-1].get("input_col") if env_state.data_inspections else None

        # Step 1b: execute current requirement on each sample
        sample_data = []
        execute_cost = 0.0
        for sample in raw_samples:
            input_text = sample.get(input_col, str(sample)) if input_col else str(sample)
            start = datetime.now(timezone.utc).isoformat()
            exec_prompt = render_template(
                _EXECUTE_TEMPLATE,
                task_requirement=current_req,
                input=input_text,
            )
            exec_result = call_llm(self.model_name, exec_prompt,
                                   max_tokens=self.execute_max_tokens,
                                   temperature=self.temperature)
            end = datetime.now(timezone.utc).isoformat()
            predicted_output = exec_result["output"]
            execute_cost += exec_result.get("cost", 0.0)

            sample_data.append({
                "input": input_text,
                "predicted_output": predicted_output,
                "ground_truth_notes": "",
            })
            messages.append(_make_msg(
                start, end, "aunu_agent", "execute",
                exec_prompt, "data/data_interaction_execute.jinja",
                "", predicted_output, self.model_name, exec_result,
            ))

        logger.info(f"[DataInteraction] Turn {turn}: executed on {len(sample_data)} samples")

        # Step 1c: reflect — identify ambiguities in the current requirement
        previous_reflections = state.get("previous_reflections", [])

        start = datetime.now(timezone.utc).isoformat()
        reflect_prompt = render_template(
            _REFLECT_TEMPLATE,
            current_task_requirement=current_req,
            sample_data=sample_data,
            previous_reflections=previous_reflections if previous_reflections else None,
        )
        reflect_result = call_llm(self.model_name, reflect_prompt,
                                  max_tokens=self.reflect_max_tokens,
                                  temperature=self.temperature)
        end = datetime.now(timezone.utc).isoformat()
        reflection = parse_json_output(reflect_result["output"])
        output_evaluation = reflection.get("output_evaluation", reflect_result["output"])
        potential_ambiguity = reflection.get("potential_ambiguity", "")

        messages.append(_make_msg(
            start, end, "aunu_agent", "reflect",
            reflect_prompt, "data/data_interaction_reflect.jinja",
            potential_ambiguity, reflect_result["output"], self.model_name, reflect_result,
        ))
        logger.info(f"[DataInteraction] Turn {turn}: reflected")

        next_turn = turn + 1
        is_last = next_turn > self.max_turns

        # Step 1d: rewrite requirement based on reflection
        updated_reflections = list(previous_reflections) + [{
            "output_evaluation": output_evaluation,
            "potential_ambiguity": potential_ambiguity,
        }]

        start = datetime.now(timezone.utc).isoformat()
        rewrite_prompt = render_template(
            _REWRITE_TEMPLATE,
            current_task_requirement=current_req,
            output_evaluation=output_evaluation,
            potential_ambiguity=potential_ambiguity,
            previous_reflections=previous_reflections if previous_reflections else None,
            iterations_remaining=self.max_turns - turn,
        )
        rewrite_result = call_llm(self.model_name, rewrite_prompt,
                                  max_tokens=self.max_tokens, temperature=self.temperature)
        end = datetime.now(timezone.utc).isoformat()
        refined_req = rewrite_result["output"]
        logger.info(f"[DataInteraction] Turn {turn}: rewritten ({len(refined_req)} chars)")

        messages.append(_make_msg(
            start, end, "aunu_agent", "rewrite",
            rewrite_prompt, "data/data_interaction_rewrite.jinja",
            "", refined_req, self.model_name, rewrite_result,
        ))

        if not is_last:
            return {
                "messages": messages,
                "is_complete": False,
                "current_turn": next_turn,
                "zero_shot_draft": state["zero_shot_draft"],
                "task_requirement_final": refined_req,
                "previous_reflections": updated_reflections,
            }

        # ── Phase 2: submit final requirement ──────────────────────────
        start = datetime.now(timezone.utc).isoformat()
        env.step(finish(refined_req))
        end = datetime.now(timezone.utc).isoformat()

        messages.append(_make_msg(
            start, end, "aunu_agent", "finish",
            "", "", "", refined_req, self.model_name, {},
        ))
        logger.info(f"[DataInteraction] Submitted final requirement ({len(refined_req)} chars)")

        return {
            "messages": messages,
            "is_complete": True,
            "current_turn": next_turn,
            "zero_shot_draft": state["zero_shot_draft"],
            "task_requirement_final": refined_req,
            "previous_reflections": updated_reflections,
        }

    # ------------------------------------------------------------------
    # Router
    # ------------------------------------------------------------------

    def _router(self, state: AgentState) -> str:
        if state["is_complete"]:
            return "end"
        return "continue"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_graph(self):
        """Compile and return the LangGraph StateGraph."""
        workflow = StateGraph(AgentState)
        workflow.add_node("aunu_agent", self.process)
        workflow.set_entry_point("aunu_agent")
        workflow.add_conditional_edges(
            "aunu_agent",
            self._router,
            {"continue": "aunu_agent", "end": END},
        )
        return workflow.compile()

    def run(self, env, task) -> dict:
        """Run a complete data-interaction episode and return the trajectory log.

        Args:
            env: AUNUEnv instance (already initialised with evaluator).
            task: TaskInstance to solve.

        Returns:
            Trajectory log dict from env.get_trajectory_log(), augmented with
            the full agent message log under the key 'agent_messages'.
        """
        from AUNUEnv.aunu_env.users.mimic_user import MimicUser
        user = MimicUser(model_name=self.model_name)
        env.reset(task, user)
        self._env = env

        initial_state: AgentState = {
            "messages": [],
            "is_complete": False,
            "task_requirement_final": "",
            "current_turn": 0,
            "zero_shot_draft": "",
            "previous_reflections": [],
        }

        app = self.build_graph()
        all_messages = []
        for output in app.stream(initial_state):
            for _, state_update in output.items():
                if "messages" in state_update:
                    all_messages.extend(state_update["messages"])

        log = env.get_trajectory_log()
        log["agent_messages"] = [dict(m) for m in all_messages]
        return log


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_msg(
    start: str, end: str, role: str, action: str,
    prompt: str, template: str, ambiguity: str, output: str,
    model: str, result: dict,
) -> Message:
    return {
        "start_time": start,
        "end_time": end,
        "role": role,
        "action": action,
        "input": prompt,
        "prompt_template": template,
        "identified_ambiguity": ambiguity,
        "output": output,
        "llm": model,
        "input_tokens": result.get("input_tokens", 0),
        "output_tokens": result.get("output_tokens", 0),
        "cost": result.get("cost", 0.0),
    }
