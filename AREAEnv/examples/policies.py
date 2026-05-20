"""
Example policies for AREAEnv.

These are reference implementations of three interaction strategies.
Policies are external to AREAEnv — they interact with the environment
through the standard reset/step API.

Usage with run_experiment.py:
    python scripts/run_experiment.py \
        --config area_env/configs/zero_shot.yaml \
        --policy_module examples.policies.zero_shot_factory

    python scripts/run_experiment.py \
        --config area_env/configs/user_only.yaml \
        --policy_module examples.policies.user_only_factory

    python scripts/run_experiment.py \
        --config area_env/configs/data_only.yaml \
        --policy_module examples.policies.data_only_factory

Or use them directly:
    from examples.policies import ZeroShotPolicy, UserOnlyPolicy, DataOnlyPolicy
"""

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from area_env.env.actions import ask_user, inspect_data, propose_requirement_update, finish
from area_env.utils.llm import call_llm
from area_env.utils.jinja_utils import render_template
from area_env.utils.json_utils import parse_json_output

logger = logging.getLogger(__name__)

_PROMPTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../prompts/agents/area_agent")
)
_ZERO_SHOT_TEMPLATE = os.path.join(_PROMPTS_DIR, "zero_shot.jinja")
_USER_INTERACT_TEMPLATE = os.path.join(_PROMPTS_DIR, "user/user_interaction.jinja")
_PREDICTION_TEMPLATE = os.path.join(_PROMPTS_DIR, "user/task_requirement_prediction.jinja")
_EXECUTE_TEMPLATE = os.path.join(_PROMPTS_DIR, "data/data_interaction_execute.jinja")
_REFLECT_TEMPLATE = os.path.join(_PROMPTS_DIR, "data/data_interaction_reflect_v1.jinja")
_REWRITE_TEMPLATE = os.path.join(_PROMPTS_DIR, "data/data_interaction_rewrite.jinja")


# ---------------------------------------------------------------------------
# Zero-shot policy
# ---------------------------------------------------------------------------

class ZeroShotPolicy:
    """Generate a task requirement from the elevator pitch alone — no interaction.

    Single LLM call → finish.

    Args:
        user: MimicUser instance (not called, required by runner interface).
        model_name: LLM model for generation.
        temperature: Sampling temperature.
        max_tokens: Max output tokens.
    """

    def __init__(self, user, model_name: str, temperature: float = 0.0, max_tokens: int = 4096):
        self.user = user
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def run(self, env, task) -> dict:
        obs, _ = env.reset(task, self.user)

        prompt = render_template(_ZERO_SHOT_TEMPLATE, user_instruction=obs["user_request"])
        result = call_llm(self.model_name, prompt, max_tokens=self.max_tokens,
                          temperature=self.temperature)
        requirement = result["output"]
        logger.info(f"[ZeroShot] Generated requirement ({len(requirement)} chars)")

        env.step(finish(requirement))
        return env.get_trajectory_log()


# ---------------------------------------------------------------------------
# User-only policy
# ---------------------------------------------------------------------------

class UserOnlyPolicy:
    """Clarify requirements via the MIMIC user, then synthesize a final requirement.

    Flow:
      1. Zero-shot draft → propose_requirement_update
      2. Loop: ask_user (up to max_turns)
      3. Synthesize → finish

    Args:
        user: MimicUser instance.
        model_name: LLM model for all generation steps.
        max_turns: Maximum user clarification rounds.
        temperature: Sampling temperature.
        max_tokens: Max output tokens.
    """

    def __init__(self, user, model_name: str, max_turns: int = 5,
                 temperature: float = 0.0, max_tokens: int = 4096):
        self.user = user
        self.model_name = model_name
        self.max_turns = max_turns
        self.temperature = temperature
        self.max_tokens = max_tokens

    def run(self, env, task) -> dict:
        obs, _ = env.reset(task, self.user)

        # Step 1: zero-shot draft
        prompt = render_template(_ZERO_SHOT_TEMPLATE, user_instruction=obs["user_request"])
        result = call_llm(self.model_name, prompt, max_tokens=self.max_tokens,
                          temperature=self.temperature)
        zero_shot_draft = result["output"]
        obs, _, done, _ = env.step(propose_requirement_update(zero_shot_draft))

        # Step 2: user interaction loop
        for turn in range(1, self.max_turns + 1):
            if done:
                break
            chat_history = _build_chat_history(env.state.interaction_history)
            prompt = render_template(
                _USER_INTERACT_TEMPLATE,
                chat_history=chat_history if chat_history else None,
                initial_requirement=obs["user_request"],
                max_iterations=self.max_turns,
                current_iteration=turn,
            )
            result = call_llm(self.model_name, prompt, max_tokens=1024,
                              temperature=self.temperature)
            parsed = parse_json_output(result["output"])
            question = parsed.get("question", result["output"])
            logger.info(f"[UserOnly] Turn {turn}: {question[:80]}")
            obs, _, done, _ = env.step(ask_user(question))

        # Step 3: synthesize and finish
        chat_history = _build_chat_history(env.state.interaction_history)
        prompt = render_template(
            _PREDICTION_TEMPLATE,
            initial_task_requirement=obs["user_request"],
            zero_shot_draft=zero_shot_draft,
            chat_history=chat_history,
        )
        result = call_llm(self.model_name, prompt, max_tokens=self.max_tokens,
                          temperature=self.temperature)
        env.step(finish(result["output"]))
        return env.get_trajectory_log()


# ---------------------------------------------------------------------------
# Data-only policy
# ---------------------------------------------------------------------------

class DataOnlyPolicy:
    """Inspect dataset samples and iteratively rewrite the task requirement.

    Flow:
      1. Zero-shot draft → propose_requirement_update
      2. Loop (max_iterations): inspect_data → execute/reflect/rewrite → propose_requirement_update
      3. finish

    Args:
        user: MimicUser instance (not called, required by runner interface).
        model_name: LLM model for all generation steps.
        max_iterations: Number of data inspection + rewrite cycles.
        n_samples: Rows to sample per inspect_data action.
        temperature: Sampling temperature.
        max_tokens: Max output tokens.
    """

    def __init__(self, user, model_name: str, max_iterations: int = 3, n_samples: int = 3,
                 temperature: float = 0.0, max_tokens: int = 4096):
        self.user = user
        self.model_name = model_name
        self.max_iterations = max_iterations
        self.n_samples = n_samples
        self.temperature = temperature
        self.max_tokens = max_tokens

    def run(self, env, task) -> dict:
        obs, _ = env.reset(task, self.user)

        # Step 1: zero-shot draft
        prompt = render_template(_ZERO_SHOT_TEMPLATE, user_instruction=obs["user_request"])
        result = call_llm(self.model_name, prompt, max_tokens=self.max_tokens,
                          temperature=self.temperature)
        current_req = result["output"]
        obs, _, done, _ = env.step(propose_requirement_update(current_req))

        previous_reflections = []

        # Step 2: data inspection + rewrite loop
        for iteration in range(1, self.max_iterations + 1):
            if done:
                break
            iterations_remaining = self.max_iterations - iteration

            # Inspect data
            obs, _, done, info = env.step(
                inspect_data(n_samples=self.n_samples, query="inspect task data samples")
            )
            samples = info.get("data_samples", [])
            if not samples:
                logger.warning(f"[DataOnly] Iteration {iteration}: no data samples returned")
                break

            # Execute current requirement on each sample
            executed = []
            for s in samples:
                input_text = _extract_input(s)
                p = render_template(_EXECUTE_TEMPLATE, task_requirement=current_req, input=input_text)
                r = call_llm(self.model_name, p, max_tokens=1024, temperature=self.temperature)
                executed.append({"input": input_text[:500], "predicted_output": r["output"][:800]})

            # Reflect: identify ambiguities
            p = render_template(
                _REFLECT_TEMPLATE,
                current_task_requirement=current_req,
                sample_data=executed,
                previous_reflections=previous_reflections or None,
            )
            r = call_llm(self.model_name, p, max_tokens=2048, temperature=self.temperature)
            reflection = parse_json_output(r["output"])
            previous_reflections.append(reflection)
            logger.info(f"[DataOnly] Iteration {iteration} ambiguity: "
                        f"{reflection.get('potential_ambiguity', '')[:80]}")

            # Rewrite: fix ambiguities
            p = render_template(
                _REWRITE_TEMPLATE,
                current_task_requirement=current_req,
                output_evaluation=reflection.get("output_evaluation", ""),
                potential_ambiguity=reflection.get("potential_ambiguity", ""),
                previous_reflections=previous_reflections[:-1] or None,
                iterations_remaining=iterations_remaining,
            )
            r = call_llm(self.model_name, p, max_tokens=self.max_tokens, temperature=self.temperature)
            current_req = r["output"]
            obs, _, done, _ = env.step(propose_requirement_update(current_req))

        # Step 3: finish
        env.step(finish(current_req))
        return env.get_trajectory_log()


# ---------------------------------------------------------------------------
# Factory functions for use with run_experiment.py --policy_module
# ---------------------------------------------------------------------------

def zero_shot_factory(user, model_name: str = "gpt-4.1", **kwargs):
    return ZeroShotPolicy(user, model_name=model_name, **kwargs)


def user_only_factory(user, model_name: str = "gpt-4.1", max_turns: int = 5, **kwargs):
    return UserOnlyPolicy(user, model_name=model_name, max_turns=max_turns, **kwargs)


def data_only_factory(user, model_name: str = "gpt-4.1", max_iterations: int = 3,
                      n_samples: int = 3, **kwargs):
    return DataOnlyPolicy(user, model_name=model_name, max_iterations=max_iterations,
                          n_samples=n_samples, **kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_chat_history(interaction_history: list) -> list:
    return [
        {"role": e["role"], "output": e["content"]}
        for e in interaction_history
        if e["role"] in ("agent", "user")
    ]


def _extract_input(sample: dict) -> str:
    candidates = ["text", "article", "report", "script", "news_text", "document"]
    for c in candidates:
        if c in sample:
            return str(sample[c])[:1000]
    return str(next(iter(sample.values())))[:1000]
