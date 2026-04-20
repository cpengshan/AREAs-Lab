import pandas as pd
from agent_basic import InteractionState
import logging
import json
import re
import random
from datetime import datetime, timezone
import os
from jinja2 import Environment, FileSystemLoader
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from model.model_base import LLM

logger = logging.getLogger(__name__)

def _parse_json_output(text: str) -> dict:
    """Extract and parse JSON from LLM output that may have markdown fences and trailing content."""
    text = text.strip()
    match = re.search(r"```json\s*(.*?)\s*```", text, flags=re.DOTALL)
    if match:
        return json.loads(match.group(1))
    # Fallback: try raw text
    return json.loads(text)

PROMPT_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "../../prompts/agents/aunu_agent")

class AUNUAgent:
    def __init__(self, strategy: str, model: str, user_instruction_init: str,
                 dataset: str = None, max_turns: int = 5, persona: str = None,
                 user_interaction_template: str = "user_interaction.jinja",
                 zero_shot_seed: dict = None):
        self.strategy = strategy
        self.model_name = model
        self.llm = LLM(self.model_name)
        self.env = Environment(loader=FileSystemLoader(PROMPT_TEMPLATE_DIR))
        if self.strategy != "zero_shot" and dataset is not None:
            self.dataset = self.load_dataset(dataset)
        self.task_requirement_curr = user_instruction_init
        self.max_turns = max_turns
        self.role = "aunu_agent"
        self.persona = persona
        self.user_interaction_template = user_interaction_template
        # Pre-loaded zero_shot message (skips LLM call when provided)
        self.zero_shot_seed = zero_shot_seed

    
    # Ordered preference lists: first column name found in the DataFrame wins.
    _INPUT_COLUMN_CANDIDATES = ["text", "article", "report", "script", "news_text"]

    def _detect_input_column(self, df: pd.DataFrame) -> str:
        """Return the name of the primary input text column for a dataset."""
        for candidate in self._INPUT_COLUMN_CANDIDATES:
            if candidate in df.columns:
                return candidate
        # Fall back to the first object-dtype column, then the very first column.
        for col in df.columns:
            if df[col].dtype == object:
                return col
        return df.columns[0]

    def load_dataset(self, dataset: str) -> pd.DataFrame:
        """Load sampled_data.csv and attach a `_input_col` attribute indicating the primary input column."""
        data_path = os.path.join(os.path.dirname(__file__), "../../data/data_raw", dataset, "sampled_data.csv")
        try:
            df = pd.read_csv(data_path)
            df.attrs["input_col"] = self._detect_input_column(df)
            logger.info(f"Loaded dataset '{dataset}': {len(df)} rows, input column='{df.attrs['input_col']}'")
            return df
        except Exception as e:
            logger.error(f"Data loading failed for dataset '{dataset}': {e}")
            raise

    def create_task_requirement_init(self, user_instruction_init):
        """
        Create the initial task requirement based on user's initial instruction
        """
        # TODO
        task_requirement_init = None
        return task_requirement_init

    def _execute_on_sample(self, row: pd.Series) -> dict:
        """Apply current task requirement to one data row; return {input, predicted_output}."""
        input_col = self.dataset.attrs.get("input_col", row.index[0])
        input_text = str(row[input_col])
        execute_template = self.env.get_template("data_interaction_execute.jinja")
        prompt = execute_template.render(
            task_requirement=self.task_requirement_curr,
            input=input_text,
        )
        response = self.llm.generate(prompt)
        return {
            "input": input_text,
            "predicted_output": response["output"],
            "_tokens_in": response.get("input_tokens", 0),
            "_tokens_out": response.get("output_tokens", 0),
            "_cost": response.get("cost", 0.0),
        }

    def data_interaction(self, state: InteractionState, n_samples: int = 3):
        """
        Sample n_samples rows, execute the current task requirement on each,
        then ask the LLM to identify gaps and output a refined task requirement.
        """
        messages = state["messages"]
        current_iteration = sum(
            1 for m in messages if m["role"] == self.role and m["action"] == "data_interaction"
        )
        max_iterations = state.get("max_iterations", self.max_turns)

        # Sample rows
        sample_rows = self.dataset.sample(n=min(n_samples, len(self.dataset)), random_state=None)

        total_input_tokens = 0
        total_output_tokens = 0
        total_cost = 0.0
        sample_data = []
        for _, row in sample_rows.iterrows():
            result = self._execute_on_sample(row)
            total_input_tokens += result.pop("_tokens_in")
            total_output_tokens += result.pop("_tokens_out")
            total_cost += result.pop("_cost")
            sample_data.append(result)

        # Reflect and refine task requirement
        reflect_template = self.env.get_template("data_interaction.jinja")
        reflect_prompt = reflect_template.render(
            current_task_requirement=self.task_requirement_curr,
            sample_data=sample_data,
            max_iterations=max_iterations,
            current_iteration=current_iteration,
        )

        start_time = datetime.now(timezone.utc).isoformat()
        reflect_response = self.llm.generate(reflect_prompt, max_tokens=12288)
        end_time = datetime.now(timezone.utc).isoformat()

        total_input_tokens += reflect_response.get("input_tokens", 0)
        total_output_tokens += reflect_response.get("output_tokens", 0)
        total_cost += reflect_response.get("cost", 0.0)

        try:
            parsed = _parse_json_output(reflect_response["output"])
            improved_requirement = parsed.get("improved_task_requirement", reflect_response["output"])
            potential_ambiguity = parsed.get("potential_ambiguity", "")
        except (json.JSONDecodeError, TypeError):
            parsed = {}
            improved_requirement = reflect_response["output"]
            potential_ambiguity = ""

        self.task_requirement_curr = improved_requirement

        message = {
            "start_time": start_time,
            "end_time": end_time,
            "role": self.role,
            "action": "data_interaction",
            "input": reflect_prompt,
            "prompt_template": "data_interaction.jinja",
            "identified_ambiguity": potential_ambiguity,
            "output": improved_requirement,
            "llm": self.model_name,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "cost": total_cost,
            "data": {
                "sample_data": sample_data,
                "output_evaluation": parsed.get("output_evaluation", ""),
                "refinement_strategy": parsed.get("refinement_strategy", ""),
            },
        }
        logger.info(json.dumps(message))
        return message

    def user_interaction(self, state: InteractionState):
        """
        Figure out how to interact with user, i.e. generate a proper question
        """
        messages = state["messages"]
        template = self.env.get_template(self.user_interaction_template)

        max_iterations = state.get("max_iterations", 5)
        current_iteration = sum(1 for m in messages if m["role"] == self.role and m["action"] == "user_interaction")

        is_first_turn = current_iteration == 0
        if is_first_turn:
            initial_requirement = next(
                (m["output"] for m in messages if m["role"] == "mimic_user"), ""
            )
            prompt = template.render(
                initial_requirement=initial_requirement,
                chat_history=[],
                max_iterations=max_iterations,
                current_iteration=current_iteration,
            )
        else:
            chat_history = [m for m in messages if m["role"] in (self.role, "mimic_user")]
            prompt = template.render(
                initial_requirement="",
                chat_history=chat_history,
                max_iterations=max_iterations,
                current_iteration=current_iteration,
            )

        start_time = datetime.now(timezone.utc).isoformat()
        response = self.llm.generate(prompt)
        end_time = datetime.now(timezone.utc).isoformat()

        try:
            parsed = _parse_json_output(response["output"])
            question = parsed.get("question", response["output"])
            identified_ambiguity = parsed.get("identified_ambiguity", "")
        except (json.JSONDecodeError, TypeError):
            question = response["output"]
            identified_ambiguity = ""

        message = {
            "start_time": start_time,
            "end_time": end_time,
            "role": self.role,
            "action": "user_interaction",
            "input": prompt,
            "prompt_template": self.user_interaction_template,
            "identified_ambiguity": identified_ambiguity,
            "output": question,
            "llm": self.model_name,
            "input_tokens": response.get("input_tokens", 0),
            "output_tokens": response.get("output_tokens", 0),
            "cost": response.get("cost", 0.0),
        }
        logger.info(json.dumps(message))
        return message

    def reflection(self, state: InteractionState):
        """
        Check human and/or data interaction results, figure out what in the current task_requirement should be revised.
        """
        task_requirement_revised = ""  # TODO
        start_time = datetime.now(timezone.utc).isoformat()
        end_time = datetime.now(timezone.utc).isoformat()
        message = {
            "start_time": start_time,
            "end_time": end_time,
            "role": self.role,
            "action": "reflection",
            "input": "",  # TODO: rendered prompt
            "prompt_template": "",  # TODO: template name
            "identified_ambiguity": "",
            "output": task_requirement_revised,
            "llm": self.model_name,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost": 0.0,
        }
        logger.info(json.dumps(message))
        return message

    def task_requirement_revision(self, state: InteractionState):
        """
        Apply revisions based on reflection results.
        * Should we merge reflectiion with task_requirement_revision? *
        """
        task_requirement_revised = ""  # TODO
        self.task_requirement_curr = task_requirement_revised
        start_time = datetime.now(timezone.utc).isoformat()
        end_time = datetime.now(timezone.utc).isoformat()
        message = {
            "start_time": start_time,
            "end_time": end_time,
            "role": self.role,
            "action": "task_requirement_revision",
            "input": "",  # TODO: rendered prompt
            "prompt_template": "",  # TODO: template name
            "identified_ambiguity": "",
            "output": task_requirement_revised,
            "llm": self.model_name,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost": 0.0,
        }
        logger.info(json.dumps(message))
        return message


    def merge_zero_shot_data(self, state: InteractionState, user_instruction: str) -> dict:
        """
        Final step for the data strategy: merge the zero-shot draft from the start
        of the session with the final data-interaction requirement.
        """
        messages = state["messages"]

        zero_shot_requirement = next(
            (m["output"] for m in messages if m["role"] == self.role and m["action"] == "zero_shot"),
            "",
        )

        merge_template = self.env.get_template("merge_zero_shot_data.jinja")
        merge_prompt = merge_template.render(
            user_instruction=user_instruction,
            zero_shot_requirement=zero_shot_requirement,
            data_interaction_requirement=self.task_requirement_curr,
        )
        merge_start = datetime.now(timezone.utc).isoformat()
        merge_response = self.llm.generate(merge_prompt)
        merge_end = datetime.now(timezone.utc).isoformat()

        self.task_requirement_curr = merge_response["output"]

        merge_message = {
            "start_time": merge_start,
            "end_time": merge_end,
            "role": self.role,
            "action": "merge_zero_shot_data",
            "input": merge_prompt,
            "prompt_template": "merge_zero_shot_data.jinja",
            "identified_ambiguity": "",
            "output": merge_response["output"],
            "llm": self.model_name,
            "input_tokens": merge_response.get("input_tokens", 0),
            "output_tokens": merge_response.get("output_tokens", 0),
            "cost": merge_response.get("cost", 0.0),
        }
        logger.info(json.dumps(merge_message))
        return [merge_message]

    def task_requirement_prediction(self, state: InteractionState):
        """
        Synthesize the full conversation into a final predicted task requirement.
        Called once at the end of the interaction.
        """
        messages = state["messages"]
        template = self.env.get_template("task_requirement_prediction.jinja")

        initial_task_requirement = next(
            (m["output"] for m in messages if m["role"] == "mimic_user" and m["action"] == "init"),
            "",
        )
        zero_shot_draft = next(
            (m["output"] for m in messages if m["role"] == self.role and m["action"] == "zero_shot"),
            "",
        )
        chat_history = [m for m in messages if m["role"] in (self.role, "mimic_user") and m["action"] not in ("init", "zero_shot")]

        prompt = template.render(
            initial_task_requirement=initial_task_requirement,
            zero_shot_draft=zero_shot_draft,
            chat_history=chat_history,
        )

        start_time = datetime.now(timezone.utc).isoformat()
        response = self.llm.generate(prompt, max_tokens=12288)
        end_time = datetime.now(timezone.utc).isoformat()

        self.task_requirement_curr = response["output"]

        message = {
            "start_time": start_time,
            "end_time": end_time,
            "role": self.role,
            "action": "task_requirement_prediction",
            "input": prompt,
            "prompt_template": "task_requirement_prediction.jinja",
            "identified_ambiguity": "",
            "output": response["output"],
            "llm": self.model_name,
            "input_tokens": response.get("input_tokens", 0),
            "output_tokens": response.get("output_tokens", 0),
            "cost": response.get("cost", 0.0),
        }
        logger.info(json.dumps(message))
        return message

    def _route(self, state: InteractionState) -> tuple[str, dict]:
        """
        Call the router LLM to decide 'user' or 'data' for this hybrid turn.
        Returns (strategy, router_message).
        """
        messages = state["messages"]
        current_iteration = sum(
            1 for m in messages if m["role"] == self.role and m["action"] == "hybrid_router"
        )
        max_iterations = state.get("max_iterations", self.max_turns)

        user_instruction = next(
            (m["output"] for m in messages if m["role"] == "mimic_user" and m["action"] == "init"),
            "",
        )
        chat_history = [
            m for m in messages
            if m["role"] in (self.role, "mimic_user") and m["action"] not in ("init", "zero_shot")
        ]

        template = self.env.get_template("hybrid_router.jinja")
        prompt = template.render(
            user_instruction=user_instruction,
            current_task_requirement=self.task_requirement_curr,
            chat_history=chat_history,
            max_iterations=max_iterations,
            current_iteration=current_iteration,
        )

        start_time = datetime.now(timezone.utc).isoformat()
        response = self.llm.generate(prompt)
        end_time = datetime.now(timezone.utc).isoformat()

        try:
            parsed = _parse_json_output(response["output"])
            strategy = parsed.get("strategy", "user")
            reasoning = parsed.get("reasoning", "")
            if strategy not in ("user", "data"):
                strategy = "user"
        except (json.JSONDecodeError, TypeError):
            strategy = "user"
            reasoning = ""

        # Accumulate all router decisions from previous turns into a running list
        prior_list = []
        for m in messages:
            if m["role"] == self.role and m["action"] == "hybrid_router":
                prior_list.extend(m.get("data", {}).get("list", []))
        reasoning_list = prior_list + [{"turn": current_iteration, "strategy": strategy, "reasoning": reasoning}]

        router_message = {
            "start_time": start_time,
            "end_time": end_time,
            "role": self.role,
            "action": "hybrid_router",
            "input": prompt,
            "prompt_template": "hybrid_router.jinja",
            "identified_ambiguity": "",
            "output": strategy,
            "llm": self.model_name,
            "input_tokens": response.get("input_tokens", 0),
            "output_tokens": response.get("output_tokens", 0),
            "cost": response.get("cost", 0.0),
            "data": {"list": reasoning_list},
        }
        logger.info(json.dumps(router_message))
        return strategy, router_message

    def hybrid_interaction(self, state: InteractionState) -> list:
        """
        One turn of the hybrid strategy:
        1. Sample 1 data row and execute current task requirement.
        2. LLM checks gaps and refines the task requirement (hybrid_data_check.jinja).
        3. Run user_interaction to generate a question for mimic_user.
        Returns a list of messages [hybrid_data_check_msg, user_interaction_msg].
        """
        messages = state["messages"]
        current_iteration = sum(
            1 for m in messages if m["role"] == self.role and m["action"] == "hybrid_data_check"
        )
        max_iterations = state.get("max_iterations", self.max_turns)

        # Step 1: Sample 1 row and execute current task requirement
        row = self.dataset.sample(n=1).iloc[0]
        result = self._execute_on_sample(row)
        exec_input_tokens = result.pop("_tokens_in")
        exec_output_tokens = result.pop("_tokens_out")
        exec_cost = result.pop("_cost")
        sample_data = result  # {input, predicted_output}

        # Step 2: Get the most recent user feedback (if any) to incorporate
        last_user_msg = next(
            (m for m in reversed(messages) if m["role"] == "mimic_user" and m["action"] == "respond"),
            None,
        )
        user_feedback = last_user_msg["output"] if last_user_msg else ""

        # Step 3: Check gaps and refine task requirement
        check_template = self.env.get_template("hybrid_data_check.jinja")
        check_prompt = check_template.render(
            current_task_requirement=self.task_requirement_curr,
            sample=sample_data,
            user_feedback=user_feedback,
            max_iterations=max_iterations,
            current_iteration=current_iteration,
        )

        start_time = datetime.now(timezone.utc).isoformat()
        check_response = self.llm.generate(check_prompt)
        end_time = datetime.now(timezone.utc).isoformat()

        self.task_requirement_curr = check_response["output"]

        total_input_tokens = exec_input_tokens + check_response.get("input_tokens", 0)
        total_output_tokens = exec_output_tokens + check_response.get("output_tokens", 0)
        total_cost = exec_cost + check_response.get("cost", 0.0)

        check_message = {
            "start_time": start_time,
            "end_time": end_time,
            "role": self.role,
            "action": "hybrid_data_check",
            "input": check_prompt,
            "prompt_template": "hybrid_data_check.jinja",
            "identified_ambiguity": "",
            "output": check_response["output"],
            "sample_data": [sample_data],
            "llm": self.model_name,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "cost": total_cost,
        }
        logger.info(json.dumps(check_message))

        # Step 4: Generate question/refined requirement for mimic_user
        ui_state = {**state, "messages": messages + [check_message]}
        user_msg = self.user_interaction(ui_state)

        return [check_message, user_msg]

    def process(self, state: InteractionState):
        """
        The core reasoning node for the AU-NU Agent.
        """
        messages = state["messages"]
        # --- PSEUDO-CODE FOR STRATEGIES ---
        if self.strategy == "zero_shot":
            template = self.env.get_template("zero_shot.jinja")
            prompt = template.render(user_instruction=self.task_requirement_curr)

            start_time = datetime.now(timezone.utc).isoformat()
            response = self.llm.generate(prompt)
            end_time = datetime.now(timezone.utc).isoformat()

            self.task_requirement_curr = response["output"]

            message = {
                "start_time": start_time,
                "end_time": end_time,
                "role": self.role,
                "action": "zero_shot",
                "input": prompt,
                "prompt_template": "zero_shot.jinja",
                "output": response["output"],
                "llm": self.model_name,
                "input_tokens": response.get("input_tokens", 0),
                "output_tokens": response.get("output_tokens", 0),
                "cost": response.get("cost", 0.0),
            }
            logger.info(json.dumps(message))

            return {
                "messages": [message],
                "is_complete": True,
                "task_requirement_final": self.task_requirement_curr,
            }
        elif self.strategy == "persona":
            template = self.env.get_template("zero_shot.jinja")
            prompt = template.render(
                user_instruction=self.task_requirement_curr,
                persona=self.persona,
            )

            start_time = datetime.now(timezone.utc).isoformat()
            response = self.llm.generate(prompt)
            end_time = datetime.now(timezone.utc).isoformat()

            self.task_requirement_curr = response["output"]

            message = {
                "start_time": start_time,
                "end_time": end_time,
                "role": self.role,
                "action": "zero_shot",
                "input": prompt,
                "prompt_template": "zero_shot.jinja",
                "output": response["output"],
                "llm": self.model_name,
                "input_tokens": response.get("input_tokens", 0),
                "output_tokens": response.get("output_tokens", 0),
                "cost": response.get("cost", 0.0),
            }
            logger.info(json.dumps(message))

            return {
                "messages": [message],
                "is_complete": True,
                "task_requirement_final": self.task_requirement_curr,
            }
        elif self.strategy == "user":
            new_messages = []

            # First call only: seed task_requirement_curr with a zero-shot draft
            has_zero_shot = any(m["role"] == self.role and m["action"] == "zero_shot" for m in messages)
            if not has_zero_shot:
                if self.zero_shot_seed is not None:
                    # Reuse pre-loaded zero_shot output — no LLM call
                    zs_message = self.zero_shot_seed
                    self.task_requirement_curr = zs_message["output"]
                else:
                    template = self.env.get_template("zero_shot.jinja")
                    prompt = template.render(user_instruction=self.task_requirement_curr)
                    start_time = datetime.now(timezone.utc).isoformat()
                    response = self.llm.generate(prompt)
                    end_time = datetime.now(timezone.utc).isoformat()
                    self.task_requirement_curr = response["output"]
                    zs_message = {
                        "start_time": start_time,
                        "end_time": end_time,
                        "role": self.role,
                        "action": "zero_shot",
                        "input": prompt,
                        "prompt_template": "zero_shot.jinja",
                        "identified_ambiguity": "",
                        "output": response["output"],
                        "llm": self.model_name,
                        "input_tokens": response.get("input_tokens", 0),
                        "output_tokens": response.get("output_tokens", 0),
                        "cost": response.get("cost", 0.0),
                    }
                logger.info(json.dumps(zs_message))
                new_messages.append(zs_message)

            ui_message = self.user_interaction({**state, "messages": messages + new_messages})
            new_messages.append(ui_message)

            aunu_turns = sum(1 for m in messages if m["role"] == self.role and m["action"] == "user_interaction") + 1
            is_complete = aunu_turns >= self.max_turns
            if is_complete:
                pred_state = {"messages": messages + new_messages}
                pred_message = self.task_requirement_prediction(pred_state)
                new_messages.append(pred_message)
            return {
                "messages": new_messages,
                "is_complete": is_complete,
                "task_requirement_final": self.task_requirement_curr,
            }

        elif self.strategy == "data":
            new_messages = []

            # Step 1: seed task_requirement_curr with a zero-shot draft
            has_zero_shot = any(m["role"] == self.role and m["action"] == "zero_shot" for m in messages)
            if not has_zero_shot:
                if self.zero_shot_seed is not None:
                    zs_message = self.zero_shot_seed
                    self.task_requirement_curr = zs_message["output"]
                else:
                    template = self.env.get_template("zero_shot.jinja")
                    prompt = template.render(user_instruction=self.task_requirement_curr)
                    start_time = datetime.now(timezone.utc).isoformat()
                    response = self.llm.generate(prompt)
                    end_time = datetime.now(timezone.utc).isoformat()
                    self.task_requirement_curr = response["output"]
                    zs_message = {
                        "start_time": start_time,
                        "end_time": end_time,
                        "role": self.role,
                        "action": "zero_shot",
                        "input": prompt,
                        "prompt_template": "zero_shot.jinja",
                        "identified_ambiguity": "",
                        "output": response["output"],
                        "llm": self.model_name,
                        "input_tokens": response.get("input_tokens", 0),
                        "output_tokens": response.get("output_tokens", 0),
                        "cost": response.get("cost", 0.0),
                    }
                logger.info(json.dumps(zs_message))
                new_messages.append(zs_message)

            # Step 2: data interaction loop (sample → execute → reflect)
            data_turns = sum(
                1 for m in (messages + new_messages)
                if m["role"] == self.role and m["action"] == "data_interaction"
            )
            is_complete = data_turns >= self.max_turns
            if not is_complete:
                di_state = {**state, "messages": messages + new_messages, "max_iterations": self.max_turns}
                data_message = self.data_interaction(di_state)
                new_messages.append(data_message)
                data_turns += 1
                is_complete = data_turns >= self.max_turns

            # Step 3: on completion, merge zero-shot seed with final data-interaction result
            if is_complete:
                merge_state = {"messages": messages + new_messages}
                merge_messages = self.merge_zero_shot_data(merge_state, self.task_requirement_curr)
                new_messages.extend(merge_messages)

            return {
                "messages": new_messages,
                "is_complete": is_complete,
                "task_requirement_final": self.task_requirement_curr,
            }

        elif self.strategy == "hybrid":
            new_messages = []

            # Step 1: seed task_requirement_curr with a zero-shot draft
            has_zero_shot = any(m["role"] == self.role and m["action"] == "zero_shot" for m in messages)
            if not has_zero_shot:
                if self.zero_shot_seed is not None:
                    zs_message = self.zero_shot_seed
                    self.task_requirement_curr = zs_message["output"]
                else:
                    template = self.env.get_template("zero_shot.jinja")
                    prompt = template.render(user_instruction=self.task_requirement_curr)
                    start_time = datetime.now(timezone.utc).isoformat()
                    response = self.llm.generate(prompt)
                    end_time = datetime.now(timezone.utc).isoformat()
                    self.task_requirement_curr = response["output"]
                    zs_message = {
                        "start_time": start_time,
                        "end_time": end_time,
                        "role": self.role,
                        "action": "zero_shot",
                        "input": prompt,
                        "prompt_template": "zero_shot.jinja",
                        "identified_ambiguity": "",
                        "output": response["output"],
                        "llm": self.model_name,
                        "input_tokens": response.get("input_tokens", 0),
                        "output_tokens": response.get("output_tokens", 0),
                        "cost": response.get("cost", 0.0),
                    }
                logger.info(json.dumps(zs_message))
                new_messages.append(zs_message)

            # Count completed hybrid turns (each router decision = one turn)
            mix_turns = sum(
                1 for m in messages if m["role"] == self.role and m["action"] == "hybrid_router"
            )
            is_complete = mix_turns >= self.max_turns

            if not is_complete:
                mix_state = {**state, "messages": messages + new_messages, "max_iterations": self.max_turns}

                # Step 2: router decides which strategy to run this turn
                chosen_strategy, router_msg = self._route(mix_state)
                new_messages.append(router_msg)
                mix_state = {**mix_state, "messages": mix_state["messages"] + [router_msg]}

                # Step 3: run one iteration of the chosen strategy
                if chosen_strategy == "data":
                    data_msg = self.data_interaction(mix_state)
                    new_messages.append(data_msg)
                else:
                    ui_msg = self.user_interaction(mix_state)
                    new_messages.append(ui_msg)

                mix_turns += 1
                is_complete = mix_turns >= self.max_turns

            if is_complete:
                pred_state = {"messages": messages + new_messages}
                pred_message = self.task_requirement_prediction(pred_state)
                new_messages.append(pred_message)

            return {
                "messages": new_messages,
                "is_complete": is_complete,
                "task_requirement_final": self.task_requirement_curr,
            }

        # Fallback
        return {
            "messages": [],
            "is_complete": True,
            "task_requirement_final": self.task_requirement_curr,
        }
