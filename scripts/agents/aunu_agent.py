import pandas as pd
from agent_basic import InteractionState
import logging
import json
from datetime import datetime, timezone
import os
from jinja2 import Environment, FileSystemLoader
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from model.model_base import LLM

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "../../prompts/agents/aunu_agent")

class AUNUAgent:
    def __init__(self, strategy: str, model: str, user_instruction_init: str,
                 dataset: str = None, max_turns: int = 5):
        self.strategy = strategy
        self.model_name = model
        self.llm = LLM(self.model_name)
        self.env = Environment(loader=FileSystemLoader(PROMPT_TEMPLATE_DIR))
        if self.strategy != "zero_shot" and dataset is not None:
            self.dataset = self.load_dataset(dataset)
        self.task_requirement_curr = user_instruction_init
        self.max_turns = max_turns
        self.role = "aunu_agent"
    
    def load_dataset(self, dataset: str):
        """
        Load the dataset CSV file from data/data_processed/<dataset>/data.csv
        """
        data_path = os.path.join(os.path.dirname(__file__), "../../data/data_processed", dataset, "data.csv")
        try:
            df = pd.read_csv(data_path)
            return df
        except Exception as e:
            logger.info(f"Data loading failed for dataset '{dataset}': {e}")

    def create_task_requirement_init(self, user_instruction_init):
        """
        Create the initial task requirement based on user's initial instruction
        """
        # TODO
        task_requirement_init = None
        return task_requirement_init

    def data_interaction(self, state: InteractionState):
        """
        Figure out how to interact with data and get experiment results
        """
        experiment_results = "Some experiment results"  # TODO
        start_time = datetime.now(timezone.utc).isoformat()
        end_time = datetime.now(timezone.utc).isoformat()
        message = {
            "start_time": start_time,
            "end_time": end_time,
            "role": self.role,
            "action": "data_interaction",
            "input": "",  # TODO: rendered prompt
            "output": experiment_results,
            "llm": self.model_name,
        }
        logger.info(json.dumps(message))
        return message

    def user_interaction(self, state: InteractionState):
        """
        Figure out how to interact with user, i.e. generate a proper question
        """
        messages = state["messages"]
        template = self.env.get_template("user_interaction.jinja")

        is_first_turn = not any(m["role"] == self.role for m in messages)
        if is_first_turn:
            initial_requirement = next(
                (m["output"] for m in messages if m["role"] == "mimic_user"), ""
            )
            prompt = template.render(initial_requirement=initial_requirement, chat_history=[])
        else:
            chat_history = [m for m in messages if m["role"] in (self.role, "mimic_user")]
            prompt = template.render(initial_requirement="", chat_history=chat_history)

        start_time = datetime.now(timezone.utc).isoformat()
        response = self.llm.generate(prompt)
        end_time = datetime.now(timezone.utc).isoformat()

        message = {
            "start_time": start_time,
            "end_time": end_time,
            "role": self.role,
            "action": "user_interaction",
            "input": prompt,
            "output": response["output"],
            "llm": self.model_name,
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
            "output": task_requirement_revised,
            "llm": self.model_name,
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
            "output": task_requirement_revised,
            "llm": self.model_name,
        }
        logger.info(json.dumps(message))
        return message


    def task_requirement_prediction(self, state: InteractionState):
        """
        Synthesize the full conversation into a final predicted task requirement.
        Called once at the end of the interaction.
        """
        messages = state["messages"]
        template = self.env.get_template("task_requirement_prediction.jinja")

        initial_task_requirement = next(
            (m["output"] for m in messages if m["role"] == "mimic_user" and m["action"] == "init"), ""
        )
        chat_history = [m for m in messages if m["role"] in (self.role, "mimic_user") and m["action"] != "init"]

        prompt = template.render(
            initial_task_requirement=initial_task_requirement,
            chat_history=chat_history,
        )

        start_time = datetime.now(timezone.utc).isoformat()
        response = self.llm.generate(prompt)
        end_time = datetime.now(timezone.utc).isoformat()

        self.task_requirement_curr = response["output"]

        message = {
            "start_time": start_time,
            "end_time": end_time,
            "role": self.role,
            "action": "task_requirement_prediction",
            "input": prompt,
            "output": response["output"],
            "llm": self.model_name,
        }
        logger.info(json.dumps(message))
        return message

    def process(self, state: InteractionState):
        """
        The core reasoning node for the AU-NU Agent.
        """
        messages = state["messages"]
        # --- PSEUDO-CODE FOR STRATEGIES ---
        if self.strategy == "zero_shot":
            template = self.env.get_template("zeroshot.jinja")
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
                "output": response["output"],
                "llm": self.model_name,
            }
            logger.info(json.dumps(message))

            return {
                "messages": [message],
                "is_complete": True,
                "task_requirement_final": self.task_requirement_curr,
            }
        elif self.strategy == "user":
            message = self.user_interaction(state)
            aunu_turns = sum(1 for m in messages if m["role"] == self.role) + 1
            is_complete = aunu_turns >= self.max_turns
            new_messages = [message]
            if is_complete:
                # Predict final task requirement from full conversation
                pred_state = {"messages": messages + [message]}
                pred_message = self.task_requirement_prediction(pred_state)
                new_messages.append(pred_message)
            return {
                "messages": new_messages,
                "is_complete": is_complete,
                "task_requirement_final": self.task_requirement_curr,
            }

        elif self.strategy == "data":
            message = self.data_interaction(state)
            aunu_turns = sum(1 for m in messages if m["role"] == self.role) + 1
            is_complete = aunu_turns >= self.max_turns
            new_messages = [message]
            if is_complete:
                pred_state = {"messages": messages + [message]}
                pred_message = self.task_requirement_prediction(pred_state)
                new_messages.append(pred_message)
            return {
                "messages": new_messages,
                "is_complete": is_complete,
                "task_requirement_final": self.task_requirement_curr,
            }

        elif self.strategy == "mix":
            user_msg = self.user_interaction(state)
            data_msg = self.data_interaction(state)
            aunu_turns = sum(1 for m in messages if m["role"] == self.role) + 2
            is_complete = aunu_turns >= self.max_turns
            new_messages = [user_msg, data_msg]
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
