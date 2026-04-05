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

PROMPT_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "../../prompts/agents/AUNU")

class AUNUAgent:
    def __init__(self, args):
        self.strategy = args.strategy
        self.model_name = args.aunu_model
        self.llm = LLM(self.model_name)
        self.env = Environment(loader=FileSystemLoader(PROMPT_TEMPLATE_DIR))
        if self.strategy != "zero_shot":
            self.dataset = self.load_dataset(args.dataset)
        self.task_requirement_curr = args.user_instruction_init if hasattr(args, "user_instruction_init") else ""
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
        messages = state["messages"]
        experiment_results = "Some experiment results"  # TODO
        message = {"role": self.role, "action": "experiment", "content": experiment_results}
        logger.info({json.dumps(message)})
        messages.append(message)
        return messages
    
    def user_interaction(self, state: InteractionState):
        """
        Figure out how to interact with user, i.e. generate a proper question
        """
        messages = state["messages"]
        agent_question = "Tell me more" # TODO
        message = {"role": "aunu_agent", "action": "question", "content": agent_question}
        logger.info({json.dumps(message)})
        messages.append(message)
        return messages
    
    def reflection(self, state: InteractionState):
        """
        Check human and/or data interaction results, figure out what in the current task_requirement should be revised.
        """
        messages = state["messages"]
        message = {"role": "aunu_agent", "action": "reflection", "content": task_requirement_revised}
        logger.info({json.dumps(message)})
        messages.append(message)
        return messages
    
    def task_requirement_revision(self, state: InteractionState):
        """
        Apply revisions based on reflection results. 
        * Should we merge reflectiion with task_requirement_revision? *
        """
        messages = state["messages"]
        task_requirement_revised = None
        # TODO
        self.task_requirement_curr = task_requirement_revised
        message = {"role": "task_requirement_update", "content": task_requirement_revised}
        logger.info({json.dumps(message)})
        messages.append(message)
        return messages


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

            self.task_requirement_curr = response

            message = {
                "start_time": start_time,
                "end_time": end_time,
                "role": self.role,
                "action": "zero_shot",
                "input": prompt,
                "output": response,
                "llm": self.model_name,
            }
            logger.info(json.dumps(message))
            messages.append(message)

            return {
                "messages": messages,
                "is_complete": True,
                "task_requirement_final": self.task_requirement_curr,
            }
        elif self.strategy == "user_interaction":
            # TODO: 
            return {
                "messages": messages,
                "task_requirement_final": self.task_requirement_curr
            }
            
        elif self.strategy == "data_interaction":
            # TODO: 
            return {
                "messages": messages,
                "task_requirement_final": self.task_requirement_curr
            }
        # TODO: Other strategies
