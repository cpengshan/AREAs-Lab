import pandas as pd
from agent_basic import InteractionState
import logging
import json

logger = logging.getLogger(__name__)

class AUNUAgent:
    def __init__(self, strategy, user_instruction_init, data_path):
        self.strategy = strategy
        self.dataset = self.load_dataset(data_path)
        self.task_requirement_curr = self.create_task_requirement(user_instruction_init)
    
    def load_dataset(self, data_path):
        """
        Load the dataset csv file
        """
        # TODO
        df = pd.read_csv(data_path)
        return df

    def create_task_requirement_init(self, user_instruction_init):
        """
        Create the initial task requirement based on user's initial instruction
        """
        task_requirement_init = user_instruction_init
        return task_requirement_init

    def data_interaction(self, state: InteractionState):
        """
        Figure out how to interact with data and get experiment results
        """
        messages = state["messages"]
        experiment_results = "Some experiment results"  # TODO
        message = {"role": "experiment_result", "content": experiment_results}
        logger.info({json.dumps(message)})
        messages.append(message)
        return messages
    
    def user_interaction(self, state: InteractionState):
        """
        Figure out how to interact with user, i.e. generate a proper question
        """
        messages = state["messages"]
        agent_question = "Tell me more" # TODO
        message = {"role": "aunu_agent_question", "content": agent_question}
        logger.info({json.dumps(message)})
        messages.append(message)
        return messages
    
    def revise_task_requirement(self, state: InteractionState):
        """
        revise the task requirement based on the latest experiment results and/or user feedback
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
            # TODO: 
            return {
                "messages": messages,
                "task_requirement_final": self.task_requirement_curr
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
