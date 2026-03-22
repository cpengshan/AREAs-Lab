import pandas as pd
from agent_basic import InteractionState
import logging
import json

logger = logging.getLogger(__name__)

class AUNUAgent:
    def __init__(self, args):
        self.strategy = args.strategy
        self.dataset = self.load_dataset(args.data_path)
        self.task_requirement_curr = self.create_task_requirement(args.user_instruction_init)
        self.role = "aunu_agent"
    
    def load_dataset(self, data_path):
        """
        Load the dataset csv file
        """
        try:
            df = pd.read_csv(data_path)
            return df
        except:
            logger.info("Data loading failed")

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
    
    def reflectiion(self, state: InteractionState):
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
