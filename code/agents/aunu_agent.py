import pandas as pd
from agent_basic import InteractionState


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
        messages.append({"role": "experiment_result", "content": experiment_results})
        return messages
    
    def user_interaction(self, state: InteractionState):
        """
        Figure out how to interact with user, i.e. generate a proper question
        """
        messages = state["messages"]
        agent_question = "Tell me more" # TODO
        messages.append({"role": "aunu_agent", "content": agent_question})
        return messages
    
    def revise_task_requirement(self, state: InteractionState):
        """
        revise the task requirement based on the latest experiment results and/or user feedback
        """
        messages = state["messages"]
        task_requirement_revised = None
        # TODO
        self.task_requirement_curr = task_requirement_revised


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
