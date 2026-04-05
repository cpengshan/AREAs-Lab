import time
from typing import Dict, Any, Optional, List

class AUNUAgent:
    """
    AI-Assisted User Needs Understanding (AU-NU) Agent.
    Acts as a bridge between vague intent and formal specifications.
    """
    def __init__(self, strategy: str):
        # TODO
        return

    def process_turn(self, user_input: str, data_context: Optional[Any] = None) -> Dict[str, Any]:
        """
        Decision engine: Determines whether to ask a clarification question 
        or finalize the specification based on the strategy.
        """
        self.iteration += 1
        
        # Strategy Logic: Zero-Shot completes immediately
        if self.strategy == "zero_shot":
            return self._finalize(user_input)
        
        # Strategy Logic: Active Elicitation or Hybrid requires interaction
        if self.iteration < 3: 
            return {
                "action": "ASK_CLARIFICATION",
                "content": f"I see you want '{user_input}'. Could you specify the time range and data granularity?",
                "agent_state": "eliciting_requirements"
            }
        else:
            return self._finalize(user_input)

    def _finalize(self, final_input: str) -> Dict[str, Any]:
        """Generates the high-fidelity output."""
        self.specification["intent"] = final_input
        self.specification["confidence_score"] = 0.95
        return {
            "action": "EMIT_SPEC",
            "content": f"FINAL_SPEC: [Intent: {final_input}] [Resolution: High-Fidelity]",
            "agent_state": "finalized"
        }

class MimicUser:
    """
    The Evaluator/Simulator that holds the 'Ground Truth'.
    Used to test the Agent's ability to extract hidden requirements.
    """
    def __init__(self, ground_truth: str):
        self.ground_truth = ground_truth

    def generate_response(self, agent_question: str) -> str:
        """
        Simulates a human response by revealing a bit more of the 
        ground truth based on the agent's question.
        """
        # In a real scenario, this would use an LLM to parse the 
        # agent_question and provide a relevant subset of the ground_truth.
        return f"To answer your question: {self.ground_truth}. Please use weekly aggregation."

# --- Orchestration Logic ---

class InteractionManager:
    """Manages the loop between the Agent and the Mimic User."""
    def __init__(self, agent: AUNUAgent, user: MimicUser):
        self.agent = agent
        self.user = user

    def run(self, initial_prompt: str):
        print(f"--- Starting AU-NU Session | Strategy: {self.agent.strategy} ---")
        print(f"User Initial Prompt: {initial_prompt}\n")
        
        current_input = initial_prompt
        
        while True:
            # Step 1: Agent processes the current input
            response = self.agent.process_turn(current_input)
            
            if response["action"] == "EMIT_SPEC":
                print(f"✅ [FINAL SPECIFICATION ADOPTED]:\n{response['content']}")
                break
            
            # Step 2: Agent asks a question (Active Elicitation)
            print(f"🤖 Agent Question: {response['content']}")
            
            # Step 3: Mimic User provides feedback based on Ground Truth
            current_input = self.user.generate_response(response['content'])
            print(f"👤 Mimic User Feedback: {current_input}\n")
            
            time.sleep(0.5) # Simulate processing delay

# --- Main Execution ---

if __name__ == "__main__":
    # Define what the user 'actually' wants but didn't say initially
    hidden_intent = "Analyze Q4 revenue across APAC regions, excluding internal test accounts."
    
    # Initialize components
    my_agent = AUNUAgent(strategy="active_elicitation")
    my_mimic = MimicUser(hidden_intent)
    
    # Run simulation
    manager = InteractionManager(my_agent, my_mimic)
    manager.run("I need a sales report.")