import json
import logging
import os
import ast
from typing import Dict, Any
from jinja2 import Environment, FileSystemLoader
import datetime # Import datetime module

logger = logging.getLogger(__name__)

class MimicUser:
    def __init__(self, 
                persona: Dict[str, Any], 
                task_requirement_gold: str,
                user_instruction_init: str,
                communication_style_init: str,
                model: str, 
                template_path: str):
        self.persona = persona
        self.task_requirement_gold = task_requirement_gold
        self.user_instruction_init = user_instruction_init
        self.communication_style_init = communication_style_init
        self.model = model
        self.template_path = template_path

        template_dir = os.path.dirname(template_path)
        template_file = os.path.basename(template_path)
        self.env = Environment(loader=FileSystemLoader(template_dir))
        self.template = self.env.get_template(template_file)

    def _render_prompt(self, state: Dict[str, Any]) -> str:
        context = {
            "persona": self.persona,
            "task_requirement_gold": self.task_requirement_gold,
            "user_instruction_init": self.user_instruction_init,
            "communication_style_init": self.communication_style_init,
            "state_messages": state.get("messages", []),
            "last_agent_question": state["messages"][-1]["input"] if state["messages"] and state["messages"][-1]["role"] == "agent" else ""
        }
        return self.template.render(context)

    def _call_llm(self, rendered_prompt: str) -> Dict[str, Any]:
        # Placeholder for actual LLM call.
        # This should return a dictionary with at least 'feedback' or 'reply',
        # 'is_complete', and optionally 'thought' and 'llm_metadata'.
        mock_feedback = "I’m mainly looking to create a tool that can take various messy and sometimes repetitive news inputs and turn them into one clear, high-quality story that feels consistent in voice and balanced in perspective. It’s about streamlining multiple viewpoints into a single, reliable narrative that maintains our editorial standards."
        mock_thought = "The assistant is asking for a high-level summary of my goals, but I’m not fully ready to lay out all the details. I want to keep the response somewhat vague, reflecting that I know what I'm aiming for in general but haven’t pinned down all the specifics or methods yet. I want to convey that I’m looking for a refined, authoritative end result without drowning the assistant in technicalities or latent nuances just yet."
        return {
            "feedback": mock_feedback,
            "reply": mock_feedback,
            "is_complete": False,
            "thought": mock_thought,
            "llm_metadata": {"model_name": self.model, "prompt_tokens": len(rendered_prompt.split()), "completion_tokens": len(mock_feedback.split())}
        }

    def respond(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if state.get("is_complete", False):
            return {"messages": [], "is_complete": True, "raw": {}}

        rendered_prompt = self._render_prompt(state)
        llm_result = self._call_llm(rendered_prompt)

        reply = (
            llm_result.get("feedback")
            or llm_result.get("reply")
            or "I haven't thought that far ahead yet."
        )
        is_complete = bool(llm_result.get("is_complete", False))

        # Get current time for start_time and end_time
        current_time = datetime.datetime.now().isoformat()

        # Get the last agent question from the state, if available
        last_agent_question = state["messages"][-1]["input"] if state["messages"] and state["messages"][-1]["role"] == "agent" else ""

        # Updated message format as requested, adapting for 'mimic_user' role
        message = {
            "start_time": current_time,
            "end_time": current_time,
            "role": "mimic_user",
            "action": "respond",
            "input": last_agent_question,
            "output": reply,
            "llm": self.model,
            "input_tokens": llm_result.get("llm_metadata", {}).get("prompt_tokens", 0),
            "output_tokens": llm_result.get("llm_metadata", {}).get("completion_tokens", 0),
            "cost": 0.0,
        }
        logger.info("Generated mimic user message: %s", json.dumps(message))

        return {
            "messages": [message],
            "is_complete": is_complete,
            "raw": llm_result,
            "rendered_prompt": rendered_prompt,
        }

def conduct_interaction(mimic_user_instance: MimicUser, current_state: Dict[str, Any], agent_question: str, turn_number: int) -> Dict[str, Any]:
    # Simulating an agent question being added to the messages
    agent_message = {
        "start_time": None,
        "end_time": None,
        "role": "agent",
        "action": "question",
        "input": agent_question,
        "output": "",
        "llm": None
    }
    current_state["messages"].append(agent_message)

    # Mimic user responds
    response_from_mimic_user = mimic_user_instance.respond(current_state)

    # Add the mimic user's response to the state
    current_state["messages"].extend(response_from_mimic_user["messages"])
    current_state["is_complete"] = response_from_mimic_user["is_complete"]

    # Print statements adapted to the new message format
    print(f"=== Agent Question (Turn {turn_number}) ===")
    print(agent_question)
    print(f"\n=== Mimic User Thought (Turn {turn_number}) ===")
    print(response_from_mimic_user["raw"].get("thought", "No thought provided."))
    print(f"\n=== Mimic User Feedback (Turn {turn_number}) ===")
    print(response_from_mimic_user["messages"][-1]["output"])

    return current_state
