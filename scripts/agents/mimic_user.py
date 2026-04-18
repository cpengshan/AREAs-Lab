import json
import logging
import os
import re
from typing import Dict, Any
from jinja2 import Environment, FileSystemLoader
import datetime
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model.model_base import LLM

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
        self.llm = LLM(model)

    def _render_prompt(self, state: Dict[str, Any]) -> str:
        last_agent_message = next(
            (m for m in reversed(state.get("messages", [])) if m["role"] == "aunu_agent"), None
        )
        context = {
            "user_profile": self.persona,
            "gold_task_requirement": self.task_requirement_gold,
            "init_user_instruction": self.user_instruction_init,
            "communication_style_init": self.communication_style_init,
            "agent_question": last_agent_message["output"] if last_agent_message else "",
        }
        return self.template.render(context)

    def _call_llm(self, rendered_prompt: str) -> Dict[str, Any]:
        response = self.llm.generate(rendered_prompt)
        raw_output = response["output"]

        # Parse JSON from LLM output — strip markdown code fences if present
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_output.strip())
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            # Fallback: treat entire output as feedback
            parsed = {"thought": "", "feedback": raw_output}

        return {
            "feedback": parsed.get("feedback", raw_output),
            "thought": parsed.get("thought", ""),
            "is_complete": False,
            "input_tokens": response.get("input_tokens", 0),
            "output_tokens": response.get("output_tokens", 0),
            "cost": response.get("cost", 0.0),
        }

    def respond(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if state.get("is_complete", False):
            return {"messages": [], "is_complete": True, "raw": {}}

        current_time = datetime.datetime.now().isoformat()

        # First turn: no prior agent message — send user_instruction_init directly
        last_agent_message = next(
            (m for m in reversed(state.get("messages", [])) if m["role"] == "aunu_agent"), None
        )
        if last_agent_message is None:
            message = {
                "start_time": current_time,
                "end_time": current_time,
                "role": "mimic_user",
                "action": "init",
                "input": "",
                "prompt_template": "",
                "output": self.user_instruction_init,
                "llm": None,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost": 0.0,
            }
            logger.info("MimicUser init message: %s", json.dumps(message))
            return {
                "messages": [message],
                "is_complete": False,
                "raw": {},
                "rendered_prompt": "",
            }

        # Subsequent turns: respond to the agent's last question via LLM
        rendered_prompt = self._render_prompt(state)
        llm_result = self._call_llm(rendered_prompt)

        reply = (
            llm_result.get("feedback")
            or llm_result.get("reply")
            or "I haven't thought that far ahead yet."
        )
        is_complete = bool(llm_result.get("is_complete", False))

        message = {
            "start_time": current_time,
            "end_time": current_time,
            "role": "mimic_user",
            "action": "respond",
            "input": last_agent_message["output"],
            "prompt_template": os.path.basename(self.template_path),
            "output": reply,
            "llm": self.model,
            "input_tokens": llm_result.get("input_tokens", 0),
            "output_tokens": llm_result.get("output_tokens", 0),
            "cost": llm_result.get("cost", 0.0),
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
