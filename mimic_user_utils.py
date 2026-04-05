import json
import logging
from typing import Any, Dict, List, Optional
from jinja2 import Environment, FileSystemLoader
from openai import OpenAI
import os

logger = logging.getLogger(__name__)


class MimicUser:
    def __init__(
        self,
        persona: Dict[str, Any],
        task_requirement_gold: Any,
        user_instruction_init: Any,
        communication_style_init: Optional[Dict[str, Any]] = None,
        model: str = "gpt-4.1-mini",
        template_path: str = "/mnt/data/feedback.jinja",
    ):
        self.persona = persona
        self.task_requirement_gold = task_requirement_gold
        self.user_instruction_init = user_instruction_init
        self.communication_style_init = communication_style_init or {}
        self.model = model
        self.client = OpenAI()

        template_dir = os.path.dirname(template_path) or "."
        template_name = os.path.basename(template_path)
        self.env = Environment(loader=FileSystemLoader(template_dir))
        self.prompt_template_user_feedback = self.env.get_template(template_name)

    def _get_last_agent_question(self, messages: List[Dict[str, Any]]) -> str:
        for msg in reversed(messages):
            if msg.get("role") in ("agent", "assistant"):
                return msg.get("content", "")
        return ""

    def _render_prompt(self, state: Dict[str, Any]) -> str:
        messages = state.get("messages", [])
        last_agent_msg = self._get_last_agent_question(messages)

        template_vars = {
            "user_profile": self.persona,
            "init_user_instruction": self.user_instruction_init,
            "gold_task_requirement": self.task_requirement_gold,
            "communication_style": self.communication_style_init,
            # support both names so old template keeps working
            "aunu_question": last_agent_msg,
            "agent_question": last_agent_msg,
            "chat_history": messages,
        }
        return self.prompt_template_user_feedback.render(template_vars)

    def _safe_parse_json(self, text: str) -> Dict[str, Any]:
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:].strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start : end + 1])
            raise ValueError(f"Invalid JSON output from model:\n{text}")

    def _call_llm(self, rendered_prompt: str) -> Dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a mimic user simulator. Always return valid JSON with at least "
                        "a feedback field. You may also return thought and is_complete."
                    ),
                },
                {"role": "user", "content": rendered_prompt},
            ],
        )
        return self._safe_parse_json(response.choices[0].message.content)

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

        message = {"role": "mimic_user", "content": reply}
        logger.info("Generated mimic user message: %s", json.dumps(message))

        return {
            "messages": [message],
            "is_complete": is_complete,
            "raw": llm_result,
            "rendered_prompt": rendered_prompt,
        }




def conduct_interaction(mimic_user_obj, current_state, agent_question_text, turn_number):

    current_state["messages"].append({
        "role": "agent",
        "content": agent_question_text
    })
    print(f"\n=== Agent Question (Turn {turn_number}) ===")
    print(agent_question_text)

    # Mimic User response
    response = mimic_user_obj.respond(current_state)

    # Extract and print thought & feedback from the raw LLM result
    if response and response["raw"]:
        llm_result = response["raw"]
        if "thought" in llm_result:
            print(f"\n=== Mimic User Thought (Turn {turn_number}) ===")
            print(llm_result["thought"])
        if "feedback" in llm_result:
            print(f"\n=== Mimic User Feedback (Turn {turn_number}) ===")
            print(llm_result["feedback"])

    # Updated status
    current_state["messages"] = list(current_state["messages"]) + list(response["messages"])
    current_state["is_complete"] = response["is_complete"]

    return current_state






