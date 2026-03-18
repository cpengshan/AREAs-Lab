from google import genai


class MimicUser:
    def __init__(self, persona, task_requirement, model):
        self.persona = persona
        self.task_requirement = task_requirement
        self.client = genai.Client(api_key="你的API_KEY")
        self.model = model
    
    def respond(self, messages_in):
        """
        messages: Contain all the messages from both AUNU agent and mimic user so far
        """
        response = self.client.models.generate_content(
            model="gemini-3-flash",
            contents="XXX"
        )
        return message_out



message_in = [
    {"sender": "aunu_agent", "content": "XXX"}
]

message_out = [
    {"sender": "aunu_agent", "content": "XXX"},
    {"sender": "mimic_user", "content": "XXX"},
]



prompt = """

You are a mimic user, here is your persona

{{ persona }}

Here is your task

{{ task requirement }}

Here is the conversation history:

{{ messages }}

You need to respond to the last message sent by AUNU agent
"""




