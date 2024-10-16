from prompt_searcher.core.interfaces.agent import Agent
from openai import OpenAI

class OpenAIAgent(Agent):
    def __init__(self, model: str, api_key: str, **kwargs):
        self.model = model
        self.client = OpenAI(api_key=api_key, **kwargs)

    def generate_response(self, system_message: str, user_message: str) -> str:
        """
        Generate a response from the OpenAI model.

        Args:
            system_message (str): The system message providing context to the model.
            user_message (str): The user message for which the model will generate a response.

        Returns:
            str: The response generated by the model.
        """
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
        )
        return completion.choices[0].message.content

# Example usage:
# agent = OpenAIAgent(model="gpt-4o-mini", api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as an env var>"))
# response = agent.generate_response("System message", "User message")
# print(response)