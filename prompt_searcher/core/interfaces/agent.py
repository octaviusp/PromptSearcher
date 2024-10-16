class Agent:
    def __init__(self, model: str, api_key: str, client):
        self.model = model
        self.client = client(api_key=api_key)

    def generate_response(self, system_message: str, user_message: str) -> str:
        """
        Generate a response from the model.

        Args:
            system_message (str): The system message providing context to the model.
            user_message (str): The user message for which the model will generate a response.

        Returns:
            str: The response generated by the model.
        """
        raise NotImplementedError("This method should be overridden by subclasses")