class ObjectivePrompt:
    def __init__(self, initial_prompt: str):
        self.current_prompt = initial_prompt
        self.history = [(initial_prompt, None)]  # (prompt, loss_score)

    def __repr__(self) -> str:
        return self.current_prompt
    
    def __str__(self) -> str:
        return self.current_prompt

    def put_loss_to_last_prompt(self, loss_score: float) -> None:
        """
        Put the loss score to the last prompt.

        Args:
            loss_score (float): The loss score associated with the last prompt.
        """
        self.history[-1] = (self.history[-1][0], loss_score)

    def update(self, new_prompt: str) -> None:
        """
        Update the current prompt and maintain a history of previous prompts with their loss scores.

        Args:
            new_prompt (str): The new system prompt to be set as current.
            loss_score (float): The loss score associated with the previous prompt.
        """
        self.current_prompt = new_prompt
        self.history.append((new_prompt, 0))

    def get_last_prompt(self) -> str:
        """
        Get the last system prompt.

        Returns:
            str: The current system prompt.
        """
        return self.current_prompt

    def get_history(self) -> list:
        """
        Get the history of all prompts and their associated loss scores.

        Returns:
            list: A list of tuples containing (prompt, loss_score).
        """
        return self.history

    def get_best_prompt(self) -> tuple:
        """
        Get the prompt with the max loss score.

        Returns:
            tuple: A tuple containing (best_prompt, best_loss_score).
        """
        valid_history = [(prompt, score) for prompt, score in self.history if score is not None]
        if not valid_history:
            return None
        return max(valid_history, key=lambda x: x[1])
