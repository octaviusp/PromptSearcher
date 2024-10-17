from prompt_searcher.core.interfaces.loss import LossFunction
from prompt_searcher.core.interfaces.agent import Agent

class ProgressiveBackpropagation:
    def __init__(self, augmentator: Agent, desired_output: str = None):
        """
        Initialize the Backpropagation class.

        Args:
            augmentator (Agent): An agent used for augmenting and optimizing prompts.
            desired_output (str, optional): The desired output for the prompt optimization. Defaults to None.

            Desired output is optional, but it can be used to guide the optimization process towards a specific output.
        """
        self.model = augmentator
        self.desired_output = desired_output

    def optimize_prompt(self,
                        current_prompt: str,
                        score: int,
                        previous_prompt: str):
        """
        Optimize the given prompt based on the current score and previous prompt.

        This method generates a progressively improved version of the current prompt using prompt engineering techniques.
        It takes into account the current prompt, its score, and the previous prompt that didn't improve the score.

        Args:
            current_prompt (str): The current prompt to be optimized.
            score (int): The current score of the prompt based on the evaluation criteria.
            previous_prompt (str): The previous prompt that didn't improve the score.

        Returns:
            str: A progressively improved version of the prompt.

        The method uses the following steps:
        1. Constructs a system message and a user message with instructions for prompt optimization.
        2. Utilizes the augmentator agent to generate an improved prompt based on these messages.
        3. Applies prompt engineering techniques to make incremental improvements.
        4. Ensures the new prompt is clear, concise, and effective in guiding the model to produce better results.
        5. Removes any inconsistencies or contradictions to maintain coherence and clarity.

        The improved prompt is returned as a string, ready for production use without any additional explanations or examples.
        """

        computed_score_natural = f"The current score of the prompt based on the criteria is: {score}"
        system_message = "You are an AI assistant tasked with progressively improving a prompt. Your goal is to create a slightly enhanced version of the given prompt that better aligns with the desired outputs. Ensure consistency and remove any contradictions in the system prompt."
        user_message = f"""The current prompt is: {current_prompt}
        {computed_score_natural}

        {f"Desired Output: {self.desired_output}.\n" if self.desired_output else ""}

        {f"The previous prompt was: {previous_prompt}.\n" if previous_prompt else ""}

        The previous prompt did not improve the score significantly. Analyze why it didn't work and create a slightly better prompt. Make small, incremental improvements while avoiding the mistakes of the previous prompt.

        Please provide a minimally improved version of the current prompt that will generate slightly better responses for similar types of inputs. Use subtle prompt engineering techniques to make small enhancements, such as:

        1. Slightly refining instructions or context
        2. Gently clarifying any mildly ambiguous parts of the prompt
        3. Carefully incorporating a bit more relevant domain knowledge
        4. Making minor adjustments to the tone or style to better suit the task

        Ensure the new prompt is clear, concise, and guides the model to produce marginally better results. The improved prompt should maintain versatility while making very small enhancements.

        Remove any minor inconsistencies or subtle contradictions in the system prompt to maintain coherence and clarity.

        Provide only the slightly improved prompt in your response, without any additional explanation or examples. The prompt must be ready to use in the final stage, without any labels or placeholders to complete. Your response should contain only the final prompt, ready for production use.

        {f"Remember that the desired output is: {self.desired_output}" if self.desired_output else ""}

        The minimally improved prompt is:"""

        improved_prompt = self.model.generate_response(system_message, user_message).strip()
        return improved_prompt

    def _format_list(self, items: list[str]) -> str:
        """
        Format a list of items into a string with each item on a new line, prefixed with a hyphen.

        Args:
            items (list[str]): A list of strings to be formatted.

        Returns:
            str: A formatted string with each item on a new line, prefixed with a hyphen.

        Example:
            Input: ['apple', 'banana', 'cherry']
            Output: "- apple\n- banana\n- cherry"
        """
        return "\n".join(f"- {item}" for item in items)
