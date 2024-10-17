from prompt_searcher.core.interfaces.loss import LossFunction
from prompt_searcher.core.interfaces.agent import Agent

class Backpropagation:
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

        This method generates an optimized version of the current prompt using advanced prompt engineering techniques.
        It takes into account the current prompt, its score, and the previous prompt that didn't improve the score.

        Args:
            current_prompt (str): The current prompt to be optimized.
            score (int): The current score of the prompt based on the evaluation criteria.
            previous_prompt (str): The previous prompt that didn't improve the score.

        Returns:
            str: An optimized version of the prompt.

        The method uses the following steps:
        1. Constructs a system message and a user message with instructions for prompt optimization.
        2. Utilizes the augmentator agent to generate an optimized prompt based on these messages.
        3. Applies various prompt engineering techniques such as few-shot learning, chain of thought, task decomposition, etc.
        4. Ensures the new prompt is clear, concise, and effective in guiding the model to produce high-quality results.
        5. Removes any inconsistencies or contradictions to maintain coherence and clarity.

        The optimized prompt is returned as a string, ready for production use without any additional explanations or examples.
        """

        computed_score_natural = f"The current score of the prompt based on the criteria is: {score}"
        system_message = "You are an AI assistant tasked with improving a prompt. Your goal is to create an enhanced version of the given prompt that better aligns with the desired outputs. Ensure consistency and remove any contradictions in the system prompt."
        user_message = f"""The best current prompt is: {current_prompt}
        {computed_score_natural}

        {f"Desired Output: {self.desired_output}.\n" if self.desired_output else ""}

        {f"The previous prompt was: {previous_prompt}.\n" if previous_prompt else ""}

        The previous prompt did not improve the score. Analyze why it didn't work and create a new, better prompt. Avoid repeating the mistakes of the previous prompt.

        Please provide an optimized version of the current prompt that will generate better responses for similar types of inputs. Use advanced prompt engineering techniques to improve performance, including but not limited to:

        1. Few-shot learning: Create new example input-output pairs to guide the model, without using the actual inputs and outputs provided.
        2. Chain of thought: Break down complex reasoning into step-by-step explanations.
        3. Task decomposition: Split complex tasks into smaller, manageable subtasks.
        4. Persona adoption: Frame the prompt as if the model is adopting a specific expert role.
        5. Contextual priming: Provide relevant background information to set the right context.
        6. Output structuring: Specify the desired format or structure of the output.
        7. Self-consistency: Encourage the model to check its own work and refine its answers.

        Ensure the new prompt is clear, concise, and effectively guides the model to produce high-quality results. The optimized prompt should be versatile enough to handle various input types within the given domain.

        Remove any inconsistencies or contradictions in the system prompt to maintain coherence and clarity.

        Provide only the optimized prompt in your response, without any additional explanation or examples. The prompt must be ready to use in the final stage, without any labels or placeholders to complete. Your response should contain only the final prompt, ready for production use. Please provide the best prompt that you can.

        {f"Remember that the desired output is: {self.desired_output}" if self.desired_output else ""}

        The optimized prompt is:"""

        optimized_prompt = self.model.generate_response(system_message, user_message).strip()
        return optimized_prompt

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
