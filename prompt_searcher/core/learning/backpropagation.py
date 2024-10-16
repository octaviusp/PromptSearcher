from prompt_searcher.core.interfaces.loss import LossFunction
from prompt_searcher.core.interfaces.agent import Agent

class Backpropagation:
    def __init__(self, model: Agent, loss_function: LossFunction):
        self.model = model
        self.loss_function = loss_function

    def optimize_prompt(self,
                        current_prompt: str,
                        score: int,
                        x: list[str],
                        y_pred: list[str],
                        y_true: list[str]):

        computed_score_natural = f"The current score of the prompt based on the criteria is: {score}"
        system_message = "You are an AI assistant tasked with improving a prompt. Your goal is to create an enhanced version of the given prompt that better aligns with the desired outputs. Ensure consistency and remove any contradictions in the system prompt."
        user_message = f"""Current Prompt: {current_prompt}

        {computed_score_natural}

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

        Provide only the optimized prompt in your response, without any additional explanation or examples from the original inputs and outputs.

        The optimized prompt is:"""

        optimized_prompt = self.model.generate_response(system_message, user_message).strip()
        return optimized_prompt

    def _format_list(self, items: list[str]) -> str:
        return "\n".join(f"- {item}" for item in items)
