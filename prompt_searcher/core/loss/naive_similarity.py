from prompt_searcher.core.interfaces.loss import LossFunction
from prompt_searcher.core.interfaces.agent import Agent

class NaiveSimilarity(LossFunction):

    def __init__(self, model: Agent, system_message: str = None):
        self.model = model
        self.system_message = "You are an AI assistant tasked with evaluating the correctness of an answer compared to a desired answer. Your goal is to provide a score between 0 and 10 based on how correct the given answer is."
        if system_message is not None:
            self.system_message = system_message

    def score(self, y_pred: list[str], y_true: list[str]) -> float:
        total_score = 0
        for pred, true in zip(y_pred, y_true):
            user_message = f"""Given this answer: {pred}
                And this desired answer: {true}

                Provide a score based on the correctness of the answer compared to the desired answer.

                Use the following scale:
                1: Completely incorrect or irrelevant
                2: Mostly incorrect with minor relevant points
                3: Partially correct but significant errors
                4: More correct than incorrect, but still has notable mistakes
                5: Mostly correct with minor errors
                6: Correct in essence but missing some details
                7: Correct with only slight imperfections
                8: Very accurate with minimal room for improvement
                9: Nearly perfect answer
                10: Perfect answer, exactly matches the desired response

                Just answer with the score number. Your score is:"""
            response = self.model.generate_response(self.system_message, user_message)
            try:
                print("Response:", response)
                score = int(response.strip())
                total_score += score
            except ValueError:
                continue
        
    
        return total_score / len(y_pred) if y_pred else 0
