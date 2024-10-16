from typing import List, Tuple
import traceback
from prompt_searcher.core import (
    load_dataset,
    load_unsupervised_dataset,
    Backpropagation,
    ObjectivePrompt,
    LossFunction,
    Agent
)
import matplotlib.pyplot as plt

class PromptSearch:
    def __init__(
        self,
        dataset_path: str,  # Path to the dataset file
        student: Agent,  # Agent used as the student model
        loss_function: LossFunction,  # Function to calculate loss/score
        backpropagation: Backpropagation,  # Backpropagation algorithm
        objective_prompt: ObjectivePrompt,  # Initial objective prompt
        epochs: int = 5,  # Number of training epochs
        verbose: bool = True,
    ):
        """
        Initialize the PromptSearch class.

        Args:
            dataset_path (str): Path to the dataset file.
            evaluator (Agent): Agent used for evaluation.
            augmentator (Agent): Agent used for augmentation.
            student (Agent): Agent used as the student model.
            loss_function (LossFunction): Function to calculate loss/score.
            backpropagation (Backpropagation): Backpropagation algorithm.
            objective_prompt (ObjectivePrompt): Initial objective prompt.
            epochs (int, optional): Number of training epochs. Defaults to 5.
            verbose (bool, optional): Whether to print progress information. Defaults to True.
        """
        try:
            self.student = student
            self.epochs = epochs
            self.dataset_path = dataset_path
            self.verbose = verbose

            self.dataset = load_dataset(self.dataset_path)
                
            self.score_function = loss_function
            self.backpropagation = backpropagation
            self.objective_prompt = objective_prompt
            
            self.x_train = [row[0] for row in self.dataset]
            self.y_train = [row[1] for row in self.dataset]
            
            self.score_history = []
            self.best_score = 0
            self.best_prompt = self.objective_prompt.get_best_prompt()
        except Exception as e:
            if self.verbose:
                print(f"Error initializing PromptSearch: {str(e)}")
                print(traceback.format_exc())
            raise

    def train(self):
        """
        Trains the prompt search model for the specified number of epochs.

        This method iterates through the epochs, generating predictions for each prompt
        in the dataset using the current objective prompt. It then calculates the score,
        updates the score history, and optimizes the prompt using backpropagation.

        The method keeps track of the best score and corresponding prompt throughout
        the training process.

        Prints progress information for each epoch, including the current score,
        current prompt, and improved prompt.
        """
        try:
            for i in range(self.epochs):
                if self.verbose:
                    print("*"*100)
                    print(f"Epoch {i+1}/{self.epochs}")
                y_pred = []
                current_prompt = self.objective_prompt.get_last_prompt()
                
                if self.verbose:
                    print(f"****\nTesting prompt: {current_prompt}\n****")
                for input_prompt, _ in self.dataset:
                    try:
                        response = self.student.generate_response(current_prompt, input_prompt)
                        y_pred.append(response)
                    except Exception as e:
                        if self.verbose:
                            print(f"Error generating response: {str(e)}")
                            print(traceback.format_exc())
                
                try:
                    current_score = self.score_function.score(y_pred, self.y_train)
                    if self.verbose:
                        print(f"Score: {current_score}")
                    
                    self.score_history.append(current_score)
                    self.objective_prompt.put_loss_to_last_prompt(current_score)
                except Exception as e:
                    if self.verbose:
                        print(f"Error calculating score: {str(e)}")
                        print(traceback.format_exc())
                    continue
                
                previous_prompt = current_prompt

                if self.score_function.winner(self.best_score, current_score):
                    if self.verbose:
                        print(f"****\nNew best score: {current_score} with prompt: {current_prompt}\n****")
                    self.best_score = current_score
                    self.best_prompt = current_prompt
                    previous_prompt = None
                elif self.verbose:
                    print(f"- No improvement with this prompt.")

                try:
                    improved_prompt = self.backpropagation.optimize_prompt(
                        self.best_prompt, self.best_score, previous_prompt=previous_prompt 
                    )
                    self.objective_prompt.update(improved_prompt)
                except Exception as e:
                    if self.verbose:
                        print(f"Error optimizing prompt: {str(e)}")
                        print(traceback.format_exc())
        except Exception as e:
            if self.verbose:
                print(f"Error during training: {str(e)}")
                print(traceback.format_exc())

    def get_best_prompt(self) -> str:
        return self.best_prompt
    
    def get_best_score(self) -> float:
        return self.best_score
    
    def get_results(self) -> Tuple[str, float]:
        return self.best_prompt, self.best_score

    def plot_score_history(self, figsize: Tuple[int, int] = (10, 6)):
        """
        Plots the score history.

        This method creates a line plot of the score history, showing the evolution
        of the score over the epochs.

        The plot is displayed with a title, x-axis labeled 'Epoch', y-axis labeled 'Score',
        and a grid for better readability.
        """
        try:
            plt.figure(figsize=figsize)
            plt.plot(range(1, len(self.score_history) + 1), self.score_history, 'b-')
            plt.title('Score History')
            plt.xlabel('Epoch')
            plt.ylabel('Score')
            plt.grid(True)
            plt.show()
        except Exception as e:
            if self.verbose:
                print(f"Error plotting score history: {str(e)}")
                print(traceback.format_exc())

    def print_score_history(self):
        """
        Prints the score history.

        This method prints the score history, showing the score for each epoch.

        The output is formatted with the epoch number and the corresponding score.
        """
        if self.verbose:
            try:
                for epoch, score in enumerate(self.score_history, 1):
                    print(f"Epoch {epoch}: Score = {score}")
            except Exception as e:
                print(f"Error printing score history: {str(e)}")
                print(traceback.format_exc())
