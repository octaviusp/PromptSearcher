import sys
import os
import matplotlib.pyplot as plt

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from prompt_searcher.core import load_dataset, Backpropagation, NaiveSimilarity, GroqAgent, ObjectivePrompt

from dotenv import load_dotenv
from os import getenv
load_dotenv()

OPENAI_API_KEY = getenv("OPENAI_API_KEY")
GROQ_API_KEY = getenv("GROQ_API_KEY")

def main():
    print("Step 1: Load dataset and initialize agents")

    dataset = load_dataset("tests/data/math.csv")

    model = GroqAgent(api_key=GROQ_API_KEY, model="llama-3.2-1b-preview")
    evaluator = GroqAgent(api_key=GROQ_API_KEY, model="llama3-8b-8192")
    augmentator = GroqAgent(api_key=GROQ_API_KEY, model="llama3-8b-8192")

    print("Step 2: Initialize score function and backpropagation")
    score_function = NaiveSimilarity(evaluator)
    backpropagation = Backpropagation(augmentator, score_function)

    print("Step 3: Set initial system message and number of epochs")
    objective_prompt = ObjectivePrompt("You are a math teacher") # -> Objective prompt to improve
    epochs = 5

    print("Step 4: Start training loop")
    best_score = 0  # Initialize with lowest possible score
    best_system_message = objective_prompt.get_last_prompt()
    score_history = []

    x_train = [row[0] for row in dataset]
    y_train = [row[1] for row in dataset]

    for i in range(epochs):
        print(f"Epoch {i+1}/{epochs}")
        y_pred = []

        current_prompt = objective_prompt.get_last_prompt()

        print("Step 4.1: Generate predictions for each prompt in the dataset")
        for prompt, response in dataset:
            print("Prompt:", prompt)
            y_pred.append(model.generate_response(current_prompt, prompt))

        print("Step 4.2: Calculate score")
        current_score = score_function.score(y_pred, y_train)

        # If it's the first epoch, set the initial prompt's score for later comparison
        if i == 0:
            objective_prompt.set_init_loss(current_score)

        print("Score:", current_score)

        # Track score and system message
        score_history.append((current_score, current_prompt))

        # Update best score and system message if current score is better
        if current_score > best_score:
            best_score = current_score
            best_prompt = current_prompt

        print("Step 4.3: Optimize system message using backpropagation")
        improved_prompt = backpropagation.optimize_prompt(current_prompt, current_score, x_train, y_pred, y_train)

        objective_prompt.update(improved_prompt, current_score)

    print("Step 5: Training complete")

    print("*"*50)    
    print("\nFinal Results:")
    print(f"Best score: {best_score}")

    print(f"Best system prompt: {objective_prompt.get_best_prompt()}")
    print("*"*50)
    print("\nScore history:")
    epochs = range(1, len(score_history) + 1)
    scores = [score for score, _ in score_history]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, scores, 'b-')
    plt.title('Score History')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.grid(True)
    plt.show()

    for epoch, (score, prompt) in enumerate(score_history):
        print(f"Epoch {epoch + 1}: Score = {score}")
        print(f"Prompt: {prompt}\n")

if __name__ == "__main__":
    main()
