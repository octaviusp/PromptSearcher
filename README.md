![PromptSearcher Logo](./assets/prompt_searcher.webp)

# PromptSearcher

PromptSearcher is an automatic tool designed to find the best prompt in both supervised and unsupervised scenarios. This project draws inspiration from traditional neural network learning techniques.

## Overview

The core concept of PromptSearcher revolves around the idea of "gradients" in prompt engineering. Here, the gradients represent the best prompts that guide the results towards improved scores.

## Key Features

- Automatic prompt optimization
- Support for both supervised and unsupervised learning
- Inspired by neural network learning principles
- Gradient-based approach to prompt improvement

## How It Works

PromptSearcher iteratively refines prompts by analyzing the performance of each variation. The system identifies the most effective prompts (the "gradients") that lead to better outcomes, allowing for continuous improvement in prompt quality.

## Applications

This tool can be particularly useful in various fields where prompt engineering plays a crucial role, such as:

- Natural Language Processing (NLP)
- Conversational AI
- Content Generation
- Information Retrieval

## Getting Started

1. Install Poetry if you haven't already:
   ```
   pip install poetry
   ```

2. Clone the repository and navigate to the project directory:
   ```
   git clone https://github.com/octaviusp/PromptSearcher.git
   cd promptsearcher
   ```

3. Install the project dependencies:
   ```
   poetry install
   ```

4. Create a `.env` file in the project root directory and add your API keys:
   ```
   GROQ_API_KEY=your_groq_api_key
   OPENAI_API_KEY=your_openai_api_key
   ```

5. Run the main script:
   ```
   poetry run python prompt_searcher.py
   ```

## Example Usage

Here is an example of how to use PromptSearcher in your project:
