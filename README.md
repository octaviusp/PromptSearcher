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

4. Run the main script:
   ```
   poetry run python tests/main.py
   ```

## Contributing

We welcome contributions to PromptSearcher! Please follow these steps to contribute:

1. Fork the repository
2. Create a new branch for your feature or bug fix
3. Make your changes and commit them with a clear commit message
4. Push your changes to your fork
5. Create a pull request to the main repository

Please ensure your code follows the project's coding standards and includes appropriate tests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

# Author

Octavio Pavon - [octaviusp](https://github.com/octaviusp)