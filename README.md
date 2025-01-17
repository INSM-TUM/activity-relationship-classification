# Extracting Explanatory Rationales of Activity Relationships using LLMs - A Comparative Analysis
This repository contains the source code and result files for the experiments described in the paper of same name.

## Abstract
Business Process Redesign (BPR) is essential for adapting processes to technological advancements, legislative changes, and sustainability standards. Despite its significance, BPR faces challenges due to limited automated support, particularly in classifying activity relationships that govern execution order. This comparative analysis investigates the use of Large Language Models (LLMs) to automate the extraction of explanatory rationales—laws, business rules, and best practices—from textual data, addressing the traditionally manual and resource-intensive retrieval process. By comparing four LLM prompting techniques (Vanilla, Few-Shot, Chain-of-Thought, and their combination), we evaluate their effectiveness in classifying relationships based on contextual origins. Our findings show that Few-Shot and Chain-of-Thought approaches significantly enhance precision, recall, and F1 scores. Furthermore, smaller, cost-effective LLMs, such as GPT-4o mini, achieved performance comparable to larger models, making advanced classification accessible to organizations with limited resources.

## Overview

The Activity Relationship Classifier is a Python-based tool designed to classify the relationships between pairs of activities in a business process. The classification is based on predefined categories and can also identify dependencies due to the law of nature.

## Features

- Classifies relationships into contextual origin categories (Governmental Law, Best Practice, Business Rule), and classifies Laws of Nature.
- Supports multiple models for classification, including OpenAI, Anthropic, and Vertex AI models.
- Utilizes different methods for classification: vanilla prompting, chain-of-thought, few shot learning, and a mix of few shot learning with chain-of-thought.
- Optionally uses Retrieval-Augmented Generation (RAG) for context retrieval.

## Requirements

- Python 3.8 or higher
- Required Python packages (listed in `requirements.txt`)

## Installation

1. Clone the repository:
    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

3. [If you want to run the VertexAI models] Setup Authentication for VertexAI by following [this guide](https://cloud.google.com/vertex-ai/generative-ai/docs/start/quickstarts/quickstart-multimodal#python).

4. Set up the necessary environment variables by creating a `.env` file in the project root directory and adding the API keys for the models you want to run:
    ```env
    OPENAI_API_KEY=<your_openai_api_key>
    ANTHROPIC_API_KEY=<your_anthropic_api_key>
    REPLICATE_API_TOKEN=<your_replicate_api_token> # For RAG embeddings but can also be done locally
    VERTEXAI_PROJECT_NAME=<your vertexai project id>
    VERTEXAI_LOCATION=<location in which you want requests to vertexai to be processed>
    ```

## Usage

### Command-Line Interface

Run the classifier using the command-line interface:

```sh
python main.py --folder_name <folder_name> [--model <model>] [--method <method>] [--rag | --no-rag] [--path <path>]
```

#### Arguments

- `--folder_name` (string, required): The folder name containing the input files (`desc.txt`, `activities.txt`, `truth.txt`, `examples.txt`), results folder will be created in this folder. The folder is best named according to the process name.
- `--model` (string, optional): The model to use for classification. Defaults to `claude-3-5-haiku-latest`. Check `constants.py` for supported models.
- `--method` (string, optional): The method to use for classification. Defaults to `vanilla`. Check `constants.py` for supported methods.
- `--rag` (optional): Whether to use RAG for context retrieval. Defaults to `True`. Use `--no-rag` to disable.
- `--path` (string, optional): The path to the parent folder of the folder_name containing the input files. Defaults to the same directory as the Python file.

### Example

```sh
python main.py --folder_name example_folder --model gpt-4o --method few-cot --no-rag
```

### Input Files

The input folder **must** contain the following files. An example is provided in the `thesis_process` directory in the repository:

- `desc.txt`: A text file containing the process description.
- `activities.txt`: A text file containing the list of activities, one per line.
- `truth.csv`: A CSV file containing the ground truth for evaluation. Check the `thesis_process` folder for the structure.
- `examples (few).txt`: (Only necessary for methods with few shot learning) A text file containing 3-5 examples of correct labelling for few shot learning, the answer to the prompts should be JSON formatted
- `examples (few-cot).txt`: (Only necessary for methods with few shot learning) A text file containing 3-5 examples of correct labelling for few shot learning, the answer to the prompts should be JSON formatted for `few` and in natural language for `few-cot`.

### Output

The classifier results will be saved in a CSV file in the `results` directory that is by default stored inside a `results` subfolder of the input folder. The result file name will include the tsted model and method as well as the current timestamp.

### Evaluation

To evaluate the performance of the classifier output, use the `calculate_stats` method from the [`stats.py`](./stats.py) file.
For an example usage, refer to [`experiments.py`](./experiments.py)

## Structure of this Repository

### Code Files
- `main.py`: Main script to run the classifier.
- `prompts.py`: Defines the text blocks from which used prompts are built.
- `experiments.py`: Utility method to calculate classifier performance.
- `stats.py`: Contains functions for calculating evaluation metrics.
- `constants.py`: Contains constants used in the project.
- `rag.py`: Contains the RAG implementation (if applicable).

### Example Data
- `thesis_process/`: Contains example files for the thesis registration and submission process.
    - `interview_transcripts/` : Contains the interview transcripts used as basis for the process description, as described in the paper.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact

For any questions or issues, please contact `touqanzahi@gmail.com`.
