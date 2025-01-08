# Extracting Explanatory Rationales of Activity Relationships using LLMs - A Comparative Analysis
This repository contains the source code and result files for the experiments described in the paper of same name.

## Abstract
Business Process Redesign (BPR) is essential for adapting processes to technological advancements, legislative changes, and sustainability standards. Despite its significance, BPR faces challenges due to limited automated support, particularly in classifying activity relationships that govern execution order. This comparative analysis investigates the use of Large Language Models (LLMs) to automate the extraction of explanatory rationales—laws, business rules, and best practices—from textual data, addressing the traditionally manual and resource-intensive retrieval process. By comparing four LLM prompting techniques (Vanilla, Few-Shot, Chain-of-Thought, and their combination), we evaluate their effectiveness in classifying relationships based on contextual origins. Our findings show that Few-Shot and Chain-of-Thought approaches significantly enhance precision, recall, and F1 scores. Furthermore, smaller, cost-effective LLMs, such as GPT-4o mini, achieved performance comparable to larger models, making advanced classification accessible to organizations with limited resources.


## Structure of the Repository
In this repository you find
* **Testing Use Case:** We developed the experimental setup based on the testing use case of the airport check-in process.
  * Process Descriptions
  * Ground Truth
  * Prompts
* **Thesis Process:** This use case scenario was used for the final experiments
  * Interview Trasncripts
  * Process Descriptions
  * Ground Truth
  * Prompts
* **Results** 
* **Python Scripts**


## Technical Details
The core components of the setup involve LLM access via the respective APIs, Python for interaction, and the RAG pipeline for context retrieval. Python 3.12 was chosen as the primary programming language due to the availability of an extensive ecosystem of libraries allowing interactions with LLMs. Specifically, the \textit{openai} package was used for OpenAI's GPT, the _anthropic_ package for Antrophic's Claude, and the Vertex AI SDK for Python\footnote{For configuration, we followed the guide provided in: https://cloud.google.com/python/docs/reference/aiplatform/latest} for Google's Gemini. To ensure storage of sensitive API keys and to safely upload the project to GitHub, the \textit{dotenv} package was utilized to load environment variables. 
