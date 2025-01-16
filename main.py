import argparse
import json
import pathlib
import textwrap
import time
from datetime import datetime
from typing import Any

import openai
import vertexai
from anthropic import Anthropic
from dotenv import load_dotenv
from vertexai.generative_models import GenerativeModel, Part, Content
from ollama import Client

from datetime import datetime

from constants import *
from rag import Rag

load_dotenv()

def log(message: str):
    print(f'[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] '+message)

def _create_first_message(activity1: str, activity2: str, context) -> list:
    return [{"role": "user", "content": f"Apart from the law of nature, which of the categories best describes the contextual origin of why {activity1} occurs before {activity2}? Explain why you chose this category and not another one. If none of the categories apply to the relationship, explain why it is not an instance of any of the categories. After discussing the contextual origin, discuss if the ordering is due to a law of nature.\nContext:\n{context}"}]

def _create_second_message() -> list:
    return [
        {
            "role": "user",
            "content": textwrap.dedent("""Structure your answer in the following format without any additional text, and replace the placeholders with the correct values:
                {
                    "First Activity": "-",
                    "Second Activity": "-",
                    "Category": "-",
                    "Justification": "-",
                    "Law of Nature": "-"
                }

                If none of the three contextual origin categories apply to the relationship, put a dash in the Category and Justification fields and do not include any other text for justifying your decision. Otherwise, put the category you chose in the "Category" field, the justification for your choice in the "Justification" field. In the "Law of Nature" field, if the answer is yes, then you should put justification in the value, if it is no, then only put a single dash.""")
        }
    ]

class Classifier:
    """
    A class to classify relationships between activities based on a given process description.

    Attributes:
        model (str): The model to use for classification.
        method (str): The method to use for classification.
        is_rag (bool): Whether to use RAG for context retrieval.
        # truth_file (str): The path to the truth file.
        activities (list): A list of activities to classify.
        system_prompt (str): The system prompt for the model.
        context (str): The context for classification.
        result_file (file object): The file to write the classification results.

    Methods:
        classify_relationships(): Classifies the relationships between activities.
    """
    def __init__(self, input_folder: pathlib.Path | str, model: str, method: str, rag: bool, vertexai_project_name: str = None, vertexai_location: str = None):
        if isinstance(input_folder, str):
            input_folder = pathlib.Path(input_folder)
        if not input_folder.exists():
            raise ValueError("Folder not found")

        self.model = model
        self.platform = get_platform(model)
        self.method = method
        self.is_rag = rag

        # if not (folder_name / "truth.csv").exists():
        #     raise ValueError("Truth file not found")
        # self.truth_file = folder_name / "truth.csv"

        activities_path = input_folder / "activities.txt"
        if not activities_path.exists():
            raise ValueError("Activities file not found")
        with open(activities_path, "r", encoding="utf-8") as f:
            self.activities = [line.strip() for line in f if line.strip()]

        self._validate_method()

        example_path = input_folder / "examples.txt"
        if not example_path.exists() and method in ["few", "few-cot"]:
            raise ValueError("Example file not provided for few-shot learning")
        self._build_system_prompt(example_path)

        self._initialize_clients(vertexai_project_name, vertexai_location)

        process_desc_path = input_folder / "desc.txt"
        if not process_desc_path.exists():
            raise ValueError("Process description file not provided")
        self._load_process_description(process_desc_path)

        if self.is_rag:
            self._init_rag(input_folder)
        self._create_result_path(input_folder)

    def _build_system_prompt(self, example_path: pathlib.Path):
        base_text = textwrap.dedent("""
        You are an assistant business process re-designer. Your job is to explain the context behind the ordering of a pair of activities, given the pair of activities and the process description, by categorizing the reason of the specific order in zero or one of the three following categories:
        1- Governmental Law: Rules created and enforced by governmental institutions to regulate business behaviour (i.e. Customer cannot cash a cheque without validating their documents).
        2- Best Practice: Procedures usually accepted by the organization's staff or industry-wide to be superior to alternatives, but are not required to be followed nor enforced by any stakeholder (i.e. Following up with patients after treatment).
        3- Business Rule: Rules that are under full jurisdiction of the stakeholders of the process (i.e. organization or suppliers) that can change or discard this rule at their own discretion (i.e. holding regular meetings after starting project).
    
        Separate from the other categories, you need to decide if the relationship is due to a law of nature, which is an inviolable relationship where the second activity cannot precede the second activity due to either a deadlock occurring or due to a data (i.e. You cannot reply to a message without receiving it), resource dependency (i.e. You cannot print a document without having paper), or logical dependency from the first activity.""").lstrip("\n")

        vanilla_text = textwrap.dedent("""
        Structure your answer in the following format without any additional text, and replace the placeholders with the correct values:
        {
            "First Activity": "-",
            "Second Activity": "-",
            "Category": "-",
            "Justification": "-",
            "Law of Nature": "-"
        }
        If none of the categories apply to the relationship, put a dash in the Category and Justification fields and do not include any other text for justifying your decision. Otherwise, put the category you chose in the "Category" field, the justification for your choice in the "Justification" field. If the relationship is an instance of a Law of Nature, you should provide justification in the "Law of Nature" field for why the relationship is due to a law of nature, if not, you should put a dash in the field.
        You will receive the prompt as "What is the relationship between [First Activity] and [Second Activity]?", the first activity always occurs in time before the second activity. Return only the JSON response with no other text outside the JSON.""")

        self.system_prompt = base_text
        if self.method in ["vanilla", "few"]:
            self.system_prompt += "\n" + vanilla_text
        if self.method in ["few", "few-cot"]:
            with open(example_path, "r", encoding="utf-8") as f:
                self.system_prompt += "\n" + f.read()

    def _validate_method(self):
        if self.method not in methods:
            raise ValueError(f"Method {self.method} not supported")

    def _initialize_clients(self, vertexai_project_name: str = None, vertexai_location: str = None):
        if self.platform == Platforms.OPEN_AI:
            self.openai_client = openai.OpenAI()
        elif self.platform == Platforms.ANTHROPIC:
            self.claude_client = Anthropic()
        elif self.platform == Platforms.VERTEX:
            if not vertexai_project_name:
                raise ValueError("Project name not provided for Vertex AI, must be provided for Vertex AI models")
            if not vertexai_location:
                raise ValueError("Location not provided for Vertex AI, must be provided for Vertex AI models")
            vertexai.init(project=vertexai_project_name, location=vertexai_location)
        elif self.platform == Platforms.OLLAMA:
            self.ollama_client = Client(
                host='http://localhost:11434', # TODO put URL in env file
                headers={'Content-Type': 'application/json'}
            )
        if self.model not in all_models:
            raise ValueError("Model not supported")

    def _load_process_description(self, process_desc_path: pathlib.Path):
        with open(process_desc_path, encoding="utf8", mode="r") as f:
            process_desc = f.read()
            self.process_desc = process_desc

    def _create_result_path(self, input_folder: pathlib.Path):
        # (absolute_path / "results").mkdir(parents=True, exist_ok=True)
        now = datetime.now()
        dt_string = now.strftime("%d%m%Y-%H%M%S")
        result_file_name = input_folder.name + "-" + dt_string
        folder_name = self.model.replace(":", "-")
        if self.is_rag:
            (input_folder / "results" / self.method / "rag" / folder_name).mkdir(parents=True, exist_ok=True)
            self.result_file = (input_folder / "results" / self.method / "rag" / folder_name / (result_file_name + ".csv")).open("x")
        else:
            (input_folder / "results" / self.method / folder_name).mkdir(parents=True, exist_ok=True)
            self.result_file = (input_folder / "results" / self.method / folder_name / (result_file_name + ".csv")).open("x")

    def _init_rag(self, input_folder: pathlib.Path):
        self.rag_engine = Rag(input_folder.name)
        self.rag_engine.load_embeddings(self.process_desc)

    def classify_relationships(self):
        if self.method in ["vanilla", "few"]:
            self._classify_normal()
        elif self.method in ["cot", "few-cot"]:
            self._classify_cot()

        # stats.calculate_stats(self.truth_file, self.result_file.name, method=self.method)

    def _classify_normal(self):
        self._write_result_header()
        for i in range(len(self.activities)):
            for j in range(i + 1, len(self.activities)):
                context = self._get_context([self.activities[i], self.activities[j]])
                messages = [{"role": "user", "content": f"What is the relationship between {self.activities[i]} and {self.activities[j]}?\nContext:\n{context}"}]
                parse = self._get_model_response(messages, self.system_prompt)
                parse = self._parse_response(parse)
                self._write_result_row(parse, self.activities[i], self.activities[j])
            self.result_file.flush()

    def _classify_cot(self):
        self._write_result_header()
        for i in range(len(self.activities)):
            for j in range(i + 1, len(self.activities)):
                context = self._get_context([self.activities[i], self.activities[j]])
                messages = _create_first_message(self.activities[i], self.activities[j], context)
                cot = self._get_model_response(messages, self.system_prompt)
                messages.append({"role": "assistant", "content": cot})
                messages.extend(_create_second_message())
                parse = self._get_model_response(messages, self.system_prompt)
                parse = self._parse_response(parse)
                self._write_result_row(parse, self.activities[i], self.activities[j])
            self.result_file.flush()

    def _write_result_header(self):
        self.result_file.write("sep=;\nFirst Activity;Second Activity;" + ";".join(categories) + f";{lon}\n")

    def _get_context(self, activities: list) -> list | str:
        if self.is_rag:
            return self.rag_engine.return_related(activities)
        return self.process_desc

    def _get_model_response(self, messages: list, system_prompt : str = None) -> Any:
        if self.platform == Platforms.OPEN_AI:
            ret = self._call_openai(messages, system_prompt)
        elif self.platform == Platforms.ANTHROPIC:
            ret = self._call_anthropic(messages, system_prompt)
        elif self.platform == Platforms.VERTEX:
            ret = self._call_vertex(messages, system_prompt)
        elif self.platform == Platforms.OLLAMA:
            ret = self._call_ollama(messages, system_prompt)
        else:
            raise ValueError("Model not supported")
        # print(ret + "\n" + "-" * 100)
        return ret

    def _write_result_row(self, parse: dict[str, str], activity1: str, activity2: str):
        self.result_file.write(f"{activity1};{activity2};")
        for category in categories:
            self.result_file.write(f"{parse.get('Justification', '-') if category == parse.get('Category') else '-'};")
        self.result_file.write(f"{parse.get('Law of Nature')}\n")

    def _call_openai(self, messages: list[dict], temperature: float = 0, system_prompt : str = None) -> Any:
        if system_prompt:
            messages.insert(0,{"role": "system", "content": system_prompt})
        completion = self.openai_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        return completion.choices[0].message.content

    def _call_anthropic(self, messages: list[dict], temperature: float = 0, system_prompt : str = None) -> Any:
        completion = self.claude_client.messages.create(
            model=self.model,
            system=system_prompt,
            messages=messages,
            temperature=temperature,
            max_tokens=1024
        )
        return completion.content[0].text

    def _call_vertex(self, messages: list[dict], temperature: float = 0, system_prompt : str = None) -> Any:
        model_gen = GenerativeModel(
            model_name=self.model,
            system_instruction=[system_prompt]
        )
        chat = model_gen.start_chat() if len(messages) == 1 else model_gen.start_chat(history=self._create_history(messages))
        for _ in range(5):
            try:
                response = chat.send_message(
                    content=messages[-1]["content"],
                    generation_config={
                        "max_output_tokens": 1024,
                        "temperature": temperature,
                        "top_p": 0.95,
                    },
                    safety_settings=safety_settings,
                    stream=False,
                )
            except Exception as e:
                print(e)
                time.sleep(15)
                continue
            else:
                break
        else:
            raise Exception("Failed to send message to vertex AI")
        return response.text
    
    def _call_ollama(self, messages: list[dict], system_prompt : str = None) -> Any:

        # print('Ill now call ollama')
        # print(messages)
        if system_prompt:
            messages.insert(0,{"role": "system", "content": system_prompt})

        # print('Messages:'+str(messages))
        response = self.ollama_client.chat(model='llama3', messages=messages)

        # print(self.model + ": " + response.message.content)
        return response.message.content

    def _create_history(self, messages: list[dict]) -> list:
        history = []
        for message in messages[:-1]:
            role = message["role"] if message["role"] == "user" else "model"
            content = message["content"]
            history.append(Content(role=role, parts=[Part.from_text(content)]))
        return history

    def _parse_response(self, response: str) -> dict:
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return self._fix_json(response)

    def _fix_json(self, response: str) -> dict:
        messages = [{"role": "user", "content": f"Could you check if this JSON string is valid and properly escaped, if not, please fix it and return it to me without any additional text. Make sure it's valid to be parsed in python\n{response}"}]

        fixed_response = self._get_model_response(messages) # No system prompt!
        return json.loads(fixed_response)

    def __del__(self):
        if self.result_file:
            self.result_file.close()

def main():
    """
    Main function to classify relationships between activities.

    This function sets up the argument parser to accept command-line arguments, parses the arguments,
    and initializes the Classifier class with the provided arguments. It then calls the classify_relationships
    method to perform the classification.

    Command-line arguments:
    --folder_name (str, required): The folder name containing the following input files: desc.txt, activities.txt, truth.txt.
    --model (str, optional): The model to use for classification. Defaults to 'claude-3-5-haiku-latest'. Check constants.py for supported models.
    --method (str, optional): The method to use for classification. Defaults to 'vanilla'. Check constants.py for supported methods.
    --rag (bool, optional): Whether to use RAG for context retrieval. Defaults to True. Use --no-rag to disable.
    --path (str, optional): The path to the folder containing input files. Defaults to the same directory as the Python file.
    --vertexai_project_name (str, optional): The Vertex AI project name.
    --vertexai_location (str, optional): The Vertex AI location.

    Returns:
    None
    """
    parser = argparse.ArgumentParser(description="Classify relationships between activities.")
    parser.add_argument("--folder_name", required=True, type=str, help="The folder name containing the following input files: desc.txt, activities.txt, truth.txt")
    parser.add_argument("--model", type=str, default="claude-3-5-haiku-latest", help="The model to use for classification. Check constants.py for supported models.")
    parser.add_argument("--method", type=str, default="vanilla", help="The method to use for classification. Check constants.py for supported methods.")
    parser.add_argument("--rag", type=bool, action=argparse.BooleanOptionalAction, default=True, help="Whether to use RAG for context retrieval. Defaults to True. Use --no-rag to disable.")
    parser.add_argument("--path", type=str, default=str(pathlib.Path(__file__).parent), help="The path to the folder containing input files. Defaults to the same directory as the Python file.")
    parser.add_argument("--vertexai_project_name", type=str, help="The Vertex AI project name")
    parser.add_argument("--vertexai_location", type=str, help="The Vertex AI location")

    args = parser.parse_args()

    folder_path = pathlib.Path(args.path) / args.folder_name
    cls = Classifier(input_folder=folder_path, model=args.model, method=args.method, rag=args.rag,
                     vertexai_project_name=args.vertexai_project_name, vertexai_location=args.vertexai_location)
    cls.classify_relationships()

if __name__ == "__main__":
    main()