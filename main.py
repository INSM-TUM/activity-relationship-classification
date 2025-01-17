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
from os import getenv
from dotenv import load_dotenv
from vertexai.generative_models import GenerativeModel, Part, Content
from ollama import Client

from datetime import datetime

from constants import *
from util import *
from rag import Rag
import prompts

load_dotenv()



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
    def __init__(self, input_folder: pathlib.Path | str, model: str, method: str, rag: bool, dry_run : bool = False, output_folder: pathlib.Path | str = None):
        log('Classifier '+str(locals()))
        if isinstance(input_folder, str):
            input_folder = pathlib.Path(input_folder)
        if not input_folder.exists():
            raise ValueError("Folder not found")

        self.model = model
        self.platform = get_platform(model)
        self.method = method
        self.is_rag = rag
        self.dry_run = dry_run

        # if not (folder_name / "truth.csv").exists():
        #     raise ValueError("Truth file not found")
        # self.truth_file = folder_name / "truth.csv"

        activities_path = input_folder / "activities.txt"
        if not activities_path.exists():
            raise ValueError("Activities file not found")
        with open(activities_path, "r", encoding="utf-8") as f:
            self.activities = [line.strip() for line in f if line.strip()]

        self._validate_method()

        example_path = input_folder / f"examples ({method}).txt"
        if not example_path.exists() and method in ["few", "few-cot"]:
            raise ValueError("Example file not provided for few-shot learning")
        self._build_system_prompt(example_path)

        self._initialize_clients()

        process_desc_path = input_folder / "desc.txt"
        if not process_desc_path.exists():
            raise ValueError("Process description file not provided")
        self._load_process_description(process_desc_path)

        if self.is_rag:
            self._init_rag(input_folder)
        
        output_folder = output_folder or (input_folder / "results") 
        output_folder = pathlib.Path(output_folder)
        self._create_result_path(input_folder, output_folder)

    def _build_system_prompt(self, example_path: pathlib.Path):
        self.system_prompt = prompts.base_text
        if self.method in ["vanilla", "few"]:
            self.system_prompt += "\n" + prompts.how_to_format + '\n' + prompts.what_queries
        if self.method in ["cot", "few-cot"]:
            self.system_prompt += "\n" + prompts.cot_queries
        if self.method in ["few", "few-cot"]:
            with open(example_path, "r", encoding="utf-8") as f:
                example = f.read()
                self.system_prompt += "\n" + example

    def _validate_method(self):
        if self.method not in methods:
            raise ValueError(f"Method {self.method} not supported")

    def _initialize_clients(self):
        if self.dry_run:
            return
        elif self.platform == Platforms.OPEN_AI:
            self.client = openai.OpenAI()
        elif self.platform == Platforms.ANTHROPIC:
            self.client = Anthropic()
        elif self.platform == Platforms.VERTEX:
            vertexai_project_name = getenv('VERTEXAI_PROJECT_NAME')
            vertexai_location = getenv('VERTEXAI_LOCATION')
            if not vertexai_project_name:
                raise ValueError("Project name not provided for Vertex AI, must be provided for Vertex AI models")
            if not vertexai_location:
                raise ValueError("Location not provided for Vertex AI, must be provided for Vertex AI models")
            vertexai.init(project=vertexai_project_name, location=vertexai_location)
        elif self.platform == Platforms.OLLAMA:
            self.client = Client(
                host='http://localhost:11434', # TODO put URL in env file
                headers={'Content-Type': 'application/json'}
            )
        if self.model not in all_models:
            raise ValueError("Model not supported")

    def _load_process_description(self, process_desc_path: pathlib.Path):
        with open(process_desc_path, encoding="utf8", mode="r") as f:
            process_desc = f.read()
            self.process_desc = process_desc

    def _create_result_path(self, input_folder: pathlib.Path, output_folder: pathlib.Path):
        # (absolute_path / "results").mkdir(parents=True, exist_ok=True)
        dt_string = datetime.now().strftime("%d%m%Y-%H%M%S")
        result_file_name = f"{input_folder.name}-{dt_string}--{self.model.replace(':', '-')}-{self.method}{'_rag' if self.is_rag else ''}"
        output_folder.mkdir(parents=True, exist_ok=True)
        self.result_file = (output_folder / (result_file_name + ".csv")).open("x")

    def _init_rag(self, input_folder: pathlib.Path):
        log('Initializing RAG')
        self.rag_engine = Rag(input_folder.name, self.process_desc)

    def classify_relationships(self):
        
        self._write_result_header()        
        for i in range(len(self.activities)):
            activity1 = self.activities[i]
            for j in range(i + 1, len(self.activities)):
                activity2 = self.activities[j]
                context = self._get_context([activity1, activity2])
                log(f'Classifying "{activity1}"->"{activity2}"')

                if self.method in ["vanilla", "few"]:
                    parse = self._classify_normal(activity1, activity2, context)
                elif self.method in ["cot", "few-cot"]:
                    parse = self._classify_cot(activity1, activity2, context)

                self._write_result_row(parse, activity1, activity2)
            self.result_file.flush()

        # stats.calculate_stats(self.truth_file, self.result_file.name, method=self.method)
        return self.result_file.name

    def _classify_normal(self, activity1, activity2, context):
        messages = [{"role": "user", "content": prompts.create_query(activity1, activity2, context)}]
        parse = self._get_model_response(messages, self.system_prompt)
        return self._parse_response(parse)

    # TODO this mixes prompt chaining an chain of thought
    # proposed fix of kerstin: adding the cot prompt in the self.system_prompt and being more explanatory in the examples
    def _classify_cot(self, activity1, activity2, context):
        messages = [{"role": "user", "content": prompts.create_query(activity1, activity2, context)}]
        cot_response = self._get_model_response(messages, self.system_prompt)
        messages.append({"role": "assistant", "content": cot_response})
        messages.append({"role": "user", "content": prompts.how_to_format})
        parse = self._get_model_response(messages, self.system_prompt)
        return self._parse_response(parse)

    def _write_result_header(self):
        self.result_file.write("sep=;\nFirst Activity;Second Activity;" + ";".join(categories) + f";{lon}\n")

    def _get_context(self, activities: list) -> list | str:
        if self.is_rag:
            return self.rag_engine.return_related(activities)
        return self.process_desc

    def _get_model_response(self, messages: list, system_prompt : str = None) -> Any:
        if self.dry_run :
            print('====Messages=====')
            # print(messages)
            print('====system_prompt=====')
            # print(system_prompt)
            ret = """{
    "First Activity": "-",
    "Second Activity": "-",
    "Category": "-",
    "Justification": "-",
    "Law of Nature": "-"
}"""
        elif self.platform == Platforms.OPEN_AI:
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

    def _call_openai(self, messages: list[dict], system_prompt : str = None, temperature: float = 0) -> Any:
        if system_prompt:
            messages.insert(0,{"role": "system", "content": system_prompt})
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        return completion.choices[0].message.content

    def _call_anthropic(self, messages: list[dict], system_prompt : str = None, temperature: float = 0) -> Any:
        completion = self.client.messages.create(
            model=self.model,
            system=system_prompt,
            messages=messages,
            temperature=temperature,
            max_tokens=1024
        )
        return completion.content[0].text

    def _call_vertex(self, messages: list[dict], system_prompt : str = None, temperature: float = 0) -> Any:
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
                time.sleep(15) # Vertex AI has a request limit of 5/minute, so waiting 4 times 15 seconds should solve the problem
                continue
            else:
                break
        else:
            raise Exception("Failed to send message to vertex AI")
        return response.text
    
    def _call_ollama(self, messages: list[dict], system_prompt : str = None) -> Any:

        if system_prompt:
            messages.insert(0,{"role": "system", "content": system_prompt})

        response = self.client.chat(
            model='llama3', 
            messages=messages,
            options = {
                # "num_ctx": 48000
            },
        )
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
            log('JSON Error')
            return {
                "First Activity": "-",
                "Second Activity": "-",
                "Category": "-",
                "Justification": "JSON ERROR",
                "Law of Nature": "-"
            }

    def __del__(self):
        if hasattr(self, 'result_file'):
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