import json
import pathlib
import time
from datetime import datetime
from typing import Any

import openai
import vertexai
from anthropic import Anthropic
from dotenv import load_dotenv
from vertexai.generative_models import GenerativeModel, Part, Content

from constants import *
import stats
from rag import Rag

load_dotenv()


class Classifier:
    def __init__(self, activities: list, model: str = "gpt-4o-mini", sys_path: str = "airport/sys_desc_vanilla.txt",
                 process_desc_path: str = "airport/desc.txt", is_rag: bool = False,
                 rag_collection_name: str = "airport",
                 method: str = "normal", truth_file: str = "airport/Airport_GroundTruth.csv"):

        with open(pathlib.Path(__file__).parent / sys_path, "r", encoding="utf-8") as f:
            form = f.read()

        if method not in methods:
            raise ValueError("Method not supported")
        self.method = method

        self.result_file = None
        self.model = model
        if self.model in models_openai or method == "consensus":
            self.openai_client = openai.OpenAI()
            self.openai_sys = {"role": "system", "content": form}
        if self.model in models_anthropic or method == "consensus":
            self.claude_client = Anthropic()
            self.anthropic_sys = form
        if self.model in models_vertex or method == "consensus":
            vertexai.init(project="solid-scope-430714-m9", location="europe-west3")  # us-central1
            self.vertex_sys = form
        if model not in all_models:
            raise ValueError("Model not supported")

        self.is_rag = is_rag
        if self.is_rag:
            self.rag_engine = Rag(rag_collection_name)

        self.activities = activities
        self.truth_file = truth_file

        with open(pathlib.Path(__file__).parent / process_desc_path, encoding="utf8", mode="r") as f:
            self.process_desc = f.read()
            self.context = self.process_desc

        self.create_path(rag_collection_name)

    def create_path(self, name):
        pathlib.Path("results").mkdir(parents=True, exist_ok=True)
        now = datetime.now()
        dt_string = now.strftime("%d%m%Y-%H%M%S")
        result_file_name = name + "-" + dt_string
        folder_name = self.model.replace(":", "-")
        if self.is_rag:
            self.rag_engine.load_embeddings(self.process_desc)
            pathlib.Path(f"results/{self.method}/rag").mkdir(parents=True, exist_ok=True)
            pathlib.Path(f"results/{self.method}/rag/" + folder_name).mkdir(parents=True, exist_ok=True)
            self.result_file = pathlib.Path(f"results/{self.method}/rag/" + folder_name + "/" + result_file_name
                                            + ".csv").open("x")
        else:
            pathlib.Path(f"results/{self.method}/" + folder_name).mkdir(parents=True, exist_ok=True)
            self.result_file = pathlib.Path(f"results/{self.method}/" + folder_name + "/" + result_file_name
                                            + ".csv").open("x")

    def classify_relationships(self):
        if self.method == "normal":
            self.normal()
        elif self.method == "cot":
            self.cot()
        elif self.method == "few":
            self.normal()
        elif self.method == "few-cot":
            self.cot()

        stats.calculate_stats(self.truth_file, self.result_file.name, method=self.method)

    def normal(self):
        count = 0
        self.result_file.write("sep=;\nFirst Activity;Second Activity;" + ";".join(categories) + f";{lon};Comments\n")
        for i in range(len(self.activities)):
            for j in range(i + 1, len(self.activities)):
                # Get the context using RAG
                if self.is_rag:
                    context = self.rag_engine.return_related([self.activities[i], self.activities[j]])
                else:
                    context = self.context

                # ASK FOR ALL CATEGORIES AT ONCE
                # OPENAI
                if hasattr(self, "openai_sys"):
                    messages = [
                        self.openai_sys,
                        {"role": "user",
                         "content": "Context:\n" + (context if isinstance(context, str) else '\n'.join(context))},
                        {"role": "user", "content": "What is the relationship between " + self.activities[i]
                                                    + " and " + self.activities[j] + "?"}]
                    parse = self.call_openai(messages, self.model)

                # ANTHROPIC:
                elif hasattr(self, "claude_client"):
                    sys_cached = [
                        {
                            "type": "text",
                            "text": self.anthropic_sys
                        },
                        {
                            "type": "text",
                            "text": self.context,
                            "cache_control": {"type": "ephemeral"}
                        }
                    ]
                    # messages = [
                    #     {"role": "user",
                    #      "content": "Here is some context about the process:\n"
                    #      + (context if isinstance(context, str) else '\n'.join(context))
                    #      + "\n What is the relationship between " + self.activities[i] + " and "
                    #      + self.activities[j] + "?"}]
                    messages = [
                        {"role": "user",
                         "content": "What is the relationship between " + self.activities[i] + " and " +
                                    self.activities[j]}
                    ]
                    parse = self.claude_client.beta.prompt_caching.messages.create(
                        model=self.model,
                        system=sys_cached,
                        messages=messages,
                        max_tokens=1024
                    )
                    # print("Input tokens: " + str(parse.usage.input_tokens))
                    # print("Cache creation tokens: " + str(parse.usage.cache_creation_input_tokens))
                    # print("Cache read tokens: " + str(parse.usage.cache_read_input_tokens))
                    # print("Output tokens: " + str(parse.usage.output_tokens))
                    print(parse.content[0].text)
                    parse = parse.content[0].text
                    # parse = self.call_anthropic(messages, self.model, self.anthropic_sys)

                # VERTEX:
                elif hasattr(self, "vertex_sys"):
                    messages = [
                        {"role": "user",
                         "content": "Here is some context about the process:\n" + (context if isinstance(context, str)
                                                                                   else '\n'.join(
                             context)) + "What is the relationship between " + self.activities[i] + " and "
                                    + self.activities[j] + "?"}]
                    parse = self.call_vertex(messages, self.model, self.vertex_sys)
                else:
                    raise ValueError("Model not supported")

                if parse[0] == '`':
                    parse = parse[8:-4]
                try:
                    parse = json.loads(parse)
                except json.JSONDecodeError:
                    parse = self.fix_json(parse)

                self.result_file.write(self.activities[i] + ";" + self.activities[j] + ";")
                for category in categories:
                    if category == parse["Category"]:
                        self.result_file.write(parse["Justification"] + ";")
                    else:
                        self.result_file.write("-;")
                try:
                    if str(parse["Category"]).lower() == "law of nature":
                        self.result_file.write(parse["Justification"] + ";")
                    else:
                        self.result_file.write(parse["Law of Nature"] + ";")  # + \n
                except KeyError:
                    self.result_file.write("-;")
                self.result_file.write(json.dumps(parse).replace('\n', ' ').replace('\r', '') + "\n")
                print("---------------------------------------------------")
                count += 1
            self.result_file.flush()
        print(count)

    def cot(self):
        self.result_file.write("sep=;\nFirst Activity;Second Activity;" + ";".join(categories) + f";{lon};Comments\n")
        for i in range(len(self.activities)):
            for j in range(i + 1, len(self.activities)):
                # Get the context using RAG CONTEXT CURRENTLY IN SYSTEM PROMPT
                # if self.is_rag:
                #     context = self.rag_engine.return_related([self.activities[i], self.activities[j]])
                # else:
                #     context = self.context
                # context = context if isinstance(context, str) else '\n'.join(context)

                # first_m = f"Apart from the law of nature, which of the categories best describes the contextual origin of why {self.activities[i]} occurs before {self.activities[j]}? Explain why you chose this category and not another one. If none of the categories apply to the relationship, explain why it is not an instance of any of the categories. After discussing the contextual origin, discuss if the ordering is due to a law of nature. ### Context: {context}"
                first_m = f"Apart from the law of nature, which of the categories best describes the contextual origin of why {self.activities[i]} occurs before {self.activities[j]}? Explain why you chose this category and not another one. If none of the categories apply to the relationship, explain why it is not an instance of any of the categories. After discussing the contextual origin, discuss if the ordering is due to a law of nature."

                # OPENAI
                messages = [
                    {"role": "user",
                     "content": first_m}
                ]
                if hasattr(self, "openai_sys"):
                    messages = [
                        self.openai_sys,
                        {"role": "user",
                         "content": first_m},
                    ]
                    parse = self.call_openai(messages, self.model)

                # ANTHROPIC:
                elif hasattr(self, "claude_client"):
                    sys_cached = [
                        {
                            "type": "text",
                            "text": self.anthropic_sys,
                            "cache_control": {"type": "ephemeral"}
                        }
                    ]
                    parse = self.claude_client.beta.prompt_caching.messages.create(
                        model=self.model,
                        system=sys_cached,
                        messages=messages,
                        max_tokens=1024
                    )
                    # print("Input tokens: " + str(parse.usage.input_tokens))
                    # print("Cache creation tokens: " + str(parse.usage.cache_creation_input_tokens))
                    # print("Cache read tokens: " + str(parse.usage.cache_read_input_tokens))
                    # print("Output tokens: " + str(parse.usage.output_tokens))
                    print(parse.content[0].text)
                    parse = parse.content[0].text

                # VERTEX:
                elif hasattr(self, "vertex_sys"):
                    parse = self.call_vertex(messages, self.model, self.vertex_sys)
                else:
                    raise ValueError("Model not supported")

                the_cot = parse
                messages.append({"role": "assistant", "content": parse})
                messages.append({"role": "user",
                                 "content": """
                                 Structure your answer in the following format without any additional text, and replace the placeholders with the correct values:
{
    "First Activity": "-",
    "Second Activity": "-",
    "Category": "-",
    "Justification": "-"
    "Law of Nature": "-"
}

If none of the three contextual origin categories apply to the relationship, put a dash in the Category and Justification fields and do not include any other text for justifying your decision. Otherwise, put the category you chose in the "Category" field, the justification for your choice in the "Justification" field. In the "Law of Nature" field you should put yes or no in the answer key with a justification. 
                                 """})

                if hasattr(self, "openai_sys"):
                    messages.insert(0, self.openai_sys)
                    parse = self.call_openai(messages, self.model)

                elif hasattr(self, "claude_client"):
                    sys_cached = [
                        {
                            "type": "text",
                            "text": self.anthropic_sys,
                            "cache_control": {"type": "ephemeral"}
                        }
                    ]
                    parse = self.claude_client.beta.prompt_caching.messages.create(
                        model=self.model,
                        system=sys_cached,
                        messages=messages,
                        max_tokens=1024
                    )
                    print("Input tokens: " + str(parse.usage.input_tokens))
                    print("Cache creation tokens: " + str(parse.usage.cache_creation_input_tokens))
                    print("Cache read tokens: " + str(parse.usage.cache_read_input_tokens))
                    print("Output tokens: " + str(parse.usage.output_tokens))
                    print(parse.content[0].text)
                    parse = parse.content[0].text

                elif hasattr(self, "vertex_sys"):
                    parse = self.call_vertex(messages, self.model, self.vertex_sys)
                else:
                    raise ValueError("Model not supported")

                if parse[0] == '`':
                    parse = parse[8:-4]
                try:
                    parse = json.loads(parse)
                except json.JSONDecodeError:
                    parse = self.fix_json(parse)

                self.result_file.write(self.activities[i] + ";" + self.activities[j] + ";")
                for category in categories:
                    if category == parse["Contextual Origin"]["Category"]:
                        self.result_file.write(parse["Contextual Origin"]["Justification"] + ";")
                    else:
                        self.result_file.write("-;")
                try:
                    if str(parse["Law of Nature"]["Answer"]).lower() == "yes" or str(
                            parse["Law of Nature"]["Answer"]).lower() == "yes.":
                        self.result_file.write(parse["Law of Nature"]["Justification"] + ";")
                    else:
                        self.result_file.write("-;")
                except KeyError:
                    self.result_file.write("-;")
                self.result_file.write(the_cot.replace('\n', ' ').replace('\r', '').replace(';', ',') + " " +
                                       json.dumps(parse).replace('\n', ' ').replace('\r', '').replace(';', ',') + "\n")
                print("---------------------------------------------------")
            self.result_file.flush()

    def fix_json(self, parse: str) -> dict:
        self.openai_client = openai.OpenAI()
        if parse[0] == '`':
            parse = parse[8:-4]
        try:
            json.loads(parse)
        except json.JSONDecodeError:
            messages = [{"role": "user", "content": "Could you check if this JSON string is valid and properly escaped,"
                                                    "if not, please fix it and return it to me without any additional"
                                                    "text. Make sure its valid to be parsed in python\n" + parse}]
            parse = self.call_openai(messages, "gpt-4o-mini")
            if parse[0] == '`':
                parse = parse[8:-4]
        return json.loads(parse)

    def call_openai(self, messages: list[dict], model: str, temperature: float = 0) -> Any:
        completion = self.openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        print(model + ": " + completion.choices[0].message.content)
        # print(completion.choices[0].message.content)
        return completion.choices[0].message.content

    def call_anthropic(self, messages: list[dict], model: str, sys: str = "", temperature: float = 0) -> Any:
        completion = self.claude_client.messages.create(
            model=model,
            system=sys,
            messages=messages,
            temperature=temperature,
            max_tokens=1024
        )
        print(model + ": " + completion.content[0].text)
        # print(completion.content[0].text)
        return completion.content[0].text

    def call_vertex(self, messages: list[dict], model: str, sys: str = "", temperature: float = 0) -> Any:
        model_gen = GenerativeModel(
            model,
            system_instruction=[sys]
        )
        if len(messages) == 1:
            chat = model_gen.start_chat()
        else:
            hist = []
            for message_i in range(len(messages) - 1):
                role = messages[message_i]["role"] if messages[message_i]["role"] == "user" else "model"
                content = messages[message_i]["content"]
                hist.append(Content(role=role, parts=[Part.from_text(content)]))
            chat = model_gen.start_chat(history=hist)

        for i in range(5):
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

        print(model + ": " + response.text)
        return response.text

    def __del__(self):
        if self.result_file:
            self.result_file.close()

travel_acts = [
    "Fill out travel request",
    "Attach documents",
    "Sign Travel Request",
    "Scan and Send documents",
    "Check documents",
    "Go on the business trip",
    "Fill out travel reimbursement request",
    "Attach invoices",
    "Scan and send reimbursement request",
    "Check reimbursement request",
    "Initiate bank transfer"
]

thesis_acts = [
    "Search for a topic",
    "Conduct informal meeting to explain topic",
    "Write and submit proposal",
    "Get registered by chair on Koinon",
    "Student accept thesis on Koinon",
    "Start writing thesis",
    "Conduct regular catch-up meetings",
    "Submit thesis on Koinon",
    "Present thesis in Colloquium"
]
acts = ['Scan ticket',
        'Change number of bags',
        'Change seat',
        'Check validity of documents',
        'Weigh baggage',
        'Cancel check-in',
        'Process payment',
        'Check-in luggage',
        'Load luggage'
]

cls = Classifier(thesis_acts,
                 "claude-3-5-sonnet-20240620",
                 method="cot",
                 sys_path="interviews/cot_sys2.txt",
                 process_desc_path="interviews/desc.txt",
                 truth_file="interviews/truth.csv",
                 is_rag=False,
                 rag_collection_name="thesis")
cls.classify_relationships()
