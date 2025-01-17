import vertexai.preview.generative_models as generative_models
from enum import Enum



class Platforms(Enum):
    OPEN_AI = 'openAI'
    ANTHROPIC = 'anthropic'
    VERTEX = 'vertex'
    OLLAMA = 'ollama'

models = {
    Platforms.OPEN_AI : ["gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini"],
    Platforms.ANTHROPIC : ["claude-3-5-haiku-latest", "claude-3-5-sonnet-20240620"],
    Platforms.VERTEX : ["gemini-1.5-flash-001", "gemini-1.5-pro-001"],
    Platforms.OLLAMA : ["llama3"]
}

def get_platform(model_name):
    return next((platform for platform, model_names in models.items() if model_name in model_names), None)


all_models = set().union(*[models[platform] for platform in Platforms ])

categories = ["Governmental Law", "Best Practice", "Business Rule"]
methods = ["vanilla", "cot", "few", "few-cot"]
lon = "Law of Nature"
all_categories = categories + [lon]

safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
}
