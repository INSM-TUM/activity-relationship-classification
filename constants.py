import vertexai.preview.generative_models as generative_models

models_openai = ["gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini", "ft:gpt-3.5-turbo-1106:personal:airport-first:9qNNlgWe", "gpt-4o-2024-08-06"]
models_anthropic = ["claude-3-haiku-20240307", "claude-3-5-sonnet-20240620", "claude-3-opus-20240229"]
models_vertex = ["gemini-1.5-flash-001", "gemini-1.5-pro-001", "gemini-experimental",
                 "gemini-flash-experimental", "gemini-pro-experimental"]
all_models = models_openai + models_anthropic + models_vertex
categories = ["Governmental Law", "Best Practice", "Business Rule"]
methods = ["normal", "cot", "few", "few-cot"]
lon = "Law of Nature"
all_categories = categories + [lon]

safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
}
