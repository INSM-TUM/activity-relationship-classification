from main import Classifier
from constants import *

for model in all_models:#['gemini-1.5-flash-001']:#
    for method in methods:#['few']: #
        cls = Classifier(input_folder='./thesis_process', model=model, method=method, rag=True, dry_run=False)
        cls.classify_relationships()