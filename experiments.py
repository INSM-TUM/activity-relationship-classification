from main import Classifier
from constants import *
import pathlib
import stats
from datetime import datetime


input_folder = pathlib.Path('./thesis_process')
truth_file = input_folder / "truth.csv"
dt_string = datetime.now().strftime("%d%m%Y-%H%M%S")

if not truth_file.exists():
    raise ValueError("Truth file not found")

for model in all_models:#['gemini-1.5-flash-001']:#
    for method in methods:#['few']: #
        cls = Classifier(input_folder=input_folder, model=model, method=method, rag=True, dry_run=False, output_folder=input_folder / 'results' / ('run-'+dt_string))
        result_file = cls.classify_relationships()
        stats.calculate_stats(truth_file, result_file, output_folder=input_folder / 'results' / ('run-'+dt_string) / 'stats')

# cls = Classifier(input_folder=input_folder, model='gemini-1.5-pro-001', method='vanilla', rag=False, dry_run=False, output_folder=input_folder / 'results' / ('run-'+dt_string))
# result_file = cls.classify_relationships()

        