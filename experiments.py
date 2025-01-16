from main import Classifier
from constants import *
import pathlib
import stats


input_folder = pathlib.Path('./thesis_process')
for model in ['gemini-1.5-flash-001']:#all_models:#
    for method in ['few']: #methods:#
        cls = Classifier(input_folder=input_folder, model=model, method=method, rag=False, dry_run=False)
        result_file = cls.classify_relationships()

        
        truth_file = input_folder / "truth.csv"
        if not truth_file.exists():
            raise ValueError("Activities file not found")
        print(truth_file)
        print(result_file)
        stats.calculate_stats(truth_file, result_file)