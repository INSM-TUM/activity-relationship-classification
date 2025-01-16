import pathlib

from constants import categories, all_categories
import pandas as pd


def change_xlsx_to_csv(name):
    df = pd.read_excel(f'{name}.xlsx')

    df = df[
        ["First Activity", "Second Activity", "Governmental Law", "Best Practice", "Business Rule", "Law of Nature"]]
    df.to_csv(f'{name}.csv', sep=";", index=False)

def calculate_stats(truth_file: str, generated_file: str, method: str = "normal"):
    cats_count = {"Governmental Law": {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "precision": 0, "recall": 0, "F1-Score": 0},
                  "Best Practice": {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "precision": 0, "recall": 0, "F1-Score": 0},
                  "Business Rule": {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "precision": 0, "recall": 0, "F1-Score": 0},
                  "Law of Nature": {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "precision": 0, "recall": 0, "F1-Score": 0}}
    with open(truth_file, 'r', encoding='utf-8') as f, open(generated_file, 'r', encoding='ISO-8859-1') as g:
        if not f or not g:
            print("File not found")
            return
        f.readline(), g.readline()  # Skip seperator
        # if f.readline() != g.readline():
        #     print("Files have different columns")
        #     return

        count = 0
        for line_truth, line_generated in zip(f, g):
            count += 1
            truth = (line_truth.strip('\n')).split(";")
            generated = (line_generated.strip('\n')).split(";")
            # if len(truth) != len(generated):
            #     print(f"Line {count} has different number of columns")
            #     return

            # check if we compare the same pair of activities
            if truth[0] != generated[0]:
                print(f"First activity is different in line {count}")
                return
            if truth[1] != generated[1]:
                print(f"Second activity is different in line {count}")
                return
            
            # compute confusion matrix values for each category
            for i in range(2, len(all_categories) + 2):
                # check if both the generated result and the ground truth classify the pair of activities in the current category i (True Positives)
                if (truth[i] != "" and truth[i] != "-") and (generated[i] != "" and generated[i] != "-"):
                    cats_count[all_categories[i - 2]]["tp"] += 1
                # check if both the generated result and the ground truth did not classify the pair of activities as the current category i (True Negatives)
                elif (truth[i] == "" or truth[i] == "-") and (generated[i] == "" or generated[i] == "-"):
                    cats_count[all_categories[i - 2]]["tn"] += 1
                # check if the generated result did not classify the pair of activities as the current category i but the truth did (False Negatives)
                elif (truth[i] != "" and truth[i] != "-") and (generated[i] == "" or generated[i] == "-"):
                    cats_count[all_categories[i - 2]]["fn"] += 1
                # check if the truth did not classify the pair of activities as the current category i but the generated result did (False Positives)
                elif (truth[i] == "" or truth[i] == "-") and (generated[i] != "" and generated[i] != "-"):
                    cats_count[all_categories[i - 2]]["fp"] += 1

        # compute precision, recall, and F1-score for each category
        for category in cats_count.keys():
            cats_count[category]["precision"] = precision(cats_count[category]["tp"], cats_count[category]["fp"])
            cats_count[category]["recall"] = recall(cats_count[category]["tp"], cats_count[category]["fn"])
            cats_count[category]["F1-Score"] = f1_score(cats_count[category]["tp"], cats_count[category]["fp"], cats_count[category]["fn"])
        
        # compute average precision, recall, and F1-Score for the categories best practice, governmental law, and business rule
        average_metrics = {"precision": 0, "recall": 0, "F1-Score": 0}
        for metric in average_metrics:
            average = 0
            for category in categories:
                average += cats_count[category][metric] 
            average_metrics[metric] = average / len(average_metrics)

        pathlib.Path("stats").mkdir(parents=True, exist_ok=True)
        # if consensus:
        #     file_name = f"stats/consensus-{pathlib.Path(truth_file).stem}-{pathlib.Path(generated_file).stem}.csv"
        # else:
        file_name = f"./stats/{method}-{pathlib.Path(truth_file).stem}-{pathlib.Path(generated_file).parent.name}-{pathlib.Path(generated_file).stem}.csv"
        print(file_name)
        with open(file_name, "w") as stats_file:
            stats_file.write("Category,TP,FN,FP,TN,Precision,Recall,F1\n")
            
            for cat in all_categories:
                stats_file.write(f"{cat},")
                stats_file.write(f"{cats_count[cat]['tp']},")
                stats_file.write(f"{cats_count[cat]['fn']},")
                stats_file.write(f"{cats_count[cat]['fp']},")
                stats_file.write(f"{cats_count[cat]['tn']},")
                stats_file.write(f"{precision(cats_count[cat]['tp'], cats_count[cat]['fp']):.3f},")
                stats_file.write(f"{recall(cats_count[cat]['tp'], cats_count[cat]['fn']):.3f},")
                stats_file.write(
                    f"{f1_score(cats_count[cat]['tp'], cats_count[cat]['fp'], cats_count[cat]['fn']):.3f}\n")
            
            stats_file.write("Average (without law of nature),")
            stats_file.write(f"-,")
            stats_file.write(f"-,")
            stats_file.write(f"-,")
            stats_file.write(f"-,")
            stats_file.write(f"{average_metrics["precision"]:.3f},")
            stats_file.write(f"{average_metrics["recall"]:.3f},")
            stats_file.write(
                f"{average_metrics["F1-Score"]:.3f}\n")

def precision(tp: int, fp: int):
    try:
        return tp / (tp + fp)
    except ZeroDivisionError:
        print("No positive predictions")
        return 1.000


def recall(tp: int, fn: int):
    try:
        return tp / (tp + fn)
    except ZeroDivisionError:
        print("No positive cases")
        return 1.000


def f1_score(tp: int, fp: int, fn: int):
    try:
        return (2 * tp) / (2 * tp + fp + fn)
    except ZeroDivisionError:
        print("No positive cases or predictions")
        return 1.000

# Example usage
if __name__ == '__main__':
    calculate_stats('./thesis_process/truth.csv', './thesis_process/results/vanilla/rag/claude-3-5-haiku-latest/thesis_process-14012025-224724.csv', method='vanilla')