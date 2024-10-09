import pathlib

from constants import all_categories
import pandas as pd


def change_xlsx_to_csv(name):
    df = pd.read_excel(f'{name}.xlsx')

    df = df[
        ["First Activity", "Second Activity", "Governmental Law", "Best Practice", "Business Rule", "Law of Nature"]]
    df.to_csv(f'{name}.csv', sep=";", index=False)


def calculate_stats(truth_file: str, generated_file: str, consensus: bool = False, method: str = "normal"):
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0
    cats_count = {"Governmental Law": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
                  "Best Practice": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
                  "Business Rule": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
                  "Law of Nature": {"tp": 0, "fp": 0, "fn": 0, "tn": 0}}
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
            if truth[0] != generated[0]:
                print(f"First activity is different in line {count}")
                return
            if truth[1] != generated[1]:
                print(f"Second activity is different in line {count}")
                return
            for i in range(2, len(all_categories) + 2):
                if (truth[i] != "" and truth[i] != "-") and (generated[i] != "" and generated[i] != "-"):
                    total_tp += 1
                    cats_count[all_categories[i - 2]]["tp"] += 1
                elif (truth[i] == "" or truth[i] == "-") and (generated[i] == "" or generated[i] == "-"):
                    total_tn += 1
                    cats_count[all_categories[i - 2]]["tn"] += 1
                elif (truth[i] != "" and truth[i] != "-") and (generated[i] == "" or generated[i] == "-"):
                    total_fn += 1
                    cats_count[all_categories[i - 2]]["fn"] += 1
                elif (truth[i] == "" or truth[i] == "-") and (generated[i] != "" and generated[i] != "-"):
                    total_fp += 1
                    cats_count[all_categories[i - 2]]["fp"] += 1
        print(f"TP: {total_tp}")
        print(f"FN: {total_fn}")
        print(f"FP: {total_fp}")
        print(f"TN: {total_tn}")
        print(f"Precision: {precision(total_tp, total_fp):.3f}")
        print(f"Recall: {recall(total_tp, total_fn):.3f}")
        print(
            f"F1-score: {f1_score(total_tp, total_fp, total_fn):.3f}")
        for cat in all_categories:
            print("-" * 30)
            print(f"Category: {cat}")
            print(f"TP: {cats_count[cat]['tp']}")
            print(f"FN: {cats_count[cat]['fn']}")
            print(f"FP: {cats_count[cat]['fp']}")
            print(f"TN: {cats_count[cat]['tn']}")
            print(f"Precision: {precision(cats_count[cat]['tp'], cats_count[cat]['fp']):.3f}")
            print(f"Recall: {recall(cats_count[cat]['tp'], cats_count[cat]['fn']):.3f}")
            print(
                f"F1-score: {f1_score(cats_count[cat]['tp'], cats_count[cat]['fp'], cats_count[cat]['fn']):.3f}")

        pathlib.Path("stats").mkdir(parents=True, exist_ok=True)
        # if consensus:
        #     file_name = f"stats/consensus-{pathlib.Path(truth_file).stem}-{pathlib.Path(generated_file).stem}.csv"
        # else:
        file_name = f"stats/{method}-{pathlib.Path(truth_file).stem}-{pathlib.Path(generated_file).parent.name}-{pathlib.Path(generated_file).stem}.csv"
        with open(file_name, "w") as stats_file:
            stats_file.write("Category,TP,FN,FP,TN,Precision,Recall,F1\n")
            stats_file.write("All,")
            stats_file.write(f"{total_tp},")
            stats_file.write(f"{total_fn},")
            stats_file.write(f"{total_fp},")
            stats_file.write(f"{total_tn},")
            stats_file.write(f"{precision(total_tp, total_fp):.3f},")
            stats_file.write(f"{recall(total_tp, total_fn):.3f},")
            stats_file.write(
                f"{f1_score(total_tp, total_fp, total_fn):.3f}\n")
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
