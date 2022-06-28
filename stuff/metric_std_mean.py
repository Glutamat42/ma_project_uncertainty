import pandas as pd
import json
import os

paths = [
    "/home/markus/workspace/data/final_data/bbb_paper/outputs_dino_best_",
    "/home/markus/workspace/data/final_data/bbb_paper/outputs_supervised_best_",
    "/home/markus/workspace/data/final_data/bbb_paper/outputs_supervised_comparable_"
]

indexes = [1, 2, 3]


def calc(path):
    print(path)
    metric = 0
    for i in indexes:
        metric += json.load(open(os.path.join(path + str(i), "ensemble_1", "plots_metrics/metrics.json")))[-1]["uncertainty"][
                      "miscal_area"] / len(indexes)
    print(metric)


for p in paths:
    calc(p)
