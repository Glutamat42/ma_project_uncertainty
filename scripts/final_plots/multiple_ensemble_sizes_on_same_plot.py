import json
import os

# config, most likely options to change at top
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.utils.evaluation.helpers import asImg

base_dir = "/home/markus/workspace/data/final_data/ensembles/outputs_supervised_best/"
base_dir = "/home/markus/workspace/data/final_data/ensembles/outputs_dino_best/"
base_dir = "/home/markus/workspace/data/final_data/aleatoric/aleatoric_ensemble/output_aleatoric_supervised_best/"
# base_dir = "/home/markus/workspace/data/final_data/aleatoric/aleatoric_ensemble/output_aleatoric_dino_best/"

ensemble_dirs = ["ensemble_1", "ensemble_3", "ensemble_5", "ensemble_10", "ensemble_20"]
# format: lambda to access metric inside dict (x = dict from one epoch) from metrics.json, label for plots, start at ensemble_dirs index
# (to skip ensemble size 1 for uct metrics)
metrics_to_plot = [(lambda x: x['validation_mse'], 'MSE', 0),
                   (lambda x: x['uncertainty']['miscal_area'], 'miscalibration area', 0),
                   (lambda x: x['uncertainty']['nll'], 'NLL', 0),
                   (lambda x: x['uncertainty']['sharpness'], 'sharpness', 0)]
save_plots = True

relative_metrics_path = "plots_metrics/metrics.json"
out_dir = "grouped"
# end config

out_path = os.path.join(base_dir, out_dir)
data = [json.load(open(os.path.join(base_dir, ens_dir, relative_metrics_path))) for ens_dir in ensemble_dirs]

Path(out_path).mkdir(exist_ok=True)

data_plot = []
for ens_index in range(len(data)):
    ens = data[ens_index]
    metrics_cur_ens = {x[1]: [] for x in metrics_to_plot}
    for epoch in ens:
        for metric in metrics_to_plot:
            if ens_index >= metric[2]:  # skip uct metrics for ens size 1 (without uct)
                metrics_cur_ens[metric[1]].append(metric[0](epoch))
    data_plot.append(metrics_cur_ens)


def save_or_show_plot(data_rows, row_labels, title):
    linear_scale = ['miscalibration area']
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
    x = [i + 1 for i in range(len(data_rows[-1]))]

    fig, ax = plt.subplots()
    for i in range(len(data_rows)):
        if len(data_rows[i]) == 0:
            continue
        ax.plot(x[:len(data_rows[i])], data_rows[i], color=colors[i], alpha=0.25)
        ax.plot(x[:len(data_rows[i])], pd.Series(data_rows[i]).rolling(window=5, min_periods=1, center=True).mean(), label=row_labels[i], color=colors[i])
    ax.axes.set_yscale('linear' if title in linear_scale else 'log')
    ax.grid(which='both')
    ax.set_title(title)
    ax.set_xlim(left=1, right=len(data_rows[-1]))
    ax.xaxis.set_ticks([1] + list(range(5, len(data_rows[-1]) - 1, 5)) + [len(data_rows[-1])])
    ax.legend()

    if save_plots:
        img = asImg(None)
        filename = f"{title}"
        directory = os.path.join(out_path)
        Path(directory).mkdir(exist_ok=True)
        img.save(os.path.join(directory, filename), "JPEG", quality=85, optimize=True)
    else:
        plt.show()


for metric in metrics_to_plot:
    data_rows = [np.array(x[metric[1]]) for x in data_plot]
    save_or_show_plot(data_rows, ensemble_dirs, metric[1])
