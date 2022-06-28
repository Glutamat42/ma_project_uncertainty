import json
import os
from pathlib import Path

import matplotlib.ticker
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

from src.utils.evaluation.helpers import asImg

"""
load metrics from multiple models (metrics.json, including uct metrics) and plot their change over training / epochs
with mean and std and std*2
"""

# config, most likely options to change at top
model_count = 10
# base_dir = "/home/markus/workspace/data/final_data/bbb_paper/outputs_supervised_whole_training"
# base_dir = "/home/markus/workspace/data/final_data/bbb_paper/outputs_dino_whole_training"
# base_dir = "/home/markus/workspace/data/final_data/bbb_graves/outputs_supervised_whole_training"
# base_dir = "/home/markus/workspace/data/final_data/bbb_graves/outputs_dino_whole_training"
# base_dir = "/home/markus/workspace/data/final_data/aleatoric/aleatoric/outputs_supervised_whole_training/"
base_dir = "/home/markus/workspace/data/final_data/aleatoric/aleatoric/outputs_dino_whole_training"
# base_dir = "/home/markus/workspace/data/final_data/mcbn/4_outputs_supervised_whole_training/"

# model_count = 3  # IMPORTANT: CHANGE THIS VALUE AS REQUIRED
# base_dir = "/home/markus/workspace/data/final_data/mcbn/4_outputs_dino_ONLY_HEAD_whole_training/"


save_plots = True
relative_metrics_path = "ensemble_1/plots_metrics/metrics.json"
out_dir = "grouped"
# end config

out_path = os.path.join(base_dir, out_dir)
data = [json.load(open(os.path.join(base_dir, str(i), relative_metrics_path))) for i in range(1, model_count + 1)]

Path(out_path).mkdir(exist_ok=True)

means_stds = {
    'MSE': {'mean': [], 'std': []},
    'miscalibration area': {'mean': [], 'std': []},
    'NLL': {'mean': [], 'std': []},
    'sharpness': {'mean': [], 'std': []},
}

for i in range(max([len(x) for x in data])):
    cur_ep = []
    for model in data:
        if len(model) > i:
            cur_ep.append(model[i])
    vals = np.array([x['validation_mse'] for x in cur_ep])
    means_stds['MSE']['mean'].append(vals.mean())
    means_stds['MSE']['std'].append(vals.std())
    vals = np.array([x['uncertainty']['miscal_area'] for x in cur_ep])
    means_stds['miscalibration area']['mean'].append(vals.mean())
    means_stds['miscalibration area']['std'].append(vals.std())
    vals = np.array([x['uncertainty']['nll'] for x in cur_ep])
    means_stds['NLL']['mean'].append(vals.mean())
    means_stds['NLL']['std'].append(vals.std())
    vals = np.array([x['uncertainty']['sharpness'] for x in cur_ep])
    means_stds['sharpness']['mean'].append(vals.mean())
    means_stds['sharpness']['std'].append(vals.std())

json.dump(means_stds, open(os.path.join(out_path, "metrics.json"), "w"), indent=4)


def save_or_show_plot(means, stds, title):
    linear_scale = ['miscalibration area']
    name_mappings = {
        'miscalibration area': {'y_label': 'Miscalibration area', 'title': None},
        'NLL': {'y_label': 'NLL', 'title': None},
        'sharpness': {'y_label': 'Sharpness', 'title': None},
        'MSE': {'y_label': 'MSE', 'title': None},
    }

    x = [i + 1 for i in range(len(means))]

    plt.style.use(['science', 'ieee'])

    fig, ax = plt.subplots()
    # fig.subplots_adjust(left=0.15)
    ax.plot(x, means, color='k', label="mean")
    ax.fill_between(x, means - stds, means + stds, alpha=0.3, color='k', label=r'$\pm 1*\sigma$')
    ax.fill_between(x, means - stds * 2, means + stds * 2, alpha=0.15, color='k', label=r'$\pm 2*\sigma$')
    ax.axes.set_yscale('linear' if title in linear_scale else 'log')
    # ax.axes.yaxis.set_minor_locator(ticker.LogitLocator(0.1))
    # ax.grid(which='both')
    # ax.set_title(title)

    # set x axis labels
    ax.set_xlim(left=1, right=len(means))
    # ax.set_ylim(bottom=8.7e-2, top=1.3e-1)
    tick_locations = [1] + list(range(5, len(means) - 1, 5)) + [len(means)]
    ax.xaxis.set_ticks(tick_locations)
    ax.axes.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator(tick_locations))

    # set y axis line locations
    if title in linear_scale:
        ax.axes.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(1))

    ax.legend()
    # axis labels
    ax.set_xlabel("Epoch")
    ax.set_ylabel(name_mappings[title]['y_label'])
    # aspect ratio
    ax.set_aspect(1. / ax.get_data_ratio())

    if save_plots:
        img = asImg(None)
        filename = f"{title}"
        directory = os.path.join(out_path)
        Path(directory).mkdir(exist_ok=True)
        img.save(os.path.join(directory, filename) + ".png", "PNG", quality=95, optimize=True)
    else:
        plt.show()


for key in means_stds.keys():
    save_or_show_plot(np.array(means_stds[key]['mean']), np.array(means_stds[key]['std']), key)
