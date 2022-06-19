import argparse
import json
import multiprocessing
import os
import re
from multiprocessing import Pool
from pathlib import Path
import matplotlib.pyplot as plt


import numpy as np
import pandas as pd

from src.utils.evaluation.helpers import generate_evaluation_metrics, generate_evaluation_plots, asImg
from src.utils.evaluation.metrics import mse


def evaluate_one_epoch(custom_config, path, csv):
    csv_path = os.path.join(path, csv)
    epoch_nr = int(re.compile('(?<=ep)\d{3}').search(csv).group(0))

    # load csv
    predictions = pd.read_csv(csv_path)

    # add missing field
    predictions['squared_error'] = np.square(predictions.prediction - predictions.target)

    # min std because uct toolbox expects std to be > 0 (does not allow std of 0)
    predictions['std'] = predictions['std'].clip(lower=1e-6)

    # calc/gen plots and metrics for cur epoch
    metrics = generate_evaluation_metrics(predictions, mse(predictions)),
    generated_plots = generate_evaluation_plots(custom_config, predictions, mode="save", epoch=epoch_nr-1)

    return metrics, generated_plots, epoch_nr


def create_evaluation(config, path, save_plots=True):
    """

    Args:
        config:
        path:
        save_plots: if false: show instead of save

    Returns:

    """
    # create config object with correct names for main project functions
    custom_config = argparse.Namespace(data_path=config.dataset_dir, dataset=config.meta['dataset_type'], output_dir=path)

    # calculate per batch ensemble metrics and generate plots

    # for debugging: single threaded
    # for epoch_nr, csv in enumerate(tqdm.tqdm(os.listdir(path))):
    #     metrics, _, index = evaluate_one_epoch(custom_config, os.path.join(path, csv), epoch_nr)
    #     all_metrics.append((index, metrics))

    print("start calculating metrcis and generating per epoch plots. This will take a bit (but not too long)")
    csv_list = [x for x in os.listdir(path) if 'predictions.csv' in x]
    with Pool(multiprocessing.cpu_count()) as p:
        future = p.starmap_async(func=evaluate_one_epoch, iterable=[(custom_config, path, csv) for csv in csv_list])
        results = future.get()

    # sort results (even for single threaded the correct order is not guaranteed because the files might be listed in wrong order)
    # and split der results to individual variables
    results.sort(key=lambda x: x[2])
    metrics = [x[0][0] for x in results]
    plots = [x[1] for x in results]
    x = [x[2] for x in results]

    # plots with metrics over time (what normally wandb does)
    def save_or_show_plot(metric, title, log_scale=True):
        plt.plot(x, metric)
        plt.yscale('log' if log_scale else 'linear')
        plt.grid(which='both')
        plt.title(title)

        if save_plots:
            img = asImg(None)
            filename = f"{title}"
            directory = os.path.join(path, 'plots_metrics')
            Path(directory).mkdir(exist_ok=True)
            img.save(os.path.join(directory, filename), "JPEG")  # WebP would save ~22%, but Windows has bad support for it
        else:
            plt.show()

    save_or_show_plot([x['validation_mse'] for x in metrics], 'validation_mse')
    save_or_show_plot([x['validation_mse_equally_weighted'] for x in metrics], 'validation_mse_equally_weighted')
    save_or_show_plot([x['validation_mse_extreme_values'] for x in metrics], 'validation_mse_extreme_values')
    save_or_show_plot([x['validation_mae'] for x in metrics], 'validation_mae')
    save_or_show_plot([x['validation_rmse'] for x in metrics], 'validation_rmse', log_scale=False)

    save_or_show_plot([x['uncertainty']['miscal_area'] for x in metrics], 'miscal_area', log_scale=False)
    save_or_show_plot([x['uncertainty']['nll'] for x in metrics], 'nll', log_scale=False)
    save_or_show_plot([x['uncertainty']['crps'] for x in metrics], 'crps')
    save_or_show_plot([x['uncertainty']['sharpness'] for x in metrics], 'sharpness')

    # save metrics (array with all metrics per epoch for each epoch)
    json.dump(metrics, open(os.path.join(path, "plots_metrics", "metrics.json"), "w"))
    # TODO: summary (best values, best epoch etc)

    print("done")



