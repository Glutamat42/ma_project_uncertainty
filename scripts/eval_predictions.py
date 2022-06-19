import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.evaluation.helpers import convert_normed_values_to_angles, generate_evaluations


def parse_args():
    parser = argparse.ArgumentParser('')

    # @formatter:off
    parser.add_argument('--source_file', required=True, type=str, nargs='+', help="predictions csv file, if multiple given it's assumed they are individual ensemble runs. They have to be all for the exact same test set!")
    parser.add_argument('--dataset_dir', required=True, type=str)
    # @formatter:on

    options = parser.parse_args()

    # add dataset meta file
    meta_file_path = os.path.join(options.dataset_dir, 'meta.json')
    if not Path(meta_file_path).is_file():
        raise Exception(f"could not find dataset meta file at {meta_file_path}")
    options.meta = json.load(open(meta_file_path, "r"))
    if 'canSteering' in options.meta:
        options.meta['label'] = options.meta['canSteering']
    if "dataset_type" not in options.meta:
        options.meta["dataset_type"] = "driving"

    # for sample frames plot the name "data_path" is required
    options.data_path = options.dataset_dir
    # required for sample frames plot, during "normal" program execution this is set via cli interface,
    # here it's only available from the dataset meta
    options.dataset = options.meta['dataset_type']

    return options


def load_predictions_file(source_file):
    assert isinstance(source_file, str)
    return pd.read_csv(source_file,
                       dtype={'filename': object,
                              'prediction': np.float32,
                              'std': np.float32,
                              'target': np.float32,
                              'prediction_as_angle': np.float32,
                              'target_as_angle': np.float32,
                              })


def load_ensemble(source_file, meta):
    files = [load_predictions_file(item) for item in source_file]

    if 'std' in files[0]:
        print("WARNING: source files already contain 'std', ignoring it")

    row_list = []
    for row_i in range(len(files[0])):
        pred_array = np.array([x.iloc[row_i]['prediction'] for x in files])
        mean = pred_array.mean()
        std = pred_array.std()
        row_list.append({'filename': files[0].iloc[row_i]['filename'],
                         'prediction': mean,
                         'std': std,
                         'target': files[0].iloc[row_i]['target'],
                         'prediction_as_angle': convert_normed_values_to_angles(meta['label']['std'],
                                                                                meta['label']['mean'],
                                                                                mean,
                                                                                config.meta["dataset_type"] == "driving"),
                         'std_as_angle':  std * meta['label']['std'],
                         'target_as_angle': files[0].iloc[row_i]['target_as_angle']})
    return pd.DataFrame(row_list)


def main(config):
    if len(config.source_file) == 1:
        df = load_predictions_file(config.source_file[0])
    else:
        df = load_ensemble(config.source_file, config.meta)

    df['squared_error'] = np.square(df.prediction - df.target)

    metrics, _ = generate_evaluations(config, df, df['squared_error'].mean(), 'show')
    print(metrics)


if __name__ == '__main__':
    config = parse_args()
    main(config)
