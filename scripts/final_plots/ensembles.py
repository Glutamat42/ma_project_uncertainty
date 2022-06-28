import argparse
import ast
import json
import multiprocessing
import os
import re
import time
from collections import OrderedDict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm

from scripts.final_plots.uct_evaluation import create_evaluation
from src.utils.evaluation.helpers import convert_normed_values_to_angles


def parse_args():
    parser = argparse.ArgumentParser('')

    # @formatter:off
    parser.add_argument('--models_dir', required=True, type=str, help="This folder should only contain output folders of the ensemble models (2022_10_6__17_44_41___fnisr, 2022_10_6__17_44_09___wupra, ...).")
    parser.add_argument('--output_dir', required=True, type=str, help="Outputs will be stored here. Don't use models_dir for that! Choose empty/non existing folder.")
    parser.add_argument('--dataset_dir', required=True, type=str)
    parser.add_argument('--predictions_prefix', default='2nd_ep', type=str, help="Prefix of the csv files to use. Probably either 'ep' or '2nd_ep'")
    parser.add_argument('--target_mse', default=None, type=float, help="Don't use the best epoch during training, instead use the last one before the validation_mse got better than this value. Usefull if a specific mse value is required. Negative values will disable filtering (also no \"best mse\" filter)")
    parser.add_argument('--ensemble_size', default=None, type=int, help="specify ensemble size to calculate. If None calculate all possible sizes of 1,3,5,10,20")
    parser.add_argument('--filter_index_to_use', default=1, type=int, help="Only use with ensemble_size=1. After sorting use the model at specified index - 1. Eg 1 will use the highest ranked model, 2 will use the 2nd highest ranked model (without target_mse this would be the model with the 2nd best mse, with target_mse the model with the 2nd nearest mse and NOT the 2nd nearest checkpoint)")
    # @formatter:on

    options = parser.parse_args()
    if options.filter_index_to_use != 1 and options.ensemble_size != 1:
        raise Exception("only use filter_index_to_use together with ensemble_size=1")

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
    # options.data_path = options.dataset_dir
    # required for sample frames plot, during "normal" program execution this is set via cli interface,
    # here it's only available from the dataset meta
    # options.dataset = options.meta['dataset_type']

    return options


def load_training_config(model_path):
    return json.load(open(os.path.join(model_path, "config.json")))


# noinspection PyTypeChecker
def load_metadata(config, target_mse=None):
    path = config.models_dir

    summaries = OrderedDict()
    for model in os.listdir(path):
        print(f"model: {model}")
        model_path = os.path.join(path, model)
        filename = [i for i in os.listdir(model_path) if i.startswith("checkpoint_best") and i.endswith(".txt")][0]
        file = os.path.join(model_path, filename)
        print(f"processing file: '{file}'")

        model_summary = json.load(open(file))
        print(f"best epoch: {model_summary['epoch']}")
        summaries[model] = model_summary

    # sort asc by validation mse
    # while this sort should be irrelevant now because it is sorted again later, I'll still leave it here because it's also structuring it
    # ('log': summaries[i])
    summaries = OrderedDict(
        (i, {'log': summaries[i]}) for i in sorted(summaries, key=lambda item: summaries[item]['validation_mse']['best']))

    # load metric collector history (losses for each epoch until best)
    for model in summaries.keys():
        logdata = open(os.path.join(path, model, 'log.txt')).readlines()
        # for each row that contains "| supervised epoch " run a regex to get the stringified dict and parse it back to dict
        res = [ast.literal_eval(re.compile('(?<=finished: ).*(?= \()').search(x).group(0)) for x in logdata if "| supervised epoch " in x]

        if len(res) == 0:
            # for dino the metric collector state dict is not logged, need to work around a bit
            # get current epoch metrics
            res = [ast.literal_eval(re.compile('(?<=after : ).*(?= \()').search(x).group(0)) for x in logdata if
                   "after : {'train_dino_loss" in x]
            res = [{k: {'last_value': v} for k, v in x.items()} for x in res]
            # get unchanged_for_epochs from already loaded summaries
            res[-1]['validation_mse']['unchanged_for_epochs'] = 100 - (summaries[model]['log']['epoch'])

        # drop everything after "best" epoch (where loss increased again)
        if target_mse is None:
            if res[-1]['validation_mse']['unchanged_for_epochs'] > 0:
                res = res[:-res[-1]['validation_mse']['unchanged_for_epochs']]

        # if there is a target_mse set, search closest epoch and remove summaries after that epoch
        if target_mse is not None and target_mse >= 0:
            closest_index = len(res)-1
            smallest_distance = abs(config.target_mse - res[-1]['validation_mse']['last_value'])
            for u in range(len(res)):
                cur_distance = abs(config.target_mse - res[u]['validation_mse']['last_value'])
                if cur_distance < smallest_distance:
                    closest_index = u
                    smallest_distance = cur_distance
            res = res[:closest_index+1]
            print(f"closest value for target mse: {res[closest_index]['validation_mse']['last_value']} with error of {smallest_distance}")
        summaries[model]['epoch_summaries'] = res

    # sort (again) because sorting might be wrong now if target_mse is used
    #summaries = OrderedDict((i, summaries[i]) for i in sorted(summaries, key=lambda x: summaries[x]['epoch_summaries'][-1]['validation_mse']['last_value']))
    # if target_mse is active: sort by closest deviation
    if target_mse is not None:
        summaries = OrderedDict(
            (i, summaries[i]) for i in sorted(summaries, key=lambda x: abs(config.target_mse - summaries[x]['epoch_summaries'][-1]['validation_mse']['last_value'])))

    # iterate again over it to calculate some stats
    best_mse = []
    models_till_now = []
    with open(os.path.join(config.output_dir, "ensemble_stats.txt"), "w") as fp:
        for model in summaries.keys():
            res = summaries[model]['epoch_summaries']
            # statistics
            best_mse.append(res[-1]['validation_mse']['last_value'])  # = best in case target_mse is None
            models_till_now.append(model)

            msg = f"top {len(best_mse)} models validation: mean: {sum(best_mse) / len(best_mse)} min: {min(best_mse)} max: {max(best_mse)}"
            print(msg)
            fp.write(f"{msg}\n")

            summaries[model]['ensemble_stats_till_this_model'] = {
                'mean': sum(best_mse) / len(best_mse),
                'min': min(best_mse),
                'max': max(best_mse),
                'included_models': models_till_now.copy()
            }

    # append predictions for each epoch
    for model in tqdm.tqdm(summaries.keys()):
        model_dir = os.path.join(path, model)

        # i changed output naming schemes to fix a bug which resulted in a new bug. The following lines work around the naming bug
        alternative_pattern = False
        model_train_config = load_training_config(model_dir)
        if model_train_config['mode'] == "dino" and len([x for x in os.listdir(model_dir) if "supervised_train_for_backbone_ep" in x]) > 0:
            alternative_pattern = True

        all_files = os.listdir(model_dir)
        predictions = []
        for i in range(len(summaries[model]['epoch_summaries'])):
            if alternative_pattern:
                pattern1 = f"{config.predictions_prefix}"
                pattern2 = f"_ep{i:03}__"
                filename = [i for i in all_files if i.startswith(pattern1) and pattern2 in i][0]
            elif model_train_config['mode'] == "dino":
                pattern = f"{config.predictions_prefix}backbone_ep{i:03}_"
                filename = [i for i in all_files if i.startswith(pattern)][0]
            else:
                pattern = f"{config.predictions_prefix}{i + 1:03}_"
                filename = [i for i in all_files if i.startswith(pattern)][0]
            filepath = os.path.join(model_dir, filename)
            predictions.append(pd.read_csv(filepath))
        summaries[model]['predictions'] = predictions

    return summaries


# defining outside because multiprocessing can't pickle it otherwise
def process_ensemble_ep(ep_nr, summaries, size, output_dir_name, config):
    data_current_epoch = []
    # get predictions for top "size" models at epoch "i" if there are that many epochs, else from last epoch existing
    for model_nr in range(size):
        if config.filter_index_to_use != 1:
            # can only happen with ensemble_size = 1
            model_nr = config.filter_index_to_use -1
        model_name = list(summaries.keys())[model_nr]
        epoch_nr = ep_nr if ep_nr < len(summaries[model_name]['predictions']) else len(summaries[model_name]['predictions']) - 1
        data_current_epoch.append(summaries[model_name]['predictions'][epoch_nr])

    # cut too long results (in case of different gpu count on different runs it can happen that on one gpu there are some sample more processed -> drop them)
    min_len = min(len(x) for x in data_current_epoch)
    data_current_epoch = [x[:min_len - 1] for x in data_current_epoch]

    # calculate mean and std
    if 'std' in data_current_epoch[0]:
        means_pred = np.array([x['prediction'] for x in data_current_epoch])
        stds_pred = np.array([x['std'] for x in data_current_epoch])

        # for ensemble size one this equals to np.mean(stds_pred) with stds_pred having a length of 1 per sample
        vars = np.mean(means_pred ** 2, axis=0) - np.mean(means_pred, axis=0) ** 2 + np.mean(stds_pred ** 2, axis=0)

        means = means_pred.mean(axis=0)
        stds = vars ** 0.5
    else:
        means = np.array([x['prediction'] for x in data_current_epoch]).mean(axis=0)
        stds = np.array([x['prediction'] for x in data_current_epoch]).std(axis=0)

    # generate new dataframe
    std_series = pd.Series(stds, name='std')
    mean_series = pd.Series(means, name='prediction')
    pred_as_angle_series = pd.Series(convert_normed_values_to_angles(config.meta['label']['std'],
                                                                     config.meta['label']['mean'],
                                                                     means,
                                                                     config.meta["dataset_type"] == "driving"), name="prediction_as_angle")
    std_as_angle_series = pd.Series(stds * config.meta['label']['std'], name="std_as_angle")
    predictions_with_uct_cur_ep = pd.concat([data_current_epoch[0]['filename'],
                                             mean_series,
                                             std_series,
                                             data_current_epoch[0]['target'],
                                             pred_as_angle_series,
                                             std_as_angle_series,
                                             data_current_epoch[0]['target_as_angle']],
                                            axis=1)

    # save as csv
    prefix = config.predictions_prefix if 'ep' in config.predictions_prefix else config.predictions_prefix + "ep"
    predictions_with_uct_cur_ep.to_csv(os.path.join(output_dir_name, f"{prefix}{(ep_nr + 1):03}_predictions.csv.gz"),
                                       compression='gzip')


def calculate_ensemble(config, summaries, size):
    print(f"calculating ensemble with size {size}")
    if config.filter_index_to_use != 1:
        print(f"filter_index_to_use={config.filter_index_to_use}")

    # create output directory
    output_dir_name = os.path.join(config.output_dir, f"ensemble_{size}")
    Path(output_dir_name).mkdir(parents=True, exist_ok=False)

    # save ensemble meta info
    json.dump(summaries[list(summaries.keys())[size - 1]]['ensemble_stats_till_this_model'],
              open(os.path.join(output_dir_name, 'ensemble_meta.json'), 'w'))

    # define abort condition.
    max_epoch_number_of_ensemble = 0
    for model_nr in range(size):
        if config.filter_index_to_use != 1:
            # can only happen with ensemble_size = 1
            model_nr = config.filter_index_to_use - 1
        model_name = list(summaries.keys())[model_nr]
        if len(summaries[model_name]['predictions']) > max_epoch_number_of_ensemble:
            max_epoch_number_of_ensemble = len(summaries[model_name]['predictions'])

    # after code rework it's now that fast that multiprocessing becomes counterproductive if compression
    # is gzip. With xz it might be still a bit faster
    # -> disabling it, but leaving code here
    if True:
        for i in tqdm.tqdm(range(max_epoch_number_of_ensemble)):
            process_ensemble_ep(i, summaries, size, output_dir_name, config)
    else:
        # start calculation with pool because it's relatively slow + some status logging
        with Pool(multiprocessing.cpu_count()) as p:
            print("generating ensemble csv files. This will take a while.")
            future = p.starmap_async(func=process_ensemble_ep,
                                     iterable=[(i, summaries, size, output_dir_name, config) for i in range(max_epoch_number_of_ensemble)])

            start_timestamp = int(time.time())
            ep_done = 0
            while ep_done < max_epoch_number_of_ensemble:
                ep_done = len(os.listdir(output_dir_name))
                if ep_done > 0:
                    eta = (int(time.time()) - start_timestamp) / len(os.listdir(output_dir_name)) * (max_epoch_number_of_ensemble - ep_done)
                    eta_m, eta_s = divmod(eta, 60)
                    print(f"\r[{ep_done:03}/{max_epoch_number_of_ensemble}] eta {int(eta_m):02}:{int(eta_s):02} m:s", end="")
                else:
                    print(f"\r[{ep_done:03}/{max_epoch_number_of_ensemble}] eta --:-- m:s", end="")
                time.sleep(1)

            future.wait()

    return output_dir_name


def main(config):
    # if output_dir already exists: abort
    if Path(config.output_dir).exists():
        raise Exception("output_dir already existing")

    # create output directory
    Path(config.output_dir).mkdir(parents=True, exist_ok=False)

    # load metadata including all predictions csv files
    summaries = load_metadata(config, target_mse=config.target_mse)

    # define ensemble sizes to calculate
    if config.ensemble_size is not None:
        ensemble_sizes=[config.ensemble_size]
    else:
        ensemble_sizes = [x for x in [1, 3, 5, 10, 20, 40, 50] if x <= len(summaries)]
    # ensemble_sizes = [x for x in [10] if x <= len(summaries)]
    print(f"There are {len(summaries)} models -> calculate ensemble sizes: {ensemble_sizes}")

    # calculate ensembles and collect their output directories
    ensemble_dirs = []
    for i in ensemble_sizes:
        ensemble_dirs.append(calculate_ensemble(config, summaries, i))

    print(ensemble_dirs)
    return ensemble_dirs


if __name__ == '__main__':
    config = parse_args()

    ensemble_dirs = main(config)
    # ensemble_dirs = ["/home/markus/workspace/data/final_data/bbb_graves/outputs_dino_whole_training/3/ensemble_1"]
    #ensemble_dirs = ["/home/markus/workspace/data/final_data/bbb_graves/outputs_supervised_whole_training/2/ensemble_1/"]

    for dir in ensemble_dirs:
        create_evaluation(config, dir)
