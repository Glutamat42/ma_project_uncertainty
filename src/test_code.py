import logging
import os
import time
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cv2 import cv2
from torch import nn
from torchvision.transforms import transforms

from src.uncertainty.bbb_utils import bbb_predict, bbb_aleatoric_predict
from src.utils.common import dist_tqdm
from src.utils.create_models import get_data_loader
from src.project_enums import EnumSamplerTypes, EnumUncertaintyTypes
from src.utils import utils
from src.utils.evaluation.helpers import convert_normed_values_to_angles
from src.uncertainty.uncertainty import mcbn_predict, mcbn_get_bn_params, mcbn_predict_aleatoric
from src.utils.evaluation.plots import draw_values_on_frame


def test(model, config, csv_name, save_predictions=False, save_predictions_filename_prefix=''):
    """ run test, model should already be loaded and including feature extractor and head

    Args:
        model:
        config:

    Returns:

    """
    log = logging.getLogger("test")
    # copy.deepcopy has many problems, eg seem to no work (anymore?!?!) with blitz. pickle seems to be more reliable and is pretty fast
    _forward_hooks = model._forward_hooks  # workaround for wandb.watch (cannot be pickled)
    model._forward_hooks = None
    _model = pickle.loads(pickle.dumps(model))
    model._forward_hooks = _forward_hooks
    model = _model

    model.cuda()
    model.eval()

    # synchronize batch norms (if any)
    # disabled DDP since its not much difference but results in problems e.g. with syncing predictions
    # if utils.has_batchnorms(model):
    #     model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # model = nn.parallel.DistributedDataParallel(model, device_ids=[config.gpu])
    # assert utils.is_main_process()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.Resize((config.net_resolution[1], config.net_resolution[0]))
    ])
    test_loader = get_data_loader(data_path=config.data_path_test,
                                  csv_name=csv_name,
                                  batch_size=config.batch_size_per_gpu,
                                  num_workers=config.num_workers,
                                  shuffle=False,
                                  transform=transform,
                                  dataset=config.dataset,
                                  manual_seed=config.seed,
                                  enable_src_image_passthrough=False,
                                  sampler_type=EnumSamplerTypes.SequentialSampler)  # Sequential sampler because running in non-distributed mode

    if config.uncertainty_type in [EnumUncertaintyTypes.mcbn, EnumUncertaintyTypes.aleatoric_mcbn]:
        bn_std_var_buffer = mcbn_get_bn_params(config, model, transform)

    lossfunc = nn.MSELoss().cuda()
    total_loss = 0
    rows_list = []
    pbar = dist_tqdm(test_loader, desc=f"test")
    for i, batch in enumerate(pbar):
        src_inp, target, meta = batch

        # move to gpu
        inp = src_inp['cameraFront'].cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        with torch.no_grad():
            if config.uncertainty_type == EnumUncertaintyTypes.bayes_by_backprop:
                output, std = bbb_predict(model, inp, num_predictions=config.uncertainty_iters)
            elif config.uncertainty_type == EnumUncertaintyTypes.mcbn:
                # noinspection PyUnboundLocalVariable
                output, std, _ = mcbn_predict(model, inp, bn_std_var_buffer, config.uncertainty_iters)
            elif config.uncertainty_type == EnumUncertaintyTypes.aleatoric:
                _output = model(inp)
                output = _output[:, 0].view(-1, 1)
                # during training std is squared to calculate var.
                # To avoid negative std values and mimic the training behavior .abs() is required here
                std = _output[:, 1].view(-1,1).abs()
            elif config.uncertainty_type == EnumUncertaintyTypes.aleatoric_bbb:
                output, std = bbb_aleatoric_predict(model, inp, num_predictions=config.uncertainty_iters)
            elif config.uncertainty_type == EnumUncertaintyTypes.aleatoric_mcbn:
                # noinspection PyUnboundLocalVariable
                output, std, _ = mcbn_predict_aleatoric(model, inp, bn_std_var_buffer, config.uncertainty_iters)
            else:
                output = model(inp)

        if config.show_predictions:
            for j in range(len(output)):
                output_val = convert_normed_values_to_angles(config.meta["test"]['label']['std'],
                                                             config.meta["test"]['label']['mean'],
                                                             output[j].item(),
                                                             config.meta["test"]["dataset_type"] == "driving")
                target_val = convert_normed_values_to_angles(config.meta["test"]['label']['std'],
                                                             config.meta["test"]['label']['mean'],
                                                             target[j].item(),
                                                             config.meta["test"]["dataset_type"] == "driving")
                if 'orig_image' in meta:
                    frame = meta['orig_image'][j]
                else:
                    frame = src_inp['cameraFront'][j]

                # TODO: dont know the exact type of frame, but it isnt ndarray. Since i fixed draw_values_on_frame to work with ndarray this
                # call might fail (untested on changes)
                image = draw_values_on_frame(frame, target_val, output_val)

                cv2.imshow("cur img", image)
                cv2.waitKey()

        loss = lossfunc(output, target)  # TODO: this is an eval func
        for j in range(len(output)):
            # maybe faster if copying output to cpu first
            log_dict = {'target': target[j].item(), 'prediction': output[j].item(), 'filename': meta['filename'][j]}
            if config.uncertainty_type != EnumUncertaintyTypes.none:
                log_dict['std'] = std[j].item()
            rows_list.append(log_dict)

        total_loss += loss.item()
        if utils.is_main_process():
            pbar.set_postfix({'avg_loss': "{:.4f}".format(total_loss / (i + 1))})

    df = pd.DataFrame(rows_list)
    df["prediction_as_angle"] = convert_normed_values_to_angles(config.meta['test']['label']['std'],
                                                                config.meta['test']['label']['mean'],
                                                                df["prediction"],
                                                                config.meta["test"]["dataset_type"] == "driving")
    df["target_as_angle"] = convert_normed_values_to_angles(config.meta['test']['label']['std'],
                                                            config.meta['test']['label']['mean'],
                                                            df["target"],
                                                            config.meta["test"]["dataset_type"] == "driving")
    df['squared_error'] = np.square(df.prediction - df.target)

    if 'std' in df:
        df["std_as_angle"] = df['std'] * config.meta['test']['label']['std']

    if utils.is_main_process() and save_predictions:
        out_path = os.path.join(config.output_dir, save_predictions_filename_prefix + str(int(time.time())) + '_predictions.csv.gz')
        if Path(out_path).exists():
            raise Exception(f"predictions file {out_path} already exists")
        log.info(f"predictions will be saved only on main process to {out_path} ")
        df.to_csv(out_path, compression='gzip')

    return total_loss / len(test_loader), df
