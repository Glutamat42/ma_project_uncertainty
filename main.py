import datetime
import json
import logging
import logging.config
import os
import shutil
import sys
import time
from pathlib import Path

import configargparse
import torch
import torch.distributed

import wandb
from torch.backends import cudnn

from torchvision import models as torchvision_models

import matplotlib.pyplot as plt  # import here to disable log spam

from src.custom_exceptions import ModelNoProgressException

logging.config.dictConfig({'disable_existing_loggers': True, 'version': 1})
logging.getLogger('PIL').setLevel(logging.WARNING)

from src.train_dino import train_dino
from src.utils.evaluation.helpers import generate_evaluations, generate_evaluation_metrics

from src.project_enums import EnumHeadTypes, EnumUncertaintyTypes, EnumSchedulerTypes, EnumMode, EnumDatasets, EnumSamplerTypes, \
    EnumBBBCostFunction

from src.utils import utils, common
from src.utils.common import EnumAction, dist_print, init_logging, backup_project
from src.utils.create_models import create_and_load_model, create_and_load_backbone
from src.utils.utils import get_sha
from src.supervised_code import train_supervised, train_head
from src.test_code import test

log = logging.getLogger("main")


def load_config():
    torchvision_archs = sorted(name for name in torchvision_models.__dict__
                               if name.islower() and not name.startswith("__")
                               and callable(torchvision_models.__dict__[name]))

    # @formatter:off
    parser = configargparse.ArgParser(default_config_files=['configs/default.ini'] if Path('configs/default.ini').exists() else [])
    parser.add_argument('-c', '--config', required=False, is_config_file=True, help='config file path')

    general_parameters = parser.add_argument_group("general parameters", "these parameters are always relevant")
    supervised_parameters = parser.add_argument_group("supervised parameter", "params for supervised training (or head, if training dino)")
    supervised_parameters_fine_tuning = parser.add_argument_group("supervised parameter fine tuning", "finetune hyperparams")
    vit_parameters = parser.add_argument_group("vit params", "relevant if using arch = vit_*")
    dino_parameters = parser.add_argument_group("dino params", "parameters for semi-supervised dino training")
    other_parameters = parser.add_argument_group("other params", "...")
    test_parameters = parser.add_argument_group("test params", "params for testing stage")
    uncertainty_parameters = parser.add_argument_group("uncertainty params", "params for uncertainty")

    general_parameters.add_argument('--mode', type=EnumMode, action=EnumAction, default=EnumMode.none, help="""Training mode""")
    general_parameters.add_argument('--name', required=False, default=common.generate_default_filename(), type=str, help="The name is used at different places to identify a run")
    general_parameters.add_argument('--tags', type=str, nargs='+', default=[], help="Tags for this run. Allows grouping in wandb.")
    general_parameters.add_argument('--output_base_dir', default="/home/markus/outputs", type=str, help='Base output directory.Logs and checkpoints will be saved to <output_base_dir>/<name>.')
    general_parameters.add_argument('--seed', default=0, type=int, help='Random seed.')
    general_parameters.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    general_parameters.add_argument('--dataset', default=EnumDatasets.drive360, type=EnumDatasets, action=EnumAction, help='Which dataset type to use')
    general_parameters.add_argument('--resume', default=False, type=utils.bool_flag, help='Set to true to enable resuming training. Works in combination with wandb_run_id. Only dino is supported')
    general_parameters.add_argument('--override_output_dir', default=None, type=str, help='Override automatically generated output directory (based on base_output_dir). Not recommended except if resuming interrupted training')
    general_parameters.add_argument('--wandb_run_id', default=None, type=str, help='Set to wandb id of run to resume. Works in combination with resume, Only dino is supported')
    general_parameters.add_argument('--wandb_project', default="ma_uncertainty", type=str, help='wandb project name')
    # general backbone parameter
    general_parameters.add_argument('--arch', default='vit_small', type=str,choices=['vit_tiny', 'vit_small', 'vit_base', 'xcit', 'deit_tiny', 'deit_small'] + ['nvidia'] + torchvision_archs, help="""Name of architecture to train. For quick experiments with ViTs, we recommend using vit_tiny or vit_small.""")
    # general_parameters.add_argument('--arch', default='vit_small', type=str,choices=['vit_tiny', 'vit_small', 'vit_base', 'xcit', 'deit_tiny', 'deit_small'] + ['nvidia'] + torchvision_archs + torch.hub.list("facebookresearch/xcit:main"),help="""Name of architecture to train. For quick experiments with ViTs, we recommend using vit_tiny or vit_small.""")
    general_parameters.add_argument('--net_resolution', type=int, nargs='+', default=(271, 224), help='width, height')
    vit_parameters.add_argument('--patch_size', default=16, type=int, help="""Size in pixels of input square patches - default 16 (for 16x16 patches). Using smaller values leads to better performance but requires more memory. Applies only for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    vit_parameters.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate, for vit and xcit")
    general_parameters.add_argument('--plateau_detection_epochs', default=10, type=int, help="Number of epochs with no improvement before early stopping. For dino theres a separate parameter --dino_early_stopping because of it's different default value")
    general_parameters.add_argument('--repeat_on_fail', default=None, type=int, help="In some scenarios this config allows to abort training if model is collapsing. It will restart training until either it failed --repeat_on_fail times or training was successfull (does not mean loss have to be good). This is the total number of tries, meaning '1' will result in 0 retries. If this setting is set to None (omitted) it will train without aborting.")


    ### test
    general_parameters.add_argument('--data_path_test', default=None, type=str, help='Please specify path to the test data. Defaults to "data_path".')
    general_parameters.add_argument('--csv_name_test', default='test.csv', type=str)
    general_parameters.add_argument('--csv_name_test2', default=None, type=str, help="If not None: will perform a 2nd testrun with the specified csv file")
    test_parameters.add_argument("--show_predictions", default=False, type=utils.bool_flag, help="visually show predictions")
    test_parameters.add_argument('--pretrained_weights_backbone', default='', type=str, help="Path to pretrained weights to evaluate.")
    test_parameters.add_argument('--pretrained_weights_head', default='', type=str, help="Path to pretrained weights to evaluate.")


    ### supervised
    supervised_parameters.add_argument('--skip_reducing_layer', type=utils.bool_flag, default=True, help="In some models (eg resnet) there is a layer greatly reducing the output dimensionality (eg avgpool). This switch sets them to nn.Identity()")
    supervised_parameters.add_argument('--batch_size', default=256, type=int, help='number of distinct images loaded on all GPUs.')
    supervised_parameters.add_argument('--epochs', default=60, type=int, help='Number of epochs of training.')
    supervised_parameters.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of linear warmup (highest LR used during training). The learning rate is linearly scaled with the batch size, and specified here for a reference batch size of 256.""")
    supervised_parameters.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the weight decay. With ViT, a smaller value at the beginning of training works well.""")
    supervised_parameters.add_argument('--optimizer', default='adamw', type=str, choices=['adamw', 'sgd', 'lars', 'adam'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    supervised_parameters.add_argument('--data_path', default="/home/markus/datasets/drive360_small_prepared/", type=str, help='Please specify path to the ImageNet training data.')
    supervised_parameters.add_argument('--csv_name', default='train.csv', type=str)
    supervised_parameters.add_argument('--lr_scheduler', default=EnumSchedulerTypes.none, type=EnumSchedulerTypes, action=EnumAction, help='select scheduler to use for learning rate')
    supervised_parameters.add_argument('--wd_scheduler', default=EnumSchedulerTypes.none, type=EnumSchedulerTypes, action=EnumAction, help='select scheduler to use for weight decay')
    supervised_parameters.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the end of optimization. We use a cosine LR schedule with linear warmup.""")
    supervised_parameters.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the weight decay. We use a cosine schedule for WD and using a larger decay by the end of training improves performance for ViTs.""")
    supervised_parameters.add_argument("--warmup_epochs", default=0, type=int, help="Number of epochs for the linear learning-rate warm up.")
    supervised_parameters.add_argument("--sampler_type", default=EnumSamplerTypes.DistributedSampler, type=EnumSamplerTypes, action=EnumAction, help='Sampler type, DistributedSampler is a "simple" sampler, UpscaleDistributedSampler scales up minority classes, DynamicDistributedSampler uses dynamic downsampling (see catalys docs: DynamicBalanceClassSampler)')
    supervised_parameters.add_argument('--clip_grad', type=float, default=None, help="""Maximal parameter gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can help optimization for larger ViT architectures. 0 for disabling.""")
    supervised_parameters.add_argument('--head_arch', type=EnumHeadTypes, default=EnumHeadTypes.RegressionHead, action=EnumAction, help="select head architecture for supervised learning")
    supervised_parameters.add_argument('--saveckp_freq', default=None, type=int, help='Save checkpoint every x epochs.')
    # fine tuning
    supervised_parameters_fine_tuning.add_argument('--RandomResizedCrop_scale_min', default=0.8, type=float, help="Set both to 1 to disable")
    supervised_parameters_fine_tuning.add_argument('--RandomResizedCrop_scale_max', default=1.0, type=float)
    supervised_parameters_fine_tuning.add_argument('--ColorJitter_brightness', default=0.5, type=float)
    supervised_parameters_fine_tuning.add_argument('--ColorJitter_contrast', default=0.4, type=float)
    supervised_parameters_fine_tuning.add_argument('--ColorJitter_saturation', default=0.3, type=float)
    supervised_parameters_fine_tuning.add_argument('--ColorJitter_hue', default=0.03, type=float)
    supervised_parameters_fine_tuning.add_argument('--ColorJitter_probability', default=0.8, type=float, help="Set to 0 to disable ColorJitter")
    supervised_parameters_fine_tuning.add_argument('--GaussianBlur_kernel_size', default=3, type=int)
    supervised_parameters_fine_tuning.add_argument('--GaussianBlur_sigma_min', default=0.1, type=float)
    supervised_parameters_fine_tuning.add_argument('--GaussianBlur_sigma_max', default=5.0, type=float)
    supervised_parameters_fine_tuning.add_argument('--GaussianBlur_probability', default=0.8, type=float, help="Set to 0 to disable GaussianBlur")


    ### uncertainty
    uncertainty_parameters.add_argument('--uncertainty_type', default=EnumUncertaintyTypes.none, type=EnumUncertaintyTypes, action=EnumAction, help="""Uncertainty algo""")
    uncertainty_parameters.add_argument('--mcbn_batchsize', default=32, type=int, help="""Batchsize for mcbn bn mean and var calculation. Too low values will be unstable. """)
    uncertainty_parameters.add_argument('--uncertainty_iters', default=64, type=int, help="""Number of predictions per sample for uncertainty approaches where applicable (not for aleatoric and ensembles). Higher values should provide more stable estimations""")
    uncertainty_parameters.add_argument('--aleatoric_complex_head', default=False, type=utils.bool_flag, help="""use more complex head for aleatoric uncertainty, if available for head (only RegressionHead)""")
    uncertainty_parameters.add_argument('--bbb_cost_function', default=EnumBBBCostFunction.graves, type=EnumBBBCostFunction, action=EnumAction, help="Select cost function for BBB")


    ### dino
    dino_parameters.add_argument('--dino_saveckp_freq', default=None, type=int, help='Save checkpoint every x epochs.')
    dino_parameters.add_argument('--dino_test_freq', default=5, type=int, help='train head and test; 0 for only after last dino epoch')
    dino_parameters.add_argument('--dino_batch_size', default=64, type=int, help='number of distinct images loaded on all GPUs.')
    dino_parameters.add_argument('--dino_epochs', default=100, type=int, help='Number of epochs of training.')
    dino_parameters.add_argument("--dino_lr", default=0.0005, type=float, help="""Learning rate at the end of linear warmup (highest LR used during training). The learning rate is linearly scaled with the batch size, and specified here for a reference batch size of 256.""")
    dino_parameters.add_argument('--dino_weight_decay', type=float, default=0.04, help="""Initial value of the weight decay. With ViT, a smaller value at the beginning of training works well.""")
    dino_parameters.add_argument('--dino_optimizer', default='adamw', type=str, choices=['adamw', 'sgd', 'lars', 'adam'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    dino_parameters.add_argument('--dino_csv_name', default=None, type=str, help="defaults to csv_name")
    dino_parameters.add_argument('--dino_data_path', default=None, type=str, help='Please specify path to the training data. Defaults to --data_path')
    dino_parameters.add_argument('--dino_lr_scheduler', default=EnumSchedulerTypes.cosine, type=EnumSchedulerTypes, action=EnumAction, help='select scheduler to use for learning rate')
    dino_parameters.add_argument('--dino_wd_scheduler', default=EnumSchedulerTypes.cosine, type=EnumSchedulerTypes, action=EnumAction, help='select scheduler to use for weight decay')
    dino_parameters.add_argument('--dino_min_lr', type=float, default=1e-6, help="""Target LR at the end of optimization. We use a cosine LR schedule with linear warmup.""")
    dino_parameters.add_argument('--dino_weight_decay_end', type=float, default=0.4, help="""Final value of the weight decay. We use a cosine schedule for WD and using a larger decay by the end of training improves performance for ViTs.""")
    dino_parameters.add_argument("--dino_sampler_type", default=EnumSamplerTypes.DistributedSampler, type=EnumSamplerTypes, action=EnumAction, help='Sampler type, DistributedSampler is a "simple" sampler, UpscaleDistributedSampler scales up minority classes, DynamicDistributedSampler uses dynamic downsampling (see catalys docs: DynamicBalanceClassSampler)')
    dino_parameters.add_argument("--dino_early_stopping", default=25, type=int, help='Stop training if no improvement for given amount of epochs. Determining "improvement" might not only depend on the "main" loss function but also other parameters. Testing probably does not happen every epoch, the actual value of this variable might be higher: ceil(dino_early_stopping / dino_test_freq) * dino_test_freq')
    # Model parameters
    dino_parameters.add_argument('--dino_out_dim', default=65536, type=int, help="""Dimensionality of the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    dino_parameters.add_argument('--dino_norm_last_layer', default=True, type=utils.bool_flag, help="""Whether or not to weight normalize the last layer of the DINO head. Not normalizing leads to better performance but can make the training unstable. In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    dino_parameters.add_argument('--dino_momentum_teacher', default=0.996, type=float, help="""Base EMA parameter for teacher update. The value is increased to 1 during training with cosine schedule. We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    dino_parameters.add_argument('--dino_use_bn_in_head', default=False, type=utils.bool_flag, help="Whether to use batch normalizations in projection head (Default: False)")
    # Temperature teacher parameters
    dino_parameters.add_argument('--dino_warmup_teacher_temp', default=0.04, type=float, help="""Initial value for the teacher temperature: 0.04 works well in most cases. Try decreasing it if the training loss does not decrease.""")
    dino_parameters.add_argument('--dino_teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup) of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend starting with the default value of 0.04 and increase this slightly if needed.""")
    dino_parameters.add_argument('--dino_warmup_teacher_temp_epochs', default=0, type=int, help='Number of warmup epochs for the teacher temperature (Default: 30).')
    dino_parameters.add_argument('--dino_freeze_last_layer', default=1, type=int, help="""Number of epochs during which we keep the output layer fixed. Typically doing so during the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    dino_parameters.add_argument("--dino_warmup_epochs", default=10, type=int, help="Number of epochs for the linear learning-rate warm up.")
    # Multi-crop parameters
    dino_parameters.add_argument('--dino_global_crops_scale', type=float, nargs='+', default=(0.4, 1.), help="""Scale range of the cropped image before resizing, relatively to the origin image. Used for large global view cropping. When disabling multi-crop (--dino_local_crops_number 0), we recommand using a wider range of scale ("--dino_global_crops_scale 0.14 1." for example)""")
    dino_parameters.add_argument('--dino_local_crops_number', type=int, default=8, help="""Number of small local views to generate. Set this parameter to 0 to disable multi-crop training. When disabling multi-crop we recommend to use "--dino_global_crops_scale 0.14 1." """)
    dino_parameters.add_argument('--dino_local_crops_scale', type=float, nargs='+', default=(0.05, 0.4), help="""Scale range of the cropped image before resizing, relatively to the origin image. Used for small local view cropping of multi-crop.""")
    dino_parameters.add_argument('--dino_clip_grad', type=float, default=3.0, help="""Maximal parameter gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can help optimization for larger ViT architectures. 0 for disabling.""")


    other_parameters.add_argument("--dist_url", default="env://", type=str, help="""url used to set up distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    other_parameters.add_argument("--local_rank", default=None, type=int, help="Unused, left for compatibility, use environment variable instead")
    other_parameters.add_argument('--use_fp16', type=utils.bool_flag, default=False, help="""Whether or not to use half precision for training. Improves training time and memory requirements, but can provoke instability and slight decay of performance. We recommend disabling mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    # @formatter:on

    options = parser.parse_args()

    # print(config)
    # print("----------")
    # print(parser.format_help())
    # print("----------")
    if 'LOCAL_RANK' in os.environ and os.environ['LOCAL_RANK'] == 0:
        dist_print(parser.format_values())  # useful for logging where different settings came from

    return options


def main(config):
    utils.fix_random_seeds(config.seed)
    cudnn.benchmark = True  # might cause randomness / prevent reproducibility; see https://pytorch.org/docs/stable/notes/randomness.html

    log.info(f"git state - {get_sha()}")

    start_time = time.time()

    if config.mode == EnumMode.test:
        if utils.is_main_process():
            model, backbone_epoch = create_and_load_model(config)
            if config.csv_name_test2 is not None:
                # perform test run with 2nd csv file. Only predictions csv will be saved, nothing else will happen
                validation_loss, predictions = test(model,
                                          config,
                                          config.csv_name_test2,
                                          save_predictions=True,
                                          save_predictions_filename_prefix=f"2nd_ep{str(backbone_epoch).zfill(3)}_")
                _metrics = generate_evaluation_metrics(predictions, validation_loss)
                log.debug(f"2nd csv metrics: {_metrics}")
            loss, predictions = test(model,
                                     config,
                                     config.csv_name_test,
                                     save_predictions=True,
                                     save_predictions_filename_prefix=f"ep{str(backbone_epoch).zfill(3)}_")

            metrics, _ = generate_evaluations(config, predictions, loss, 'save')
            log.info(metrics)
    elif config.mode == EnumMode.supervised:
        train_supervised(config)
    elif config.mode == EnumMode.dino:
        train_dino(config)
    elif config.mode == EnumMode.head:
        backbone, embed_dim, backbone_info = create_and_load_backbone(config)
        train_head(config,
                   backbone=backbone,
                   embed_dim=embed_dim,
                   cur_epoch=backbone_info['epoch'],
                   backbone_uuid=backbone_info['uuid'],
                   silent_mode=False,
                   wandb_watch_model=True)
    elif config.mode == EnumMode.none:
        log.error(f"mode == none is only allowed if resuming training. Either set a valid mode or check your resume config")
        sys.exit(1)
    else:
        raise Exception(f'unknown mode {config.mode}')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    log.info('Runtime {}'.format(total_time_str))


def additional_config_ops_resume(config):
    # set per gpu batch size
    config.batch_size_per_gpu = int(config.batch_size / utils.get_world_size())
    config.dino_batch_size_per_gpu = int(config.dino_batch_size / utils.get_world_size())
    log.info(f"actual batchsize per gpu: {config.batch_size_per_gpu}, for dino: {config.dino_batch_size_per_gpu}")

    return config


def additional_config_ops(config):
    # set per gpu batch size
    config.batch_size_per_gpu = int(config.batch_size / utils.get_world_size())
    config.dino_batch_size_per_gpu = int(config.dino_batch_size / utils.get_world_size())
    log.info(f"actual batchsize per gpu: {config.batch_size_per_gpu}, for dino: {config.dino_batch_size_per_gpu}")

    # data_path_test defaults to data_path
    if config.data_path_test is None:
        config.data_path_test = config.data_path

    if config.dino_csv_name is None:
        config.dino_csv_name = config.csv_name

    if config.dino_data_path is None:
        config.dino_data_path = config.data_path

    # some validation
    assert config.warmup_epochs < config.epochs
    assert config.dino_warmup_epochs < config.dino_epochs

    # warnings
    if config.sampler_type == EnumSamplerTypes.DynamicDistributedSampler:
        log.warning("WARNING: DynamicDistributedSampler is broken")
    if config.mode == EnumMode.dino and config.epochs > 10:
        log.warning("WARNING: dino mode is selected and head epoch count is high (above 10). "
                    "This will result in higher training durations and is probably unnecessary. Consider decreasing it.")

    # exit in invalid states
    if config.uncertainty_type == EnumUncertaintyTypes.aleatoric_mcbn and config.mode != EnumMode.test:
        raise Exception("aleatoric mcbn is only valid for evaluation. To train aleatoric mcbn run training in aleatoric mode")

    # add dataset meta file
    train_meta_path = os.path.join(config.data_path, 'meta.json')
    train_dino_meta_path = os.path.join(config.dino_data_path, 'meta.json')
    test_meta_path = os.path.join(config.data_path_test, 'meta.json')
    if not Path(test_meta_path).is_file():
        raise Exception(f"could not find test dataset meta file at {test_meta_path}")
    if not Path(train_meta_path).is_file():
        raise Exception(f"could not find train dataset meta file at {train_meta_path}")
    config.meta = {"train": json.load(open(train_meta_path, "r")),
                   "train_dino": json.load(open(train_dino_meta_path, "r")),
                   "test": json.load(open(test_meta_path, "r"))}

    # rename some parameters to more generic names
    if 'canSteering' in config.meta['train']:
        config.meta['train']['label'] = config.meta['train']['canSteering']
    if 'canSteering' in config.meta['test']:
        config.meta['test']['label'] = config.meta['test']['canSteering']
    if "dataset_type" not in config.meta['train']:
        config.meta['train']["dataset_type"] = "driving"
    if "dataset_type" not in config.meta['test']:
        config.meta['test']["dataset_type"] = "driving"
    if "dataset_type" not in config.meta['train_dino']:
        config.meta['train_dino']["dataset_type"] = "driving"
    if "dataset_type" not in config.meta['train_dino']:
        config.meta['train_dino']["dataset_type"] = "driving"

    return config


def init_and_backup():
    config = load_config()
    utils.init_distributed_mode(config)

    # sync name between processes, this is required if there is no name set and it will be generated randomly instead
    name = [config.name] if utils.is_main_process() else [None]
    torch.distributed.broadcast_object_list(name, src=0)
    config.name = name[0]
    log.info(f"name of run: {config.name}")

    # set output directory
    if config.override_output_dir is not None:
        config.output_dir = config.override_output_dir
    else:
        config.output_dir = os.path.join(config.output_base_dir,
                                         f"{datetime.date.today().isocalendar()[0]}_{datetime.date.today().isocalendar()[1]}",
                                         config.name)
    dist_print(f"output directory: {config.output_dir}")

    # create output dir, required for init_logging
    resume_training = False

    if Path(config.output_dir).exists():
        # Be careful if using this hacky switch. It could result in permanent data loss if this folder contains relevant data
        if 'DELETE_EXISTING_FOLDER' in os.environ and int(os.environ['DELETE_EXISTING_FOLDER']) == 1:
            shutil.rmtree(config.output_dir)
        else:
            resume_training = True
            log.info(f"output_dir {config.output_dir} already exists.")
    # for consistency between threads each thread has to have this check finished before main thread can create the directory
    torch.distributed.barrier()
    if not Path(config.output_dir).exists():
        if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:  # is not distributed or is local rank 0
            Path(config.output_dir).mkdir(parents=True)
            log.debug(f"created directory {config.output_dir}")
    torch.distributed.barrier()  # had exceptions in init_logging because the other processes were faster than the main process
    init_logging(target_dir=config.output_dir, rank=utils.get_rank())

    if resume_training:
        if config.resume:
            log.info("Folder already existed and resume flag is set to true: resume training. "
                     "Its (currently) only planned to resume dino training.")
            log.info("wandb does not support resuming runs started as part of a sweep")
            log.warning("Logs are appended to wandb. Since training probably did not stop exactly at epoch end this might seem a bit "
                        "strange, like it would not resume correctly. But that's normal and expected based on how wandb logging and resume "
                        "works.")

            old_config = vars(config)
            config.__dict__ = json.load(open(os.path.join(config.output_dir, "config.json")))

            # keep some values / allow changing for resuming
            config.rank = old_config['rank']
            config.world_size = old_config['world_size']
            config.gpu = old_config['gpu']
            config.num_workers = old_config['num_workers']

            # re-set some values
            config = additional_config_ops_resume(config)

            log.debug("config loaded and replaced from config.json")
            if utils.is_main_process():
                if "wandb_run_id" in config:
                    wandb.init(id=config.wandb_run_id, resume="must", project=config.wandb_project)
                    log.debug("resume wandb run")
                else:
                    wandb.init(config=vars(config),
                               project=config.wandb_project,  # disabled for sweep
                               entity="glutamat",
                               name=config.name,
                               tags=[config.arch, config.mode.value, config.dataset] + config.tags)
                    log.warning(f"loaded config is too old and does not contain wandb_run_id so resuming wandb progress is not possible "
                                f"-> starting new wandb run. run_id of new run: {wandb.run.id}")
        else:
            log.error(f"Folder {config.output_dir} already existed but resume flag is False.")
            sys.exit(1)
    else:
        config = additional_config_ops(config)

        if utils.is_main_process():
            wandb.init(config=vars(config),
                       project=config.wandb_project,  # disabled for sweep
                       entity="glutamat",
                       name=config.name,
                       tags=[config.arch, config.mode.value, config.dataset] + config.tags)

            # get run_id, pre-generating is not possible because sweep does not support setting id manually
            config.wandb_run_id = wandb.run.id

            backup_project(config.output_dir)
            with open(os.path.join(config.output_dir, 'config.json'), 'w') as fp:
                json.dump(vars(config), fp, indent=4)
            log.debug('Backup source code and config')
    if utils.is_main_process():
        wandb_url = wandb.run.get_url()
        log.info(f"wandb url: {wandb_url if wandb_url is not None else 'not available (probably in offline mode)'}")

    config.resume_training = resume_training

    return config


if __name__ == '__main__':
    config = init_and_backup()

    remaining_tries = config.repeat_on_fail if config.repeat_on_fail is not None else 1
    # if config.repeat_on_fail is None this loop doesn't matter at all, the exception will not be thrown if config.repeat_on_fail is None
    while remaining_tries > 0:
        try:
            main(config)
            break  # training was successful (or at least not interrupted by ModelNoProgressException
        except ModelNoProgressException as e:
            remaining_tries -= 1
            log.warning(e)
            log.warning(f"ModelNoProgress Exception was thrown. {remaining_tries} tries remaining.")
            if remaining_tries > 0:
                config.seed += 1
                log.info(f"setting seed to {config.seed}")
            else:
                if not utils.is_main_process():
                    time.sleep(1)  #make sure main process quits first so that main process log (and stdout) contains the error messages
                log.error(f"no successful training run after {config.repeat_on_fail} tries. Giving up")
                sys.exit(1)
    log.info("training finished")
