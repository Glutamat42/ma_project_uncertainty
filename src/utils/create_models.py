import logging
import math

import catalyst.data
import torch
from torch import nn

from torch.utils.data import DataLoader, SequentialSampler

import src.models.vision_transformer as vits
from torchvision import models as torchvision_models
import blitz.utils

from src.datasets.dataset_age import DatasetAge
from src.datasets.dataset_driving_dataset import DrivingDataset
from src.project_enums import EnumBackboneType, EnumSchedulerTypes, EnumSamplerTypes, EnumDatasets, EnumHeadTypes, EnumUncertaintyTypes
from src.uncertainty.bbb_utils import bbb_sample_elbo_with_mse, bbb_aleatoric_sample_elbo
from src.utils import utils
from src.datasets.datasetv2 import Drive360
from src.models.models import PilotNet, RegressionHead, BackboneHeadWrapper, RegressionHeadBBB, SimpleRegressionHead
from src.utils.checkpoints import load_head, load_backbone
from src.utils.common import FixedDistributedSamplerWrapper

log = logging.getLogger("create_model")


def get_data_loader(data_path, csv_name, transform, num_workers, batch_size, manual_seed, shuffle=True, flip=False,
                    dataset=EnumDatasets.drive360, enable_src_image_passthrough=False, sampler_type=EnumSamplerTypes.DistributedSampler):
    if dataset == EnumDatasets.drive360:
        dataset = Drive360
    elif dataset == EnumDatasets.driving_dataset:
        dataset = DrivingDataset
    elif dataset == EnumDatasets.age:
        dataset = DatasetAge
    else:
        raise Exception('invalid dataset')
    dataset = dataset(data_dir=data_path,
                      csv_name=csv_name,
                      transform=transform,
                      flip=flip,
                      enable_src_image_passthrough=enable_src_image_passthrough)

    # def seed_worker(worker_id):
    #     worker_seed = torch.initial_seed() % 2 ** 32
    #     np.random.seed(worker_seed)
    #     random.seed(worker_seed)
    #
    # g = torch.Generator()
    # g.manual_seed(manual_seed)

    if sampler_type == EnumSamplerTypes.DistributedSampler:
        sampler = torch.utils.data.DistributedSampler(dataset, shuffle=shuffle, seed=manual_seed)
    elif sampler_type == EnumSamplerTypes.UpscaleDistributedSampler:
        sampler = catalyst.data.BalanceClassSampler(labels=dataset.dataframe.bins_canSteering, mode="upsampling")
        sampler = catalyst.data.DistributedSamplerWrapper(sampler=sampler, shuffle=shuffle)  # TODO seed
    elif sampler_type == EnumSamplerTypes.DynamicDistributedSampler:
        # https://arxiv.org/abs/1901.06783
        log.warning("Using DynamicSampler results in a wrong calculation of total iterations for schedulers. They should still work but "
                    + "they will probably not reach the target value (calculated total_its should be higher than the actual iterations")
        log.warning("If continuing training from a checkpoint and using DynamicDistributedSampler might require setting"
                    + " start_epoch, but in theory set_epoch() should be enough")
        sampler = catalyst.data.DynamicBalanceClassSampler(labels=dataset.dataframe.bins_canSteering)
        sampler = FixedDistributedSamplerWrapper(sampler=sampler, shuffle=shuffle)  # TODO seed
    elif sampler_type == EnumSamplerTypes.SequentialSampler:
        sampler = SequentialSampler(dataset)
    else:
        raise NotImplementedError()

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             sampler=sampler,
                             pin_memory=True,
                             drop_last=True,
                             # worker_init_fn=seed_worker,
                             # generator=g
                             )
    log.debug(f"Data loaded: there are {len(dataset)} images.")
    return data_loader


def gen_backbone(config, backbone_type=EnumBackboneType.default, skip_reducing_layer=True):
    """

    Args:
        config:
        backbone_type:
        skip_reducing_layer: In some models (eg resnet) there is a layer greatly reducing the output dimensionality (eg avgpool).
            This switch sets them to nn.Identity()

    Returns:

    """
    # NOTE: Implementations of skip_reducing_layer have also tbd in dino_code for train_head(...)

    # we changed the name DeiT-S for ViT-S to avoid confusions
    config.arch = config.arch.replace("deit", "vit")
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if config.arch in vits.__dict__.keys():
        if backbone_type == EnumBackboneType.default or backbone_type == EnumBackboneType.student:
            backbone = vits.__dict__[config.arch](patch_size=config.patch_size,
                                                  drop_path_rate=config.drop_path_rate)
        else:
            backbone = vits.__dict__[config.arch](patch_size=config.patch_size)
        embed_dim = backbone.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif config.arch in torchvision_models.__dict__.keys():
        backbone = torchvision_models.__dict__[config.arch]()
        embed_dim = backbone.fc.weight.shape[1]
        if 'resnet' in config.arch and skip_reducing_layer:
            backbone.avgpool = nn.Identity()
            embed_dim = embed_dim * math.ceil(config.net_resolution[0] / 32) * math.ceil(config.net_resolution[1] / 32)
    elif config.arch == "nvidia":
        backbone = PilotNet()
        if backbone_type == EnumBackboneType.teacher or backbone_type == EnumBackboneType.student:
            raise Exception("Not working because output size is different for local and global crops for dino")
            embed_dim = 64 * 21 * 21  # 224x224
        else:
            embed_dim = 64 * math.ceil(config.net_resolution[0] / 11) * math.ceil(config.net_resolution[1] / 11)
    # if the network is a XCiT
    elif config.arch in torch.hub.list("facebookresearch/xcit:main"):
        if backbone_type == EnumBackboneType.default or backbone_type == EnumBackboneType.student:
            backbone = torch.hub.load('facebookresearch/xcit:main',
                                      config.arch,
                                      pretrained=False,
                                      drop_path_rate=config.drop_path_rate)
        else:
            backbone = torch.hub.load('facebookresearch/xcit:main',
                                      config.arch,
                                      pretrained=False)
        embed_dim = backbone.embed_dim

    else:
        raise Exception(f"Unknow architecture: {config.arch}")

    # Disable fc and head layers of models. Eg for resnet there is a fc layer specific for imagenet. Don't want these layers here
    backbone.fc, backbone.head = torch.nn.Identity(), torch.nn.Identity()

    return backbone, embed_dim


def gen_optimizer_and_schedulers(params_groups,
                                 data_loader,
                                 optimizer_algo,
                                 lr,
                                 min_lr,
                                 warmup_epochs,
                                 batch_size,
                                 lr_scheduler,
                                 wd_scheduler,
                                 weight_decay,
                                 weight_decay_end,
                                 epochs):
    if optimizer_algo == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif optimizer_algo == "adam":
        optimizer = torch.optim.Adam(params_groups)
    elif optimizer_algo == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif optimizer_algo == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches

    # ============ init schedulers ... ============
    # scaled_lr = lr * (batch_size * utils.get_world_size()) / 256.
    # scaled_min_lr = min_lr * (batch_size * utils.get_world_size()) / 256.
    # no scaling anymore
    scaled_lr = lr
    scaled_min_lr = min_lr
    if lr_scheduler == EnumSchedulerTypes.cosine:
        lr_schedule = utils.cosine_scheduler(
            scaled_lr,  # linear scaling rule
            scaled_min_lr,
            epochs,
            len(data_loader),
            warmup_epochs=warmup_epochs,
        )
    elif lr_scheduler == EnumSchedulerTypes.none:
        lr_schedule = utils.linear_scheduler(
            scaled_lr,
            epochs,
            len(data_loader),
            end_value=scaled_lr
        )
    elif lr_scheduler == EnumSchedulerTypes.linear:
        lr_schedule = utils.linear_scheduler(
            scaled_lr,
            epochs,
            len(data_loader),
            end_value=scaled_min_lr,
            warmup_epochs=warmup_epochs
        )
    elif lr_scheduler == EnumSchedulerTypes.exponential:
        lr_schedule = utils.exponential_scheduler(
            scaled_lr,
            epochs,
            len(data_loader),
            end_value=scaled_min_lr,
            warmup_epochs=warmup_epochs
        )

    if wd_scheduler == EnumSchedulerTypes.cosine:
        wd_schedule = utils.cosine_scheduler(
            weight_decay,
            weight_decay_end,
            epochs,
            len(data_loader),
        )
    elif wd_scheduler == EnumSchedulerTypes.none:
        wd_schedule = utils.linear_scheduler(
            weight_decay,
            epochs,
            len(data_loader),
            end_value=weight_decay
        )
    elif wd_scheduler == EnumSchedulerTypes.linear:
        wd_schedule = utils.linear_scheduler(
            weight_decay,
            epochs,
            len(data_loader),
            end_value=weight_decay_end
        )
    elif wd_scheduler == EnumSchedulerTypes.exponential:
        wd_schedule = utils.exponential_scheduler(
            weight_decay,
            epochs,
            len(data_loader),
            end_value=weight_decay_end
        )

    return optimizer, lr_schedule, wd_schedule


def create_and_load_model(config):
    """ load model (or just backbone) from checkpoints """
    # supervised
    backbone, embed_dim, backbone_info = create_and_load_backbone(config)
    head = create_head(embed_dim,
                       uncertainty_type=config.uncertainty_type,
                       head_arch=config.head_arch,
                       advanced_aleatoric=config.aleatoric_complex_head)

    load_head(config.pretrained_weights_head, config, head=head)
    # todo: check backbone_uuid

    if config.uncertainty_type == EnumUncertaintyTypes.bayes_by_backprop:
        _BackboneHeadWrapper = blitz.utils.variational_estimator(BackboneHeadWrapper)
        setattr(_BackboneHeadWrapper, "sample_elbo", bbb_sample_elbo_with_mse)
        model = _BackboneHeadWrapper(backbone=backbone, head=head, arch=config.arch)
    elif config.uncertainty_type == EnumUncertaintyTypes.aleatoric_bbb:
        _BackboneHeadWrapper = blitz.utils.variational_estimator(BackboneHeadWrapper)
        setattr(_BackboneHeadWrapper, "sample_elbo", bbb_aleatoric_sample_elbo)
        model = _BackboneHeadWrapper(backbone=backbone, head=head, arch=config.arch)
    else:
        model = BackboneHeadWrapper(backbone=backbone, head=head, arch=config.arch)
    return model, backbone_info['epoch']


def create_and_load_backbone(config):
    """ create backbone model and load weights from backbone or dino checkpoint """
    backbone, embed_dim = gen_backbone(config, backbone_type=EnumBackboneType.default, skip_reducing_layer=config.skip_reducing_layer)
    backbone_info = load_backbone(config.pretrained_weights_backbone, config, backbone, dino_key="teacher")
    return backbone, embed_dim, backbone_info


def create_head(embed_dim: int, head_arch: EnumHeadTypes, uncertainty_type=EnumUncertaintyTypes.none, advanced_aleatoric=False):
    if head_arch == EnumHeadTypes.RegressionHead:
        if uncertainty_type == EnumUncertaintyTypes.none or uncertainty_type == EnumUncertaintyTypes.mcbn:
            return RegressionHead(embed_dim, num_labels=1, advanced_aleatoric=advanced_aleatoric)
        elif uncertainty_type in [EnumUncertaintyTypes.aleatoric, EnumUncertaintyTypes.aleatoric_mcbn]:
            return RegressionHead(embed_dim, aleatoric=True, advanced_aleatoric=advanced_aleatoric)
        elif uncertainty_type in [EnumUncertaintyTypes.bayes_by_backprop, EnumUncertaintyTypes.aleatoric_bbb]:
            if utils.get_world_size() > 1:
                log.warning(
                    "Blitz implementation of bayes by backprop does not support multi gpu training (https://github.com/piEsposito/blitz-bayesian-deep-learning/issues/84)")
            return RegressionHeadBBB(embed_dim, num_labels=1, aleatoric=uncertainty_type == EnumUncertaintyTypes.aleatoric_bbb)
        else:
            raise NotImplementedError(f"creating RegressionHead for uncertainty type {uncertainty_type} is not implemented")
    elif head_arch == EnumHeadTypes.SimpleRegressionHead:
        if uncertainty_type == EnumUncertaintyTypes.none or uncertainty_type == EnumUncertaintyTypes.mcbn:
            return SimpleRegressionHead(embed_dim, num_labels=1)
        elif uncertainty_type in [EnumUncertaintyTypes.aleatoric, EnumUncertaintyTypes.aleatoric_mcbn]:
            return SimpleRegressionHead(embed_dim, aleatoric=True)
        elif uncertainty_type in [EnumUncertaintyTypes.bayes_by_backprop, EnumUncertaintyTypes.aleatoric_bbb]:
            raise NotImplementedError("bbb for SimpleRegressionHead not implemented")
        else:
            raise NotImplementedError(f"creating SimpleRegressionHead for uncertainty type {uncertainty_type} is not implemented")
    else:
        raise NotImplementedError(f"creating {head_arch} is not implemented")
