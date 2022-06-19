import copy
import logging
import math

import wandb
from blitz.utils.minibatch_weighting import minibatch_weight

from torch import nn
from torchvision.transforms import transforms
import blitz.utils

from src.custom_exceptions import ModelNoProgressException
from src.uncertainty.bbb_utils import bbb_aleatoric_sample_elbo, bbb_sample_elbo_with_mse
from src.utils.MetricCollector import MetricCollector, EnumMetricParams
from src.utils.common import dist_tqdm, dist_wandb_log
from src.utils.create_models import gen_backbone, EnumBackboneType, get_data_loader, gen_optimizer_and_schedulers, create_head
from src.utils.checkpoints import save_supervised
from src.project_enums import EnumUncertaintyTypes, EnumMode, EnumBBBCostFunction
from src.utils import utils

from src.models.models import BackboneHeadWrapper
from src.utils.evaluation.helpers import generate_evaluation_metrics, generate_evaluation_plots
from src.utils.evaluation import metrics
from src.test_code import test

log = logging.getLogger("supervised")


def train_supervised(config):
    backbone, embed_dim = gen_backbone(config, backbone_type=EnumBackboneType.default, skip_reducing_layer=config.skip_reducing_layer)
    backbone.train()
    _supervised_training(config,
                         only_head=False,
                         backbone=backbone,
                         embed_dim=embed_dim,
                         wandb_watch_model=True)


def train_head_dino(config, backbone, cur_epoch, embed_dim, backbone_uuid, silent_mode=True, disable_save=False):
    # create a copy just to be sure... and the backbone might get modified for head training, eg backbone.avgpool = nn.Identity()
    backbone = copy.deepcopy(backbone)

    if 'resnet' in config.arch and config.skip_reducing_layer:
        log.debug("arch is resnet and skip_reducing_layer is true -> changing head embed_dim")
        backbone.avgpool = nn.Identity()
        embed_dim = embed_dim * math.ceil(config.net_resolution[0] / 32) * math.ceil(config.net_resolution[1] / 32)

    return train_head(config, backbone, cur_epoch, embed_dim, backbone_uuid, silent_mode, disable_save)


def train_head(config, backbone, cur_epoch, embed_dim, backbone_uuid, silent_mode=True, disable_save=False, wandb_watch_model=False):
    backbone.eval()
    backbone.requires_grad = False
    for param in backbone.parameters():
        param.requires_grad = False

    summary = _supervised_training(config,
                                   only_head=True,
                                   silent_mode=silent_mode,
                                   backbone=backbone,
                                   embed_dim=embed_dim,
                                   out_name_prefix=f"backbone_ep{str(cur_epoch).zfill(3)}",
                                   backbone_uuid=backbone_uuid,
                                   disable_save=disable_save,
                                   wandb_watch_model=wandb_watch_model)
    backbone.requires_grad = True
    backbone.train()
    return summary


def _supervised_training(config,
                         backbone,
                         embed_dim,
                         batch_size_per_gpu=None,
                         only_head=True,
                         silent_mode=False,
                         out_name_prefix='',
                         backbone_uuid=None,
                         disable_save=False,
                         wandb_watch_model=False):
    """

    Args:
        config:
        backbone:
        embed_dim:
        batch_size_per_gpu:
        only_head:
        silent_mode: disable wandb log, save and log less stuff
        out_name_prefix:
        epochs:
        backbone_uuid:

    Returns:

    """
    csv_name = config.csv_name
    epochs = config.epochs
    if batch_size_per_gpu is None: batch_size_per_gpu = config.batch_size_per_gpu

    head = create_head(embed_dim,
                       uncertainty_type=config.uncertainty_type,
                       head_arch=config.head_arch,
                       advanced_aleatoric=config.aleatoric_complex_head)
    head.train()

    if config.uncertainty_type == EnumUncertaintyTypes.bayes_by_backprop:
        _BackboneHeadWrapper = blitz.utils.variational_estimator(BackboneHeadWrapper)
        setattr(_BackboneHeadWrapper, "sample_elbo", bbb_sample_elbo_with_mse)
        model = _BackboneHeadWrapper(backbone=backbone, head=head, arch=config.arch)
        model = model.cuda()
        # syncbatchnorms not required, bbb does only support single gpu anyways
    elif config.uncertainty_type == EnumUncertaintyTypes.aleatoric_bbb:
        log.warning("aleatoric bbb should work, but i never got it working. Some references in sourcecode below this line: ")
        # aleatoric uncertainty & combining with epistemic: What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?
        # sample implementation of aleatoric & epistemic with variational inference (comparable to bbb):
        # https://github.com/marcobellagente93/Bayesian_Regression/blob/main/Bayesian_regression_notebooks/noisy_sin_example.ipynb
        # best run was with loss quite stable a bit above 1 (trained ~120 epochs): aleatoric_bbb_most_stable_run (might be with different loss func)
        _BackboneHeadWrapper = blitz.utils.variational_estimator(BackboneHeadWrapper)
        setattr(_BackboneHeadWrapper, "sample_elbo", bbb_aleatoric_sample_elbo)
        model = _BackboneHeadWrapper(backbone=backbone, head=head, arch=config.arch)
        model = model.cuda()
        # syncbatchnorms not required, bbb does only support single gpu anyways
    else:
        model = BackboneHeadWrapper(backbone=backbone, head=head, arch=config.arch)
        # synchronize batch norms (if any)
        if utils.has_batchnorms(model):
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.cuda()
        model = nn.parallel.DistributedDataParallel(model, device_ids=[config.gpu])

    supervised_transform = transforms.Compose([
        transforms.Resize((config.net_resolution[1], config.net_resolution[0])) if config.RandomResizedCrop_scale_min == 1
        else transforms.RandomResizedCrop(size=(config.net_resolution[1], config.net_resolution[0]),
                                          scale=(config.RandomResizedCrop_scale_min, config.RandomResizedCrop_scale_max)),
        transforms.RandomApply([transforms.ColorJitter(brightness=config.ColorJitter_brightness,
                                                       contrast=config.ColorJitter_contrast,
                                                       saturation=config.ColorJitter_saturation,
                                                       hue=config.ColorJitter_hue)],
                               p=config.ColorJitter_probability),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(config.GaussianBlur_kernel_size, config.GaussianBlur_kernel_size),
                                                        sigma=(config.GaussianBlur_sigma_min, config.GaussianBlur_sigma_max))],
                               p=config.GaussianBlur_probability),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    supervised_loader = get_data_loader(data_path=config.data_path,
                                        csv_name=csv_name,
                                        batch_size=batch_size_per_gpu,
                                        num_workers=config.num_workers,
                                        shuffle=True,
                                        transform=supervised_transform,
                                        flip=True,
                                        dataset=config.dataset,
                                        manual_seed=config.seed,
                                        sampler_type=config.sampler_type)
    params_groups = utils.get_params_groups(model)
    optimizer, lr_schedule, wd_schedule = gen_optimizer_and_schedulers(params_groups=params_groups,
                                                                       data_loader=supervised_loader,
                                                                       optimizer_algo=config.optimizer,
                                                                       lr=config.lr,
                                                                       min_lr=config.min_lr,
                                                                       warmup_epochs=config.warmup_epochs,
                                                                       batch_size=batch_size_per_gpu,
                                                                       lr_scheduler=config.lr_scheduler,
                                                                       wd_scheduler=config.wd_scheduler,
                                                                       weight_decay=config.weight_decay,
                                                                       weight_decay_end=config.weight_decay_end,
                                                                       epochs=epochs
                                                                       )

    # ============ preparing loss ... ============
    if config.uncertainty_type in [EnumUncertaintyTypes.aleatoric, EnumUncertaintyTypes.aleatoric_bbb]:
        loss_func = nn.GaussianNLLLoss(reduction='mean').cuda()
    else:
        loss_func = nn.MSELoss().cuda()

    # loading checkpoints could be implemented here
    # I stopped saving the information required to continue training to save storage space
    # Training the supervised part is relatively fast, so in case I would like to continue training I'll just start a new one

    # wandb model watcher. Provides some fancy model stats in webinterface
    if wandb_watch_model and utils.is_main_process():
        wandb.watch(model, loss_func, log="all", log_freq=len(supervised_loader))

    start_epoch = 0
    metric_collector_config = [
        {'name': 'validation_mse',
         EnumMetricParams.use_for_plateau_detection: True,
         EnumMetricParams.use_for_is_best: True,
         EnumMetricParams.patience: config.plateau_detection_epochs},
        {'name': 'validation_mse_equally_weighted', EnumMetricParams.use_for_plateau_detection: True},
        {'name': 'validation_mse_extreme_values'},
        {'name': 'validation_mae'},
        {'name': 'validation_rmse'},
        {'name': 'train_loss'},
        {'name': 'train_mse'},
    ]
    if config.uncertainty_type != EnumUncertaintyTypes.none:
        metric_collector_config += [
            {'name': 'miscal_area'},
            {'name': 'nll'},
            {'name': 'crps'},
            {'name': 'sharpness'},
        ]
    metric_collector = MetricCollector(metric_collector_config)

    initial_dataloader_length = len(supervised_loader)  # for dynamicsampler
    other_log_data = {}  # eg images/plots
    for epoch in range(start_epoch, epochs):
        supervised_loader.sampler.set_epoch(epoch)

        train_stats = train_one_epoch_supervised(model,
                                                 loss_func,
                                                 supervised_loader,
                                                 optimizer,
                                                 epoch,
                                                 epochs,
                                                 only_head,
                                                 config,
                                                 lr_schedule,
                                                 wd_schedule,
                                                 silent_mode=silent_mode,
                                                 initial_dataloader_length=initial_dataloader_length)

        # test, evaluate test results and log
        if not silent_mode or epoch == (epochs - 1):
            base_filename = f"ep{str(epoch + 1).zfill(3)}_supervised_train{('_for_' + out_name_prefix + '_') if len(out_name_prefix) > 0 else ''}"
            # run on all gpus for early stopping
            # can probably be easily parallelized with synced predictions with
            # https://github.com/pytorch/vision/blob/afda28accbc79035384952c0359f0e4de8454cb3/references/detection/utils.py#L70
            # but potential savings are only around 10s per execution
            if config.csv_name_test2 is not None:
                # perform test run with 2nd csv file. Only predictions csv will be saved, nothing else will happen
                validation_loss, _ = test(model,
                                          config,
                                          config.csv_name_test2,
                                          save_predictions=True,
                                          save_predictions_filename_prefix=f"2nd_{base_filename}_")
                log.debug(f"2nd csv test loss: {validation_loss}")

            validation_loss, predictions = test(
                model, config, config.csv_name_test, save_predictions=True,
                save_predictions_filename_prefix=f"{base_filename}_")

            _metrics = generate_evaluation_metrics(predictions, None)
            log_dict = {
                'validation_mse': validation_loss,
                'validation_mse_equally_weighted': _metrics['validation_mse_equally_weighted'],
                'validation_mse_extreme_values': _metrics['validation_mse_extreme_values'],
                'validation_mae': _metrics['validation_mae'],
                'validation_rmse': _metrics['validation_rmse'],
                'train_loss': train_stats['loss'],
                'train_mse': train_stats['mse']
            }
            if 'uncertainty' in _metrics:
                log_dict = {**log_dict, **_metrics['uncertainty']}
            is_best, is_plateau = metric_collector.step(log_dict)

            if utils.is_main_process():
                # generate plots, save on disk and prepare for wand.log
                # do this here because wandb.log block is not executed if silent_mode is True but epoch == (epochs - 1)
                # but plots are still required
                other_log_data['images'] = {k: wandb.Image(v)
                                            for k, v
                                            in generate_evaluation_plots(config, predictions, 'save', out_name_prefix, epoch).items()}

                # save models
                if not disable_save:
                    save_supervised(epoch=epoch + 1,
                                    arch=config.arch,
                                    arch_head=config.head_arch,
                                    optimizer=optimizer,
                                    loss_func=loss_func,
                                    head=head,
                                    backbone=backbone if not only_head else None,
                                    uuid_backbone=backbone_uuid if only_head else None,
                                    config=config,
                                    persistent=epoch == (epochs - 1),
                                    is_best=is_best and not silent_mode,
                                    # don't need that separate checkpoint in case of silent_mode because test will only run once
                                    allow_resume_training=False,
                                    # normally not required for supervised because training is fast, but would require much more storage
                                    metrics=metric_collector)
                if config.saveckp_freq and (epoch + 1) % config.saveckp_freq == 0:
                    save_supervised(epoch=epoch + 1,
                                    arch=config.arch,
                                    arch_head=config.head_arch,
                                    optimizer=optimizer,
                                    loss_func=loss_func,
                                    head=head,
                                    backbone=backbone if not only_head else None,
                                    uuid_backbone=backbone_uuid if only_head else None,
                                    config=config,
                                    persistent=True,
                                    allow_resume_training=False,
                                    metrics=metric_collector)

            if not silent_mode:  # should _not_ run if silent_mode is false but epoch == (epochs - 1)
                if utils.is_main_process():
                    # log summary to logfile
                    log.debug(f"supervised epoch {epoch} finished: {metric_collector.get_latest_values()}")
                    # log to wandb + plots to wandb
                    wandb_log_dict = {
                        'epoch': epoch + 1,
                        'validation_mse': metric_collector.get_last_value('validation_mse'),
                        'validation_best_mse': metric_collector['validation_mse'][EnumMetricParams.best],
                        'validation_mse_extreme_values': metric_collector.get_last_value('validation_mse_extreme_values'),
                        'validation_mse_equally_weighted_bins': metric_collector.get_last_value('validation_mse_equally_weighted'),
                        'validation_best_mse_equally_weighted_bins': metric_collector['validation_mse_equally_weighted'][
                            EnumMetricParams.best],
                        'validation_loss_mae': metric_collector.get_last_value('validation_mae'),
                        'validation_rmse': metric_collector.get_last_value('validation_rmse'),
                        'relative_mse_overfitting': metrics.relative_overfitting(metric_collector.get_last_value('validation_mse'),
                                                                                 metric_collector.get_last_value('train_mse')),
                        'train_mse': metric_collector.get_last_value('train_mse'),

                        **other_log_data['images']
                    }
                    if 'uncertainty' in _metrics:
                        wandb_log_dict = {**wandb_log_dict, **{
                            'miscal_area': metric_collector.get_last_value('miscal_area'),
                            'nll': metric_collector.get_last_value('nll'),
                            'crps': metric_collector.get_last_value('crps'),
                            'sharpness': metric_collector.get_last_value('sharpness'),
                        }}
                    dist_wandb_log(wandb_log_dict)

                # early stop
                if is_plateau:
                    log.info("plateau detection triggered: stopping training")
                    break

        # if repeat_on_fail is on check abort conditions
        if config.repeat_on_fail is not None:
            # eleatoric head; values for dataset 4_5k supervised
            if config.mode == EnumMode.head and config.uncertainty_type == EnumUncertaintyTypes.aleatoric:
                if epoch == 0 and train_stats['mse'] > 1.0:
                    raise ModelNoProgressException("train mse after first epoch is larger than 1.0")
                if epoch == 3 and metric_collector['validation_mse'][EnumMetricParams.best] > 0.4:
                    raise ModelNoProgressException("validation mse after 4th epoch is larger than 0.4")
                if epoch == 9 and metric_collector['validation_mse'][EnumMetricParams.best] > 0.12:
                    raise ModelNoProgressException("validation mse after 10th epoch is larger than 0.12")

    return metric_collector, other_log_data


def train_one_epoch_supervised(model, loss_func, data_loader, optimizer, epoch, epochs, only_head, config, lr_schedule, wd_schedule,
                               silent_mode, initial_dataloader_length=None):
    total_loss = 0
    total_mse_loss = 0

    mse_loss_func = None
    if not isinstance(loss_func, nn.MSELoss) or config.uncertainty_type == EnumUncertaintyTypes.bayes_by_backprop:
        mse_loss_func = nn.MSELoss()

    pbar = dist_tqdm(data_loader, desc=f"train [{epoch + 1}/{epochs}]")
    for i, batch in enumerate(pbar):
        if initial_dataloader_length is not None:
            it = len(data_loader) * epoch + i  # global training iteration
        else:
            it = initial_dataloader_length * epoch + i
        inp, target, _ = batch

        for param_index, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if param_index == 0:  # only the first group is regularized, see :func:`~dino.utils.get_params_groups`
                param_group["weight_decay"] = wd_schedule[it]

        # move to gpu
        inp = inp['cameraFront'].cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        model.zero_grad()
        if config.uncertainty_type == EnumUncertaintyTypes.none:
            output = model(inp)
            loss = loss_func(output, target)
        elif config.uncertainty_type == EnumUncertaintyTypes.aleatoric:
            output = model(inp)
            output_mu = output[:, 0].view(-1, 1)
            output_std = output[:, 1].view(-1, 1)
            loss = loss_func(output_mu, target, output_std ** 2)
            loss_mse = mse_loss_func(output_mu, target)
        elif config.uncertainty_type in [EnumUncertaintyTypes.bayes_by_backprop, EnumUncertaintyTypes.aleatoric_bbb]:
            if config.bbb_cost_function == EnumBBBCostFunction.samples_per_epoch:
                # 1/batch (epoch) size is suggested in github issue at: https://github.com/piEsposito/blitz-bayesian-deep-learning/issues/79
                # (---likely based on https://papers.nips.cc/paper/2015/file/bc7316929fe1545bf0b98d114ee3ecb8-Paper.pdf--- nope this is says 1/minibatch size)
                cost_func = 1. / (inp.shape[0] * len(data_loader))
            elif config.bbb_cost_function == EnumBBBCostFunction.graves:
                # original BBB authors suggests
                # - 1/minibatch count (" Graves (2011) proposes minimising the mini-batch cost for minibatch..."
                # https://arxiv.org/pdf/1505.05424.pdf
                # based on https://papers.nips.cc/paper/2011/file/7eb3c8be3d411e8ebfab08eba5f49632-Paper.pdf)
                cost_func = 1. / len(data_loader)
            elif config.bbb_cost_function == EnumBBBCostFunction.paper_proposal:
                # - (2^(M-i))/(2^M-1) as proposed in BBB paper
                cost_func = minibatch_weight(batch_idx=i, num_batches=len(data_loader))
            elif config.bbb_cost_function == EnumBBBCostFunction.paper_proposal_all_epochs:
                # because this approach results in loss peak (like collapsing) at each new epoch the following variation calculates it over
                # all epochs, instead of only the current epoch, like described in the paper. This approach showed way too high uct preds
                # epoch starts at zero, epochs at 1
                cost_func = minibatch_weight(batch_idx=epoch * len(data_loader) + i, num_batches=epochs * len(data_loader))
            else:
                raise NotImplementedError(config.bbb_cost_function)

            loss, loss_mse = model.sample_elbo(
                inputs=inp,
                labels=target,
                criterion=loss_func,
                criterion_mse=mse_loss_func,
                sample_nbr=3,  # "quality" of std, 5 might be a good spot between loss, uct quality and performance
                complexity_cost_weight=cost_func
            )
        else:
            raise NotImplementedError()

        if not math.isfinite(loss.item()):
            raise Exception("Loss is {}, stopping training".format(loss.item()))

        loss.backward()
        if config.clip_grad:
            # print('using gradient clipping, might not work correctly')
            utils.clip_gradients(model, config.clip_grad)
        optimizer.step()

        total_loss += loss.item()
        if mse_loss_func:
            # noinspection PyUnboundLocalVariable
            total_mse_loss += loss_mse.item()
        else:
            total_mse_loss = total_loss

        if not silent_mode:
            dist_wandb_log({'lr': optimizer.param_groups[0]["lr"], 'train_loss': loss})
        if utils.is_main_process():
            pbar.set_postfix({'avg_loss': "{:.4f}".format(total_loss / (i + 1)), 'lr': lr_schedule[it]})

    return {"loss": total_loss / len(data_loader), "mse": total_mse_loss / len(data_loader)}
