import json
import logging
import math
import os
import uuid

import torch
import wandb
from torch import nn

from src.dino_code import DataAugmentationDINO, DINOHead, DINOLoss
from src.supervised_code import train_head_dino
from src.utils import utils
from src.utils.MetricCollector import MetricCollector, EnumMetricParams

from src.utils.common import dist_tqdm, dist_wandb_log
from src.utils.create_models import gen_backbone, get_data_loader, gen_optimizer_and_schedulers
from src.utils.checkpoints import load_dino, save_dino
from src.project_enums import EnumBackboneType
from src.utils.evaluation import metrics


def train_dino(config):
    log = logging.getLogger("dino")

    transform = DataAugmentationDINO(
        config.dino_global_crops_scale,
        config.dino_local_crops_scale,
        config.dino_local_crops_number,
    )
    data_loader = get_data_loader(data_path=config.dino_data_path,
                                  csv_name=config.dino_csv_name,
                                  batch_size=config.dino_batch_size_per_gpu,
                                  num_workers=config.num_workers,
                                  shuffle=True,
                                  flip=False,  # DataAugmentationDINO is already doing this
                                  transform=transform,
                                  dataset=config.dataset,
                                  manual_seed=config.seed,
                                  sampler_type=config.dino_sampler_type)

    ### generate models
    student, embed_dim = gen_backbone(config, backbone_type=EnumBackboneType.student, skip_reducing_layer=False)
    teacher, _ = gen_backbone(config, backbone_type=EnumBackboneType.teacher, skip_reducing_layer=False)
    log.debug("dataloader and models built")

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropWrapper(student, DINOHead(
        embed_dim,
        config.dino_out_dim,
        use_bn=config.dino_use_bn_in_head,
        norm_last_layer=config.dino_norm_last_layer,
    ))
    teacher = utils.MultiCropWrapper(
        teacher,
        DINOHead(embed_dim, config.dino_out_dim, config.dino_use_bn_in_head),
    )
    student, teacher = student.cuda(), teacher.cuda()
    log.debug("models now have MultiCropWrapper and are on gpu")

    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)
        log.debug("SyncBatchNorm.convert_sync_batchnorm ran")

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[config.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    log.debug("added SyncBatchNorm")
    student = nn.parallel.DistributedDataParallel(student, device_ids=[config.gpu])
    log.debug("added DDP")
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    log.debug("initialized teacher weights")
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    log.debug("grad calculation for teacher disabled")
    log.info(f"Student and Teacher are built: they are both {config.arch} network.")

    # ============ preparing loss ... ============
    dino_loss = DINOLoss(
        config.dino_out_dim,
        config.dino_local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        config.dino_warmup_teacher_temp,
        config.dino_teacher_temp,
        config.dino_warmup_teacher_temp_epochs,
        config.dino_epochs,
    ).cuda()

    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(config.dino_momentum_teacher, 1,
                                               config.dino_epochs, len(data_loader))

    params_groups = utils.get_params_groups(student)
    optimizer, lr_schedule, wd_schedule = gen_optimizer_and_schedulers(params_groups=params_groups,
                                                                       data_loader=data_loader,
                                                                       optimizer_algo=config.dino_optimizer,
                                                                       lr=config.dino_lr,
                                                                       min_lr=config.dino_min_lr,
                                                                       warmup_epochs=config.dino_warmup_epochs,
                                                                       batch_size=config.dino_batch_size_per_gpu,
                                                                       lr_scheduler=config.dino_lr_scheduler,
                                                                       wd_scheduler=config.dino_wd_scheduler,
                                                                       weight_decay=config.dino_weight_decay,
                                                                       weight_decay_end=config.dino_weight_decay_end,
                                                                       epochs=config.dino_epochs)

    # TODO: aleatoric uct stuff (dont forget wandb_log)
    metric_collector = MetricCollector([
        {'name': 'validation_mse',
         EnumMetricParams.use_for_plateau_detection: True,
         EnumMetricParams.patience: math.ceil(config.dino_early_stopping / config.dino_test_freq),
         EnumMetricParams.use_for_is_best: True},
        {'name': 'validation_mse_equally_weighted',
         EnumMetricParams.use_for_plateau_detection: True,
         EnumMetricParams.patience: math.ceil(config.dino_early_stopping / config.dino_test_freq)},
        {'name': 'validation_mse_extreme_values'},
        {'name': 'validation_mae'},
        {'name': 'validation_rmse'},
        {'name': 'train_dino_loss'},
        {'name': 'train_head_loss'},
        {'name': 'train_head_mse'},
    ])

    # ============ optionally resume training ... ============
    start_epoch = 0
    checkpoint_name = f"checkpoint_dino_{config.arch}.pth"
    if config.resume_training:
        checkpoint_dict = load_dino(os.path.join(config.output_dir, checkpoint_name), config, student.module, teacher.module,
                                    optimizer, dino_loss)

        if not checkpoint_dict["loaded_optimizer"] or not checkpoint_dict["loaded_loss_func"]:
            raise Exception("checkpoint does not support resume training")

        if checkpoint_dict["version"] == 3 and "metrics" not in checkpoint_dict:
            log.error("Checkpoint is too old and does not contain MetricCollector state dict. Using a new one. Might not work as expected")
        else:
            metric_collector.load_state_dict(json.loads(checkpoint_dict["metrics"]))
        start_epoch = checkpoint_dict["epoch"]
        log.info(f"resumed from checkpoint: {checkpoint_name}")

    # TODO: watch model, should work but untested
    if utils.is_main_process():
        wandb.watch(teacher, dino_loss, log="all", log_freq=2 * len(data_loader))

    log.debug("Starting DINO training !")
    for epoch in range(start_epoch, config.dino_epochs):
        epoch_plus_one = epoch + 1
        data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of DINO ... ============
        loss = train_one_epoch(student, teacher, teacher_without_ddp, dino_loss,
                               data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
                               epoch, config)

        # train head and use it to eval the backbone
        model_uuid = str(uuid.uuid4()) if utils.is_main_process() else "not_main_process_should_not_save_model"
        is_best = False
        is_plateau = False
        save_checkpoints_this_epoch = config.dino_saveckp_freq and epoch_plus_one % config.dino_saveckp_freq == 0
        if epoch == config.dino_epochs - 1 or config.dino_test_freq != 0 and epoch_plus_one % config.dino_test_freq == 0:
            validation_summary, other_log_data = train_head_dino(config,
                                                                 backbone=teacher.module.backbone,
                                                                 embed_dim=embed_dim,
                                                                 cur_epoch=epoch,
                                                                 backbone_uuid=model_uuid,
                                                                 silent_mode=True,
                                                                 # saving makes only sense if the backbone is also saved
                                                                 # disable_save=not save_checkpoints_this_epoch
                                                                 # nvm always saving head because it might be this is the best
                                                                 # dino_checkpoint, but I don't know that here
                                                                 disable_save=False)

            is_best, is_plateau = metric_collector.step({
                'validation_mse': validation_summary['validation_mse'][EnumMetricParams.best],
                'validation_mse_equally_weighted': validation_summary['validation_mse_equally_weighted']['best'],
                'validation_mse_extreme_values': validation_summary['validation_mse_extreme_values'][EnumMetricParams.best],
                'validation_mae': validation_summary['validation_mae'][EnumMetricParams.best],
                'validation_rmse': validation_summary['validation_rmse'][EnumMetricParams.best],
                'train_dino_loss': loss,
                'train_head_loss': validation_summary['train_loss'][EnumMetricParams.best],
                'train_head_mse': validation_summary['train_mse'][EnumMetricParams.best]
            })

            dist_wandb_log({
                'epoch': epoch_plus_one,
                'validation_mse': metric_collector.get_last_value('validation_mse'),
                'validation_best_mse': metric_collector['validation_mse'][EnumMetricParams.best],
                'validation_mse_extreme_values': metric_collector.get_last_value('validation_mse_extreme_values'),
                'validation_mse_equally_weighted_bins': metric_collector.get_last_value('validation_mse_equally_weighted'),
                'validation_best_mse_equally_weighted_bins': metric_collector['validation_mse_equally_weighted'][EnumMetricParams.best],
                'validation_loss_mae': metric_collector.get_last_value('validation_mae'),
                'validation_rmse': metric_collector.get_last_value('validation_rmse'),
                'relative_mse_overfitting': metrics.relative_overfitting(metric_collector.get_last_value('validation_mse'),
                                                                         metric_collector.get_last_value('train_head_mse')),

                'train_head_loss': metric_collector.get_last_value('train_head_loss'),
                'train_head_mse': metric_collector.get_last_value('train_head_mse'),

                **other_log_data
            })

        if utils.is_main_process():
            # save checkpoint
            save_dino(epoch=epoch_plus_one,
                      arch=config.arch,
                      optimizer=optimizer,
                      loss_func=dino_loss,
                      student=student,
                      teacher=teacher,
                      config=config,
                      allow_resume_training=True,
                      checkpoint_uuid=model_uuid,
                      is_best=is_best,
                      metrics=metric_collector)
            # periodically keeping a checkpoint
            if save_checkpoints_this_epoch:
                save_dino(epoch=epoch_plus_one,
                          arch=config.arch,
                          optimizer=optimizer,
                          loss_func=dino_loss,
                          student=student,
                          teacher=teacher,
                          config=config,
                          persistent=True,
                          allow_resume_training=False,
                          checkpoint_uuid=model_uuid,  # passing it to both saves is ok here because they are the exact same model
                          metrics=metric_collector)
        if is_plateau:
            log.info("plateau detection triggered: stopping training")
            break


def train_one_epoch(student,
                    teacher,
                    teacher_without_ddp,
                    dino_loss,
                    data_loader,
                    optimizer,
                    lr_schedule,
                    wd_schedule,
                    momentum_schedule,
                    epoch,
                    config):
    total_loss = 0
    pbar = dist_tqdm(data_loader, desc=f"dino [{epoch + 1}/{config.dino_epochs}]")
    for i, (images, _, _) in enumerate(pbar):
        it = len(data_loader) * epoch + i  # global training iteration

        for param_index, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if param_index == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images['cameraFront']]
        # teacher and student forward passes + compute dino loss
        teacher_output = teacher(images[:2])  # only the 2 global views pass through the teacher
        student_output = student(images)
        loss = dino_loss(student_output, teacher_output, epoch)

        if math.isinf(loss.item()):
            raise Exception("Loss is {}, stopping training".format(loss.item()))

        # student update
        optimizer.zero_grad()
        loss.backward()
        param_norms = None
        if config.dino_clip_grad:
            param_norms = utils.clip_gradients(student, config.dino_clip_grad)
        utils.cancel_gradients_last_layer(epoch, student, config.dino_freeze_last_layer)
        optimizer.step()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()

        # throttle wandb logging because the log file exceeds the file size limit of 10.4MB
        # and reducing the logging interval of these elements is not problematic
        # esp lr and wd aren't that relevant to log frequently
        if i == 0:
            dist_wandb_log({'dino_loss': loss.item(),
                            'dino_lr': optimizer.param_groups[0]["lr"],
                            'dino_weight_decay': optimizer.param_groups[0]["weight_decay"]
                            })
        elif i % 25 == 0:
            dist_wandb_log({'dino_loss': loss.item()})

        total_loss += loss.item()
        if utils.is_main_process():
            pbar.set_postfix({'avg_loss': "{:.4f}".format(total_loss / (i + 1)), 'lr': lr_schedule[it]})
    # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    return total_loss / len(data_loader)
