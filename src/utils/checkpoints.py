import json
import logging
import os
import shutil
import uuid

import torch
from torch import nn

from src.dino_code import DINOLoss
from src.utils.utils import is_main_process

log = logging.getLogger("checkpoints")


def save_dino(epoch, arch, optimizer, loss_func, student, teacher, config, metrics, persistent=False, is_best=False,
              allow_resume_training=False, checkpoint_uuid=None):
    """

    Args:
        epoch:
        arch:
        optimizer:
        loss_func:
        student:
        teacher:
        config:
        persistent:
        is_best:
        allow_resume_training:
        checkpoint_uuid: optionally pass uuid for this checkpoint here. This has to be random generated for every save. Useful if uuid is required before saving model, eg for training and saving a head

    Returns:

    """
    return _save_model(config, "dino", epoch, arch, optimizer, loss_func, metrics, student=student, teacher=teacher, persistent=persistent,
                       is_best=is_best, allow_resume_training=allow_resume_training, checkpoint_uuid=checkpoint_uuid)


def save_supervised(epoch, arch_head, optimizer, loss_func, head, config, metrics, persistent=False, is_best=False,
                    allow_resume_training=False, backbone=None, arch=None, uuid_backbone=None):
    """ backbone can be None if only head should be saved, uuid_backbone and backbone exclude each other, one has to None, the other a value """
    if uuid_backbone is None and backbone is None or uuid_backbone is not None and backbone is not None:
        raise AttributeError()
    return _save_model(config, "supervised", epoch, arch, optimizer, loss_func, metrics, backbone=backbone, head=head,
                       persistent=persistent, is_best=is_best, allow_resume_training=allow_resume_training, arch_head=arch_head,
                       uuid_backbone=uuid_backbone)


def _save_model(config, checkpoint_type, epoch, arch, optimizer, loss_func, metrics, arch_head=None, student=None, teacher=None,
                backbone=None, head=None,
                uuid_backbone=None, persistent=False, is_best=False, allow_resume_training=False, checkpoint_uuid=None):
    """

    Args:
        persistent: If False it will use a generic file name and override the previous save.
            Use True if this save should persist (and not be overwritten)

    Returns:

    """
    if not is_main_process():
        raise Exception("only call on master process")

    if checkpoint_uuid is None:
        checkpoint_uuid = str(uuid.uuid4())

    def get_dict(model):
        if model is not None:
            if isinstance(model, nn.parallel.DistributedDataParallel):
                return model.module.state_dict()
            return model.state_dict()
        return None

    checkpoint_data_lossfunc = None
    if allow_resume_training:
        if isinstance(loss_func, DINOLoss):
            checkpoint_data_lossfunc = loss_func.state_dict()
        else:
            checkpoint_data_lossfunc = loss_func

    if arch_head is None:
        head_type_as_str = None
    elif isinstance(arch_head, str):
        head_type_as_str = arch_head
    else:
        head_type_as_str = arch_head.value

    # TODO: patch_size, avgpool layers
    save_dict = {
        "version": 5,
        "type": checkpoint_type,
        "epoch": epoch,
        "uuid": checkpoint_uuid,
        "arch": arch,
        "arch_head": head_type_as_str,
        "optimizer": optimizer.state_dict() if allow_resume_training else None,
        "loss_func": checkpoint_data_lossfunc,
        "student": get_dict(student),
        "teacher": get_dict(teacher),
        "backbone": get_dict(backbone),
        "head": get_dict(head),
        "uuid_backbone": uuid_backbone,
        "config": json.dumps(vars(config)),
        "metrics": json.dumps(metrics.state_dict())
    }

    # remove "None" entries
    filtered = {k: v for k, v in save_dict.items() if v is not None}
    save_dict.clear()
    save_dict.update(filtered)

    if persistent:
        filename = f"checkpoint_{checkpoint_type}{'_for_' + uuid_backbone if uuid_backbone is not None else ''}_{arch}{('_' + arch_head) if arch_head is not None else ''}_ep{epoch:04}_{save_dict['uuid']}.pth"
    else:
        filename = f"checkpoint_{checkpoint_type}_{arch}{('_' + arch_head) if arch_head is not None else ''}.pth"

    target_full_name = os.path.join(config.output_dir, filename)
    torch.save(save_dict, target_full_name + '.tmp')
    shutil.move(target_full_name + '.tmp', target_full_name)  # just to be sure checkpoint is always valid

    # separate checkpoint for "best" model
    if is_best:
        filename = os.path.join(config.output_dir, f"checkpoint_best_{checkpoint_type}_{arch}.")
        torch.save(save_dict, filename + 'pth')
        with open(filename + "txt", "w") as f:
            json.dump({"epoch": epoch, **metrics.get_latest_values()}, f)

    return {"uuid": save_dict["uuid"]}


def load_dino(filepath, config, student, teacher, optimizer, loss_func):
    return _load_checkpoint(filepath, config, student=student, teacher=teacher, optimizer=optimizer, loss_func=loss_func,
                            checkpoint_type="dino")


def load_backbone(filepath, config, backbone, optimizer=None, loss_func=None, dino_key="teacher"):
    """

    Args:
        dino_key: if backbone is dino: use this key, valid: student, teacher

    Returns:

    """
    return _load_checkpoint(filepath, config, backbone=backbone, optimizer=optimizer, loss_func=loss_func, dino_key=dino_key,
                            checkpoint_type="backbone")


def load_head(filepath, config, head, optimizer=None, loss_func=None):
    return _load_checkpoint(filepath, config, head=head, optimizer=optimizer, loss_func=loss_func, checkpoint_type="head")


def _load_checkpoint(filepath, config, backbone=None, head=None, student=None, teacher=None, optimizer=None, loss_func=None,
                     checkpoint_type=None, dino_key=None):
    """

    Args:
        filepath:
        checkpoint_type: only required for deprecated storage format (version None); one of backbone, head, dino
        dino_key: if loading backbone from dino: choose which key to use, valid: student, teacher

    Returns:

    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"file {filepath} does not exist")

    try:
        state_dict = torch.load(filepath, map_location="cpu")
    except AttributeError as e:
        log.error(
            "Failed loading state_dict. Reason is probably that most old saves (version <= 1) were referencing some sourcecode (enums) which aren't existing anymore. Last working code version should be 72311d6793171052514f0e3b30ff8ea77abdf519. Loading on newer code is not possible.")
        raise e

    # load with version None (old)
    if 'version' not in state_dict:
        raise Exception(
            f"checkpoint '{filepath}' has old format (version: None). They are not supported anymore. Last commit with enough support for testing mode: 9460952ebd10cc839d3fb2dcc457df0bf1cdbec8")

    if state_dict['version'] not in [1, 2, 3, 4, 5]:
        raise Exception(f"checkpoint {filepath} has an invalid version: '{state_dict['version']}'")

    # load
    log.info(f"loading checkpoint {filepath} of version {state_dict['version']}")
    log.info(f"type: {state_dict['type']} epoch: {state_dict['epoch']} uuid: {state_dict['uuid']} arch: {state_dict['arch']}")

    def load_with_log_model(model, state_dict, title=''):
        # state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        try:
            model.load_state_dict(state_dict, strict=True)
            log.debug(f'{title} loaded successful')
        except:
            msg = model.load_state_dict(state_dict, strict=False)
            log.warning(f'loaded {title} with mismatching keys: {msg}')

    if state_dict['type'] == 'dino':
        assert config.arch == state_dict['arch']
        if backbone is not None:  # this is if training head based on a dino model
            log.info("loading dino as backbone")
            if state_dict["version"] <= 3:
                load_with_log_model(backbone, state_dict[dino_key], dino_key)
            else:  # might be the correct branch for older version too
                load_with_log_model(backbone,
                                    {k[9:]: state_dict[dino_key][k] for k in state_dict[dino_key].keys() if k.startswith("backbone")},
                                    'dino_key')

        else:  # this is if resuming dino training
            assert student is not None
            assert teacher is not None
            load_with_log_model(student, state_dict['student'], 'student')
            load_with_log_model(teacher, state_dict['teacher'], 'teacher')

    elif state_dict['type'] == 'head':
        assert head is not None
        if state_dict["version"] > 3:
            assert config.head_arch == state_dict['arch_head']
        load_with_log_model(head, state_dict['head'], 'head')
    elif state_dict['type'] == 'backbone':
        assert backbone is not None
        assert config.arch == state_dict['arch']
        load_with_log_model(backbone, state_dict['backbone'], 'backbone')
    elif state_dict['type'] == 'supervised':
        if head is not None:
            assert config.head_arch == state_dict['arch_head']
            load_with_log_model(head, state_dict['head'], 'head')
        else:
            assert backbone is not None
            assert config.arch == state_dict['arch']
            load_with_log_model(backbone, state_dict['backbone'], 'backbone')
    else:
        raise Exception(f"invalid type {state_dict[type]}")

    if optimizer is not None:
        if state_dict["version"] <= 4:
            optimizer.load_state_dict(state_dict['optimizer'].state_dict())
        else:
            optimizer.load_state_dict(state_dict['optimizer'])
        log.debug("loaded optimizer")
    if loss_func is not None:
        if state_dict["version"] <= 4:
            if isinstance(loss_func, DINOLoss):
                loss_func.load_state_dict(state_dict['loss_func'].state_dict())
            else:
                loss_func = state_dict['loss_func']
        else:
            if isinstance(loss_func, DINOLoss):
                loss_func.load_state_dict(state_dict['loss_func'])
            else:
                loss_func = state_dict['loss_func']
        log.debug("loaded loss func")

    filter_keys = ["student", "teacher", "head", "backbone", "optimizer", "loss_func"]
    return {
        **{k: state_dict[k] if k not in filter_keys else None for k in state_dict.keys()},
        "loaded_optimizer": optimizer is not None,
        "loaded_loss_func": loss_func is not None
    }
