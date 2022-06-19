import argparse
import datetime
import logging
import math
import os
import random
import shutil
import string
import subprocess
from enum import Enum
from pathlib import Path
from typing import Iterator

import catalyst.data

import wandb
from tqdm import tqdm

from src.utils.utils import is_main_process
from catalyst.data.dataset import DatasetFromSampler


log = logging.getLogger("common")


def dist_print(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)


def dist_wandb_log(*args, **kwargs):
    if is_main_process():
        wandb.log(*args, **kwargs)


# https://stackoverflow.com/a/60750535
class EnumAction(argparse.Action):
    """
    Argparse action for handling Enums
    """

    def __init__(self, **kwargs):
        # Pop off the type value
        enum_type = kwargs.pop("type", None)

        # Ensure an Enum subclass is provided
        if enum_type is None:
            raise ValueError("type must be assigned an Enum when using EnumAction")
        if not issubclass(enum_type, Enum):
            raise TypeError("type must be an Enum when using EnumAction")

        # Generate choices from the Enum
        kwargs.setdefault("choices", tuple(e.value for e in enum_type))

        super(EnumAction, self).__init__(**kwargs)

        self._enum = enum_type

    def __call__(self, parser, namespace, values, option_string=None):
        # Convert value back into an Enum
        value = self._enum(values)
        setattr(namespace, self.dest, value)


class CustomFormatter(logging.Formatter):
    grey = "\x1b[37;22m"
    white = "\x1b[97;22m"
    yellow = "\x1b[33;22m"
    red = "\x1b[31;22m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    out_format = "{asctime} -{name:^12s}- {levelname:8s} | {message} ({filename}:{lineno})"

    FORMATS = {
        logging.DEBUG: grey + out_format + reset,
        logging.INFO: white + out_format + reset,
        logging.WARNING: yellow + out_format + reset,
        logging.ERROR: red + out_format + reset,
        logging.CRITICAL: bold_red + out_format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, style='{')
        return formatter.format(record)


def init_logging(target_dir=None, console_log_level=logging.INFO, disabled=False, rank=0):
    """ initialize logger """
    if disabled:
        logging.disable()
    else:
        handlers = []

        console_logging_handler = logging.StreamHandler()
        console_logging_handler.setFormatter(CustomFormatter())
        console_logging_handler.setLevel(console_log_level)
        if rank == 0:
            handlers.append(console_logging_handler)

        if target_dir is not None:
            log_file_name = f'log{("_rank_" + str(rank)) if rank != 0 else ""}.txt'
            file_logging_handler = logging.FileHandler(os.path.join(target_dir, log_file_name))
            file_logging_handler.setFormatter(logging.Formatter(CustomFormatter.out_format, style="{"))
            handlers.append(file_logging_handler)

        logging.basicConfig(handlers=handlers,
                            level=logging.DEBUG)


def get_project_root() -> Path:
    filepath = Path(__file__).absolute()
    for i in range(0, 50):
        if Path(os.path.join(filepath.parents[i], '.gitignore')).exists():
            return filepath.parents[i]
    log.warning('failed to find project root, .gitignore has to exist in the same or a parent folder')


def backup_project(target_dir):
    """ backup the project to the target_dir including uncommitted changes """
    project_root = get_project_root()
    out_filename = os.path.join(target_dir, 'project_code')

    if Path(out_filename + ".tgz").exists() or Path(out_filename + ".zip").exists():
        raise FileExistsError(f"{out_filename} already exists")

    # try backing up with the help of git -> respecting .gitignore
    if not subprocess.run(f"git --version && cd {project_root} && git ls-files | tar Tczf - {out_filename}.tgz",
                          shell=True,
                          stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode == 0:
        # if backing up with git did't work backup the whole directory with python
        shutil.make_archive(out_filename, 'zip', project_root)


def dist_tqdm(obj, *args, **kwargs):
    if is_main_process():
        return tqdm(obj, ascii=True, mininterval=15, *args, **kwargs)
    else:
        return obj





class FixedDistributedSamplerWrapper(catalyst.data.DistributedSamplerWrapper):
    """ original catalyst.data.DistributedSamplerWrapper is incompatible with catalyst.data.DynamicBalanceClassSampler because of it's
    decreasing sample count. It crashed as soon as there were less than 50% of original sample count left.
    This reruns the code to update self.total_size of DistributedSampler on every __iter__() call, instead of just once in __init__()
    """

    def __iter__(self) -> Iterator[int]:
        dataset = DatasetFromSampler(self.sampler)

        if self.drop_last and len(dataset) % self.num_replicas != 0:  # type: ignore
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                # `type:ignore` is required because Dataset cannot provide a default __len__
                # see NOTE in pytorch/torch/utils/data/sampler.py
                (len(dataset) - self.num_replicas) / self.num_replicas  # type: ignore
            )
        else:
            self.num_samples = math.ceil(len(dataset) / self.num_replicas)  # type: ignore
        self.total_size = self.num_samples * self.num_replicas

        return super().__iter__()


def generate_default_filename():
    return f"{datetime.date.today().isocalendar()[0]}_{datetime.date.today().isocalendar()[1]}_{datetime.date.today().isocalendar()[2]}__" \
           f"{datetime.datetime.now().strftime('%H_%M_%S')}___{''.join(random.choice(string.ascii_lowercase) for x in range(5))}"
