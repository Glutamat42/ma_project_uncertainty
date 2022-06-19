import os
import random
from typing import Callable

import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps
import pandas as pd
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset


# dataset preperation
# preprocess datasets (generate index files containing labels):
# - generate metadata csv (and filtered version) from mat files (for wiki and imdb)
#     - adjust and run convert_mat_to_csv.py
#     - copy both generated csv files to dataset root (see next point)
# - copy from script/dataset_preparation/ to dataset root (containing folders 00,01,02,... / "original images")
#     - depending on the dataset either gen_index_AAF.sh or gen_index_imdb_wiki.sh
#     - prepare_age_dataset.py
# - some basic python libs are required (pandas, numpy, scipy, ...)
# - run the script with bash (with bash not with sh!!, will take a long time, because bash is surprisingly slow)
# - run prepare_age_dataset.py
# script will delete some files in source folder for imdb/wiki dataset (should be corrupted)
# only run once, otherwise resulting dataset will have duplicates (if running twice remove gt.json)
# output will contain the balanced and prepared dataset


class DatasetAge(Dataset):
    """ using naming from steering dataset for simpler implementation.
    It's a bit ugly but doesnt have other negative effects than a "wrong name" """
    def __init__(self, data_dir: str, csv_name: str, transform: Callable = transforms.ToTensor(), flip=False,
                 enable_src_image_passthrough=False):
        """

        Args:
            data_dir:
            csv_name:
            transform:
            flip:
            enable_src_image_passthrough: required for some fancy visualizations, but requires more ram
        """
        super().__init__()
        self.transform = transform
        self.data_dir = data_dir
        self.csv_name = csv_name
        self.flip = flip
        self.enable_src_image_passthrough = enable_src_image_passthrough

        #### reading in dataframe from csv #####
        self.dataframe = pd.read_csv(os.path.join(self.data_dir, self.csv_name),
                                     dtype={'label_norm': np.float32,
                                            'image': object,
                                            })

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        orig_image = Image.open(os.path.join(self.data_dir, row['image'])).convert('RGB')
        label = torch.tensor([row['label_norm']])

        if self.flip and random.random() > 0.5:
            orig_image = ImageOps.mirror(orig_image)

        image = self.transform(orig_image)

        if self.enable_src_image_passthrough:
            more = {'filename': row['image'], 'orig_image': TF.to_tensor(orig_image)}
        else:
            more = {'filename': row['image']}
        return {'cameraFront': image}, label, more

    def __len__(self):
        return len(self.dataframe)
