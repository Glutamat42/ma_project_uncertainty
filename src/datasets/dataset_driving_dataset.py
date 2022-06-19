import os
import random
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class DrivingDataset(Dataset):
    def __init__(self, data_dir: str, csv_name: str, transform: Callable = transforms.ToTensor(), flip=False,
                 enable_src_image_passthrough=False):
        super().__init__()
        self.transform = transform
        self.data_dir = data_dir
        self.csv_name = csv_name
        self.flip = flip
        self.enable_src_image_passthrough=enable_src_image_passthrough

        self.dataframe = pd.read_csv(os.path.join(self.data_dir, self.csv_name),
                                     names=['frame', 'steering'], sep=' ', dtype={
                'frame': object,
                'steering': np.float32
            })

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        orig_image = Image.open(os.path.join(self.data_dir, row['frame']))
        label = torch.tensor([self.dataframe['steering'].iloc[index]])

        orig_image = TF.to_tensor(orig_image)
        if self.flip and random.random() > 0.5:
            orig_image = TF.hflip(orig_image)
            label = -label

        image = self.transform(orig_image)

        return {'cameraFront': image}, label, {'filename': row['cameraFront'],
                                               'orig_image': orig_image if self.enable_src_image_passthrough else None}

    def __len__(self):
        return len(self.dataframe)
