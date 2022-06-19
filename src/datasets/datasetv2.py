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


class Drive360(Dataset):
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

        # ideas: load everything to ram in constructor
        # takes too much ram. train_chapters_1_2.csv 28 threads: 224x224x3x34000x28 = ~136GB
        # maybe with shared array: https://github.com/ptrblck/pytorch_misc/blob/master/shared_array.py
        # mem for 4 individual instances (per gpu) and pregenerated flips: ~86GB
        #
        # create second list with flipped samples: -> never... even more ram


        # convert df to np array (or tensors) (benchmark if its faster)
        #
        # from random import randrange
        # import pandas as pd
        # from timeit import default_timer as timer
        # df = pd.read_csv("/home/markus/datasets/drive360_prepared_4_5k/train_chapters_1_2.csv")
        # indexes = [randrange(len(df)-1) for x in range(64)]
        # nd_camerFront = df['cameraFront'].to_numpy()
        # nd_canSteering=df['canSteering'].to_numpy()
        # start = timer(); [df.iloc[x] for x in indexes]; print(timer()-start)
        # # ~0.015
        # start = timer(); [(nd_camerFront[x], nd_canSteering[x]) for x in indexes]; print(timer()-start)
        # # ~ 0.0003
        #
        # for one full batch (of train_chapters.csv) the estimated potential is a bit more than 1 second

        #### reading in dataframe from csv #####
        self.dataframe = pd.read_csv(os.path.join(self.data_dir, self.csv_name),
                                     dtype={'cameraFront': object,
                                            'canSpeed': np.float32,
                                            'canSteering': np.float32,
                                            'bins_canSteering': int
                                            })

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        # torchvision.io.read_image(...)
        orig_image = Image.open(os.path.join(self.data_dir, row['cameraFront']))
        # from torchvision.io.image import read_file, decode_jpeg
        # data = read_file('path_to_image.jpg')  # raw data is on CPU
        # img = decode_jpeg(data, device='cuda')  # decoded image in on GPU
        # ------
        # image = torch.from_numpy(np.array(image))
        label = torch.tensor([row['canSteering']])

        # orig_image = TF.to_tensor(orig_image)
        if self.flip and random.random() > 0.5:
            # orig_image = TF.hflip(orig_image)
            orig_image = ImageOps.mirror(orig_image)
            label = -label

        image = self.transform(orig_image)

        # import pdb; pdb.set_trace()
        # cv2.imshow("frame", image.numpy().transpose(1, 2, 0))
        # print(label)
        # cv2.waitKey()
        if self.enable_src_image_passthrough:
            more = {'filename': row['cameraFront'], 'orig_image': TF.to_tensor(orig_image)}
        else:
            more = {'filename': row['cameraFront']}
        return {'cameraFront': image}, label, more

    def __len__(self):
        return len(self.dataframe)
