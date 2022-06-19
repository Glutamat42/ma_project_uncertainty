import numpy as np
import pandas as pd
import scipy.stats

def calc(filepath):
       # https://stats.stackexchange.com/a/239982
       df = pd.read_csv(filepath,
                   dtype={'cameraFront': object,
                          'canSpeed': np.float32,
                          'canSteering': np.float32,
                          'chapter': int,
                          'bins_canSteering': int,
                          })
       # print(df['bins_canSteering'].value_counts())
       print(scipy.stats.entropy(list(df['bins_canSteering'].value_counts())))
       print(scipy.stats.entropy(list(df['bins_canSteering'].value_counts()))/np.log(40))

calc("/home/markus/datasets/drive360_prepared_4_5k/train_chapters.csv")
calc("/home/markus/datasets/drive360_prepared_4_5k/test_chapters.csv")
calc("/home/markus/datasets/drive360_prepared_4_5k/validation_chapters.csv")

calc("/home/markus/datasets/drive360_prepared_4_5k/validation_chapters_1_2.csv")


# calc("/home/markus/datasets/drive360_prepared_4_5k/train.csv")
# calc("/home/markus/datasets/drive360_prepared_4_5k/test.csv")
# calc("/home/markus/datasets/drive360_prepared_4_5k/validation.csv")
#
# calc("/home/markus/datasets/drive360_prepared_4_5k/validation_1_2.csv")

