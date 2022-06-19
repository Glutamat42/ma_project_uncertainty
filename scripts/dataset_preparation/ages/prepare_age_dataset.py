import glob
import os
import json
import shutil
import scipy.stats
import pandas as pd
import numpy as np
from pathlib import Path
from time import strftime, gmtime
from matplotlib import pyplot as plt

### config
bin_cut = None
test_split = 20
valid_split = 10
output_dir = "output"
###

# set random seed
random_seed = 42
np.random.seed(random_seed)

# fix pandas output
pd.set_option('display.max_rows', None)

# create output_dir
Path(output_dir).mkdir()

# load csv and create some plots
df = pd.read_csv("gt.csv",
                 dtype={'label': int,
                        'image': object,
                        })
df.label.hist(bins=len(df.label.unique()))
plt.savefig(os.path.join(output_dir, 'source_data_distribution.png'))
plt.close()
df.label.hist(bins=len(df.label.unique()), range=(1, 99))
plt.savefig(os.path.join(output_dir, 'source_data_distribution_valid_values.png'))
plt.close()

# remove invalid ages
df = df[df.label > 0]
df = df[df.label < 100]

# remove invalid frames based on *_filtered.csv (no person / 2 persons on source frame)
meta_df_filename_candidates = glob.glob('*_filtered.csv')
if len(meta_df_filename_candidates) == 1:
    meta_df = pd.read_csv(meta_df_filename_candidates[0], dtype={'dob': int,
                                                                 'photo_taken': int,
                                                                 'full_path': object,
                                                                 'name': object,
                                                                 'face_score': np.float64,
                                                                 'second_face_score': np.float64,
                                                                 })
    df['image'] = df['image'].apply(lambda x: os.path.normpath(x))
    meta_df['full_path'] = meta_df['full_path'].apply(lambda x: os.path.normpath(x))
    df = df[df['image'].isin(meta_df['full_path'])]

    print(f"removed invalid samples based on *_filtered.csv\n{len(df)} samples left")
else:
    print("There is no *_filtered_csv file (which might or not be expected)\n"
          "-> Not removing invalid samples based on *_filtered.csv (for obvious reasons...)")

# balance
label_sizes = df.groupby('label').size()
print(label_sizes)
min_items = int(label_sizes.max() / 10) if bin_cut is None else bin_cut
df = df.groupby('label').apply(lambda g: g.sample(
    # lookup number of samples to take
    n=min_items,
    # enable replacement (oversampling) if len is less than number of samples expected
    # replace=len(g) < min_items
    # if len(g) > min_items: sample, if len(g) <= min_items and > 0: use g, else: None
) if len(g) > min_items else g if len(g) > 0 else None).sort_values(by=['image'])

# calculate some variables and generate meta object
test_sample_count = int(len(df) / 100 * test_split)
valid_sample_count = int(len(df) / 100 * valid_split)
shannon_entropy = scipy.stats.entropy(list(df['label'].value_counts()))
balance = shannon_entropy / np.log(len(label_sizes))

# normalize to [0,1]
label_mean = -50
label_std = 100
# standardize mean=0 std=1
# label_mean = float(df['label'].mean())
# label_std = float(df['label'].std())

meta = {
    'dataset_type': 'age',
    'label': {
        'mean': label_mean,
        'std': label_std
    },
    'len': {
        'train': len(df) - test_sample_count - valid_sample_count,
        'test': test_sample_count,
        'valid': valid_sample_count
    },
    'balance': balance,
    'entropy': shannon_entropy,
    'samples_per_bin': int(min_items),
    'random_seed': random_seed,
    'created_at': strftime("%Y-%m-%d %H:%M:%S", gmtime())
}

df.label.hist(bins=len(df.label.unique()))
plt.savefig(os.path.join(output_dir, 'data_distribution.png'))
plt.close()

# normalize
df['label_norm'] = (df['label'] - label_mean) / label_std

# save meta info
with open(os.path.join(output_dir, 'meta.json'), 'w') as fp:
    json.dump(meta, fp, indent=4)


# save csv fles
def save_df(df, basename):
    """ safe dataframe with differently sized subsets """
    df_2 = df[::2]
    df_4 = df_2[::2]
    df_8 = df_4[::2]
    df_16 = df_8[::2]
    df.to_csv(os.path.join(output_dir, basename + ".csv"))
    df_2.to_csv(os.path.join(output_dir, basename + "_1_2.csv"))
    df_4.to_csv(os.path.join(output_dir, basename + "_1_4.csv"))
    df_8.to_csv(os.path.join(output_dir, basename + "_1_8.csv"))
    df_16.to_csv(os.path.join(output_dir, basename + "_1_16.csv"))


df.to_csv(os.path.join(output_dir, "balanced_gt.csv"))
train_df = df.copy()
test_df = train_df.sample(n=test_sample_count)
train_df.drop(test_df.index, inplace=True)
valid_df = train_df.sample(n=valid_sample_count)
train_df.drop(valid_df.index, inplace=True)

save_df(train_df, 'train')
save_df(test_df, 'test')
save_df(valid_df, 'validation')

# copy images
for index, line in df.iterrows():
    output_path = os.path.join(output_dir, os.path.dirname(line.image))
    Path(output_path).mkdir(exist_ok=True)
    shutil.copy(line.image, output_path)
