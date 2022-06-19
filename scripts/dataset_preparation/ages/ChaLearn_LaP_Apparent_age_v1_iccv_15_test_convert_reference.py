import pandas as pd
import numpy as np

# config
std = 13.74860141481293
mean = 37.247085073975434
# END - config

print("be sure to have set mean and std to the values of meta.json. Also be sure to use the meta.json from the dataset the model was trained on!")

df_src = pd.read_csv("Reference.csv", sep=';', names=['file', 'age', 'unknown_field'], dtype={"file": object, "age": int, "unknown_field": np.float64})

df_target = pd.DataFrame()
df_target['label'] = df_src['age']
df_target['image'] = df_src['file']
df_target['label_norm'] = (df_src['age'] - mean) / std

df_target.to_csv("test.csv")

print("test.csv generated")

