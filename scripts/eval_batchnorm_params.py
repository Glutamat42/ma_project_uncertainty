import pandas as pd
from matplotlib import pyplot as plt

df_mcbn_values_mean = pd.read_csv("../stuff/mcbn_bn_layer_params/mcbn_values_mean.csv", delimiter=";", header=None, ).iloc[:, :-1]
mean_mcbn_values_mean = df_mcbn_values_mean.mean(axis=0)
# std_mcbn_values_mean = df_mcbn_values_mean.std(axis=0)

df_mcbn_values_var = pd.read_csv("../stuff/mcbn_bn_layer_params/mcbn_values_var.csv", delimiter=";", header=None, ).iloc[:, :-1]
mean_mcbn_values_var = df_mcbn_values_var.mean(axis=0)
# std_mcbn_values_var = df_mcbn_values_var.std(axis=0)


df_orig_values_mean = pd.read_csv("../stuff/mcbn_bn_layer_params/orig_values_mean.csv", delimiter=";", header=None, ).iloc[:, :-1]
df_orig_values_var = pd.read_csv("../stuff/mcbn_bn_layer_params/orig_values_var.csv", delimiter=";", header=None, ).iloc[:, :-1]

division_mean = (mean_mcbn_values_mean / df_orig_values_mean).to_numpy()
difference_mean = (mean_mcbn_values_mean - df_orig_values_mean).to_numpy()

division_var = (mean_mcbn_values_var / df_orig_values_var).to_numpy()
difference_var = (mean_mcbn_values_var - df_orig_values_var).to_numpy()

# mean
fig, axs = plt.subplots(2, 2)
fig.subplots_adjust(hspace=.5)
axs[0, 0].hist(list(division_mean), bins=40)
axs[0, 0].title.set_text('division_mean all')
axs[0, 1].hist(list(division_mean), bins=40, range=(-5, 5))
# axs[0, 1].hist(list(division_mean), bins=40, range=(-0, 2))
axs[0, 1].title.set_text('division_mean')
axs[1, 0].hist(list(difference_mean), bins=40)
axs[1, 0].title.set_text('difference_mean all')
# axs[1, 1].hist(list(difference_mean), bins=40, range=(-0.2, 0.2))
# axs[1, 1].title.set_text('difference_mean')
plt.show()

# var
fig, axs = plt.subplots(2, 2)
fig.subplots_adjust(hspace=.5)
axs[0, 0].hist(list(division_var), bins=40)
axs[0, 0].title.set_text('division_var all')
# axs[0, 1].hist(list(division_var), bins=40, range=(.95, 1.05))
# axs[0, 1].title.set_text('division_var')
axs[1, 0].hist(list(difference_var), bins=40)
axs[1, 0].title.set_text('difference_var all')
# axs[1, 1].hist(list(difference_var), bins=40, range=(-0.2,0.2))
# axs[1, 1].title.set_text('division_var')
plt.show()
