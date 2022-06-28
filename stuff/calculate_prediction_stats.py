import json
import math
import os


# aleatoric ensemble 10 supervised
grouped_bbb_graves_dino = json.load(open("/home/markus/workspace/data/final_data/aleatoric/aleatoric_ensemble/output_aleatoric_supervised_best/ensemble_10/plots_metrics/metrics.json"))
epoch=8

# bbb graves hand selected
# grouped_bbb_graves_dino = json.load(open("/home/markus/workspace/data/final_data/bbb_graves/outputs_supervised_whole_training/grouped/metrics.json"))
# grouped_bbb_graves_dino = json.load(open("/home/markus/workspace/data/final_data/bbb_graves/outputs_dino_whole_training/grouped/metrics.json"))
# epoch=20

std_dataset=27.41716766357422

def print_value(name, data, decimals=5, rmse_degree_std=None):
    value = data[name]["mean"][epoch-1]
    std=data[name]['std'][epoch-1]

    if rmse_degree_std is not None:
        value = math.sqrt(value) * rmse_degree_std
        std = math.sqrt(std) * rmse_degree_std
        name = f"{name} degree"

    print(f"{name}")
    print(round(value, decimals))
    print(round(std, decimals))
    print(str(int(round(std/value, 2)*100)) + "%")
    print("-----------------")

print_value("MSE", grouped_bbb_graves_dino)
print_value("MSE", grouped_bbb_graves_dino, rmse_degree_std=std_dataset)
print_value("NLL", grouped_bbb_graves_dino)
print_value("miscalibration area", grouped_bbb_graves_dino)
print_value("sharpness", grouped_bbb_graves_dino)



base_folder="/home/markus/workspace/data/final_data/bbb_graves/model_supervised_20/"
epoch=20

model_dirs = os.walk(base_folder)


for x in [("3", "03"), ("5", "05"), ("10", "10"), ("20", "20")]:
    x[1]
    round(json.load(open(f"outputs_supervised_comparable_{x[1]}/ensemble_{x[0]}/plots_metrics/metrics.json"))[-1]['validation_mse'], 4)
    round(json.load(open(f"outputs_supervised_comparable_{x[1]}/ensemble_{x[0]}/plots_metrics/metrics.json"))[-1]['uncertainty']['nll'], 1)
    round(json.load(open(f"outputs_supervised_comparable_{x[1]}/ensemble_{x[0]}/plots_metrics/metrics.json"))[-1]['uncertainty']['miscal_area'], 2)