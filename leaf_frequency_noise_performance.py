from autrainer.models.leaf import LEAFNet
import torch
import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

metric_path = "results/noise/summary/metrics.csv"

root_folder = "results/noise/"

dataset_name_converter = {"AIBO-wav": "SER", "SpeechCommands-wav": "SR", "DCASE2020-wav-16k": "ASC", "DCASE2018-T3-16k": "BR"}
augmentation_type_name_converter = {"bluenoise": "Highpassed noise", "pinknoise": "Lowpassed noise", "bandpass": "Bandpass-filter"}
initialisation_column = {"LEAFNet": 0, "LEAFNet-constant": 1}

colour_converter = {"bandpass": 0, "bluenoise": 1, "pinknoise": 2}
symbol_converter = {"LEAFNet": "x", "LEAFNet-constant": "o"}
colour_converter = {"bandpass": 0, "bluenoise": 1, "pinknoise": 2}
symbol_converter = {"LEAFNet": "x", "LEAFNet-constant": "o"}

initialisation_name_converter = {"LEAFNet": "Mel-scale", "LEAFNet-constant": "Constant"}

sns.set(style="whitegrid")
initial_plotted = {}

metrics = pd.read_csv(metric_path)

plot_data = {}

for i in range(len(metrics)):
    run_name = metrics.iloc[i]["run_name"]
    initialisation = run_name.split("_")[2]
    augmentation = run_name.split("_")[1]
    augmentation_type = augmentation.split("-")[0]
    initialisation_augmentation = initialisation + "-" + augmentation_type
    if initialisation_augmentation not in plot_data.keys():
        plot_data[initialisation_augmentation] = {}
        plot_data[initialisation_augmentation] = {"uar": [], "strength": []}

    if "bandpass" in augmentation_type:
        augmentation_strength = int(augmentation.split("-")[1])
    elif "pink" in augmentation_type:
        augmentation_strength = int(augmentation.split("-")[3])
    elif "blue" in augmentation_type:
        augmentation_strength = int(augmentation.split("-")[3])
    uar = float(metrics.iloc[i]["uar"])

    plot_data[initialisation_augmentation]["strength"].append(augmentation_strength)    
    plot_data[initialisation_augmentation]["uar"].append(uar)    

plt.figure(figsize=(12, 6))
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
for initialisation_augmentation in plot_data.keys():
    split = initialisation_augmentation.split("-")
    augmentation_type = split[-1]
    initialisation = "-".join(split[:-1])
    sns.regplot(
        x=plot_data[initialisation_augmentation]["strength"], y=plot_data[initialisation_augmentation]["uar"], 
            marker=symbol_converter[initialisation], color=plt.cm.tab10.colors[colour_converter[augmentation_type]], 
            label = initialisation_name_converter[initialisation] + "_" + augmentation_type_name_converter[augmentation_type])


plt.legend(fontsize='large', ncol=3)
plt.xlabel('Augmentation Center Frequency', fontsize=28)
plt.ylabel('UAR', fontsize=28)
plt.xticks(fontsize=24)  # Adjust fontsize as needed for xtick labels
plt.yticks(fontsize=24) 

plt.tight_layout()
plt.savefig(root_folder + "performance_augmentation.pdf")
plt.clf()