from autrainer.models.leaf import LEAFNet
import torch
import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

root_folder = "results/noise/"
initialisation_name_converter = {"LEAFNet": "Mel-scale", "bark": "Bark-scale", "linear-constant": "Linear", "constant": "Constant"}
augmentation_type_name_converter = {"bluenoise": "High-passed noise", "pinknoise": "Low-passed noise", "bandpass": "Bandpass filter"}

dataset_name_converter = {"AIBO-wav": "SER", "SpeechCommands-wav": "SR", "DCASE2020-wav-16k": "ASC", "DCASE2018-T3-16k": "BR"}
initialisation_column = {"LEAFNet": 0, "LEAFNet-constant": 1}

sns.set(style="whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
initial_plotted = {}


exp_folders = [o for o in os.listdir(root_folder) 
                if os.path.isdir(os.path.join(root_folder,o))]
exp_folders.sort()

dataset = exp_folders[0].split("_")[0]

sample_rate = 16000 #important for everything here

param_dicts = {}
order_dict = {}
value_range = {}

for exp_root_folder in exp_folders:
    print(exp_root_folder)
    augmentation = exp_root_folder.split("_")[1]
    augmentation_type = augmentation.split("-")[0]
    if "bandpass" in augmentation_type:
        augmentation_strength = int(augmentation.split("-")[1])
    elif "pink" in augmentation_type:
        augmentation_strength = int(augmentation.split("-")[3])
    elif "blue" in augmentation_type:
        augmentation_strength = int(augmentation.split("-")[3])
    
    initialisation = exp_root_folder.split("_")[2]
    initialisation_augmentation = initialisation + "-" + augmentation_type

    if not "constant" in initialisation :
        continue
    
    if initialisation_augmentation not in param_dicts.keys():
        param_dicts[initialisation_augmentation] = {}
        order_dict[initialisation_augmentation] = {}
        value_range[initialisation_augmentation] = {}
        value_range[initialisation_augmentation]["Min"] = int(augmentation_strength)
        value_range[initialisation_augmentation]["Max"] = int(augmentation_strength)
    if augmentation_strength not in param_dicts[initialisation_augmentation]:
        param_dicts[initialisation_augmentation][augmentation_strength] = {}
        order_dict[initialisation_augmentation][augmentation_strength] = {}
    if int(augmentation_strength) < value_range[initialisation_augmentation]["Min"]:
        value_range[initialisation_augmentation]["Min"] = int(augmentation_strength)
    if int(augmentation_strength) > value_range[initialisation_augmentation]["Max"]:
        value_range[initialisation_augmentation]["Max"] = int(augmentation_strength)


    exp_root_folder = os.path.join(root_folder, exp_root_folder)
    # plot_dir = os.path.join(exp_root_folder, "plots")
    # os.makedirs(plot_dir, exist_ok=True)
    n_classes = 35
    n_epochs = 50

    model = LEAFNet(n_classes)
    params_of_interest = ["leaf.filterbank.center_freqs"]

    df_columns = ["epoch"] + params_of_interest



    epoch_dirs = ["_initial"]

    epoch_dirs.append("_best")


    for epoch_dir in epoch_dirs:
        model_state_file = os.path.join(exp_root_folder, epoch_dir, "model.pth.tar")
        if not os.path.isfile(model_state_file):
            continue
        model_state = torch.load(model_state_file)
        param_values = []
        for param_name in params_of_interest:
            param_values = model_state[param_name].numpy()
            if "center" in param_name:
                param_values /= (2 * np.pi / sample_rate)
            # elif "bandwidth" in param_name:
            #     param_values = (sample_rate / 2.) / param_values
            if param_name not in param_dicts[initialisation_augmentation]:
                param_dicts[initialisation_augmentation][augmentation_strength][param_name] = {}
            if not epoch_dir in param_dicts[initialisation_augmentation][augmentation_strength][param_name].keys():
                param_dicts[initialisation_augmentation][augmentation_strength][param_name][epoch_dir] = {}
                param_dicts[initialisation_augmentation][augmentation_strength][param_name][epoch_dir] = pd.DataFrame()
                if "center" in param_name:
                    order_dict[initialisation_augmentation][augmentation_strength][epoch_dir] = {}
            if "center" in param_name:
                order_dict[initialisation_augmentation][augmentation_strength][epoch_dir] = param_values.argsort()
            sorted_params = param_values[order_dict[initialisation_augmentation][augmentation_strength][epoch_dir]]
            param_dicts[initialisation_augmentation][augmentation_strength][param_name][epoch_dir] = sorted_params

cmap = plt.cm.viridis
for column, initialisation_augmentation in enumerate(param_dicts.keys()):

    ax = axes[column]
    augmentation_strengths = list(param_dicts[initialisation_augmentation].keys())
    augmentation_strengths.sort()
    for augmentation_strength in augmentation_strengths:
        for row, param_name in enumerate(param_dicts[initialisation_augmentation][augmentation_strength].keys()):
            for epoch in param_dicts[initialisation_augmentation][augmentation_strength][param_name].keys():
                if not initialisation_augmentation in initial_plotted.keys():
                    initial_plotted[initialisation_augmentation] = {}
                if not augmentation_strength in initial_plotted[initialisation_augmentation].keys():
                    initial_plotted[initialisation_augmentation][augmentation_strength] = {}
                if param_name in initial_plotted[initialisation_augmentation][augmentation_strength].keys() and epoch == "_initial":
                    continue
                values = param_dicts[initialisation_augmentation][augmentation_strength][param_name][epoch]

                if epoch == "_initial":
                    label = "Initial"
                    color = plt.cm.tab10.colors[0] 
                else:
                    norm = plt.Normalize(value_range[initialisation_augmentation]["Min"], value_range[initialisation_augmentation]["Max"])
                    label = str(augmentation_strength)
                    color = colors = cmap(norm(int(augmentation_strength)))
                    # color = dataset_color[dataset]
                matplotlib.rcParams['mathtext.fontset'] = 'stix'
                matplotlib.rcParams['font.family'] = 'STIXGeneral'
                ax.errorbar(np.arange(len(values), dtype=int), values, fmt='o', markersize=3.5, alpha=0.7, color=color, label=label)
                font = {'family': 'STIXGeneral'}
                ax.legend(fontsize='x-large', ncol=2, prop=font)
                for spine in plt.gca().spines.values():
                    spine.set_zorder(0)
                if epoch == "_initial":
                    initial_plotted[initialisation_augmentation][augmentation_strength][param_name] = True
            if column == 0:
                if "center" in param_name:
                    ax.set_ylabel("Centre Frequency [Hz]", fontsize=28, fontfamily = 'STIXGeneral')
                elif "bandwidth" in param_name:
                    ax.set_ylabel("Bandwidth [Hz]", fontsize=28)
    print(augmentation_type_name_converter[initialisation_augmentation.split("-")[-1]])
    
    ax.set_title(augmentation_type_name_converter[initialisation_augmentation.split("-")[-1]], fontsize=30, fontfamily = 'STIXGeneral')
    ax.set_xlabel("Frequency Band", fontsize=28, fontfamily = 'STIXGeneral')
    font_ticks = {'family': 'STIXGeneral', 'size': 22}
    ax.tick_params(axis='x', labelsize=22) 
    ax.tick_params(axis='y', labelsize=22) 
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                    ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontname('STIXGeneral')

          

plt.tight_layout()

plt.savefig(root_folder + "parameter_change_augmentation.pdf")
plt.clf()
