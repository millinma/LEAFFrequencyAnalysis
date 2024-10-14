import os

import matplotlib.pyplot as plt
from autrainer.models.leaf import LEAFNet
import numpy as np
import pandas as pd
import seaborn as sns
import torch


all_root = "results/default/"


dataset_name_converter = {
    "AIBO-wav": "SER",
    "SpeechCommands-wav": "SR",
    "DCASE2020-wav-16k": "ASC",
    "DCASE2018-T3-16k": "BAD",
}
initialisation_column = {"LEAFNet": 0, "LEAFNet-const": 1}
dataset_numbering = {
    "AIBO-wav": 0,
    "SpeechCommands-wav": 1,
    "DCASE2020-wav-16k": 2,
    "DCASE2018-T3-16k": 3,
}
datasets = ["AIBO-wav", "SpeechCommands-wav", "DCASE2020-wav-16k", "DCASE2018-T3-16k"]
initialisation_name_converter = {
    "LEAFNet": "Mel-scale",
    "bark": "Bark-scale",
    "linear-constant": "Linear",
    "LEAFNet-const": "Constant",
}


sns.set(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(15, 5))
initial_plotted = {}

# dataset_color = {}

for n_dataset, dataset in enumerate(datasets):
    root_folder = all_root + "training/"
    exp_folders = [
        o
        for o in os.listdir(root_folder)
        if os.path.isdir(os.path.join(root_folder, o))
    ]
    exp_folders.sort()

    sample_rate = 16000  # important for everything here

    # Speechcommand
    param_dicts = {}
    order_dict = {}

    for exp_root_folder in exp_folders:
        actual_folder = exp_root_folder
        dataset_exp = exp_root_folder.split("_")[0]

        initialisation = exp_root_folder.split("_")[1]
        if initialisation not in param_dicts.keys():
            param_dicts[initialisation] = {}
            order_dict[initialisation] = {}
        seed = exp_root_folder.split("_")[-1]

        print(exp_root_folder)

        exp_root_folder = os.path.join(root_folder, exp_root_folder)
        plot_dir = os.path.join(exp_root_folder, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        n_classes = 35

        model = LEAFNet(n_classes)

        params_of_interest = [
            "leaf.filterbank.center_freqs",
            "leaf.filterbank.bandwidths",
        ]

        df_columns = ["epoch"] + params_of_interest

        # epoch at which we invesitgate the model state
        n_epochs = 50
        epoch_dirs = ["_initial"]
        epoch_dirs.append("epoch_" + str(n_epochs))

        # collect parameters for each run
        for epoch_dir in epoch_dirs:
            model_state_file = os.path.join(
                exp_root_folder, epoch_dir, "model.pt"
            )
            if not os.path.isfile(model_state_file):
                continue
            model_state = torch.load(model_state_file)
            param_values = []
            for param_name in params_of_interest:
                # initial_plotted[initialisation][param_name] = False
                param_values = model_state[param_name].numpy()
                print(param_values)
                # rescale to the hertz scale
                if "center" in param_name:
                    param_values /= 2 * np.pi / sample_rate
                elif "bandwidth" in param_name:
                    param_values = (sample_rate / 2.0) / param_values
                if param_name not in param_dicts[initialisation]:
                    param_dicts[initialisation][param_name] = {}
                if (
                    epoch_dir
                    not in param_dicts[initialisation][param_name].keys()
                ):
                    param_dicts[initialisation][param_name][epoch_dir] = {}
                    param_dicts[initialisation][param_name][epoch_dir] = (
                        pd.DataFrame()
                    )
                    if "center" in param_name:
                        order_dict[initialisation][epoch_dir] = {}
                if (
                    "center" in param_name
                    and str(seed) not in order_dict[initialisation][epoch_dir]
                ):
                    order_dict[initialisation][epoch_dir][str(seed)] = (
                        param_values.argsort()
                    )
                sorted_params = param_values[
                    order_dict[initialisation][epoch_dir][str(seed)]
                ]
                param_dicts[initialisation][param_name][epoch_dir][
                    str(seed)
                ] = sorted_params
    
    # plotting
    for column, initialisation in enumerate(param_dicts.keys()):
        for row, param_name in enumerate(param_dicts[initialisation].keys()):
            ax = axes[row, initialisation_column[initialisation]]
            for epoch in param_dicts[initialisation][param_name].keys():
                if initialisation not in initial_plotted.keys():
                    initial_plotted[initialisation] = {}
                if (
                    param_name in initial_plotted[initialisation].keys()
                    and epoch == "_initial"
                ):
                    continue
                plot_df = pd.DataFrame()
                plot_df["mean"] = param_dicts[initialisation][param_name][
                    epoch
                ].mean(axis=1)
                plot_df["std"] = param_dicts[initialisation][param_name][
                    epoch
                ].std(axis=1)
                # plot_df = plot_df.sort_values(by='mean')
                if epoch == "_initial" and "constant" in initialisation:
                    check = param_dicts[initialisation][param_name][
                        epoch
                    ].values
                    print("")
                if epoch == "_initial":
                    label = "Initial"
                    color = plt.cm.tab10.colors[0]
                else:
                    label = dataset_name_converter[dataset]
                    color = plt.cm.tab10.colors[n_dataset + 1]
                    # color = dataset_color[dataset]
                ax.errorbar(
                    plot_df.index,
                    plot_df["mean"],
                    yerr=plot_df["std"],
                    fmt="o",
                    markersize=3,
                    alpha=0.7,
                    color=color,
                    label=label,
                )
                font = {"family": "STIXGeneral"}  # Adjust font size as needed
                if row == 0 and column == 2:
                    ncols = 1
                else:
                    ncols = 2
                ax.legend(fontsize="xx-small", prop=font, ncols=ncols)
                for spine in plt.gca().spines.values():
                    spine.set_zorder(0)
                if epoch == "_initial":
                    initial_plotted[initialisation][param_name] = True
            if row == 0:
                ax.set_title(
                    initialisation_name_converter[
                        initialisation
                    ]
                )
            else:
                ax.set_xlabel("Frequency Band")
            for item in (
                [ax.title, ax.xaxis.label, ax.yaxis.label]
                + ax.get_xticklabels()
                + ax.get_yticklabels()
            ):
                item.set_fontname("STIXGeneral")

axes[0][0].set_ylabel("Centre Frequency [Hz]")

axes[1][0].set_ylabel("Bandwidth [Hz]")
# plt.title(dataset)
# plt.suptitle(dataset, fontsize=16)
plt.tight_layout()
# plt.savefig(all_root + "parameter_change.pdf")
plt.savefig(all_root + "parameter_change_" + actual_folder + ".pdf")
plt.clf()
