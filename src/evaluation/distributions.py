import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tabulate import tabulate

sys.path.append("src")
from utils.data_handling import data_loader

FEATURE_NAMES = [
    "Arterial Blood Pressure",
    "Arterial Diastolic Blood Pressure",
    "Arterial Systolic Blood Pressure",
    "Heart Rate",
    "Mean Noninvasive Blood Pressure",
    "Noninvasive Diastolic Blood Pressure",
    "Noninvasive Systolic Blood Pressure",
    "Peripheral Capillary Oxygen Saturation",
    "Respiratory Rate",
]


def process_and_display_statistics_table(original_data, file_prefixes, epsilon_values):
    data_list = [original_data]

    for file_prefix in file_prefixes:
        for eps in epsilon_values:
            file_path = os.path.join("data_synthetic", f"{file_prefix}_{eps}eps.csv")
            synthetic_data, _ = data_loader(file_path)
            synthetic_data = synthetic_data[:, :-1]
            data_list.append(synthetic_data)

    all_stats = []
    for k, data in enumerate(data_list):
        stats_list = []
        for i in range(9):
            feature_data = data[:, i * 5 : (i + 1) * 5].flatten()

            if k == 0:
                stats_dict = {
                    "Feature": FEATURE_NAMES[i],
                    f"Mean {k + 1}": np.mean(feature_data),
                    f"SD {k + 1}": np.std(feature_data),
                }
            else:
                stats_dict = {
                    f"Mean {k + 1}": np.mean(feature_data),
                    f"SD {k + 1}": np.std(feature_data),
                }
            stats_list.append(stats_dict)
        all_stats.append(stats_list)

    headers = []
    for stats_list in all_stats:
        headers.extend(stats_list[0].keys())

    final_rows = []
    for j, stats in enumerate(all_stats):
        rows = [list(stat.values()) for stat in stats]
        if j == 0:
            final_rows = rows
        else:
            for n, row in enumerate(rows):
                final_rows[n] = final_rows[n] + row

    print(final_rows)
    print(tabulate(final_rows, headers=headers, tablefmt="grid"))


def calculate_feature_distributions(data):
    feature_distributions = [data[:, i * 5 : (i + 1) * 5].flatten() for i in range(9)]
    return feature_distributions


def plot_feature_distributions(
    distributions_list, labels, num_features=9, save=False, save_name="distribution_plot.png"
):
    plt.figure(figsize=(16, 12))

    for i in range(num_features):
        plt.subplot(3, 3, i + 1)

        for feature_data, label in zip(distributions_list, labels):
            sns.kdeplot(feature_data[i], label=label, linewidth=2)

        plt.title(FEATURE_NAMES[i], fontsize=12)
        plt.xlabel("Value", fontsize=10)
        plt.ylabel("Density", fontsize=10)
        if i != 8:
            plt.legend(loc="upper right", fontsize=8)
        else:
            plt.legend(loc="upper left", fontsize=8)

    plt.subplots_adjust(hspace=0.28, wspace=0.22)

    if save:
        save_path = os.path.join("results")
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, save_name), dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_name}")
    else:
        plt.show()


def process_and_plot_distributions(original_data, file_prefixes, epsilon_values, labels, save_name):
    original_distributions = calculate_feature_distributions(original_data)

    synthetic_distributions = []
    for file_prefix in file_prefixes:
        for eps in epsilon_values:
            file_path = os.path.join("data_synthetic", f"{file_prefix}_{eps}eps.csv")
            synthetic_data, _ = data_loader(file_path)
            print(f"Amount of true labels in {file_path}:", sum(synthetic_data[:, -1]))
            synthetic_data = synthetic_data[:, :-1]
            synthetic_distributions.append(calculate_feature_distributions(synthetic_data))

    plot_feature_distributions(
        [original_distributions] + synthetic_distributions,
        labels=labels,
        num_features=9,
        save=True,
        save_name=save_name,
    )


if __name__ == "__main__":
    training_data_features, _ = data_loader(
        os.path.join("data", "mimic-iii_preprocessed", "pickle_data", "training_data.pkl")
    )

    labels = [
        "Original Data",
        "PATEGAN (Synthcity)",
        "DPGAN (Synthcity)",
        "PATECTGAN (SmartNoise)",
        "DPCTGAN (SmartNoise)",
    ]

    file_prefixes = [
        "synthcity_pategan",
        "synthcity_dpgan",
        "smartnoise_patectgan",
        "smartnoise_dpctgan",
    ]
    eps_values = [10]

    process_and_display_statistics_table(
        original_data=training_data_features, file_prefixes=file_prefixes, epsilon_values=eps_values
    )

    # process_and_plot_distributions(
    #     training_data_features,
    #     file_prefixes=file_prefixes,
    #     epsilon_values=eps_values,
    #     labels=labels,
    #     save_name="1eps_comparison.png",
    # )
