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


def display_statistics_table(data, title):
    stats = []

    for i in range(9):
        feature_data = data[:, i * 5 : (i + 1) * 5].flatten()

        stats_dict = {
            "Feature": FEATURE_NAMES[i],
            "Mean": np.mean(feature_data),
            "Median": np.median(feature_data),
            "Std Dev": np.std(feature_data),
            "Variance": np.var(feature_data),
            "Min": np.min(feature_data),
            "Max": np.max(feature_data),
        }
        stats.append(stats_dict)

    headers = stats[0].keys()
    rows = [list(stat.values()) for stat in stats]
    print(title)
    print(tabulate(rows, headers=headers, tablefmt="grid"))


def calculate_feature_distribution(data):
    print("Data shape:", data.shape)

    feature_distributions = []

    for i in range(9):
        feature_data = data[:, i * 5 : (i + 1) * 5].flatten()
        feature_distributions.append(feature_data)

    return feature_distributions


def plot_feature_distributions(
    distributions_list, labels, num_features=9, save=False, save_name="distribution_plot.png"
):
    plt.figure(figsize=(16, 12))

    for i in range(num_features):
        plt.subplot(3, 3, i + 1)

        for feature_data, label in zip(distributions_list, labels):
            sns.kdeplot(feature_data[i], label=label, linewidth=3)

        plt.title(FEATURE_NAMES[i], fontsize=12)
        plt.xlabel("Value", fontsize=10)
        plt.ylabel("Density", fontsize=10)
        plt.legend(loc="upper right", fontsize=10)

    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    if save:
        save_path = os.path.join("results")
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, save_name), dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_name}")
    else:
        plt.show()


if __name__ == "__main__":
    data, _ = data_loader(os.path.join("data", "mimic-iii_preprocessed", "pickle_data", "original_data.pkl"))

    dpctgan_data, _ = data_loader(os.path.join("data", "synthetic_data", "smartnoise_dpctgan_1eps.csv"))
    print("Amount of true labels:", sum(dpctgan_data[:, -1]))
    dpctgan_data = dpctgan_data[:, :-1]

    patectgan_data, _ = data_loader(os.path.join("data", "synthetic_data", "smartnoise_patectgan_1eps.csv"))
    print("Amount of true labels:", sum(patectgan_data[:, -1]))
    patectgan_data = patectgan_data[:, :-1]

    patectgan3_data, _ = data_loader(os.path.join("data", "synthetic_data", "smartnoise_patectgan_3eps.csv"))
    print("Amount of true labels:", sum(patectgan3_data[:, -1]))
    patectgan3_data = patectgan3_data[:, :-1]

    original_distributions = calculate_feature_distribution(data)
    dpctgan_distributions = calculate_feature_distribution(dpctgan_data)
    patectgan_distributions = calculate_feature_distribution(patectgan_data)
    patectgan3_distributions = calculate_feature_distribution(patectgan3_data)

    plot_feature_distributions(
        [original_distributions, dpctgan_distributions, patectgan_distributions, patectgan3_distributions],
        labels=["Original Data", "DPCTGAN", "PATECTGAN 1", "PATECTGAN 3"],
        num_features=9,
        save=True,
        save_name="distribution_plot.png",
    )

    display_statistics_table(data, title="Original Data")
    display_statistics_table(dpctgan_data, title="DPCTGAN")
    display_statistics_table(patectgan_data, title="PATECTGAN 1")
    display_statistics_table(patectgan3_data, title="PATECTGAN 3")
