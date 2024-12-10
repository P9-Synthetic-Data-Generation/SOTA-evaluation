import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append("src")
from utils.load_data import data_loader


def feature_distribution(data):
    means = []
    medians = []
    std_devs = []
    mins = []
    max_values = []
    percentiles_25 = []
    percentiles_50 = []
    percentiles_75 = []

    print("Data shape:", data.shape)

    for i in range(9):
        feature_data = data[:, i * 5 : (i + 1) * 5].flatten()

        means.append(np.mean(feature_data))
        medians.append(np.median(feature_data))
        std_devs.append(np.std(feature_data))
        mins.append((np.min(feature_data), np.argmin(feature_data)))
        max_values.append((np.max(feature_data), np.argmax(feature_data)))

        percentiles_25.append(np.percentile(feature_data, 25))
        percentiles_50.append(np.percentile(feature_data, 50))
        percentiles_75.append(np.percentile(feature_data, 75))

        plt.subplot(3, 3, i + 1)
        plt.hist(feature_data, bins=50, color="skyblue", edgecolor="black", alpha=0.7)
        plt.title(f"Feature {i+1}")
        plt.xlabel("Value")
        plt.ylabel("Frequency")

    print("Means:", means)
    print("Medians:", medians)
    print("Standard Deviations:", std_devs)
    print("Min Values:", mins)
    print("Max Values:", max_values)
    print("25th Percentiles:", percentiles_25)
    print("50th Percentiles (Medians):", percentiles_50)
    print("75th Percentiles:", percentiles_75)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    data, _ = data_loader(
        os.path.join("data", "mimic-iii_preprocessed", "pickle_data", "data.pkl"),
        os.path.join("data", "mimic-iii_preprocessed", "pickle_data", "labels.pkl"),
    )

    feature_distribution(data)
