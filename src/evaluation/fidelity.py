import os
import sys

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy, ks_2samp, wasserstein_distance
from tabulate import tabulate

sys.path.append("src")
from utils.data_handling import data_loader


def calculate_feature_distributions(data):
    feature_distributions = [data[:, i * 5 : (i + 1) * 5].flatten() for i in range(9)]
    return feature_distributions


def evaluate_synthetic_data(real_data_distributions, synthetic_data_distributions, bins=30):
    """
    Evaluate the quality of synthetic data compared to real data using various metrics.

    Parameters:
        real_data_distributions (array-like): Feature-wise distributions of real dataset.
        synthetic_data_distributions (array-like): Feature-wise distributions of synthetic dataset.
        bins (int): Number of bins to use for histogram-based comparisons.

    Returns:
        dict: A dictionary containing the computed metrics.
    """
    metrics = {
        "wasserstein_distance": 0,
        "jensen_shannon_distance": 0,
        "kl_divergence": 0,
        "kolmogorov_smirnov_statistic": 0,
    }

    for real_feature, synthetic_feature in zip(real_data_distributions, synthetic_data_distributions):
        real_feature = np.ravel(real_feature)
        synthetic_feature = np.ravel(synthetic_feature)

        metrics["wasserstein_distance"] += wasserstein_distance(real_feature, synthetic_feature)

        bin_edges = np.histogram_bin_edges(real_feature, bins=bins)
        real_hist, _ = np.histogram(real_feature, bins=bin_edges, density=True)
        synthetic_hist, _ = np.histogram(synthetic_feature, bins=bin_edges, density=True)

        # Add small value to avoid log(0)
        real_hist += 1e-8
        synthetic_hist += 1e-8

        metrics["jensen_shannon_distance"] += jensenshannon(real_hist, synthetic_hist)
        metrics["kl_divergence"] += entropy(real_hist, synthetic_hist)
        ks_stat, _ = ks_2samp(real_feature, synthetic_feature)
        metrics["kolmogorov_smirnov_statistic"] += ks_stat

    for k, v in metrics.items():
        metrics[k] = v / 9  # Average across 9 features

    return metrics


def display_all_metrics(metrics_dict):
    """
    Display metrics for all datasets in a tabular format.

    Parameters:
        metrics_dict (dict): A dictionary containing dataset paths as keys and metrics as values.
    """
    table = []
    for dataset, metrics in metrics_dict.items():
        row = [dataset]
        row.extend(metrics.values())
        table.append(row)

    headers = [
        "Dataset",
        "Wasserstein Distance",
        "Jensen-Shannon Distance",
        "KL Divergence",
        "Kolmogorov-Smirnov Statistic",
    ]
    print(tabulate(table, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    synthetic_data_paths = [
        "data_synthetic/synthcity_pategan_1eps.csv",
        "data_synthetic/synthcity_pategan_5eps.csv",
        "data_synthetic/synthcity_pategan_10eps.csv",
        "data_synthetic/synthcity_dpgan_1eps.csv",
        "data_synthetic/synthcity_dpgan_5eps.csv",
        "data_synthetic/synthcity_dpgan_10eps.csv",
        "data_synthetic/smartnoise_patectgan_1eps.csv",
        "data_synthetic/smartnoise_patectgan_5eps.csv",
        "data_synthetic/smartnoise_patectgan_10eps.csv",
        "data_synthetic/smartnoise_dpctgan_1eps.csv",
        "data_synthetic/smartnoise_dpctgan_5eps.csv",
        "data_synthetic/smartnoise_dpctgan_10eps.csv",
    ]

    training_data, test = data_loader(
        os.path.join("data", "mimic-iii_preprocessed", "pickle_data", "training_data.pkl")
    )

    all_metrics = {}

    for path in synthetic_data_paths:
        synthetic_data, _ = data_loader(path)
        synthetic_data_features = synthetic_data[:, :-1]

        training_data_distributions = calculate_feature_distributions(training_data)
        synthetic_data_distributions = calculate_feature_distributions(synthetic_data)

        metrics = evaluate_synthetic_data(training_data_distributions, synthetic_data_distributions)
        all_metrics[path] = metrics

    display_all_metrics(all_metrics)
