import os
import sys

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import chisquare, entropy, ks_2samp, wasserstein_distance
from tabulate import tabulate

sys.path.append("src")
from utils.data_handling import data_loader


def evaluate_synthetic_data(real_data, synthetic_data, bins=50):
    """
    Evaluate the quality of synthetic data compared to real data using various metrics.

    Parameters:
        real_data (array-like): The real dataset.
        synthetic_data (array-like): The synthetic dataset.
        bins (int): Number of bins to use for histogram-based comparisons.

    Returns:
        dict: A dictionary containing the computed metrics.
    """
    real_data = np.ravel(real_data)
    synthetic_data = np.ravel(synthetic_data)
    metrics = {}

    metrics["wasserstein_distance"] = wasserstein_distance(real_data, synthetic_data)

    bin_edges = np.histogram_bin_edges(real_data, bins=bins)
    real_hist, _ = np.histogram(real_data, bins=bin_edges, density=True)
    synthetic_hist, _ = np.histogram(synthetic_data, bins=bin_edges, density=True)

    # Add small value to avoid log(0)
    real_hist += 1e-8
    synthetic_hist += 1e-8

    metrics["jensen_shannon_distance"] = jensenshannon(real_hist, synthetic_hist)
    metrics["kl_divergence"] = entropy(real_hist, synthetic_hist)
    metrics["inverse_kl_divergence"] = entropy(synthetic_hist, real_hist)
    chi2_stat, _ = chisquare(synthetic_hist, f_exp=real_hist)
    metrics["chi_squared_statistic"] = chi2_stat
    ks_stat, _ = ks_2samp(real_data, synthetic_data)
    metrics["kolmogorov_smirnov_statistic"] = ks_stat

    return metrics


def display_metrics(metrics):
    """
    Display the metrics in a tabular format using tabulate.

    Parameters:
        metrics (dict): A dictionary of metrics to display.
    """
    table = [[key, value] for key, value in metrics.items()]
    print(tabulate(table, headers=["Metric", "Value"], tablefmt="grid"))


if __name__ == "__main__":
    data, _ = data_loader(os.path.join("data", "mimic-iii_preprocessed", "pickle_data", "original_data.pkl"))

    pategan_data, _ = data_loader(os.path.join("data", "synthetic_data", "synthcity_pategan.csv"))
    pategan_data = pategan_data[:, :-1]

    metrics = evaluate_synthetic_data(data, pategan_data)
    display_metrics(metrics)
