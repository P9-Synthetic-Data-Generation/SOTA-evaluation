"""
Install Synthcity before running code: 'pip install synthcity'
"""

import os
import sys

import numpy as np
import pandas as pd

from synthcity.plugins import Plugins
from synthcity.utils.serialization import save_to_file

sys.path.append("src")
from utils.data_handling import data_loader


def load_mimic_data():
    data = np.load(os.path.join("data", "mimic-iii_preprocessed", "pickle_data", "train_data.pkl"), allow_pickle=True)
    labels = np.load(os.path.join("data", "mimic-iii_preprocessed", "pickle_data", "train_labels.pkl"), allow_pickle=True)

    data_reshaped = data.reshape(len(data), -1)
    labels_reshaped = labels.reshape(-1, 1)
    data_with_labels = np.concatenate((data_reshaped, labels_reshaped), axis=1)
    df_data = pd.DataFrame(data_with_labels)

    return df_data


if __name__ == "__main__":
    data, labels = data_loader(
        os.path.join("data", "mimic-iii_preprocessed", "pickle_data", "original_data.pkl"),
        os.path.join("data", "mimic-iii_preprocessed", "pickle_data", "original_labels.pkl"),
    )

    print(data.shape)
    print(labels.shape)

    data_with_labels = np.concatenate((data, labels), axis=1)
    df_data = pd.DataFrame(data_with_labels)

    epsilons = [1, 5, 10]
    for eps in epsilons:
        print(f"Started training pategan with eps {eps}.")
        syn_model = Plugins().get(
            "pategan",
            epsilon=eps,
        )
        syn_model.fit(df_data)

        os.makedirs("models", exist_ok=True)
        save_to_file(os.path.join("models", f"synthcity_pategan_eps{eps}.pkl"), syn_model)
