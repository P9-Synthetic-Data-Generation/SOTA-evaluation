import os
import sys

import numpy as np
import pandas as pd

from synthcity.plugins import Plugins
from synthcity.utils.serialization import load_from_file, save_to_file

sys.path.append("src")
from utils.data_handling import data_loader


def train(eps_values, data, models):
    for eps in eps_values:
        for name in models:
            print(f"Training {name} with {eps}...")
            model = Plugins().get(
                name,
                epsilon=eps,
            )
            model.fit(data)

            os.makedirs("models", exist_ok=True)
            save_path = os.path.join("models", f"synthcity_pategan_{eps}eps.pkl")
            save_to_file(save_path, model)
            print(f"Finished training of {name} with {eps}. Model saved to {save_path}")


def generate_synthetic_data(model_path, count):
    model = load_from_file(model_path)
    data = model.generate(count)
    df = pd.DataFrame(data.data)

    os.makedirs("data_synthetic", exist_ok=True)
    filename = model_path.split(".")[0]

    save_path = os.path.join("data_synthetic", f"{filename}.csv")
    df.to_csv(save_path, index=False)
    print(f"Finished generating synthetic data with {filename}. Synthetic data saved to {save_path}")


if __name__ == "__main__":
    features, labels = data_loader(
        os.path.join("data", "mimic-iii_preprocessed", "pickle_data", "training_data.pkl"),
        os.path.join("data", "mimic-iii_preprocessed", "pickle_data", "training_labels.pkl"),
    )

    training_data = pd.DataFrame(np.hstack((features, labels)))

    train(eps_values=[1, 5, 10], data=training_data, models=["pategan"])

    generate_synthetic_data(
        model_path=os.path.join("data", "models", "synthcity_pategan_1eps.pkl"), count=len(features)
    )
    generate_synthetic_data(
        model_path=os.path.join("data", "models", "synthcity_pategan_5eps.pkl"), count=len(features)
    )
    generate_synthetic_data(
        model_path=os.path.join("data", "models", "synthcity_pategan_10eps.pkl"), count=len(features)
    )
