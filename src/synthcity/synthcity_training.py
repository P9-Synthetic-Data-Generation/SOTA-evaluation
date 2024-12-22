import os
import sys

import numpy as np
import pandas as pd

from synthcity.plugins import Plugins

sys.path.append("src")
from utils.data_handling import data_loader


def train_and_generate_synthetic_data(eps_values, data, models):
    for eps in eps_values:
        for name in models:
            print(f"Training {name} with {eps}...")
            model = Plugins().get(
                name,
                epsilon=eps,
            )
            model.fit(data)

            synth_data = model.generate(len(data))
            df = pd.DataFrame(synth_data.data)

            os.makedirs("data_synthetic", exist_ok=True)
            save_path = os.path.join("data_synthetic", f"synthcity_{name}_{eps}eps.csv")
            df.to_csv(save_path, index=False)
            print(f"Finished training of {name} with {eps}. Data saved to {save_path}")


if __name__ == "__main__":
    features, labels = data_loader(
        os.path.join("data", "mimic-iii_preprocessed", "pickle_data_new", "training_data.pkl"),
        os.path.join("data", "mimic-iii_preprocessed", "pickle_data_new", "training_labels.pkl"),
    )

    training_data = pd.DataFrame(np.hstack((features, labels)))

    train_and_generate_synthetic_data(eps_values=[1, 5, 10], data=training_data, models=["pategan"])
