import os
import sys

import numpy as np
import pandas as pd
from snsynth import Synthesizer

sys.path.append("src")
from utils.data_handling import data_loader


def train_and_generate_synthetic_data(eps_values, data, models):
    for name in models:
        for eps in eps_values:
            synthesizer = Synthesizer.create(name, epsilon=5 + eps)

            print(f"Training {name} with {eps}...")
            synthesizer.fit(data, preprocessor_eps=5)

            synthetic_data = synthesizer.sample(len(data))

            os.makedirs("data_synthetic", exist_ok=True)
            save_path = os.path.join("data_synthetic", f"smartnoise_{name}_{eps}eps.csv")
            synthetic_data.to_csv(save_path, index=False)

            print(f"Finished training of {name} with {eps}. Synthetic data saved to {save_path}")


if __name__ == "__main__":
    features, labels = data_loader(
        os.path.join("data", "mimic-iii_preprocessed", "pickle_data", "training_data.pkl"),
        os.path.join("data", "mimic-iii_preprocessed", "pickle_data", "training_labels.pkl"),
    )
    training_data = pd.DataFrame(np.hstack((features, labels)))

    train_and_generate_synthetic_data(
        eps_values=[1, 5, 10],
        data=training_data,
        models=[
            "patectgan",
        ],
    )
