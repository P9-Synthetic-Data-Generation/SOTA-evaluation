import os
import sys

import numpy as np
import pandas as pd
from snsynth import Synthesizer

sys.path.append("src")
from utils.data_handling import data_loader


def generate_synthetic_data(eps_values, data, models):
    for eps in eps_values:
        for name in models:
            synthesizer = Synthesizer.create(name, epsilon=10 + eps)

            print(f"Training {name} with {eps}...")
            synthesizer.fit(data, preprocessor_eps=10)

            synthetic_data = synthesizer.sample(8228)

            os.makedirs("data_synthetic", exist_ok=True)
            save_path = os.path.join("data_synthetic", f"smartnoise_{name}_{eps}eps.csv")
            synthetic_data.to_csv(save_path, index=False)

            print(f"Finished training of {name} with {eps}. Synthetic data saved to {save_path}")


if __name__ == "__main__":
    features, labels = data_loader(
        os.path.join("data", "mimic-iii_preprocessed", "pickle_data", "original_data.pkl"),
        os.path.join("data", "mimic-iii_preprocessed", "pickle_data", "original_labels.pkl"),
    )
    data = pd.DataFrame(np.hstack((features, labels)))

    generate_synthetic_data(
        eps_values=[1, 5, 10],
        data=data,
        models=[
            "dpctgan",
            "patectgan",
        ],
    )
