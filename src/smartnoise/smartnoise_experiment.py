import os
import sys

import numpy as np
import pandas as pd
from snsynth import Synthesizer
from snsynth.pytorch import PytorchDPSynthesizer
from snsynth.pytorch.nn import DPCTGAN, PATEGAN

sys.path.append("src")
from utils.data_handling import data_loader


def generate_synthetic_data(data, gans):
    for name, model in gans.items():
        synth = PytorchDPSynthesizer(epsilon=1.0, gan=model, preprocessor=None)
        synth.fit(data, preprocessor_eps=0.1)

        synthetic_data = synth.sample(8228)
        synthetic_data.to_csv(os.path.join("data", "synthetic_data", f"smartnoise_{name}"), index=False)


if __name__ == "__main__":
    data, labels = data_loader(
        os.path.join("data", "mimic-iii_preprocessed", "pickle_data", "train_data.pkl"),
        os.path.join("data", "mimic-iii_preprocessed", "pickle_data", "train_labels.pkl"),
    )
    data_with_labels = np.concatenate((data, labels), axis=1)
    df_data = pd.DataFrame(data_with_labels)

    gans = {"pategan": PATEGAN(epsilon=1.0), "dpctgan": DPCTGAN()}
    generate_synthetic_data(df_data, gans)
