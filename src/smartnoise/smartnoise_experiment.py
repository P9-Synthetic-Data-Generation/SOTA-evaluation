import os
import sys

import numpy as np
import pandas as pd
from snsynth import Synthesizer
from snsynth.pytorch import PytorchDPSynthesizer
from snsynth.pytorch.nn import DPCTGAN, DPGAN, PATECTGAN, PATEGAN

sys.path.append("src")
from utils.data_handling import data_loader


def generate_synthetic_data(data, gans):
    for name, model in gans.items():
        synth = PytorchDPSynthesizer(epsilon=1.0, gan=model, preprocessor=None)
        synth.fit(data, preprocessor_eps=0.1)

        synthetic_data = synth.sample(8228)
        synthetic_data.to_csv(os.path.join("data", "synthetic_data", f"smartnoise_{name}.csv"), index=False)


if __name__ == "__main__":
    features, labels = data_loader(
        os.path.join("data", "mimic-iii_preprocessed", "pickle_data", "original_data.pkl"),
        os.path.join("data", "mimic-iii_preprocessed", "pickle_data", "original_labels.pkl"),
    )
    data = np.concatenate((features, labels), axis=1)
    data = pd.DataFrame(data)

    gans = {
        "dpctgan": DPCTGAN(),
        "pategan": PATEGAN(epsilon=1.0),
        "patectgan": PATECTGAN(regularization="dragan"),
    }

    generate_synthetic_data(data, gans)
