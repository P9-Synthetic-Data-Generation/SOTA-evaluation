import numpy as np
import pandas as pd
import os
from synthcity.plugins import Plugins


def load_mimic_data():
    data = np.load(r"C:\Users\mikke\Documents\GitHub\SOTA-evaluation\data\mimic-iii_preprocessed\pickle_data\data.pkl", allow_pickle=True)
    labels = np.load(r"C:\Users\mikke\Documents\GitHub\SOTA-evaluation\data\mimic-iii_preprocessed\pickle_data\labels.pkl", allow_pickle=True)

    data_reshaped = data.reshape(8228, -1)
    labels_reshaped = labels.reshape(-1, 1)
    data_with_labels = np.concatenate((data_reshaped, labels_reshaped), axis=1)
    df_data = pd.DataFrame(data_with_labels)

    return df_data

if __name__ == "__main__":
    data = load_mimic_data()
    print("data shape: ", data.shape)
    syn_model = Plugins().get('pategan')

    syn_model.fit(data)

    print(syn_model.generate(count = 10))


