import numpy as np


def data_loader(data_path, label_path):
    data = np.load(data_path, allow_pickle=True)
    data = data.reshape(data.shape[0], -1)
    labels = np.load(label_path, allow_pickle=True)
    labels = labels.reshape(-1, 1)

    return data, labels
