import numpy as np


def data_loader(data_path, label_path=None):
    data_type = data_path.split(".")[1]
    if data_type == "pkl":
        data = np.load(data_path, allow_pickle=True)
    elif data_type == "csv":
        data = np.loadtxt(data_path, delimiter=",", skiprows=1)

    if label_path is None:
        labels = []
    else:
        labels = np.load(label_path, allow_pickle=True)
        labels = labels.reshape(-1, 1)

    return data, labels
