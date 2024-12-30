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


if __name__ == "__main__":
    synthetic_data_paths = [
        "data_synthetic/synthcity_pategan_1eps.csv",
        "data_synthetic/synthcity_pategan_5eps.csv",
        "data_synthetic/synthcity_pategan_10eps.csv",
        "data_synthetic/synthcity_dpgan_1eps.csv",
        "data_synthetic/synthcity_dpgan_5eps.csv",
        "data_synthetic/synthcity_dpgan_10eps.csv",
        "data_synthetic/smartnoise_patectgan_1eps.csv",
        "data_synthetic/smartnoise_patectgan_5eps.csv",
        "data_synthetic/smartnoise_patectgan_10eps.csv",
        "data_synthetic/smartnoise_dpctgan_1eps.csv",
        "data_synthetic/smartnoise_dpctgan_5eps.csv",
        "data_synthetic/smartnoise_dpctgan_10eps.csv",
    ]

    for path in synthetic_data_paths:
        data, _ = data_loader(path)
        labels = data[:, -1]
        print(path, sum(labels))
