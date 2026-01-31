import numpy as np
from pathlib import Path

def load_dataset(data_dir="data/joints_occluded"):
    X, Y, M = [], [], []

    data_dir = Path(data_dir)

    for occluded_file in data_dir.glob("*_occluded.npy"):
        mask_file = data_dir / occluded_file.name.replace("_occluded", "_mask")
        clean_file = Path("data/joints_clean") / occluded_file.name.replace("_occluded", "")

        x = np.load(occluded_file)
        y = np.load(clean_file)
        m = np.load(mask_file)

        X.append(x)
        Y.append(y)
        M.append(m)

    return X, Y, M
