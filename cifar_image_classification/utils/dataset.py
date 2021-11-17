from typing import Tuple

import numpy as np

def get_mean_std(dataset) -> (Tuple, Tuple):
    data_r = np.dstack([dataset[i][1][:, :, 0] for i in range(len(dataset))])
    data_g = np.dstack([dataset[i][1][:, :, 1] for i in range(len(dataset))])
    data_b = np.dstack([dataset[i][1][:, :, 2] for i in range(len(dataset))])
    mean = (np.mean(data_r), np.mean(data_g), np.mean(data_b))
    std = (np.std(data_r), np.std(data_g), np.std(data_b))

    return mean, std