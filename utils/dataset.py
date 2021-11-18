from typing import Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from configs import train_data_configs

def get_mean_std(dataset) -> (Tuple, Tuple):
    """compute the mean and std of  dataset
        Args:
            for example: cifar100_training_dataset or cifar100_test_dataset
            which derived from class torch.utils.data
        Returns:
            a tuple contains mean, std value of entire dataset
    """

    data_r = np.dstack([dataset[i][1][:, :, 0] for i in range(len(dataset))])
    data_g = np.dstack([dataset[i][1][:, :, 1] for i in range(len(dataset))])
    data_b = np.dstack([dataset[i][1][:, :, 2] for i in range(len(dataset))])
    mean = (np.mean(data_r), np.mean(data_g), np.mean(data_b))
    std = (np.std(data_r), np.std(data_g), np.std(data_b))

    return mean, std

def denormalize(images, mean, std):
    means = torch.tensor(mean).reshape(1, 3, 1, 1)
    stds = torch.tensor(std).reshape(1, 3, 1, 1)
    return images * stds + means

def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_xticks([])
        ax.set_yticks([])
        denormalized_images = denormalize(images, train_data_configs.MEAN, train_data_configs.STD)
        ax.imshow(make_grid(denormalized_images[:64], nrow=8).permute(1, 2, 0).clamp(0, 1))
        break