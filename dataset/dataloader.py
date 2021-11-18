from torch.utils.data import DataLoader
import torch.utils.data.dataset as DataSet
from torchvision import datasets

from configs import train_data_configs, test_data_configs, global_configs
from augmentation import augment_cifar100_train, augment_cifar100_basic


class Dataset:
    def __init__(self, image_resolution=global_configs.IMAGE_RESOLUTION):
        self.image_resolution = image_resolution
    
    @staticmethod
    def load_data(data_set: DataSet, batch_size, num_workers, shuffle):
        return DataLoader(data_set, batch_size, num_workers, shuffle)

    def get_train_data(self, augmentation=False, mean=train_data_configs.MEAN, std=train_data_configs.STD):
        transformer = augment_cifar100_basic(mean, std)
        if augmentation:
            transformer = augment_cifar100_train(self.image_resolution, mean, std)
        data_set = datasets.CIFAR100(root='./data', train=True, download=True,
                                     transform=transformer)

        return self.load_data(data_set, train_data_configs.BATCH_SIZE,
                              train_data_configs.NUM_WORKERS, train_data_configs.SHUFFLE)

    def get_test_data(self, mean=test_data_configs.MEAN, std=test_data_configs.STD):
        transformer = augment_cifar100_basic(mean, std)
        data_set = datasets.CIFAR100(root='./data', train=False, download=True,
                                     transform=transformer)
        return self.load_data(data_set, test_data_configs.BATCH_SIZE,
                              test_data_configs.NUM_WORKERS, test_data_configs.SHUFFLE)