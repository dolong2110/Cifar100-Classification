import torch

from utils import devices
from dataset.dataloader import Dataset
import models
from training import learning
from configs import global_configs

def main(args):
    device = devices.get_default_device()
    if torch.device == 'cpu':
        return
    model = models.get_model(args.model)

    data = Dataset(args.image_size)
    cifar100_training_data = data.get_train_data(augmentation=True)
    cifar100_test_data = data.get_test_data()
    train_dl = devices.DeviceDataLoader(cifar100_training_data, device)
    valid_dl = devices.DeviceDataLoader(cifar100_test_data, device)

    history = [learning.evaluate(model, valid_dl)]
    history += learning.fit_one_cycle(global_configs.EPOCH, global_configs.MAX_LR, model, args.model,
                                      train_dl, valid_dl, grad_clip=global_configs.GRAD_CLIP,
                                      weight_decay=global_configs.WEIGHT_DECAY, opt_func=torch.optim.Adam)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate model")

    parser.add_argument("--model", type=str, default="",
                        help="model")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="batch_size")
    parser.add_argument("--csv_dir", type=str, default="",
                        help="csv directory")
    parser.add_argument("--data_dir", type=str, default="",
                        help="data directory")
    parser.add_argument("--num_workers", type=int, default= 2,
                        help="num workers") # number of thread
    parser.add_argument("--image_size", type=tuple, default=(640, 640),
                        help="image size")
    arguments = parser.parse_args()
    main(arguments)