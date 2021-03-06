import torch

from utils import devices
from dataset.dataloader import Dataset
import models
from training import learning
from configs import global_configs
from plots import graph

def main(args):
    device = devices.get_default_device()
    if torch.device == 'cuda':
        return
    model = models.get_model(args.model)
    model = devices.to_device(model, device)

    data = Dataset(args.image_size)
    cifar100_training_data = data.get_train_data(augmentation=args.augmentation)
    cifar100_test_data = data.get_test_data()
    train_dl = devices.DeviceDataLoader(cifar100_training_data, device)
    valid_dl = devices.DeviceDataLoader(cifar100_test_data, device)

    history = [learning.evaluate(model, valid_dl)]
    history += learning.fit_one_cycle(global_configs.EPOCH, global_configs.MAX_LR, model, args.model,
                                      train_dl, valid_dl, weight_decay=global_configs.WEIGHT_DECAY,
                                      grad_clip=global_configs.GRAD_CLIP, opt_func=torch.optim.Adam)

    # graph.plot_accuracies(history)
    # graph.plot_losses(history)
    # graph.plot_lrs(history)
    torch.save(history, 'cifar100.th')

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
                        help="num workers")  # number of threads
    parser.add_argument("--image_size", type=tuple, default=32,
                        help="image size")
    parser.add_argument("--augmentation", type=bool, default=False,
                        help="augmentation")
    arguments = parser.parse_args()
    main(arguments)