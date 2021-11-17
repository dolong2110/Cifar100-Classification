from utils import devices
from dataset import dataloader

import torchmetrics


class EvaluateClassifier:
    def __init__(self,
                 model_path: str,
                 device = None,
                 image_size: tuple = (640, 640)):
        super().__init__()
        self.model_path = model_path
        self.device = torch.device(device)
        self.image_size = image_size
        self._load_pytorch_lightning_model(model_path)

    def _load_pytorch_lightning_model(self, model_path):
        self.model = ClassificationModel.load_from_checkpoint(
                                                checkpoint_path=model_path,
                                                map_location=self.device)
        self.model.to(self.device)

    def forward(self, image):
        with torch.no_grad():
            predicted_label = self.model(image)
            return predicted_label

def main():
    device = devices.get_device()
    model = EvaluateClassifier(model_path=args.model_path,
                               device=device,
                               image_size=args.image_size)

    model.eval()
    data = dataloader.Dataset(args.image_size)
    cifar100_training_data = data.get_train_data(augmentation=True)
    cifar100_test_data = data.get_test_data()
    acc_metrics = torchmetrics.Accuracy()
    f1_metrics = torchmetrics.F1()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate model")

    parser.add_argument("--model_path", type=str, default="",
                                help="model_path")
    parser.add_argument("--batch_size", type=int, default=16,
                                help="batch_size")
    parser.add_argument("--csv_dir", type=str, default="",
                                help="csv directory")
    parser.add_argument("--data_dir", type=str, default="",
                                help="data directory")
    parser.add_argument("--num_workers", type=int, default= 2,
                                help="num workers")
    parser.add_argument("--image_size", type=tuple, default=(640,640),
                                help="image size")
    args = parser.parse_args()
    main(args)