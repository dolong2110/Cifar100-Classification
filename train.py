import torch
import torchmetrics
from tqdm import tqdm

from utils import devices
from dataset.dataloader import Dataset
from model import ImageClassificationModel


class EvaluateClassifier:
    def __init__(self,
                 model_type: str,
                 device = None,
                 image_size: tuple = (640, 640)):
        super().__init__()
        self.model_type = model_type
        self.device = torch.device(device)
        self.image_size = image_size
        self._load_pytorch_lightning_model(model_type)

    def _load_pytorch_lightning_model(self, model_type):
        self.model = ImageClassificationModel.load_from_checkpoint(
                                                checkpoint_path=model_type,
                                                map_location=self.device)
        self.model.to(self.device)

    def forward(self, image):
        with torch.no_grad():
            predicted_label = self.model(image)
            return predicted_label

def main(args):
    device = devices.get_default_device()
    model = EvaluateClassifier(model_type=args.model,
                               device=device,
                               image_size=args.image_size)

    model.eval()
    data = Dataset(args.image_size)
    cifar100_training_data = data.get_train_data(augmentation=True)
    # cifar100_test_data = data.get_test_data()
    f1_metrics = torchmetrics.F1()

    for data in tqdm(cifar100_training_data):
        image, label = data
        image = image.to(device)
        predicted_label = model.forward(image)
        _ = f1_metrics(predicted_label, label)

    avg_f1 = f1_metrics.compute()
    print("F1 Score:", avg_f1)

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
                        help="num workers")
    parser.add_argument("--image_size", type=tuple, default=(640, 640),
                        help="image size")
    arguments = parser.parse_args()
    main(arguments)