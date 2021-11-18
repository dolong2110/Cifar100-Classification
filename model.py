import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS

from models import get_model

class ImageClassificationModel(pl.LightningModule):
    def __init__(self, model_name: str = None):
        super().__init__()
        self.model = get_model(model_name)

    def forward(self, images):
        predicted_label = self.model(images)
        return predicted_label

    def training_step(self, train_batch):
        images, labels = train_batch
        predicted_label = self.forward(images)  # Generate predictions
        loss = F.cross_entropy(predicted_label, labels)  # Calculate loss
        return loss

    def validation_step(self, val_batch):
        images, labels = val_batch
        predicted_label = self.forward(images)  # Generate predictions
        loss = F.cross_entropy(predicted_label, labels)  # Calculate loss
        acc = accuracy(predicted_label, labels)  # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

def accuracy(outputs, labels):
    _, predicts = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(predicts == labels).item() / len(predicts))