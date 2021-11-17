from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import get_model

class ImageClassificationBase(pl.LightningModule):
    def __init__(self, model_name: str = None):
        super().__init__()
        self.model = get_model(model_name)

    def forward(self, input):
        predicted_label = self.model(input)
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
        acc = accuracy(out, labels)  # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    @staticmethod
    def validation_epoch_end(outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accuracy = [x['val_acc'] for x in outputs]
        epoch_accuracy = torch.stack(batch_accuracy).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_accuracy.item()}

    @staticmethod
    def epoch_end(epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))

def accuracy(outputs, labels):
    _, predicts = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(predicts == labels).item() / len(predicts))