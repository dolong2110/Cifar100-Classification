import torch
import torch.nn as nn
import torch.nn.functional as F

def measure_accuracy(predicted, labels):
    _, predicted_labels = torch.max(predicted, dim=1)
    return torch.tensor(torch.sum(predicted_labels == labels).item() / len(predicted_labels))


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        predicted = self(images)  # Generate predictions
        loss = F.cross_entropy(predicted, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        predicted = self(images)  # Generate predictions
        loss = F.cross_entropy(predicted, labels)  # Calculate loss
        accuracy = measure_accuracy(predicted, labels)  # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': accuracy}

    @staticmethod
    def validation_epoch_end(outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    @staticmethod
    def epoch_end(epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))