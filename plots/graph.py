import matplotlib.pyplot as plt
import numpy as np

def plot_accuracies(history):
    """
    Plotted the accuracy Graph
    """

    accuracies = [train_epoch['val_acc'] for train_epoch in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs')

def plot_losses(history):
    """
    Training and Validation loss graph
    """

    train_losses = [train_epoch.get('train_loss') for train_epoch in history]
    val_losses = [train_epoch['val_loss'] for train_epoch in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')

def plot_lrs(history):
    """
    Learning Rate Graph
    """

    lrs = np.concatenate([x.get('lrs', []) for x in history])
    plt.plot(lrs)
    plt.xlabel('Batch no.')
    plt.ylabel('Learning rate')
    plt.title('Learning Rate vs. Batch no.')