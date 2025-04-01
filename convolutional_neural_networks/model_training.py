# Code partly based on: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

import os
import time
import matplotlib.pyplot as plt
from tempfile import TemporaryDirectory
import torch
import pandas as pd

def train_epoch(model, dataloader, criterion, optimizer, device, phase='train'):

    running_loss = 0.0
    running_corrects = 0

    if phase == 'train':
        model.train()  # Set model to training mode
    else:
        model.eval()   # Set model to evaluate mode

    # Iterate over data.
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(phase == 'train'):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # backward + optimize only if in training phase
            if phase == 'train':
                loss.backward()
                optimizer.step()

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)

    return epoch_loss, epoch_acc


def train_model(model, criterion, optimizer, scheduler, dataloaders, device, num_epochs=25):

    since = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        train_loss_history = []
        train_acc_history = []
        val_loss_history = []
        val_acc_history = []

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'valid']:
                epoch_loss, epoch_acc = train_epoch(
                    model, dataloaders[phase], criterion, optimizer, device, phase
                )

                if phase == 'train':
                    train_loss_history.append(epoch_loss)
                    train_acc_history.append(epoch_acc)
                else:
                    val_loss_history.append(epoch_loss)
                    val_acc_history.append(epoch_acc)

                if phase == 'train':
                    scheduler.step()

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'valid' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path, weights_only=True))

    return model, train_loss_history, train_acc_history, val_loss_history, val_acc_history


def plot_training_history(train_loss, train_acc, val_loss, val_acc, title, folder):

    os.makedirs(folder, exist_ok=True)

    train_acc = [acc.cpu().numpy() for acc in train_acc]
    val_acc = [acc.cpu().numpy() for acc in val_acc]
    epochs = range(1, len(train_loss) + 1)

    loss_data = {
        'Epoch': epochs,
        'Train Loss': train_loss,
        'Validation Loss': val_loss
    }
    acc_data = {
        'Epoch': epochs,
        'Train Accuracy': train_acc,
        'Validation Accuracy': val_acc,
    }

    loss_df = pd.DataFrame(loss_data)
    acc_df = pd.DataFrame(acc_data)
    loss_df.to_csv(os.path.join(folder, 'loss.csv'), index=False)
    acc_df.to_csv(os.path.join(folder, 'acc.csv'), index=False)

    plt.figure(figsize=(12, 5))
    plt.suptitle(title, fontsize=16)

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'bo-', label='Train')
    plt.plot(epochs, val_loss, 'ro-', label='Valid')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, 'bo-', label='Train')
    plt.plot(epochs, val_acc, 'ro-', label='Valid')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig(os.path.join(folder, 'plots.pdf'), format='pdf')
    plt.show()