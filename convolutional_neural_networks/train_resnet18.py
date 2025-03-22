import sys
import torch
import torchvision.models as models
import torchvision
import torchvision.transforms as transforms
from download_dataset import get_train_dataset_path, get_test_dataset_path
import numpy as np
import pandas as pd
import os
import torch.optim as optim
import torch.nn as nn
import sys
import argparse

def create_data_loader(data_path, batch_size):
    # define mean and std for normalizing the dataset
    cinic_mean = [0.47889522, 0.47227842, 0.43047404]
    cinic_std = [0.24205776, 0.23828046, 0.25874835]

    # define standard data loaders
    data_loader = torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder(data_path,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=cinic_mean, std=cinic_std)
                ])),
            batch_size=batch_size,
            shuffle=True
        )
    
    return data_loader


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--output_folder', type=str, default='output')
    parser.add_argument('--weight_decay', type=float, default=0.0)

    args = parser.parse_args()
    return args


def train_resnet18(num_epochs, learning_rate, batch_size, output_folder, weight_decay):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_path = get_train_dataset_path()
    test_path = get_test_dataset_path()
    train_dl = create_data_loader(train_path, batch_size)
    test_dl = create_data_loader(test_path, batch_size)
    resnet18 = models.resnet18(weights=None).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(resnet18.parameters(), lr=learning_rate, weight_decay=weight_decay)

    metric_data = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        total_train_loss = 0.0

        for i, data in enumerate(train_dl, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = resnet18(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total_train_loss += loss.item()

            if i % 25 == 24:
                # print(f'[{epoch + 1}, {i + 1:5d} / {len(train_dl)}] loss: {running_loss / 25:.3f}')
                running_loss = 0.0

        avg_train_loss = total_train_loss / len(train_dl)
        print(f'Epoch {epoch + 1}, Training Loss: {avg_train_loss:.3f}')

        total_test_loss = 0.0
        with torch.no_grad():
            for data in test_dl:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = resnet18(inputs)
                loss = criterion(outputs, labels)
                total_test_loss += loss.item()

        avg_test_loss = total_test_loss / len(test_dl)
        print(f'Epoch {epoch + 1}, Test Loss: {avg_test_loss:.3f}')

        metrics = {}
        metrics['epoch'] = epoch
        metrics['train_loss'] = avg_train_loss
        metrics['test_loss'] = avg_test_loss
        metric_data.append(metrics)

    out_file_path = os.path.join(output_folder, f'resnet18_e{epoch}_lr{learning_rate}_bs{batch_size}_wd{weight_decay}.csv')
    df_metric_data = pd.DataFrame(metric_data)
    df_metric_data.to_csv(out_file_path, index=False)
    print('Finished Training')


def main():
    args = parse_arguments()
    num_epochs = args.epochs
    learning_rate = args.lr
    batch_size = args.batch_size
    output_folder = args.output_folder
    weight_decay = args.weight_decay
    train_resnet18(num_epochs, learning_rate, batch_size, output_folder, weight_decay)


if __name__ == '__main__':
    main()