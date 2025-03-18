import sys
import torch
import torchvision.models as models
import torchvision
import torchvision.transforms as transforms
from download_dataset import get_train_dataset_path, get_test_dataset_path
import numpy as np
import pandas as pd
import os


def embed_data(data_path: str, model) -> tuple[torch.tensor, torch.tensor]:
    '''
    Function for loading data from data_path,
    embedding them using model and returning tensors
    with embeddings and the second one with labels
    '''

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
            batch_size=128,
            shuffle=False
        )

    # embed the data
    embeddings = []
    labels = []
    print(f'Loading from {data_path}')
    for idx, (images, label) in enumerate(data_loader):
        if idx % 25 == 0:
            print(f'=== {idx} / {len(data_loader)} ===')
        with torch.no_grad():
            embedding = model(images)
            embeddings.append(embedding)
        labels.append(label)
    
    # transform list of tensors to tensor, same for labels
    embeddings = [x.squeeze(dim=[2, 3]) for x in embeddings]
    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels)

    return embeddings, labels


def main():
    '''
    The main function of the program
    '''
    if len(sys.argv) < 3:
        raise Exception('Invalid number of parameters, should be 2')
    
    _, out_path, file_prefix = sys.argv
    
    # load model
    resnet18 = models.resnet18(weights='IMAGENET1K_V1')
    backbone = torch.nn.Sequential(*list(resnet18.children())[:-1])

    train_path = get_train_dataset_path()
    test_path = get_test_dataset_path()
    
    for path, name in [(train_path, 'train'), (test_path, 'test')]:
        # specify output file path and skip iterations, if they both exist
        out_embeddings_path = os.path.join(out_path, f'{file_prefix}_{name}_X.csv')
        out_labels_path = os.path.join(out_path, f'{file_prefix}_{name}_y.csv')
        if os.path.isfile(out_embeddings_path) and os.path.isfile(out_labels_path):
            print(f'Files {out_embeddings_path} and {out_labels_path} exist, skipping')
            continue

        embeddings, labels = embed_data(path, backbone)

        # convert tensors to DataFrames and rename columns
        embeddings_df = pd.DataFrame(embeddings.numpy())
        embeddings_df.columns = [f'x_{x}' for x in embeddings_df.columns]
        labels_df = pd.DataFrame(labels.numpy())
        labels_df.columns = ['label']

        # save DataFrames to csv
        print(f'Saving {name}...')
        embeddings_df.to_csv(out_embeddings_path, index=False)
        labels_df.to_csv(out_labels_path, index=False)


if __name__ == '__main__':
    '''
    Arguments:
    out_path: str - specifies the directory to which embeddings will be saved
    file_prefix: str - specifies the prefix of the generated embedding csv files

    It loads the model, extracts backbone from it,
    downloads and load the CINIC10 dataset, embeds it and saves it to
    separate csv files
    '''

    main()