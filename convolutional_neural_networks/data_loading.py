# Code partly based on: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

import torch
import os
from torchvision import datasets, models, transforms

class Data:

    def __init__(self, data_dir, train_transform = None, valid_transform = None, test_transform = None, batch_size = 32, shuffle = True, num_workers = 4):
        self.load_datasets(data_dir, train_transform, valid_transform, test_transform, batch_size, shuffle, num_workers)

    def default_train_transform(self, cinic_mean = [0.47889522, 0.47227842, 0.43047404], cinic_std = [0.24205776, 0.23828046, 0.25874835]):
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cinic_mean, cinic_std)
        ])

    def default_valid_transform(self, cinic_mean = [0.47889522, 0.47227842, 0.43047404], cinic_std = [0.24205776, 0.23828046, 0.25874835]):
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(cinic_mean, cinic_std)
        ])

    def default_test_transform(self, cinic_mean = [0.47889522, 0.47227842, 0.43047404], cinic_std = [0.24205776, 0.23828046, 0.25874835]):
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(cinic_mean, cinic_std)
        ])

    def load_datasets(self, data_dir, train_transform = None, valid_transform = None, test_transform = None, batch_size = 4, shuffle = True, num_workers = 4):

        data_transforms = {
            'train': train_transform if train_transform else self.default_train_transform(),
            'valid': valid_transform if valid_transform else self.default_valid_transform(),
            'test': test_transform if test_transform else self.default_test_transform()
        }

        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['train', 'valid', 'test']}
        self.dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                    shuffle=shuffle, num_workers=num_workers)
                    for x in ['train', 'valid', 'test']}

        self.dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}
        self.class_names = image_datasets['train'].classes