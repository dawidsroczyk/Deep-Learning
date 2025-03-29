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
import torch.nn.functional as F
import sys
import argparse
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


class FewShotEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten()
        )
    
    def forward(self, x):
        x = self.blocks(x)
        return x


from torch.utils.data import DataLoader, Subset, RandomSampler

def create_class_dataloaders(data_path, batch_size, num_workers=2):
    full_dataset = datasets.ImageFolder(data_path)
    class_names = full_dataset.classes
    
    cinic_mean = [0.47889522, 0.47227842, 0.43047404]
    cinic_std = [0.24205776, 0.23828046, 0.25874835]
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cinic_mean, std=cinic_std)
    ])
    
    dataset = datasets.ImageFolder(data_path, transform=transform)
    
    class_indices = {i: [] for i in range(len(class_names))}
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)
    
    class_loaders = {}
    for class_idx, indices in class_indices.items():
        subset = Subset(dataset, indices)
        sampler = RandomSampler(subset, replacement=True)
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            shuffle=False
        )
        class_loaders[class_idx] = loader
    
    return class_loaders, class_names

def episode(embedder,
            criterion,
            support_class_loaders, 
            support_class_names, 
            query_class_loaders, 
            query_class_names,
            C,
            ):
    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
    S_means = []
    Q = []

    for idx, c in enumerate(C):
        scl = support_class_loaders[support_class_names.index(c)]
        qcl = query_class_loaders[query_class_names.index(c)]

        S_c = next(iter(scl))
        (S_images, _) = S_c
        Q_c = next(iter(qcl))
        (Q_images, _) = Q_c

        S_embeddings = embedder(S_images)
        S_embeddings_mean = torch.mean(S_embeddings, axis=0)
        S_means.append(S_embeddings_mean)

        Q_embeddings = embedder(Q_images)
        Q.append((idx, Q_embeddings))

    S_means = torch.vstack(S_means)

    loss = 0
    for idx, Q_embeds in Q:
        cos_sims = cos(Q_embeds.unsqueeze(1), S_means.unsqueeze(0))
        probs = F.softmax(cos_sims, dim=1)
        log_probs = torch.log(probs)
        targets = torch.full((probs.shape[0],), idx)
        loss += criterion(log_probs, targets)
    
    return loss / (len(C) * (support_class_loaders.batch_size + query_class_loaders.batch_size))

def train(embedder: FewShotEmbedder,
          train_classes,
          test_classes,
          M,
          S_num,
          Q_num,
          num_epochs,
          train_episodes,
          test_episodes):
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(embedder.parameters(), lr=1e-4, weight_decay=1e-6)
    support_class_loaders, support_class_names = create_class_dataloaders(get_train_dataset_path(), S_num)
    query_class_loaders, query_class_names = create_class_dataloaders(get_train_dataset_path(), Q_num)
    for epoch in range(num_epochs):
        total_loss = 0.0
        for episode_num in range(train_episodes):
            C = np.random.choice(train_classes, M)
            optimizer.zero_grad()
            loss = episode(embedder, criterion, support_class_loaders, support_class_names,
                           query_class_loaders, query_class_names, C)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            total_test_loss = 0.0
            for episode_num in range(test_episodes):
                C = np.random.choice(test_classes, M)
                loss = episode(embedder, criterion, support_class_loaders, support_class_names,
                               query_class_loaders, query_class_names, C)
                total_test_loss += loss.item()
        
        train_loss = total_loss / train_episodes
        test_loss = test_loss / test_episodes

        print(train_loss, test_loss)
            