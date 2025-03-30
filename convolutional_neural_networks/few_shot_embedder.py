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
from torch.utils.data import DataLoader, Subset, RandomSampler


class FewShotEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.3),  # Added dropout
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.3),  # Added dropout
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.3),  # Added dropout
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.3),  # Added dropout
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Flatten()
        )
    
    def forward(self, x):
        x = self.blocks(x)
        return x

def create_class_dataloaders(data_path, batch_size, num_workers=2):
    full_dataset = datasets.ImageFolder(data_path)
    class_names = full_dataset.classes
    cinic_mean = [0.47889522, 0.47227842, 0.43047404]
    cinic_std = [0.24205776, 0.23828046, 0.25874835]
    

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=cinic_mean, std=cinic_std)
    ])
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
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
        # Reduce num_workers if issues persist
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            sampler=RandomSampler(subset, replacement=True),
            num_workers=1,
            shuffle=False,
            persistent_workers=False  # Add this line
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
        # Support phase
        scl = support_class_loaders[support_class_names.index(c)]
        S_images, _ = next(iter(scl))
        S_embeddings = embedder(S_images)
        S_means.append(S_embeddings.mean(dim=0))  # Mean across support samples

        # Query phase
        qcl = query_class_loaders[query_class_names.index(c)]
        Q_images, _ = next(iter(qcl))
        Q_embeddings = embedder(Q_images)
        Q.append((idx, Q_embeddings))

    S_means = torch.stack(S_means)  # [M, embedding_dim]

    loss = 0
    total_samples = 0
    for idx, Q_embeds in Q:
        # Calculate similarities [query_batch_size, M]
        cos_sims = cos(Q_embeds.unsqueeze(1), S_means.unsqueeze(0))
        
        # Calculate loss
        log_probs = F.log_softmax(cos_sims, dim=1)
        targets = torch.full((Q_embeds.shape[0],), idx, device=Q_embeds.device)
        loss += criterion(log_probs, targets) * Q_embeds.shape[0]  # Weight by number of queries
        total_samples += Q_embeds.shape[0]
    
    return loss / total_samples  # Normalize by total query samples

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
    # optimizer = optim.Adam(embedder.parameters(), lr=1e-4, weight_decay=1e-6)
    optimizer = optim.AdamW(embedder.parameters(), lr=1e-4, weight_decay=1e-4)  # Changed to AdamW with stronger weight decay
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=5, factor=0.5, verbose=True
    )

    # Create loaders with explicit batch sizes
    support_class_loaders, support_class_names = create_class_dataloaders(get_train_dataset_path(), S_num)
    query_class_loaders, query_class_names = create_class_dataloaders(get_train_dataset_path(), Q_num)
    
    for epoch in range(num_epochs):
        # Training phase
        embedder.train()
        total_loss = 0.0
        for _ in range(train_episodes):
            C = np.random.choice(train_classes, M, replace=False)
            optimizer.zero_grad()
            loss = episode(embedder, criterion, 
                          support_class_loaders, support_class_names,
                          query_class_loaders, query_class_names,
                          C)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Evaluation phase
        embedder.eval()
        total_test_loss = 0.0
        with torch.no_grad():
            for _ in range(test_episodes):
                C = np.random.choice(test_classes, M, replace=False)
                loss = episode(embedder, criterion,
                              support_class_loaders, support_class_names,
                              query_class_loaders, query_class_names,
                              C)
                total_test_loss += loss.item()
        
        # Logging
        avg_train_loss = total_loss / train_episodes
        avg_test_loss = total_test_loss / test_episodes
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Test Loss: {avg_test_loss:.4f}")