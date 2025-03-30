import sys
import torch
import torchvision.models as models
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import os
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torch.utils.data import DataLoader, Subset, RandomSampler
from download_dataset import get_train_dataset_path, get_test_dataset_path
from torchvision import datasets

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class FewShotEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Flatten()
        )
    
    def forward(self, x):
        return self.blocks(x)

def create_class_dataloaders(data_path, batch_size, num_workers=2):
    full_dataset = datasets.ImageFolder(data_path)
    class_names = full_dataset.classes
    cinic_mean = [0.47889522, 0.47227842, 0.43047404]
    cinic_std = [0.24205776, 0.23828046, 0.25874835]
    
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
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
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            sampler=RandomSampler(subset, replacement=True),
            num_workers=num_workers,
            pin_memory=True,  # Speeds up GPU transfer
            persistent_workers=num_workers > 0
        )
        class_loaders[class_idx] = loader
    
    return class_loaders, class_names

def episode(embedder, criterion, support_class_loaders, support_class_names, 
            query_class_loaders, query_class_names, C):
    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
    S_means = []
    Q = []

    embedder.eval() if not embedder.training else embedder.train()
    
    with torch.set_grad_enabled(embedder.training):
        for idx, c in enumerate(C):
            # Support phase
            scl = support_class_loaders[support_class_names.index(c)]
            S_images, _ = next(iter(scl))
            S_images = S_images.to(device)
            S_embeddings = embedder(S_images)
            S_means.append(S_embeddings.mean(dim=0))

            # Query phase
            qcl = query_class_loaders[query_class_names.index(c)]
            Q_images, _ = next(iter(qcl))
            Q_images = Q_images.to(device)
            Q_embeddings = embedder(Q_images)
            Q.append((idx, Q_embeddings))

        S_means = torch.stack(S_means)  # [M, embedding_dim]

        loss = 0
        total_samples = 0
        for idx, Q_embeds in Q:
            cos_sims = cos(Q_embeds.unsqueeze(1), S_means.unsqueeze(0))
            log_probs = F.log_softmax(cos_sims, dim=1)
            targets = torch.full((Q_embeds.shape[0],), idx, device=device)
            loss += criterion(log_probs, targets) * Q_embeds.shape[0]
            total_samples += Q_embeds.shape[0]
    
    return loss / total_samples if total_samples > 0 else torch.tensor(0.0)

def train(embedder: FewShotEmbedder,
          train_classes,
          test_classes,
          M=5,
          S_num=5,
          Q_num=15,
          num_epochs=50,
          train_episodes=100,
          test_episodes=50):
    
    embedder = embedder.to(device)
    criterion = nn.NLLLoss().to(device)
    optimizer = optim.AdamW(embedder.parameters(), lr=1e-4, weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=5, factor=0.5, verbose=True
    )

    # Create loaders
    support_class_loaders, support_class_names = create_class_dataloaders(
        get_train_dataset_path(), S_num
    )
    query_class_loaders, query_class_names = create_class_dataloaders(
        get_train_dataset_path(), Q_num
    )
    
    best_loss = float('inf')
    for epoch in range(num_epochs):
        # Training
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
        
        # Validation
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
        
        avg_train_loss = total_loss / train_episodes
        avg_test_loss = total_test_loss / test_episodes
        
        # Update scheduler
        scheduler.step(avg_test_loss)
        
        # Save best model
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            torch.save(embedder.state_dict(), 'best_embedder.pth')
        
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Test Loss: {avg_test_loss:.4f}, "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

if __name__ == "__main__":
    # Example usage
    embedder = FewShotEmbedder()
    num_classes = 1000  # Adjust based on your dataset
    train_classes = list(range(num_classes))
    test_classes = list(range(num_classes))
    
    train(embedder, train_classes, test_classes)