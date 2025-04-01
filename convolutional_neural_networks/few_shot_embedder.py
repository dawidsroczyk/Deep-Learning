import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, RandomSampler
from torchvision import datasets, transforms
import numpy as np
import gc
from download_dataset import get_train_dataset_path, get_test_dataset_path

# Memory optimization settings
torch.backends.cudnn.benchmark = True  # Faster convolutions without changing architecture
torch.autograd.set_detect_anomaly(False)  # Disable for performance

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Keep your exact architecture without changes
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
    # Keep your exact transform pipeline
    cinic_mean = [0.47889522, 0.47227842, 0.43047404]
    cinic_std = [0.24205776, 0.23828046, 0.25874835]
    
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(10),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=cinic_mean, std=cinic_std)
    ])
    
    dataset = datasets.ImageFolder(data_path, transform=transform)
    
    # Create class indices (unchanged)
    class_indices = {i: [] for i in range(len(dataset.classes))}
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)
    
    # Memory-optimized loader creation
    class_loaders = {}
    for class_idx, indices in class_indices.items():
        subset = Subset(dataset, indices)
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            sampler=RandomSampler(subset, replacement=True),
            num_workers=min(2, num_workers),  # Reduced workers to save memory
            pin_memory=True,
            persistent_workers=False  # Disabled to save memory
        )
        class_loaders[class_idx] = loader
    
    return class_loaders, dataset.classes

def process_batch(embedder, images):
    """Helper function to process batches with memory cleanup"""
    images = images.to(device, non_blocking=True)
    embeddings = embedder(images)
    return embeddings

def episode(embedder, criterion, support_class_loaders, support_class_names, 
            query_class_loaders, query_class_names, C):
    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
    S_means = []
    Q = []
    
    # Process support samples with memory cleanup
    for idx, c in enumerate(C):
        # Support phase with memory management
        scl = support_class_loaders[support_class_names.index(c)]
        S_images, _ = next(iter(scl))
        S_embeddings = process_batch(embedder, S_images)
        S_means.append(S_embeddings.mean(dim=0))
        del S_images, S_embeddings
        
        # Query phase with memory management
        qcl = query_class_loaders[query_class_names.index(c)]
        Q_images, _ = next(iter(qcl))
        Q_embeddings = process_batch(embedder, Q_images)
        Q.append((idx, Q_embeddings))
        del Q_images, Q_embeddings
        
        torch.cuda.empty_cache()

    S_means = torch.stack(S_means)  # [M, embedding_dim]

    loss = 0
    total_samples = 0
    
    # Process query samples in smaller chunks if needed
    for idx, Q_embeds in Q:
        chunk_size = 32  # Process queries in chunks to save memory
        for i in range(0, Q_embeds.size(0), chunk_size):
            chunk = Q_embeds[i:i+chunk_size]
            
            cos_sims = cos(chunk.unsqueeze(1), S_means.unsqueeze(0))
            log_probs = F.log_softmax(cos_sims, dim=1)
            targets = torch.full((chunk.size(0),), idx, device=device)
            
            loss += criterion(log_probs, targets) * chunk.size(0)
            total_samples += chunk.size(0)
        
        del Q_embeds
        torch.cuda.empty_cache()
    
    del S_means, Q
    torch.cuda.empty_cache()
    
    return loss / max(total_samples, 1)

def train(embedder, train_classes, test_classes, M=5, S_num=5, Q_num=15,
          num_epochs=50, train_episodes=100, test_episodes=50):
    
    embedder = embedder.to(device)
    criterion = nn.NLLLoss().to(device)
    optimizer = optim.AdamW(embedder.parameters(), lr=1e-4, weight_decay=1e-4)
    
    # Gradient accumulation to reduce memory peaks
    accum_steps = 2  
    
    for epoch in range(num_epochs):
        # Training phase with memory management
        embedder.train()
        train_loss = 0
        
        # Create fresh loaders each epoch to prevent memory buildup
        support_loaders, support_classes = create_class_dataloaders(
            get_train_dataset_path(), S_num
        )
        query_loaders, query_classes = create_class_dataloaders(
            get_train_dataset_path(), Q_num
        )
        
        for episode_idx in range(train_episodes):
            C = np.random.choice(train_classes, M, replace=False)
            
            loss = episode(embedder, criterion, 
                         support_loaders, support_classes,
                         query_loaders, query_classes,
                         C)
            
            # Gradient accumulation
            loss = loss / accum_steps
            loss.backward()
            
            if (episode_idx + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()
            
            train_loss += loss.item() * accum_steps
            
            # Periodic memory cleanup
            if episode_idx % 10 == 0:
                gc.collect()
        
        # Validation phase
        embedder.eval()
        test_loss = 0
        with torch.no_grad():
            val_support, val_support_classes = create_class_dataloaders(
                get_test_dataset_path(), S_num
            )
            val_query, val_query_classes = create_class_dataloaders(
                get_test_dataset_path(), Q_num
            )
            
            for _ in range(test_episodes):
                C = np.random.choice(test_classes, M, replace=False)
                loss = episode(embedder, criterion,
                              val_support, val_support_classes,
                              val_query, val_query_classes,
                              C)
                test_loss += loss.item()
        
        avg_train = train_loss / train_episodes
        avg_test = test_loss / test_episodes
        
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss: {avg_train:.4f}, "
              f"Test Loss: {avg_test:.4f}")
        
        # Save checkpoint with memory cleanup
        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state': embedder.state_dict(),
                'loss': avg_test,
            }, f"checkpoint_epoch{epoch}.pth")
            torch.cuda.empty_cache()

if __name__ == "__main__":
    # Initialize with your exact parameters
    embedder = FewShotEmbedder()
    num_classes = 1000  # Keeping your original parameter
    train_classes = list(range(num_classes))
    test_classes = list(range(num_classes))
    
    try:
        train(embedder, train_classes, test_classes,
              M=5, S_num=5, Q_num=15,  # Your original parameters
              num_epochs=50, train_episodes=100, test_episodes=50)
    except RuntimeError as e:
        print(f"Error: {e}")
        print("If memory error persists, try reducing num_workers or using smaller image size")