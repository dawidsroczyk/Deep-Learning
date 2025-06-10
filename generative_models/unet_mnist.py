import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor
import math
from diffusers import DDPMScheduler
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import wandb
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, embed_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.proj = nn.Linear(embed_channels, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x, embedding):
        x = self.conv1(x)
        embedding = self.proj(embedding).view(-1, x.size(1), 1, 1)
        x = F.relu(x + embedding)
        x = self.conv2(x)
        x = F.relu(x)
        return x

class UNET(nn.Module):
    def __init__(self, T=1000, emb_dim=16, data_channels=3):
        super().__init__()

        self.embed = nn.Embedding(T, emb_dim)

        self.conv1 = ConvBlock(data_channels, 16, emb_dim)
        self.conv2 = ConvBlock(16, 32, emb_dim)

        self.bottleneck = ConvBlock(32, 64, emb_dim)

        self.upscale1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv3 = ConvBlock(64, 32, emb_dim)
        self.upscale2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.conv4 = ConvBlock(32, 16, emb_dim)

        self.conv5 = nn.Conv2d(16, data_channels, kernel_size=1)
    
    def forward(self, x, t):
        if isinstance(t, int):
            t = torch.tensor([t], dtype=torch.long, device=x.device)
        elif isinstance(t, torch.Tensor) and t.dim() == 0:
            t = t.unsqueeze(0)
        elif isinstance(t, torch.Tensor):
            t = t.to(dtype=torch.long, device=x.device)

        embedding = self.embed(t)

        conv1 = self.conv1(x, embedding)
        conv2 = self.conv2(F.max_pool2d(conv1, 2), embedding)

        conv3 = self.upscale1(self.bottleneck(F.max_pool2d(conv2, 2), embedding))
        conv3 = torch.cat([conv2, conv3], 1)
        conv3 = self.conv3(conv3, embedding)

        conv4 = self.upscale2(conv3)
        conv4 = torch.cat([conv1, conv4], 1)
        conv4 = self.conv4(conv4, embedding)

        conv5 = self.conv5(conv4)

        return conv5

def sample_image(model, scheduler, channels, img_size, T, device):
    with torch.no_grad():
        img = torch.randn((1, channels, img_size, img_size), device=device)
        images = []
        
        for t in scheduler.timesteps:
            t_tensor = torch.tensor([t], device=device)
            noise_pred = model(img, t_tensor)
            img = scheduler.step(model_output=noise_pred, timestep=t, sample=img).prev_sample
            
            # Store intermediate images for visualization
            if t % 100 == 0 or t == 0 or t == T-1:  # Store every 100 steps
                images.append(img.cpu())
        
        # Create a grid of images at different timesteps
        grid = make_grid(torch.cat(images, dim=0), nrow=len(images))
        return img, grid

def train():
    # Initialize wandb
    wandb.init(
        project="diffusion-model", 
        entity="your-username",
        mode='offline')
    
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear"
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = 10
    T = 1000
    batch_size = 32
    model = UNET(T=T, data_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = datasets.MNIST(
        root='./data',
        train=True,
        transform=transform,
        download=True
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Log hyperparameters
    wandb.config = {
        "learning_rate": 1e-3,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "T": T
    }

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            optimizer.zero_grad()
            noise = torch.rand_like(images, device=device)
            t = torch.randint(0, T, (batch_size,))
            images_noisy = scheduler.add_noise(images, noise, t)
            noise_pred = model(images_noisy, t)
            loss = criterion(noise_pred, noise)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Log batch loss
            wandb.log({"batch_loss": loss.item()})
            print(f'Epoch {epoch+1}/{num_epochs}, Batch {idx}/{len(dataloader)}, Loss: {loss.item():.4f}')
        
        # Calculate average epoch loss
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_epoch_loss:.4f}')
        
        # Generate sample images at the end of each epoch
        model.eval()
        sample_img, sample_grid = sample_image(model, scheduler, 1, 28, T, device)
        
        # Convert grid to numpy for visualization
        grid_np = sample_grid.permute(1, 2, 0).numpy()
        grid_np = np.clip(grid_np, 0, 1)
        
        # Log epoch metrics and images
        wandb.log({
            "epoch_loss": avg_epoch_loss,
            "generated_images": wandb.Image(grid_np, caption=f"Epoch {epoch+1}"),
            "epoch": epoch+1
        })
        
        # Save model checkpoint
        torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")
        wandb.save(f"model_epoch_{epoch+1}.pth")

def main():
    train()
    wandb.finish()

if __name__ == '__main__':
    main()