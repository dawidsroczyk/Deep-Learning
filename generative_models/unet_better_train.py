import argparse
import math
import json
import os
from datetime import datetime
import data_manager as dm
import torch
import torchvision
import torchvision.transforms.functional as F
from torchvision.utils import make_grid
from diffusers import DDPMScheduler
from unet_better_model import *
import random
import numpy as np

def normalize(x):
    return 2 * x - 1

def denormalize(x):
    return (x + 1) / 2

def generate_sample_images(model, noise_scheduler, device, n_samples=8):
    """Generate sample images during training"""
    model.eval()
    with torch.no_grad():
        # Use noise_scheduler from diffusers
        x = torch.randn(n_samples, 3, 32, 32).to(device)
        for step in noise_scheduler.timesteps:
            t = torch.tensor([step], device=device).expand(x.size(0))
            pred_noise = model(x, t)
            x = noise_scheduler.step(pred_noise, step, x).prev_sample
        x = denormalize(x).clamp(0, 1)
    model.train()
    return x

def log_config_info(model, exp_dir, batch_size, epochs, lr, model_name, img_size, seed):
    """Log model architecture and training configuration to a file"""
    with open(f"{exp_dir}/config.txt", "w") as f:
        # Write training configuration
        f.write("Training Configuration:\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Learning rate: {lr}\n")
        f.write(f'Model name: {model_name}\n')
        f.write(f'Image size: {img_size}\n')
        f.write(f'Random seed: {seed}\n')
        f.write("\n")
        
        # Write model architecture summary
        f.write("Model Architecture:\n")
        f.write(str(model))
        
        # Write parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        f.write("\n\nParameter Counts:\n")
        f.write(f"Total Parameters: {total_params:,}\n")
        f.write(f"Trainable Parameters: {trainable_params:,}\n")

def train(batch_size=128, epochs=80, lr=1e-3, model_name='Model', img_size=32, seed=42):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    # Replace with diffusers scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        beta_schedule="linear",
        clip_sample=False
    )
    
    model = get_model(model_name)().to(device)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((img_size, img_size)), 
        torchvision.transforms.ToTensor(), 
        normalize
    ])

    data_loader = dm.create_full_dataset_dataloader(dm.full_dataset_path_kaggle(), batch_size, transform)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = f"experiments_diffusion/exp_{timestamp}"
    os.makedirs(exp_dir, exist_ok=True)
    
    # Log configuration information
    log_config_info(model, exp_dir, batch_size, epochs, lr, model_name, img_size, seed)
    
    # Initialize metrics dictionary
    metrics = {
        'config': {
            'batch_size': batch_size,
            'epochs': epochs,
            'learning_rate': lr,
            'device': str(device),
            'model_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'model_architecture': str(model).replace('\n', ' '),
            'scheduler_config': {
                'num_train_timesteps': noise_scheduler.config.num_train_timesteps,
                'beta_start': noise_scheduler.config.beta_start,
                'beta_end': noise_scheduler.config.beta_end,
                'beta_schedule': noise_scheduler.config.beta_schedule
            }
        },
        'epoch_losses': []
    }

    # Train
    for epoch in range(epochs):
        loss_epoch = 0
        n = 0
        for x, _ in data_loader:
            x = x.to(device)
            optimizer.zero_grad()
            
            # Sample noise to add to the images
            noise = torch.randn_like(x)
            
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, 
                (x.size(0),),
                device=device
            ).long()
            
            # Add noise to the clean images according to the timestep
            noisy_images = noise_scheduler.add_noise(x, noise, timesteps)
            
            # Predict the noise residual
            pred_noise = model(noisy_images, timesteps)
            loss = criterion(pred_noise, noise)
            loss.backward()
            optimizer.step()
            
            loss_epoch += loss.item()
            n += x.size(0)
        
        epoch_loss = loss_epoch / n
        metrics['epoch_losses'].append(epoch_loss)
        
        # Save metrics to JSON file
        with open(f"{exp_dir}/metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Generate and save sample images
        sample_images = generate_sample_images(model, noise_scheduler, device)
        grid = make_grid(sample_images, nrow=8, padding=2)
        grid = F.to_pil_image(grid)
        grid.save(f"{exp_dir}/epoch_{epoch:03d}.png")
        
        print(f"Epoch {epoch}, Loss {epoch_loss:.6f}, Images saved to {exp_dir}/epoch_{epoch:03d}.png")

    torch.save(model.state_dict(), f'{exp_dir}/final_model.pth')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Diffusion Process with Configurable Parameters")
    parser.add_argument('--batch-size', type=int, default=128, help="Batch size")
    parser.add_argument('--epochs', type=int, default=80, help="Number of epochs")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--model', type=str, default='Model', help='Model name')
    parser.add_argument('--img_size', type=int, default=32, help='Size of generated images')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    train(batch_size=args.batch_size, epochs=args.epochs, lr=args.lr, model_name=args.model, img_size=args.img_size, seed=args.seed)
