import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from variational_autoencoder import VariationalAutoencoder
import os
import data_manager as dm

def train_variational_autoencoder(
    epochs,
    img_size,
    emb_dim,
    batch_size=32,
    learning_rate=1e-3,
    optimizer_type='adam',  # 'adam' or 'sgd'
    momentum=0.9,  # only for SGD
    latent_sample_scale=2.0,  # scale for visualization samples
    visualize_every=1,  # visualize every N epochs
    num_visualization_samples=5,
    save_dir="results",
    device=None
):
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = VariationalAutoencoder(img_size, emb_dim, device=device).to(device)
    
    # Initialize optimizer
    if optimizer_type.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    criterion = nn.MSELoss()

    def visualize_results(model, dataloader, epoch, latent_dim=2, num_samples=5):
        model.eval()
        with torch.no_grad():
            test_images = next(iter(dataloader))[:num_samples].to(device)
            reconstructions = model(test_images)
            
            # Define unique mu and sigma for each image
            mu = torch.randn(num_samples, latent_dim).to(device) * latent_sample_scale
            sigma = torch.rand(num_samples, latent_dim).to(device) * 0.5 + 0.1  # std between 0.1 and 0.6
            
            eps = torch.randn(num_samples, latent_dim).to(device)
            random_z = mu + sigma * eps
            
            generated_images = model.decoder(random_z)
            
            fig, axes = plt.subplots(3, num_samples, figsize=(15, 6))
            
            for i in range(num_samples):
                axes[0, i].imshow(test_images[i].permute(1, 2, 0).cpu().numpy().clip(0, 1))
                axes[0, i].set_title("Original")
                axes[0, i].axis('off')
                
                axes[1, i].imshow(reconstructions[i].permute(1, 2, 0).cpu().numpy().clip(0, 1))
                axes[1, i].set_title("Reconstructed")
                axes[1, i].axis('off')
                
                axes[2, i].imshow(generated_images[i].permute(1, 2, 0).cpu().numpy().clip(0, 1))
                axes[2, i].set_title("Generated")
                axes[2, i].axis('off')
            
            plt.suptitle(f"Epoch {epoch + 1} Results")
            plt.tight_layout()
            
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(f"{save_dir}/epoch_{epoch+1}.png")
            plt.close()  # Close the figure to free memory
        
        model.train()

    # Create dataloader
    dataloader = dm.create_full_dataset_dataloader(dm.full_dataset_path_kaggle(), batch_size)

    def train_one_epoch(epoch_index):
        running_loss = 0.0
        for i, images in enumerate(dataloader):
            images = images.to(device)
            
            # Print progress
            if i % 100 == 0:
                print(f'Batch {i}/{len(dataloader)}')
            
            # Training step
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(images, outputs)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        return running_loss / len(dataloader)

    # Training loop
    for epoch in range(epochs):
        print(f'### EPOCH {epoch + 1} / {epochs} ###')
        avg_loss = train_one_epoch(epoch)
        print(f"Average Loss: {avg_loss:.4f}")
        
        # Visualization
        if (epoch + 1) % visualize_every == 0 or epoch == epochs - 1:
            visualize_results(
                model, 
                dataloader, 
                epoch, 
                latent_dim=emb_dim, 
                num_samples=num_visualization_samples
            )
    
    return model