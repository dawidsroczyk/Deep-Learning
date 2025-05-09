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
        in_channels=3,
        base_channels=128,
        num_blocks=4,
        kernel_size=2,
        stride=2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def vae_loss(recon_x, x, mu, logvar, beta=1.0):
        MSE = nn.MSELoss(reduction='sum')(recon_x, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return MSE + beta * KLD
    # criterion = nn.MSELoss()
    model = VariationalAutoencoder(
        img_size=img_size, 
        emb_dimension=emb_dim, 
        device=device,
        in_channels=in_channels,
        base_channels=base_channels,
        num_blocks=4,
        kernel_size=2,
        stride=2).to(device)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def visualize_results(model, dataloader, epoch, latent_dim=2, num_samples=5):
        model.eval()
        with torch.no_grad():
            test_images = next(iter(dataloader))[:num_samples].to(device)
            reconstructions = model(test_images)[0]
            
            # Define unique mu and sigma for each image
            mu = torch.randn(num_samples, latent_dim).to(device) * 2  # e.g. scaled for variety
            sigma = torch.rand(num_samples, latent_dim).to(device) * 0.5 + 0.1  # avoid very small std

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
            
            os.makedirs("results", exist_ok=True)
            plt.savefig(f"results/epoch_{epoch+1}.png")
            plt.show()
        
        model.train()


    dataloader = dm.create_full_dataset_dataloader(dm.full_dataset_path_kaggle(), 32)

    def train_one_epoch(epoch_index):
        model.train()
        total_loss = 0.0
        for i, images in enumerate(dataloader):
            images = images.to(device)
            if i % 100 == 0:
                print(f'{i}/{len(dataloader)}')
            optimizer.zero_grad()
            outputs, mean, log_var = model(images)
            # loss = criterion(images, outputs)
            loss = vae_loss(outputs, images, mean, log_var, beta=1.0)
            loss.backward()
            optimizer.step()

            total_loss +- loss.item()
        print(f'Loss: {total_loss}')
    
    for epoch in range(epochs):
        print(f'### EPOCH {epoch + 1} / {epochs} ###')
        train_one_epoch(epoch)
        visualize_results(model, dataloader, epoch, latent_dim=emb_dim, num_samples=5)
