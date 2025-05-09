import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from variational_autoencoder import VariationalAutoencoder
import os
import data_manager as dm

def train_variational_autoencoder(epochs, img_size, emb_dim):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.MSELoss()
    model = VariationalAutoencoder(img_size, emb_dim, device=device).to(device)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def visualize_results(model, dataloader, epoch, latent_dim=2, num_samples=5, device='cuda'):
        model.eval()
        with torch.no_grad():
            test_images = next(iter(dataloader))[0][:num_samples].to(device)

            reconstructions = model(test_images)

            custom_means = torch.randn(num_samples, latent_dim, device=device) * 2.0  # random means ~ N(0, 2)
            custom_stds = torch.rand(num_samples, latent_dim, device=device) * 2.0 + 0.1  # random stds in [0.1, 2.1]

            eps = torch.randn(num_samples, latent_dim, device=device)
            random_z = custom_means + custom_stds * eps

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
        for i, images in enumerate(dataloader):
            images = images.to(device)
            if i % 100 == 0:
                print(f'{i}/{len(dataloader)}')
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(images, outputs)
            loss.backward()
            optimizer.step()
    
    for epoch in range(epochs):
        print(f'### EPOCH {epoch + 1} / {epochs} ###')
        train_one_epoch(epoch)
        visualize_results(model, dataloader, epoch, latent_dim=emb_dim, num_samples=5)
