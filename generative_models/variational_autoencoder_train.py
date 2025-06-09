import torch
import torch.nn as nn
from variational_autoencoder import VariationalAutoencoder, save_sample_images
import data_manager as dm
from datetime import datetime
import os
import json
import matplotlib.pyplot as plt

class VAETrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_experiment()
        self.model = self.init_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.dataloader = dm.create_full_dataset_dataloader(
            dm.full_dataset_path_kaggle(), 
            config['batch_size']
        )
        
    def setup_experiment(self):
        """Create experiment directory and save config"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = f"experiments_vae/exp_{timestamp}"
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(f"{self.exp_dir}/recon", exist_ok=True)
        os.makedirs(f"{self.exp_dir}/generated", exist_ok=True)
        
        with open(f"{self.exp_dir}/config.json", 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def init_model(self):
        model = VariationalAutoencoder(
            img_size=self.config['img_size'],
            emb_dimension=self.config['emb_dim'],
            device=self.device,
            in_channels=self.config['in_channels'],
            base_channels=self.config['base_channels'],
            num_blocks=self.config['num_blocks'],
            kernel_size=self.config['kernel_size'],
            stride=self.config['stride']
        ).to(self.device)
        
        # Save model architecture
        with open(f"{self.exp_dir}/model_architecture.txt", 'w') as f:
            f.write(str(model))
            total_params = sum(p.numel() for p in model.parameters())
            f.write(f"\n\nTotal Parameters: {total_params:,}")
        
        return model
    
    def vae_loss(self, recon_x, x, mu, logvar):
        recon_loss = nn.MSELoss(reduction='sum')(recon_x, x)
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.config['vae_beta'] * kld_loss
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kld = 0.0
        
        for i, images in enumerate(self.dataloader):
            images = images[0].to(self.device)
            self.optimizer.zero_grad()
            
            recon, mu, logvar = self.model(images)
            loss = self.vae_loss(recon, images, mu, logvar)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_recon += (loss.item() - self.config['vae_beta'] * (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()).item()))
            total_kld += (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()).item())
            
            # if i % 100 == 0:
            #     print(f'Batch {i}/{len(self.dataloader)} - Loss: {loss.item()/images.size(0):.4f}')
        
        avg_loss = total_loss / len(self.dataloader.dataset)
        avg_recon = total_recon / len(self.dataloader.dataset)
        avg_kld = total_kld / len(self.dataloader.dataset)
        
        return avg_loss, avg_recon, avg_kld
    
    def visualize_results(self, epoch, num_samples=5):
        self.model.eval()
        with torch.no_grad():
            test_images = next(iter(self.dataloader))[0][:num_samples].to(self.device)
            reconstructions, _, _ = self.model(test_images)
            
            # Save images
            save_sample_images(test_images.cpu(), f"{self.exp_dir}/epoch_{epoch:03d}.png")
            save_sample_images(reconstructions.cpu(), f"{self.exp_dir}/recon/epoch_{epoch:03d}.png")
            
            # Generate random samples
            z = torch.randn(num_samples, self.config['emb_dim']).to(self.device)
            generated = self.model.decoder(z)
            save_sample_images(generated.cpu(), f"{self.exp_dir}/generated/epoch_{epoch:03d}.png")
            
            # Create comparison figure
            fig, axes = plt.subplots(3, num_samples, figsize=(15, 6))
            for i in range(num_samples):
                axes[0,i].imshow(test_images[i].permute(1,2,0).cpu().numpy().clip(0,1))
                axes[1,i].imshow(reconstructions[i].permute(1,2,0).cpu().numpy().clip(0,1))
                axes[2,i].imshow(generated[i].permute(1,2,0).cpu().numpy().clip(0,1))
                for ax in axes[:,i]: ax.axis('off')
            
            plt.tight_layout()
            plt.savefig(f"{self.exp_dir}/comparison_epoch_{epoch:03d}.png")
            plt.close()
    
    def train(self):
        metrics = {
            'loss': [],
            'recon_loss': [],
            'kld_loss': []
        }
        
        for epoch in range(self.config['epochs']):
            print(f'Epoch {epoch+1}/{self.config["epochs"]}')
            loss, recon_loss, kld_loss = self.train_epoch(epoch)
            
            metrics['loss'].append(loss)
            metrics['recon_loss'].append(recon_loss)
            metrics['kld_loss'].append(kld_loss)
            
            # Save metrics
            with open(f"{self.exp_dir}/metrics.json", 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Visualize and save checkpoint
            if (epoch + 1) % self.config['save_interval'] == 0:
                self.visualize_results(epoch+1)
                # torch.save(self.model.state_dict(), f"{self.exp_dir}/model_epoch_{epoch+1}.pth")
            
            print(f'Loss: {loss:.4f} (Recon: {recon_loss:.4f}, KLD: {kld_loss:.4f})')
        
        torch.save(self.model.state_dict(), f"{self.exp_dir}/final_model.pth")

if __name__ == "__main__":
    config = {
        'epochs': 100,
        'img_size': 64,
        'emb_dim': 200,
        'in_channels': 3,
        'base_channels': 64,
        'num_blocks': 4,
        'kernel_size': 2,
        'stride': 2,
        'vae_beta': 2.0,
        'batch_size': 32,
        'learning_rate': 1e-3,
        'save_interval': 1
    }
    
    trainer = VAETrainer(config)
    trainer.train()