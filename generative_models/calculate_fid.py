import torch
import data_manager as dm
from unet_better_model import get_model
import os
from diffusers import DDPMScheduler
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import json
import torch
from variational_autoencoder import VariationalAutoencoder

noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        beta_schedule="linear",
        clip_sample=False
    )
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DiffusionModel:
    def __init__(self, exp_path, noise_scheduler, device):
        with open(os.path.join(exp_path, 'config.txt'), 'r') as file:
            lines = file.readlines()
            model_name = [x for x in lines if x.startswith('Model name:')][0].rstrip().split()[-1]
        self.model = get_model(model_name)()
        model_path = os.path.join(exp_path, 'final_model.pth')
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model = self.model.to(device)
        self.noise_scheduler = noise_scheduler
        self.device = device
        self.img_size = 32
    
    def generate_noise(self, seed=42, num_samples=1):
        torch.manual_seed(seed)
        x = torch.randn(num_samples, 3, 32, 32).to(self.device)
        return x
    
    def generate_image(self, noise):
        noise_scheduler = self.noise_scheduler
        device = self.device
        model = self.model
        
        def denormalize(x):
            return (x + 1) / 2
        
        model.eval()
        with torch.no_grad():
            x = noise
            for step in noise_scheduler.timesteps:
                t = torch.tensor([step], device=device).expand(x.size(0))
                pred_noise = model(x, t)
                x = noise_scheduler.step(pred_noise, step, x).prev_sample
            x = denormalize(x).clamp(0, 1)
        model.train()
        return x

def create_vae_from_config(config_path, device='cpu'):
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model = VariationalAutoencoder(
        img_size=config.get('img_size', 64),
        emb_dimension=config.get('emb_dim', 2),
        device=device,
        in_channels=config.get('in_channels', 3),
        base_channels=config.get('base_channels', 128),
        num_blocks=config.get('num_blocks', 4),
        kernel_size=config.get('kernel_size', 2),
        stride=config.get('stride', 2)
    )
    
    return model.to(device)

class VAE_Model:
    def __init__(self, exp_path, device):
        config_path = os.path.join(exp_path, 'config.json')
        self.device = device
        self.model = create_vae_from_config(config_path, device=self.device)
        model_path = os.path.join(exp_path, 'final_model.pth')
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval() 
        self.exp_path = exp_path
        self.emb_dim = self.model.emb_dimension
        self.img_size = 64
    
    def generate_noise(self, num_samples=1, mean=0.0, std=1.0, seed=42):
        torch.manual_seed(seed)
        return torch.randn(num_samples, self.emb_dim, device=self.device) * std + mean
    
    def generate_image(self, noise=None, num_samples=1):
        self.model.eval()
        with torch.no_grad():
            if noise is None:
                noise = self.generate_noise(num_samples=num_samples)
            noise = noise.to(self.device)
            generated = self.model.decoder(noise)
            if generated.min() < -0.5:
                generated = torch.clamp(generated, -1, 1)
            else:
                generated = torch.clamp(generated, 0, 1)
        return generated

num_images = os.listdir(dm.full_dataset_path_kaggle()).__len__()
print(num_images)

import os
import torch
import torchvision
from torchvision.utils import save_image
from tqdm import tqdm
from cleanfid import fid
import shutil

batch_size = 128
gen_dir = 'generated_images'
real_images_path = dm.full_dataset_path_kaggle()

vae_fid_scores = []

# Ensure root save folder exists
# os.makedirs(save_root, exist_ok=True)

for exp_path in tqdm(os.listdir('experiments_vae')):
    exp_dir = os.path.join('experiments_vae', exp_path)
    if 'final_model.pth' not in os.listdir(exp_dir):
        continue

    print(f'Processing {exp_path}...')

    model_iter = VAE_Model(exp_dir, device)

    # Create directory to store generated images
    if os.path.exists(gen_dir):
        shutil.rmtree(gen_dir)
    os.makedirs(gen_dir, exist_ok=True)

    with torch.no_grad():
        generated_count = 0
        img_idx = 0
        while generated_count < num_images:
            z = model_iter.generate_noise(batch_size)
            samples = model_iter.generate_image(z)
            
            # Clamp and scale to [0,1] for saving
            samples = torch.clamp(samples, 0, 1)

            for img in samples:
                save_image(img, os.path.join(gen_dir, f"{img_idx:05d}.png"))
                img_idx += 1
                generated_count += 1
                if generated_count >= num_images:
                    break

    fid_score = fid.compute_fid(dm.full_dataset_path_kaggle(), gen_dir, verbose=False, dataset_res=64)
    print(f"FID for {exp_path}: {fid_score}")
    vae_fid_scores.append((exp_path, fid_score))

    with open("vae_fid_scores.txt", "a") as f:
        f.write(f"{exp_path},{fid_score}\n")


diffusion_fid_scores = []
for exp_path in tqdm(os.listdir('experiments_diffusion')):
    exp_dir = os.path.join('experiments_diffusion', exp_path)
    if 'final_model.pth' not in os.listdir(exp_dir):
        continue

    print(f'Processing {exp_path}...')

    model_iter = DiffusionModel(exp_dir, noise_scheduler, device)

    # Create directory to store generated images
    if os.path.exists(gen_dir):
        shutil.rmtree(gen_dir)
    os.makedirs(gen_dir, exist_ok=True)

    with torch.no_grad():
        generated_count = 0
        img_idx = 0
        while generated_count < num_images:
            z = model_iter.generate_noise(batch_size)
            samples = model_iter.generate_image(z)
            
            # Clamp and scale to [0,1] for saving
            samples = torch.clamp(samples, 0, 1)

            for img in samples:
                save_image(img, os.path.join(gen_dir, f"{img_idx:05d}.png"))
                img_idx += 1
                generated_count += 1
                if generated_count >= num_images:
                    break

    fid_score = fid.compute_fid(dm.full_dataset_path_kaggle(), gen_dir, verbose=False, dataset_res=64)
    print(f"FID for {exp_path}: {fid_score}")
    diffusion_fid_scores.append((exp_path, fid_score))

    with open("diffusion_fid_scores.txt", "a") as f:
        f.write(f"{exp_path},{fid_score}\n")