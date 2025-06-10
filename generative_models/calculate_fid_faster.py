import os
import torch
import random
import shutil
from tqdm import tqdm
from PIL import Image
from cleanfid import fid
import data_manager as dm
from torchvision.utils import save_image
from unet_better_model import get_model
from diffusers import DDPMScheduler
from variational_autoencoder import VariationalAutoencoder
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_images = 256
batch_size = 256
seed = 42
real_images_path = dm.full_dataset_path_kaggle()
subset_real_dir = "real_images_subset_256"
gen_dir = "generated_images_256"

noise_scheduler = DDPMScheduler(
    num_train_timesteps=1000,
    beta_start=1e-4,
    beta_end=0.02,
    beta_schedule="linear",
    clip_sample=False
)

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
            generated = torch.clamp(generated, 0, 1)
        return generated

def prepare_real_subset():
    if os.path.exists(subset_real_dir):
        print(f"Using existing real images subset at {subset_real_dir}")
        return
    
    os.makedirs(subset_real_dir, exist_ok=True)
    random.seed(seed)
    all_real_images = os.listdir(real_images_path)
    selected_real_images = random.sample(all_real_images, num_images)
    
    for img_name in tqdm(selected_real_images, desc="Copying real subset"):
        src = os.path.join(real_images_path, img_name)
        dst = os.path.join(subset_real_dir, img_name)
        shutil.copy(src, dst)
    print(f"Created fixed subset of {num_images} real images at {subset_real_dir}")

def evaluate_model(model_class, experiments_dir, model_type="VAE"):
    fid_scores = []
    
    for exp_path in tqdm(os.listdir(experiments_dir), desc=f"Evaluating {model_type}"):
        exp_dir = os.path.join(experiments_dir, exp_path)
        if 'final_model.pth' not in os.listdir(exp_dir):
            continue

        if os.path.exists(gen_dir):
            shutil.rmtree(gen_dir)
        os.makedirs(gen_dir, exist_ok=True)

        if model_type == "VAE":
            model = VAE_Model(exp_dir, device)
        else:
            model = DiffusionModel(exp_dir, noise_scheduler, device)

        with torch.no_grad():
            if model_type == "VAE":
                z = model.generate_noise(num_samples=num_images, seed=seed)
                samples = model.generate_image(z)
            else:
                z = model.generate_noise(num_samples=num_images, seed=seed)
                samples = model.generate_image(z)
            
            samples = torch.clamp(samples, 0, 1)
            
            for i, img in enumerate(samples):
                save_image(img, os.path.join(gen_dir, f"{i:05d}.png"))

        fid_score = fid.compute_fid(
            subset_real_dir, 
            gen_dir, 
            mode="clean",
            num_workers=0,
            verbose=False
        )
        
        print(f"{model_type} FID ({exp_path}): {fid_score:.2f}")
        fid_scores.append((exp_path, fid_score))
        
        result_file = f"{model_type.lower()}_fid_scores_256.txt"
        with open(result_file, "a") as f:
            f.write(f"{exp_path},{fid_score}\n")
    
    return fid_scores

if __name__ == "__main__":
    random.seed(seed)
    torch.manual_seed(seed)
    
    prepare_real_subset()
    
    print("\n===== Evaluating VAEs =====")
    vae_scores = evaluate_model(VAE_Model, "experiments_vae", "VAE")
    
    print("\n===== Evaluating Diffusion Models =====")
    diffusion_scores = evaluate_model(DiffusionModel, "experiments_diffusion", "Diffusion")
    
    print("\n===== Final FID Scores (256 images) =====")
    print("VAE Scores:", vae_scores)
    print("Diffusion Scores:", diffusion_scores)