import numpy as np
import matplotlib.pyplot as plt
import os
import random
import torch
from scipy.interpolate import interp1d
from torchvision.utils import save_image, make_grid
from torchvision import datasets, transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from PIL import Image
import os
from tqdm import tqdm

from models.dcgan import Generator

def display_results(dir):
    D_losses = np.loadtxt(os.path.join(os.path.join(dir, 'loss'), 'netD.txt'))
    G_losses = np.loadtxt(os.path.join(os.path.join(dir, 'loss'), 'netG.txt'))

    D_real_mean_out = np.loadtxt(os.path.join(os.path.join(dir, 'mean_out'), 'netD_real.txt'))
    D_fake_mean_out = np.loadtxt(os.path.join(os.path.join(dir, 'mean_out'), 'netD_fake.txt'))

    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(dir, 'loss', 'loss.png'))
    plt.show()

    plt.figure(figsize=(10,5))
    plt.title("Discriminator Mean Scores During Training")
    plt.plot(D_real_mean_out, label="Real")
    plt.plot(D_fake_mean_out, label="Fake")
    plt.xlabel("Iterations")
    plt.ylabel("Mean")
    plt.legend()
    plt.savefig(os.path.join(dir, 'mean_out', 'mean_out.png'))
    plt.show()

def save_fake_imgs(ngpu, nz, ngf, nc, device, dir, random_state=42):

    random.seed(random_state)
    torch.manual_seed(random_state)

    netG = Generator(ngpu, nz, ngf, nc).to(device)
    netG.load_state_dict(torch.load(os.path.join(os.path.join(dir, 'nets'), 'netG')))

    fixed_noise = torch.randn(100, nz, 1, 1, device=device)
    with torch.no_grad():
        fake = netG(fixed_noise).detach().cpu()

    if not os.path.exists(os.path.join(dir, 'fake_imgs')):
        os.mkdir(os.path.join(dir, 'fake_imgs'))

    for i, img in enumerate(fake):
        save_image(img, os.path.join(os.path.join(dir, 'fake_imgs'), f'fake_{i}.png'))


def get_save_interp_vectors(nz, device, dir):
    if not os.path.exists(os.path.join(dir, 'interpol')):
        os.mkdir(os.path.join(dir, 'interpol'))

    start = torch.randn(nz, device=device)
    end = torch.randn(nz, device=device)

    linfit = interp1d([1, 10], torch.vstack([start, end]).cpu().numpy(), axis=0)
    interp_vectors = [linfit(i) for i in range(1, 10 + 1)]  # 10 interpolation points

    np.savetxt(os.path.join(dir, 'interpol', 'interpol.csv'), 
               np.array(interp_vectors))

    return torch.tensor(np.array(interp_vectors), device=device).float()

def generate_interpol_dcgan(dir, ngpu, nz, ngf, nc, device):
    interp_vectors = get_save_interp_vectors(nz, device, dir)
    
    interp_vectors = interp_vectors.reshape(10, nz, 1, 1).float()
    
    netG = Generator(ngpu, nz, ngf, nc).to(device)
    netG.load_state_dict(torch.load(os.path.join(dir, 'nets', 'netG')))
    
    with torch.no_grad():
        fake = netG(interp_vectors).detach().cpu()
    
    grid = make_grid(fake, padding=2, normalize=True, nrow=5)
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.savefig(os.path.join(dir, 'interpol', 'interpol.png'))
    plt.close()


def load_images_from_folder(folder, transform, max_images=None):

    images = []
    for i, filename in enumerate(sorted(os.listdir(folder))):
        if max_images and i >= max_images:
            break
        path = os.path.join(folder, filename)
        try:
            image = Image.open(path).convert('RGB')
            images.append(transform(image))
        except:
            continue
    return torch.stack(images)


def calculate_fid(real_path, fake_path, device, max_images=100):

    transform = transforms.Compose([
        transforms.Resize((299, 299)),  # InceptionV3 expects 299x299
        transforms.ToTensor()  # Converts to [0,1] range (torch.float32)
    ])

    real_images = load_images_from_folder(real_path, transform, max_images)
    fake_images = load_images_from_folder(fake_path, transform, max_images)

    real_images = (real_images * 255).type(torch.uint8)
    fake_images = (fake_images * 255).type(torch.uint8)

    real_images = real_images.to(device)
    fake_images = fake_images.to(device)

    fid = FrechetInceptionDistance(feature=2048).to(device)

    batch_size = 50
    for i in range(0, len(real_images), batch_size):
        fid.update(real_images[i:i+batch_size], real=True)
    for i in range(0, len(fake_images), batch_size):
        fid.update(fake_images[i:i+batch_size], real=False)

    return fid.compute().item()