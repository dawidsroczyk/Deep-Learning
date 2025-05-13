import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from diffusers import DDPMScheduler
from tqdm.auto import tqdm
from UNet import UNet
import data_manager as dm

def train_unet(
    dataset_path,
    epochs=100,
    img_size=64,
    batch_size=32,
    learning_rate=0.001,
    hidden_channels=[128, 256, 512, 1024],
    time_emb_dim=32,
    num_train_timesteps=1000,
    save_model_path=None,
    show_samples=True,
    sample_interval=10
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    noise_scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps)

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    train_dataloader = dm.create_full_dataset_dataloader(
        path=dataset_path,
        batch_size=batch_size,
        transform=transform
    )

    model = UNet(
        in_channels=3,
        hidden_channels=hidden_channels,
        conv_kernel=3,
        conv_stride=1,
        max_pool_kernel=2,
        max_pool_stride=2,
        up_conv_kernel=2,
        up_conv_stride=2,
        time_emb_dim=time_emb_dim
    ).to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    with tqdm(range(epochs), desc="Total Training", unit="epoch") as epoch_pbar:
        for epoch in epoch_pbar:
            model.train()
            epoch_loss = 0
            
            with tqdm(train_dataloader, desc=f"Epoch {epoch+1}", leave=False, unit="batch") as batch_pbar:
                for images in batch_pbar:
                    images = images.to(device)
                    noise = torch.randn_like(images)
                    timesteps = torch.randint(
                        0, num_train_timesteps, 
                        (images.shape[0],), device=device
                    )

                    noisy_images = noise_scheduler.add_noise(images, noise, timesteps)
                    noise_pred = model(noisy_images, timesteps)
                    loss = criterion(noise_pred, noise)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    batch_pbar.set_postfix(loss=loss.item())
                    epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_dataloader)
            epoch_pbar.set_postfix(avg_loss=avg_loss)

            if show_samples and (epoch+1) % sample_interval == 0:
                model.eval()
                with torch.no_grad():
                    sample = model(
                        torch.randn(1, 3, img_size, img_size, device=device),
                        torch.tensor([num_train_timesteps-1], device=device)
                    )
                    plt.imshow(sample.squeeze().permute(1,2,0).cpu().numpy() * 0.5 + 0.5)
                    plt.title(f"Epoch {epoch+1}")
                    plt.axis('off')
                    plt.show()

    if save_model_path:
        torch.save(model.state_dict(), save_model_path)
        print(f"Model saved to {save_model_path}")

    return model