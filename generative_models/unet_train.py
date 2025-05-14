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
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # Model initialization ON TARGET DEVICE
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
    ).to(device)  # This is crucial!

    # Training setup
    noise_scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Data pipeline
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

    # Training loop
    with tqdm(range(epochs), desc="Total Training") as epoch_pbar:
        for epoch in epoch_pbar:
            model.train()
            epoch_loss = 0
            
            with tqdm(train_dataloader, desc=f"Epoch {epoch+1}", leave=False) as batch_pbar:
                for images in batch_pbar:
                    # Ensure everything is on same device
                    images = images.to(device)
                    noise = torch.randn_like(images, device=device)  # Explicit device
                    
                    timesteps = torch.randint(
                        0, num_train_timesteps, 
                        (images.shape[0],), 
                        device=device  # Explicit device
                    )

                    noisy_images = noise_scheduler.add_noise(images, noise, timesteps)
                    noise_pred = model(noisy_images, timesteps)
                    loss = criterion(noise_pred, noise)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    batch_pbar.set_postfix(loss=loss.item())
                    epoch_loss += loss.item()

            # Epoch statistics
            avg_loss = epoch_loss / len(train_dataloader)
            epoch_pbar.set_postfix(avg_loss=avg_loss)

            # Sample generation
            if show_samples and (epoch+1) % sample_interval == 0:
                model.eval()
                with torch.no_grad():
                    x = torch.randn(1, 3, img_size, img_size, device=device)
                    for t in reversed(range(0, num_train_timesteps)):
                        x = model(x, torch.tensor([t], device=device))
                    sample = x.clamp(-1, 1)
                    
                    plt.figure(figsize=(4,4))
                    plt.imshow(sample.squeeze().permute(1,2,0).cpu().numpy() * 0.5 + 0.5)
                    plt.title(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")
                    plt.axis('off')
                    plt.show()

    # Model saving
    if save_model_path:
        torch.save(model.state_dict(), save_model_path)
        print(f"💾 Model saved to {save_model_path}")

    return model