import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from diffusers import DDPMScheduler
from tqdm.auto import tqdm
from UNet import UNet
import data_manager as dm
import numpy as np

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
    sample_interval=10,
    gradient_accumulation_steps=1
):
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # Model initialization
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
    
    # Print model summary
    print(f"🧠 Model architecture:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Training setup
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear"
    )
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    # Data pipeline
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # Fixed for 3 channels
    ])
    
    train_dataloader = dm.create_full_dataset_dataloader(
        path=dataset_path,
        batch_size=batch_size,
        transform=transform
    )

    # Training statistics
    train_losses = []
    best_loss = float('inf')

    # Training loop
    with tqdm(range(epochs), desc="Total Training") as epoch_pbar:
        for epoch in epoch_pbar:
            model.train()
            epoch_loss = 0
            optimizer.zero_grad()
            
            with tqdm(train_dataloader, desc=f"Epoch {epoch+1}", leave=False) as batch_pbar:
                for step, images in enumerate(batch_pbar):
                    images = images.to(device)
                    
                    # Sample noise to add to the images
                    noise = torch.randn_like(images)
                    
                    # Sample a random timestep for each image
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, 
                        (images.shape[0],),
                        device=device
                    ).long()

                    # Add noise to the clean images according to the noise magnitude at each timestep
                    noisy_images = noise_scheduler.add_noise(images, noise, timesteps)
                    
                    # Predict the noise residual
                    noise_pred = model(noisy_images, timesteps)
                    loss = criterion(noise_pred, noise)
                    
                    # Normalize loss for gradient accumulation
                    loss = loss / gradient_accumulation_steps
                    loss.backward()

                    # Gradient accumulation
                    if (step + 1) % gradient_accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()

                    batch_pbar.set_postfix(loss=loss.item() * gradient_accumulation_steps)
                    epoch_loss += loss.item() * gradient_accumulation_steps

            # Update learning rate
            avg_loss = epoch_loss / len(train_dataloader)
            scheduler.step(avg_loss)
            train_losses.append(avg_loss)
            
            # Update progress bar
            epoch_pbar.set_postfix(
                avg_loss=avg_loss,
                lr=optimizer.param_groups[0]['lr']
            )

            # Sample generation
            if show_samples and (epoch+1) % sample_interval == 0:
                model.eval()
                with torch.no_grad():
                    # Generate random noise
                    x = torch.randn(1, 3, img_size, img_size, device=device)
                    
                    # Denoising loop
                    for t in tqdm(reversed(range(0, num_train_timesteps)), desc="Sampling", leave=False):
                        timestep = torch.tensor([t], device=device)
                        noise_pred = model(x, timestep)
                        
                        # Compute less noisy image
                        x = noise_scheduler.step(noise_pred, t, x).prev_sample
                    
                    # Post-process sample
                    sample = x.clamp(-1, 1).cpu()
                    sample = sample.squeeze().permute(1, 2, 0).numpy()
                    sample = sample * 0.5 + 0.5  # Scale to [0, 1]
                    
                    # Plot
                    plt.figure(figsize=(4,4))
                    plt.imshow(np.clip(sample, 0, 1))
                    plt.title(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")
                    plt.axis('off')
                    plt.show()

            # Save best model
            if avg_loss < best_loss and save_model_path:
                best_loss = avg_loss
                torch.save(model.state_dict(), save_model_path)
                print(f"💾 Best model saved to {save_model_path} with loss {best_loss:.4f}")

    # Plot training curve
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Curve')
    plt.legend()
    plt.show()

    return model