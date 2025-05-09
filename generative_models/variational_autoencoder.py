import torch
import torch.nn as nn

class VariationalEncoder(nn.Module):
    def __init__(self, img_size=64, emb_dimension=2, device='cpu', 
                 in_channels=3, base_channels=128, num_blocks=4, 
                 kernel_size=2, stride=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device
        self.emb_dimension = emb_dimension
        
        def create_block(in_channels, out_channels, kernel_size, stride):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU()
            )
        
        # Create encoder blocks
        self.blocks = nn.ModuleList()
        current_channels = in_channels
        for i in range(num_blocks):
            self.blocks.append(
                create_block(current_channels, base_channels, kernel_size, stride)
            )
            current_channels = base_channels
        
        self.flatten = nn.Flatten()
        
        # Calculate the linear layer input dimension
        downsampled_size = img_size // (stride ** num_blocks)
        in_dim = (downsampled_size ** 2) * base_channels
        
        self.mean = nn.Linear(in_dim, emb_dimension)
        self.log_var = nn.Linear(in_dim, emb_dimension)
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        
        x = self.flatten(x)
        
        mean = self.mean(x)
        log_var = self.log_var(x)
        
        eps = torch.randn(size=(x.shape[0], self.emb_dimension), device=self.device)
        x = mean + torch.exp(0.5 * log_var) * eps
        
        return x, mean, log_var

class Decoder(nn.Module):
    def __init__(self, emb_dimension=2, img_size=64, device='cpu',
                 out_channels=3, base_channels=128, num_blocks=4,
                 kernel_size=2, stride=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.img_size = img_size
        self.device = device
        self.base_channels = base_channels
        
        def create_block(in_channels, out_channels, kernel_size, stride):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU()
            )
        
        # Calculate the linear layer output dimension
        downsampled_size = img_size // (stride ** num_blocks)
        in_dim = (downsampled_size ** 2) * base_channels
        
        self.lin_1 = nn.Linear(emb_dimension, in_dim)
        self.batch_norm_1 = nn.BatchNorm1d(in_dim)
        self.leaky_relu_1 = nn.LeakyReLU()
        
        # Create decoder blocks
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(
                create_block(base_channels, base_channels, kernel_size, stride)
            )
        
        self.final_conv2d = nn.ConvTranspose2d(base_channels, out_channels, 1, 1)
    
    def forward(self, x):
        x = self.lin_1(x)
        x = self.batch_norm_1(x)
        x = self.leaky_relu_1(x)
        
        # Reshape to start the convolutional transpose path
        downsampled_size = self.img_size // (2 ** len(self.blocks))
        x = torch.reshape(x, (-1, self.base_channels, downsampled_size, downsampled_size))
        
        for block in self.blocks:
            x = block(x)
        
        x = self.final_conv2d(x)
        return x

class VariationalAutoencoder(nn.Module):
    def __init__(self, img_size=64, emb_dimension=2, device='cpu', 
                 in_channels=3, base_channels=128, num_blocks=4,
                 kernel_size=2, stride=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.img_size = img_size
        self.emb_dimension = emb_dimension
        self.device = device
        
        # Initialize encoder and decoder with matching architecture
        encoder_params = {
            'img_size': img_size,
            'emb_dimension': emb_dimension,
            'device': device,
            'in_channels': in_channels,
            'base_channels': base_channels,
            'num_blocks': num_blocks,
            'kernel_size': kernel_size,
            'stride': stride
        }
        
        decoder_params = {
            'emb_dimension': emb_dimension,
            'img_size': img_size,
            'device': device,
            'out_channels': in_channels,
            'base_channels': base_channels,
            'num_blocks': num_blocks,
            'kernel_size': kernel_size,
            'stride': stride
        }
        
        self.encoder = VariationalEncoder(**encoder_params)
        self.decoder = Decoder(**decoder_params)
    
    def forward(self, x):
        x, mean, log_var = self.encoder(x)
        x = self.decoder(x)
        return x, mean, log_var