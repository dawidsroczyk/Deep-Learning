import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import data_manager as dm

class VariationalEncoder(nn.Module):
    def __init__(self, img_size=64, emb_dimension=2, device='cpu', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device

        def create_block(in_channels, out_channels, kernel_size=2, stride=2):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU()
            )
        self.emb_dimension = emb_dimension
        
        in_channels = 3
        out_channels = 128
        kernel_size = 2
        stride = 2

        self.block_1 = create_block(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
        self.block_2 = create_block(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
        self.block_3 = create_block(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
        self.block_4 = create_block(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
        self.flatten = nn.Flatten()
        in_dim = img_size * img_size // 256 * out_channels
        self.mean = nn.Linear(in_dim, emb_dimension)
        self.log_var = nn.Linear(in_dim, emb_dimension)
    
    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.flatten(x)

        mean = self.mean(x)
        log_var = self.log_var(x)
        
        eps = torch.randn(size=(x.shape[0], self.emb_dimension), device=self.device)
        x = mean + torch.exp(0.5 * log_var) * eps

        return x

class Decoder(nn.Module):
    def __init__(self, emb_dimension=2, img_size=64, device='cpu', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.img_size = img_size
        self.out_channels = 128
        self.device = device

        out_channels = 128
        in_channels = 3
        kernel_size = 2
        stride = 2

        def create_block(in_channels, out_channels):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, in_channels, kernel_size, stride),
                nn.BatchNorm2d(in_channels),
                nn.LeakyReLU()
            )

        in_dim = img_size * img_size // 256 * out_channels

        self.lin_1 = nn.Linear(emb_dimension, in_dim)
        self.batch_norm_1 = nn.BatchNorm1d(in_dim)
        self.leaky_relu_1 = nn.LeakyReLU()

        self.block_1 = create_block(out_channels, out_channels)
        self.block_2 = create_block(out_channels, out_channels)
        self.block_3 = create_block(out_channels, out_channels)
        self.block_4 = create_block(out_channels, out_channels)

        self.final_conv2d = nn.ConvTranspose2d(out_channels, in_channels, 1, 1)
    
    def forward(self, x):

        x = self.lin_1(x)
        x = self.batch_norm_1(x)
        x = self.leaky_relu_1(x)

        x = torch.reshape(x, (-1, self.out_channels, self.img_size // 16, self.img_size // 16))

        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)

        x = self.final_conv2d(x)

        return x

class VariationalAutoencoder(nn.Module):
    def __init__(self, img_size, emb_dimension, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.img_size = img_size
        self.emb_dimension = emb_dimension
        self.device = device
        
        self.encoder = VariationalEncoder(img_size, emb_dimension, device)
        self.decoder = Decoder(emb_dimension, img_size, device)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x