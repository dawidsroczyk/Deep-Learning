import argparse
import math
import json
import os
from datetime import datetime
import data_manager as dm
import torch
import torchvision
import torchvision.transforms.functional as F
from torchvision.utils import make_grid
from diffusers import DDPMScheduler

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[x]

class ResnetBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, embed_channels):
        super(ResnetBlock, self).__init__()
        self.in_layers = torch.nn.Sequential(
            torch.nn.GroupNorm(16, in_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        self.emb_layers = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(embed_channels, out_channels)
        )
        self.out_layers = torch.nn.Sequential(
            torch.nn.GroupNorm(16, out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        if in_channels != out_channels:
            self.shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = torch.nn.Identity()

    def forward(self, x, embedding):
        _input = x
        x = self.in_layers(x)
        emb_out = self.emb_layers(embedding).view(-1, x.size(1), 1, 1)
        x = x + emb_out
        x = self.out_layers(x)
        return x + self.shortcut(_input)

class Upsample(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv(x)
        return x

class Downsample(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class Model(torch.nn.Module):
    def __init__(self, image_channels=3, embed_dim=64):
        super(Model, self).__init__()

        self.embed = torch.nn.Sequential(
            PositionalEncoding(embed_dim),
            torch.nn.Linear(embed_dim, embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(embed_dim, embed_dim),
        )

        self.conv_in = torch.nn.Conv2d(image_channels, 16, kernel_size=3, padding=1)
        self.enc1_1 = ResnetBlock(16, 16, embed_dim)
        self.enc1_2 = ResnetBlock(16, 32, embed_dim)
        self.downconv1 = Downsample(32, 32)
        self.enc2_1 = ResnetBlock(32, 32, embed_dim)
        self.enc2_2 = ResnetBlock(32, 64, embed_dim)
        self.downconv2 = Downsample(64, 64)
        self.bottleneck_1 = ResnetBlock(64, 64, embed_dim)
        self.bottleneck_2 = ResnetBlock(64, 64, embed_dim)
        self.upconv2 = Upsample(64, 64)
        self.dec2_1 = ResnetBlock(128, 64, embed_dim)
        self.dec2_2 = ResnetBlock(64, 32, embed_dim)
        self.upconv1 = Upsample(32, 32)
        self.dec1_1 = ResnetBlock(64, 32, embed_dim)
        self.dec1_2 = ResnetBlock(32, 16, embed_dim)
        self.norm_out = torch.nn.GroupNorm(16, 16)
        self.conv_out = torch.nn.Conv2d(16, image_channels, kernel_size=3, padding=1)

    def forward(self, x, t):
        emb = self.embed(t)

        x = self.conv_in(x)
        x = self.enc1_1(x, emb)
        enc1 = self.enc1_2(x, emb)
        x = self.downconv1(enc1)
        x = self.enc2_1(x, emb)
        enc2 = self.enc2_2(x, emb)
        x = self.downconv2(enc2)
        x = self.bottleneck_1(x, emb)
        x = self.bottleneck_2(x, emb)
        x = self.upconv2(x)
        x = torch.cat([x, enc2], 1)
        x = self.dec2_1(x, emb)
        x = self.dec2_2(x, emb)
        x = self.upconv1(x)
        x = torch.cat([x, enc1], 1)
        x = self.dec1_1(x, emb)
        x = self.dec1_2(x, emb)
        x = self.norm_out(x)
        x = torch.nn.functional.relu(x)
        x = self.conv_out(x)
        return x

class ModelMoreChannels(torch.nn.Module):
    def __init__(self, num_steps=1000):
        super(ModelMoreChannels, self).__init__()

        embed_dim = 32 * 4
        self.embed = torch.nn.Sequential(
            PositionalEncoding(32, num_steps),
            torch.nn.Linear(32, embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(embed_dim, embed_dim),
            torch.nn.ReLU(),
        )

        self.conv_in = torch.nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.enc1_1 = ResnetBlock(32, 32, embed_dim)
        self.enc1_2 = ResnetBlock(32, 64, embed_dim)
        self.downconv1 = Downsample(64, 64)
        self.enc2_1 = ResnetBlock(64, 64, embed_dim)
        self.enc2_2 = ResnetBlock(64, 128, embed_dim)
        self.downconv2 = Downsample(128, 128)
        self.bottleneck_1 = ResnetBlock(128, 128, embed_dim)
        self.bottleneck_2 = ResnetBlock(128, 128, embed_dim)
        self.upconv2 = Upsample(128, 128)
        self.dec2_1 = ResnetBlock(256, 128, embed_dim)
        self.dec2_2 = ResnetBlock(128, 64, embed_dim)
        self.upconv1 = Upsample(64, 64)
        self.dec1_1 = ResnetBlock(128, 64, embed_dim)
        self.dec1_2 = ResnetBlock(64, 32, embed_dim)
        self.norm_out = torch.nn.GroupNorm(32, 32)
        self.conv_out = torch.nn.Conv2d(32, 3, kernel_size=3, padding=1)

    def forward(self, x, t):
        emb = self.embed(t)

        x = self.conv_in(x)
        x = self.enc1_1(x, emb)
        enc1 = self.enc1_2(x, emb)
        x = self.downconv1(enc1)
        x = self.enc2_1(x, emb)
        enc2 = self.enc2_2(x, emb)
        x = self.downconv2(enc2)
        x = self.bottleneck_1(x, emb)
        x = self.bottleneck_2(x, emb)
        x = self.upconv2(x)
        x = torch.cat([x, enc2], 1)
        x = self.dec2_1(x, emb)
        x = self.dec2_2(x, emb)
        x = self.upconv1(x)
        x = torch.cat([x, enc1], 1)
        x = self.dec1_1(x, emb)
        x = self.dec1_2(x, emb)
        x = self.norm_out(x)
        x = torch.nn.functional.silu(x)
        x = self.conv_out(x)
        return x

class ModelMoreChannels2(torch.nn.Module):
    def __init__(self, num_steps=1000):
        super(ModelMoreChannels2, self).__init__()

        embed_dim = 64 * 4
        self.embed = torch.nn.Sequential(
            PositionalEncoding(64, num_steps),
            torch.nn.Linear(64, embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(embed_dim, embed_dim),
            torch.nn.ReLU(),
        )

        self.conv_in = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.enc1_1 = ResnetBlock(64, 64, embed_dim)
        self.enc1_2 = ResnetBlock(64, 128, embed_dim)
        self.downconv1 = Downsample(128, 128)
        self.enc2_1 = ResnetBlock(128, 128, embed_dim)
        self.enc2_2 = ResnetBlock(128, 256, embed_dim)
        self.downconv2 = Downsample(256, 256)
        self.bottleneck_1 = ResnetBlock(256, 256, embed_dim)
        self.bottleneck_2 = ResnetBlock(256, 256, embed_dim)
        self.upconv2 = Upsample(256, 256)
        self.dec2_1 = ResnetBlock(512, 256, embed_dim)
        self.dec2_2 = ResnetBlock(256, 128, embed_dim)
        self.upconv1 = Upsample(128, 128)
        self.dec1_1 = ResnetBlock(256, 128, embed_dim)
        self.dec1_2 = ResnetBlock(128, 64, embed_dim)
        self.norm_out = torch.nn.GroupNorm(32, 64)
        self.conv_out = torch.nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x, t):
        emb = self.embed(t)

        x = self.conv_in(x)
        x = self.enc1_1(x, emb)
        enc1 = self.enc1_2(x, emb)
        x = self.downconv1(enc1)
        x = self.enc2_1(x, emb)
        enc2 = self.enc2_2(x, emb)
        x = self.downconv2(enc2)
        x = self.bottleneck_1(x, emb)
        x = self.bottleneck_2(x, emb)
        x = self.upconv2(x)
        x = torch.cat([x, enc2], 1)
        x = self.dec2_1(x, emb)
        x = self.dec2_2(x, emb)
        x = self.upconv1(x)
        x = torch.cat([x, enc1], 1)
        x = self.dec1_1(x, emb)
        x = self.dec1_2(x, emb)
        x = self.norm_out(x)
        x = torch.nn.functional.silu(x)
        x = self.conv_out(x)
        return x


def get_model(name):
    model_dict = {
        'Model': Model,
        'ModelMoreChannels': ModelMoreChannels,
        'ModelMoreChannels2': ModelMoreChannels2
    }
    if name not in model_dict.keys():
        raise Exception('Invalid Model name')
    
    return model_dict[name]