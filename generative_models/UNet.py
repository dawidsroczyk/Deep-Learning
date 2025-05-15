import torch.nn as nn
import torch
import math

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)

class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride

        self.positional_encoding = PositionalEncoding(dropout=0.0)
        
        self.block = nn.Sequential(
            self.create_single_block(input_channels, output_channels, kernel_size, stride),
            self.create_single_block(output_channels, output_channels, kernel_size, stride),
        )
    
    def forward(self, x, time_emb=None):
        x = self.positional_encoding(x)
        x = self.block[0](x)
        if self.time_emb_proj is not None and time_emb is not None:
            time_emb = self.time_emb_proj(time_emb)
            time_emb = time_emb.unsqueeze(-1).unsqueeze(-1)
            x = x + time_emb
        x = self.block[1](x)
        return x
    
    def create_single_block(self, in_channels, out_channels, kernel_size, stride):
        padding = kernel_size // 2
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

class UNet(nn.Module):
    def __init__(self, in_channels: int, 
                 hidden_channels: list[int], 
                 conv_kernel, 
                 conv_stride, 
                 max_pool_kernel,
                 max_pool_stride,
                 up_conv_kernel, 
                 up_conv_stride,
                 time_emb_dim=32,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Store all parameters
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.conv_kernel = conv_kernel
        self.conv_stride = conv_stride
        self.max_pool_kernel = max_pool_kernel
        self.max_pool_stride = max_pool_stride
        self.up_conv_kernel = up_conv_kernel
        self.up_conv_stride = up_conv_stride
        
        assert len(hidden_channels) > 0

        # Encoder (downsampling path)
        self.forward_conv_blocks = nn.ModuleList()
        iter_in_channels = in_channels
        for channels in hidden_channels[:-1]:
            self.forward_conv_blocks.append(
                ConvBlock(iter_in_channels, channels, conv_kernel, conv_stride)
            )
            iter_in_channels = channels

        # Middle block
        self.middle_block = ConvBlock(iter_in_channels, hidden_channels[-1], conv_kernel, conv_stride)

        # Decoder (upsampling path)
        self.reverse_conv_blocks = nn.ModuleList()
        self.reverse_deconv_layers = nn.ModuleList()
        iter_in_channels = hidden_channels[-1]
        for channels in reversed(hidden_channels[:-1]):
            self.reverse_deconv_layers.append(
                nn.ConvTranspose2d(2*channels, channels, up_conv_kernel, up_conv_stride)
            )
            self.reverse_conv_blocks.append(
                ConvBlock(2*channels, channels, conv_kernel, conv_stride, time_emb_dim)
            )

        # Final layer
        self.final_layer = nn.Conv2d(hidden_channels[0], in_channels, 1, 1)
        
        # Max pooling layer (not trainable)
        self.max_pool = nn.MaxPool2d(max_pool_kernel, max_pool_stride)
    
    def forward(self, x, timesteps, return_dict=False):
        # Ensure timesteps are on same device as model
        timesteps = timesteps.to(x.device)
        
        # Time embedding
        time_emb = self.time_embed(timesteps)
        
        # Encoder path
        outputs = []
        for conv_block in self.forward_conv_blocks:
            x = conv_block(x, time_emb)
            outputs.append(x)
            x = self.max_pool(x)
        
        # Middle block
        x = self.middle_block(x, time_emb)
        
        # Decoder path
        for deconv, conv_block in zip(self.reverse_deconv_layers, self.reverse_conv_blocks):
            x = deconv(x)
            x_concat = outputs.pop()
            # Ensure spatial dimensions match
            # if x_concat.shape[2:] != x.shape[2:]:
            #     x = F.interpolate(x, size=x_concat.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x_concat, x], dim=1)
            x = conv_block(x, time_emb)
        
        # Final output
        x = self.final_layer(x)

        return {"sample": x} if return_dict else x