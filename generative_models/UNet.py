import torch
import torch.nn as nn
import torch.nn.functional as F
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
        if x.dim() == 4:
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1).reshape(B, H*W, C)
            x = x + self.pe[:H*W]
            x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        else:
            x = x + self.pe[:x.size(1)]
        return self.dropout(x)

class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, time_emb_dim=None):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        
        self.positional_encoding = PositionalEncoding(d_model=output_channels)
        
        self.time_emb_proj = None
        if time_emb_dim is not None:
            self.time_emb_proj = nn.Linear(time_emb_dim, output_channels)
        
        self.block = nn.Sequential(
            self.create_single_block(input_channels, output_channels, kernel_size, stride),
            self.create_single_block(output_channels, output_channels, kernel_size, stride),
        )
    
    def forward(self, x, time_emb=None):
        x = self.block[0](x)
        
        x = self.positional_encoding(x)
        
        if self.time_emb_proj is not None and time_emb is not None:
            time_emb = self.time_emb_proj(time_emb)
            time_emb = time_emb.view(-1, self.output_channels, 1, 1)
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
    def __init__(self, 
                 in_channels: int, 
                 hidden_channels: list[int], 
                 conv_kernel: int = 3,
                 conv_stride: int = 1,
                 max_pool_kernel: int = 2,
                 max_pool_stride: int = 2,
                 up_conv_kernel: int = 2, 
                 up_conv_stride: int = 2,
                 time_emb_dim: int = 32,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.time_emb_dim = time_emb_dim
        
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        ) if time_emb_dim > 0 else None
        
        self.forward_conv_blocks = nn.ModuleList()
        iter_in_channels = in_channels
        for channels in hidden_channels[:-1]:
            self.forward_conv_blocks.append(
                ConvBlock(iter_in_channels, channels, conv_kernel, conv_stride)
            )
            iter_in_channels = channels

        self.middle_block = ConvBlock(
            hidden_channels[-2] if len(hidden_channels) > 1 else iter_in_channels,
            hidden_channels[-1], conv_kernel, conv_stride, time_emb_dim
        )

        self.reverse_conv_blocks = nn.ModuleList()
        self.reverse_deconv_layers = nn.ModuleList()
        for i in range(len(hidden_channels)-1, 0, -1):
            self.reverse_deconv_layers.append(
                nn.ConvTranspose2d(hidden_channels[i], hidden_channels[i-1], 
                                  up_conv_kernel, up_conv_stride)
            )
            self.reverse_conv_blocks.append(
                ConvBlock(2*hidden_channels[i-1], hidden_channels[i-1], 
                         conv_kernel, conv_stride, time_emb_dim)
            )

        self.final_layer = nn.Conv2d(hidden_channels[0], in_channels, 1)
        
        self.max_pool = nn.MaxPool2d(max_pool_kernel, max_pool_stride)
    
    def forward(self, x, timesteps=None, return_dict=False):
        time_emb = None
        if self.time_embed is not None and timesteps is not None:
            timesteps = timesteps.view(-1, 1).float()
            time_emb = self.time_embed(timesteps)
        
        outputs = []
        for conv_block in self.forward_conv_blocks:
            x = conv_block(x, time_emb)
            outputs.append(x)
            x = self.max_pool(x)
        
        x = self.middle_block(x, time_emb)
        
        for deconv, conv_block in zip(self.reverse_deconv_layers, self.reverse_conv_blocks):
            x = deconv(x)
            x_concat = outputs.pop()
            
            if x.shape != x_concat.shape:
                x = F.interpolate(x, size=x_concat.shape[2:], mode='bilinear', align_corners=False)
            
            x = torch.cat([x_concat, x], dim=1)
            x = conv_block(x, time_emb)
        
        x = self.final_layer(x)

        return {"sample": x} if return_dict else x