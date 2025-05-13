import torch.nn as nn
import torch
import math

class TimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        self.register_buffer('emb', emb)

    def forward(self, timesteps):
        emb = timesteps[:, None] * self.emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, time_emb_dim=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        
        self.time_emb_proj = None
        if time_emb_dim is not None:
            self.time_emb_proj = nn.Linear(time_emb_dim, output_channels)
        
        self.block = nn.Sequential(
            self.create_single_block(input_channels, output_channels, kernel_size, stride),
            self.create_single_block(output_channels, output_channels, kernel_size, stride),
        )
    
    def forward(self, x, time_emb=None):
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

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.conv_kernel = conv_kernel
        self.conv_stride = conv_stride
        self.max_pool_kernel = max_pool_kernel
        self.max_pool_stride = max_pool_stride
        self.up_conv_kernel = up_conv_kernel
        self.up_conv_stride = up_conv_stride
        self.time_emb_dim = time_emb_dim

        self.time_embed = TimestepEmbedding(time_emb_dim)
        
        assert len(hidden_channels) > 0

        forward_conv_blocks = []
        iter_in_channels = in_channels
        for idx, channels in enumerate(hidden_channels[:-1]):
            out_c = hidden_channels[idx]
            conv_block = ConvBlock(iter_in_channels, out_c, conv_kernel, conv_stride, time_emb_dim)
            forward_conv_blocks.append(conv_block)
            iter_in_channels = out_c
        self.forward_conv_blocks = forward_conv_blocks

        middle_block = ConvBlock(iter_in_channels, hidden_channels[-1], conv_kernel, conv_stride, time_emb_dim)
        self.middle_block = middle_block

        reverse_conv_blocks = []
        reverse_deconv_layers = []
        iter_in_channels = hidden_channels[-1]
        for idx, channels in (list(enumerate(hidden_channels))[:-1])[::-1]:
            conv_block = ConvBlock(2*channels, channels, conv_kernel, conv_stride, time_emb_dim)
            deconv_layer = nn.ConvTranspose2d(2*channels, channels, up_conv_kernel, up_conv_stride)
            reverse_conv_blocks.append(conv_block)
            reverse_deconv_layers.append(deconv_layer)
        self.reverse_conv_blocks = reverse_conv_blocks
        self.reverse_deconv_layers = reverse_deconv_layers

        final_layer = nn.Conv2d(hidden_channels[0], in_channels, 1, 1)
        self.final_layer = final_layer
    
    def forward(self, x, timesteps, return_dict=False):
        time_emb = self.time_embed(timesteps)
        
        outputs = []
        max_pool_layer = nn.MaxPool2d(self.max_pool_kernel, self.max_pool_stride)

        for i in range(len(self.hidden_channels) - 1):
            x = self.forward_conv_blocks[i](x, time_emb)
            outputs.append(x)
            x = max_pool_layer(x)
        
        x = self.middle_block(x, time_emb)
        
        for i in range(len(self.hidden_channels) - 1):
            x = self.reverse_deconv_layers[i](x)
            x_concat = outputs[-1-i]
            x_concat = x_concat[:, :, :x.shape[2], :x.shape[2]]
            x = torch.concat((x_concat, x), dim=1)
            x = self.reverse_conv_blocks[i](x, time_emb)
        
        x = self.final_layer(x)

        if return_dict:
            return {"sample": x}
        return x