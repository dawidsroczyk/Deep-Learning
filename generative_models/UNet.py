import torch.nn as nn
import torch

class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.block = nn.Sequential(
            self.create_single_block(input_channels, output_channels, kernel_size, stride),
            self.create_single_block(output_channels, output_channels, kernel_size, stride),
            # self.create_single_block(output_channels, output_channels, kernel_size, stride)
        )
    
    def forward(self, x):
        return self.block(x)
    
    def create_single_block(self, in_channels, out_channels, kernel_size, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride),
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

        assert len(hidden_channels) > 0

        forward_conv_blocks = []
        iter_in_channels = in_channels
        for idx, channels in enumerate(hidden_channels[:-1]):
            out_c = hidden_channels[idx]
            conv_block = ConvBlock(iter_in_channels, out_c, conv_kernel, conv_stride)
            forward_conv_blocks.append(conv_block)
            iter_in_channels = out_c
        self.forward_conv_blocks = forward_conv_blocks

        middle_block = ConvBlock(iter_in_channels, hidden_channels[-1], conv_kernel, conv_stride)
        self.middle_block = middle_block

        reverse_conv_blocks = []
        reverse_deconv_layers = []
        iter_in_channels = hidden_channels[-1]
        for idx, channels in (list(enumerate(hidden_channels))[:-1])[::-1]:
            conv_block = ConvBlock(2*channels, channels, conv_kernel, conv_stride)
            deconv_layer = nn.ConvTranspose2d(2*channels, channels, up_conv_kernel, up_conv_stride)
            reverse_conv_blocks.append(conv_block)
            reverse_deconv_layers.append(deconv_layer)
        self.reverse_conv_blocks = reverse_conv_blocks
        self.reverse_deconv_layers = reverse_deconv_layers

        final_layer = nn.Conv2d(hidden_channels[0], in_channels, 1, 1)
        self.final_layer = final_layer
    
    def forward(self, x):
        outputs = []
        max_pool_layer = nn.MaxPool2d(self.max_pool_kernel, self.max_pool_stride)

        for i in range(len(self.hidden_channels) - 1):
            x = self.forward_conv_blocks[i](x)
            outputs.append(x)
            x = max_pool_layer(x)
        
        x = self.middle_block(x)
        
        for i in range(len(self.hidden_channels) - 1):
            x = self.reverse_deconv_layers[i](x)
            x_concat = outputs[-1-i]
            x_concat = x_concat[:, :, :x.shape[2], :x.shape[2]]
            x = torch.concat((x_concat, x), dim=1)
            x = self.reverse_conv_blocks[i](x)
        
        x = self.final_layer(x)

        return x

def example_unet():
    '''
    Returns unet configuration from a paper
    '''
    return UNet(
    in_channels=3, 
    hidden_channels=[64, 128, 256, 512, 1024],
    conv_kernel=3,
    conv_stride=1,
    max_pool_kernel=2,
    max_pool_stride=2,
    up_conv_kernel=2,
    up_conv_stride=2
)