import torch
import torch.nn as nn
from .spectral_convolution import SpectralConvolution
from .mlp import MLP

class FourierBlock(nn.Module):
    def __init__(self, modes, in_channels, out_channels, hidden_size, num_hidden, activation=nn.GELU(), bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size or 8
        self.num_hidden = num_hidden or 4
        self.activation = activation
        self.modes = modes
        self.dim = len(self.modes)
        self.bias = bias
        
        # Fourier layer 
        self.fourier = SpectralConvolution(in_channels, out_channels, modes)
        
        # MLP layer
        self.mlp = MLP(len(self.modes), in_channels, out_channels, hidden_size, activation)
        
        # Convolution layer
        if self.dim == 2:
            self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        elif self.dim == 3:
            self.conv = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        else:
            self.conv = nn.Conv1d(in_channels, out_channels, 3, padding=1)
            
    def forward(self, x):
        """
        x: torch.Tensor
            Input tensor of shape [batch, channels, *sizes]
        
        """
        sizes = x.size()
        
        if self.bias:
            bias = x
        
        # Fourier layer
        x_ft = self.fourier(x)
        
        # MLP layer
        x_mlp = self.mlp(x)
        
        # Convolution layer
        x_conv = self.conv(x).view(*sizes)
        
        # Add 
        x = x_ft + x_mlp + x_conv
        if self.bias:
            x = x + bias
        return x
    
    
    
# x = torch.randn(1, 1, 32, 32, 32)
# fourier_block = FourierBlock([1, 2, 3], 1, 1, 64, 3)
# output = fourier_block(x)
# print(x.shape)
# print(output.shape)