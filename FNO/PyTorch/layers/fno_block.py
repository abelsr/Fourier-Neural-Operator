import torch
import torch.nn as nn
from typing import List, Union
from .spectral_convolution import SpectralConvolution
from .mlp import MLP

class FourierBlock(nn.Module):
    """
        # Fourier block.
        
        This block consists of three layers:
        1. Fourier layer: SpectralConvolution
        2. MLP layer: MLP
        3. Convolution layer: Convolution
        
    """
    def __init__(self, modes: Union[List[int], int], in_channels: int, out_channels: int, hidden_size: int = None, activation: nn.Module = nn.GELU(), bias: bool = False) -> None:
        """        
        Parameters:
        -----------
        modes: List[int] or int (Required)
            Number of Fourier modes to use in the Fourier layer (SpectralConvolution). Example: [1, 2, 3] or 4
        in_channels: int (Required)
            Number of input channels
        out_channels: int (Required)
            Number of output channels
        hidden_size: int (Optional)
            Number of hidden units in the MLP layer
        activation: nn.Module (Optional)
            Activation function to use in the MLP layer. Default: nn.GELU()
        bias: bool (Optional)
            Whether to add bias to the output. Default: False
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.activation = activation
        self.modes = modes
        self.dim = len(self.modes)
        self.bias = bias
        
        # Fourier layer (SpectralConvolution)
        self.fourier = SpectralConvolution(in_channels, out_channels, modes)
        
        # MLP layer
        if self.hidden_size is not None:
            self.mlp = MLP(len(self.modes), in_channels, out_channels, hidden_size, activation)
        
        # Convolution layer
        if self.dim == 2:
            self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        elif self.dim == 3:
            self.conv = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        else:
            self.conv = nn.Conv1d(in_channels, out_channels, 3, padding=1)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
        ----------
        x: torch.Tensor
            Input tensor of shape [batch, channels, *sizes]
        
        Returns:
        -------
        x: torch.Tensor
            Output tensor of shape [batch, channels, *sizes]
        """
        assert x.size(1) == self.in_channels, f"Input channels must be {self.in_channels} but got {x.size(1)} channels instead."
        sizes = x.size()
        
        if self.bias:
            bias = x
        
        # Fourier layer
        x_ft = self.fourier(x)
        
        # MLP layer
        if self.hidden_size is not None:
            x_mlp = self.mlp(x)
        
        # Convolution layer
        if self.dim == 2 and self.dim == 3:
            x_conv = self.conv(x)
        else:
            x_conv = self.conv(x.reshape(sizes[0], self.in_channels, -1)).reshape(*sizes)
        
        # Add
        x = x_ft + x_conv
        if self.hidden_size is not None:
            x = x + x_mlp
        if self.bias:
            x = x + bias
        # Activation
        x = self.activation(x)
        return x
