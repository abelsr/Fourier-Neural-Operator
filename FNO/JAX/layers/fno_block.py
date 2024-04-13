
# TODO: Make FourierBlock a subclass of nn.Module

import jax.numpy as jnp
from jax import random, jit, vmap
from flax import linen as nn
from spectral_convolution import SpectralConvolution
from mlp import MLP

class FourierBlock(nn.Module):
    """
        # Fourier block.
        
        This block consists of three layers:
        1. Fourier layer: SpectralConvolution
        2. MLP layer: MLP
        3. Convolution layer: Convolution
        
        
        Parameters:
        -----------
        modes: List[int] or Int (Required)
            Number of Fourier modes to use in the Fourier layer (SpectralConvolution). Example: [1, 2, 3] or 4
        in_channels: int (Required)
            Number of input channels
        out_channels: int (Required)
            Number of output channels
        hidden_size: int (Optional)
            Number of hidden units in the MLP layer
        num_hidden: int (Optional)
            Number of hidden layers in the MLP layer
        activation: nn.Module (Optional)
            Activation function to use in the MLP layer. Default: nn.GELU()
        bias: bool (Optional)
            Whether to add bias to the output. Default: False
        
    """
    modes: list
    in_channels: int
    out_channels: int
    hidden_size: int = 8
    num_hidden: int = 4
    activation: nn.Module = nn.gelu
    bias: bool = False
    
    def setup(self):
        self.dim = len(self.modes)
        
        # Fourier layer (SpectralConvolution)
        self.fourier = SpectralConvolution(self.in_channels, self.out_channels, self.modes)
        
        # MLP layer
        self.mlp = MLP(len(self.modes), self.in_channels, self.out_channels, self.hidden_size, self.activation)
        
        # Convolution layer
        self.conv = nn.Conv(self.in_channels, 3, padding=1)
            
    @nn.compact
    def __call__(self, x):
        """
        Parameters:
        ----------
        x: torch.Tensor
            Input tensor of shape [batch, channels, *sizes]
        
        Returns:
        --------
        x: torch.Tensor
            Output tensor of shape [batch, channels, *sizes]
        """
        sizes = x.shape
        
        if self.bias:
            bias = x
        
        # Fourier layer
        x_ft = self.fourier(x)
        
        # MLP layer
        x_mlp = self.mlp(x)
        
        # Convolution layer
        
        x_conv = self.conv(x)
        
        # Add 
        x = x_ft + x_mlp + x_conv
        if self.bias:
            x = x + bias
        return x

