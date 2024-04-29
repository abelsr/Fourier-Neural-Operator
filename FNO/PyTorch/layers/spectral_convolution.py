import torch
import torch.nn as nn
from typing import List, Tuple


class SpectralConvolution(nn.Module):
    """
    Spectral Convolution layer implementation.
    """
    def __init__(self, in_channels: int, out_channels: int, modes: List[int]):
        """
        Initialize SpectralConvolution layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            modes (List[int]): List of modes for spectral convolution.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.dim = len(self.modes)
        self.mix_matrix = self.get_mix_matrix(self.dim)
        
        # Scale factor for weights
        self.scale = 1 / (in_channels * out_channels)
        
        # Weights
        self.weights = nn.ParameterList([
            nn.Parameter(self.scale * torch.ones(in_channels, out_channels, *self.modes, dtype=torch.cfloat))
            for _ in range(2 ** (self.dim - 1))
        ])
        
    @staticmethod
    def complex_mult(input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Perform complex multiplication between input and weights.

        Args:
            input (torch.Tensor): Input tensor. [batch_size, in_channels, *sizes]
            weights (torch.Tensor): Weights tensor. [in_channels, out_channels, *sizes]

        Returns:
            torch.Tensor: Result of complex multiplication. [batch_size, out_channels, *sizes]
        """
        return torch.einsum('bi...,io...->bo...', input, weights)
    
    @staticmethod
    def get_mix_matrix(dim: int) -> torch.Tensor:
        """
        Generate a mixing matrix for spectral convolution.

        Args:
            dim (int): Dimension of the mixing matrix.

        Returns:
            torch.Tensor: Mixing matrix.
        """
        # Create a lower triangular matrix with -1 in the diagonal and 1 in the rest
        mix_matrix = torch.tril(torch.ones((dim, dim), dtype=torch.float)) - 2*torch.eye(dim, dtype=torch.float)
        
        # Subtract -2 to the last row
        mix_matrix[-1] = mix_matrix[-1] - 2
        
        # The last element of the last row is 1
        mix_matrix[-1, -1] = 1
        
        # The zeros of the mix matrix are converted to 1
        mix_matrix[mix_matrix == 0] = 1
    
        # Add a row of ones at the beginning
        mix_matrix = torch.cat((torch.ones((1, dim), dtype=torch.float), mix_matrix), dim=0)
        
        return mix_matrix
    
    def mix_weights(self, out_ft: torch.Tensor, x_ft: torch.Tensor, weights: List[torch.Tensor]) -> torch.Tensor:
        """
        Mix the weights for spectral convolution.

        Args:
            out_ft (torch.Tensor): Output tensor in Fourier space.
            x_ft (torch.Tensor): Input tensor in Fourier space.
            weights (List[torch.Tensor]): List of weights tensors.

        Returns:
            torch.Tensor: Mixed weights tensor.
        """
        slices = tuple(slice(None, mode) for mode in self.modes)
        
        # Mixing weights
        
        # First weight
        out_ft[(Ellipsis,) + slices] = self.complex_mult(x_ft[(Ellipsis,) + slices], weights[0])
        
        if len(weights) == 1:
            return out_ft
        
        # Rest of the weights
        for i in range(1, len(weights)):
            modes = self.mix_matrix[i].squeeze().tolist()
            slices = tuple(slice(-mode, None) if sign < 0 else slice(None, mode) for sign, mode in zip(modes, self.modes))
            out_ft[(Ellipsis,) + slices] = self.complex_mult(x_ft[(Ellipsis,) + slices], weights[i])
        
        return out_ft
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SpectralConvolution layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        batch_size, _, *sizes = x.shape
        
        # Fourier transform
        x_ft = torch.fft.fftn(x, dim=tuple(range(-self.dim, 0)))
        
        # Initialize output
        out_ft = torch.zeros(batch_size, self.out_channels, *sizes, dtype=torch.cfloat, device=x.device)
        # Reduce the last dimension to x.shape[-1]//2+1
        out_ft = out_ft[..., :x.shape[-1]//2+1]
        
        # Mixing weights
        out_ft = self.mix_weights(out_ft, x_ft, self.weights)
        
        # Inverse Fourier transform to real space
        out = torch.fft.irfftn(out_ft, dim=tuple(range(-self.dim, 0)), s=sizes)
        
        return out