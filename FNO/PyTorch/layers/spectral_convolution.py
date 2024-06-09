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
        
        # Weights for real and imaginary parts separately
        self.weights_real = nn.ParameterList([
            nn.Parameter(self.scale * torch.ones(in_channels, out_channels, *self.modes, dtype=torch.float))
            for _ in range(self.mix_matrix.shape[0] - 1)
        ])
        self.weights_imag = nn.ParameterList([
            nn.Parameter(self.scale * torch.ones(in_channels, out_channels, *self.modes, dtype=torch.float))
            for _ in range(self.mix_matrix.shape[0] - 1)
        ])
        
    @staticmethod
    def complex_mult(input_real: torch.Tensor, input_imag: torch.Tensor, weights_real: torch.Tensor, weights_imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform complex multiplication between input and weights.

        Args:
            input_real (torch.Tensor): Real part of input tensor. [batch_size, in_channels, *sizes]
            input_imag (torch.Tensor): Imaginary part of input tensor. [batch_size, in_channels, *sizes]
            weights_real (torch.Tensor): Real part of weights tensor. [in_channels, out_channels, *sizes]
            weights_imag (torch.Tensor): Imaginary part of weights tensor. [in_channels, out_channels, *sizes]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Real and imaginary parts of the result. [batch_size, out_channels, *sizes]
        """
        out_real = torch.einsum('bi...,io...->bo...', input_real, weights_real) - torch.einsum('bi...,io...->bo...', input_imag, weights_imag)
        out_imag = torch.einsum('bi...,io...->bo...', input_real, weights_imag) + torch.einsum('bi...,io...->bo...', input_imag, weights_real)
        return out_real, out_imag
    
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
    
    def mix_weights(self, out_ft_real: torch.Tensor, out_ft_imag: torch.Tensor, x_ft_real: torch.Tensor, x_ft_imag: torch.Tensor, weights_real: List[torch.Tensor], weights_imag: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mix the weights for spectral convolution.

        Args:
            out_ft_real (torch.Tensor): Real part of output tensor in Fourier space.
            out_ft_imag (torch.Tensor): Imaginary part of output tensor in Fourier space.
            x_ft_real (torch.Tensor): Real part of input tensor in Fourier space.
            x_ft_imag (torch.Tensor): Imaginary part of input tensor in Fourier space.
            weights_real (List[torch.Tensor]): List of real part weights tensors.
            weights_imag (List[torch.Tensor]): List of imaginary part weights tensors.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Mixed weights tensor (real and imaginary parts).
        """
        slices = tuple(slice(None, mode) for mode in self.modes)
        
        # Mixing weights
        
        # First weight
        out_ft_real[(Ellipsis,) + slices], out_ft_imag[(Ellipsis,) + slices] = self.complex_mult(
            x_ft_real[(Ellipsis,) + slices], x_ft_imag[(Ellipsis,) + slices], weights_real[0], weights_imag[0]
        )
        
        if len(weights_real) == 1:
            return out_ft_real, out_ft_imag
        
        # Rest of the weights
        for i in range(1, len(weights_real)):
            modes = self.mix_matrix[i].squeeze().tolist()
            slices = tuple(slice(-mode, None) if sign < 0 else slice(None, mode) for sign, mode in zip(modes, self.modes))
            out_ft_real[(Ellipsis,) + slices], out_ft_imag[(Ellipsis,) + slices] = self.complex_mult(
                x_ft_real[(Ellipsis,) + slices], x_ft_imag[(Ellipsis,) + slices], weights_real[i], weights_imag[i]
            )
        
        return out_ft_real, out_ft_imag
        
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
        
        # Split into real and imaginary parts
        x_ft_real, x_ft_imag = x_ft.real, x_ft.imag
        
        # Initialize output
        out_ft_real = torch.zeros(batch_size, self.out_channels, *sizes, dtype=torch.float, device=x.device)
        out_ft_imag = torch.zeros(batch_size, self.out_channels, *sizes, dtype=torch.float, device=x.device)
        # Reduce the last dimension to x.shape[-1]//2+1
        out_ft_real = out_ft_real[..., :x.shape[-1]//2+1]
        out_ft_imag = out_ft_imag[..., :x.shape[-1]//2+1]
        
        # Mixing weights
        out_ft_real, out_ft_imag = self.mix_weights(out_ft_real, out_ft_imag, x_ft_real, x_ft_imag, self.weights_real, self.weights_imag)
        
        # Combine real and imaginary parts back into a complex tensor
        out_ft = torch.complex(out_ft_real, out_ft_imag)
        
        # Inverse Fourier transform to real space
        out = torch.fft.irfftn(out_ft, dim=tuple(range(-self.dim, 0)), s=sizes)
        
        return out