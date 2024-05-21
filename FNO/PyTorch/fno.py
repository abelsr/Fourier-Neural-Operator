import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import FourierBlock
from typing import List

class FNO(nn.Module):
    """
    FNO (Fourier Neural Operator) model for solving PDEs using deep learning.
    """
    def __init__(self, modes: List[int], num_fourier_layers: int, in_channels: int, lifting_channels: int, projection_channels:int, out_channels: int, mid_channels: int, activation: nn.Module, **kwargs: bool):
        """
        Initialize the FNO model.

        Args:
            modes (List[int]): List of integers representing the number of Fourier modes along each dimension.
            num_fourier_layers (int): Number of Fourier blocks to use in the model.
            in_channels (int): Number of input channels.
            lifting_channels (int): Number of channels in the lifting layer.
            out_channels (int): Number of output channels.
            mid_channels (int): Number of channels in the intermediate layers.
            activation (nn.Module): Activation function to use.
            **kwargs (bool): Additional keyword arguments.

        Keyword Args:
            add_grid (bool): Whether to use grid information in the model.
            padding (List[int]): Padding to apply to the input tensor. [pad_dim1, pad_dim2, ...]
        """
        super().__init__()
        self.modes = modes
        self.dim = len(modes)
        self.num_fourier_layers = num_fourier_layers
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.lifting_channels = lifting_channels
        self.projection_channels = projection_channels
        self.activation = activation
        self.add_grid = kwargs.get('add_grid', False)
        self.padding = kwargs.get('padding', None)
        self.sizes = [0] * self.dim
        
        
        # Format the padding
        if self.padding is not None:
            # Padd is a list of integers representing the padding along each dimension, so we need to convert it to a tuple
            self.padding = [(0, 0), (0, 0)] + [(p, p) for p in self.padding]
            # Flatten the padding
            self.padding = sum(self.padding, ())
            # Slice for removing padding [:, :, padding[0]:-padding[1], padding[2]:-padding[3],...]
            self.slice = tuple(slice(p, -p) if p > 0 else slice(None) for p in self.padding[2::2])
            
            

        # Lifting layer (P)
        if self.lifting_channels is not None:
            self.p1 = nn.Linear(self.in_channels + (self.dim if self.add_grid else 0), self.lifting_channels)
            self.p2 = nn.Linear(self.lifting_channels, self.mid_channels)
        else:
            self.p1 = nn.Linear(self.in_channels + (self.dim if self.add_grid else 0), self.mid_channels)
        

        # Fourier blocks
        self.fourier_blocks = nn.ModuleList([
            FourierBlock(modes, mid_channels, mid_channels, activation=activation)
            for _ in range(num_fourier_layers)
        ])

        # Projection layer (Q)
        self.q1 = nn.Linear(self.mid_channels,self.projection_channels)
        self.final = nn.Linear(self.projection_channels, self.out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the FNO model.

        Args:
            x (torch.Tensor): Input tensor. [batch, channels, *sizes]

        Returns:
            torch.Tensor: Output tensor. [batch, channels, *sizes]
        """
        batch, in_channels, *sizes = x.size()
        assert len(sizes) == self.dim, "Input tensor must have the same number of dimensions as the number of modes. Got {} dimensions, expected {}.".format(len(sizes), self.dim)
        
        # Permute the dimensions [batch, channels, *sizes] -> [batch, *sizes, channels]
        x = x.permute(0, *range(2, self.dim + 2), 1)

        # If grid is enabled, set the grid
        if self.add_grid:
            for i in range(len(sizes)):
                if sizes[i] != self.sizes[i] or self.grids[0].shape[0] != batch:
                    self.set_grid(x)
                    break
            x = torch.cat((x, self.grids), dim=-1)

        # Lifting layer
        x = self.p1(x)
        if self.lifting_channels is not None:
            x = self.p2(x)

        # Permute the dimensions [batch, *sizes, channels] -> [batch, channels, *sizes]
        x = x.permute(0, -1, *range(1, self.dim + 1))
        
        # Pad the input tensor
        if self.padding is not None:
            x = F.pad(x, self.padding[::-1])

        # Fourier blocks
        for fourier_block in self.fourier_blocks:
            x = fourier_block(x)
            
        # Remove padding
        if self.padding is not None:
            x = x[(Ellipsis,) + tuple(self.slice)]

        # Permute the dimensions [batch, channels, *sizes] -> [batch, *sizes, channels]
        x = x.permute(0, *range(2, self.dim + 2), 1)

        # Projection layer
        x = self.q1(x)

        # Activation
        x = self.activation(x)

        # Final layer
        x = self.final(x)

        return x.permute(0, -1, *range(1, self.dim + 1))

    def set_grid(self, x: torch.Tensor) -> None:
        """
        Set the grid information for the FNO model.

        Args:
            x (torch.Tensor): Input tensor.

        """
        batch, *sizes, in_channels = x.size()
        self.grids = []
        self.sizes = sizes

        # Create a grid
        for dim in range(self.dim):
            new_shape = [1] * (self.dim + 2)
            new_shape[dim + 1] = sizes[dim]
            repeats = [1] + sizes + [1]
            repeats[dim + 1] = 1
            repeats[0] = batch
            grid = torch.linspace(0, 1, sizes[dim], device=x.device, dtype=torch.float).reshape(*new_shape).repeat(repeats)
            self.grids.append(grid)
        
        self.grids = torch.cat(self.grids, dim=-1)
        
