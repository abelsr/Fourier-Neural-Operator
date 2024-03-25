import torch
import torch.nn as nn
from .layers import FourierBlock
import torch


class FNO(nn.Module):
    def __init__(self, modes, num_fourier_layers, in_channels, out_channels, mid_channels, activation, **kwargs):
        super().__init__()
        self.modes = modes
        self.dim = len(modes)
        self.num_fourier_layers = num_fourier_layers
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.activation = activation
        self.is_grid = kwargs.get('is_grid', False)
        self.sizes = [0] * self.dim
        
        # Lifting layer (P)
        self.p = nn.Linear(self.in_channels + (self.dim if self.is_grid else 0), self.in_channels)
        
        # Fourier blocks
        self.fourier_blocks = nn.ModuleList([
            FourierBlock(modes, in_channels, in_channels, out_channels, activation)
            for _ in range(num_fourier_layers)
        ])
        
        # Projection layer (Q)
        self.q = nn.Linear(self.in_channels, self.mid_channels)
        self.final = nn.Linear(self.mid_channels, self.out_channels)
        
    def forward(self, x):
        batch, channels, *sizes = x.size()
        
        # If grid is enabled, set the grid
        if self.is_grid:
            self.set_grid(x)
            for i in range(len(sizes)):
                if sizes[i] != self.sizes[i]:
                    break
            
            x = x.reshape(batch, *sizes, channels)
            reshapes = [1] * len(x.size())
            reshapes[0] = batch
            batched_grid = [grid.repeat(reshapes) for grid in self.grids]
            x = torch.cat((*batched_grid, x), dim=-1)
        
        
        # Lifting layer
        x = self.p(x)
        
        # Permute the dimensions [batch, *sizes, channels] -> [batch, channels, *sizes]
        x = x.permute(0, -1, *range(1, self.dim + 1))
        
        # Fourier blocks
        for fourier_block in self.fourier_blocks:
            x = fourier_block(x)
            
        # Permute the dimensions [batch, channels, *sizes] -> [batch, *sizes, channels]
        x = x.permute(0, *range(2, self.dim + 2), 1)
        
        # Projection layer
        x = self.q(x)
        
        # Activation
        x = self.activation(x)
        
        # Final layer
        x = self.final(x)
        
        return x.permute(0, -1, *range(1, self.dim + 1))
        
        
    def set_grid(self, x):
        batch, channels, *sizes = x.size()
        self.grids = []
        self.sizes = sizes
        
        # Create a grid
        for dim in range(self.dim):
            new_shape = [1] * (self.dim + 2)
            new_shape[dim + 1] = sizes[dim]
            repeats = [1] + sizes + [1]
            repeats[dim + 1] = 1
            grid = torch.linspace(0, 1, sizes[dim], requires_grad=True, device=x.device, dtype=torch.float).reshape(new_shape).repeat(repeats).clone().detach()
            self.grids.append(grid)
            
            
x = torch.randn(10, 2, 32, 32, 32)

model = FNO(modes=[13, 13, 12], num_fourier_layers=4, in_channels=2, out_channels=2, mid_channels=5, activation=nn.GELU(), is_grid=True)
out = model(x)
print(out.size())