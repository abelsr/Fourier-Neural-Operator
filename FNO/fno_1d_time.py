"""
Implementation of Fourier Neural Operator for 1D Time-Dependent Problem using PyTorch.

Author: Abel Santillan Rodriguez
Date: 01/11/2023

Based on the paper: Fourier Neural Operator for Parametric Partial Differential Equations
"""
# Libraries
import torch
from torch import nn
from torch.nn import functional as F


# Spectral Convolution in Fourier Space (2D) [x, t]
class SpectralConv2D(nn.Module):
    """
    Spectral Convolution in Fourier Space (2D) [x, t]
    
    Parameters:
    -----------
    * in_channels: int - Number of input channels
    * out_channels: int - Number of output channels
    * modes: int - Number of Fourier modes to multiply, default = 4 (2 modes per dimension)
    """
    def __init__(self, in_channels, out_channels, modes=4):
        super(SpectralConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 2 modes per dimension max, i.e. 4 modes for 2D
        self.modes = modes
        
        # Scale used to initialize weights
        self.scale = 1 / (in_channels * out_channels)
        
        # Weights for different convolution operations in Fourier space
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes, modes,
                                                             modes, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes, modes,
                                                             modes, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes, modes,
                                                             modes, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes, modes,
                                                             modes, dtype=torch.cfloat))

    def complex_multi_2D(self, inputs, weights):
        # Complex multiplication between inputs and weights
        return torch.einsum("bixy,ioxy->boxy", inputs, weights)
    
    def forward(self, x):
        batchsize = x.shape[0]
        
        # Calculating the Fourier coefficients, i.e. applying FFT to x and swapping the order of dimensions
        x_ft = torch.fft.rfftn(x, dim=[-2,-1])
        
        # Initialize output tensor in Fourier space with the following dimension [batch, out, x[-2], x[-1]/2]
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        
        # Multiply the relevant Fourier modes with the corresponding weights
        out_ft[:, :, :self.modes, :self.modes] = self.complex_multi_2D(x_ft[:, :, :self.modes, :self.modes], self.weights1)
        
        out_ft[:, :, -self.modes:, :self.modes] = self.complex_multi_2D(x_ft[:, :, -self.modes:, :self.modes], self.weights2)
        
        out_ft[:, :, :self.modes, -self.modes:] = self.complex_multi_2D(x_ft[:, :, :self.modes, -self.modes:], self.weights3)

        out_ft[:, :, -self.modes:, -self.modes:] = self.complex_multi_2D(x_ft[:, :, -self.modes:, -self.modes:], self.weights4)
        
        
        # Calculate the inverse Fourier transform to get the final output in real space
        result = torch.fft.irfftn(out_ft, s=(x.size(-2), x.size(-1)))
        return result
    
# Fourier Layer Block
class FourierLayerBlock(nn.Module):
    """
    Fourier Layer Block
    
    Parameters:
    -----------
    * modes: int - Number of Fourier modes to multiply, default = 6 (2 modes per dimension)
    * layers: int - Number of layers in the block, default = 4
    """
    def __init__(self, modes=4, layers=4):
        super(FourierLayerBlock, self).__init__()
        self.modes = modes
        self.layers = layers
        
        # 3D Fourier Spectral Convolution Layer
        self.rama_superior = SpectralConv2D(self.width, self.width, self.modes)
        
        # 1D Convolution Layer
        self.rama_inferior = nn.Conv1d(self.width, self.width, 1)

    def forward(self, x, batchsize, size_x, size_y):
        # Forward pass in the upper branch (3D Fourier Spectral Convolution Layer)
        x_sup = self.rama_superior(x)
        
        # Forward pass in the lower branch (1D Convolution Layer) (W)
        x_inf = self.rama_inferior(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        
        # Sum the outputs of both branches and apply a ReLU activation function
        x = x_sup + x_inf # (+)
        x = F.relu(x) # (𝜎)
        return x

# Fourier Neural Operator 1D Time-Dependent
class FNO1DTime(nn.Module):
    """
    Fourier Neural Operator 2D Time-Dependent
    
    Parameters:
    -----------
    * modes: int - Number of Fourier modes to multiply, default = 4 (2 modes per dimension)
    * width: int - Number of channels in the hidden layers, default = 10
    * layers: int - Number of layers in the block, default = 4
    """
    def __init__(self, modes=6, width=10, layers=4):
        super(FNO1DTime, self).__init__()
        self.modes = modes
        self.width = width
        self.layers = layers
        self.size_x = 0
        self.size_t = 0
        
        # Input Layer (P) [batch, in, x, y, t]
        self.fc0 = nn.Linear(self.size_t + 2, self.width)
        
        # Array of Fourier Layer Blocks
        self.module = nn.ModuleList()
        for _ in range(self.layers):
            self.module.append(FourierLayerBlock(self.modes, self.width))
            
        # Linear Layers (Q)
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        batchsize, size_x, size_t = x.shape
        if self.size_x != size_x or self.size_t != size_t:
            self.set_grid(x)
        x = x.reshape(batchsize, size_x, 1, size_t).repeat([1, 1, self.size_t, 1])
        x = torch.cat((
                        self.gridx.repeat([batchsize, 1, 1, 1, 1]),
                        self.gridt.repeat([batchsize, 1, 1, 1, 1]),
                        x),
                        dim=-1)
        _, size_x, size_y, _ = x.shape
        
        # Forward pass through the input layer (P)
        x = self.fc0(x)
        
        # Swap the order of dimensions to [batch, in, x, t]
        x = x.permute(0, 3, 1, 2)
        
        # Forward pass through the Fourier Layer Blocks
        for mod in self.module:
            x = mod(x, batchsize, size_x, size_y)
        
        # Linear Layers (Q) and ReLU activation function
        
        # Swap the order of dimensions to [batch, in, x, t]
        x = x.permute(0, 2, 3, 1)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x
    
    def set_grid(self, x):
        _, self.size_x, self.size_t = x.shape
        self.gridx = torch.tensor(torch.linspace(0, 1, self.Sx), dtype=torch.float, device=x.device).reshape(1, self.Sx, 1, 1, 1).repeat([1, 1, self.Sy, self.T, 1])
        self.gridt = torch.tensor(torch.linspace(0, 1, self.T+1)[1:], dtype=torch.float, device=x.device).reshape(1, 1, 1, self.T, 1).repeat([1, self.Sx, self.Sy, 1, 1])