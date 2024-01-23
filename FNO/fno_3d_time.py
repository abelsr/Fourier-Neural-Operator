"""
Implementation of Fourier Neural Operator for 3D Time-Dependent Problem using PyTorch.

Author: Abel Santillan Rodriguez
Date: 01/11/2023

Based on the paper: Fourier Neural Operator for Parametric Partial Differential Equations
"""
# Libraries
import torch
from torch import nn
from torch.nn import functional as F

# Spectral Convolution in Fourier Space (4D) [x, y, z, t]
class SpectralConv4D(nn.Module):
    """
    Spectral Convolution in Fourier Space (4D) [x, y, z, t]
    
    Parameters:
    -----------
    * in_channels: int - Number of input channels
    * out_channels: int - Number of output channels
    * modes: int - Number of Fourier modes to multiply, default = 6 (2 modes per dimension)
    """
    def __init__(self, in_channels, out_channels, modes=6):
        super(SpectralConv4D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 2 modes per dimension max, i.e. 6 modes for 3D
        self.modes = modes
        
        # Scale used to initialize weights
        self.scale = 1 / (in_channels * out_channels)
        
        # Weights for different convolution operations in Fourier space
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes, modes,
                                                             modes, modes, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes, modes,
                                                             modes, modes, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes, modes,
                                                             modes, modes, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes, modes,
                                                             modes, modes, dtype=torch.cfloat))

    def complex_multi_4D(self, inputs, weights):
        # Complex multiplication between inputs and weights
        return torch.einsum("bixyzt,ioxyzt->boxyzt", inputs, weights)
    
    def forward(self, x):
        batchsize = x.shape[0]
        
        # Calculating the Fourier coefficients, i.e., applying FFT to x and swapping the order of dimensions
        x_ft = torch.fft.fftn(x, dim=[-4, -3, -2, -1])
        
        # Initialize output tensor in Fourier space with the following dimension [batch, out, x[-4], x[-3], x[-2], x[-1]]
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-4), x.size(-3), x.size(-2), x.size(-1), dtype=torch.cfloat, device=x.device)
        
        # Multiply the relevant Fourier modes with the corresponding weights
        out_ft[:, :, :self.modes, :self.modes, :self.modes, :self.modes] = self.complex_multi_4D(x_ft[:, :, :self.modes, :self.modes, :self.modes, :self.modes], self.weights1)
        
        out_ft[:, :, -self.modes:, :self.modes, :self.modes, :self.modes] = self.complex_multi_4D(x_ft[:, :, -self.modes:, :self.modes, :self.modes, :self.modes], self.weights2)
        
        out_ft[:, :, :self.modes, -self.modes:, :self.modes, :self.modes] = self.complex_multi_4D(x_ft[:, :, :self.modes, -self.modes:, :self.modes, :self.modes], self.weights3)

        out_ft[:, :, -self.modes:, -self.modes:, :self.modes, :self.modes] = self.complex_multi_4D(x_ft[:, :, -self.modes:, -self.modes:, :self.modes, :self.modes], self.weights4)
        
        # Calculate the inverse Fourier transform to get the final output in real space
        result = torch.fft.ifftn(out_ft, s=(x.size(-4), x.size(-3), x.size(-2), x.size(-1)))
        return result

# Fourier Layer Block for 3D Time
class FourierLayerBlock3DTime(nn.Module):
    """
    Fourier Layer Block for 3D Time
    
    Parameters:
    -----------
    * modes: int - Number of Fourier modes to multiply, default = 6 (2 modes per dimension)
    * width: int - Number of channels in the hidden layers, default = 4
    """
    def __init__(self, modes=6, width=4):
        super(FourierLayerBlock3DTime, self).__init__()
        self.modes = modes
        self.width = width
        
        # 4D Fourier Spectral Convolution Layer
        self.rama_superior = SpectralConv4D(self.width, self.width, self.modes)
        
        # 1D Convolution Layer
        self.rama_inferior = nn.Conv1d(self.width, self.width, 1)

    def forward(self, x, batchsize, size_x, size_y, size_z, size_t):
        # Forward pass in the upper branch (4D Fourier Spectral Convolution Layer)
        x_sup = self.rama_superior(x)
        
        # Forward pass in the lower branch (1D Convolution Layer) (W)
        x_inf = self.rama_inferior(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z, size_t)
        
        # Sum the outputs of both branches and apply a ReLU activation function
        x = x_sup + x_inf # (+)
        x = F.relu(x) # (𝜎)
        return x

# Fourier Neural Operator 3D Time-Dependent
class FNO3DTime(nn.Module):
    """
    Fourier Neural Operator 3D Time-Dependent
    
    Parameters:
    -----------
    * modes: int - Number of Fourier modes to multiply, default = 6 (2 modes per dimension)
    * width: int - Number of channels in the hidden layers, default = 10
    * layers: int - Number of layers in the block, default = 4
    """
    def __init__(self, modes=6, width=10, layers=4, ti=10):
        super(FNO3DTime, self).__init__()
        self.modes = modes
        self.width = width
        self.layers = layers
        self.size_x = 0
        self.size_y = 0
        self.size_z = 0
        self.size_t = 0
        
        # Input Layer (P) [batch, in, x, y, z, t]
        self.fc0 = nn.Linear(ti+4, self.width)
        
        # Array of Fourier Layer Blocks for 3D Time
        self.module = nn.ModuleList()
        for _ in range(self.layers):
            self.module.append(FourierLayerBlock3DTime(self.modes, self.width))
            
        # Linear Layers (Q)
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        batchsize, size_x, size_y, size_z, size_t = x.shape
        if self.size_x != size_x or self.size_y != size_y or self.size_z != size_z or self.size_t != size_t:
            self.set_grid(x)
        x = x.reshape(batchsize, size_x, size_y, size_z, 1, size_t).repeat([1, 1, 1, 1, self.size_t, 1])
        x = torch.cat((
                        self.gridx.repeat([batchsize, 1, 1, 1, 1, 1]).clone(),
                        self.gridy.repeat([batchsize, 1, 1, 1, 1, 1]).clone(),
                        self.gridz.repeat([batchsize, 1, 1, 1, 1, 1]).clone(),
                        self.gridt.repeat([batchsize, 1, 1, 1, 1, 1]).clone(),
                        x),
                        dim=-1)
        _, size_x, size_y, size_z, size_w, _ = x.shape
        
        # Forward pass through the input layer (P)
        x = self.fc0(x)
        
        # Swap the order of dimensions to [batch, in, x, y, z, in]
        x = x.permute(0, 5, 1, 2, 3, 4)
        
        # Forward pass through the Fourier Layer Blocks for 3D Time
        for mod in self.module:
            x = mod(x, batchsize, size_x, size_y, size_z, size_w)
        
        # Linear Layers (Q) and ReLU activation function
        
        # Swap the order of dimensions to [batch, in, x, y, z, t]
        x = x.permute(0, 2, 3, 4, 5, 1)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x
    
    def set_grid(self, x):
        _, self.size_x, self.size_y, self.size_z, self.size_t = x.shape
        self.gridx = torch.linspace(0, 1, self.size_x, dtype=torch.float, device=x.device).reshape(1, self.size_x, 1, 1, 1, 1).repeat([1, 1, self.size_y, self.size_z, self.size_t, 1]).clone().detach().requires_grad_(True)
        self.gridy = torch.linspace(0, 1, self.size_y, dtype=torch.float, device=x.device).reshape(1, 1, self.size_y, 1, 1, 1).repeat([1, self.size_x, 1, self.size_z, self.size_t, 1]).clone().detach().requires_grad_(True)
        self.gridz = torch.linspace(0, 1, self.size_z, dtype=torch.float, device=x.device).reshape(1, 1, 1, self.size_z, 1, 1).repeat([1, self.size_x, self.size_y, 1, self.size_t, 1]).clone().detach().requires_grad_(True)
        self.gridt = torch.linspace(0, 1, self.size_t+1, dtype=torch.float, device=x.device)[1:].reshape(1, 1, 1, 1, self.size_t, 1).repeat([1, self.size_x, self.size_y, self.size_z, 1, 1]).clone().detach().requires_grad_(True)
