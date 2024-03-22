"""
Implementation of Fourier Neural Operator for 2D Time-Dependent Problem using PyTorch.

Author: Abel Santillan Rodriguez
Date: 01/11/2023

Based on the paper: Fourier Neural Operator for Parametric Partial Differential Equations
"""
# Libraries
import torch
from torch import nn
from torch.nn import functional as F

# Spectral Convolution in Fourier Space (3D) [x, y, t]
class SpectralConv3D(nn.Module):
    """
    Spectral Convolution in Fourier Space (3D) [x, y, t]
    
    Parameters:
    -----------
    * in_channels: int - Number of input channels
    * out_channels: int - Number of output channels
    * modes: List[int] - Number of Fourier modes to multiply, default = 6 (2 modes per dimension) at most N/2 + 1
    """
    def __init__(self, in_channels, out_channels, modes=(6, 6, 6)):
        super(SpectralConv3D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # N/2 + 1 modes per dimension at most
        self.modes = modes
        
        # Scale used to initialize weights
        self.scale = 1 / (in_channels * out_channels)
        
        # Weights for different convolution operations in Fourier space
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes[0], modes[1], modes[2], dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes[0], modes[1], modes[2], dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes[0], modes[1], modes[2], dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes[0], modes[1], modes[2], dtype=torch.cfloat))

    def complex_multi_3D(self, inputs, weights):
        # Complex multiplication between inputs and weights
        return torch.einsum("bixyz,ioxyz->boxyz", inputs, weights)
    
    def forward(self, x):
        batchsize = x.shape[0]
        
        # Calculating the Fourier coefficients, i.e. applying FFT to x and swapping the order of dimensions
        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1])
        
        # Initialize output tensor in Fourier space with the following dimension [batch, out, x[-3], x[-2], x[-1]/2]
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        
        # Multiply the relevant Fourier modes with the corresponding weights
        out_ft[:, :, :self.modes[0],  :self.modes[1],  :self.modes[2]] = self.complex_multi_3D(x_ft[:, :, :self.modes[0],  :self.modes[1],  :self.modes[2]], self.weights1)
        out_ft[:, :, -self.modes[0]:, :self.modes[1],  :self.modes[2]] = self.complex_multi_3D(x_ft[:, :, -self.modes[0]:, :self.modes[1],  :self.modes[2]], self.weights2)
        out_ft[:, :, :self.modes[0],  -self.modes[1]:, :self.modes[2]] = self.complex_multi_3D(x_ft[:, :, :self.modes[0],  -self.modes[1]:, :self.modes[2]], self.weights3)
        out_ft[:, :, -self.modes[0]:, -self.modes[1]:, :self.modes[2]] = self.complex_multi_3D(x_ft[:, :, -self.modes[0]:, -self.modes[1]:, :self.modes[2]], self.weights4)
        
        # Calculate the inverse Fourier transform to get the final output in real space
        result = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return result
    
# Fourier Layer Block
class FourierLayerBlock(nn.Module):
    """
    Fourier Layer Block
    
    Parameters:
    -----------
    * modes: List[int] - Number of Fourier modes to multiply, default = 6 (2 modes per dimension)
    * layers: int - Number of layers in the block, default = 4
    """
    def __init__(self, modes=(6, 6, 6), width=4, norm=None):
        super(FourierLayerBlock, self).__init__()
        self.modes = modes
        self.width = width
        self.norm = norm
        
        # 3D Fourier Spectral Convolution Layer
        self.rama_superior = SpectralConv3D(self.width, self.width, self.modes)
        
        # 1D Convolution Layer
        self.rama_inferior = nn.Conv1d(self.width, self.width, 1)
        
        # Normalization Layer (e.g., BatchNorm3d or LayerNorm)
        if self.norm:
            self.norm = nn.BatchNorm3d(self.width)

    def forward(self, x, batchsize, size_x, size_y, size_z):
        # Forward pass in the upper branch (3D Fourier Spectral Convolution Layer)
        x_sup = self.rama_superior(x)
        
        # Forward pass in the lower branch (1D Convolution Layer) (W)
        x_inf = self.rama_inferior(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        
        # Sum the outputs of both branches and apply a ReLU activation function
        x = x_sup + x_inf # (+)
        x = F.relu(x) # (𝜎)
        if self.norm:
            x = self.norm(x)  # Normalization
        return x
    
# Fourier Neural Operator 2D Time-Dependent
class FNO2DTime(nn.Module):
    """
    Fourier Neural Operator 2D Time-Dependent
    
    Parameters:
    -----------
    * modes: List[int] - Number of Fourier modes to multiply, default = 6 (2 modes per dimension)
    * width: int - Number of channels in the hidden layers, default = 10
    * layers: int - Number of layers in the block, default = 4
    * ti: int - Number of input features, default = 10
    """
    def __init__(self, modes=(6, 6, 6), width=10, layers=4, channels=10, padding=None, dropout=None, norm=None):
        """
        Fourier Neural Operator 2D Time-Dependent
        
        Parameters:
        -----------
        * modes: List[int] - Number of Fourier modes to multiply, default = 6 (2 modes per dimension)
        * width: int - Number of channels in the hidden layers, default = 10
        * layers: int - Number of layers in the block, default = 4
        * channels: int - Number of input channels, default = 10
        * padding: int - Padding size, default = 0
        """
        super(FNO2DTime, self).__init__()
        self.modes = modes
        self.width = width
        self.layers = layers
        self.padding = padding
        self.size_x = 0
        self.size_y = 0
        self.size_t = 0
        self.dropout = dropout
        self.channels = channels
        
        # Input Layer (P) [batch, in, x, y, t]
        self.fc0 = nn.Linear(self.channels+3, self.width)
        
        # Array of Fourier Layer Blocks
        self.module = nn.ModuleList()
        for _ in range(self.layers):
            self.module.append(FourierLayerBlock(self.modes, self.width, norm))
            
        # Linear Layers (Q)
        self.fc1 = nn.Linear(self.width, 128)
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(128, 1)
    
    def __repr__(self):
        return f"{__class__.__name__}(modes={self.modes}, width={self.width}, layers={self.layers}, channels={self.channels}, padding={self.padding}, dropout={self.dropout}, norm={self.norm})"

    def forward(self, x):
        batchsize, size_x, size_y, size_t = x.shape
        
        # Check modes (N/2 + 1 modes per dimension at most)
        if self.padding:
            assert self.modes[0] <= (size_x+self.padding[0])//2 + 1, "Error: The number of modes in x is too large"
            assert self.modes[1] <= (size_y+self.padding[1])//2 + 1, "Error: The number of modes in y is too large"
            assert self.modes[2] <= (size_t+self.padding[2])//2 + 1, "Error: The number of modes in t is too large"
        else:
            assert self.modes[0] <= size_x//2 + 1, "Error: The number of modes in x is too large"
            assert self.modes[1] <= size_y//2 + 1, "Error: The number of modes in y is too large"
            assert self.modes[2] <= size_t//2 + 1, "Error: The number of modes in t is too large"

        if self.size_x != size_x or self.size_y != size_y or self.size_t != size_t:
            self.set_grid(x)
        x = x.reshape(batchsize, size_x, size_y, size_t, 1).repeat([1, 1, 1, 1, self.channels])
        x = torch.cat((
                        self.gridx.repeat([batchsize, 1, 1, 1, 1]).clone(),
                        self.gridy.repeat([batchsize, 1, 1, 1, 1]).clone(),
                        self.gridt.repeat([batchsize, 1, 1, 1, 1]).clone(),
                        x),
                        dim=-1)
        _, size_x, size_y, size_z, _ = x.shape
        
        # Forward pass through the input layer (P)
        x = self.fc0(x)
        
        # Swap the order of dimensions to [batch, channels, x, y, t]
        x = x.permute(0, 4, 1, 2, 3)
        
        # Add padding to the input tensor [batch, channels, x+padding, y+padding, t+padding]
        if self.padding:
            x = F.pad(x, (0, 0, 0, self.padding[0], self.padding[1], 0))
            x = F.pad(x, (0,self.padding[2]), "constant", 0)
        
        
        # Forward pass through the Fourier Layer Blocks
        for mod in self.module:
            batchsize, _, size_x, size_y, size_z = x.shape
            x = mod(x, batchsize, size_x, size_y, size_z)
        
        # Linear Layers (Q) and ReLU activation function
        
        # Remove padding
        if self.padding:
            x = x[:, :, :-self.padding[0], :-self.padding[1], :-self.padding[2]]
        
        # Swap the order of dimensions to [batch, in, x, y, t]
        x = x.permute(0, 2, 3, 4, 1)
        
        x = self.fc1(x)
        x = F.gelu(x)
        if self.dropout:
            x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)

        return x
    
    def set_grid(self, x):
        _, self.size_x, self.size_y, self.size_t = x.shape
        self.gridx = torch.linspace(0, 1, self.size_x, dtype=torch.float, device=x.device).reshape(1, self.size_x, 1, 1, 1).repeat([1, 1, self.size_y, self.size_t, 1]).clone().detach().requires_grad_(True)
        self.gridy = torch.linspace(0, 1, self.size_y, dtype=torch.float, device=x.device).reshape(1, 1, self.size_y, 1, 1).repeat([1, self.size_x, 1, self.size_t, 1]).clone().detach().requires_grad_(True)
        self.gridt = torch.linspace(0, 1, self.size_t+1, dtype=torch.float, device=x.device)[1:].reshape(1, 1, 1, self.size_t, 1).repeat([1, self.size_x, self.size_y, 1, 1]).clone().detach().requires_grad_(True)