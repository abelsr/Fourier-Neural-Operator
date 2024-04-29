import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, dim: int, in_channels: int, out_channels: int, mid_channels: int, activation: nn.Module = nn.GELU()):
        """
        Multi-Layer Perceptron (MLP) module.

        Args:
            dim (int): The dimensionality of the input data. Can be 1, 2, or 3.
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            mid_channels (int): Number of channels in the intermediate layer.
            activation (torch.nn.Module, optional): Activation function to be applied after the first convolutional layer. 
                Defaults to `torch.nn.GELU()`.

        """
        super().__init__()
        self.dim = dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.activation = activation
        if self.dim == 2:
            self.mlp1 = nn.Conv2d(self.in_channels, self.mid_channels, 1)
            self.mlp2 = nn.Conv2d(self.mid_channels, self.out_channels, 1)
        elif self.dim == 3:
            self.mlp1 = nn.Conv3d(self.in_channels, self.mid_channels, 1)
            self.mlp2 = nn.Conv3d(self.mid_channels, self.out_channels, 1)
        else:
            self.mlp1 = nn.Conv1d(self.in_channels, self.mid_channels, 1)
            self.mlp2 = nn.Conv1d(self.mid_channels, self.out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, *spatial_dims).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, *spatial_dims).

        """
        x = self.mlp1(x)
        x = self.activation(x)
        x = self.mlp2(x)
        return x