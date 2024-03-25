import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, dim, in_channels, out_channels, mid_channels, activation=nn.GELU()):
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

    def forward(self, x):
        x = self.mlp1(x)
        x = self.activation(x)
        x = self.mlp2(x)
        return x