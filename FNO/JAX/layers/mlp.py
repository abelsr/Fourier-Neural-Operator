import jax.numpy as jnp
from flax import linen as nn
from jax import random
from typing import Callable

class MLP(nn.Module):
    dim: int
    in_channels: int
    out_channels: int
    mid_channels: int
    activation: Callable = nn.gelu
    
    def setup(self):
        kernel_size = (1,) * self.dim
        # if self.dim == 2:
        #     self.mlp1 = nn.Conv(self.in_channels, kernel_size=(1, 1))
        #     self.mlp2 = nn.Conv(self.mid_channels, kernel_size=(1, 1))
        #     self.mlp3 = nn.Conv(self.out_channels, kernel_size=(1, 1))
        # elif self.dim == 3:
        #     self.mlp1 = nn.Conv(self.in_channels, kernel_size=(1, 1, 1))
        #     self.mlp2 = nn.Conv(self.mid_channels, kernel_size=(1, 1, 1))
        #     self.mlp3 = nn.Conv(self.out_channels, kernel_size=(1, 1, 1))
        # else:
        self.mlp1 = nn.Conv(self.in_channels, kernel_size=kernel_size)
        self.mlp2 = nn.Conv(self.mid_channels, kernel_size=kernel_size)
        self.mlp3 = nn.Conv(self.out_channels, kernel_size=kernel_size)

    # @nn.compact
    def __call__(self, x):
        # Permute dimensions to (B, C, x1, x2, ..., xN) -> (B, x1, x2, ..., xN, C)
        x = jnp.permute_dims(x, (0, *range(2, 2 + self.dim), 1))
        x = self.mlp1(x)
        x = self.activation(x)
        x = self.mlp2(x)
        x = self.activation(x)
        x = self.mlp3(x)
        # Permute dimensions back to (B, x1, x2, ..., xN, C) -> (B, C, x1, x2, ..., xN)
        x = jnp.permute_dims(x, (0, self.dim + 1, *range(1, 1 + self.dim)))
        return x


# Example usage
# key = random.PRNGKey(0)
# x = random.normal(key, (1, 16, 32, 32))
# model = MLP(dim=2, in_channels=16, out_channels=1, mid_channels=64)
# params = model.init(key, x)
# y = model.apply(params, x)
# print(x.shape, "->", y.shape)

