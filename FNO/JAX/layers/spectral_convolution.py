import jax
from jax import random, numpy as jnp
from flax import linen as nn
from jax import random, jit, vmap
from flax import linen as nn

class SpectralConvolution(nn.Module):
    in_channels: int
    out_channels: int
    modes: list
    
    def setup(self):
        self.dim = len(self.modes)
        self.mix_matrix = self.get_mix_matrix(self.dim)
        
        # Scale factor for weights
        self.scale = 1 / (self.in_channels * self.out_channels)
        
        # Weights
        self.weights = [
            jnp.array(self.scale * jnp.ones((self.in_channels, self.out_channels, *self.modes)), dtype=jnp.complex64)
            for _ in range(len(self.mix_matrix + 1))
        ]
        
    @staticmethod
    def complex_mult(input, weights):
        return jnp.einsum('bi...,io...->bo...', input, weights)
    
    @staticmethod
    def get_mix_matrix(dim):
        # Create a lower triangular matrix with -1 in the diagonal and 1 in the rest
        mix_matrix = jnp.tril(jnp.ones((dim, dim), dtype=jnp.float32)) - 2*jnp.eye(dim, dtype=jnp.float32)
        
        # Subtract -2 to the last row
        mix_matrix = mix_matrix.at[-1].set(mix_matrix[-1] - 2)
        
        # The last element of the last row is 1
        mix_matrix = mix_matrix.at[-1, -1].set(1)
        
        # The zeros of the mix matrix are converted to 1
        mix_matrix = mix_matrix.at[mix_matrix == 0].set(1)
        
        # Add a row of ones at the beginning
        mix_matrix = jnp.concatenate((jnp.ones((1, dim), dtype=jnp.float32), mix_matrix), axis=0)
        
        return mix_matrix
    
    def mix_weights(self, out_ft, x_ft, weights):
        slices = tuple(slice(None, mode) for mode in self.modes)
        
        # Mixing weights
        
        # First weight
        out_ft = self.complex_mult(x_ft[(Ellipsis,) + slices], weights[0])
        
        if len(weights) == 1:
            return out_ft
        
        # Rest of the weights
        for i in range(1, len(weights)):
            modes = self.mix_matrix[i].squeeze().tolist()
            slices = tuple(slice(-mode, None) if sign < 0 else slice(None, mode) for sign, mode in zip(modes, self.modes))
            out_ft = self.complex_mult(x_ft[(Ellipsis,) + slices], weights[i])
        
        return out_ft
        
    @nn.compact
    def __call__(self, x):
        batch_size, _, *sizes = x.shape
        
        # Fourier transform
        x_ft = jnp.fft.fftn(x, axes=tuple(range(-self.dim, 0)))
        
        # Initialize output
        out_ft = jnp.zeros((batch_size, self.out_channels, *sizes), dtype=jnp.complex64)
        # Reduce the last dimension to x.shape[-1]//2+1
        out_ft = out_ft[..., :x.shape[-1]//2+1]
        
        # Mixing weights
        out_ft = self.mix_weights(out_ft, x_ft, self.weights)
        
        # Inverse Fourier transform to real space
        out = jnp.fft.irfftn(out_ft, s=tuple(sizes), axes=tuple(range(-self.dim, 0)))
        
        return out