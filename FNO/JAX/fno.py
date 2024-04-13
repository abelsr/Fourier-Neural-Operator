import jax
import equinox as eqx
from .layers import FNOBlock1d
from typing import List

# TODO: Make this FNO model more general for any dimension
class FNO1d(eqx.Module):
    lifting: eqx.nn.Conv1d
    fno_blocks: List[FNOBlock1d]
    projection: eqx.nn.Conv1d

    def __init__(self,in_channels,out_channels,modes,width,activation,n_blocks = 4,*,key,):
        key, lifting_key = jax.random.split(key)
        self.lifting = eqx.nn.Conv1d(in_channels,width,1,key=lifting_key,)

        self.fno_blocks = []
        for i in range(n_blocks):
            key, subkey = jax.random.split(key)
            self.fno_blocks.append(FNOBlock1d(width,width,modes,activation,key=subkey,))

        key, projection_key = jax.random.split(key)
        self.projection = eqx.nn.Conv1d(width,out_channels,1,key=projection_key,)

    def __call__(self,x,):
        x = self.lifting(x)

        for fno_block in self.fno_blocks:
            x = fno_block(x)

        x = self.projection(x)

        return x