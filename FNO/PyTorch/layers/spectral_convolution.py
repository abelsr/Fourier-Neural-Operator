from typing import List, Tuple, Optional, Union

import torch
import torch.nn as nn

import tensorly as tl
from tensorly.decomposition import tucker, parafac, tensor_train

# Set TensorLy to use PyTorch as the backend
tl.set_backend('pytorch')


class SpectralConvolution(nn.Module):
    """
    Spectral Convolution layer optimized with support for tensor factorization,
    mixed-precision training, and N-dimensional data.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        modes (List[int]): List of modes for spectral convolution in each dimension.
        factorization (str, optional): Type of factorization to use ('dense', 'tucker', 'cp', 'tt').
                                       Defaults to 'dense' (no factorization).
        rank (int, optional): Rank for low-rank factorization. Defaults to 16.
        bias (bool, optional): Whether to include a bias term in the layer. Defaults to True.
        **kwargs: Additional parameters.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: List[int],
        factorization: str = 'tucker',
        rank: int = 16,
        bias: bool = True,
        **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.dim = len(self.modes)
        self.factorization = factorization.lower()
        self.rank = rank

        # Validate factorization type
        if self.factorization not in ['dense', 'tucker', 'cp', 'tt']:
            raise ValueError("Unsupported factorization. Choose from 'dense', 'tucker', 'cp', 'tt'.")

        # Generate the mixing matrix
        self.mix_matrix = self.get_mix_matrix(self.dim)

        # Weight factorization based on selected type
        if self.factorization == 'dense':
            # Full weights without factorization
            weight_shape = (in_channels, out_channels, *self.modes)
            self.weights_real = nn.Parameter(
                nn.init.xavier_uniform_(torch.empty(weight_shape, dtype=torch.float32))
            )
            self.weights_imag = nn.Parameter(
                nn.init.xavier_uniform_(torch.empty(weight_shape, dtype=torch.float32))
            )
        else:
            # Initialize the full weight tensor for factorization
            full_weight_shape = (in_channels, out_channels, *self.modes)
            full_weight_real = nn.init.xavier_uniform_(torch.empty(full_weight_shape, dtype=torch.float32))
            full_weight_imag = nn.init.xavier_uniform_(torch.empty(full_weight_shape, dtype=torch.float32))

            # Apply the selected factorization separately for real and imaginary parts
            if self.factorization == 'tucker':
                core_real, factors_real = tucker(full_weight_real, rank=[self.rank] * (2 + self.dim))
                core_imag, factors_imag = tucker(full_weight_imag, rank=[self.rank] * (2 + self.dim))
                self.core_real = nn.Parameter(core_real)
                self.core_imag = nn.Parameter(core_imag)
                self.factors_real = nn.ParameterList([nn.Parameter(factor) for factor in factors_real])
                self.factors_imag = nn.ParameterList([nn.Parameter(factor) for factor in factors_imag])
            elif self.factorization == 'cp':
                factors_cp_real = parafac(full_weight_real, rank=self.rank)
                factors_cp_imag = parafac(full_weight_imag, rank=self.rank)
                self.weights_cp_real = nn.Parameter(factors_cp_real[0])
                self.weights_cp_imag = nn.Parameter(factors_cp_imag[0])
                self.factors_cp_real = nn.ParameterList([nn.Parameter(factor) for factor in factors_cp_real[1]])
                self.factors_cp_imag = nn.ParameterList([nn.Parameter(factor) for factor in factors_cp_imag[1]])
            elif self.factorization == 'tt':
                factors_tt_real = tensor_train(full_weight_real, rank=self.rank)
                factors_tt_imag = tensor_train(full_weight_imag, rank=self.rank)
                self.factors_tt_real = nn.ParameterList([nn.Parameter(factor) for factor in factors_tt_real])
                self.factors_tt_imag = nn.ParameterList([nn.Parameter(factor) for factor in factors_tt_imag])

        # Optional bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels, dtype=torch.float32))
        else:
            self.bias = None

    @staticmethod
    def complex_mult(input_real: torch.Tensor, input_imag: torch.Tensor, weights_real: torch.Tensor, weights_imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs complex multiplication between input and weights.

        Args:
            input_real (torch.Tensor): Real part of the input. [batch_size, in_channels, *sizes]
            input_imag (torch.Tensor): Imaginary part of the input. [batch_size, in_channels, *sizes]
            weights_real (torch.Tensor): Real part of the weights. [in_channels, out_channels, *sizes]
            weights_imag (torch.Tensor): Imaginary part of the weights. [in_channels, out_channels, *sizes]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Real and imaginary parts of the result. [batch_size, out_channels, *sizes]
        """
        out_real = torch.matmul(input_real, weights_real) - torch.matmul(input_imag, weights_imag)
        out_imag = torch.matmul(input_real, weights_imag) + torch.matmul(input_imag, weights_real)
        return out_real, out_imag

    @staticmethod
    def get_mix_matrix(dim: int) -> torch.Tensor:
        """
        Generates a mixing matrix for spectral convolution.

        Args:
            dim (int): Dimension of the mixing matrix.

        Returns:
            torch.Tensor: Mixing matrix.

        The mixing matrix is generated in the following steps:
        1. Create a lower triangular matrix filled with ones and subtract 2 times the identity matrix to introduce negative values.
        2. Subtract 2 from the last row to ensure a distinct pattern for mixing.
        3. Set the last element of the last row to 1 to maintain a consistent matrix structure.
        4. Convert all zero elements to 1, ensuring no zero values are present.
        5. Add a row of ones at the beginning to provide an additional mixing row.
        """
        # Step 1: Create a lower triangular matrix with -1 on the diagonal and 1 elsewhere
        mix_matrix = torch.tril(torch.ones((dim, dim), dtype=torch.float32)) - 2 * torch.eye(dim, dtype=torch.float32)

        # Step 2: Subtract 2 from the last row
        mix_matrix[-1] = mix_matrix[-1] - 2

        # Step 3: Set the last element of the last row to 1
        mix_matrix[-1, -1] = 1

        # Step 4: Convert zeros in the mixing matrix to 1
        mix_matrix[mix_matrix == 0] = 1

        # Step 5: Add a row of ones at the beginning
        mix_matrix = torch.cat((torch.ones((1, dim), dtype=torch.float32), mix_matrix), dim=0)

        return mix_matrix

    def mix_weights(
        self,
        out_ft_real: torch.Tensor,
        out_ft_imag: torch.Tensor,
        x_ft_real: torch.Tensor,
        x_ft_imag: torch.Tensor,
        weights_real: Union[List[torch.Tensor], torch.Tensor],
        weights_imag: Union[List[torch.Tensor], torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mixes weights for spectral convolution.

        Args:
            out_ft_real (torch.Tensor): Real part of the output tensor in Fourier space.
            out_ft_imag (torch.Tensor): Imaginary part of the output tensor in Fourier space.
            x_ft_real (torch.Tensor): Real part of the input tensor in Fourier space.
            x_ft_imag (torch.Tensor): Imaginary part of the input tensor in Fourier space.
            weights_real (List[torch.Tensor] or torch.Tensor): Real weights.
            weights_imag (List[torch.Tensor] or torch.Tensor): Imaginary weights.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Mixed output tensors (real and imaginary parts).
        """
        # Slicing indices based on the mixing matrix
        slices = tuple(slice(None, min(mode, x_ft_real.size(i + 2))) for i, mode in enumerate(self.modes))

        # Mix weights
        # First weight
        out_ft_real[(Ellipsis,) + slices], out_ft_imag[(Ellipsis,) + slices] = self.complex_mult(
            x_ft_real[(Ellipsis,) + slices], x_ft_imag[(Ellipsis,) + slices],
            weights_real[0][(Ellipsis,) + slices], weights_imag[0][(Ellipsis,) + slices]
        )

        if isinstance(weights_real, list) and len(weights_real) > 1:
            # Remaining weights
            for i in range(1, len(weights_real)):
                modes = self.mix_matrix[i].squeeze().tolist()
                slices = tuple(
                    slice(-min(mode, x_ft_real.size(j + 2)), None) if sign < 0 else slice(None, min(mode, x_ft_real.size(j + 2)))
                    for j, (sign, mode) in enumerate(zip(modes, self.modes))
                )
                out_ft_real[(Ellipsis,) + slices], out_ft_imag[(Ellipsis,) + slices] = self.complex_mult(
                    x_ft_real[(Ellipsis,) + slices], x_ft_imag[(Ellipsis,) + slices],
                    weights_real[i][(Ellipsis,) + slices], weights_imag[i][(Ellipsis,) + slices]
                )

        return out_ft_real, out_ft_imag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the spectral convolution layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, in_channels, D1, D2, ..., DN).

        Returns:
            torch.Tensor: Output tensor of shape (batch, out_channels, D1, D2, ..., DN).
        """
        batch_size, _, *sizes = x.shape

        # Ensure input has the expected number of dimensions
        if len(sizes) != self.dim:
            raise ValueError(f"Expected input to have {self.dim + 2} dimensions (including batch and channel), but got {len(sizes) + 2}")

        # Apply N-dimensional FFT
        x_ft = torch.fft.fftn(x, dim=tuple(range(-self.dim, 0)), norm='ortho')

        # Separate into real and imaginary parts
        x_ft_real, x_ft_imag = x_ft.real, x_ft.imag

        # Initialize output tensors in Fourier space
        out_ft_real = torch.zeros(batch_size, self.out_channels, *sizes, dtype=x_ft_real.dtype, device=x.device)
        out_ft_imag = torch.zeros(batch_size, self.out_channels, *sizes, dtype=x_ft_imag.dtype, device=x.device)

        # Apply weight mixing based on factorization type
        if self.factorization == 'dense':
            # Use weights directly
            out_ft_real, out_ft_imag = self.mix_weights(
                out_ft_real, out_ft_imag, x_ft_real, x_ft_imag, self.weights_real, self.weights_imag
            )
        elif self.factorization == 'tucker':
            # Reconstruct weights from Tucker factorization and use them directly
            out_ft_real, out_ft_imag = self.mix_weights(
                out_ft_real, out_ft_imag, x_ft_real, x_ft_imag,
                tl.tucker_to_tensor((self.core_real, [factor for factor in self.factors_real])),
                tl.tucker_to_tensor((self.core_imag, [factor for factor in self.factors_imag]))
            )
        elif self.factorization == 'cp':
            # Reconstruct weights from CP factorization and use them directly
            out_ft_real, out_ft_imag = self.mix_weights(
                out_ft_real, out_ft_imag, x_ft_real, x_ft_imag,
                tl.cp_to_tensor((self.weights_cp_real, [factor for factor in self.factors_cp_real])),
                tl.cp_to_tensor((self.weights_cp_imag, [factor for factor in self.factors_cp_imag]))
            )
        elif self.factorization == 'tt':
            # Reconstruct weights from TT factorization and use them directly
            out_ft_real, out_ft_imag = self.mix_weights(
                out_ft_real, out_ft_imag, x_ft_real, x_ft_imag,
                tl.tt_to_tensor(self.factors_tt_real),
                tl.tt_to_tensor(self.factors_tt_imag)
            )

        # Combine real and imaginary parts
        out_ft = torch.complex(out_ft_real, out_ft_imag)

        # Apply IFFT to return to spatial domain
        out = torch.fft.ifftn(out_ft, dim=tuple(range(-self.dim, 0)), s=sizes, norm='ortho').real

        # Add bias if present
        if self.bias is not None:
            out = out + self.bias.view(1, -1, *([1] * self.dim))

        return out