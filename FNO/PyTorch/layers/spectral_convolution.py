import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Union

# Opcional: Si deseas utilizar TensorLy para factorización avanzada
# pip install tensorly
import tensorly as tl
from tensorly.decomposition import tucker, parafac, tensor_train

# Configurar TensorLy para usar PyTorch como backend
tl.set_backend('pytorch')


class SpectralConvolution(nn.Module):
    """
    Capa de Convolución Espectral optimizada con soporte para factorización de tensores,
    entrenamiento de precisión mixta y datos N-dimensionales.

    Args:
        in_channels (int): Número de canales de entrada.
        out_channels (int): Número de canales de salida.
        modes (List[int]): Lista de modos para la convolución espectral en cada dimensión.
        factorization (str, opcional): Tipo de factorización a utilizar ('dense', 'tucker', 'cp', 'tt').
                                        Por defecto es 'dense' (sin factorización).
        rank (int, opcional): Rango para la factorización de bajo rango. Por defecto es 4.
        bias (bool, opcional): Si se incluye un sesgo en la capa. Por defecto es True.
        **kwargs: Otros parámetros adicionales.
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

        # Validar la factorización
        assert self.factorization in ['dense', 'tucker', 'cp', 'tt'], \
            "Factorización no soportada. Elige entre 'dense', 'tucker', 'cp', 'tt'."

        # Generar la matriz de mezcla
        self.mix_matrix = self.get_mix_matrix(self.dim)

        # Factorización de pesos según el tipo seleccionado
        if self.factorization == 'dense':
            # Pesos completos sin factorización
            weight_shape = (in_channels, out_channels, *self.modes)
            self.weights_real = nn.Parameter(
                torch.randn(weight_shape, dtype=torch.float32) * (1 / (in_channels * out_channels))**0.5
            )
            self.weights_imag = nn.Parameter(
                torch.randn(weight_shape, dtype=torch.float32) * (1 / (in_channels * out_channels))**0.5
            )
        else:
            # Inicializar el tensor de pesos completo para factorización
            full_weight_shape = (in_channels, out_channels, *self.modes)
            full_weight_real = torch.randn(full_weight_shape, dtype=torch.float32) * (1 / (in_channels * out_channels))**0.5
            full_weight_imag = torch.randn(full_weight_shape, dtype=torch.float32) * (1 / (in_channels * out_channels))**0.5

            # Aplicar la factorización seleccionada por separado para real e imag
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

        # Sesgo opcional
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels, dtype=torch.float32))
        else:
            self.bias = None

    @staticmethod
    def complex_mult(input_real: torch.Tensor, input_imag: torch.Tensor, weights_real: torch.Tensor, weights_imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Realiza la multiplicación compleja entre la entrada y los pesos.

        Args:
            input_real (torch.Tensor): Parte real de la entrada. [batch_size, in_channels, *sizes]
            input_imag (torch.Tensor): Parte imaginaria de la entrada. [batch_size, in_channels, *sizes]
            weights_real (torch.Tensor): Parte real de los pesos. [in_channels, out_channels, *sizes]
            weights_imag (torch.Tensor): Parte imaginaria de los pesos. [in_channels, out_channels, *sizes]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Partes real e imaginaria del resultado. [batch_size, out_channels, *sizes]
        """
        out_real = torch.einsum('bci...,cio...->bco...', input_real, weights_real) - torch.einsum('bci...,cio...->bco...', input_imag, weights_imag)
        out_imag = torch.einsum('bci...,cio...->bco...', input_real, weights_imag) + torch.einsum('bci...,cio...->bco...', input_imag, weights_real)
        return out_real, out_imag

    @staticmethod
    def get_mix_matrix(dim: int) -> torch.Tensor:
        """
        Genera una matriz de mezcla para la convolución espectral.

        Args:
            dim (int): Dimensión de la matriz de mezcla.

        Returns:
            torch.Tensor: Matriz de mezcla.
        """
        # Crear una matriz triangular inferior con -1 en la diagonal y 1 en el resto
        mix_matrix = torch.tril(torch.ones((dim, dim), dtype=torch.float32)) - 2 * torch.eye(dim, dtype=torch.float32)

        # Restar 2 a la última fila
        mix_matrix[-1] = mix_matrix[-1] - 2

        # El último elemento de la última fila es 1
        mix_matrix[-1, -1] = 1

        # Los ceros de la matriz de mezcla se convierten en 1
        mix_matrix[mix_matrix == 0] = 1

        # Añadir una fila de unos al principio
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
        Mezcla los pesos para la convolución espectral.

        Args:
            out_ft_real (torch.Tensor): Parte real del tensor de salida en el espacio de Fourier.
            out_ft_imag (torch.Tensor): Parte imaginaria del tensor de salida en el espacio de Fourier.
            x_ft_real (torch.Tensor): Parte real del tensor de entrada en el espacio de Fourier.
            x_ft_imag (torch.Tensor): Parte imaginaria del tensor de entrada en el espacio de Fourier.
            weights_real (List[torch.Tensor] o torch.Tensor): Pesos reales.
            weights_imag (List[torch.Tensor] o torch.Tensor): Pesos imaginarios.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tensores de salida mezclados (parte real e imaginaria).
        """
        # Índices de cortes según la matriz de mezcla
        slices = tuple(slice(None, min(mode, x_ft_real.size(i + 2))) for i, mode in enumerate(self.modes))

        # Mezclar pesos
        # Primer peso
        out_ft_real[(Ellipsis,) + slices], out_ft_imag[(Ellipsis,) + slices] = self.complex_mult(
            x_ft_real[(Ellipsis,) + slices], x_ft_imag[(Ellipsis,) + slices],
            weights_real[0][(Ellipsis,) + slices], weights_imag[0][(Ellipsis,) + slices]
        )

        if isinstance(weights_real, list) and len(weights_real) > 1:
            # Resto de los pesos
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
        Paso hacia adelante de la capa de convolución espectral.

        Args:
            x (torch.Tensor): Tensor de entrada de forma (batch, in_channels, D1, D2, ..., DN).

        Returns:
            torch.Tensor: Tensor de salida de forma (batch, out_channels, D1, D2, ..., DN).
        """
        batch_size, _, *sizes = x.shape

        # Aplicar la FFT N-dimensional
        x_ft = torch.fft.fftn(x, dim=tuple(range(-self.dim, 0)), norm='ortho')

        # Separar en partes real e imaginaria
        x_ft_real, x_ft_imag = x_ft.real, x_ft.imag

        # Inicializar los tensores de salida en el espacio de Fourier
        out_ft_real = torch.zeros(batch_size, self.out_channels, *sizes, dtype=x_ft_real.dtype, device=x.device)
        out_ft_imag = torch.zeros(batch_size, self.out_channels, *sizes, dtype=x_ft_imag.dtype, device=x.device)

        # Aplicar mezcla de pesos según la factorización
        if self.factorization == 'dense':
            weights_real = self.weights_real
            weights_imag = self.weights_imag
        elif self.factorization == 'tucker':
            # Reconstruir los pesos desde la factorización Tucker
            weight_recon_real = tl.tucker_to_tensor((self.core_real, [factor for factor in self.factors_real]))
            weight_recon_imag = tl.tucker_to_tensor((self.core_imag, [factor for factor in self.factors_imag]))
            weights_real = weight_recon_real
            weights_imag = weight_recon_imag
        elif self.factorization == 'cp':
            # Reconstruir los pesos desde la factorización CP
            weight_recon_real = tl.cp_to_tensor((self.weights_cp_real, [factor for factor in self.factors_cp_real]))
            weight_recon_imag = tl.cp_to_tensor((self.weights_cp_imag, [factor for factor in self.factors_cp_imag]))
            weights_real = weight_recon_real
            weights_imag = weight_recon_imag
        elif self.factorization == 'tt':
            # Reconstruir los pesos desde la factorización TT
            # Tensor Train requiere reconstrucción secuencial
            weight_recon_real = tl.tt_to_tensor(self.factors_tt_real)
            weight_recon_imag = tl.tt_to_tensor(self.factors_tt_imag)
            weights_real = weight_recon_real
            weights_imag = weight_recon_imag


        # Aplicar mezcla de pesos
        out_ft_real, out_ft_imag = self.mix_weights(
            out_ft_real, out_ft_imag, x_ft_real, x_ft_imag, weights_real, weights_imag
        )

        # Combinar partes real e imaginaria
        out_ft = torch.complex(out_ft_real, out_ft_imag)

        # Aplicar la IFFT para volver al dominio espacial
        out = torch.fft.ifftn(out_ft, dim=tuple(range(-self.dim, 0)), s=sizes, norm='ortho').real

        # Añadir el sesgo si está presente
        if self.bias is not None:
            out = out + self.bias.view(1, -1, *([1] * self.dim))

        return out
