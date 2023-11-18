# Fourier Neural Operator

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.0-red.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8.0-blue.svg)](https://www.python.org/)
![Google Colab](https://img.shields.io/badge/Google%20Colab-Open%20In%20Colab-yellow.svg)
![alpha](https://img.shields.io/badge/alpha-0.1.0-orange.svg)

This repository contains a PyTorch implementation of the following paper:

> [**Fourier Neural Operator for Parametric Partial Differential Equations**](https://arxiv.org/abs/2010.08895)

Also it contains a code to solve and generate data for Navier-Stokes equations which are given by:

$$
\begin{align*}
\partial_t w(x,t) + u(x,t) \cdot \nabla w(x,t) &= \nu \Delta w(x,t) + f(x)\\
\nabla \cdot u(x,t) &= 0\\
w(x,0) &= w_0(x)
\end{align*}
$$


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

Also if you prefer install by yourself, you can install the following packages:

- `torch>=1.7.0`
- `matplotlib`
- `numpy`
- `scipy`
- `tqdm`
- `h5py`


## Usage

This repository contains Jupyter notebooks to solve and generate data for Navier-Stokes equations, and to train and test the model. 


