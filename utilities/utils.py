import numpy as np
import torch
from scipy.io import loadmat

class MatlabFileReader:
    """
    A class for reading MATLAB files.

    Args:
        file_path (str): The path to the MATLAB file.
        device (str, optional): The device to store the data (default is 'cpu').
        to_numpy (bool, optional): Convert the data to NumPy array (default is True).
        to_tensor (bool, optional): Convert the data to PyTorch tensor (default is False).

    Methods:
        read_file(section='data'): Read the specified section of the MATLAB file.

    Attributes:
        file_path (str): The path to the MATLAB file.
        device (str): The device to store the data.
        to_numpy (bool): Convert the data to NumPy array.
        to_tensor (bool): Convert the data to PyTorch tensor.
    """
    def __init__(self, file_path, **kwargs):
        self.file_path = file_path
        self.device = kwargs.get('device', 'cpu') # default is 'cpu'
        self.to_numpy = kwargs.get('to_numpy', True) # default is True
        self.to_tensor = kwargs.get('to_tensor', False) # default is False
        self._load_file_()
        
    def _load_file_(self):
        """
        Load the MATLAB file.
        """
        self.mat_data = loadmat(self.file_path)
        self.keys = list(self.mat_data.keys())
        

    def read_file(self, section='data'):
        """
        Read the specified section of the MATLAB file.

        Args:
            section (str, optional): The section to read from the MATLAB file (default is 'data').

        Returns:
            numpy_array (ndarray or Tensor): The data read from the MATLAB file.
        """
        if section not in self.keys:
            raise ValueError(f'Invalid section {section}.')
        numpy_array = self.mat_data[section]
        if self.to_tensor:
            numpy_array = torch.from_numpy(numpy_array).to(self.device)
        return numpy_array
    
    def __repr__(self):
        return f'MatlabFileReader(file_path={self.file_path}, device={self.device}, to_numpy={self.to_numpy}, to_tensor={self.to_tensor})'
    
    
