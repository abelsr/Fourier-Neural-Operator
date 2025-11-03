import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import lightning as L
from loguru import logger
from pytorch_lightning.loggers import CSVLogger
from lightning.pytorch.callbacks import BatchSizeFinder, ModelSummary, RichProgressBar

from FNO.PyTorch import FNO
from losses.lploss import LpLoss
from utilities.utils import MatlabFileReader

# configs
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('medium')

logger.info("Libraries imported and configurations set.")

class LitFNO(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss = LpLoss(size_average=False)
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat.reshape(x.shape[0], -1), y.reshape(x.shape[0], -1))
        loss_train = self.loss(y_hat.reshape(1, -1), y.reshape(1, -1))
        self.log('train_loss', loss_train, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat.reshape(1, -1), y.reshape(1, -1))
        self.log('val_loss', loss, prog_bar=True, logger=True)
        
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
        
logger.info("LitFNO class defined.")
        
# Dataset
class Dataset3D(TensorDataset):
    def __init__(self, data):
        self.input  = data[:, :, :, :10]
        self.output = data[:, :, :, 10:20]
        self.data_size = data.shape[0]
        self.size_x = data.shape[1]
        self.size_y = data.shape[2]
        self.size_t = self.input.shape[3]
        self.input = self.input.reshape(self.data_size, self.size_x, self.size_y, 1, self.size_t).repeat(1, 1, 1, self.size_t, 1)
        self.input = self.get_grid().permute(0, 4, 1, 2, 3)


    def get_grid(self):
        x = torch.linspace(0, 1, self.size_x)
        y = torch.linspace(0, 1, self.size_y)
        t = torch.linspace(0, 1, self.size_t)
        x = x.reshape(1, -1, 1, 1, 1)
        y = y.reshape(1, 1, -1, 1, 1)
        t = t.reshape(1, 1, 1, -1, 1)
        x = x.repeat(self.data_size, 1, self.size_y, self.size_t, 1)
        y = y.repeat(self.data_size, self.size_x, 1, self.size_t, 1)
        t = t.repeat(self.data_size, self.size_x, self.size_y, 1, 1)
        return torch.cat((x, y, t, self.input), dim=-1)

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        return self.input[idx], self.output[idx]
    
class NavierStokesDataModule(L.LightningDataModule):
    def __init__(self, data, batch_size=1, num_workers=32):
        super().__init__()
        self.data = data
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        train_length = int(0.8 * self.data.shape[0])
        self.data_train = Dataset3D(self.data[:train_length, ...])
        self.data_eval = Dataset3D(self.data[train_length:, ...])

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.data_eval, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

logger.info("Dataset3D class defined.")

data = MatlabFileReader(
    file_path='/home/jorge/astro/abel_dev/Fourier-Neural-Operator/notebooks/ns_s_128_N_400_data_20251102_224600.mat', 
    to_tensor=True
)
data = data.read_file('u')
data = data[:, :, :, :20]
data2 = MatlabFileReader(
    file_path='/home/jorge/astro/abel_dev/Fourier-Neural-Operator/notebooks/ns_s_128_N_200_data.mat', 
    to_tensor=True
)
data2 = data2.read_file('u')
data = torch.cat((data, data2), dim=0)
logger.info("Datasets concatenated. Final data shape: {}", data.shape)
train_length = int(0.8 * data.shape[0])
navier_stokes_dm = NavierStokesDataModule(data, batch_size=8, num_workers=32)
logger.info("DataLoader objects created.")

# model
model = FNO(modes=[16, 16, 16],
            num_fourier_layers=8,
            in_channels=13,
            lifting_channels=16,
            projection_channels=16,
            mid_channels=64,
            out_channels=1,
            activation=nn.GELU(),
            n_fno_blocks_per_layer=1,)
model = LitFNO(model)
logger.info("Model instantiated.")

# logger
csv_logger = CSVLogger('logs', name='fno', flush_logs_every_n_steps=1)
logger.info("CSV Logger created.")

# train model
logger.info("Starting training...")
trainer = L.Trainer(
    max_epochs=50, 
    accelerator='gpu', 
    logger=[csv_logger], # type: ignore
    devices=[0, 1],
    precision='bf16-mixed',
    strategy='ddp_find_unused_parameters_true',
    callbacks=[RichProgressBar(), ModelSummary(max_depth=6)],
    enable_model_summary=False
)
trainer.fit(model, navier_stokes_dm)