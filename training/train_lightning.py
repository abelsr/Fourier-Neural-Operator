import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from FNO.PyTorch import FNO
from utilities.utils import MatlabFileReader
from losses.lploss import LpLoss
import lightning as L
from pytorch_lightning.loggers import CSVLogger

# configs
torch.set_float32_matmul_precision('medium')

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
        return torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        
        
# Dataset
class Dataset3D(TensorDataset):
    def __init__(self, data):
        self.input  = data[:, :, :, :10]
        self.output = data[:, :, :, 10:]
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
    
data = MatlabFileReader(r'D:\Abel Santillan Rodriguez\Documents\Personal\Projects\Fourier Neural Operator\data\NavierStokes_V1e-5_N1200_T20.mat', to_tensor=True)
data = data.read_file('u')
data_train, data_eval = Dataset3D(data[:800, ...]), Dataset3D(data[800:1000, ...])
train_loader = DataLoader(data_train, batch_size=16, shuffle=True)
eval_loader = DataLoader(data_eval, batch_size=16, shuffle=False)

# model
model = FNO(modes=[8, 8, 6],
            num_fourier_layers=8,
            in_channels=13,
            lifting_channels=128,
            projection_channels=128,
            mid_channels=64,
            out_channels=1,
            activation=nn.GELU(),
            padding=(0,0,3))
model = LitFNO(model)

# logger
logger = CSVLogger('logs', name='fno')

# train model
trainer = L.Trainer(max_epochs=50, accelerator='gpu', logger=logger)
trainer.fit(model, train_loader, eval_loader)