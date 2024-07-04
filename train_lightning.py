import argparse
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
# from FNO.PyTorch import FNO
from utilities.utils import MatlabFileReader
from losses.lploss import LpLoss
import lightning as L
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from neuralop.models import FNO

class LitFNO(L.LightningModule):
    def __init__(self, model, lr=1e-3, weight_decay=1e-4):
        super().__init__()
        self.model = model
        self.loss = LpLoss(size_average=False)
        self.lr = lr
        self.weight_decay = weight_decay
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat.reshape(x.shape[0], -1), y.reshape(x.shape[0], -1))
        loss_train = self.loss(y_hat.reshape(1, -1), y.reshape(1, -1))
        mse = F.mse_loss(y_hat.reshape(1, -1), y.reshape(1, -1))
        self.log('train_loss', loss_train, prog_bar=True, sync_dist=True, logger=True, on_step=False, on_epoch=True)
        self.log('mse_train', mse, prog_bar=True, sync_dist=True, logger=True, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat.reshape(1, -1), y.reshape(1, -1))
        mse = F.mse_loss(y_hat.reshape(1, -1), y.reshape(1, -1))
        self.log('val_loss', loss, prog_bar=True, sync_dist=True, logger=True, on_step=False, on_epoch=True)
        self.log('mse_val', mse, prog_bar=True, sync_dist=True, logger=True, on_step=False, on_epoch=True)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        
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

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--n_gpus', type=int, default=1)
    argparser.add_argument('--n_nodes', type=int, default=1)
    argparser.add_argument('--batch_size', type=int, default=32)
    argparser.add_argument('--epochs', type=int, default=500)
    argparser.add_argument('--lr', type=float, default=1e-3)
    argparser.add_argument('--weight_decay', type=float, default=1e-4)
    argparser.add_argument('--data_path', type=str, default=r'./data/Navier Stokes/NavierStokes_V1e-5_N1200_T20.mat')
    argparser.add_argument('--output_path', type=str, default='experiments/NavierStokes')
    argparser.add_argument('--experiment_name', type=str, default='NavierStokes')
    args = argparser.parse_args()
    
    n_gpus = args.n_gpus
    n_nodes = args.n_nodes
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    weight_decay = args.weight_decay
    data_path = args.data_path
    output_path = args.output_path
    experiment_name = args.experiment_name
    
    
    data = MatlabFileReader(data_path, to_tensor=True)
    data = data.read_file('u')
    data_train, data_eval = Dataset3D(data[:800, ...]), Dataset3D(data[800:1000, ...])
    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=25)
    eval_loader = DataLoader(data_eval, batch_size=batch_size, shuffle=False, num_workers=25)

    # model
    # model = FNO(modes=[8, 8, 6],
    #             num_fourier_layers=8,
    #             in_channels=13,
    #             lifting_channels=128,
    #             projection_channels=128,
    #             mid_channels=64,
    #             out_channels=1,
    #             activation=nn.GELU(),
    #             padding=(0, 0, 3))
    
    model = FNO(n_modes=(64, 64, 6), 
                hidden_channels=64,
                n_layers=8,
                in_channels=13,
                lifting_channels=128,
                projection_channels=128,
                out_channels=1,
                factorization='tucker',
                implementation='factorized',
                fno_block_precision='mixed',
                rank=0.05)
    
    model = LitFNO(model, lr=lr, weight_decay=weight_decay)

    # logger
    logger = CSVLogger(output_path, name=f"csv_logs_{experiment_name}")
    tensorboard = TensorBoardLogger(output_path, name=f"tb_logs_{experiment_name}")

    # train model
    trainer = L.Trainer(max_epochs=epochs, logger=[logger, tensorboard], devices=n_gpus, num_nodes=n_nodes)
    t0 = time.time()
    trainer.fit(model, train_loader, eval_loader)
    tf = time.time()
    formated_time = pd.to_datetime(tf - t0, unit='s')   
    print(f"Training time: {formated_time.strftime('%H:%M:%S')}")
    