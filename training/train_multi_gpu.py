import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import time
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from FNO.PyTorch import FNO
from torch.optim import Adam
from utilities.utils import MatlabFileReader
from losses.lploss import LpLoss
from torch.utils.data import DataLoader, TensorDataset
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


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

def ddp_setup(rank: int, world_size: int):
    """
    Initializes the distributed process group.
    Args:
        rank: rank of the current process
        world_size: total number of processes
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.cuda.set_device(rank)
    init_process_group(backend='nccl', rank=rank, world_size=world_size)

class Trainer:
    def __init__(self, model: nn.Module, train_data: DataLoader, eval_data: DataLoader, optimizer: torch.optim.Optimizer, gpu_id: int, save_every: int):
        self.gpu_id = gpu_id
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.eval_data = eval_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = DDP(self.model, device_ids=[self.gpu_id])
        self.loss_fn = LpLoss(size_average=False)
        self.loss_hist, self.accuracy_hist = [], []
        self.loss_hist_test, self.accuracy_hist_test = [], []
        self.progress = None

    def _run_batch(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """
        Runs a single batch for training.
        Args:
            x: input tensor
            y: target tensor
        """
        self.optimizer.zero_grad()
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat.reshape(x.shape[0], -1), y.reshape(x.shape[0], -1))
        loss.backward()
        self.optimizer.step()

    def _eval_batch(self, data_loader: DataLoader) -> float:
        """
        Evaluates the model on a given data loader.
        Args:
            data_loader: data loader for evaluation
        """
        self.model.eval()
        losses, mses = [], []
        with torch.no_grad():
            for x, y in data_loader:
                x, y = x.to(self.gpu_id), y.to(self.gpu_id)
                y_hat = self.model(x)
                loss = self.loss_fn(y_hat.reshape(1, -1), y.reshape(1, -1))
                mse = F.mse_loss(y_hat.reshape(1, -1), y.reshape(1, -1), reduction='mean')
                losses.append(loss.item())
                mses.append(mse.item())
        return torch.mean(torch.tensor(losses)).item(), torch.mean(torch.tensor(mses)).item()

    def _run_epoch(self, epoch: int) -> None:
        """
        Runs a single epoch for training.
        Args:
            epoch: current epoch number
        """
        self.model.train()
        self.train_data.sampler.set_epoch(epoch)
        for x, y in self.train_data:
            x, y = x.to(self.gpu_id), y.to(self.gpu_id)
            self._run_batch(x, y)
        
    def _save_checkpoint(self, epoch: int) -> None:
        """
        Saves the model checkpoint.
        Args:
            epoch: current epoch number
        """
        if self.gpu_id == 0:
            ckp = self.model.module.state_dict()
            PATH = f"checkpoint_{epoch}.pt"
            torch.save(ckp, PATH)
            print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int) -> None:
        """
        Trains the model.
        Args:
            max_epochs: maximum number of epochs
        """
        self.progress = tqdm(range(max_epochs)) if self.progress is None else self.progress
        for epoch in self.progress:
            self._run_epoch(epoch)
            train_loss, train_mse = self._eval_batch(self.train_data)
            test_loss, test_mse = self._eval_batch(self.eval_data)
            if self.gpu_id == 0:
                if epoch % self.save_every == 0 and epoch > 0:
                    self._save_checkpoint(epoch)
                self.progress.set_description(f"Epoch: {epoch} | Loss train: {train_loss:.4f} | Accuracy (MSE): {train_mse:.4f} | Loss test: {test_loss:.4f} | Accuracy test: {test_mse:.4f}")
                self.loss_hist.append(train_loss)
                self.accuracy_hist.append(train_mse)
                self.loss_hist_test.append(test_loss)
                self.accuracy_hist_test.append(test_mse)
        if self.gpu_id == 0:
            self._save_metrics()
            self._save_final_checkpoint()
            self._plot_metrics()

    def _save_metrics(self):
        """
        Saves the training metrics history.
        """
        print("Training done")
        print("Saving train metrics history...")
        df = pd.DataFrame({
            'loss': self.loss_hist,
            'accuracy': self.accuracy_hist,
            'loss_test': self.loss_hist_test,
            'accuracy_test': self.accuracy_hist_test
        })
        df.to_csv("distributed_metrics.csv")
        print("Metrics saved")

    def _save_final_checkpoint(self):
        """
        Saves the final model checkpoint.
        """
        final_ckp = self.model.module.state_dict()
        torch.save(final_ckp, "checkpoint_final.pt")
        print("Final checkpoint saved")
        
    def _plot_metrics(self):
        """
        Plots the training and evaluation metrics and saves the plot as a PDF.
        """
        epochs = range(len(self.loss_hist))
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(epochs, self.loss_hist, label='Training Loss')
        plt.plot(epochs, self.loss_hist_test, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()
        
        plt.subplot(2, 1, 2)
        plt.plot(epochs, self.accuracy_hist, label='Training MSE')
        plt.plot(epochs, self.accuracy_hist_test, label='Validation MSE')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Squared Error')
        plt.legend()
        plt.grid()
        
        plt.tight_layout()
        plt.savefig('training_metrics.pdf')
        print("Metrics plot saved as training_metrics.pdf")

def load_train_objs():
    """
    Loads the training objects including the dataset, model, and optimizer.
    """
    dataset = MatlabFileReader("./data/NavierStokes_V1e-5_N1200_T20.mat", to_tensor=True).read_file("u")
    train_set, eval_set = Dataset3D(dataset[:800, ...]), Dataset3D(dataset[800:1000, ...])
    model = FNO(
        modes=[8, 8, 6],
        num_fourier_layers=8,
        in_channels=13,
        lifting_channels=128,
        projection_channels=128,
        mid_channels=64,
        out_channels=1,
        activation=nn.GELU(),
        distributed=True
    )
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    return train_set, eval_set, model, optimizer

def prepare_dataloader(dataset, batch_size):
    """
    Prepares the data loader with distributed sampling.
    Args:
        dataset: the dataset to load
        batch_size: size of each batch
    """
    return DataLoader(dataset, batch_size=batch_size, pin_memory=True, shuffle=False, sampler=DistributedSampler(dataset))

def main(rank, world_size, save_every, total_epochs, batch_size):
    """
    Main function for distributed training.
    Args:
        rank: rank of the current process
        world_size: total number of processes
        save_every: frequency of saving checkpoints
        total_epochs: total number of epochs to train
        batch_size: batch size for training
    """
    ddp_setup(rank, world_size)
    train_set, eval_set, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(train_set, batch_size)
    eval_data = prepare_dataloader(eval_set, batch_size)
    trainer = Trainer(model, train_data, eval_data, optimizer, rank, save_every)
    try:
        trainer.train(total_epochs)
    finally:
        destroy_process_group()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Simple distributed training")
    parser.add_argument("--total_epochs", type=int, default=100, help="Total epochs to train the model")
    parser.add_argument("--save_every",   type=int, default=10,  help="Frequency of saving checkpoints")
    parser.add_argument("--batch_size",   type=int, default=32,  help="Input batch size on each device (default=32)")
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    print(f"Training model for {args.total_epochs} epochs")
    t0 = time.time()
    mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size), nprocs=world_size)
    tf = time.time()
    time_final = time.strftime("%H:%M:%S", time.gmtime(tf - t0))
    print(f"Training done in {time_final}")