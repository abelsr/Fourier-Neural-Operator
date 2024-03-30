"""
## Train function for FNO models

This module contains the train function for FNO models.

### Contents:
* `train_epoch`: Trains the model for one epoch
* `eval_epoch`: Evaluates the model for one epoch
* `train_model`: Trains the model
"""
import torch
from FNO.lploss import LpLoss
import numpy as np
import pandas as pd
import os
import torch.nn.functional as F
from itertools import islice as take
from tqdm import trange
from typing import List
import pandas as pd
from time import time
from torch.utils.tensorboard import SummaryWriter

def train_epoch(dataloader, model, optimizer, scheduler, loss_function, device='cpu'):
    """
    train_epoch: Trains the model for one epoch
    
    Parameters:
    -----------
    * dataloader: torch.utils.data.DataLoader - Dataloader with training data
    * model: torch.nn.Module - Model to train
    * optimizer: torch.optim - Optimizer to use
    * scheduler: torch.optim.lr_scheduler - Scheduler to use
    * loss_function: torch.nn - Loss function to use
    """
    
    # Train for each batch
    for x, y in dataloader:
        batch_size, sx, sy, T_pred = y.shape
        out = model(x.to(device)).reshape(batch_size, sx, sy, T_pred)
        
        # Compute loss
        l2 = loss_function(out.reshape(batch_size, -1), y.to(device).reshape(batch_size, -1))

        # Clear gradients
        optimizer.zero_grad()

        # Backpropagation
        l2.backward()

        # Update weights
        optimizer.step()
        
    if scheduler is not None:
        scheduler.step()
    model.eval()
    
def eval_epoch(dataloader, model, loss_function, num_batches=None, device='cpu') -> List[List[float]]:
    """
    Evaluates the model for one epoch
    
    Parameters:
    -----------
    * dataloader: torch.utils.data.DataLoader - Dataloader with training data
    * model: torch.nn.Module - Model to train
    * loss_function: torch.nn - Loss function to use
    * num_batches: int - Number of batches to evaluate
    * device: str - Device to use (cpu or cuda)
    """
    # Freeze model parameters
    with torch.no_grad():
        # Historiales
        losses, mses = [], []
        
        # Evaluates this epoch with num_batches
        test_l2 = 0
        for a, u_true in take(dataloader, num_batches):
            batch_size, sx, sy, T_pred = u_true.shape
            out = model(a.to(device)).reshape(batch_size, sx, sy, T_pred)
            
            # Evaluate loss
            test_l2 = loss_function(out.reshape(1, -1), u_true.to(device).reshape(1, -1)).item()
            test_mse = F.mse_loss(out.reshape(1, -1), u_true.to(device).reshape(1, -1), reduction='mean')
            losses.append(test_l2)
            mses.append(test_mse.cpu())
            
        # Get mean loss and mean mse
        loss = np.mean(losses)
        mse = np.mean(mses)
        
        return loss, mse
    
    
def train_model(model, train_dataloader, test_dataloader, epochs=20, device='cpu', lr=1e-3, scheduler_step=100, scheduler_gamma=0.05, train_batches=None, test_batches=None, **kwargs) -> List[List[float]]:
    """
    train_model: Trains the model
    
    Parameters:
    -----------
    * model: torch.nn.Module - Model to train
    * train_dataloader: torch.utils.data.DataLoader - Dataloader with training data
    * test_dataloader: torch.utils.data.DataLoader - Dataloader
    * lr: float - Learning rate
    * epochs: int - Number of epochs
    * scheduler_step: int - Scheduler step
    * scheduler_gamma: float - Scheduler gamma
    * train_batches: int - Number of batches to train
    * test_batches: int - Number of batches to evaluate
    * device: str - Device to use (cpu or cuda)
    * timer: bool - Timer
    
    **kwargs:
    --------
    * loss_function: torch.nn - Loss function to use (default = LpLoss)
    * optimizer: torch.optim - Optimizer to use (default = Adam)
    * scheduler: torch.optim.lr_scheduler - Scheduler to use (default = StepLR)
    * save_every: int - Save every x epochs (default = 50, -1 to save only at the end)
    * model_name: str - Model name (default = 'fno_model')
    * tensorboard: bool - Tensorboard (default = False)
    
    Returns:
    --------
    * loss_hist: List[List[float]] - History of losses
    * mse_hist: List[List[float]] - History of mses
    """
    # Loss function
    loss_function = kwargs.get('loss_function', 
                               LpLoss(size_average=False))
    
    # Optimizer
    opt = kwargs.get('optimizer',
                     torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4))
    
    # Scheduler
    scheduler = kwargs.get('scheduler', 
                           torch.optim.lr_scheduler.StepLR(opt, step_size=scheduler_step, gamma=scheduler_gamma))
    
    # Save every x epochs
    save_every = kwargs.get('save_every', 50)
    if save_every > epochs:
        save_every = epochs
    
    # Model name
    model_name: str = kwargs.get('model_name', 'fno_model')
    
    # Tensorboard
    tensorboard = kwargs.get('tensorboard', False)
    if tensorboard is True:
        print("To use tensorboard, run: tensorboard --logdir='{model_name}/tensorboard'".format(model_name=model_name))
    
    # Create folder to save models
    if not os.path.exists(model_name):
        os.makedirs(model_name)
    
    # Timer
    timer: bool = kwargs.get('timer', False)
    
    # Initialize histories
    loss_hist, mse_hist = [], []
    
    # Start timer
    if timer is True:
        start = time()
    
    # Train for each epoch
    for epoch in trange(epochs):
        
        # Train the epoch
        train_epoch(train_dataloader, model, opt, scheduler, loss_function, device)

        # Evaluates the epoch in training
        trn_loss, trn_mse = eval_epoch(train_dataloader, model, loss_function, train_batches, device)
        
        # Evaluates the epoch in testing
        tst_loss, tst_mse = eval_epoch(test_dataloader, model, loss_function, test_batches, device)

        # Save histories
        loss_hist.append([trn_loss, tst_loss])
        mse_hist.append([trn_mse, tst_mse])
        
        # Save model every x epochs and history
        if epoch % save_every == 0:
            torch.save(model, f"{model_name}/model_{epoch}_epochs.pt")
            # Save histories in a single csv file
            pd.concat([pd.DataFrame(loss_hist, columns=['train_loss', 'test_loss']), 
                       pd.DataFrame(mse_hist, columns=['train_mse', 'test_mse'])], 
                      axis=1).to_csv(f"{model_name}/history.csv")
            
        # Write to tensorboard
        if tensorboard is True:
            writer = SummaryWriter(f"{model_name}/tensorboard") # To use tensorboard, run tensorboard --logdir={model_name}/tensorboard
            # Write scalars to tensorboard to visualize them in the same plot
            writer.add_scalars('Loss', {'train': trn_loss, 'test': tst_loss}, epoch)
            writer.add_scalars('MSE', {'train': trn_mse, 'test': tst_mse}, epoch)
            writer.flush()
            
    
    
    # Save model and history
    torch.save(model, f"{model_name}/model_{epochs}_epochs.pt")
    pd.concat([pd.DataFrame(loss_hist, columns=['train_loss', 'test_loss']), 
               pd.DataFrame(mse_hist, columns=['train_mse', 'test_mse'])], axis=1).to_csv(f"{model_name}/history.csv")
    
    # End timer
    if timer is True:
        end = time()
    
    # if timer is True, return the time with the histories
    if timer is True:
        return {'results': [pd.DataFrame(loss_hist, columns=['train', 'test']), pd.DataFrame(mse_hist, columns=['train', 'test'])],
                'time': end - start}
    else:
        return {'results': [pd.DataFrame(loss_hist, columns=['train', 'test']), pd.DataFrame(mse_hist, columns=['train', 'test'])]}