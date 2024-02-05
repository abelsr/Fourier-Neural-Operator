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
import torch.nn.functional as F
from itertools import islice as take
from tqdm import trange
from typing import List
import pandas as pd
from time import time

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
        out = model(x.to(device)).view(batch_size, sx, sy, T_pred)
        
        # Compute loss
        mse = F.mse_loss(out.view(batch_size, -1), y.to(device).view(batch_size, -1), reduction='mean')
        l2 = loss_function(out.view(batch_size, -1), y.to(device).view(batch_size, -1))

        # Clear gradients
        optimizer.zero_grad()

        # Backpropagation
        l2.backward()

        # Update weights
        optimizer.step()

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
            out = model(a.to(device)).view(batch_size, sx, sy, T_pred)
            
            # Evaluate loss
            test_l2 = loss_function(out.view(1, -1), u_true.to(device).view(1, -1)).item()
            test_mse = F.mse_loss(out.view(1, -1), u_true.to(device).view(1, -1), reduction='mean')
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
    
    Returns:
    --------
    * loss_hist: List[List[float]] - History of losses
    * mse_hist: List[List[float]] - History of mses
    """
    # Loss function
    loss_function = kwargs.get('loss_function', 
                               LpLoss(size_average=False))
    # loss_function = torch.nn.L1Loss()
    
    # Optimizer
    opt = kwargs.get('optimizer',
                     torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4))
    
    # Scheduler
    scheduler = kwargs.get('scheduler', 
                           torch.optim.lr_scheduler.StepLR(opt, step_size=scheduler_step, gamma=scheduler_gamma))
    
    # Timer
    timer = kwargs.get('timer', False)
    
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
        
    # End timer
    if timer is True:
        end = time()
    
    # if timer is True, return the time with the histories
    if timer is True:
        return {'results': [pd.DataFrame(loss_hist, columns=['train', 'test']), pd.DataFrame(mse_hist, columns=['train', 'test'])],
                'time': end - start}
    else:
        return {'results': [pd.DataFrame(loss_hist, columns=['train', 'test']), pd.DataFrame(mse_hist, columns=['train', 'test'])]}