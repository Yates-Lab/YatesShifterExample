"""
Poisson loss functions for DataYatesV1.

This module contains loss functions and related utilities for Poisson distributed data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def calc_poisson_bits_per_spike(robs, rhat, eps=1e-9):
    """
    Calculate bits per spike for Poisson distributed data.
    
    Parameters:
    -----------
    robs : torch.Tensor
        Observed spike counts
    rhat : torch.Tensor
        Predicted firing rates
    eps : float
        Small constant to avoid numerical issues
        
    Returns:
    --------
    torch.Tensor
        Bits per spike for each unit
    """
    # Ensure inputs are tensors
    if not isinstance(robs, torch.Tensor):
        robs = torch.tensor(robs, dtype=torch.float32)
    if not isinstance(rhat, torch.Tensor):
        rhat = torch.tensor(rhat, dtype=torch.float32)
    
    # Calculate log-likelihood
    # log p(y|λ) = y log(λ) - λ - log(y!)
    # For Poisson, we can ignore the log(y!) term when comparing models
    
    # Calculate mean firing rates
    mean_obs = robs.mean(dim=0)
    
    # Calculate log-likelihood for model predictions
    ll_model = (robs * torch.log(rhat + eps) - rhat).mean(dim=0)
    
    # Calculate log-likelihood for mean firing rate model (baseline)
    ll_mean = (robs * torch.log(mean_obs + eps) - mean_obs).mean(dim=0)
    
    # Calculate bits per spike
    # BPS = (ll_model - ll_mean) / mean_obs / log(2)
    bps = (ll_model - ll_mean) / (mean_obs + eps) / math.log(2)
    
    return bps

class PoissonBPSAggregator(nn.Module):
    """
    Module to calculate bits per spike by aggregating log likelihoods across batches.
    
    This module accumulates observed and predicted spike counts across batches
    and then calculates bits per spike at the end.
    """
    def __init__(self, device='cpu'):
        super().__init__()
        self.reset()
        self.device = device
        
    def reset(self):
        """Reset the aggregator."""
        self.robs = []
        self.rhat = []
        self.dfs = None
        self.init = True
        
    def __call__(self, batch):
        """
        Accumulate observed and predicted spike counts from a batch.
        
        Parameters:
        -----------
        batch : dict
            Batch dictionary containing 'robs' and 'rhat' keys
        """
        assert 'robs' in batch.keys(), "'robs' must be a key in the batch"
        assert 'rhat' in batch.keys(), "'rhat' must be a key in the batch"
        
        if self.init == True:
            if 'dfs' in batch.keys():
                self.dfs = []
            self.init = False
            
        self.robs.append(batch['robs'].detach().cpu())
        self.rhat.append(batch['rhat'].detach().cpu())
        
        if self.dfs is not None and 'dfs' in batch.keys():
            self.dfs.append(batch['dfs'].detach().cpu())
            
    def closure(self):
        """
        Calculate bits per spike from accumulated data.
        
        Returns:
        --------
        torch.Tensor
            Bits per spike for each unit
        """
        robs = torch.cat(self.robs, dim=0)
        rhat = torch.cat(self.rhat, dim=0)
        
        if self.dfs is not None:
            dfs = torch.cat(self.dfs, dim=0)
            # Apply mask if available
            if dfs.shape[1] == 1:
                dfs = dfs.expand(-1, robs.shape[1])
            robs = robs * dfs
            rhat = rhat * dfs
            
        return calc_poisson_bits_per_spike(robs, rhat)
