import torch
import numpy as np
from copy import deepcopy
from .general import ensure_tensor

def get_memory_footprint(t):
    '''
    Get the memory footprint of a tensor, sparse_coo_tensor, or numpy array
    '''
    if isinstance(t, np.ndarray):
        return t.nbytes
    if isinstance(t, torch.Tensor):
        return t.element_size() * t.nelement()
    if isinstance(t, torch.sparse_coo_tensor):
        return t.element_size() * t.values().nelement() + t.element_size() * t.indices().nelement()
    
    return 0

def get_memory_footprints_str(t):
    '''
    Get the memory footprint of a tensor, sparse_coo_tensor, or numpy array as a string with human readable units
    '''
    size = get_memory_footprint(t)
    suffixes = ['B', 'KB', 'MB', 'GB', 'TB']
    i = 0
    while size > 1024 and i < len(suffixes):
        size = size / 1024
        i += 1
    return f'{size:.2f} {suffixes[i]}'


class DictDataset(torch.utils.data.Dataset):
    '''
    A generic PyTorch dataset that stores covariates in a dictionary. Also holds metadata about the dataset.
    '''
    def __init__(self,
        data, metadata={}, replicates=False):
        '''
        Parameters
        ----------
        data : dict
            Dictionary of covariates. Each covariate is a torch tensor and must have the same first dimension length.
        metadata : dict
            Dictionary of metadata.
        replicates : bool
            If true, DictDataset object will return a new DictDataset object when indexed.
        '''
        self.metadata = metadata
        self.covariates = data
        for key in self.covariates.keys():
            self.covariates[key] = ensure_tensor(self.covariates[key])

        # Check to be sure all covariates have the same first dimension length
        # This is the number of samples in the dataset
        first_dim = self.covariates[list(self.covariates.keys())[0]].shape[0]
        for key in self.covariates.keys():
            if self.covariates[key].shape[0] != first_dim:
                raise ValueError(f'Covariate {key} has different first dimension length than other covariates. {self.covariates[key].shape[0]} != {first_dim}')

        # Send all covariates to the same device
        # If they are on different devices, send them to the CPU
        # If they are on the same device, then nothing to do
        cov_devices = set([v.device for v in self.covariates.values()])
        if len(cov_devices) > 1:
            self.device = 'cpu'
            self.to('device')
        else:
            self.device = list(cov_devices)[0] 

        self.replicates = replicates

    def __getitem__(self, index):
        '''
        Return the covariates with a given name if the index is a string. Get a subset of the dataset if the index can be used to slice an array. 
        If replcates is False, 
        '''
        # if key return the value
        if isinstance(index, str):
            return self.covariates[index]

        covariates = {cov: self.covariates[cov][index,...] for cov in self.keys()}
        if not self.replicates:
            return covariates

        return DictDataset(covariates, metadata=self.metadata, replicates=True)  
    
    def add_covariate(self, key, value):
        '''
        Add a covariate to the dataset
        '''
        value = ensure_tensor(value)
        assert value.shape[0] == len(self), f'Covariate {key} has different first dimension length than other covariates. {value.shape[0]} != {len(self)}'
        value = value.to(self.device)
        self.covariates[key] = value
        return self

    def remove_covariate(self, key):
        '''
        Remove a covariate from the dataset
        '''
        if key in self.covariates:
            del self.covariates[key]
        else:
            print(f'{key} not in dataset, doing nothing.')

    def to(self, device):
        '''
        Move all covariates to a device
        '''
        self.covariates = {k: v.to(device) for k,v in self.covariates.items() if isinstance(v, torch.Tensor)}
        return self

    def keys(self):
        return self.covariates.keys()
    
    def __repr__(self): 
        rep = 'DictDataset\n'
        for k,v in self.covariates.items():
            rep += f'\t{k} ({v.device}): {get_memory_footprints_str(v)} - {" x ".join([str(s) for s in v.shape])} ({v.dtype})\n'
        if self.metadata:
            rep += '\tMetadata:\n'
            for k,v in self.metadata.items():
                rep += f'\t\t{k}: {type(v).__name__}\n'
        return rep
        
    def __len__(self):
        return self.covariates[list(self.covariates.keys())[0]].shape[0]

    def __setitem__(self, key, value):
        self.add_covariate(key, value)

    def __contains__(self, key):
        return key in self.covariates
    
    def concatenate(self, other):
        '''
        Concatenate another dataset to this one
        '''
        for k in self.covariates.keys():
            self.covariates[k] = torch.cat((self.covariates[k], other[k]), dim=0)
        
        self.metadata.update(other.metadata)

        return self

    def __add__(self, other):
        return self.concatenate(other)
    
    def save(self, f):
        '''
        Save the dataset to a file 
        '''
        d = {
            'covariates': self.covariates,
            'metadata': self.metadata
        }
        torch.save(d, f)

    @staticmethod
    def load(f):
        '''
        Load the dataset from a file
        '''
        data = torch.load(f, weights_only=False)
        for k in data['covariates'].keys():
            data['covariates'][k].requires_grad = False
        if 'covariates' not in data:
            raise ValueError('File does not contain covariates')
        if 'metadata' not in data:
            data['metadata'] = {}
        return DictDataset(data['covariates'], metadata=data['metadata'])

    @staticmethod
    def collate_fn(batch):
        '''
        Collate function for a PyTorch DataLoader.
        '''
        
        covariates = {k: torch.cat([b[k] for b in batch], dim=0) for k in batch[0].keys()}
        metadata = batch[0].metadata
        return DictDataset(covariates, metadata=metadata)

    def copy(self):
        '''
        Return a copy of the dataset, including covariates.
        '''
        copy_covariates = {k: v.clone() for k, v in self.covariates.items()}
        copy_metadata = deepcopy(self.metadata)
        return DictDataset(copy_covariates, metadata=copy_metadata, replicates=self.replicates)

