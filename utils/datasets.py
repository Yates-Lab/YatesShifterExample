import torch
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from .general import ensure_tensor
from .spikes import bin_spikes
from .exp.general import get_clock_functions, get_trial_protocols
from .exp.gratings import GratingsTrial
from .exp.gaborium import GaboriumTrial

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

def generate_gaborium_dataset(exp, ks_results, roi_src, pix_interp, ep_interp, valid_interp, dt=1/240, metadata={}, trial_subset=.5):
    protocols = get_trial_protocols(exp)
    ptb2ephys, _ = get_clock_functions(exp)
    
    st = ks_results.spike_times
    clu = ks_results.spike_clusters
    cids = np.unique(clu)

    # Export Gaborium dataset
    gaborium_trials = [(iT, GaboriumTrial(exp['D'][iT], exp['S'])) for iT in range(len(exp['D'])) if protocols[iT] == 'ForageGabor']
    print(f'There are {len(gaborium_trials)} Gaborium trials. Using {trial_subset*100:.0f}% of them.')
    n_trials = int(len(gaborium_trials) * trial_subset)
    print(f'Using {n_trials} trials.')
    trial_inds = np.random.choice(len(gaborium_trials), n_trials, replace=False)
    trial_inds = np.sort(trial_inds) # maintain trial order
    gaborium_trials = [gaborium_trials[iT] for iT in trial_inds]
    gaborium_dict = {
        't_bins': [],
        'trial_inds': [],
        'stim': [],
        'robs': [],
        'dpi_pix': [],
        'eyepos': [],
        'dpi_valid': [],
        'roi': [],
    }
    for iT, trial in tqdm(gaborium_trials, 'Regenerating Gaborium Stimulus'):
        # get flip times in ephys time
        flip_times = ptb2ephys(trial.flip_times)

        # Setup bins
        trial_bin_edges = np.arange(flip_times[0], flip_times[-1], dt)
        trial_bins = trial_bin_edges[:-1] + dt/2
        gaborium_dict['t_bins'].append(trial_bins)
        gaborium_dict['trial_inds'].append(np.ones_like(trial_bins) * iT)

        # Get DPI
        trial_dpi = pix_interp(trial_bins)
        gaborium_dict['dpi_pix'].append(trial_dpi)
        gaborium_dict['eyepos'].append(ep_interp(trial_bins))
        gaborium_dict['dpi_valid'].append(valid_interp(trial_bins))

        # Get ROI
        trial_roi = trial_dpi[...,None].astype(int) + roi_src[None,...]
        gaborium_dict['roi'].append(trial_roi)

        # Get the frame index for each bin 
        frame_inds = np.searchsorted(flip_times, trial_bins) - 1
        frame_inds[frame_inds < 0] = 0

        # Sample the stimulus for each frame
        trial_stim = trial.get_frames(frame_inds, roi=trial_roi)
        gaborium_dict['stim'].append(trial_stim)

        # Bin spikes
        trial_robs = bin_spikes(st, trial_bin_edges, clu, cids)
        gaborium_dict['robs'].append(trial_robs)

    if gaborium_trials:
        for k, v in gaborium_dict.items():
            gaborium_dict[k] = np.concatenate(v)

        return DictDataset(gaborium_dict, metadata=metadata)
    else:
        return None
def generate_gratings_dataset(exp, ks_results, roi_src, 
                              pix_interp, ep_interp, valid_interp, dt=1/240, 
                              metadata={}, trial_subset=1):
    protocols = get_trial_protocols(exp)
    ptb2ephys, _ = get_clock_functions(exp)
    st = ks_results.spike_times
    clu = ks_results.spike_clusters
    cids = np.unique(clu)

    # Export Gratings dataset
    trials = [(iT, GratingsTrial(exp['D'][iT], exp['S'])) for iT in range(len(exp['D'])) if protocols[iT] == 'ForageGrating']
    print(f'There are {len(trials)} Gratings trials. Using {trial_subset*100:.0f}% of them.')
    n_trials = int(len(trials) * trial_subset)
    print(f'Using {n_trials} trials.')
    trial_inds = np.random.choice(len(trials), n_trials, replace=False)
    trial_inds = np.sort(trial_inds)
    trials = [trials[iT] for iT in trial_inds]
    gratings_dict = {
            't_bins': [],
            'stim': [],
            'stim_phase': [],
            'sf': [],
            'ori': [],
            'robs': [],
            'dpi_pix': [],
            'eyepos': [],
            'dpi_valid': [],
            'roi': [],
            'trial_inds': [],
        }
    for iT, trial in tqdm(trials, 'Gratings trials'):
        # get flip times in ephys time
        flip_times = ptb2ephys(trial.flip_times)

        # Setup bins
        trial_bin_edges = np.arange(flip_times[0], flip_times[-1], dt)
        trial_bins = trial_bin_edges[:-1] + dt/2
        gratings_dict['t_bins'].append(trial_bins)
        gratings_dict['trial_inds'].append(np.ones_like(trial_bins) * iT)

        # Get DPI
        trial_dpi = pix_interp(trial_bins)
        gratings_dict['dpi_pix'].append(trial_dpi)
        gratings_dict['eyepos'].append(ep_interp(trial_bins))
        gratings_dict['dpi_valid'].append(valid_interp(trial_bins))

        # Get ROI
        trial_roi = trial_dpi[...,None].astype(int) + roi_src[None,...]
        gratings_dict['roi'].append(trial_roi)

        # Get the frame index for each bin 
        frame_inds = np.searchsorted(flip_times, trial_bins) - 1
        frame_inds[frame_inds < 0] = 0

        # Sample the stimulus for each frame
        trial_stim = trial.get_frames(frame_inds, roi=trial_roi)
        gratings_dict['stim'].append(trial_stim)
        trial_stim_phase = trial.get_frames_phase(frame_inds, roi=trial_roi)
        gratings_dict['stim_phase'].append(trial_stim_phase)

        # Get the spatial frequency and orientation for each frame
        trial_sf = trial.spatial_frequencies[frame_inds]
        gratings_dict['sf'].append(trial_sf)
        trial_ori = trial.orientations[frame_inds]
        gratings_dict['ori'].append(trial_ori)

        # Bin spikes
        trial_robs = bin_spikes(st, trial_bin_edges, clu, cids)
        gratings_dict['robs'].append(trial_robs)

    if trials:
        for k, v in gratings_dict.items():
            gratings_dict[k] = np.concatenate(v)

        return DictDataset(gratings_dict, metadata=metadata)
    else:
        return None