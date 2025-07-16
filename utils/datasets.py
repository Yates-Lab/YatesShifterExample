import torch
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from typing import Dict, Any, List, Tuple, Union
from .general import ensure_tensor
from .spikes import bin_spikes
from .exp.general import get_clock_functions, get_trial_protocols
from .exp.gratings import GratingsTrial
from .exp.gaborium import GaboriumTrial
from .exp.backimage import BackImageTrial

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

class CombinedEmbeddedDataset(torch.utils.data.Dataset):
    '''
    A dataset that combines multiple DictDatasets and embeds time into the 
    channel dimension at index time.
    
    This class allows working with multiple datasets as if they were a single dataset,
    while also handling time embedding for temporal data processing.
    '''
    def __init__(
        self, 
        dsets: Union[DictDataset, List[DictDataset]], 
        dsets_inds: Union[torch.Tensor, np.ndarray, List[Union[torch.Tensor, np.ndarray]]], 
        keys_lags: Dict[str, Union[None, int, List[int], np.ndarray, torch.Tensor]], 
        device: str = 'cpu'
    ):
        '''
        Parameters
        ----------
        dsets : DictDataset or list of DictDatasets
            The datasets to combine. If a single dataset is provided, it will be wrapped in a list.
        dsets_inds : torch.Tensor, np.ndarray, or list of these
            Indices of each dataset to serve. These specify which samples from each dataset
            will be included in the combined dataset.
        keys_lags : dict {covariate: lags}
            The covariate keys to embed and the number of lags to embed.
            If lags is array-like, it is the indices of the lags to embed into second dimension.
            If lags is None, the covariate will be included without embedding (i.e. at the current time).
            If lags is an integer, a lagged version of the covariate will be included without embedding.
        device : str, default='cpu'
            The device to store the data on.
        '''
        super(CombinedEmbeddedDataset, self).__init__()
        self.device = device
        
        # Convert single dataset to list for consistent handling
        if isinstance(dsets, torch.utils.data.Dataset):
            dsets = [dsets]
        elif not isinstance(dsets, list):
            raise ValueError('dsets must be a list of datasets')
        
        self.n_dsets = len(dsets)
        self.dsets = dsets
        
        # Validate that all datasets are DictDatasets and move them to the specified device
        for dset in dsets:
            assert isinstance(dset, DictDataset), 'dsets must be a list of DictDatasets'
            dset.to(device)
        
        # Convert single indices array to list for consistent handling
        if isinstance(dsets_inds, torch.Tensor) or isinstance(dsets_inds, np.ndarray):
            dsets_inds = [dsets_inds]
        
        # Validate that we have indices for each dataset
        assert len(dsets) == len(dsets_inds), 'dsets and dsets_inds must have the same length'
        self.dset_inds = dsets_inds

        # Assemble all indices that can be served along with the dataset index
        # This creates a tensor where each row is [dataset_idx, sample_idx]
        self.inds = torch.cat(
            [
                torch.stack([
                    iD * torch.ones_like(self.dset_inds[iD], dtype=torch.long),
                    ensure_tensor(self.dset_inds[iD], dtype=torch.long)
                ], dim=1)
                for iD in range(self.n_dsets)
            ]
        , dim=0)

        self.set_keys_lags(keys_lags)  # Fixed method name

    def set_keys_lags(self, keys_lags: Dict[str, Union[None, int, List[int], np.ndarray, torch.Tensor]]) -> None:
        '''
        Set the keys and lags for the dataset. This will also check that the requested keys are in all datasets and have consistent shapes and dtypes.

        Parameters
        ----------
        keys_lags : dict {covariate: lags}
            The covariate keys to embed and the number of lags to embed.
            If lags is array-like, it is the indices of the lags to embed into second dimension.
            If lags is None, the covariate will be included without embedding (i.e. at the current time).
        '''
        # Transform the requested keys into the appropriate format
        # Negative lags represent looking into the past
        self.keys_lags = deepcopy(keys_lags)
        self.keys_inds = deepcopy(keys_lags)
        for key in self.keys_inds:
            if self.keys_inds[key] is None:
                # No lag, use current time
                self.keys_inds[key] = 0  # This is handled as an int in __getitem__
            elif isinstance(self.keys_inds[key], int):
                # Single lag, convert to negative for past
                self.keys_inds[key] = -self.keys_inds[key]
            elif isinstance(self.keys_inds[key], (list, tuple, np.ndarray, torch.Tensor)):
                # Multiple lags, convert to tensor and make negative for past
                self.keys_inds[key] = -ensure_tensor(self.keys_inds[key], dtype=torch.long)
            else:
                raise ValueError('lags must be None, int, or array-like')

        # Check that the requested keys are in all datasets and have consistent shapes and dtypes
        self.keys_dims: Dict[str, torch.Size] = {}
        self.keys_dtypes: Dict[str, torch.dtype] = {}
        for key in self.keys_inds:
            for dset in self.dsets:
                if key in dset:
                    if key not in self.keys_dims:
                        # Store dimensions and dtype for first occurrence
                        self.keys_dims[key] = dset[key].shape[1:]
                        self.keys_dtypes[key] = dset[key].dtype
                    else:
                        # Verify consistency across datasets
                        assert dset[key].shape[1:] == self.keys_dims[key], f'{key} has inconsistent shape across datasets'
                        assert dset[key].dtype == self.keys_dtypes[key], f'{key} has inconsistent dtype across datasets'
                else:
                    raise ValueError(f'{key} not in dataset {dset}')
    
    
    def to(self, device: str) -> 'CombinedEmbeddedDataset':
        '''
        Move all covariates to a device
        
        Parameters
        ----------
        device : str
            The device to move the data to
            
        Returns
        -------
        CombinedEmbeddedDataset
            Self, for method chaining
        '''
        self.device = device
        for dset in self.dsets:
            dset.to(device)
        return self
        
    def __len__(self) -> int:
        '''
        Returns the total number of samples in the combined dataset
        '''
        return len(self.inds)

    def __getitem__(self, idx: Union[int, slice, List[int], torch.Tensor]) -> Dict[str, torch.Tensor]:
        '''
        Get items from the combined dataset with time embedding
        
        Parameters
        ----------
        idx : int, slice, list, or tensor
            Indices to retrieve from the combined dataset
            
        Returns
        -------
        dict
            Dictionary of covariates with time embedding applied
        '''
        sample_inds = self.inds[idx]
        should_squeeze = False
        if sample_inds.ndim == 1:
            # Handle single index case by adding a dimension
            sample_inds = sample_inds[None, :]
            should_squeeze = True
            
        n_samples = len(sample_inds)

        # Preallocate the output dictionary with appropriate dimensions
        out_dict: Dict[str, torch.Tensor] = {}
        for key in self.keys_inds:
            dims = (n_samples, )
            if isinstance(self.keys_inds[key], torch.Tensor):
                # Add lag dimension for time-embedded features
                dims += (self.keys_inds[key].shape[0],)
            dims += self.keys_dims[key]
            out_dict[key] = torch.zeros(dims, dtype=self.keys_dtypes[key], device=self.device)

        # Sample the datasets and insert into the output dictionary
        # First get unique dataset indices and their mapping back to the original indices
        dsets_req, dset_final_inds = torch.unique(sample_inds[:, 0], return_inverse=True)
        for iD in range(len(dsets_req)):
            dset_idx = dsets_req[iD]
            dset = self.dsets[dset_idx]
            # Get sample indices for this dataset
            dset_inds = sample_inds[dset_final_inds == iD, 1]
            
            # Apply time embedding for each key
            for key, lags in self.keys_inds.items():
                if isinstance(lags, torch.Tensor):
                    # Multiple lags: create indices for each lag
                    keys_inds = dset_inds[:,None] + lags[None, :] 
                elif isinstance(lags, int):
                    # Single lag: add lag to indices
                    keys_inds = dset_inds + lags

                # Retrieve data with time embedding and store in output
                out_dict[key][dset_final_inds == iD] = dset[key][keys_inds]
        
        # permute stim if necessary
        if 'stim' in out_dict and out_dict['stim'].ndimension() == 5:
            out_dict['stim'] = out_dict['stim'].permute(0, 2, 1, 3, 4)
                
        # Remove batch dimension if we added it
        if should_squeeze:
            for key in out_dict:
                    out_dict[key] = out_dict[key].squeeze(0)

        return out_dict

    def shallow_copy(self) -> 'CombinedEmbeddedDataset':
        '''
        Return a shallow copy of the dataset, copying all internal data except the datasets.
        The returned copy references the same dataset objects but has independent copies
        of indices and other internal data structures.
        
        Returns
        -------
        CombinedEmbeddedDataset
            A new dataset instance with copied internal data but references to the same datasets
        '''
        # Create new instance with same datasets but copies of other data
        copy_dset_inds = [inds.clone() for inds in self.dset_inds]
        
        # Create new instance with references to same datasets
        copy_dataset = CombinedEmbeddedDataset(
            self.dsets, 
            copy_dset_inds, 
            self.keys_lags, # a deepcopy is made at initialization
            device=self.device
        )
        
        return copy_dataset
    
    def get_dataset_index(self, dset_name: str) -> int:
        '''
        Get the index of a specific sub-dataset within the combined dataset.

        Parameters
        ----------
        dset_name : str
            The name of the sub-dataset to find.

        Returns
        -------
        int
            The index of the specified sub-dataset.
        '''
        dset_names = [d.metadata['name'] for d in self.dsets]
        dset_idx = dset_names.index(dset_name)
        return dset_idx
    
    def get_dataset_inds(self, dset_name: str) -> torch.Tensor:
        '''
        Get the indices for a specific sub-dataset within the combined dataset.

        Parameters
        ----------
        dset_name : str
            The name of the sub-dataset to retrieve indices for.

        Returns
        -------
        inds : torch.Tensor
            The indices for the specified sub-dataset.
        '''
        dset_idx = self.get_dataset_index(dset_name)
        inds = self.inds[self.inds[:,0]==dset_idx]
        return inds

    def get_inds_from_times(self, times: Union[torch.Tensor, np.ndarray, list, float])->torch.Tensor:
        """
        Convert timestamps to dataset indices.

        This function maps time points to the corresponding indices in the dataset,
        handling multiple trials and datasets.

        Parameters
        ----------
        dset : CombinedEmbeddedDataset
            The dataset to map times to.
        times : Union[torch.Tensor, np.ndarray, list, float]
            Time points to convert to indices.

        Returns
        -------
        torch.Tensor
            Dataset indices corresponding to the input times.
        """
        times = ensure_tensor(times)

        inds = []
        # Iterate through each sub-dataset
        for dset_ind, ds in enumerate(self.dsets):
            t_bins = ds['t_bins'].flatten()
            trials = ds['trial_inds'].flatten()
            unique_trials = torch.unique(trials)

            # Process each trial separately
            for iT in unique_trials:
                # Get time bins for this trial
                t_mask = torch.nonzero(trials == iT).flatten()
                t_mask = t_mask.to(self.device)
                trial_times = t_bins[t_mask]

                # Calculate time bin width
                dt = torch.median(torch.diff(trial_times))

                # Create bin edges for time discretization
                trial_edges = torch.cat([trial_times-dt/2, trial_times[[-1]]+dt/2])
                trial_edges = trial_edges.to(self.device)

                # Find which bin each time falls into
                trial_time_inds = torch.bucketize(times, trial_edges) - 1

                # Filter out invalid indices
                valid_inds = (trial_time_inds >= 0) & (trial_time_inds < len(trial_edges)-1)
                trial_time_inds = trial_time_inds[valid_inds]

                # Map to dataset indices
                time_inds = t_mask[trial_time_inds]

                # Store dataset index and time index pairs
                inds.append(torch.stack([dset_ind*torch.ones_like(time_inds), time_inds], dim=1))

        # Combine all indices
        inds = torch.cat(inds, dim=0)
        return inds

    def cast(self, dtype, target_keys=None, protect_keys=None) -> 'CombinedEmbeddedDataset':
        '''
        Convert all floating point tensors in all datasets to the specified dtype
    
        Parameters
        ----------
        dtype : torch.dtype
            The dtype to convert tensors to
        
        Returns
        -------
        CombinedEmbeddedDataset
            Self, for method chaining
        '''
        for dset in self.dsets:
            dset.cast(dtype, target_keys=target_keys, protect_keys=protect_keys)

        # Update stored dtype information
        if target_keys is None:
            target_keys = self.keys_dtypes.keys()

        for key in target_keys:
            if protect_keys is not None and key in protect_keys:
                continue
            if torch.can_cast(self.keys_dtypes[key], dtype):
                self.keys_dtypes[key] = dtype

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
                              metadata={}):
    protocols = get_trial_protocols(exp)
    ptb2ephys, _ = get_clock_functions(exp)
    st = ks_results.spike_times
    clu = ks_results.spike_clusters
    cids = np.unique(clu)

    # Export Gratings dataset
    trials = [(iT, GratingsTrial(exp['D'][iT], exp['S'])) for iT in range(len(exp['D'])) if protocols[iT] == 'ForageGrating']
    print(f'There are {len(trials)} Gratings trials.')
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
    for iT, trial in tqdm(trials, 'Generating Gratings dataset'):
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

def generate_backimage_dataset(exp, ks_results, roi_src, pix_interp, ep_interp, valid_interp, dt=1/240, metadata={}, trial_subset=.5):
    protocols = get_trial_protocols(exp)
    ptb2ephys, _ = get_clock_functions(exp)
    
    st = ks_results.spike_times
    clu = ks_results.spike_clusters
    cids = np.unique(clu)

    # Export BackImage dataset
    backimage_trials = [(iT, BackImageTrial(exp['D'][iT], exp['S'])) for iT in range(len(exp['D'])) if protocols[iT] == 'BackImage']
    print(f'There are {len(backimage_trials)} BackImage trials. Using {trial_subset*100:.0f}% of them.')
    n_trials = int(len(backimage_trials) * trial_subset)
    print(f'Using {n_trials} trials.')
    trial_inds = np.random.choice(len(backimage_trials), n_trials, replace=False)
    trial_inds = np.sort(trial_inds) # maintain trial order
    backimage_trials = [backimage_trials[iT] for iT in trial_inds]

    backimage_dict = {
        't_bins': [],
        'trial_inds': [],
        'stim': [],
        'robs': [],
        'dpi_pix': [],
        'eyepos' : [],
        'dpi_valid': [],
        'roi': [],
    }
    for iT, trial in tqdm(backimage_trials, 'BackImage trials'):
        # Setup bins
        trial_onset = ptb2ephys(trial.image_onset_ptb)
        trial_offset = ptb2ephys(trial.image_offset_ptb)
        trial_bin_edges = np.arange(trial_onset, trial_offset, dt)
        trial_bins = trial_bin_edges[:-1] + dt/2
        backimage_dict['t_bins'].append(trial_bins)
        backimage_dict['trial_inds'].append(np.ones_like(trial_bins, dtype=np.int64) * iT)

        # Get DPI
        trial_dpi = pix_interp(trial_bins)
        trial_dpi_valid = valid_interp(trial_bins)
        backimage_dict['dpi_pix'].append(trial_dpi)
        backimage_dict['eyepos'].append(ep_interp(trial_bins))
        backimage_dict['dpi_valid'].append(trial_dpi_valid)

        # Get ROI
        trial_roi = trial_dpi[...,None].astype(int) + roi_src[None,...]
        backimage_dict['roi'].append(trial_roi)

        # Sample the stimulus for each frame
        trial_stim = trial.get_roi(trial_roi)
        backimage_dict['stim'].append(trial_stim)

        # Bin spikes
        robs_trial = bin_spikes(st, trial_bin_edges, clu, cids)
        backimage_dict['robs'].append(robs_trial)

    if backimage_trials:
        for k, v in backimage_dict.items():
            backimage_dict[k] = np.concatenate(v)

        return DictDataset(backimage_dict, metadata=metadata)
    else:
        return None