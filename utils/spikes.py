import numpy as np
from .general import ensure_ndarray
from pathlib import Path
from functools import cached_property
import pandas as pd

def bin_spikes(spike_times, t_bin_edges, spike_clusters=None, cids=None, method = 'sparse', oob='clip'):
    """
    Bin spike times into time bins. 

    Parameters
    ----------
    spike_times : np.ndarray (n_spikes,)
        Spike times in seconds
    t_bin_edges : np.ndarray (n_bins+1,)
        Time bin edges
    spike_clusters : np.ndarray (n_spikes,) dtype=int32 or castable
        Unit ids of the spikes. If None, all spikes are assumed to be from the same unit, and the output is a 1D array.
    cids : np.ndarray (n_units,) dtype=int32 or castable
        The list of clusters, in any order, to include in the computation. That order will be used
        in the output array. If None, order the clusters by unit id from `spike_clusters`.
    method : str
        'sparse' or 'hist'. If 'sparse', a sparse tensor is used to count the spikes (Jake's method).
        If 'hist', histogram function is used to count the spikes (slower).
    oob : str
        'clip' or 'edge'. If 'clip', spikes outside the timebins are removed from the count. 
        If 'edge', output has 2 extra bins at the beginning and end to count the spikes outside the timebins.
    
    Returns
    -------
    counts : np.ndarray (n_bins,) or (n_bins+2,)
        Count of spikes in each time bin. If oob='edge', the output has 2 extra bins at the beginning and end.

    author: RKR 2/7/2024
    """

    spike_times = ensure_ndarray(spike_times).squeeze()
    if spike_times.ndim == 0:
        spike_times = spike_times[None]
    assert spike_times.ndim == 1, 'spike_times must be a 1D array'
    t_bin_edges = ensure_ndarray(t_bin_edges).squeeze()
    assert t_bin_edges.ndim == 1, 't_bin_edges must be a 1D array'
    assert len(t_bin_edges) > 1, 't_bin_edges must have at least 2 elements'
    assert np.all(np.diff(t_bin_edges) > 0), 't_bin_edges must be monotonically increasing'

    # Handle empty input
    if len(spike_times) == 0:
        if cids is None:
            n_units = 1
        else:
            n_units = len(cids)
        return np.zeros((len(t_bin_edges)-1, n_units), dtype=int)

    squeeze_output = False
    if spike_clusters is None:
        assert len(spike_times.shape) == 1, 'spike_times must be a 1D array if ids is not provided'
        spike_clusters = np.zeros(len(spike_times), dtype=np.int32)
        squeeze_output = True
    else:
        spike_clusters = ensure_ndarray(spike_clusters, dtype=np.int32).squeeze()
        assert len(spike_times) == len(spike_clusters), 'spike_times and ids must have the same length'

        if spike_clusters.ndim == 0:
            spike_clusters = spike_clusters[None]

    assert spike_clusters.ndim == 1, 'ids must be a 1D array'

    assert method in ['sparse', 'hist'], 'method must be either "sparse" or "hist"'
    assert oob in ['clip', 'edge'], 'oob must be either "clip" or "edge"'

    # Make spike times monotonic if not already
    if not np.all(np.diff(spike_times) >= 0):
        print("Spike times are not sorted, sorting")
        sort_inds = np.argsort(spike_times)
        spike_times = spike_times[sort_inds]
        spike_clusters = spike_clusters[sort_inds]

    # remove spikes outside the timebins
    if oob == 'clip':
        i0 = np.searchsorted(spike_times, t_bin_edges[0])
        i1 = np.searchsorted(spike_times, t_bin_edges[-1])
        spike_times = spike_times[i0:i1]
        spike_clusters = spike_clusters[i0:i1]

    # Initialize cids if not provided and check if the ids are valid
    spike_cluster_ids = np.unique(spike_clusters)
    if cids is None:
        cids = np.sort(spike_cluster_ids)
    else:
        cids = ensure_ndarray(cids).squeeze()
        assert cids.ndim == 1, 'cids must be a 1D array'

    # Filter the spikes to only include the clusters in cids
    if not np.all(np.isin(spike_cluster_ids, cids)):
        cids_mask = np.isin(spike_clusters, cids)
        spike_times = spike_times[cids_mask]
        spike_clusters = spike_clusters[cids_mask]

    n_cids = len(cids)
    cids2inds = np.zeros(np.max(cids)+1, dtype=int)
    cids2inds[cids] = np.arange(n_cids)
    spike_inds = cids2inds[spike_clusters]
    # add -inf to the beginning to catch the spikes before the first bin
    t_bins = np.concatenate([[-np.inf], t_bin_edges, [np.inf]]) 
    n_spikes = len(spike_times)
    
    if method == 'sparse':
        # Jake's method (super fast)
        # digitize the spike times into the time bins
        # then count the number of spikes in each bin
        # using a sparse tensor to count the spikes
        import torch
        counts = torch.sparse_coo_tensor(
                        np.asarray([np.digitize(spike_times, t_bins)-1, spike_inds]),
                        np.ones(n_spikes), 
                        (len(t_bins)-1, n_cids), 
                        dtype=torch.float32
                    ).to_dense()
        counts = counts.numpy()
    elif method == 'hist':
        # slower method
        counts, *_ = np.histogramdd(np.stack([spike_times, spike_inds],axis=1), bins=[t_bins, np.arange(n_cids+1)])
    else:
        raise ValueError(f'Invalid method: {method}')
    
    if oob == 'edge':
        counts = np.squeeze(counts)
    else:
        counts =  np.squeeze(counts[1:-1])
    
    if not squeeze_output and counts.ndim == 1:
        counts = counts[None]
    
    return counts

def convert_samples_to_time(samples, adfreq, ts=None, fn=None):
    """
    Convert samples into timestamps using sampling rate and start-time
    
    Parameters:
    -----------
    samples : array-like
        Vector of sample indices
    adfreq : float
        Sampling rate
    ts : array-like, optional
        Vector of recording fragment start timestamps
    fn : array-like, optional
        Vector of fragment sample counts
        
    Returns:
    --------
    times : ndarray
        Converted timestamps
        
    Notes:
    ------
    ts and fn are necessary to adjust sample times for recordings that were
    paused or are not continuous

    jly 2025-02-07 wrote it
    """
    
    # Handle default arguments
    if fn is None:
        fn = np.max(samples)
    if ts is None:
        ts = 0
        
    # Convert inputs to numpy arrays if they aren't already
    samples = np.asarray(samples)

    # Handle scalar inputs by converting to 1-element arrays
    ts = np.atleast_1d(np.asarray(ts))
    fn = np.atleast_1d(np.asarray(fn))
    
    n_fragments = fn.size
    fb = ts  # fragment begin times
    
    # Calculate sample boundaries
    sb = np.concatenate(([0], np.cumsum(fn[:-1])))  # sample begins
    se = np.cumsum(fn)  # sample ends
    
    # Initialize output array
    times = np.zeros(samples.size)
    
    # Process each fragment
    for ff in range(n_fragments):  # Python uses 0-based indexing
        idx = (samples >= sb[ff]) & (samples <= se[ff])
        times[idx] = ((samples[idx] - sb[ff]) + 1) / adfreq + fb[ff]
    
    return times

class KilosortResults:
    def __init__(self, directory):
        if isinstance(directory, str):
            directory = Path(directory)
        assert isinstance(directory, Path), 'directory must be a string or Path object'
        assert directory.exists(), f'{directory} does not exist'
        assert directory.is_dir(), f'{directory} is not a directory'
        self.directory = directory

        # Move directory to sorter_output if it is a kilosort4 output directory
        if (directory / 'sorter_output').exists():
            directory = directory / 'sorter_output'

        self.spike_times_file = directory / 'spike_times.npy'
        assert self.spike_times_file.exists(), f'{self.spike_times_file} does not exist'
        self._spike_times = None # TODO: I don't like that spike "times" are actually samples 

        self.spike_amplitudes_file = directory / 'amplitudes.npy'
        assert self.spike_amplitudes_file.exists(), f'{self.spike_amplitudes_file} does not exist'
        self._spike_amplitudes = None

        self.st_file = directory / 'full_st.npy'
        if not self.st_file.exists():
            print(f'Warning: {self.st_file} does not exist. Use Kilosort4 with save_extra_vars=True to generate.')
        self.kept_spikes_file = directory / 'kept_spikes.npy'
        if not self.kept_spikes_file.exists():
            print(f'Warning: {self.kept_spikes_file} does not exist. Use Kilosort4 with save_extra_vars=True to generate.')

        self.spike_clusters_file = directory / 'spike_clusters.npy'
        assert self.spike_clusters_file.exists(), f'{self.spike_clusters_file} does not exist'

        self.spike_templates_file = directory / 'spike_templates.npy'
        assert self.spike_templates_file.exists(), f'{self.spike_templates_file} does not exist'

        self.cluster_labels_file = directory / 'cluster_KSLabel.tsv'
        assert self.cluster_labels_file.exists(), f'{self.cluster_labels_file} does not exist'

        # check if ephys_metadata.json exists two levels up
        ephys_metadata_file = directory / 'ephys_metadata.json'
        if ephys_metadata_file.exists():
            import json
            with open(ephys_metadata_file, 'r') as f:
                self.ephys_metadata = json.load(f)
        
    @cached_property
    def spike_times(self):
        '''
        This now properly returns times if that info is available
        '''
        spike_times = np.load(self.spike_times_file)
        if hasattr(self, 'ephys_metadata'):
            return convert_samples_to_time(spike_times, self.ephys_metadata['sample_rate'], self.ephys_metadata['block_start_times'], self.ephys_metadata['block_n_samples'])
        else:
            print('Warning: ephys_metadata not found. Returning samples instead of times.')

        return spike_times
    
    @cached_property
    def spike_samples(self):
        return np.load(self.spike_times_file)
    
    @cached_property
    def spike_amplitudes(self):
        return self.st[:,2]

    @cached_property
    def st(self): 
        st = np.load(self.st_file)
        spikes = np.load(self.kept_spikes_file)
        return st[spikes]
    
    @cached_property
    def spike_clusters(self):
        return np.load(self.spike_clusters_file)

    @cached_property
    def spike_templates(self):
        return np.load(self.spike_templates_file)

    @cached_property
    def cluster_labels(self):
        return pd.read_csv(self.cluster_labels_file, sep='\t')