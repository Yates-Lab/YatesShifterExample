#%%
# ============================================================================
# 1. IMPORT LIBRARIES AND SET UP ENVIRONMENT
# ============================================================================

# Standard scientific computing libraries
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import os
import copy

# PyTorch for deep learning
import torch
import torch.nn as nn
import torch.nn.functional as F

# Scipy for signal processing and interpolation
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d

# Custom utility modules for neural data analysis
from utils.loss import PoissonBPSAggregator
from utils.modules import SplitRelu, NonparametricReadout, StackedConv2d
from utils.datasets import DictDataset, generate_gaborium_dataset, generate_gratings_dataset
from utils.general import set_seeds, ensure_tensor
from utils.grid_sample import grid_sample_coords
from utils.rf import calc_sta, plot_stas
from utils.exp.general import get_clock_functions, get_trial_protocols
from utils.exp.dots import dots_rf_map_session
from utils.spikes import KilosortResults

# Matplotlib utilities
from matplotlib.patches import Circle

# Seaborn for advanced plotting
import seaborn as sns

# ============================================================================
# CONFIGURATION AND SETUP
# ============================================================================

# Set random seeds for reproducibility across runs
RANDOM_SEED = 1002
set_seeds(RANDOM_SEED)

# Configure PyTorch for optimal performance
torch.set_float32_matmul_precision('medium')  # Trade precision for speed

# Set device for computation (GPU if available, otherwise CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Define data directory path
data_dir = Path('/home/ryanress/YatesShifterExample/data')

#%%
# ============================================================================
# 2. LOAD SHIFTED DATASET
# ============================================================================

# Load experimental metadata and stimulus information
f_dset = data_dir / 'gaborium_corrected.dset'

gabor_dataset = DictDataset.load(f_dset)
gabor_dataset['stim'] = gabor_dataset['stim'].float()
gabor_dataset['stim'] = (gabor_dataset['stim'] - 127) / 255
print(gabor_dataset)

f_natural_images = data_dir / 'natural_images.dset'
ni_dataset = DictDataset.load(f_natural_images)
ni_dataset['stim'] = ni_dataset['stim'].float()
ni_dataset['stim'] = (ni_dataset['stim'] - 127) / 255
print(ni_dataset)

#%%
# plot a gif of 240 frames of stim from each dataset side by side
import matplotlib.animation as animation
from IPython.display import HTML

# Set up the figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.set_title('Gabor Dataset')
ax2.set_title('Natural Images Dataset')
ax1.axis('off')
ax2.axis('off')

# Number of frames to animate
n_frames = min(240, gabor_dataset['stim'].shape[0], ni_dataset['stim'].shape[0])

# Initialize images
im1 = ax1.imshow(gabor_dataset['stim'][0], cmap='gray', animated=True)
im2 = ax2.imshow(ni_dataset['stim'][0], cmap='gray', animated=True)

def animate(frame):
    """Update function for animation"""
    im1.set_array(gabor_dataset['stim'][frame])
    ax1.set_title(f'Gabor Dataset \n Eye Position: {gabor_dataset['eyepos'][frame,0]:.3f}, {gabor_dataset['eyepos'][frame,1]:.3f}')
    im2.set_array(ni_dataset['stim'][frame])
    ax2.set_title(f'Natural Images Dataset \n Eye Position: {ni_dataset['eyepos'][frame,0]:.3f}, {ni_dataset['eyepos'][frame,1]:.3f}')
    fig.suptitle(f'Time: {frame/240*1000:.1f} ms')
    return [im1, im2]

# Create animation
anim = animation.FuncAnimation(fig, animate, frames=n_frames, 
                              interval=50, blit=True, repeat=True)

plt.close(fig)
# Display in notebook
HTML(anim.to_jshtml())

#%%
# We are now going to try to model the cells responses to these two conditions using CNNs
# First, we are going to do QC to determine which cells to include
# Then, we are going to fit a CNN to each dataset separately and see how they generalize
# Finally, we are going to fit a CNN to both datasets together
#%%

def get_valid_inds(dset, n_lags):
    dpi_valid = dset['dpi_valid']
    new_trials = torch.diff(dset['trial_inds'], prepend=torch.tensor([-1])) != 0
    valid = ~new_trials
    valid &= (dpi_valid > 0)

    for iL in range(n_lags):
        valid &= torch.roll(valid, iL)
    
    inds = torch.where(valid)[0]

    return inds

n_lags = 25
n_units = gabor_dataset['robs'].shape[1]
n_y, n_x = gabor_dataset['stim'].shape[1:3]
gabor_inds = get_valid_inds(gabor_dataset, n_lags)
ni_inds = get_valid_inds(ni_dataset, n_lags)
#%%
# 1) QC - Refractory, Subthreshold Spikes, Visually Responsive
# 2) Train, Val, Test Split
# 3) Fit LQM
# 4) Fit CNN
# 5) Compare across good cells
# 6) 
# Quality thresholds for unit selection
SNR_THRESHOLD = 6  # Signal-to-noise ratio threshold

print("Analyzing neural response timing...")

# Calculate spike-triggered stimulus energy (STE)
# STE measures how much stimulus energy drives each neuron at different time lags
# This helps us find the optimal delay between stimulus and neural response
spike_triggered_energies = calc_sta(
    gabor_dataset['stim'], gabor_dataset['robs'],
    n_lags, inds=gabor_inds, device=device, batch_size=10000,
    stim_modifier=lambda x: x**2,  # Square stimulus to get energy
    progress=True
).cpu().numpy()

print(f"Computed STEs for {spike_triggered_energies.shape[0]} units, "
      f"{spike_triggered_energies.shape[1]} time lags")

# Find the optimal response lag for each neuron
# We look for the lag that gives the strongest, most reliable response
signal_strength = np.abs(spike_triggered_energies -
                        np.median(spike_triggered_energies, axis=(2,3), keepdims=True))

# Apply spatial smoothing to reduce noise
smoothing_kernel = [0, 2, 2, 2]  # No temporal smoothing, 2-pixel spatial smoothing
signal_strength = gaussian_filter(signal_strength, smoothing_kernel)

# Calculate signal-to-noise ratio for each unit and lag
noise_level = np.median(signal_strength[:, 0], axis=(1,2))  # Use first lag as noise estimate
signal_to_noise = np.max(signal_strength, axis=(1,2,3)) / noise_level

plt.figure()
plt.hist(signal_to_noise, bins=100)
plt.axvline(SNR_THRESHOLD, color='red', linestyle='--')
plt.xlabel('Signal-to-Noise Ratio')
plt.ylabel('Count')
plt.title('Distribution of Signal-to-Noise Ratios')
plt.show()

visually_responsive_units = np.where(signal_to_noise > SNR_THRESHOLD)[0]
print(f'{len(visually_responsive_units)} / {n_units} units are visually responsive')
#%%
# Refractory period

# Load spike sorting results from Kilosort4
ks4_dir = data_dir / 'Allen_2022-04-13_ks4'
ks_results = KilosortResults(ks4_dir)
spike_times = ks_results.spike_times  # When each spike occurred
spike_amplitudes = ks_results.spike_amplitudes
spike_clusters = ks_results.spike_clusters  # Which neuron each spike came from
cluster_ids = np.unique(spike_clusters)  # List of all neuron IDs
print(f"Loaded {len(cluster_ids)} units with {len(spike_times)} total spikes")

#%%
from utils.qc import compute_min_contam_props, plot_min_contam_prop

refractory_periods = np.exp(np.linspace(np.log(1e-3), np.log(10e-3), 100))
min_contam_props, firing_rates = compute_min_contam_props(
    spike_times, spike_clusters, refractory_periods=refractory_periods, ref_acg_t_start=.25e-3, progress=True)
#%%
contam_props = min_contam_props.min(axis=1)

plt.figure()
plt.hist(contam_props, bins=50)
plt.show()

for prop in [0, 20, 50, 80, 100]:
    closest_ind = np.argmin(np.abs(contam_props - prop/100))
    print(f'{prop}%: {closest_ind}')
    isis = np.diff(spike_times[spike_clusters == closest_ind]) * 1000

    fig, axs = plot_min_contam_prop(spike_times[spike_clusters == closest_ind], min_contam_props[closest_ind], refractory_periods)

    axs.set_title(f'Unit {closest_ind} - {len(isis)} spikes ({firing_rates[closest_ind]:.2f} Hz) - {contam_props[closest_ind]*100:.1f}% contamination')
    plt.show()

#%%
# Note, for this analysis we don't care about fitting MUA
CONTAMINATION_THRESHOLD = .75 #1 

included_units = np.intersect1d(np.where(contam_props <= CONTAMINATION_THRESHOLD)[0], visually_responsive_units)
print(f'{len(included_units)} / {len(cluster_ids)} units will be included in modeling')
gabor_dataset['robs'] = gabor_dataset['robs'][:, included_units]
ni_dataset['robs'] = ni_dataset['robs'][:, included_units]

#%%
example_unit = 5
fig, axs = plt.subplots(1, 2, figsize=(12, 5), width_ratios=[2, 1])
plt.subplot(121)
plt.hist2d(spike_times[spike_clusters == example_unit], spike_amplitudes[spike_clusters == example_unit], bins=(200, 50), cmap='Purples')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (a.u.)')
plt.title(f'Unit {example_unit}')
plt.subplot(122)
plt.hist(np.diff(spike_times[spike_clusters == example_unit])*1000, bins=np.linspace(0, 10, 50))
plt.xlabel('ISI (s)')
plt.show()

#%%

from utils.qc import analyze_amplitude_truncation, plot_amplitude_truncation

amplitude_results = []

for iU in cluster_ids:
    st_clu = spike_times[spike_clusters == iU]
    amp_clu = spike_amplitudes[spike_clusters == iU]

    window_blocks, valid_blocks, popts, mpcts = analyze_amplitude_truncation(st_clu, amp_clu, max_isi=np.inf)
    amplitude_results.append({
        'cid' : iU,
        'window_blocks' : window_blocks,
        'valid_blocks' : valid_blocks,
        'popts' : popts,
        'mpcts' : mpcts,
    })



#%%

example_unit = 5

fig, axs = plot_amplitude_truncation(
    spike_times[spike_clusters == example_unit],
    spike_amplitudes[spike_clusters == example_unit], 
    amplitude_results[example_unit]['window_blocks'],
    amplitude_results[example_unit]['valid_blocks'],
    amplitude_results[example_unit]['mpcts'],
)
axs.set_title(f'Unit {example_unit}')
plt.show()


#%%

MPCT_THRESHOLD = 30

gabor_dfs = torch.zeros_like(gabor_dataset['robs'])
ni_dfs = torch.zeros_like(ni_dataset['robs'])

for iU, uid in enumerate(included_units):
    st_clu = spike_times[spike_clusters == uid]
    window_blocks = np.array(amplitude_results[uid]['window_blocks']).flatten()
    if len(window_blocks) == 0:
        continue
    window_times = st_clu[window_blocks]
    
    mpcts = amplitude_results[uid]['mpcts']
    # if np.allclose(mpcts, 50):
    #     print(f'Unit {uid} is low amplitude, including as MUA')
    #     gabor_dfs[:, iU] = 1
    #     ni_dfs[:, iU] = 1
        #continue
    mpct_interpolant = interp1d(window_times, np.repeat(mpcts, 2), kind='linear', fill_value=50, bounds_error=False)
    mpct_bins = mpct_interpolant(gabor_dataset['t_bins'])
    gabor_dfs[:, iU] = torch.from_numpy(mpct_bins < MPCT_THRESHOLD).float()
    mpct_bins = mpct_interpolant(ni_dataset['t_bins'])
    ni_dfs[:, iU] = torch.from_numpy(mpct_bins < MPCT_THRESHOLD).float()

print(f'Fraction of bins passing MPCT threshold. Gabor: {gabor_dfs.mean().item():.3f}, NI: {ni_dfs.mean().item():.3f}')

gabor_dataset['dfs'] = gabor_dfs
ni_dataset['dfs'] = ni_dfs

#%%

MIN_SPIKE_COUNT = 500  # Minimum number of spikes required per unit
spikes_after_dfs_gabor = (gabor_dataset['robs'] * gabor_dataset['dfs']).sum(0)
spikes_after_dfs_ni = (ni_dataset['robs'] * ni_dataset['dfs']).sum(0)
good_units = np.where((spikes_after_dfs_gabor > MIN_SPIKE_COUNT) & (spikes_after_dfs_ni > MIN_SPIKE_COUNT))[0]
print(f'{len(good_units)} / {len(included_units)} units have enough spikes after MPCT')
gabor_dataset['robs'] = gabor_dataset['robs'][:, good_units]
gabor_dataset['dfs'] = gabor_dataset['dfs'][:, good_units]
ni_dataset['robs'] = ni_dataset['robs'][:, good_units]
ni_dataset['dfs'] = ni_dataset['dfs'][:, good_units]

#%%

stas = calc_sta(gabor_dataset['stim'] - gabor_dataset['stim'].mean(), gabor_dataset['robs'], 
                n_lags, inds=gabor_inds,device='cuda', batch_size=10000,
                progress=True).cpu().numpy()

plot_stas(stas[:, :, None, :, :])
#%%
stes = calc_sta(gabor_dataset['stim'], gabor_dataset['robs'], 
                n_lags,device='cuda', batch_size=10000,
                stim_modifier=lambda x: x**2, progress=True).cpu().numpy()

stes -= stes.mean(axis=(1, 2,3), keepdims=True)

plot_stas(stes[:, :, None, :, :])

#%%

from utils.datasets import CombinedEmbeddedDataset

keys_lags = {
    'robs': 0,
    'dfs': 0,
    'stim': np.arange(n_lags),
}

train_val_split = 0.8


def split_inds_by_trial(dset, inds, splits, seed=1002):
    '''
    Split indices by trial into training and validation sets.

    Parameters
    ----------
    dset : DictDataset
        The dataset containing trial indices.
    inds : torch.Tensor
        The indices to split.
    splits : list of floats
        The fractions of indices to use for each split. The sum of the splits should be 1 or less.
    seed : int, optional
        The random seed. The default is 1002.

    Returns
    -------
    list of torch.Tensor
        The indices for each split.
    '''

    assert np.sum(splits) <= 1, 'Splits must sum to 1 or less'

    set_seeds(seed)
    trials = dset['trial_inds'].unique()
    rand_trials = torch.randperm(len(trials))
    split_inds = [int(len(trials) * split) for split in splits]
    split_trials = [trials[rand_trials[sum(split_inds[:i]):sum(split_inds[:i+1])]] for i in range(len(split_inds))]
    split_inds = [inds[torch.isin(dset['trial_inds'][inds], split_trials[i])] for i in range(len(split_inds))]
    return split_inds


train_val_test_split = [.6, .2, .2]

gabor_train_inds, gabor_val_inds, gabor_test_inds = split_inds_by_trial(gabor_dataset, gabor_inds, train_val_test_split)
print(f'Gaborium sample split: {len(gabor_train_inds) / len(gabor_inds):.3f} train, {len(gabor_val_inds) / len(gabor_inds):.3f} val, {len(gabor_test_inds) / len(gabor_inds):.3f} test')

gabor_train_dataset = CombinedEmbeddedDataset(gabor_dataset,
                                    gabor_train_inds,
                                    keys_lags,
                                    'cpu')


gabor_val_dataset = CombinedEmbeddedDataset(gabor_dataset,
                                   gabor_val_inds,
                                   keys_lags,
                                   'cpu')

gabor_test_dataset = CombinedEmbeddedDataset(gabor_dataset,
                                   gabor_test_inds,
                                   keys_lags,
                                   'cpu')


ni_train_inds, ni_val_inds, ni_test_inds = split_inds_by_trial(ni_dataset, ni_inds, train_val_test_split)
print(f'Natural images sample split: {len(ni_train_inds) / len(ni_inds):.3f} train, {len(ni_val_inds) / len(ni_inds):.3f} val, {len(ni_test_inds) / len(ni_inds):.3f} test')

ni_train_dataset = CombinedEmbeddedDataset(ni_dataset,
                                    ni_train_inds,
                                    keys_lags,
                                    'cpu')

ni_val_dataset = CombinedEmbeddedDataset(ni_dataset,
                                   ni_val_inds,
                                   keys_lags,
                                   'cpu')

ni_test_dataset = CombinedEmbeddedDataset(ni_dataset,
                                   ni_test_inds,
                                   keys_lags,
                                   'cpu')

both_train_dataset = CombinedEmbeddedDataset([gabor_dataset, ni_dataset],
                                    [gabor_train_inds, ni_train_inds],
                                    keys_lags,
                                    'cpu')

both_val_dataset = CombinedEmbeddedDataset([gabor_dataset, ni_dataset],
                                    [gabor_val_inds, ni_val_inds],
                                    keys_lags,
                                    'cpu')

both_test_dataset = CombinedEmbeddedDataset([gabor_dataset, ni_dataset],
                                    [gabor_test_inds, ni_test_inds],
                                    keys_lags,
                                    'cpu')

#%%
test_batch = both_train_dataset[:64]

for k, v in test_batch.items():
    print(k, v.shape)
# %%

from utils.modules import WindowedConv2d, SplitRelu
from utils.reg import laplacian, locality_conv, local_2d

class SpatioTemporalResNet(nn.Module):
    '''
    A spatiotemporal cnn model with time only in the first layer.
    '''

    def __init__(self, n_lags, n_y, n_x, n_units, temporal_channels, res_channels, kernel_size, n_layers, baseline_rates=None):
        super(SpatioTemporalResNet, self).__init__()

        self.temporal_layer = nn.Conv2d(n_lags, temporal_channels, kernel_size=1, bias=False) 
        self.temporal_activation = SplitRelu()
        self.layers = nn.ModuleList()
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        
        for iC in range(n_layers):
            self.layers.append(WindowedConv2d(temporal_channels*2 if iC == 0 else res_channels, 
                                              res_channels, kernel_size=kernel_size, bias=True))

        # Add projection layer if temporal_channels != res_channels
        if temporal_channels != res_channels:
            self.channel_projection = nn.Conv2d(temporal_channels*2, res_channels, kernel_size=1, bias=False)
        else:
            self.channel_projection = None

        contraction = (kernel_size - 1) * n_layers
        output_dims = [res_channels, n_y - contraction, n_x - contraction]
        self.readout = NonparametricReadout(output_dims, n_units, bias=True)

        inv_softplus = lambda x, beta=1: torch.log(torch.exp(beta*x) - 1) / beta
        if baseline_rates is not None:
            assert len(baseline_rates) == n_units, 'init_rates must have the same length as n_units'
            self.readout.bias.data = inv_softplus(
                ensure_tensor(baseline_rates, device=self.readout.bias.device)
            )

    def forward(self, batch, debug=False):
        x = batch['stim']
        if debug:
            print(f'Input: {x.shape}')
        x = self.temporal_layer(x)
        x = self.temporal_activation(x)
        if debug:
            print(f'Temporal: {x.shape}')

        # Store the first layer input for residual connections
        residual = x
        if self.channel_projection is not None:
            residual = self.channel_projection(residual)

        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = F.relu(x)
            
            # Add residual connection (crop residual to match current x size)
            if residual.shape[-2:] != x.shape[-2:]:
                # Calculate how much to crop from each side
                h_diff = residual.shape[-2] - x.shape[-2]
                w_diff = residual.shape[-1] - x.shape[-1]
                h_crop = h_diff // 2
                w_crop = w_diff // 2
                residual = residual[..., h_crop:h_crop+x.shape[-2], w_crop:w_crop+x.shape[-1]]
            
            x = x + residual
            residual = x  # Update residual for next layer
            
            if debug:
                print(f'Layer {i+1}: {x.shape}')
                
        x = self.readout(x)
        if debug:
            print(f'Readout: {x.shape}')
        batch['rhat'] = F.softplus(x)
        return batch
    
    def temporal_smoothness_regularization(self):
        return laplacian(self.temporal_layer.weight, dims=1)
    
    def plot_weights(self):
        temporal_weights = self.temporal_layer.weight.detach().cpu().numpy()
        plt.figure()
        plt.plot(temporal_weights.squeeze().T)
        plt.title('Temporal Weights')
        plt.show()
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'plot_weights'):
                fig, axs = layer.plot_weights()
                fig.suptitle(f'Layer {i+1}')
                plt.show()
        fig, axs = self.readout.plot_weights()
        fig.suptitle(f'Readout')
        plt.show()

# Update the model instantiation
_, n_y, n_x = gabor_dataset['stim'].shape
n_units = gabor_dataset['robs'].shape[1]
temporal_channels = 4
res_channels = 8
kernel_size = 13
n_layers = 4

baseline_rates = (gabor_dataset['robs'][gabor_train_inds] * gabor_dataset['dfs'][gabor_train_inds]).sum(0) / gabor_dataset['dfs'][gabor_train_inds].sum(0)

model = SpatioTemporalResNet(n_lags, n_y, n_x, n_units, temporal_channels, res_channels, kernel_size, n_layers, baseline_rates)

batch = model(test_batch, True)
model.plot_weights()

#%%

class SpatioTemporalCNN(nn.Module):
    '''
    A spatiotemporal cnn model with time only in the first layer.
    '''

    def __init__(self, n_lags, n_y, n_x, n_units, channels, activations, kernel_sizes, baseline_rates=None):
        super(SpatioTemporalCNN, self).__init__()

        self.temporal_layer = nn.Conv2d(n_lags, channels[0], kernel_size=1, bias=False) 
        self.layers = nn.ModuleList()
        self.layers.append(activations[0])
        for iC in range(len(channels) - 1):
            prev_channels = channels[iC] * 2 if isinstance(activations[iC], SplitRelu) else channels[iC]
            self.layers.append(WindowedConv2d(prev_channels, channels[iC+1], kernel_size=kernel_sizes[iC], bias=False))
            self.layers.append(activations[iC+1])

        contraction = np.sum(kernel_sizes) - len(kernel_sizes) 
        output_dims = [channels[-1], n_y - contraction, n_x - contraction]
        if isinstance(activations[-1], SplitRelu):
            output_dims[0] *= 2
        self.readout = NonparametricReadout(output_dims, n_units, bias=True)

        inv_softplus = lambda x, beta=1: torch.log(torch.exp(beta*x) - 1) / beta
        if baseline_rates is not None:
            assert len(baseline_rates) == n_units, 'init_rates must have the same length as n_units'
            self.readout.bias.data = inv_softplus(
                ensure_tensor(baseline_rates, device=self.readout.bias.device)
            )

    def forward(self, batch, debug=False):
        x = batch['stim']
        if debug:
            print(f'Input: {x.shape}')
        x = self.temporal_layer(x)
        if debug:
            print(f'Temporal: {x.shape}')

        for layer in self.layers:
            x = layer(x)
            if debug:
                print(f'Layer {layer}: {x.shape}')
        x = self.readout(x)
        if debug:
            print(f'Readout: {x.shape}')
        batch['rhat'] = F.softplus(x)
        return batch

    def temporal_smoothness_regularization(self):
        return laplacian(self.temporal_layer.weight, dims=1)
    
    def plot_weights(self):
        temporal_weights = self.temporal_layer.weight.detach().cpu().numpy()
        plt.figure()
        plt.plot(temporal_weights.squeeze().T)
        plt.title('Temporal Weights')
        plt.show()
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'plot_weights'):
                fig, axs = layer.plot_weights()
                fig.suptitle(f'Layer {i+1}')
                plt.show()
        fig, axs = self.readout.plot_weights()
        fig.suptitle(f'Readout')
        plt.show()
        
channels = [4, 8, 8]
activations = [SplitRelu(), SplitRelu(), SplitRelu()]
kernel_sizes = [16, 16]

model = SpatioTemporalCNN(n_lags, n_y, n_x, n_units, channels, activations, kernel_sizes, baseline_rates)

batch = model(test_batch, True)

model.plot_weights()
# %%
# ============================================================================
# 3. TRAINING FUNCTION
# ============================================================================

def masked_poisson_nll_loss(output, target, dfs=None):
    """
    Compute masked Poisson negative log-likelihood loss.

    Parameters
    ----------
    output : torch.Tensor
        Model predictions
    target : torch.Tensor
        Target values
    dfs : torch.Tensor, optional
        Degrees of freedom mask for weighting samples

    Returns
    -------
    torch.Tensor
        Computed loss
    """
    loss = F.poisson_nll_loss(output, target, log_input=False, full=False, reduction='none')
    if dfs is not None:
        if dfs.shape[1] == 1:
            dfs = dfs.expand(-1, loss.shape[1])
        loss = loss * dfs
        loss = loss.sum() / dfs.sum()
    else:
        loss = loss.mean()
    return loss


def train_model(model, train_dataset, val_dataset,
                n_epochs=10, lr=3e-3, weight_decay=1e-4,
                smoothness_lambda=1e-4, batch_size=256, patience=2,
                device='cuda', num_workers=None, plot_weights=True,
                verbose=True):
    """
    Train a spatiotemporal model on neural data.

    Parameters
    ----------
    model : nn.Module
        The model to train (SpatioTemporalResNet or SpatioTemporalCNN)
    train_dataset : Dataset
        Training dataset
    val_dataset : Dataset
        Validation dataset
    n_epochs : int, optional
        Number of training epochs. Default is 10.
    lr : float, optional
        Learning rate. Default is 3e-3.
    weight_decay : float, optional
        Weight decay for optimizer. Default is 1e-4.
    smoothness_lambda : float, optional
        Regularization strength for temporal smoothness. Default is 1e-4.
    batch_size : int, optional
        Batch size for training. Default is 256.
    patience : int, optional
        Early stopping patience. Default is 2.
    device : str, optional
        Device to train on. Default is 'cuda'.
    num_workers : int, optional
        Number of workers for data loading. Default is os.cpu_count()//2.
    plot_weights : bool, optional
        Whether to plot model weights each epoch. Default is True.
    verbose : bool, optional
        Whether to print training progress. Default is True.

    Returns
    -------
    dict
        Dictionary containing:
        - 'model': trained model with best weights loaded
        - 'train_losses': list of training losses per epoch
        - 'val_losses': list of validation losses per epoch
        - 'train_bps': list of training bits per spike per epoch
        - 'val_bps': list of validation bits per spike per epoch
        - 'step_losses': list of losses per training step
        - 'step_numbers': list of step numbers
        - 'best_epoch': epoch with best validation loss
    """
    if num_workers is None:
        num_workers = os.cpu_count() // 2

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # Move model to device
    model = model.to(device)

    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Initialize BPS aggregator
    bps_agg = PoissonBPSAggregator()

    # Lists to store metrics
    train_losses = []
    train_bps = []
    val_losses = []
    val_bps = []
    step_losses = []
    step_numbers = []

    # Early stopping variables
    best_val_loss = np.inf
    best_state = None
    patience_count = 0
    step = 0

    if verbose:
        print(f"Starting training for {n_epochs} epochs...")
        print(f"Training batches per epoch: {len(train_loader)}")

    # Training loop
    for epoch in range(n_epochs):
        if verbose:
            print(f"\nEpoch {epoch+1}/{n_epochs}")

        if plot_weights and hasattr(model, 'plot_weights'):
            model.plot_weights()

        # Training phase
        model.train()
        epoch_train_losses = []
        bps_agg.reset()

        train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}", disable=not verbose)
        for batch in train_pbar:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            optimizer.zero_grad()
            batch = model(batch)

            # Accumulate for BPS calculation
            bps_agg(batch)

            # Calculate loss
            loss = masked_poisson_nll_loss(batch['rhat'], batch['robs'], batch['dfs'])

            # Add smoothness regularization if model supports it
            if hasattr(model, 'temporal_smoothness_regularization'):
                loss += smoothness_lambda * model.temporal_smoothness_regularization()

            # Backward pass
            loss.backward()
            optimizer.step()

            # Store loss for plotting
            step_loss = loss.item()
            step_losses.append(step_loss)
            step_numbers.append(step)
            epoch_train_losses.append(step_loss)

            # Update progress bar
            if verbose:
                train_pbar.set_postfix({'loss': f'{step_loss:.4f}'})
            step += 1

        # Calculate average training loss for epoch
        avg_train_loss = np.mean(epoch_train_losses)
        train_losses.append(avg_train_loss)
        train_bps.append(bps_agg.closure().cpu().numpy())
        bps_agg.reset()

        if verbose:
            print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Train BPS: {train_bps[-1].mean():.4f}")

        # Validation phase
        model.eval()
        val_loss_total = 0
        val_samples = 0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}", disable=not verbose)
            for batch in val_pbar:
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}

                # Forward pass
                batch = model(batch)

                # Calculate validation loss
                val_loss = masked_poisson_nll_loss(batch['rhat'], batch['robs'], batch['dfs'])

                n_samples = batch['dfs'].sum().item()
                val_loss_total += val_loss.item() * n_samples
                val_samples += n_samples

                # Accumulate for BPS calculation
                bps_agg(batch)

                # Update progress bar
                if verbose:
                    val_pbar.set_postfix({'loss': f'{val_loss.item():.4f}'})

        # Calculate validation metrics
        avg_val_loss = val_loss_total / val_samples if val_samples > 0 else 0
        val_losses.append(avg_val_loss)

        # Calculate validation BPS
        val_bps.append(bps_agg.closure().cpu().numpy())
        bps_agg.reset()

        if verbose:
            print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val BPS: {val_bps[-1].mean():.4f}")

        # Early stopping
        if epoch > 0 and val_losses[-1] >= best_val_loss:
            patience_count += 1
            if verbose:
                print(f"❌ Validation loss did not improve from {val_losses[-2]:.4f} to {val_losses[-1]:.4f}")
            if patience_count >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}: No improvement in validation loss for {patience} epochs")
                break
        else:
            best_val_loss = val_losses[-1]
            patience_count = 0
            if epoch > 0 and verbose:
                print(f"✅ Validation loss improved from {val_losses[-2]:.4f} to {val_losses[-1]:.4f}")
            best_state = copy.deepcopy(model.state_dict())

    # Load best model state
    best_epoch = np.argmin(val_losses)
    if verbose:
        print("\nTraining completed!")
        print(f"Best validation bits per spike: {val_bps[best_epoch].mean():.4f} on epoch {best_epoch+1}")
        print(f'Loading best model from epoch {best_epoch+1}')

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_bps': train_bps,
        'val_bps': val_bps,
        'step_losses': step_losses,
        'step_numbers': step_numbers,
        'best_epoch': best_epoch
    }

def plot_training_summary(training_results):
    # Extract results
    train_losses = training_results['train_losses']
    val_losses = training_results['val_losses']
    train_bps = training_results['train_bps']
    val_bps = training_results['val_bps']
    step_losses = training_results['step_losses']
    step_numbers = training_results['step_numbers']
    # Create plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Training Results', fontsize=16)

    # Plot 1: Training loss per step
    axes[0].plot(step_numbers, step_losses, alpha=0.7, linewidth=0.5)
    axes[0].set_xlabel('Training Step')
    axes[0].set_ylabel('Poisson NLL Loss')
    axes[0].set_title('Training Loss per Step')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Training and validation loss
    axes[1].plot(range(1, len(train_losses) + 1), train_losses, 'b-o', label='Training Loss')
    axes[1].plot(range(1, len(val_losses) + 1), val_losses, 'r-o', label='Validation Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Average Poisson NLL Loss')
    axes[1].set_title('Training vs Validation Loss')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Plot 3: Training and validation bits per spike
    axes[2].plot(range(1, len(train_bps) + 1), np.mean(np.array(train_bps), axis=1), 'b-o', label='Training BPS')
    axes[2].plot(range(1, len(val_bps) + 1), np.mean(np.array(val_bps), axis=1), 'r-o', label='Validation BPS')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Bits per Spike')
    axes[2].set_title('Training vs Validation BPS')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plt.show()




# %%
# ============================================================================
# 3. EXAMPLE USAGE OF TRAINING FUNCTION
# ============================================================================

# Training parameters
training_config = {
    'n_epochs': 10,
    'lr': 3e-3,
    'weight_decay': 1e-4,
    'smoothness_lambda': 1e-4,
    'batch_size': 256,
    'patience': 2,
    'device': device,
    'plot_weights': True,
    'verbose': True
}

# Create model
gabor_model = SpatioTemporalResNet(n_lags, n_y, n_x, n_units, temporal_channels, res_channels, kernel_size, n_layers, baseline_rates)
gabor_results = train_model(
    model=gabor_model,
    train_dataset=gabor_train_dataset,
    val_dataset=gabor_val_dataset,
    **training_config
)
plot_training_summary(gabor_results)

#%%
ni_model = SpatioTemporalResNet(n_lags, n_y, n_x, n_units, temporal_channels, res_channels, kernel_size, n_layers, baseline_rates)
ni_results = train_model(
    model=ni_model,
    train_dataset=ni_train_dataset,
    val_dataset=ni_val_dataset,
    **training_config
)
plot_training_summary(ni_results)

#%%
both_model = SpatioTemporalResNet(n_lags, n_y, n_x, n_units, temporal_channels, res_channels, kernel_size, n_layers, baseline_rates)
both_results = train_model(
    model=both_model,
    train_dataset=both_train_dataset,
    val_dataset=both_val_dataset,
    **training_config
)
plot_training_summary(both_results)


#%%
# ============================================================================
# 4. PLOTTING RESULTS
# ============================================================================

plot_training_summary(gabor_results)
plot_training_summary(ni_results)
plot_training_summary(both_results)
# %%
gabor_model = gabor_results['model']
gabor_model.plot_weights()
ni_model = ni_results['model']
ni_model.plot_weights()
both_model = both_results['model']
both_model.plot_weights()

#%%
# ============================================================================
# 5. MODEL EVALUATION ON TEST SETS
# ============================================================================

def evaluate_model_on_dataset(model, dataset, batch_size=256, device='cuda', desc="Evaluating"):
    """
    Evaluate a model on a dataset and return BPS per unit.

    Parameters
    ----------
    model : nn.Module
        The trained model to evaluate
    dataset : CombinedEmbeddedDataset
        The dataset to evaluate on
    batch_size : int, optional
        Batch size for evaluation. Default is 256.
    device : str, optional
        Device to run evaluation on. Default is 'cuda'.
    desc : str, optional
        Description for progress bar. Default is "Evaluating".

    Returns
    -------
    numpy.ndarray
        Bits per spike for each unit
    """
    model.eval()
    bps_aggregator = PoissonBPSAggregator()

    # Create data loader
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            batch = model(batch)

            # Accumulate for BPS calculation
            bps_aggregator(batch)

    # Calculate final BPS
    bps = bps_aggregator.closure().cpu().numpy()
    bps_aggregator.reset()

    return bps

# Evaluate all three models on both test sets
print("Evaluating models on test sets...")
print("=" * 60)

# Dictionary to store all results
evaluation_results = {}

# Model names for cleaner plotting
model_names = ['Gabor Model', 'NI Model', 'Both Model']
models = [gabor_model, ni_model, both_model]
test_datasets = {'Gabor Test': gabor_test_dataset, 'NI Test': ni_test_dataset}

# Evaluate each model on each test set
for model_idx, (model_name, model) in enumerate(zip(model_names, models)):
    evaluation_results[model_name] = {}
    print(f"\nEvaluating {model_name}...")

    for test_name, test_dataset in test_datasets.items():
        print(f"  On {test_name}...")
        bps = evaluate_model_on_dataset(
            model, test_dataset,
            batch_size=256, device=device,
            desc=f"{model_name} on {test_name}"
        )
        evaluation_results[model_name][test_name] = bps
        print(f"    Mean BPS: {bps.mean():.4f} ± {bps.std():.4f}")

print("\n" + "=" * 60)
print("Evaluation completed!")

#%%
# ============================================================================
# 6. VISUALIZATION OF BPS DISTRIBUTIONS
# ============================================================================

def plot_bps_distributions(evaluation_results, min_bps=-10, save_path=None):
    """
    Plot BPS distributions for all models and test sets.

    Parameters
    ----------
    evaluation_results : dict
        Dictionary containing BPS results for each model and test set
    save_path : str, optional
        Path to save the plot. If None, plot is displayed.
    """
    # Set up the figure
    fig, axes = plt.subplots(1, 2, figsize=(8, 8))

    # Colors for different models
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    model_names = list(evaluation_results.keys())
    test_names = ['Gabor Test', 'NI Test']

    # Plot box plots with swarm plots for each test set
    for test_idx, test_name in enumerate(test_names):
        ax = axes[test_idx]

        # Prepare data for matplotlib boxplot and seaborn swarmplot
        bps_data = [evaluation_results[model_name][test_name] for model_name in model_names]
        positions = np.arange(-.4, len(model_names) - .4)
        
        # Plot matplotlib box plot first (behind)
        box_parts = ax.boxplot(bps_data, positions=positions, patch_artist=True, widths=.2)
        
        for median in box_parts['medians']:
            median.set_color('red')
            median.set_linewidth(2)
            median.set_alpha(0.6)
            median.set_zorder(10)
        
        for whisker in box_parts['whiskers']:
            whisker.set_linewidth(2)
            whisker.set_alpha(0.6)

        # Color the boxes
        for patch, color in zip(box_parts['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        
        # Prepare data for seaborn swarmplot

        plot_data = []
        for model_name in model_names:
            bps_values = evaluation_results[model_name][test_name]
            for bps in bps_values:
                plot_data.append({'Model': model_name, 'BPS': bps})
        
        plot_df = pd.DataFrame(plot_data)
        
        # Plot swarm plot on top
        # turn off warnings
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sns.swarmplot(data=plot_df, x='Model', y='BPS', palette=colors, ax=ax, size=5)
        
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5, label='Chance Level')
        if test_idx == 0:
            ax.legend()
        
        ax.set_ylabel('BPS (bits per spike)')
        ax.set_title(f'Model Performance on {test_name}')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-.8, 2.4])
        ax.set_xticks(np.arange(-.2, len(model_names) - .2))
        ax.set_xticklabels(model_names)
        ax.set_xlabel('')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()

    y_lims = [ax.get_ylim() for ax in axes]
    y_min = min([lim[0] for lim in y_lims])
    y_min = max(y_min, min_bps)
    y_max = max([lim[1] for lim in y_lims])
    for ax in axes:
        ax.set_ylim([y_min, y_max])
    

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()

# Create the visualization
plot_bps_distributions(evaluation_results)

#%%
# The primary takeaway is that CNN models do not generalize across conditions, at least for this dataset




