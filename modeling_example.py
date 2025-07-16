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

dataset = DictDataset.load(f_dset)
dataset['stim'] = dataset['stim'].float()
dataset['stim'] = (dataset['stim'] - dataset['stim'].mean()) / 255
dataset['stim'] = dataset['stim'][:,15:-15,15:-15] # Cropping to speed up computations

print(dataset)
n_lags = 25


#%%

def get_inds(dset, n_lags):
    dpi_valid = dset['dpi_valid']
    new_trials = torch.diff(dset['trial_inds'], prepend=torch.tensor([-1])) != 0
    dfs = ~new_trials
    dfs &= (dpi_valid > 0)

    for iL in range(n_lags):
        dfs &= torch.roll(dfs, iL)
    
    dfs = dfs.float()
    dfs = dfs[:, None]
    return dfs

n_units = dataset['robs'].shape[1]
n_y, n_x = dataset['stim'].shape[1:3]
gaborium_inds = get_inds(dataset, n_lags).squeeze().nonzero(as_tuple=True)[0]
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
    dataset['stim'], dataset['robs'],
    n_lags, inds=gaborium_inds, device=device, batch_size=10000,
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
CONTAMINATION_THRESHOLD = 0.6

included_units = np.intersect1d(np.where(contam_props < 0.8)[0], visually_responsive_units)
print(f'{len(included_units)} / {len(cluster_ids)} units will be included in modeling')

dataset['robs'] = dataset['robs'][:, included_units]

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
def plot_amplitude_truncation(spike_times, spike_amplitudes, 
                              window_blocks, valid_blocks, mpcts, axs=None):
    '''
    Create visualization of amplitude truncation analysis.
    
    Parameters
    ----------
    spike_times : array-like
        Spike timing data
    spike_amplitudes : array-like
        Spike amplitude data
    window_blocks : array-like
        Window boundary indices
    valid_blocks : array-like
        Valid block boundary indices
    mpcts : array-like
        Missing percentages for each window
        
    Returns
    -------
    tuple
        (figure handle, axes handles)
    '''
    window_block_times = np.array([[spike_times[i0], spike_times[i1]] for i0, i1 in window_blocks])
    if window_block_times.ndim == 1:
        window_block_times = window_block_times[np.newaxis, :]

    # Create mask for valid regions
    valid_mask = np.zeros(len(spike_times), dtype=bool)
    for i0, i1 in valid_blocks:
        valid_mask[i0:i1] = True

    # Create figure with two subplots
    if axs is None:
        fig, axs = plt.subplots(1, 1, figsize=(8, 6))
    else:
        fig = axs.get_figure()
    
    # Plot amplitude vs time
    xlim = [spike_times[0], spike_times[-1]]
    axs.hist2d(spike_times, spike_amplitudes, bins=(200, 50), cmap='Blues')
    axs.set_xlabel('Time (s)')
    axs.set_ylabel('Amplitude (a.u.)')
    axs.set_title(f'Amplitude vs Time')
    axs.set_xlim(xlim)

    axs2 = axs.twinx()
    for pct, (i0, i1) in zip(mpcts, window_blocks):
        axs2.plot([spike_times[i0], spike_times[i1]], [pct, pct], color='red', linewidth=3.5)

    axs2.set_ylim([0, 52])
    axs2.set_yticks([0, 50])
    axs2.set_yticklabels(['0%', '50%'])
    axs2.tick_params(axis='y', colors='red')
    axs2.set_ylabel('Missing %', color='red')
    return fig, axs

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


MPCT_THRESHOLD = 40

dfs = torch.zeros_like(dataset['robs'])

for iU, uid in enumerate(included_units):
    st_clu = spike_times[spike_clusters == uid]
    window_blocks = np.array(amplitude_results[uid]['window_blocks']).flatten()
    if len(window_blocks) == 0:
        continue
    window_times = st_clu[window_blocks]
    
    mpcts = amplitude_results[uid]['mpcts']
    mpct_interpolant = interp1d(window_times, np.repeat(mpcts, 2), kind='linear', fill_value=50, bounds_error=False)
    mpct_bins = mpct_interpolant(dataset['t_bins'])
    dfs[:, iU] = torch.from_numpy(mpct_bins < MPCT_THRESHOLD).float()

print(f'Fraction of bins passing MPCT threshold: {dfs.float().mean().item():.3f}')

dataset['dfs'] = dfs

#%%

MIN_SPIKE_COUNT = 500  # Minimum number of spikes required per unit
spikes_after_dfs = (dataset['robs'] * dataset['dfs']).sum(0)
print(spikes_after_dfs)
good_units = np.where(spikes_after_dfs > MIN_SPIKE_COUNT)[0]
print(f'{len(good_units)} / {len(included_units)} units have enough spikes after MPCT')
dataset['robs'] = dataset['robs'][:, good_units]
dataset['dfs'] = dataset['dfs'][:, good_units]



#%%

# stas = calc_sta(dataset['stim'], dataset['robs'], 
#                 n_lags,device='cuda', batch_size=10000,
#                 progress=True).cpu().numpy()

# plot_stas(stas[:, :, None, :, :])
# #%%
# stes = calc_sta(dataset['stim'], dataset['robs'], 
#                 n_lags,device='cuda', batch_size=10000,
#                 stim_modifier=lambda x: x**2, progress=True).cpu().numpy()

# stes -= stes.mean(axis=(1, 2,3), keepdims=True)

# plot_stas(stes[:, :, None, :, :])


#%%

#%%

from utils.datasets import CombinedEmbeddedDataset

keys_lags = {
    'robs': 0,
    'dfs': 0,
    'stim': np.arange(n_lags),
}

train_val_split = 0.8


def split_inds_by_trial(dset, inds, train_val_split, seed=1002):
    '''
    Split indices by trial into training and validation sets.

    Parameters
    ----------
    dset : DictDataset
        The dataset containing trial indices.
    inds : torch.Tensor
        The indices to split.
    train_val_split : float
        The fraction of indices to use for training.
    seed : int, optional
        The random seed. The default is 1002.

    Returns
    -------
    train_inds : torch.Tensor
        The indices for training.
    val_inds : torch.Tensor
        The indices for validation.
    '''

    set_seeds(seed)
    trials = dset['trial_inds'].unique()
    rand_trials = torch.randperm(len(trials))
    train_trials = trials[rand_trials[:int(len(trials) * train_val_split)]]
    train_inds = inds[torch.isin(dset['trial_inds'][inds], train_trials)]
    val_trials = trials[rand_trials[int(len(trials) * train_val_split):]]
    val_inds = inds[torch.isin(dset['trial_inds'][inds], val_trials)]
    return train_inds, val_inds


gaborium_train_inds, gaborium_val_inds = split_inds_by_trial(dataset, gaborium_inds, train_val_split, seed=1002)
print(f'Gaborium sample split: {len(gaborium_train_inds) / len(gaborium_inds):.3f} train, {len(gaborium_val_inds) / len(gaborium_inds):.3f} val')

train_dataset = CombinedEmbeddedDataset(dataset,
                                    gaborium_train_inds,
                                    keys_lags,
                                    'cpu')

val_dataset = CombinedEmbeddedDataset(dataset,
                                   gaborium_val_inds,
                                   keys_lags,
                                   'cpu')

#%%

class LinearQuadraticModel(nn.Module):
    def __init__(self, n_lags, n_units, baseline_rates=None):
        super(LinearQuadraticModel, self).__init__()
        



#%%
batch_size = 256

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count()//2)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count()//2)

#%%
batch = next(iter(train_loader))

for k, v in batch.items():
    print(k, v.shape)
# %%

from utils.modules import WindowedConv2d, SplitRelu
from utils.reg import laplacian, locality_conv, local_2d

class SpatiotemporalCNNModel(nn.Module):
    '''
    A spatiotemporal cnn model with time only in the first layer.
    '''

    def __init__(self, n_lags, n_y, n_x, n_units, channels, activations, kernel_sizes, baseline_rates=None):
        super(SpatiotemporalCNNModel, self).__init__()

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
        
_, n_y, n_x = dataset['stim'].shape
n_units = dataset['robs'].shape[1]
channels = [4, 8, 8]
activations = [SplitRelu(), SplitRelu(), SplitRelu()]
kernel_sizes = [16, 16]

baseline_rates = dataset['robs'][gaborium_train_inds].mean(0)

model = SpatiotemporalCNNModel(n_lags, n_y, n_x, n_units, channels, activations, kernel_sizes, baseline_rates)

batch = model(batch, True)

model.plot_weights()

# %%
# ============================================================================
# 3. TRAINING LOOP WITH PLOTTING
# ============================================================================

n_epochs = 10
lr = 1e-3
wd = 1e-4

# Move model to device
model = SpatiotemporalCNNModel(n_lags, n_y, n_x, n_units, channels, activations, kernel_sizes, baseline_rates)
model = model.to(device)

# Setup optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
loss_fn = nn.PoissonNLLLoss(log_input=False, full=False, reduction='none')
bps_agg = PoissonBPSAggregator()

# Lists to store losses for plotting
train_losses = []
train_bps = []
val_losses = []
val_bps = []
step_losses = []
step_numbers = []

print(f"Starting training for {n_epochs} epochs...")
print(f"Training batches per epoch: {len(train_loader)}")
print(f"Validation batches per epoch: {len(val_loader)}")

# Training loop
step = 0
for epoch in range(n_epochs):
    print(f"\nEpoch {epoch+1}/{n_epochs}")

    model.plot_weights()

    # Training phase
    model.train()
    epoch_train_losses = []
    bps_agg.reset()

    train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
    for batch_idx, batch in enumerate(train_pbar):
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass
        optimizer.zero_grad()
        batch = model(batch)

        # Accumulate for BPS calculation
        bps_agg(batch)

        # Calculate loss
        loss = loss_fn(batch['rhat'], batch['robs'])

        # Apply mask if available
        if 'dfs' in batch:
            dfs = batch['dfs']
            if dfs.shape[1] == 1:
                dfs = dfs.expand(-1, loss.shape[1])
            loss = loss * dfs
            loss = loss.sum() / dfs.sum()
        else:
            loss = loss.mean()

        # Backward pass
        loss.backward()
        optimizer.step()

        # Store loss for plotting
        step_loss = loss.item()
        step_losses.append(step_loss)
        step_numbers.append(step)
        epoch_train_losses.append(step_loss)

        # Update progress bar
        train_pbar.set_postfix({'loss': f'{step_loss:.4f}'})
        step += 1

    # Calculate average training loss for epoch
    avg_train_loss = np.mean(epoch_train_losses)
    train_losses.append(avg_train_loss)
    train_bps.append(bps_agg.closure().mean().item())
    bps_agg.reset()
    print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Train BPS: {train_bps[-1]:.4f}")

    # Validation phase
    model.eval()
    val_loss_total = 0
    val_samples = 0

    with torch.no_grad():
        val_pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}")
        for batch in val_pbar:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            batch = model(batch)

            # Calculate validation loss
            val_loss = loss_fn(batch['rhat'], batch['robs'])

            # Apply mask if available
            if 'dfs' in batch:
                dfs = batch['dfs']
                if dfs.shape[1] == 1:
                    dfs = dfs.expand(-1, val_loss.shape[1])
                val_loss = val_loss * dfs
                val_loss_total += val_loss.sum().item()
                val_samples += dfs.sum().item()
            else:
                val_loss_total += val_loss.sum().item()
                val_samples += val_loss.numel()

            # Accumulate for BPS calculation
            bps_agg(batch)

    # Calculate validation metrics
    avg_val_loss = val_loss_total / val_samples if val_samples > 0 else 0
    val_losses.append(avg_val_loss)

    # Calculate validation BPS
    val_bps.append(bps_agg.closure().mean().item())
    bps_agg.reset()

    print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val BPS: {val_bps[-1]:.4f}")

print("\nTraining completed!")

#%%
# ============================================================================
# 4. PLOTTING RESULTS
# ============================================================================

# Create plots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Training Results', fontsize=16)

# Plot 1: Training loss per step
axes[0, 0].plot(step_numbers, step_losses, alpha=0.7, linewidth=0.5)
axes[0, 0].set_xlabel('Training Step')
axes[0, 0].set_ylabel('Poisson NLL Loss')
axes[0, 0].set_title('Training Loss per Step')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Training loss per epoch (smoothed)
axes[0, 1].plot(range(1, len(train_losses) + 1), train_losses, 'b-o', label='Training Loss')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Average Poisson NLL Loss')
axes[0, 1].set_title('Average Training Loss per Epoch')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend()

# Plot 3: Validation loss per epoch
axes[1, 0].plot(range(1, len(val_losses) + 1), val_losses, 'r-o', label='Validation Loss')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Average Poisson NLL Loss')
axes[1, 0].set_title('Validation Loss per Epoch')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend()

# Plot 4: Combined training and validation loss
axes[1, 1].plot(range(1, len(train_losses) + 1), train_losses, 'b-o', label='Training Loss')
axes[1, 1].plot(range(1, len(val_losses) + 1), val_losses, 'r-o', label='Validation Loss')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Average Poisson NLL Loss')
axes[1, 1].set_title('Training vs Validation Loss')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend()

plt.tight_layout()
plt.show()

# Print final statistics
print(f"\nFinal Training Loss: {train_losses[-1]:.4f}")
print(f"Final Validation Loss: {val_losses[-1]:.4f}")

# %%










