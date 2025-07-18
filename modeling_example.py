#%%
# ============================================================================
# 1. IMPORT LIBRARIES AND SET UP ENVIRONMENT
# ============================================================================

# Standard scientific computing libraries
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
from utils.modules import SplitRelu, NonparametricReadout
from utils.datasets import DictDataset
from utils.general import set_seeds, ensure_tensor
from utils.rf import calc_sta, plot_stas
from utils.spikes import KilosortResults
from utils.datasets import CombinedEmbeddedDataset
from utils.modules import WindowedConv2d, SplitRelu
from utils.reg import laplacian


#%%
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
data_dir = Path('./data')

#%%
# ============================================================================
# 2. LOAD SHIFTED DATASETS
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
# ============================================================================
# 3. DATA PREPROCESSING AND QUALITY CONTROL
# ============================================================================
# Before training neural network models, we need to:
# 1. Identify valid time points for analysis (avoiding trial boundaries)
# 2. Perform quality control to select high-quality neural units
# 3. Split data into training, validation, and test sets

def get_valid_inds(dset, n_lags):
    """
    Find valid time indices for analysis, excluding trial boundaries and invalid periods.

    When analyzing neural responses to visual stimuli, we need to exclude:
    - Trial boundaries (where stimulus context changes abruptly)
    - Periods marked as invalid in the dataset
    - Time points too close to trial starts (within n_lags frames)

    Parameters
    ----------
    dset : DictDataset
        Dataset containing trial indices and validity markers
    n_lags : int
        Number of stimulus history frames needed for each prediction

    Returns
    -------
    torch.Tensor
        Indices of valid time points for analysis
    """
    # Get validity markers from dataset (e.g., eye tracking quality)
    dpi_valid = dset['dpi_valid']

    # Detect trial boundaries (where trial index changes)
    new_trials = torch.diff(dset['trial_inds'], prepend=torch.tensor([-1])) != 0

    # Start with points that are not at trial boundaries and are marked valid
    valid = ~new_trials
    valid &= (dpi_valid > 0)

    # Exclude points too close to trial boundaries (need n_lags history)
    for iL in range(n_lags):
        valid &= torch.roll(valid, iL)

    # Return indices of valid time points
    inds = torch.where(valid)[0]
    return inds

# ============================================================================
# ANALYSIS PARAMETERS
# ============================================================================

# Number of stimulus history frames to include in model
n_lags = 25  # 25 frames = ~100ms at 240Hz (typical retinal response latency)

# Get dataset dimensions
n_units = gabor_dataset['robs'].shape[1]  # Number of recorded neurons
n_y, n_x = gabor_dataset['stim'].shape[1:3]  # Stimulus spatial dimensions

# Find valid time indices for both datasets
gabor_inds = get_valid_inds(gabor_dataset, n_lags)
ni_inds = get_valid_inds(ni_dataset, n_lags)

print(f"Dataset dimensions:")
print(f"  Stimulus: {n_y} x {n_x} pixels")
print(f"  Units: {n_units} kilosort clusters")
print(f"  Valid timepoints - Gabor: {len(gabor_inds)}, Natural Images: {len(ni_inds)}")

#%%
# ============================================================================
# 3.1 NEURAL UNIT QUALITY CONTROL
# ============================================================================
"""
Just because Kilosort identifies a cluster of waveforms doesn't mean
the cluster corresponds to a single neuron (Type 1 errors), or that 
the waveforms contain all the spikes the neuron fires (Type 2 errors).

We apply several quality control checks to select the units we model.

1. Visual responsiveness:
   - Measured using spike-triggered stimulus energy (STE)
   - Finds clusters that respond to stimulus area in localized area of the visual field

2. Refractory period violations:
   - Real neurons have ~1-2ms refractory periods where they cannot spike
   - Violations suggest contamination from nearby neurons

3. Amplitude distribution:
   - Kilosort sets a hard amplitude threshold on spike detection
   - Many spikes from a given neuron may fall below this threshold and not be detected
   - The amplitude distribution of detected spikes can reveal this by 
     the presence of a truncation in the amplitude distribution
   - The distribution of missing spikes can change over time, so must be tracked
"""

# Quality thresholds for unit selection
SNR_THRESHOLD = 6  # Signal-to-noise ratio threshold for visual responsiveness

# ============================================================================
# 3.1.1 VISUAL RESPONSIVENESS ANALYSIS
# ============================================================================

print("Analyzing neural response timing and visual responsiveness...")
print("Computing spike-triggered stimulus energies...")

# Calculate spike-triggered stimulus energy (STE)
# STE measures how much stimulus energy drives each neuron at different time lags
# This is like a "reverse correlation" - what stimulus patterns preceded each spike?
spike_triggered_energies = calc_sta(
    gabor_dataset['stim'], gabor_dataset['robs'],
    n_lags, inds=gabor_inds, device=device, batch_size=10000,
    stim_modifier=lambda x: x**2,  # Square stimulus to get energy (not contrast)
    progress=True
).cpu().numpy()

print(f"Computed STEs for {spike_triggered_energies.shape[0]} units, "
      f"{spike_triggered_energies.shape[1]} time lags")

# Find the optimal response lag for each neuron
# We look for the lag that gives the strongest, most reliable response
# Subtract median to remove baseline and focus on stimulus-driven modulation
signal_strength = np.abs(spike_triggered_energies -
                        np.median(spike_triggered_energies, axis=(2,3), keepdims=True))

# Apply spatial smoothing to reduce noise while preserving signal structure
smoothing_kernel = [0, 2, 2, 2]  # [time, y, x] - no temporal, 2-pixel spatial smoothing
signal_strength = gaussian_filter(signal_strength, smoothing_kernel)

# Calculate signal-to-noise ratio for each unit
# Use first lag (earliest time) as noise estimate since visual responses are delayed
noise_level = np.median(signal_strength[:, 0], axis=(1,2))
signal_to_noise = np.max(signal_strength, axis=(1,2,3)) / noise_level

# Visualize the distribution of signal-to-noise ratios
plt.figure(figsize=(10, 6))
plt.hist(signal_to_noise, bins=100, alpha=0.7, edgecolor='black')
plt.axvline(SNR_THRESHOLD, color='red', linestyle='--', linewidth=2,
           label=f'Threshold = {SNR_THRESHOLD}')
plt.xlabel('Signal-to-Noise Ratio')
plt.ylabel('Number of Units')
plt.title('Distribution of Visual Responsiveness (Signal-to-Noise Ratios)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Select visually responsive units
visually_responsive_units = np.where(signal_to_noise > SNR_THRESHOLD)[0]
print(f'✅ {len(visually_responsive_units)} / {n_units} units pass visual responsiveness criteria')
print(f'   ({len(visually_responsive_units)/n_units*100:.1f}% of recorded units)')

if len(visually_responsive_units) == 0:
    print("⚠️  WARNING: No units pass visual responsiveness criteria!")
    print("   Consider lowering SNR_THRESHOLD or checking data quality")
#%%
# ============================================================================
# 3.1.2 REFRACTORY PERIOD ANALYSIS
# ============================================================================
"""
Real neurons have a refractory period (~1-2ms) after each spike during which
they cannot fire again. Violations of this biological constraint indicate:
- Contamination from nearby neurons (spike sorting errors)
- Noise artifacts being classified as spikes

We analyze inter-spike interval (ISI) distributions to estimate contamination.
This analysis was originally developed by the Steinmetz Lab and adapted by Ryan Ressmeyer.
For the original code see: 
    https://github.com/SteinmetzLab/slidingRefractory
"""

print("Loading spike sorting results and analyzing refractory periods...")

# Load spike sorting results from Kilosort4
ks4_dir = data_dir / 'Allen_2022-04-13_ks4'
ks_results = KilosortResults(ks4_dir)

# Extract spike data
spike_times = ks_results.spike_times  # When each spike occurred (in seconds)
spike_amplitudes = ks_results.spike_amplitudes  # Amplitude of each spike
spike_clusters = ks_results.spike_clusters  # Which neuron each spike came from
cluster_ids = np.unique(spike_clusters)  # List of all neuron IDs

print(f"Loaded spike data:")
print(f"  {len(cluster_ids)} units")
print(f"  {len(spike_times)} total spikes")
print(f"  Recording duration: {spike_times.max():.1f} seconds")

#%%
# Compute contamination estimates across different refractory period assumptions
from utils.qc import compute_min_contam_props, plot_min_contam_prop

print("Computing contamination estimates...")

# Test range of refractory periods (1-10ms)
refractory_periods = np.exp(np.linspace(np.log(1e-3), np.log(10e-3), 100))

# Compute minimum contamination proportion for each unit
# This estimates what fraction of spikes are likely contamination
min_contam_props, firing_rates = compute_min_contam_props(
    spike_times, spike_clusters,
    refractory_periods=refractory_periods,
    ref_acg_t_start=.25e-3,  # Start analysis at 0.25ms (absolute refractory period)
    progress=True
)

# Take the minimum contamination across all tested refractory periods
contam_props = min_contam_props.min(axis=1)

# Visualize contamination distribution
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(contam_props, bins=50, alpha=0.7, edgecolor='black')
plt.xlabel('Contamination Proportion')
plt.ylabel('Number of Units')
plt.title('Distribution of Estimated Contamination')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(firing_rates, contam_props, alpha=0.6)
plt.xlabel('Firing Rate (Hz)')
plt.ylabel('Contamination Proportion')
plt.title('Contamination vs Firing Rate')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Show examples of units with different contamination levels
print("\nExample units with different contamination levels:")
for prop in [0, 20, 50, 80, 100]:
    closest_ind = np.argmin(np.abs(contam_props - prop/100))
    unit_id = cluster_ids[closest_ind]

    print(f'{prop}% contamination example: Unit {unit_id}')

    # Plot detailed analysis for this unit
    fig, axs = plot_min_contam_prop(
        spike_times[spike_clusters == unit_id],
        min_contam_props[closest_ind],
        refractory_periods
    )

    # Add informative title
    n_spikes = np.sum(spike_clusters == unit_id)
    axs.set_title(f'Unit {unit_id} - {n_spikes} spikes '
                 f'({firing_rates[closest_ind]:.2f} Hz) - '
                 f'{contam_props[closest_ind]*100:.1f}% contamination')
    plt.show()

#%%
# ============================================================================
# 3.1.3 APPLY CONTAMINATION THRESHOLD
# ============================================================================

# Set contamination threshold
CONTAMINATION_THRESHOLD = 0.75  # Allow up to 75% contamination

print(f"Applying contamination threshold of {CONTAMINATION_THRESHOLD*100}%...")

# Find units that pass both visual responsiveness AND contamination criteria
contamination_pass = np.where(contam_props <= CONTAMINATION_THRESHOLD)[0]
included_units = np.intersect1d(contamination_pass, visually_responsive_units)

print(f"Quality control summary:")
print(f"  Visual responsiveness: {len(visually_responsive_units)}/{len(cluster_ids)} units")
print(f"  Contamination < {CONTAMINATION_THRESHOLD*100}%: {len(contamination_pass)}/{len(cluster_ids)} units")
print(f"  ✅ Both criteria: {len(included_units)}/{len(cluster_ids)} units")
print(f"  Final inclusion rate: {len(included_units)/len(cluster_ids)*100:.1f}%")

# Update datasets to include only high-quality units
gabor_dataset['robs'] = gabor_dataset['robs'][:, included_units]
ni_dataset['robs'] = ni_dataset['robs'][:, included_units]

#%%
# ============================================================================
# 3.1.4 VISUALIZE SPIKE CHARACTERISTICS
# ============================================================================

# Show example spike characteristics for a representative unit
if len(included_units) > 5:
    example_unit_idx = 5
    example_unit = cluster_ids[included_units[example_unit_idx]]
else:
    example_unit_idx = 0
    example_unit = cluster_ids[included_units[example_unit_idx]]

print(f"Examining spike characteristics for Unit {example_unit}...")

fig, axs = plt.subplots(1, 2, figsize=(12, 5), width_ratios=[2, 1])

# Plot 1: Spike amplitudes over time
plt.subplot(121)
unit_mask = spike_clusters == example_unit
unit_times = spike_times[unit_mask]
unit_amps = spike_amplitudes[unit_mask]

plt.hist2d(unit_times, unit_amps, bins=(200, 50), cmap='Purples')
plt.xlabel('Time (s)')
plt.ylabel('Spike Amplitude (a.u.)')
plt.title(f'Unit {example_unit}: Spike Amplitudes Over Time')
plt.colorbar(label='Spike Count')

# Plot 2: Inter-spike interval distribution
plt.subplot(122)
isis = np.diff(unit_times) * 1000  # Convert to milliseconds
plt.hist(isis, bins=np.linspace(0, 10, 50), alpha=0.7, edgecolor='black')
plt.axvline(2, color='red', linestyle='--', label='2ms refractory period')
plt.xlabel('Inter-Spike Interval (ms)')
plt.ylabel('Count')
plt.title(f'ISI Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Unit {example_unit} statistics:")
print(f"  Total spikes: {len(unit_times)}")
print(f"  Firing rate: {len(unit_times)/(unit_times.max()-unit_times.min()):.2f} Hz")
print(f"  Refractory violations (<2ms): {np.sum(isis < 2)} ({np.sum(isis < 2)/len(isis)*100:.2f}%)")

#%%
# ============================================================================
# 3.1.5 AMPLITUDE STABILITY ANALYSIS
# ============================================================================
"""
Kilosort uses a hard amplitude threshold to detect spikes.
This can lead to missing spikes from weak neural responses.

We track the "missing percentage" (MPCT) over time to identify periods
of poor recording quality that should be excluded from analysis.
"""

print("Analyzing spike amplitude stability over time...")

from utils.qc import analyze_amplitude_truncation, plot_amplitude_truncation

# Analyze amplitude stability for all units
amplitude_results = []

print("Computing amplitude truncation analysis for all units...")
for iU in tqdm(cluster_ids, desc="Analyzing units"):
    # Get spikes for this unit
    st_clu = spike_times[spike_clusters == iU]
    amp_clu = spike_amplitudes[spike_clusters == iU]

    # Analyze amplitude changes over time
    window_blocks, valid_blocks, popts, mpcts = analyze_amplitude_truncation(
        st_clu, amp_clu, max_isi=np.inf
    )

    amplitude_results.append({
        'cid': iU,
        'window_blocks': window_blocks,
        'valid_blocks': valid_blocks,
        'popts': popts,
        'mpcts': mpcts,  # Missing percentage over time
    })

print(f"Completed amplitude analysis for {len(amplitude_results)} units")

#%%
# ============================================================================
# 3.1.6 VISUALIZE AMPLITUDE STABILITY EXAMPLE
# ============================================================================

# Show detailed amplitude analysis for the same example unit
example_unit_global_idx = np.where(cluster_ids == example_unit)[0][0]

print(f"Detailed amplitude analysis for Unit {example_unit}:")

fig, axs = plot_amplitude_truncation(
    spike_times[spike_clusters == example_unit],
    spike_amplitudes[spike_clusters == example_unit],
    amplitude_results[example_unit_global_idx]['window_blocks'],
    amplitude_results[example_unit_global_idx]['valid_blocks'],
    amplitude_results[example_unit_global_idx]['mpcts'],
)

axs.set_title(f'Unit {example_unit}: Amplitude Stability Analysis\n'
             f'Red line shows estimated "missing percentage" over time')
plt.show()

# Explain what we're seeing
mpcts = amplitude_results[example_unit_global_idx]['mpcts']
if len(mpcts) > 0:
    print(f"Missing percentage range: {np.min(mpcts):.1f}% - {np.max(mpcts):.1f}%")
    print(f"Average missing percentage: {np.mean(mpcts):.1f}%")
else:
    print("No amplitude windows detected for this unit")


#%%
# ============================================================================
# 3.1.7 APPLY AMPLITUDE-BASED QUALITY CONTROL
# ============================================================================
"""
Here we use the amplitude analysis to create a time-varying quality mask
for each unit. We refer to this mask as the "data filters" (dfs) for each unit.
This allows us to exclude periods of poor recording quality from the loss
calculation when training the model. 
"""

# Set threshold for missing percentage
MPCT_THRESHOLD = 30  # Exclude periods with >30% estimated missing spikes

print(f"Creating time-varying quality masks (MPCT threshold: {MPCT_THRESHOLD}%)...")

# Initialize quality masks (degrees of freedom) for both datasets
gabor_dfs = torch.zeros_like(gabor_dataset['robs'])
ni_dfs = torch.zeros_like(ni_dataset['robs'])

# Process each included unit
for iU, uid in enumerate(included_units):
    # Get spike times for this unit
    st_clu = spike_times[spike_clusters == uid]

    # Get amplitude analysis results
    uid_idx = np.where(cluster_ids == uid)[0][0]  # Find index in amplitude_results
    window_blocks = np.array(amplitude_results[uid_idx]['window_blocks']).flatten()

    if len(window_blocks) == 0:
        # No amplitude windows detected - include all timepoints
        gabor_dfs[:, iU] = 1
        ni_dfs[:, iU] = 1
        continue

    # Get times corresponding to amplitude windows
    window_times = st_clu[window_blocks]
    mpcts = amplitude_results[uid_idx]['mpcts']

    # Interpolate missing percentages to dataset time bins
    # Use linear interpolation with constant extrapolation
    mpct_interpolant = interp1d(
        window_times, np.repeat(mpcts, 2),
        kind='linear', fill_value=50, bounds_error=False
    )

    # Apply to both datasets
    mpct_bins_gabor = mpct_interpolant(gabor_dataset['t_bins'])
    gabor_dfs[:, iU] = torch.from_numpy(mpct_bins_gabor < MPCT_THRESHOLD).float()

    mpct_bins_ni = mpct_interpolant(ni_dataset['t_bins'])
    ni_dfs[:, iU] = torch.from_numpy(mpct_bins_ni < MPCT_THRESHOLD).float()

# Add quality masks to datasets
gabor_dataset['dfs'] = gabor_dfs
ni_dataset['dfs'] = ni_dfs

print(f'Time bins passing MPCT threshold:')
print(f'  Gabor dataset: {gabor_dfs.mean().item():.1%}')
print(f'  Natural images dataset: {ni_dfs.mean().item():.1%}')

#%%
# ============================================================================
# 3.1.8 FINAL UNIT SELECTION
# ============================================================================
"""
Apply final criterion: units must have sufficient spikes in both datasets
after all quality control measures.
"""

MIN_SPIKE_COUNT = 500  # Minimum number of high-quality spikes required per unit

print(f"Applying final spike count criterion (minimum {MIN_SPIKE_COUNT} spikes)...")

# Count high-quality spikes for each unit in each dataset
spikes_after_qc_gabor = (gabor_dataset['robs'] * gabor_dataset['dfs']).sum(0)
spikes_after_qc_ni = (ni_dataset['robs'] * ni_dataset['dfs']).sum(0)

# Find units with sufficient spikes in BOTH datasets
sufficient_spikes = (spikes_after_qc_gabor > MIN_SPIKE_COUNT) & (spikes_after_qc_ni > MIN_SPIKE_COUNT)
final_units = np.where(sufficient_spikes)[0]

print(f"Final unit selection:")
print(f"  Units after initial QC: {len(included_units)}")
print(f"  Units with >{MIN_SPIKE_COUNT} spikes in Gabor: {(spikes_after_qc_gabor > MIN_SPIKE_COUNT).sum()}")
print(f"  Units with >{MIN_SPIKE_COUNT} spikes in NI: {(spikes_after_qc_ni > MIN_SPIKE_COUNT).sum()}")
print(f"  ✅ Final units for modeling: {len(final_units)}")

# Update datasets to include only final units
gabor_dataset['robs'] = gabor_dataset['robs'][:, final_units]
gabor_dataset['dfs'] = gabor_dataset['dfs'][:, final_units]
ni_dataset['robs'] = ni_dataset['robs'][:, final_units]
ni_dataset['dfs'] = ni_dataset['dfs'][:, final_units]

# Print summary statistics
print(f"\nFinal dataset summary:")
print(f"  Total high-quality spikes in Gabor: {(gabor_dataset['robs'] * gabor_dataset['dfs']).sum().item():.0f}")
print(f"  Total high-quality spikes in NI: {(ni_dataset['robs'] * ni_dataset['dfs']).sum().item():.0f}")
print(f"  Average firing rate per unit: {(gabor_dataset['robs'] * gabor_dataset['dfs']).sum(0).mean().item():.1f} spikes")

#%%
# ============================================================================
# 3.2 VISUALIZE RECEPTIVE FIELDS
# ============================================================================
"""
Now that we have high-quality units, let's visualize their receptive fields
using spike-triggered averages (STAs). This helps us understand what visual
features each neuron responds to.
"""

print("Computing spike-triggered averages (STAs) for final units...")

# Compute STAs using contrast (mean-subtracted stimulus)
stas = calc_sta(
    gabor_dataset['stim'] - gabor_dataset['stim'].mean(),
    gabor_dataset['robs'],
    n_lags, inds=gabor_inds, device=device, batch_size=10000,
    progress=True
).cpu().numpy()

print(f"Computed STAs for {stas.shape[0]} units")

# Visualize receptive fields
print("Displaying receptive field maps...")
fig, axs = plot_stas(stas[:, :, None, :, :])
axs.set_title('Spike-Triggered Average (STA)')
plt.show()

#%%
# Also compute spike-triggered energies for comparison
print("Computing spike-triggered energies (STEs)...")

stes = calc_sta(
    gabor_dataset['stim'], gabor_dataset['robs'],
    n_lags, device=device, batch_size=10000,
    stim_modifier=lambda x: x**2,  # Square for energy
    progress=True
).cpu().numpy()

# Remove mean to highlight stimulus-driven modulation
stes -= stes.mean(axis=(1, 2, 3), keepdims=True)

print("Displaying energy-based receptive field maps...")
fig, axs = plot_stas(stes[:, :, None, :, :])
axs.set_title('Spike-Triggered Energy (STE)')
plt.show()

#%%
# ============================================================================
# 4. DATA SPLITTING AND DATASET PREPARATION
# ============================================================================
"""
For robust model evaluation, we need to split our data into:
- Training set: Used to optimize model parameters
- Validation set: Used for hyperparameter tuning and early stopping
- Test set: Used for final, unbiased performance evaluation

Important: We split by TRIALS, not individual timepoints, to avoid
data leakage between sets.
"""


# Define which data keys need which temporal lags
keys_lags = {
    'robs': 0,        # Neural responses (current timepoint)
    'dfs': 0,         # Quality flags (current timepoint)
    'stim': np.arange(n_lags),  # Stimulus history (past n_lags timepoints)
}

print("Setting up data splitting strategy...")
print(f"Using {n_lags} stimulus history frames for each prediction")


#%%
# ============================================================================
# 4.1 TRIAL-BASED DATA SPLITTING FUNCTION
# ============================================================================

def split_inds_by_trial(dset, inds, splits, seed=1002):
    """
    Split data indices by trial to avoid data leakage between train/val/test sets.

    This is crucial for neural data analysis because consecutive timepoints
    within a trial are highly correlated. Random splitting would allow the
    model to "cheat" by learning from nearby timepoints.

    Parameters
    ----------
    dset : DictDataset
        Dataset containing trial indices
    inds : torch.Tensor
        Valid time indices to split
    splits : list of float
        Fraction of trials for each split (must sum to ≤1)
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    list of torch.Tensor
        Time indices for each split
    """
    assert np.sum(splits) <= 1, 'Split fractions must sum to 1 or less'

    # Set random seed for reproducible splits
    set_seeds(seed)

    # Get unique trial IDs and randomize their order
    trials = dset['trial_inds'].unique()
    rand_trials = torch.randperm(len(trials))

    # Calculate number of trials for each split
    split_sizes = [int(len(trials) * split) for split in splits]

    # Assign trials to each split
    split_trials = []
    start_idx = 0
    for size in split_sizes:
        end_idx = start_idx + size
        split_trials.append(trials[rand_trials[start_idx:end_idx]])
        start_idx = end_idx

    # Convert trial assignments to time index assignments
    split_inds = []
    for trial_subset in split_trials:
        # Find time indices belonging to trials in this split
        mask = torch.isin(dset['trial_inds'][inds], trial_subset)
        split_inds.append(inds[mask])

    return split_inds

#%%
# ============================================================================
# 4.2 CREATE TRAIN/VALIDATION/TEST SPLITS
# ============================================================================

# Define split proportions
train_val_test_split = [0.6, 0.2, 0.2]  # 60% train, 20% val, 20% test

print("Creating trial-based data splits...")

# Split Gabor dataset
gabor_train_inds, gabor_val_inds, gabor_test_inds = split_inds_by_trial(
    gabor_dataset, gabor_inds, train_val_test_split
)

print(f'Gabor dataset splits:')
print(f'  Train: {len(gabor_train_inds):,} samples ({len(gabor_train_inds)/len(gabor_inds):.1%})')
print(f'  Val:   {len(gabor_val_inds):,} samples ({len(gabor_val_inds)/len(gabor_inds):.1%})')
print(f'  Test:  {len(gabor_test_inds):,} samples ({len(gabor_test_inds)/len(gabor_inds):.1%})')

# Split Natural Images dataset
ni_train_inds, ni_val_inds, ni_test_inds = split_inds_by_trial(
    ni_dataset, ni_inds, train_val_test_split
)

print(f'\nNatural Images dataset splits:')
print(f'  Train: {len(ni_train_inds):,} samples ({len(ni_train_inds)/len(ni_inds):.1%})')
print(f'  Val:   {len(ni_val_inds):,} samples ({len(ni_val_inds)/len(ni_inds):.1%})')
print(f'  Test:  {len(ni_test_inds):,} samples ({len(ni_test_inds)/len(ni_inds):.1%})')

#%%
# ============================================================================
# 4.3 CREATE PYTORCH DATASETS
# ============================================================================

print("\nCreating PyTorch datasets with temporal embedding...")

# Individual dataset splits
gabor_train_dataset = CombinedEmbeddedDataset(gabor_dataset, gabor_train_inds, keys_lags, 'cpu')
gabor_val_dataset = CombinedEmbeddedDataset(gabor_dataset, gabor_val_inds, keys_lags, 'cpu')
gabor_test_dataset = CombinedEmbeddedDataset(gabor_dataset, gabor_test_inds, keys_lags, 'cpu')

ni_train_dataset = CombinedEmbeddedDataset(ni_dataset, ni_train_inds, keys_lags, 'cpu')
ni_val_dataset = CombinedEmbeddedDataset(ni_dataset, ni_val_inds, keys_lags, 'cpu')
ni_test_dataset = CombinedEmbeddedDataset(ni_dataset, ni_test_inds, keys_lags, 'cpu')

# Combined datasets (for training on both stimulus types)
both_train_dataset = CombinedEmbeddedDataset(
    [gabor_dataset, ni_dataset], [gabor_train_inds, ni_train_inds], keys_lags, 'cpu'
)
both_val_dataset = CombinedEmbeddedDataset(
    [gabor_dataset, ni_dataset], [gabor_val_inds, ni_val_inds], keys_lags, 'cpu'
)
both_test_dataset = CombinedEmbeddedDataset(
    [gabor_dataset, ni_dataset], [gabor_test_inds, ni_test_inds], keys_lags, 'cpu'
)

print("Dataset creation completed!")

#%%
# ============================================================================
# 4.4 TEST DATASET STRUCTURE
# ============================================================================

# Examine the structure of our processed datasets
print("Examining dataset structure...")
test_batch = both_train_dataset[:64]

print("Batch contents:")
for k, v in test_batch.items():
    print(f"  {k}: {v.shape} ({v.dtype})")

print(f"\nStimulus tensor interpretation:")
print(f"  Shape: [batch_size, n_lags, height, width]")
print(f"  Contains {test_batch['stim'].shape[1]} frames of stimulus history")
print(f"  Spatial resolution: {test_batch['stim'].shape[2]} x {test_batch['stim'].shape[3]} pixels")

# %%
# ============================================================================
# 5. NEURAL NETWORK ARCHITECTURE
# ============================================================================
"""
This section defines the neural network architecture for modeling neural responses.
We use a lightweight spatiotemporal CNN with residual connections to capture both spatial
and temporal dependencies in the stimulus-response relationship.
"""

class SpatioTemporalResNet(nn.Module):
    """
    Spatiotemporal CNN with residual connections for modeling neural responses.

    Architecture:
    1. Temporal layer: Processes stimulus history with 1x1 convolutions
    2. Spatial layers: Process spatial features with residual connections
    3. Readout layer: Maps features to individual neuron predictions

    The residual connections help with gradient flow during training and
    allow the model to learn both simple and complex feature combinations.
    """

    def __init__(self, n_lags, n_y, n_x, n_units, temporal_channels, res_channels,
                 kernel_size, n_layers, baseline_rates=None):
        """
        Parameters
        ----------
        n_lags : int
            Number of stimulus history frames
        n_y, n_x : int
            Spatial dimensions of stimulus
        n_units : int
            Number of neurons to model
        temporal_channels : int
            Number of channels after temporal processing
        res_channels : int
            Number of channels in residual layers
        kernel_size : int
            Spatial kernel size for convolutional layers
        n_layers : int
            Number of spatial processing layers
        baseline_rates : array-like, optional
            Baseline firing rates for initializing readout bias
        """
        super(SpatioTemporalResNet, self).__init__()

        # Temporal processing: combine stimulus history
        self.temporal_layer = nn.Conv2d(n_lags, temporal_channels, kernel_size=1, bias=False)
        self.temporal_activation = SplitRelu()  # Separate positive/negative channels

        # Spatial processing layers with residual connections
        self.layers = nn.ModuleList()
        self.kernel_size = kernel_size
        self.n_layers = n_layers

        for iC in range(n_layers):
            in_channels = temporal_channels*2 if iC == 0 else res_channels
            self.layers.append(WindowedConv2d(in_channels, res_channels,
                                            kernel_size=kernel_size, bias=True))

        # Channel projection for residual connections (if needed)
        if temporal_channels*2 != res_channels:
            self.channel_projection = nn.Conv2d(temporal_channels*2, res_channels,
                                              kernel_size=1, bias=False)
        else:
            self.channel_projection = None

        # Calculate output spatial dimensions after convolutions
        contraction = (kernel_size - 1) * n_layers
        output_dims = [res_channels, n_y - contraction, n_x - contraction]

        # Readout layer: map spatial features to neuron responses
        self.readout = NonparametricReadout(output_dims, n_units, bias=True)

        # Initialize readout bias with baseline firing rates
        if baseline_rates is not None:
            assert len(baseline_rates) == n_units, 'baseline_rates must match n_units'
            # Convert rates to log-space for softplus activation
            inv_softplus = lambda x: torch.log(torch.exp(x) - 1)
            self.readout.bias.data = inv_softplus(
                ensure_tensor(baseline_rates, device=self.readout.bias.device)
            )

    def forward(self, batch, debug=False):
        """
        Forward pass through the network.

        Parameters
        ----------
        batch : dict
            Batch containing 'stim' key with stimulus tensor
        debug : bool, optional
            If True, print intermediate tensor shapes

        Returns
        -------
        dict
            Input batch with added 'rhat' key containing predictions
        """
        x = batch['stim']  # Shape: [batch, n_lags, height, width]
        if debug:
            print(f'Input stimulus: {x.shape}')

        # Temporal processing: combine stimulus history
        x = self.temporal_layer(x)  # Shape: [batch, temporal_channels, height, width]
        x = self.temporal_activation(x)  # Shape: [batch, temporal_channels*2, height, width]
        if debug:
            print(f'After temporal processing: {x.shape}')

        # Store input for residual connections
        residual = x
        if self.channel_projection is not None:
            residual = self.channel_projection(residual)

        # Spatial processing with residual connections
        for i, layer in enumerate(self.layers):
            x = layer(x)  # Spatial convolution
            x = F.relu(x)  # Nonlinearity

            # Add residual connection (crop residual to match current x size)
            if residual.shape[-2:] != x.shape[-2:]:
                # Calculate spatial cropping needed due to convolution
                h_diff = residual.shape[-2] - x.shape[-2]
                w_diff = residual.shape[-1] - x.shape[-1]
                h_crop = h_diff // 2
                w_crop = w_diff // 2
                residual = residual[..., h_crop:h_crop+x.shape[-2], w_crop:w_crop+x.shape[-1]]

            x = x + residual  # Residual connection
            residual = x  # Update residual for next layer

            if debug:
                print(f'After spatial layer {i+1}: {x.shape}')

        # Readout: map spatial features to neuron predictions
        x = self.readout(x)  # Shape: [batch, n_units]
        if debug:
            print(f'After readout: {x.shape}')

        # Apply softplus to ensure positive firing rates
        batch['rhat'] = F.softplus(x)
        return batch

    def temporal_smoothness_regularization(self):
        """
        Compute temporal smoothness regularization term.

        Returns
        -------
        torch.Tensor
            Regularization loss term
        """
        return laplacian(self.temporal_layer.weight, dims=1)

    def plot_weights(self, name=None):
        """Visualize learned model weights."""
        prepend = (name + ' - ') if name is not None else ''
        # Plot temporal filters
        temporal_weights = self.temporal_layer.weight.detach().cpu().numpy()
        plt.figure(figsize=(10, 6))
        plt.plot(temporal_weights.squeeze().T)
        plt.xlabel('Time Lag')
        plt.ylabel('Weight')
        plt.title(prepend + 'Learned Temporal Filters')
        plt.grid(True, alpha=0.3)
        plt.show()

        # Plot spatial filters for each layer
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'plot_weights'):
                fig, _ = layer.plot_weights()
                fig.suptitle(prepend + f'Spatial Layer {i+1} Filters')
                plt.show()

        # Plot readout weights
        fig, _ = self.readout.plot_weights()
        fig.suptitle(prepend + 'Readout Weights (Spatial to Neural Mapping)')
        plt.show()

#%%
# ============================================================================
# 5.1 MODEL INSTANTIATION AND TESTING
# ============================================================================

# Get dataset dimensions for model architecture
_, n_y, n_x = gabor_dataset['stim'].shape
n_units = gabor_dataset['robs'].shape[1]

# Model hyperparameters
temporal_channels = 4   # Number of temporal feature channels
res_channels = 8       # Number of channels in residual layers
kernel_size = 13       # Spatial kernel size (13x13 pixels)
n_layers = 4          # Number of spatial processing layers

print(f"Model architecture parameters:")
print(f"  Input dimensions: {n_lags} lags × {n_y} × {n_x} pixels")
print(f"  Output units: {n_units} neurons")
print(f"  Temporal channels: {temporal_channels}")
print(f"  Spatial channels: {res_channels}")
print(f"  Kernel size: {kernel_size}×{kernel_size}")
print(f"  Spatial layers: {n_layers}")

# Calculate baseline firing rates for initialization
baseline_rates = (gabor_dataset['robs'][gabor_train_inds] * gabor_dataset['dfs'][gabor_train_inds]).sum(0) / gabor_dataset['dfs'][gabor_train_inds].sum(0)
print(f"Baseline firing rates: {baseline_rates.mean():.3f} ± {baseline_rates.std():.3f} spikes/bin")

# Create and test model
print("\nCreating ResNet model...")
model = SpatioTemporalResNet(n_lags, n_y, n_x, n_units, temporal_channels, res_channels, kernel_size, n_layers, baseline_rates)

print("Testing forward pass...")
batch = model(test_batch, debug=True)
print(f"✅ Model forward pass successful!")
print(f"   Prediction shape: {batch['rhat'].shape}")
print(f"   Prediction range: {batch['rhat'].min():.3f} - {batch['rhat'].max():.3f}")

print("\nVisualizing initial model weights...")
model.plot_weights()

# %%
# ============================================================================
# 6. MODEL TRAINING FRAMEWORK
# ============================================================================
"""
This section implements the training pipeline for neural network models.
We use Poisson loss (appropriate for spike count data) with data filtering
and regularization to prevent overfitting.

Key components:
1. Masked Poisson loss: Handles variable data quality
2. Early stopping: Prevents overfitting
3. Regularization: Encourages smooth temporal filters, which may be more biologically plausible
"""

def masked_poisson_nll_loss(output, target, dfs=None):
    """
    Compute masked Poisson negative log-likelihood loss.

    The Poisson distribution is appropriate for modeling neural spike counts
    because spikes are discrete events with a natural rate parameter.

    The masking allows us to exclude low-quality time periods from training,
    which is crucial when working with real neural data.

    Parameters
    ----------
    output : torch.Tensor
        Model predictions (firing rates), shape [batch, n_units]
    target : torch.Tensor
        Target spike counts, shape [batch, n_units]
    dfs : torch.Tensor, optional
        Quality mask (degrees of freedom), shape [batch, n_units] or [batch, 1]
        Values should be 0 (exclude) or 1 (include)

    Returns
    -------
    torch.Tensor
        Computed loss (scalar)
    """
    # Compute Poisson NLL for each sample and unit
    loss = F.poisson_nll_loss(output, target, log_input=False, full=False, reduction='none')

    if dfs is not None:
        # Expand mask if needed
        if dfs.shape[1] == 1:
            dfs = dfs.expand(-1, loss.shape[1])

        # Apply quality mask and normalize by valid samples
        loss = loss * dfs
        loss = loss.sum() / dfs.sum()
    else:
        # Simple average if no masking
        loss = loss.mean()

    return loss


#%%
# ============================================================================
# 6.1 TRAINING FUNCTION
# ============================================================================

def train_model(model, train_dataset, val_dataset,
                n_epochs=10, lr=3e-3, weight_decay=1e-4,
                smoothness_lambda=1e-4, batch_size=256, patience=2,
                device='cuda', num_workers=None, plot_weights=True,
                verbose=True):
    """
    Train a spatiotemporal CNN model on neural data with comprehensive monitoring.

    This function implements best practices for neural network training:
    - Early stopping to prevent overfitting
    - Learning rate scheduling (via AdamW optimizer)
    - Regularization for biologically plausible filters
    - Comprehensive metrics tracking (loss and bits per spike)
    - Quality masking for real neural data

    The "bits per spike" metric is particularly important in computational
    neuroscience - it measures how much information the model captures
    about each spike, with higher values indicating better predictions.

    Parameters
    ----------
    model : nn.Module
        The model to train (SpatioTemporalResNet or SpatioTemporalCNN)
    train_dataset : CombinedEmbeddedDataset
        Training dataset with stimulus and response data
    val_dataset : CombinedEmbeddedDataset
        Validation dataset for monitoring overfitting
    n_epochs : int, optional
        Maximum number of training epochs. Default is 10.
    lr : float, optional
        Learning rate for AdamW optimizer. Default is 3e-3.
    weight_decay : float, optional
        L2 regularization strength. Default is 1e-4.
    smoothness_lambda : float, optional
        Temporal smoothness regularization strength. Default is 1e-4.
    batch_size : int, optional
        Batch size for training. Default is 256.
    patience : int, optional
        Early stopping patience (epochs without improvement). Default is 2.
    device : str, optional
        Device for computation ('cuda' or 'cpu'). Default is 'cuda'.
    num_workers : int, optional
        Number of workers for data loading. Default is os.cpu_count()//2.
    plot_weights : bool, optional
        Whether to visualize model weights each epoch. Default is True.
    verbose : bool, optional
        Whether to print detailed training progress. Default is True.

    Returns
    -------
    dict
        Comprehensive training results containing:
        - 'model': trained model with best validation weights loaded
        - 'train_losses': list of training losses per epoch
        - 'val_losses': list of validation losses per epoch
        - 'train_bps': list of training bits per spike per epoch
        - 'val_bps': list of validation bits per spike per epoch
        - 'step_losses': list of losses per training step
        - 'step_numbers': list of step numbers for plotting
        - 'best_epoch': epoch number with best validation performance
    """
    # ========================================================================
    # TRAINING SETUP
    # ========================================================================

    if num_workers is None:
        num_workers = os.cpu_count() // 2

    # Create data loaders with appropriate settings
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True if device == 'cuda' else False
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True if device == 'cuda' else False
    )

    # Move model to computation device
    model = model.to(device)

    # Setup AdamW optimizer (better than Adam for most cases)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Initialize bits-per-spike aggregator for performance monitoring
    bps_agg = PoissonBPSAggregator()

    # Initialize metric tracking lists
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
        print(f"🚀 Starting training for up to {n_epochs} epochs...")
        print(f"   Training batches per epoch: {len(train_loader)}")
        print(f"   Validation batches per epoch: {len(val_loader)}")
        print(f"   Device: {device}")
        print(f"   Optimizer: AdamW (lr={lr}, weight_decay={weight_decay})")
        print(f"   Early stopping patience: {patience} epochs")

    # ========================================================================
    # MAIN TRAINING LOOP
    # ========================================================================

    for epoch in range(n_epochs):
        if verbose:
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch+1}/{n_epochs}")
            print(f"{'='*60}")

        # Visualize model weights (helpful for debugging)
        if plot_weights and hasattr(model, 'plot_weights'):
            print("📊 Current model weights:")
            model.plot_weights()

        # ====================================================================
        # TRAINING PHASE
        # ====================================================================
        model.train()  # Set model to training mode
        epoch_train_losses = []
        bps_agg.reset()

        train_pbar = tqdm(train_loader, desc=f"🔥 Training", disable=not verbose)
        for batch in train_pbar:
            # Move batch to computation device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            optimizer.zero_grad()
            batch = model(batch)

            # Accumulate predictions for BPS calculation
            bps_agg(batch)

            # Calculate primary loss (Poisson NLL with quality masking)
            loss = masked_poisson_nll_loss(batch['rhat'], batch['robs'], batch['dfs'])

            # Add temporal smoothness regularization (encourages smooth filters)
            if hasattr(model, 'temporal_smoothness_regularization'):
                reg_loss = model.temporal_smoothness_regularization()
                loss += smoothness_lambda * reg_loss

            # Backward pass and optimization step
            loss.backward()
            optimizer.step()

            # Track training progress
            step_loss = loss.item()
            step_losses.append(step_loss)
            step_numbers.append(step)
            epoch_train_losses.append(step_loss)

            # Update progress bar with current loss
            if verbose:
                train_pbar.set_postfix({
                    'loss': f'{step_loss:.4f}',
                    'step': step
                })
            step += 1

        # Calculate epoch-level training metrics
        avg_train_loss = np.mean(epoch_train_losses)
        train_losses.append(avg_train_loss)
        train_bps.append(bps_agg.closure().cpu().numpy())
        bps_agg.reset()

        if verbose:
            print(f"📈 Training Results:")
            print(f"   Average Loss: {avg_train_loss:.4f}")
            print(f"   Average BPS: {train_bps[-1].mean():.4f} ± {train_bps[-1].std():.4f}")

        # ====================================================================
        # VALIDATION PHASE
        # ====================================================================
        model.eval()  # Set model to evaluation mode (disables dropout, etc.)
        val_loss_total = 0
        val_samples = 0

        with torch.no_grad():  # Disable gradient computation for efficiency
            val_pbar = tqdm(val_loader, desc=f"🔍 Validation", disable=not verbose)
            for batch in val_pbar:
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}

                # Forward pass (no gradients needed)
                batch = model(batch)

                # Calculate validation loss
                val_loss = masked_poisson_nll_loss(batch['rhat'], batch['robs'], batch['dfs'])

                # Accumulate weighted loss (important for proper averaging with masking)
                n_samples = batch['dfs'].sum().item()
                val_loss_total += val_loss.item() * n_samples
                val_samples += n_samples

                # Accumulate for BPS calculation
                bps_agg(batch)

                # Update progress bar
                if verbose:
                    val_pbar.set_postfix({'loss': f'{val_loss.item():.4f}'})

        # Calculate epoch-level validation metrics
        avg_val_loss = val_loss_total / val_samples if val_samples > 0 else 0
        val_losses.append(avg_val_loss)

        # Calculate validation BPS
        val_bps.append(bps_agg.closure().cpu().numpy())
        bps_agg.reset()

        if verbose:
            print(f"📊 Validation Results:")
            print(f"   Average Loss: {avg_val_loss:.4f}")
            print(f"   Average BPS: {val_bps[-1].mean():.4f} ± {val_bps[-1].std():.4f}")

        # ====================================================================
        # EARLY STOPPING LOGIC
        # ====================================================================

        # Check if validation performance improved
        if epoch > 0 and val_losses[-1] >= best_val_loss:
            patience_count += 1
            if verbose:
                print(f"⚠️  No improvement: {val_losses[-2]:.4f} → {val_losses[-1]:.4f}")
                print(f"   Patience: {patience_count}/{patience}")

            # Stop training if patience exceeded
            if patience_count >= patience:
                if verbose:
                    print(f"🛑 Early stopping triggered!")
                    print(f"   No improvement for {patience} consecutive epochs")
                break
        else:
            # Validation improved - save best model state
            best_val_loss = val_losses[-1]
            patience_count = 0

            if epoch > 0 and verbose:
                print(f"✅ Validation improved: {val_losses[-2]:.4f} → {val_losses[-1]:.4f}")

            # Save the best model weights
            best_state = copy.deepcopy(model.state_dict())

    # ========================================================================
    # TRAINING COMPLETION AND MODEL RESTORATION
    # ========================================================================

    # Find the epoch with best validation performance
    best_epoch = np.argmin(val_losses)

    if verbose:
        print(f"\n{'='*60}")
        print(f"🎯 TRAINING COMPLETED!")
        print(f"{'='*60}")
        print(f"Total epochs run: {len(val_losses)}")
        print(f"Best epoch: {best_epoch+1}")
        print(f"Best validation loss: {val_losses[best_epoch]:.4f}")
        print(f"Best validation BPS: {val_bps[best_epoch].mean():.4f} ± {val_bps[best_epoch].std():.4f}")

        if patience_count >= patience:
            print(f"Training stopped early due to no improvement")
        else:
            print(f"Training completed all {n_epochs} epochs")

    # Restore the best model weights
    if best_state is not None:
        model.load_state_dict(best_state)
        if verbose:
            print(f"✅ Loaded best model weights from epoch {best_epoch+1}")
    else:
        if verbose:
            print("⚠️  No best state saved - using final model weights")

    # Return comprehensive training results
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

#%%
# ============================================================================
# 6.2 TRAINING VISUALIZATION FUNCTION
# ============================================================================

def plot_training_summary(training_results):
    """
    Create comprehensive visualization of training progress.

    Shows three key aspects:
    1. Step-by-step training loss (shows convergence behavior)
    2. Epoch-level train/val loss (shows overfitting)
    3. Epoch-level train/val BPS (shows model performance)

    Parameters
    ----------
    training_results : dict
        Results dictionary from train_model function
    """
    # Extract results
    train_losses = training_results['train_losses']
    val_losses = training_results['val_losses']
    train_bps = training_results['train_bps']
    val_bps = training_results['val_bps']
    step_losses = training_results['step_losses']
    step_numbers = training_results['step_numbers']
    best_epoch = training_results['best_epoch']

    # Create comprehensive plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Training Progress Summary', fontsize=16, fontweight='bold')

    # Plot 1: Training loss per step (shows detailed convergence)
    axes[0].plot(step_numbers, step_losses, alpha=0.7, linewidth=0.8, color='steelblue')
    axes[0].set_xlabel('Training Step')
    axes[0].set_ylabel('Poisson NLL Loss')
    axes[0].set_title('Training Loss per Step\n(Shows convergence behavior)')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Training and validation loss per epoch
    epochs = range(1, len(train_losses) + 1)
    axes[1].plot(epochs, train_losses, 'b-o', label='Training Loss', markersize=4)
    axes[1].plot(epochs, val_losses, 'r-o', label='Validation Loss', markersize=4)
    axes[1].axvline(best_epoch + 1, color='green', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch+1})')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Average Poisson NLL Loss')
    axes[1].set_title('Training vs Validation Loss\n(Shows overfitting)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Plot 3: Training and validation bits per spike
    train_bps_mean = np.array([bps.mean() for bps in train_bps])
    val_bps_mean = np.array([bps.mean() for bps in val_bps])

    axes[2].plot(epochs, train_bps_mean, 'b-o', label='Training BPS', markersize=4)
    axes[2].plot(epochs, val_bps_mean, 'r-o', label='Validation BPS', markersize=4)
    axes[2].axvline(best_epoch + 1, color='green', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch+1})')
    axes[2].axhline(0, color='black', linestyle=':', alpha=0.5, label='Chance Level')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Bits per Spike')
    axes[2].set_title('Model Performance (BPS)\n(Higher is better)')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print(f"\n📊 Training Summary:")
    print(f"   Final training BPS: {train_bps_mean[-1]:.4f}")
    print(f"   Final validation BPS: {val_bps_mean[-1]:.4f}")
    print(f"   Best validation BPS: {val_bps_mean[best_epoch]:.4f} (epoch {best_epoch+1})")
    print(f"   Performance gap: {train_bps_mean[-1] - val_bps_mean[-1]:.4f} BPS")




# %%
# ============================================================================
# 7. MODEL TRAINING EXPERIMENTS
# ============================================================================
"""
Now we train three different models to test our main research question:
Do CNN models trained on one visual stimulus generalize to different conditions?

We train:
1. Gabor-only model: Trained only on Gabor stimuli
2. Natural images-only model: Trained only on natural images
3. Combined model: Trained on both stimulus types

This experimental design allows us to test cross-condition generalization.
"""

#%%
# ============================================================================
# 7.1 TRAINING CONFIGURATION
# ============================================================================

# Define training hyperparameters
training_config = {
    'n_epochs': 20,              # Maximum epochs (early stopping may end sooner)
    'lr': 1e-3,                  # Learning rate
    'weight_decay': 1e-4,        # L2 regularization strength
    'smoothness_lambda': 1e-4,   # Temporal smoothness regularization
    'batch_size': 256,           # Batch size 
    'patience': 2,               # Early stopping patience
    'device': device,            # Use GPU if available
    'plot_weights': False,        # Visualize weights during training
    'verbose': True              # Print detailed progress
}

#%%
# ============================================================================
# 7.2 EXPERIMENT 1: GABOR-ONLY MODEL
# ============================================================================

print(f"\nTraining model on Gabor stimuli only")
print(f"Training samples: {len(gabor_train_dataset):,}")
print(f"Validation samples: {len(gabor_val_dataset):,}")

# Create fresh model for Gabor training
gabor_model = SpatioTemporalResNet(n_lags, n_y, n_x, n_units, temporal_channels,
                                  res_channels, kernel_size, n_layers, baseline_rates)

# Train the model
gabor_results = train_model(
    model=gabor_model,
    train_dataset=gabor_train_dataset,
    val_dataset=gabor_val_dataset,
    **training_config
)

# Visualize training progress
print("\nGabor Model Training Summary:")
plot_training_summary(gabor_results)

#%%
# ============================================================================
# 7.3 EXPERIMENT 2: NATURAL IMAGES-ONLY MODEL
# ============================================================================

print(f"\nTraining model on Natural Images only")
print(f"Training samples: {len(ni_train_dataset):,}")
print(f"Validation samples: {len(ni_val_dataset):,}")

# Create fresh model for Natural Images training
ni_model = SpatioTemporalResNet(n_lags, n_y, n_x, n_units, temporal_channels,
                               res_channels, kernel_size, n_layers, baseline_rates)

# Train the model
ni_results = train_model(
    model=ni_model,
    train_dataset=ni_train_dataset,
    val_dataset=ni_val_dataset,
    **training_config
)

# Visualize training progress
print("\nNatural Images Model Training Summary:")
plot_training_summary(ni_results)

#%%
# ============================================================================
# 7.4 EXPERIMENT 3: COMBINED MODEL
# ============================================================================

print(f"\nTraining model on both stimulus types")
print(f"Training samples: {len(both_train_dataset):,}")
print(f"Validation samples: {len(both_val_dataset):,}")

# Create fresh model for combined training
both_model = SpatioTemporalResNet(n_lags, n_y, n_x, n_units, temporal_channels,
                                 res_channels, kernel_size, n_layers, baseline_rates)

# Train the model
both_results = train_model(
    model=both_model,
    train_dataset=both_train_dataset,
    val_dataset=both_val_dataset,
    **training_config
)

# Visualize training progress
print("\nCombined Model Training Summary:")
plot_training_summary(both_results)


#%%
# ============================================================================
# 7.5 COMPARE TRAINED MODELS
# ============================================================================

print("\n🔍 COMPARING TRAINED MODELS")
print("=" * 60)

# Extract trained models for easier reference
gabor_model = gabor_results['model']
ni_model = ni_results['model']
both_model = both_results['model']

# Compare training performance
print("Training Performance Summary:")
print(f"  Gabor model - Final val BPS: {gabor_results['val_bps'][-1].mean():.4f}")
print(f"  NI model - Final val BPS: {ni_results['val_bps'][-1].mean():.4f}")
print(f"  Combined model - Final val BPS: {both_results['val_bps'][-1].mean():.4f}")

# Visualize learned features
print("\nPlotting learned model weights...")

print("\nGabor Model Weights:")
gabor_model.plot_weights()

print("\nNatural Images Model Weights:")
ni_model.plot_weights()

print("\nCombined Model Weights:")
both_model.plot_weights()

print("\nKey observations to look for:")
print("- How do temporal filters differ between models?")
print("- Are spatial filters qualitatively different for natural images?")

#%%
# ============================================================================
# 8. CROSS-CONDITION GENERALIZATION TESTING
# ============================================================================
"""
Now we will test how well each model generalizes across stimulus conditions. 

We evaluate each model on both test sets to measure:
1. Within-condition performance (how well models perform on trained stimuli)
2. Cross-condition generalization (how well they transfer to out-of-distribution stimuli)
"""

def evaluate_model_on_dataset(model, dataset, batch_size=256, device='cuda', desc="Evaluating"):
    """
    Evaluate a trained model on a test dataset.

    This function computes the bits per spike (BPS) metric, which measures
    how well the model predicts neural responses. Higher BPS indicates
    better model performance.

    Parameters
    ----------
    model : nn.Module
        The trained model to evaluate
    dataset : CombinedEmbeddedDataset
        The test dataset to evaluate on
    batch_size : int, optional
        Batch size for evaluation. Default is 256.
    device : str, optional
        Device to run evaluation on. Default is 'cuda'.
    desc : str, optional
        Description for progress bar. Default is "Evaluating".

    Returns
    -------
    numpy.ndarray
        Bits per spike for each unit (shape: [n_units])
    """
    model.eval()  # Set to evaluation mode
    bps_aggregator = PoissonBPSAggregator()

    # Create data loader (no shuffling needed for evaluation)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True if device == 'cuda' else False
    )

    with torch.no_grad():  # Disable gradients for efficiency
        for batch in tqdm(loader, desc=desc):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            batch = model(batch)

            # Accumulate predictions for BPS calculation
            bps_aggregator(batch)

    # Calculate final BPS per unit
    bps = bps_aggregator.closure().cpu().numpy()
    bps_aggregator.reset()

    return bps

#%%
# ============================================================================
# 8.1 COMPREHENSIVE MODEL EVALUATION
# ============================================================================

print("\nEvaluating all models on both test sets...")

# Dictionary to store all evaluation results
evaluation_results = {}

# Define models and test datasets
model_names = ['Gabor Model', 'NI Model', 'Both Model']
models = [gabor_model, ni_model, both_model]
test_datasets = {
    'Gabor Test': gabor_test_dataset,
    'NI Test': ni_test_dataset
}

print(f"\nTest set sizes:")
for name, dataset in test_datasets.items():
    print(f"  {name}: {len(dataset):,} samples")

# Evaluate each model on each test set
for model_idx, (model_name, model) in enumerate(zip(model_names, models)):
    evaluation_results[model_name] = {}
    print(f"\nEvaluating {model_name}...")

    for test_name, test_dataset in test_datasets.items():
        print(f"   Testing on {test_name}...")

        # Evaluate model performance
        bps = evaluate_model_on_dataset(
            model, test_dataset,
            batch_size=256, device=device,
            desc=f"{model_name} → {test_name}"
        )

        # Store results
        evaluation_results[model_name][test_name] = bps

        # Print summary statistics
        print(f"     Mean BPS: {bps.mean():.4f} ± {bps.std():.4f}")
        print(f"     Median BPS: {np.median(bps):.4f}")
        print(f"     Units with BPS > 0: {np.sum(bps > 0)}/{len(bps)} ({np.sum(bps > 0)/len(bps)*100:.1f}%)")

print("Cross-condition evaluation completed!")

#%%
# ============================================================================
# 8.2 ANALYZE GENERALIZATION PATTERNS
# ============================================================================

for model_name in model_names:
    gabor_bps = evaluation_results[model_name]['Gabor Test'].mean()
    ni_bps = evaluation_results[model_name]['NI Test'].mean()

    print(f"\n{model_name}:")
    print(f"  Gabor Test BPS: {gabor_bps:.4f}")
    print(f"  NI Test BPS: {ni_bps:.4f}")

    if 'Gabor' in model_name:
        print(f"  Cross-condition drop: {gabor_bps - ni_bps:.4f} BPS")
    elif 'NI' in model_name:
        print(f"  Cross-condition drop: {ni_bps - gabor_bps:.4f} BPS")

#%%
# ============================================================================
# 8.3 VISUALIZATION OF RESULTS
# ============================================================================

def plot_bps_distributions(evaluation_results, min_bps=-10, save_path=None):
    """
    Visualize of model performance across conditions.

    Parameters
    ----------
    evaluation_results : dict
        Dictionary containing BPS results for each model and test set
    min_bps : float, optional
        Minimum BPS value to display (clips very negative outliers)
    save_path : str, optional
        Path to save the plot. If None, plot is displayed.
    """
    # Set up the figure with publication-quality styling
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

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

# Create the key results visualization
print("\n📈 Creating results visualization...")
plot_bps_distributions(evaluation_results)

# %%
