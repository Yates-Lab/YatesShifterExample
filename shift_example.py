#%%
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.loss import PoissonBPSAggregator
from utils.modules import SplitRelu, NonparametricReadout
from utils.datasets import DictDataset, generate_gaborium_dataset, generate_gratings_dataset
from utils.general import set_seeds, ensure_tensor
from utils.grid_sample import grid_sample_coords
from utils.rf import calc_sta, plot_stas
from scipy.ndimage import gaussian_filter
from utils.modules import StackedConv2d 
from utils.exp.general import get_clock_functions, get_trial_protocols 
from utils.spikes import KilosortResults
from matplotlib.patches import Circle
from scipy.interpolate import interp1d
from utils.exp.dots import dots_rf_map_session
from pathlib import Path
from mat73 import loadmat
import pandas as pd
# potential speedup for matmul
torch.set_float32_matmul_precision('medium')

set_seeds(1002) # for reproducibility

device = 'cuda'
data_dir = Path('/home/ryanress/YatesShifterExample/data')

exp_file = data_dir / 'Allen_2022-04-13_struct.mat'
exp = loadmat(exp_file)
ptb2ephys, vpx2ephys = get_clock_functions(exp)

ks4_dir = data_dir / 'Allen_2022-04-13_ks4'
ks_results = KilosortResults(ks4_dir)
st = ks_results.spike_times
clu = ks_results.spike_clusters
cids = np.unique(clu)

ddpi_file = data_dir / 'Allen_2022-04-13_ddpi.csv'
dpi = pd.read_csv(ddpi_file)
t_ephys = dpi['t_ephys'].values
dpi_pix = dpi[['dpi_i', 'dpi_j']].values
dpi_valid = dpi['valid'].values
dpi_interp = interp1d(t_ephys, dpi_pix, kind='linear', fill_value='extrapolate', axis=0)
valid_interp = interp1d(t_ephys, dpi_valid, kind='nearest', fill_value='extrapolate')

# Convert DPI signal from pixel locations to degrees (using small angle approximation)
# Metadata for all datasets
screen_resolution = (exp['S']['screenRect'][2:] - exp['S']['screenRect'][:2]).astype(int)
screen_width = exp['S']['screenWidth']
screen_distance = exp['S']['screenDistance']
screen_height = screen_width * screen_resolution[1] / screen_resolution[0]
center_pix = np.flipud((screen_resolution + 1) / 2)

ppd = exp['S']['pixPerDeg'] # pixels per degree
dpi_deg = (dpi_pix - center_pix) / ppd
dpi_deg[:,0] *= -1
dpi_deg = dpi_deg[:,[1,0]]
dpi_deg_interp = interp1d(t_ephys, dpi_deg, kind='linear', fill_value='extrapolate', axis=0)
#%%

# Parameters for RF mapping 
dt = 1/240 # seconds
lags = np.arange(7, 14) # frames
rf_roi_deg = np.array([[-4, 4], [-4, 4]]) # [[az0, az1], [el0, el1]]
dxy_deg = .25 # degrees

rf_dict = dots_rf_map_session(exp, dpi, ks_results,
                              dt=dt, lags=lags, roi_deg=rf_roi_deg, dxy_deg=dxy_deg)
rf = rf_dict['rf']
j_edges = rf_dict['j_edges']
i_edges = rf_dict['i_edges']
rf_pix = rf_dict['rf_pix']
rf_deg = rf_dict['rf_deg']


#%%
# Parameters for dataset rois
roi_r = 1.5 # degrees
r_pix = int(roi_r * ppd)
roi_src = np.stack([rf_pix - r_pix, rf_pix + r_pix + 1], axis=1)

plt.figure()
plt.imshow(rf, extent=[j_edges[0], j_edges[-1], i_edges[-1], i_edges[0]], aspect='auto')
plt.axvline(0, color='aliceblue', linestyle='--', alpha=.5)
plt.axhline(0, color='aliceblue', linestyle='--', alpha=.5)
for r in np.arange(1, 5):
    plt.gca().add_patch(Circle((0, 0), r*ppd, fill=False, color='aliceblue', linestyle='--', alpha=.5))
    plt.text(np.sqrt(2)/2*r*ppd+10, -np.sqrt(2)/2*r*ppd+10, f'{r}°', color='aliceblue', fontsize=14)
plt.plot([roi_src[1, 0], roi_src[1, 1], roi_src[1, 1], roi_src[1, 0], roi_src[1, 0]],
        [roi_src[0, 0], roi_src[0, 0], roi_src[0, 1], roi_src[0, 1], roi_src[0, 0]], 'r')
plt.plot([rf_pix[1]], [rf_pix[0]], 'rx', markersize=10)
plt.colorbar(label='Spike rate (Hz)')
plt.title(f'Forward Correlation ({lags[0]*dt*1000:.1f}-{lags[-1]*dt*1000:.1f} ms)\nRF: {rf_deg[0]:.1f}°, {rf_deg[1]:.1f}° | pix: [{rf_pix[0]}, {rf_pix[1]}]')
plt.xlabel('Pixels (azimuth)')
plt.ylabel('Pixels (elevation)')
plt.show()

#%%
protocols = get_trial_protocols(exp)

metadata={
    'screen_resolution': screen_resolution,
    'screen_width': screen_width,
    'screen_height': screen_height,
    'screen_distance': screen_distance,
    'ppd': ppd,
    'roi_src': roi_src,
}

#%%
dset_file = data_dir / 'gaborium.dset'
if dset_file.exists():
    dset = DictDataset.load(dset_file)
else:
    dset = generate_gaborium_dataset(exp, ks_results, roi_src, 
                                     dpi_interp, dpi_deg_interp, valid_interp, 
                                     dt=dt, metadata=metadata)
    dset.save(dset_file)

#%%

valid_radius = 10
n_lags = 20
    
# Normalize stimulus
dset['stim'] = dset['stim'].float()
dset['stim'] = (dset['stim'] - dset['stim'].mean()) / dset['stim'].std()

# Create a binary mask for valid eye positions
valid_mask = np.logical_and.reduce([
    np.abs(dset['eyepos'][:,0]) < valid_radius, 
    np.abs(dset['eyepos'][:,1]) < valid_radius,
    dset['dpi_valid']
])
for iL in range(n_lags):
    valid_mask &= np.roll(valid_mask, 1, axis=0)
valid_inds = np.where(valid_mask)[0]
#%%
snr_thresh = 5
min_num_spikes = 500

# Calculate spike-triggered stimulus energy (STE)
# Determine maximally responsive lag for each cluster
stes = calc_sta(dset['stim'], dset['robs'], 
                n_lags, inds=valid_inds, device=device, batch_size=10000,
                stim_modifier=lambda x: x**2, progress=True).cpu().numpy()

# Find maximum energy lag for each cluster
signal = np.abs(stes - np.median(stes, axis=(2,3), keepdims=True))
sigma = [0, 2, 2, 2]
signal = gaussian_filter(signal, sigma)
noise = np.median(signal[:,0], axis=(1,2))
snr_per_lag = np.max(signal, axis=(2,3)) / noise[:,None]
cluster_lag = snr_per_lag.argmax(axis=1)

# Shift robs to maximally responsive lag for each cluster 
n_frames = len(dset['robs'])
robs = []
for iC in range(dset['robs'].shape[1]):
    lag = cluster_lag[iC]
    max_frame = n_frames + lag - n_lags
    robs.append(dset['robs'][lag:max_frame,iC])
robs = torch.stack(robs, axis=1)

dset.replicates = True
dset = dset[:-n_lags]
dset['robs'] = robs
dset = dset[valid_mask[:-n_lags]]
dset.replicates = False

# Remove bad clusters
good_ix = np.where(np.max(snr_per_lag, axis=1) > snr_thresh)[0]
good_ix = np.intersect1d(good_ix, np.where(dset['robs'].sum(0) > min_num_spikes)[0])
dset['robs'] = dset['robs'][:,good_ix]
n_units = len(good_ix)

# Plot STEs
fig, axs = plot_stas((stes - np.median(stes, axis=(2,3), keepdims=True))[:,:,None,:,:])
for iC in good_ix:
    iL = cluster_lag[iC]
    x0, x1 = iL, (iL+1)
    y0, y1 = -iC-1, -iC
    axs.plot([x0, x1, x1, x0, x0], [y1, y1, y0, y0, y1], 'r-')
plt.show()
print(f'Removed {len(good_ix)} bad clusters. {len(dset["robs"][0])} clusters remain.')

#%%
for key in dset.keys():
    if torch.is_tensor(dset[key]):
        dset[key] = dset[key].to(torch.float32)

print(dset)
#%%
grid_radius = 25
# Construct grid centered on peak of spatial STEs
_, _, n_y, n_x = stes.shape
weights = (dset['robs'].sum(dim=0) / dset['robs'].sum()).cpu().numpy()
ste_max = np.zeros((n_y, n_x))
for i, iC in enumerate(good_ix):
    lag = cluster_lag[iC]
    ste_max += stes[iC, lag] * weights[i]

grid_center = np.array((n_x, n_y)) // 2
grid = torch.stack(
            torch.meshgrid(
                torch.arange(-grid_radius,grid_radius+1), 
                torch.arange(-grid_radius,grid_radius+1),
                indexing='xy'
            ), 
            dim=-1).float()
grid += grid_center[None,None,:]

x_min, x_max = grid[...,0].min(), grid[...,0].max()
y_min, y_max = grid[...,1].min(), grid[...,1].max()

fig, axs = plt.subplots(1, 1, figsize=(6, 6))
im = axs.imshow(ste_max, cmap='viridis')
fig.colorbar(im, ax=axs)
axs.plot([x_min, x_min, x_max, x_max, x_min], [y_min, y_max, y_max, y_min, y_min], color='red')
axs.scatter([grid_center[0]], [grid_center[1]], color='red', marker='x')
axs.set_title('Population STE')
axs.set_xlabel('X (pixels)')
axs.set_ylabel('Y (pixels)')
plt.show()

# store meta data
dset.metadata['valid_radius'] = valid_radius
dset.metadata['grid_center'] = grid_center
dset.metadata['snr_thresh'] = snr_thresh
dset.metadata['min_num_spikes'] = min_num_spikes
dset.metadata['grid'] = grid

print(dset)

#%%

# Main class
class MLPPixelShifter(nn.Module):
    def __init__(self, grid, 
                hidden_dims=100,
                weight_init_multiplier=1, 
                input_dim=2,
                anchored=True,
                mode='bilinear') -> None:
        
        super(MLPPixelShifter, self).__init__()
        self.input_dim = input_dim
        self.grid = nn.Parameter(grid.float(), requires_grad=False) # grid is tensor of shape (n_row, n_col, 2)
        self.hidden_dims = hidden_dims
        self.weight_init_multiplier = weight_init_multiplier
        self.set_mode(mode)

        self.anchored = anchored

        # If hidden_dims is a scalar integer, convert it to a list with one element
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        # Dynamically create the layers using hidden_dims list
        layers = []
        for i in range(len(hidden_dims)):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dims[0]))
            else:
                layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i], bias=True))
            layers.append(nn.GELU())

        layers.append(nn.Linear(hidden_dims[-1], 2, bias=True))

        self.layers = nn.Sequential(*layers)

        # Initialize weights for all layers
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                layer.weight.data *= weight_init_multiplier
        
    def anchor(self):
        p0 = self.layers(torch.zeros(self.input_dim, device=self.grid.device))
        self.layers[-1].bias.data -= p0 # subtract the shift at the origin from the bias

    def forward(self, x):
        if self.anchored:
            self.anchor()
        x['stim_in'] = x['stim']
        stim = x['stim']
        if stim.ndim == 3:
            stim = stim.unsqueeze(1) # add channel dimension

        n_frames, _,n_y, n_x = stim.shape

        shift = x['eyepos']
        shift_out = self.layers(shift).squeeze(dim=1)
        grid_shift = shift_out[:, None, None, :] + self.grid[None, ...]
        _, n_y_grid, n_x_grid, _ = grid_shift.shape

        frame_grid = torch.arange(n_frames, device=self.grid.device).float() \
                          .repeat(n_y_grid, n_x_grid, 1) \
                          .permute(2, 0, 1) \
                          .unsqueeze(-1)

        sample_grid = torch.cat([grid_shift, frame_grid], dim=-1) # 1 x T x Y x X x 3 (x, y, frame)
        extent = [
                [0, n_frames - 1],
                [0, n_y - 1],
                [0, n_x - 1]
        ]

        out = grid_sample_coords(stim.permute(1,0,2,3)[None,...], # 1 x C x T x Y x X  
                                 sample_grid[None,...], 
                                 extent, 
                                 mode=self.mode,
                                 padding_mode='zeros',
                                 align_corners=True,
                                 no_grad=False)
        out = out.squeeze(dim=(0, 1))
        x['shift_out'] = shift_out
        x['stim'] = out

        return x
    def set_mode(self, mode):
        assert mode in ['nearest', 'bilinear'], 'mode must be "nearest" or "bilinear"'
        self.mode = mode

    def plot_shifts(self, x_min, x_max, y_min, y_max, image_resolution=50, quiver_resoluiton=10):

        with torch.no_grad():
            x_im = torch.linspace(x_min, x_max, image_resolution, device=self.grid.device)
            y_im = torch.linspace(y_min, y_max, image_resolution, device=self.grid.device)
            xy_im = torch.stack(
                torch.meshgrid(x_im, y_im, indexing='xy'),
                dim=-1)
                
            x_qv = torch.linspace(x_min, x_max, quiver_resoluiton, device=self.grid.device)
            y_qv = torch.linspace(y_min, y_max, quiver_resoluiton, device=self.grid.device)
            xy_qv = torch.stack(
                torch.meshgrid(x_qv, y_qv, indexing='xy'),
                dim=-1)

            shift_im = self.layers(xy_im).norm(dim=-1)
            shift_qv = self.layers(xy_qv)        
        fig, axs = plt.subplots(1,1, figsize=(6, 6))
        im = axs.imshow(shift_im.cpu(), extent=[x_min, x_max, y_min, y_max], origin='lower')
        fig.colorbar(im, ax=axs)
        axs.quiver(xy_qv[...,0].cpu(),
                   xy_qv[...,1].cpu(), 
                   shift_qv[...,1].cpu(), 
                   -shift_qv[...,0].cpu(), 
                   color='red')
        return fig, axs

# CNN model
class StimulusCNN(nn.Module):
    def __init__(self, dims, kernel_sizes,
                channels,
                n_units, strides=None,
                normalize_spatial_weights=False,
                fr_init = None,
                 ) -> None:
        super(StimulusCNN, self).__init__()
        
        self.dims = dims
        self.normalize_spatial_weights = normalize_spatial_weights

        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes]
        if isinstance(channels, int):
            channels = [channels]
        assert len(kernel_sizes) == len(channels), 'kernel_sizes and channels must have the same length'
        if strides is None:
            strides = [1] * len(kernel_sizes)
        assert len(strides) == len(kernel_sizes), 'strides must have the same length as kernel_sizes'
        
        n_layers = len(kernel_sizes)
        layers = []
        for i in range(n_layers):
            in_channels = dims[0] if i == 0 else channels[i-1] * 2
            out_channels = channels[i]
            layers.append(StackedConv2d(in_channels, out_channels, kernel_sizes[i], stride=strides[i], bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(SplitRelu())

        # Calculate the output dimensions taking into account kernel sizes, strides, and pooling
        h, w = dims[1], dims[2]
        for i in range(n_layers):
            h = (h - kernel_sizes[i]) // strides[i] + 1
            w = (w - kernel_sizes[i]) // strides[i] + 1

        dims_out = [channels[-1]*2, h, w]
        readout = NonparametricReadout(dims_out, n_units)
            
        layers.append(readout)
        layers.append(nn.Softplus())

        self.layers = nn.Sequential(*layers)

        inv_softplus = lambda x, beta=1: torch.log(torch.exp(beta*x) - 1) / beta
        if fr_init is not None:
            assert len(fr_init) == n_units, 'init_rates must have the same length as n_units'
            self.layers[-2].bias.data = inv_softplus(
                ensure_tensor(fr_init, device=self.layers[-2].bias.device)
            )


    def forward(self, x):
        # normalize spatial weights
        for layer in self.layers:
            if hasattr(layer, 'normalize_spatial_weights'):
                layer.normalize_spatial_weights()

        stim = x['stim'].float()
        if stim.ndim == 3:  # missing channel dimension
            stim = stim.unsqueeze(1) 
        x['rhat'] = self.layers(stim).squeeze()
        return x

class ShifterModel(nn.Module):
    def __init__(self, shifter_hparams, cnn_hparams):
        super(ShifterModel, self).__init__()
        self.shifter = MLPPixelShifter(**shifter_hparams)
        self.cnn = StimulusCNN(**cnn_hparams)
    
    def forward(self, x):
        x = self.shifter(x)
        x = self.cnn(x)
        return x

#%%

# Data loaders
loader_hparams = {
    'batch_size': 128,
    'num_workers': os.cpu_count(),
    'pin_memory': True,
}

train_set, val_set = torch.utils.data.random_split(dset, [.8, .2])
train_loader = torch.utils.data.DataLoader(train_set, **loader_hparams, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, **loader_hparams, shuffle=False)

# shifter params
shifter_kwargs = {
    'grid': dset.metadata['grid'],
    'weight_init_multiplier': 1,
    'hidden_dims': [400],
}
# CNN params
cnn_kwargs = {
    'dims': (1, dset.metadata['grid'].shape[0], dset.metadata['grid'].shape[1]),
    'kernel_sizes': [21, 11],
    'channels': [16, 16],
    'n_units': n_units,
    'normalize_spatial_weights': False,
    'fr_init': train_set[:]['robs'].mean(dim=0),
}

# %%

optimizer_hparams = {
    'lr': 1e-3,
    'weight_decay': 1e-3,
    'betas': (0.9, 0.999),
}
model = ShifterModel(shifter_kwargs, cnn_kwargs)
optimizer = torch.optim.AdamW(model.parameters(), **optimizer_hparams)

val_bps_aggregator = PoissonBPSAggregator()
n_epochs = 1

def train_epoch():
    for batch in (pbar := tqdm(train_loader, f'Epoch {epoch}')):
        optimizer.zero_grad()
        output = model(batch)
        loss = F.poisson_nll_loss(output['rhat'], batch['robs'], log_input=False, full=False)
        loss.backward()
        optimizer.step()
        #scheduler.step(loss, epoch)
        pbar.set_postfix({'loss': loss.item()})

def validate():
    with torch.no_grad():
        for batch in tqdm(val_loader, f'Validation'):
            output = model(batch)
            val_bps_aggregator(output)
        val_bps = val_bps_aggregator.closure().mean().item()
        print(f'Validation BPS: {val_bps}')
        val_bps_aggregator.reset()
        return val_bps

fig, axs = model.shifter.plot_shifts(-dset.metadata['valid_radius'], dset.metadata['valid_radius'], -dset.metadata['valid_radius'], dset.metadata['valid_radius'])
axs.set_title('Shifter Initialization')
axs.set_xlabel('X (deg)')
axs.set_ylabel('Y (deg)')
axs.images[0].colorbar.set_label('Shift (pixel)')
plt.show()
    
validate()
for epoch in range(n_epochs):
    train_epoch()
    validate()

fig, axs = model.shifter.plot_shifts(-dset.metadata['valid_radius'], dset.metadata['valid_radius'], -dset.metadata['valid_radius'], dset.metadata['valid_radius'])
axs.set_title('Final shifts')
axs.set_xlabel('X (deg)')
axs.set_ylabel('Y (deg)')
axs.images[0].colorbar.set_label('Shift (pixel)')
plt.show()
# %%
# Get and display stas pre and post shifting

i_slc = slice(grid_center[0] - grid_radius, grid_center[0] + grid_radius + 1)
j_slc = slice(grid_center[1] - grid_radius, grid_center[1] + grid_radius + 1)
pre_stas = calc_sta(dset['stim'][:,i_slc, j_slc], dset['robs'], [0], batch_size=2048, device=device, progress=True).detach().cpu().numpy().squeeze()
pre_stes = calc_sta(dset['stim'][:,i_slc, j_slc], dset['robs'], [0], batch_size=2048, device=device, stim_modifier=lambda x: x**2, progress=True).detach().cpu().numpy().squeeze()

with torch.no_grad():
    shifted_dset = model.shifter(dset[:])
shifted_stas = calc_sta(shifted_dset['stim'], dset['robs'], [0], batch_size=2048, device=device, progress=True).detach().cpu().numpy().squeeze()
shifted_stes = calc_sta(shifted_dset['stim'], dset['robs'], [0], batch_size=2048, device=device, stim_modifier=lambda x: x**2, progress=True).detach().cpu().numpy().squeeze()

#%%
stas = np.stack([pre_stas, shifted_stas], axis=1)
stas /= np.max(np.abs(stas), axis=(1,2,3), keepdims=True)
stes = np.stack([pre_stes, shifted_stes], axis=1)
stes -= np.median(stes, axis=(2,3), keepdims=True)
stes /= np.max(np.abs(stes), axis=(1,2,3), keepdims=True)

#%%
n_cols = 5
n_rows = np.ceil(n_units / n_cols).astype(int)
fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*2, n_rows*2))
for iU in range(n_units):
    ax = axs.flatten()[iU]
    ax.set_title(f'Unit {good_ix[iU]}')
    ax.imshow(stas[iU,0], cmap='coolwarm', vmin=-1, vmax=1, 
                extent=[0, 1, 0, 1])
    ax.imshow(stes[iU,0], cmap='coolwarm', vmin=-1, vmax=1, 
                extent=[1, 2, 0, 1])
    ax.imshow(stas[iU,1], cmap='coolwarm', vmin=-1, vmax=1, 
                extent=[0, 1, 1, 2])
    ax.imshow(stes[iU,1], cmap='coolwarm', vmin=-1, vmax=1, 
                extent=[1, 2, 1, 2])
    ax.plot([0, 2], [1, 1], 'k-', linewidth=1)
    ax.plot([1, 1], [0, 2], 'k-', linewidth=1)
    ax.set_xlim([0, 2])
    ax.set_ylim([2, 0])
    ax.set_xticks([0.5, 1.5], ['STA', 'STE'])
    ax.set_yticks([0.5, 1.5], ['Pre', 'Post'])
    
for ax in axs.flatten()[n_units:]:
    ax.axis('off')

# reduce the distance between subplots
fig.subplots_adjust(hspace=0.01, wspace=0.03)
# add colorbar
fig.tight_layout()
plt.show()
#%%

# gratings index

#%%

with torch.no_grad(): 
    dpi_shifts = model.shifter.layers(torch.from_numpy(dpi_deg).float()).squeeze().numpy()
dpi_pix_shifted = dpi_pix + np.fliplr(dpi_shifts)
dpi_shifted_interp = interp1d(t_ephys, dpi_pix_shifted, kind='linear', fill_value='extrapolate', axis=0)

roi_gratings = np.stack([rf_pix, rf_pix + 1], axis=1)

gratings_dset_file = data_dir / 'gratings.dset'
if gratings_dset_file.exists():
    gratings_dset = DictDataset.load(gratings_dset_file)
else:
    gratings_dset = generate_gratings_dataset(exp, ks_results, roi_gratings, 
                                              dpi_interp, dpi_deg_interp, valid_interp, dt=dt, metadata=metadata)
    gratings_shifted_dset = generate_gratings_dataset(exp, ks_results, roi_gratings, 
                                                      dpi_shifted_interp, dpi_deg_interp, valid_interp, dt=dt, metadata=metadata)
    gratings_dset['stim_shifted'] = gratings_shifted_dset['stim']
    gratings_dset['stim_phase_shifted'] = gratings_shifted_dset['stim_phase']
    gratings_dset.save(gratings_dset_file)

#%%

dfs = np.logical_and.reduce([
    np.abs(gratings_dset['eyepos'][:,0]) < valid_radius,
    np.abs(gratings_dset['eyepos'][:,1]) < valid_radius,
    gratings_dset['dpi_valid']
]).astype(np.float32)
gratings_dset['dfs'] = dfs

#%%
from utils.general import fit_sine

robs = gratings_dset['robs'].numpy()

sf = gratings_dset['sf'].numpy()
sfs = np.unique(sf)
ori = gratings_dset['ori'].numpy()
oris = np.unique(ori)
# one-hot embed sfs and oris
sf_ori_one_hot = np.zeros((len(robs), len(sfs), len(oris)))
for i in range(len(robs)):
    sf_idx = np.where(sfs == sf[i])[0][0]
    ori_idx = np.where(oris == ori[i])[0][0]
    sf_ori_one_hot[i, sf_idx, ori_idx] = 1

sf_ori_sta = calc_sta(sf_ori_one_hot, robs.astype(np.float64), 
                        n_lags, 
                        reverse_correlate=False, progress=True).numpy() / dt

sf_sta = calc_sta(sf_ori_one_hot.sum(2, keepdims=True), robs.astype(np.float64), 
                    n_lags, reverse_correlate=False, progress=True).numpy().squeeze() / dt


temporal_tuning = np.linalg.norm(sf_sta, axis=2)
peak_lags = np.argmax(temporal_tuning, axis=1)
sf_tuning = np.linalg.norm(sf_sta[:,peak_lags], axis=1)
peak_sf = np.argmax(sf_tuning, axis=1)
sf_snr = sf_tuning[np.arange(len(sf_tuning)), peak_sf] / np.mean(sf_tuning, axis=1)
ori_tuning = sf_ori_sta[np.arange(len(sf_ori_sta)), peak_lags, peak_sf]
peak_ori = np.argmax(ori_tuning, axis=1)
ori_snr = ori_tuning[np.arange(len(ori_tuning)), peak_ori] / np.mean(ori_tuning, axis=1)

phases = []
spikes = []
shifted_phases = []
shifted_spikes = []
for iU in tqdm(range(len(sf_sta))):
    sf_idx = peak_sf[iU]
    ori_idx = peak_ori[iU]
    lag = peak_lags[iU]

    sf_ori_idx = np.where(sf_ori_one_hot[:, sf_idx, ori_idx] > 0)[0]
    sf_ori_idx = sf_ori_idx[(sf_ori_idx + lag) < len(robs)] # only keep indices that have enough frames after lag

    # Compute phase for each frame
    stim_phases = gratings_dset['stim_phase'][sf_ori_idx].numpy().squeeze()
    if stim_phases.ndim == 3:
        stim_phases = stim_phases[:,stim_phases.shape[1]//2, stim_phases.shape[2]//2]
    stim_phases_shifted = gratings_dset['stim_phase_shifted'][sf_ori_idx].numpy().squeeze()
    if stim_phases_shifted.ndim == 3:
        stim_phases_shifted = stim_phases_shifted[:,stim_phases_shifted.shape[1]//2, stim_phases_shifted.shape[2]//2]
    stim_spikes = robs[sf_ori_idx + lag, iU]
    filters = gratings_dset['dfs'].numpy().squeeze()[sf_ori_idx + lag]  # use the same indices as spikes

    invalid = (stim_phases <= 0) | (filters == 0) # -1 indicates off screen or probe, 0 indicates sampled out of ROI
    phases.append(stim_phases[~invalid])
    spikes.append(stim_spikes[~invalid])

    invalid = (stim_phases_shifted <= 0) | (filters == 0) # -1 indicates off screen or probe, 0 indicates sampled out of ROI
    shifted_phases.append(stim_phases_shifted[~invalid])
    shifted_spikes.append(stim_spikes[~invalid])

n_phase_bins = 8
phase_bin_edges = np.linspace(0, 2*np.pi, n_phase_bins + 1)
phase_bins = np.rad2deg((phase_bin_edges[:-1] + phase_bin_edges[1:]) / 2)
n_phases = np.zeros((len(sf_sta), n_phase_bins))
n_spikes = np.zeros((len(sf_sta), n_phase_bins))
phase_response = np.zeros((len(sf_sta), n_phase_bins))
phase_response_ste = np.zeros((len(sf_sta), n_phase_bins))

n_phase_shifted = np.zeros((len(sf_sta), n_phase_bins))
n_spikes_shifted = np.zeros((len(sf_sta), n_phase_bins))
shifted_phase_response = np.zeros((len(sf_sta), n_phase_bins))
shifted_phase_response_ste = np.zeros((len(sf_sta), n_phase_bins))

for iU in tqdm(range(len(sf_sta))):
    unit_phases = phases[iU]
    unit_spikes = spikes[iU]
    # Count spikes per phase bin
    phase_bin_inds = np.digitize(unit_phases, phase_bin_edges) - 1  # bin index for each phase
    for i in range(n_phase_bins):
        n_phases[iU, i] = np.sum(phase_bin_inds == i)
        n_spikes[iU, i] = unit_spikes[phase_bin_inds == i].sum() / dt
        phase_response_ste[iU, i] = unit_spikes[phase_bin_inds == i].std() / np.sqrt(n_phases[iU,i]) / dt
    phase_response[iU] = n_spikes[iU] / n_phases[iU]

    unit_phases = shifted_phases[iU]
    unit_spikes = shifted_spikes[iU]
    # Count spikes per phase bin
    phase_bin_inds = np.digitize(unit_phases, phase_bin_edges) - 1  # bin index for each phase
    for i in range(n_phase_bins):
        n_phase_shifted[iU, i] = np.sum(phase_bin_inds == i)
        n_spikes_shifted[iU, i] = unit_spikes[phase_bin_inds == i].sum() / dt
        shifted_phase_response_ste[iU, i] = unit_spikes[phase_bin_inds == i].std() / np.sqrt(n_phases[iU,i]) / dt
    shifted_phase_response[iU] = n_spikes_shifted[iU] / n_phase_shifted[iU]
    
results = []
for iU in tqdm(range(len(sf_sta))):
    unit_phases = phases[iU]
    unit_spikes = spikes[iU]
    if np.sum(unit_spikes) < 50:
        results.append(None)
        continue
    results.append(fit_sine(unit_phases, unit_spikes, omega=1.0, variance_source='observed_y'))

shifted_results = []
for iU in tqdm(range(len(sf_sta))):
    unit_phases = shifted_phases[iU]
    unit_spikes = shifted_spikes[iU]
    if np.sum(unit_spikes) < 100:
        shifted_results.append(None)
        continue
    shifted_results.append(fit_sine(unit_phases, unit_spikes, omega=1.0, variance_source='observed_y'))

#%%
cid = 90
plt.figure()
plt.imshow(sf_ori_sta[cid, peak_lags[cid], :] * 240, cmap='viridis')
plt.show()
#%%
plt.figure()
plt.scatter(phases[cid], spikes[cid])
plt.show()
plt.figure()
plt.scatter(shifted_phases[cid], shifted_spikes[cid])
plt.xlim([0, 2*np.pi])
plt.show()


#%%
def plot_phase_response(res, ax=None):
    if res is None:
        raise ValueError('No fit')
    amp = res['amplitude']
    amp_se = res['amplitude_se']
    phase_offset = res['phase_offset_rad']
    phase_offset_se = res['phase_offset_rad_se']
    C = res['C']
    mi = res['modulation_index']
    mi_se = res['modulation_index_se']
    if np.isnan(mi) or np.isnan(mi_se) or np.isnan(amp) or np.isnan(amp_se) or np.isnan(phase_offset) or np.isnan(phase_offset_se):
        raise ValueError('NaN in fit')

    smoothed_phases = np.linspace(0, 2*np.pi, 100)
    smoothed_fit = amp * np.sin(smoothed_phases + phase_offset) + C
    smoothed_fit_max = (amp+amp_se) * np.sin(smoothed_phases + phase_offset) + C
    smoothed_fit_min = (amp-amp_se) * np.sin(smoothed_phases + phase_offset) + C

    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(np.rad2deg(smoothed_phases), smoothed_fit/dt, color='red')
    ax.fill_between(np.rad2deg(smoothed_phases), smoothed_fit_min/dt, smoothed_fit_max/dt, color='red', alpha=0.2)
    ylim = ax.get_ylim()
    ax.set_ylim([0, ylim[1]])
    ax.set_xlabel('Phase (radians)')
    ax.set_ylabel('Spikes / second')

for iU in [90]:#range(len(results)):
    res = results[iU]
    res_shifted = shifted_results[iU]

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].errorbar(phase_bins, phase_response[iU], yerr=phase_response_ste[iU], fmt='o-', ecolor='C0', capsize=5, zorder=0)
    axs[1].errorbar(phase_bins, shifted_phase_response[iU], yerr=shifted_phase_response_ste[iU], fmt='o-', ecolor='C0', capsize=5, zorder=0)
    ax0_title = f'Unit {iU}\nOriginal'
    ax1_title = f'Unit {iU}\nShifted'
    try:
        plot_phase_response(res, ax=axs[0])
        plot_phase_response(res_shifted, ax=axs[1])
        ax0_title += f'\nModulation Index {res["modulation_index"]:.2f} +/- {res["modulation_index_se"]:.2f}'
        ax1_title += f'\nModulation Index {res_shifted["modulation_index"]:.2f} +/- {res_shifted["modulation_index_se"]:.2f}'
    except Exception as e:
        print(f'Error plotting unit {iU}')
    axs[0].set_title(ax0_title)
    axs[1].set_title(ax1_title)
    plt.show()
# %%

mi_original = np.array([res['modulation_index'] if res is not None else np.nan for res in results])
mi_original_se = np.array([res['modulation_index_se'] if res is not None else np.nan for res in results])
mi_shifted = np.array([res['modulation_index'] if res is not None else np.nan for res in shifted_results])
mi_shifted_se = np.array([res['modulation_index_se'] if res is not None else np.nan for res in shifted_results])
n_spikes = np.array([np.sum(spikes[iU]) for iU in range(len(sf_sta))])

plt.figure()
plt.scatter(mi_original, mi_shifted, c=n_spikes, cmap='viridis')
plt.gca().set_aspect('equal', adjustable='box')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('Original Modulation Index')
plt.ylabel('Shifted Modulation Index')
plt.show()
#%%

