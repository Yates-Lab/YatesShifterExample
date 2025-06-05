import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math

# list which modules are available to import
__all__ = ['Noise', 'SplitRelu', 'WindowedConv2d', 'Conv2d', 'StackedConv2d']

class Noise(nn.Module):
    '''
    Add Gaussian noise to the input tensor. This acts as a regularizer during training.

    Parameters:
    -----------
    sigma : float
        Standard deviation of the Gaussian noise.
    '''
    def __init__(self, sigma=0.1) -> None:
        super().__init__()
        self.sigma = sigma
    def forward(self, x):
        if self.training:
            x = x + torch.randn_like(x) * self.sigma
        return x

class SplitRelu(nn.Module):
    def __init__(self, split=1) -> None:
        super().__init__()
        self.split = split
    def forward(self, x):
        return torch.cat([torch.relu(x), torch.relu(-x)], dim=self.split)
   
class Conv(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError

    def plot_weights(self):
        weights = self.weight.detach().cpu().numpy()

        c_out, c_in, _, _ = weights.shape
        # One subplot per input channel
        n_in_cols = np.ceil(np.sqrt(c_in)).astype(int)
        n_in_rows = np.ceil(c_in / n_in_cols).astype(int)

        # One imshow per output channel
        n_out_cols = np.ceil(np.sqrt(c_out)).astype(int)
        n_out_rows = np.ceil(c_out / n_out_cols).astype(int)

        fig, axs = plt.subplots(n_in_rows, n_in_cols, figsize=(8, 8))
        if n_in_cols == 1 and n_in_rows == 1:
            axs = np.array([[axs]])
        for iI in range(n_in_cols * n_in_rows):
            ax = axs.flatten()[iI]
            ax.axis('off')

            if iI > c_in - 1:
                continue

            for iO in range(c_out):
                r = iO // n_out_rows
                c = iO % n_out_cols
                ws = weights[iO, iI]
                ws_max = np.abs(ws).max()
                ax.imshow(weights[iO, iI],
                          cmap='coolwarm', 
                          extent=[c, c+1, -r, -r-1],
                          vmin=-ws_max, vmax=ws_max)
            ax.set_title(f'Input Channel {iI}', fontsize=8, pad=2)
            ax.set_xlim([0, n_out_cols])
            ax.set_ylim([-n_out_rows, 0])
        
        return fig, axs
    
class Conv2d(Conv):
    '''
    A basic 2D convolution with standard options. Only use instead of pytorch default because of plot weights functionality
    '''

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)

    def forward(self, x):
        return self.conv(x)
    
    @property
    def weight(self):
        return self.conv.weight

class StackedConv2d(Conv):
    '''
    A 2D convolutional layer with multiple layers of convolutional filters that are stacked to approximate a bigger kernel.

    Parameters:
    '''

    def __init__(self, in_channels,
                out_channels, kernel_size,
                sub_kernel_size=3, padding=0,
                padding_mode='replicate',
                groups=1,
                bias=False,
                dropout_percentage=0, **kwargs):
        
        super().__init__()
        
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        assert kernel_size[0] == kernel_size[1], "spatial kernel size must be square"
        self.kernel_size = kernel_size[0] # (H, W, num_lags)
        self.num_layers = (kernel_size[0] - sub_kernel_size) // (sub_kernel_size-1) + 1
        self.sub_kernel_size = sub_kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding
        self.padding_mode = padding_mode

        # first
        convs = [nn.Conv2d(in_channels, out_channels, kernel_size=sub_kernel_size,
                padding=padding, padding_mode=padding_mode,
                groups=groups, # can be groupwise to reduce parameters
                bias=False, # only the last layer gets the optional bias
                **kwargs)]

        if dropout_percentage>0:
            convs.append(nn.Dropout2d(dropout_percentage))
        
        for i in range(1, self.num_layers):
            convs.append(nn.Conv2d(out_channels, out_channels, kernel_size=sub_kernel_size,
                padding=padding, padding_mode=padding_mode, groups=groups,
                bias=bias if i == self.num_layers-1 else False, 
                **kwargs))

            if dropout_percentage>0:
                convs.append(nn.Dropout2d(dropout_percentage))

        self.convs = nn.Sequential(*convs)
    
    @property
    def weight(self):
        # combine the weights of all the layers to get the full-size kernel
        H = W = self.kernel_size
        # add padding if necessary
        if self.padding == 0:
            H = W = H + (self.sub_kernel_size-1) * (self.num_layers - 1)
        x = torch.zeros(self.in_channels, self.in_channels, H, W, device=self.convs[0].weight.device)
        for i in range(self.in_channels):
            x[i, i, H//2, W//2] = 1.0
        w = self.convs(x)
        return w.permute(1,0,2,3)
    
    def forward(self, x):
        return self.convs(x)

class WindowedConv2d(Conv):
    '''
    A 2D convolutional layer with a windowed filter. The window must be in torch.signal.windows, and is applied to each dimension independently.

    Parameters:
    -----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple
        Size of the convolutional kernel.
    stride : int or tuple
        Stride of the convolutional kernel.
    padding : int or tuple
        Padding of the convolutional kernel.
    dilation : int or tuple
        Dilation of the convolutional kernel.
    groups : int
        Number of groups for grouped convolution.
    bias : bool
        Whether to include a bias term.
    padding_mode : str
        Padding mode of the convolutional kernel. window_type : str
        Type of window function to use.
    window_kwargs : dict
        Additional keyword arguments to pass to the window function.
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 dilation=1, groups=1, bias=True, padding_mode='zeros', 
                 window_type='hann', window_kwargs={}) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        assert hasattr(torch.signal.windows, window_type), f'Window type {window_type} not found in torch.signal.windows'
        window_function = getattr(torch.signal.windows, window_type)
        win1 = window_function(kernel_size[0], **window_kwargs)
        win2 = window_function(kernel_size[1], **window_kwargs)
        window = win1[:, None] * win2[None, :]
        window = window / window.max()
        self.window = nn.Parameter(window[None, None, ...], requires_grad=False)

    def forward(self, x):
        weights = self.conv.weight * self.window
        return self.conv._conv_forward(x, weights, self.conv.bias)

    @property
    def weight(self):
        return self.conv.weight * self.window

class SeparableWindowedConv2D(Conv):
    '''
    A depth/space separable convolutional layer with a windowed kernel.
    This layer is used to model a low-rank linear receptive field. 
    '''

    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, dilation=1, bias=True, 
                 window_type='hann', window_kwargs={}) -> None:
        super().__init__()
        self.channel_weights = nn.Parameter(
            F.normalize(
                torch.randn(out_channels, in_channels, 1, 1),
                p=2, dim=(1,2,3)
            ), 
            requires_grad=True)

        self.spatial = WindowedConv2d(
            out_channels, out_channels, kernel_size, 
            stride, padding, dilation, groups=out_channels, bias=bias,
            window_type=window_type, window_kwargs=window_kwargs
        )

    def forward(self, x):
        # set magnitude to 1 for depthwise kernel
        self.channel_weights.data = F.normalize(self.channel_weights, p=2, dim=(1,2,3))
        x = F.conv2d(x, self.channel_weights)
        x = self.spatial(x)
        return x

    def plot_weights(self):
        channel_weights = self.channel_weights.detach().cpu().numpy()
        spatial_weights = self.spatial.weight.detach().cpu().numpy()

        n_out, _, _, _ = channel_weights.shape

        fig, axs = plt.subplots(n_out, 2, figsize=(4, 8))
        for i in range(n_out):
            axs[i, 0].axhline(0, color='k', linestyle='--')
            axs[i, 0].plot(channel_weights[i].squeeze())
            axs[i, 0].set_ylabel(f'Channel {i}', fontsize=10, labelpad=10)
            if i < n_out - 1:
                axs[i,0].set_xticks([])
                axs[i,0].set_xticklabels([])
            axs[i, 0].set_yticks([])
            axs[i, 0].set_yticklabels([])
            vmax = np.max(np.abs(spatial_weights[i]))
            axs[i, 1].imshow(spatial_weights[i].squeeze(), cmap='coolwarm', vmin=-vmax, vmax=vmax)
            axs[i, 1].axis('off') 
        return fig, axs

class BaseFactorizedReadout(nn.Module):
    """
    Base class for factorized readouts that separates the feature mapping 
    (via a 1x1 convolution) from the spatial readout. It provides a common 
    plotting utility that visualizes the feature weights and the effective 
    spatial weights.
    """
    def __init__(self, dims, n_units, bias=True):
        super().__init__()
        self.dims = dims  # (channels, H, W)
        self.n_units = n_units
        # Common feature mapping: maps input channels to n_units.
        self.features = nn.Conv2d(dims[0], n_units, kernel_size=1, bias=False)
        if bias:
            self.bias = nn.Parameter(torch.zeros(n_units))
    
    def forward(self, x):
        # Must be implemented by subclasses.
        raise NotImplementedError("Subclasses should implement forward method.")
    
    def get_spatial_weights(self):
        # Should return a tensor of shape (n_units, H, W) representing
        # the effective spatial weights.
        raise NotImplementedError("Subclasses should implement get_spatial_weights method.")
    
    def plot_weights(self):
        """
        Plot the feature weights and the effective spatial weights in a single figure with two axes.
        The spatial weights are arranged in a grid using the imshow extent.
        """
        # Extract feature weights.
        # The feature weights are from a 1x1 conv: shape (n_units, in_channels, 1, 1)
        # Squeeze to (n_units, in_channels)
        feature_weights = self.features.weight.detach().cpu().numpy().squeeze()
        # Extract spatial weights: expected shape (n_units, H, W)
        spatial_weights = self.get_spatial_weights().detach().cpu().numpy()
        n_units = spatial_weights.shape[0]
        
        # Create a single figure with two vertically stacked axes.
        fig, axs = plt.subplots(2, 1, figsize=(6, 10))
        
        # Plot feature weights.
        feat_max = np.abs(feature_weights).max()
        axs[0].imshow(feature_weights.T, cmap='coolwarm', interpolation='none', vmin=-feat_max, vmax=feat_max)
        axs[0].set_title('Feature Weights')
        axs[0].set_xlabel('Unit')
        axs[0].set_ylabel('Channel')
        
        # Determine grid size for spatial weights.
        n_cols = int(np.ceil(np.sqrt(n_units)))
        n_rows = int(np.ceil(n_units / n_cols))
        
        # For spatial weights, loop over each unit and display it in the correct grid position.
        for i in range(n_units):
            r = i // n_cols
            c = i % n_cols
            ws = spatial_weights[i]
            ws_max = np.abs(ws).max()
            axs[1].imshow(ws, cmap='coolwarm', extent=[c, c+1, -r, -r-1],
                          vmin=-ws_max, vmax=ws_max)
        
        axs[1].axis('off')
        axs[1].set_title('Spatial Weights')
        axs[1].set_xlim([0, n_cols])
        axs[1].set_ylim([-n_rows, 0])
        fig.tight_layout()
        return fig, axs

# -----------------------------------------------------
# Subclass 1: Nonparametric Readout (FactorizedReadout)
# -----------------------------------------------------
class NonparametricReadout(BaseFactorizedReadout):
    def __init__(self, dims, n_units, bias=True):
        super().__init__(dims, n_units, bias)
        # The spatial component is implemented as a grouped convolution.
        self.locations = WindowedConv2d(n_units, n_units, kernel_size=dims[1:], groups=n_units, bias=False)
        if bias:
            self.bias = nn.Parameter(torch.zeros(n_units), requires_grad=True)
        
        # Initialize the location weights with a Gaussian-like profile.
        self._initialize_gaussian(center=(dims[1] // 2, dims[2] // 2),
                                  radius=min(dims[1], dims[2]) // 4,
                                  sigma_scale=0.3)
        self.normalize_spatial_weights()
    
    def forward(self, x):
        x = self.features(x)  # (batch, n_units, H, W)
        x = self.locations(x)  # (batch, n_units, 1, 1) typically
        x = x.squeeze()
        if hasattr(self, 'bias'):
            x = x + self.bias[None, :]
        return x
    
    def normalize_spatial_weights(self):
        with torch.no_grad():
            # Normalize each unit's weights to unit norm
            F.normalize(self.locations.weight, p=2, dim=[1, 2, 3], out=self.locations.weight)
    
    def _initialize_gaussian(self, center, radius, sigma_scale=0.3):
        # Initialize location weights with a Gaussian shape (with slight random offsets)
        weight_shape = self.locations.weight.shape  # (n_units, 1, H, W)
        n_units, channels_per_group, n_y, n_x = weight_shape
        
        y_coords, x_coords = torch.meshgrid(
            torch.arange(n_y, dtype=torch.float32),
            torch.arange(n_x, dtype=torch.float32),
            indexing='ij'
        )
        y_center, x_center = center
        sigma = radius * sigma_scale
        
        for unit in range(n_units):
            offset_scale = radius * 0.1
            y_offset = torch.randn(1).item() * offset_scale
            x_offset = torch.randn(1).item() * offset_scale
            unit_y_center = y_center + y_offset
            unit_x_center = x_center + x_offset
            unit_gaussian = torch.exp(-((y_coords - unit_y_center) ** 2 +
                                        (x_coords - unit_x_center) ** 2) / (2 * sigma ** 2))
            unit_gaussian = unit_gaussian / unit_gaussian.sum()
            for c in range(channels_per_group):
                self.locations.weight.data[unit, c] = unit_gaussian

    def get_spatial_weights(self):
        # Return spatial weights from the locations convolution.
        return self.locations.weight.detach().cpu().squeeze(1)

# -----------------------------------------------------
# Subclass 2: Gaussian Readout
# -----------------------------------------------------
class GaussianReadout(BaseFactorizedReadout):
    def __init__(self, dims, n_units, bias=False):
        super().__init__(dims, n_units, bias)
        # Define learnable Gaussian parameters.
        center = (dims[1] / 2, dims[2] / 2)
        self.mean = nn.Parameter(torch.tensor([[center[0], center[1]]] * n_units, dtype=torch.float32))
        radius = min(dims[1], dims[2]) / 4
        self.std = nn.Parameter(torch.ones(n_units, 2, dtype=torch.float32) * (radius * 0.3))
        self.theta = nn.Parameter(torch.zeros(n_units, dtype=torch.float32))
        
        # Create a spatial grid (pixel coordinates)
        grid_y, grid_x = torch.meshgrid(
            torch.arange(dims[1], dtype=torch.float32),
            torch.arange(dims[2], dtype=torch.float32),
            indexing='ij'
        )
        grid = torch.stack([grid_y, grid_x], dim=-1)  # shape: (H, W, 2)
        self.register_buffer('grid', grid)
    
    def compute_gaussian_mask(self):
        # Clamp std to avoid division by zero.
        std = self.std.clamp(min=1e-3)
        mean = self.mean.unsqueeze(1).unsqueeze(1)  # (n_units, 1, 1, 2)
        std = std.unsqueeze(1).unsqueeze(1)           # (n_units, 1, 1, 2)
        theta = self.theta  # (n_units,)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        R = torch.stack([
            torch.stack([cos_theta, -sin_theta], dim=-1),
            torch.stack([sin_theta,  cos_theta], dim=-1)
        ], dim=-2)  # (n_units, 2, 2)
        
        grid = self.grid.unsqueeze(0)  # (1, H, W, 2)
        centered_grid = grid - mean     # (n_units, H, W, 2)
        rotated_grid = torch.einsum('nhwi,nij->nhwj', centered_grid, R)
        exponent = -0.5 * ((rotated_grid / std) ** 2).sum(dim=-1)
        gaussian_mask = torch.exp(exponent)
        gaussian_mask = gaussian_mask / (gaussian_mask.sum(dim=(-1,-2), keepdim=True) + 1e-8)

        return gaussian_mask
    
    def forward(self, x):
        feat = self.features(x)  # (batch, n_units, H, W)
        gaussian_mask = self.compute_gaussian_mask()  # (n_units, H, W)
        out = (feat * gaussian_mask.unsqueeze(0)).sum(dim=(-2, -1))
        if hasattr(self, 'bias'):
            out = out + self.bias
        return out
    
    def get_spatial_weights(self):
        # The effective spatial weights in this readout are given by the Gaussian mask.
        return self.compute_gaussian_mask().detach().cpu()

# -----------------------------------------------------
# Subclass 3: Hybrid Readout
# -----------------------------------------------------
class HybridReadout(BaseFactorizedReadout):
    def __init__(self, dims, n_units, bias=True):
        super().__init__(dims, n_units, bias)
        # Nonparametric spatial weights (learnable, unconstrained map).
        self.spatial_weights = nn.Parameter(.1*torch.randn(n_units, 1, dims[1], dims[2]))
        if bias:
            self.bias = nn.Parameter(torch.zeros(n_units))
            
        center = (dims[1] / 2, dims[2] / 2)
        radius = min(dims[1], dims[2]) / 3
        
        # # Initialize nonparametric weights with a Gaussian-like profile.
        # y_coords, x_coords = torch.meshgrid(
        #     torch.arange(dims[1], dtype=torch.float32),
        #     torch.arange(dims[2], dtype=torch.float32),
        #     indexing='ij'
        # )
        # for unit in range(n_units):
        #     offset_scale = radius * 0.1
        #     y_offset = torch.randn(1).item() * offset_scale
        #     x_offset = torch.randn(1).item() * offset_scale
        #     unit_center_y = center[0] + y_offset
        #     unit_center_x = center[1] + x_offset
        #     sigma = radius * 0.3
        #     unit_gaussian = torch.exp(-((y_coords - unit_center_y) ** 2 +
        #                                 (x_coords - unit_center_x) ** 2) / (2 * sigma ** 2))
        #     unit_gaussian = unit_gaussian / unit_gaussian.sum()
        #     self.spatial_weights.data[unit, 0] = unit_gaussian
        
        # Learnable Gaussian parameters.
        self.mean = nn.Parameter(torch.tensor([[center[0], center[1]]] * n_units, dtype=torch.float32))
        self.std = nn.Parameter(torch.ones(n_units, 2, dtype=torch.float32) * (radius * 0.3))
        self.theta = nn.Parameter(torch.zeros(n_units, dtype=torch.float32))
        # self.mean.requires_grad = True
        # self.std.requires_grad = True
        # self.theta.requires_grad = True
        
        grid_y, grid_x = torch.meshgrid(
            torch.arange(dims[1], dtype=torch.float32),
            torch.arange(dims[2], dtype=torch.float32),
            indexing='ij'
        )
        grid = torch.stack([grid_y, grid_x], dim=-1)  # (H, W, 2)
        self.register_buffer('grid', grid)
        
        self.normalize_spatial_weights()
    
    def normalize_spatial_weights(self):
        with torch.no_grad():
            self.spatial_weights.data = F.normalize(self.spatial_weights.data, p=2, dim=(1,2,3))
    
    def compute_gaussian_mask(self):
        std = self.std.clamp(min=1e-3)
        mean = self.mean.unsqueeze(1).unsqueeze(1)
        std = std.unsqueeze(1).unsqueeze(1)
        theta = self.theta
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        R = torch.stack([
            torch.stack([cos_theta, -sin_theta], dim=-1),
            torch.stack([sin_theta,  cos_theta], dim=-1)
        ], dim=-2)
        grid = self.grid.unsqueeze(0)
        centered_grid = grid - mean
        rotated_grid = torch.einsum('nhwi,nij->nhwj', centered_grid, R)
        exponent = -0.5 * ((rotated_grid / std) ** 2).sum(dim=-1)
        gaussian_mask = torch.exp(exponent)
        gaussian_mask = gaussian_mask / (gaussian_mask.sum(dim=(-1,-2), keepdim=True) + 1e-8)
        return gaussian_mask
    
    def forward(self, x):
        feat = self.features(x)  # (batch, n_units, H, W)
        gaussian_mask = self.compute_gaussian_mask()  # (n_units, H, W)
        A = self.spatial_weights.squeeze(1)  # (n_units, H, W)
        final_weight = A * gaussian_mask  # Hybrid effective spatial weights
        out = (feat * final_weight.unsqueeze(0)).sum(dim=(-2, -1))
        if hasattr(self, 'bias'):
            out = out + self.bias
        return out
    
    def get_spatial_weights(self):
        # Return the effective spatial weights (nonparametric weights modulated by the Gaussian).
        gaussian_mask = self.compute_gaussian_mask().detach().cpu()
        A = self.spatial_weights.detach().cpu().squeeze(1)
        return A * gaussian_mask
    