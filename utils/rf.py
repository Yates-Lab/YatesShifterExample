import numpy as np
import torch
from tqdm import tqdm
from unittest.mock import Mock
from .general import ensure_tensor
import matplotlib.pyplot as plt

class Gaussian2D:
    '''
    A 2D Gaussian function for fitting receptive fields or other punctate 2D data.
        
    Parameters:
    -----------
    amp : float
        Amplitude of the Gaussian.
    x0 : float
        x-coordinate of the center of the Gaussian.
    y0 : float
        y-coordinate of the center of the Gaussian.
    sigma_x : float
        Standard deviation of the Gaussian in the x-direction.
    sigma_y : float
        Standard deviation of the Gaussian in the y-direction.
    theta : float
        Rotation angle of the Gaussian in radians.
    bias : float
        Bias of the Gaussian.
    '''
    def __init__(self, amp, x0, y0, sigma_x, sigma_y, theta, bias):
        self.amp = amp
        self.x0 = x0
        self.y0 = y0
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.theta = theta
        self.bias = bias

    @staticmethod
    def est_p0(image, x = None, y = None):
        '''
        Estimate initial parameters for fitting a 2D Gaussian to an image.

        Parameters:
        -----------
        image : numpy.ndarray
            The image to fit the Gaussian to.
        x : numpy.ndarray
            The x-coordinates of the image. Default: None (np.arange(image.shape[1])).
        y : numpy.ndarray
            The y-coordinates of the image. Default: None (np.arange(image.shape[0])).
        '''
        if x is None:
            x = np.arange(image.shape[1])
        if y is None:
            y = np.arange(image.shape[0])
        max_i, max_j = np.unravel_index(np.argmax(image), image.shape)
        x0 = x[max_j]
        y0 = y[max_i]
        amp = image[max_i, max_j]
        sigma_x = 20
        sigma_y = 20
        theta = 0
        bias = np.median(image)
        return [amp, x0, y0, sigma_x, sigma_y, theta, bias]
    
    @staticmethod
    def forward(x, y, amp, x0, y0, sigma_x, sigma_y, theta, bias):
        '''
        Evaluate a 2D Gaussian at the specified coordinates.

        Parameters:
        -----------
        x : numpy.ndarray
            The x-coordinates to evaluate the Gaussian at.
        y : numpy.ndarray
            The y-coordinates to evaluate the Gaussian at.
        amp : float
            Amplitude of the Gaussian.
        x0 : float
            x-coordinate of the center of the Gaussian.
        y0 : float
            y-coordinate of the center of the Gaussian.
        sigma_x : float
            Standard deviation of the Gaussian in the x-direction.
        sigma_y : float
            Standard deviation of the Gaussian in the y-direction.
        theta : float
            Rotation angle of the Gaussian in radians.
        bias : float
            Bias of the Gaussian.
        '''
        x, y = x - x0, y - y0
        a = np.cos(theta)**2/(2*sigma_x**2) + np.sin(theta)**2/(2*sigma_y**2)
        b = -np.sin(2*theta)/(4*sigma_x**2) + np.sin(2*theta)/(4*sigma_y**2)
        c = np.sin(theta)**2/(2*sigma_x**2) + np.cos(theta)**2/(2*sigma_y**2)
        return amp * np.exp(-(a*x**2 + 2*b*x*y + c*y**2)) + bias

    def __call__(self, x, y):
        return self.forward(x, y, self.amp, self.x0, self.y0, self.sigma_x, self.sigma_y, self.theta, self.bias)
    
    def __repr__(self):
        return f'Gaussian2D(amp={self.amp}, x0={self.x0}, y0={self.y0}, sigma_x={self.sigma_x}, sigma_y={self.sigma_y}, theta={self.theta}, bias={self.bias})'
    
    def fit(self, image, x = None, y = None):
        '''
        Fit the Gaussian to some data

        Parameters:
        -----------
        image : numpy.ndarray
            The image to fit the Gaussian to.
        x : numpy.ndarray
            The x-coordinates of the image.
            Must either have the same shape as z or be 1D.
            If None, np.arange(image.shape[1]) is used.
        y : numpy.ndarray
            The y-coordinates of the image.
            Must either have the same shape as z or be 1D.
            If None, np.arange(image.shape[0]) is used.
        '''
        from scipy.optimize import curve_fit

        if x is None:
            x = np.arange(image.shape[1])
        if y is None:
            y = np.arange(image.shape[0])
        if x.ndim == 1 and y.ndim == 1:
            x, y = np.meshgrid(x, y)

        assert np.all(image.shape == x.shape) and np.all(image.shape == y.shape), 'x, y, and image must have the same shape'

        def f(xy, amp, x0, y0, sigma_x, sigma_y, theta, bias):
            x, y = xy
            image = self.forward(x, y, amp, x0, y0, sigma_x, sigma_y, theta, bias)
            return image.ravel()
        
        p0 = [self.amp, self.x0, self.y0, self.sigma_x, self.sigma_y, self.theta, self.bias]
        try:
            popt, pcov = curve_fit(f, (x, y), image.ravel(), p0=p0)
        except:
            print('Fit failed')
            self.amp, self.x0, self.y0, self.sigma_x, self.sigma_y, self.theta, self.bias = [np.nan]*7
            return self


        self.amp, self.x0, self.y0, self.sigma_x, self.sigma_y, self.theta, self.bias = popt
        # check if fit is unreliable
        if np.any(np.diag(pcov) > 1e3):
            pass
            #print('Fit may have failed')
            #self.amp, self.x0, self.y0, self.sigma_x, self.sigma_y, self.theta, self.bias = [np.nan]*7
        return self
    
    def plot(self, x, y, ax=None):
        if x.ndim == 1 and y.ndim == 1:
            x, y = np.meshgrid(x, y)

        if ax is None:
            ax = plt.gca()
        ax.imshow(self(x, y), cmap='Reds')
        return ax

def calc_sta(stim, robs, lags, dfs=None, inds=None, stim_modifier=lambda x: x, reverse_correlate=True, batch_size=None, device=None, progress=False):
    '''
    Calculates the spike-triggered average (STA) for a given stimulus and response.

    Parameters:
    -----------
    stim : numpy.ndarray or torch.Tensor
        The stimulus data. Shape: (n_frames, n_channels, n_y, n_x) or (n_frames, n_y, n_x)
    robs : numpy.ndarray or torch.Tensor
        Observed rate for the neural data. Shape: (n_frames, n_units)
    lags : int or list
        The number of lags to calculate the STA for. If an int, the STA will be calculated for lags 0 to lags-1.
        If a list, the STA will be calculated for the specified lags.
    dfs : numpy.ndarray or torch.Tensor
        Datafilters for the neural data. Shape: (n_frames, n_units). Used to weight the response data.
        If None, the response data will not be weighted.
        If a 1D array, the datafilter will be repeated for all units.
    inds : numpy.ndarray or torch.Tensor
        Indices to calculate the STA for. Default: None (all indices).
    stim_modifier : function
        Function to modify the stimulus data before calculating the STA. Default: lambda x: x
    reverse_correlate : bool
        If True, the STA will be normalized by the number of spikes. If False, the STA will be normalized by the number of frames.
    batch_size : int
        The batch size to use for calculating the STA. Default: None (all frames).
    device : torch.device
        The device to use for the calculations. Default: None (CPU).
    progress : bool
        If True, a progress bar will be displayed. Default: False
        
    Returns:
    --------
    torch.Tensor
        The spike-triggered average. Shape: (n_units, n_lags, n_channels, n_y, n_x)

    author: RKR 2/7/2024 (largely copied from Dekel)
    '''
    stim = ensure_tensor(stim, dtype=torch.float32)
    # If missing channel dimension, add it
    if stim.dim() == 3:
        stim = stim.unsqueeze(1)

    robs = ensure_tensor(robs, dtype=torch.float32)
    dfs = ensure_tensor(dfs, dtype=torch.float32) if dfs is not None else None
    inds = ensure_tensor(inds, dtype=torch.long) if inds is not None else None

    if robs.dim() == 1:
        robs = robs[:,None]

    if dfs is not None and dfs.dim() == 1:
        # repeat dfs over all units
        dfs = dfs[:,None].repeat(1, robs.shape[1])
    
    n_frames, n_c, n_y, n_x = stim.shape
    n_frames_robs, n_units  = robs.shape
    
    assert n_frames == n_frames_robs, f"Number of frames in stim ({n_frames}) does not match number of frames in robs ({n_frames_robs})"

    if dfs is not None:
        n_frames_dfs, n_units_dfs = dfs.shape
        assert n_units == n_units_dfs, f"Number of units in robs ({n_units}) does not match number of units in dfs ({n_units_dfs})"
        assert n_frames == n_frames_dfs, f"Number of frames in dfs ({n_frames_dfs}) does not match number of frames in stim ({n_frames})"

    if device is None:
        device = stim.device

    if inds is None:
        inds = torch.arange(stim.shape[0])

    if isinstance(lags, int):
        lags = [l for l in range(lags)]
    
    n_lags = len(lags)
    
    sts = torch.zeros((n_units, n_lags, n_c, n_y, n_x), dtype=torch.float64).to(device)
    div = torch.zeros((n_units, n_lags, n_c, n_y, n_x), dtype=torch.float64).to(device)
    if batch_size is None:
        batch_size = n_frames
    n_batches = int(np.ceil(len(inds) / batch_size))

    pbar = tqdm(total=n_lags*n_batches, desc='Calculating STA') if progress else Mock()
    try:
        for iL,lag in enumerate(lags):  
            ix = inds[inds < n_frames-lag] # prevent indexing out of bounds
            for iB in range(0, len(ix), batch_size):
                bs = slice(iB, iB+batch_size)
                inds_batch = ix[bs]
                stim_batch = stim_modifier(stim[inds_batch,...].to(device))
                robs_batch = robs[inds_batch+lag,:].to(device)
                if dfs is not None:
                    dfs_batch = dfs[inds_batch+lag,:].to(device)
                    robs_batch = robs_batch * dfs_batch
                    if reverse_correlate:
                        div += robs_batch.sum(dim=0)[:,None,None,None,None]
                    else:
                        div[:,iL] += (stim_batch[None,...] * (dfs_batch.T)[...,None,None,None]).sum(dim=1)
                else:
                    if reverse_correlate:
                        div += robs_batch.sum(dim=0)[:,None,None,None,None]
                    else:
                        div[:,iL] += stim_batch.sum(dim=0)[None,...]
                sts[:,iL] += torch.einsum('bcij, bn->ncij',stim_batch, robs_batch)
                torch.cuda.empty_cache()
                pbar.update(1)
    finally:
        pbar.close()

    out = torch.squeeze(sts / div, dim=2)
    return out

def plot_stas(sta, row_labels:list = None, col_labels:list = None, share_scale=False, ax=None):
    """
    Plots STAs across lags.

    Parameters
    ----------
    sta : np.ndarray
        STA with shape (n_rows, n_lags, n_channels, n_y, n_x) or (n_lags, n_channels, n_y, n_x)
    """
    if isinstance(sta, torch.Tensor):
        sta = sta.detach().cpu().numpy()

    if sta.ndim == 4:
        sta = sta[np.newaxis, ...]

    n_rows, n_lags, n_c, n_y, n_x= sta.shape
    if row_labels is not None:
        assert len(row_labels) == n_rows, 'Number of row labels must match number of rows in sta'
    
    scale = 1 / (np.max(np.abs(sta)) * 2)
    aspect = n_x / n_y
    imshow_kwargs = dict(aspect='equal')
    if n_c == 1:
        # imshow_kwargs['cmap'] = 'gray'
        imshow_kwargs['cmap'] = 'coolwarm'
        imshow_kwargs['vmin'] = 0
        imshow_kwargs['vmax'] = 1

    # Plot sta
    if ax is None:
        fig = plt.figure(figsize=(n_lags*aspect, n_rows))
        ax = fig.subplots(1, 1)
    else:
        fig = ax.figure
    for iR in range(n_rows):
        if not share_scale:
            scale = 1 / (np.max(np.abs(sta[iR])) * 2)

        for iL in range(n_lags):
            x0, x1 = iL*aspect, (iL+1)*aspect
            y0, y1 = -iR-1, -iR
            ax.imshow(sta[iR,iL].transpose(1,2,0) * scale + .5, 
                       extent=[x0, x1, y0, y1], 
                       **imshow_kwargs)
            ax.plot([x0, x1, x1, x0, x0], [y1, y1, y0, y0, y1], 'k-')
        
        ax.set_ylim([-n_rows-.1, .1])
        ax.set_xlim([-.1, n_lags*aspect+.1])
        # turn off 
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_aspect('equal')

        ax.set_yticks([-iR-.5 for iR in range(n_rows)])
        if row_labels is None: 
            row_labels = [f'{iR}' for iR in range(n_rows)]
        ax.set_yticklabels(row_labels)
        
        ax.set_xticks([(iL+.5)*aspect for iL in range(n_lags)])
        if col_labels is None:
            col_labels = [f'{iL}' for iL in range(n_lags)]
        ax.set_xticklabels(col_labels)
    return fig, ax

def thresholded_centroid(img, threshold=.5):
    """
    Calculate the centroid of an image after thresholding.
    
    This function computes the center of mass of an image after normalizing it
    to the range [0,1] and setting values below a threshold to zero. This is
    useful for finding the center of a receptive field while ignoring noise.
    
    Parameters:
    -----------
    img : ndarray
        2D image array
    threshold : float, default=0.5
        Values below this threshold (after normalization) will be set to zero
        
    Returns:
    --------
    tuple
        (i_centroid, j_centroid) - row and column coordinates of the centroid
    """
    # Normalize image to range [0,1]
    img = img - img.min()
    img = img / img.max()
    
    # Apply threshold to remove noise
    img[img < threshold] = 0
    
    # Get image dimensions
    n_i, n_j = img.shape
    
    # Calculate weighted average of row indices (i coordinate)
    i_centroid = (np.arange(n_i)[:,None] * img).sum() / img.sum()
    
    # Calculate weighted average of column indices (j coordinate)
    j_centroid = (np.arange(n_j)[None,:] * img).sum() / img.sum()
    
    return i_centroid, j_centroid