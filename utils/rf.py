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


#%%
import matplotlib.pyplot as plt
import torch
import os
import numpy as np
import os
import pickle
from LogGabor import LogGaborFit
from skimage import measure

#%%

def get_mask_from_contour(Im, contour):
    import scipy.ndimage as ndimage    
    # Create an empty image to store the masked array
    r_mask = np.zeros_like(Im, dtype='bool')
    # Create a contour image by using the contour coordinates rounded to their nearest integer value
    r_mask[np.round(contour[:, 0]).astype('int'), np.round(contour[:, 1]).astype('int')] = 1
    # Fill in the hole created by the contour boundary
    r_mask = ndimage.binary_fill_holes(r_mask)
    return r_mask

def get_contour(Im, thresh):
    # use skimage to find contours at a threhsold,
    # select the largest contour and return the area and center of mass

    # find contours in Im at threshold thresh
    contours = measure.find_contours(Im, thresh)
    # Select the largest contiguous contour
    contour = sorted(contours, key=lambda x: len(x))[-1]
    
    r_mask = get_mask_from_contour(Im, contour)
    # plt.imshow(r_mask, interpolation='nearest', cmap=plt.cm.gray)
    # plt.show()

    M = measure.moments(r_mask, 1)

    area = M[0, 0]
    center = np.asarray([M[1, 0] / M[0, 0], M[0, 1] / M[0, 0]])
    
    return contour, area, center


# def get_countor_metrics(sta, 
#                         ste, 
#                         cells_in_group_ids, 
#                         lags, lags_tolerance = 1, 
#                         show = True, 
#                         save_fig_path = None):
#     snr_list = []

#     if cells_in_group_ids is None:
#         cells_in_group_ids = range(len(sta))

#     # Compute SNR and contours for all cells
#     for idx, cell_id in enumerate(cells_in_group_ids):
#         # Get STA and STE for this cell
#         sta_ = sta[cell_id].squeeze()
#         ste_ = ste[cell_id].squeeze()

#         # Find the time component of maximum absolute value in STA
#         hcom, wcom, tcom = torch.where(sta_.abs() == sta_.abs().max())
#         tcom = tcom[0].item()

#         if abs(tcom - lags.index(0)) > lags_tolerance:
#             tcom = lags.index(0)


#         # Extract spatial component from STE at tcom
#         wspace = ste_[:, :, tcom]

#         # Normalize the spatial component
#         I = (wspace - wspace.min()) / (wspace.max() - wspace.min())
#         I_numpy = I.numpy()

#         # Compute contour
#         contour, area_, center = get_contour(I_numpy, 0.5)

#         # Get mask from contour
#         mask = get_mask_from_contour(I_numpy, contour)

#         # Compute SNR
#         inside_mask = mask * I_numpy
#         outside_mask = (1 - mask) * I_numpy

#         # Avoid division by zero in case std of outside mask is zero
#         outside_std = np.std(outside_mask[outside_mask != 0])
#         if outside_std == 0:
#             snr_value = 0
#         else:
#             snr_value = np.mean(inside_mask[mask == 1]) / outside_std

#         # Collect SNR, cell index, time component, and contour
#         snr_list.append((snr_value, cell_id, tcom, contour, area_))

#     # Sort cells based on the computed SNR
#     snr_sorted = sorted(snr_list, key=lambda x: x[-1], reverse=False)  # Sort in descending order

#     NC = len(cells_in_group_ids)
#     n_total_images = 2 * NC  # Each cell has STE and STA images

#     # Compute number of columns and rows for a square-like figure
#     n_cols = int(np.ceil(np.sqrt(n_total_images)))
#     if n_cols % 2 != 0:
#         n_cols += 1  # Ensure n_cols is even since each cell has two images
#     n_rows = int(np.ceil(n_total_images / n_cols))

#     # Open figure with appropriate size
#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
#     axes = axes.flatten()

#     for idx, (snr_value, cell_id, tcom, contour, area_) in enumerate(snr_sorted):
#         ste_ = ste[cell_id].squeeze()
#         sta_ = sta[cell_id].squeeze()

#         wspace_ste = ste_[:, :, tcom]
#         wspace_sta = sta_[:, :, tcom]

#         # Get individual vmax and vmin for STE and STA
#         vmax_ste = wspace_ste.abs().max().item()
#         vmin_ste = -vmax_ste

#         vmax_sta = wspace_sta.abs().max().item()
#         vmin_sta = -vmax_sta

#         # Plot STE image with contour
#         ax_ste = axes[2 * idx]
#         im_ste = ax_ste.imshow(
#             wspace_ste.detach().cpu().numpy(),
#             cmap='coolwarm',
#             vmin=vmin_ste,
#             vmax=vmax_ste,
#             interpolation='none',
#         )

#         # Overlay contour on STE image
#         if contour is not None and len(contour) > 0:
#             contour_x = contour[:, 1]
#             contour_y = contour[:, 0]
#             ax_ste.plot(contour_x, contour_y, 'r', linewidth=1)

#         ax_ste.axis('off')
#         ax_ste.set_title(f'SNR: {snr_value:.2f}')

#         # Plot STA image
#         ax_sta = axes[2 * idx + 1]
#         im_sta = ax_sta.imshow(
#             wspace_sta.detach().cpu().numpy(),
#             cmap='coolwarm',
#             vmin=vmin_sta,
#             vmax=vmax_sta,
#             interpolation='none',
#         )

#         ax_sta.axis('off')
#         ax_sta.set_title(f'Cell ID: {cell_id}')

#     # Turn off any remaining empty subplots
#     for ax in axes[2 * NC:]:
#         ax.axis('off')

#     plt.tight_layout()
#     if save_fig_path is not None:
#         plt.savefig(save_fig_path)
        


#     plt.show()
#     plt.close('all')

#     cell_id_to_countor_metrics = {cell_id: {'snr_value':snr_value, 'tcom':tcom, 'contour':contour, 'area_':area_} for snr_value, cell_id, tcom, contour, area_ in snr_sorted}
#     if save_fig_path is not None:
#         folder_dir = os.path.dirname(save_fig_path)
#         folder = os.path.join(folder_dir, 'tensors')
#         os.makedirs(folder, exist_ok = True)
#         file_name = os.path.basename(save_fig_path)
#         with open(os.path.join(folder, file_name.replace('.png', '.pkl')), 'wb') as f:
#             pickle.dump(cell_id_to_countor_metrics, f)
#     return cell_id_to_countor_metrics

def get_countor_metrics(sta, ste, 
                        cells_in_group_ids=None, 
                        show=True, 
                        save_fig_path=None,
                        sort = False):
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import pickle

    snr_list = []
    
    if cells_in_group_ids is None:
        cells_in_group_ids = range(len(sta))
    assert type(sta) == np.ndarray and type(ste) == np.ndarray
    
    # Compute SNR and contours for all cells
    for cell_id in cells_in_group_ids:
        # Get STA and STE for this cell
        sta_ = sta[cell_id].squeeze()
        ste_ = ste[cell_id].squeeze()
    
        # Use the full STE image as the spatial component
        wspace = ste_
    
        # Normalize for contour computation to [0, 1]
        ptp_wspace = np.ptp(wspace)
        if ptp_wspace != 0:
            I_for_contour = (wspace - np.min(wspace)) / ptp_wspace
        else:
            I_for_contour = wspace
    
        # Compute contour (assume get_contour returns contour, area, and center)
        contour, area_, center = get_contour(I_for_contour, 0.5)
    
        # Get mask from contour
        mask = get_mask_from_contour(I_for_contour, contour)
    
        # Compute SNR using the mask on the [0,1] normalized image
        inside_mask = mask * I_for_contour
        outside_mask = (1 - mask) * I_for_contour
    
        # Avoid division by zero
        outside_std = np.std(outside_mask[outside_mask != 0])
        if outside_std == 0:
            snr_value = 0
        else:
            snr_value = np.mean(inside_mask[mask == 1]) / outside_std
    
        # Append computed metrics for this cell
        snr_list.append((snr_value, cell_id, contour, area_))
    
    # Sort cells based on the computed area (last tuple element)
    if sort:
        snr_sorted = sorted(snr_list, key=lambda x: x[-1], reverse=False)
    else:
        snr_sorted = snr_list
    
    NC = len(cells_in_group_ids)
    n_total_images = 2 * NC  # Two images per cell: one for STE and one for STA
    
    # Compute layout: even number of columns and sufficient rows
    n_cols = int(np.ceil(np.sqrt(n_total_images)))
    if n_cols % 2 != 0:
        n_cols += 1
    n_rows = int(np.ceil(n_total_images / n_cols))
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    axes = axes.flatten()
    
    # Use imshow style with colormap 'coolwarm' and normalization [-1, 1]
    for idx, (snr_value, cell_id, contour, area_) in enumerate(snr_sorted):
        sta_ = sta[cell_id].squeeze()
        ste_ = ste[cell_id].squeeze()
    
        # Normalize STA to [-1, 1]
        max_abs_sta = np.max(np.abs(sta_))
        sta_norm = sta_ / max_abs_sta if max_abs_sta != 0 else sta_
    
        # Normalize STE to [-1, 1]
        ptp_ste = np.ptp(ste_)
        ste_norm = 2 * (ste_ - np.min(ste_)) / ptp_ste - 1 if ptp_ste != 0 else ste_
    
        # Plot normalized STE with contour overlay
        ax_ste = axes[2 * idx]
        ax_ste.imshow(ste_norm, cmap='coolwarm', vmin=-1, vmax=1, interpolation='none')
        if contour is not None and len(contour) > 0:
            # contour is assumed to be in [row, col] format
            contour_x = contour[:, 1]
            contour_y = contour[:, 0]
            ax_ste.plot(contour_x, contour_y, 'r', linewidth=1)
        ax_ste.axis('off')
        ax_ste.set_title(f'SNR: {snr_value:.2f}', fontsize=8)
    
        # Plot normalized STA image
        ax_sta = axes[2 * idx + 1]
        ax_sta.imshow(sta_norm, cmap='coolwarm', vmin=-1, vmax=1, interpolation='none')
        ax_sta.axis('off')
        ax_sta.set_title(f'Cell ID: {cell_id}', fontsize=8)
    
    # Turn off any remaining empty subplots
    for ax in axes[2 * NC:]:
        ax.axis('off')
    
    plt.tight_layout()
    if save_fig_path is not None:
        plt.savefig(save_fig_path)
    
    if show:
        plt.show()
    plt.close('all')
    
    # Create dictionary mapping cell IDs to their contour metrics
    cell_id_to_countor_metrics = {
        cell_id: {'snr_value': snr_value, 'contour': contour, 'area_': area_}
        for snr_value, cell_id, contour, area_ in snr_sorted
    }
    
    # Optionally save the metrics to a pickle file in a "tensors" folder
    if save_fig_path is not None:
        folder_dir = os.path.dirname(save_fig_path)
        folder = os.path.join(folder_dir, 'tensors')
        os.makedirs(folder, exist_ok=True)
        file_name = os.path.basename(save_fig_path)
        with open(os.path.join(folder, file_name.replace('.png', '.pkl')), 'wb') as f:
            pickle.dump(cell_id_to_countor_metrics, f)
    
    return cell_id_to_countor_metrics


# %%
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy import ndimage
import torch
from matplotlib.patches import Ellipse

# Define a 2D Gaussian function with covariance matrix components
def gaussian_2d_cov(xy, amplitude, x0, y0, sigma_xx, sigma_xy, sigma_yy, offset):
    x, y = xy
    x_diff = x - x0
    y_diff = y - y0

    # Assemble the covariance matrix
    covariance_matrix = np.array([[sigma_xx, sigma_xy], 
                                  [sigma_xy, sigma_yy]])

    # Calculate the inverse of the covariance matrix
    cov_inv = np.linalg.inv(covariance_matrix)

    # The quadratic form
    form = (x_diff * cov_inv[0, 0] + y_diff * cov_inv[0, 1]) * x_diff + \
           (x_diff * cov_inv[1, 0] + y_diff * cov_inv[1, 1]) * y_diff

    # Gaussian function with offset
    g = amplitude * np.exp(-0.5 * form) + offset
    return g.ravel()

def plot_cov_ellipse(cov_matrix, pos, nstd=2, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()

    vals, vecs = np.linalg.eigh(cov_matrix)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    width, height = 2 * nstd * np.sqrt(vals)
    ellipse = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_patch(ellipse)
    return ellipse

def fit_guassian_to_cell_STE(cell_STE, 
                             show_plots=True, 
                             normalize_STE = False,
                             no_recursive_call = False):
    # Assuming data is loaded into 'cell_STE'
    # ex_cell_id = cid[3]
    # cell_STE = STE[ex_cell_id, 0, :, :, 0].cpu().numpy()
    assert cell_STE.ndim == 2 and cell_STE.shape[0] == cell_STE.shape[1]

    # Apply Gaussian filter and normalize the data
    def get_processed_ste(cell_STE):
        I = ndimage.gaussian_filter(cell_STE, 11)
        I = torch.tensor(I)
        I = (I - I.min()) / (I.max() - I.min())
        I = I.detach().cpu().numpy()
        cell_STE = cell_STE.copy()
        cell_STE_original = cell_STE.copy()
        if normalize_STE:
            cell_STE = (cell_STE - cell_STE.min()) / (cell_STE.max() - cell_STE.min())
            cell_STE[cell_STE <0.5] = 0
        return cell_STE, I, cell_STE_original
    
    cell_STE, I, cell_STE_original = get_processed_ste(cell_STE)
   
    max_y, max_x = np.unravel_index(I.argmax(), I.shape)

    dimension = cell_STE.shape[0]
    x = np.linspace(0, dimension - 1, dimension)
    y = np.linspace(0, dimension - 1, dimension)
    x, y = np.meshgrid(x, y)

    xy = np.vstack([x.ravel(), y.ravel()])
    data = cell_STE.ravel().astype('float64')
    

    initial_guess = [1, max_x, max_y, dimension/6, 1, dimension/6, 1]
    try:
        popt, pcov = curve_fit(gaussian_2d_cov, xy, data, p0=initial_guess)
    except:
        if not no_recursive_call:
            return fit_guassian_to_cell_STE(-cell_STE, 
                                            show_plots=show_plots, 
                                            normalize_STE=normalize_STE, 
                                            no_recursive_call=True)
        return None, None
        
    fitted_data = gaussian_2d_cov(xy, *popt).reshape(dimension, dimension)
   
    MSE = np.mean((cell_STE - fitted_data)**2)
    if show_plots:
        fig, axes = plt.subplots(1, 5, figsize=(18, 6))

        image0 = axes[0].imshow(cell_STE_original, cmap='coolwarm')
        axes[0].set_title("Original Data")
        plt.colorbar(image0, ax=axes[0])

        image1 = axes[1].imshow(cell_STE, cmap='coolwarm')
        axes[1].set_title("Thresholded Data")
        plt.colorbar(image1, ax=axes[1])
        

        image2 = axes[2].imshow(fitted_data, cmap='coolwarm')
        axes[2].set_title("Fitted Gaussian Model")
        plt.colorbar(image2, ax=axes[2])

        axes[3].imshow(cell_STE, cmap='coolwarm')
        axes[3].set_title("Ellipse Overlay")
        plot_cov_ellipse(np.array([[popt[3], popt[4]], [popt[4], popt[5]]]), (popt[1], popt[2]), nstd=2, ax=axes[3], edgecolor='red', facecolor='none', linewidth=2)
        plt.colorbar(image0, ax=axes[3])

        # print((cell_STE - fitted_data)**2)
        #show on log scale
        from matplotlib import colors
        image3 = axes[4].imshow((cell_STE - fitted_data ) ** 2, cmap='coolwarm')
        # axes[4].set_title("Squared Error: " + str(MSE))
        #print MSE in with 5 significant digits to title
        axes[4].set_title("Squared Error: " + "{:.3e}".format(MSE))
        plt.colorbar(image3, ax=axes[4])


        plt.show()
        # print("Fitted parameters (Amplitude, x0, y0, sigma_xx, sigma_xy, sigma_yy, offset):", popt)

    
    # print("Mean Squared Error:", MSE)

    return popt, MSE
    # except:
    #     return None, None
#%%
def get_gaussian_fit_metrics(sta, ste, cells_in_group_ids = None, show=True, save_fig_path=None, sort = False):
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import pickle

    mse_list = []
    if cells_in_group_ids is None:
        cells_in_group_ids = range(len(sta))


    # Loop over all cells and compute MSE using the full STE image
    for cell_id in cells_in_group_ids:
        # Get STA and STE for this cell (assumed 2D)
        sta_ = sta[cell_id].squeeze()
        ste_ = ste[cell_id].squeeze()

        # Fit Gaussian to the STE image and compute MSE
        popt, MSE = fit_guassian_to_cell_STE(ste_, normalize_STE=True, show_plots=False)

        # If fit was successful, record the metrics; otherwise, assign a large MSE
        if popt is not None:
            mse_list.append((MSE, cell_id, popt))
        else:
            mse_list.append((np.inf, cell_id, None))

    if sort:
        # Sort cells based on the computed MSE (lowest first)
        mse_sorted = sorted(mse_list, key=lambda x: x[0])
    else:
        mse_sorted = mse_list

    if show:
        NC = len(cells_in_group_ids)
        n_total_images = 2 * NC  # Each cell has two images: STE and STA

        # Compute layout for a roughly square grid; ensure even number of columns
        n_cols = int(np.ceil(np.sqrt(n_total_images)))
        if n_cols % 2 != 0:
            n_cols += 1
        n_rows = int(np.ceil(n_total_images / n_cols))

        # Create figure with subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
        axes = axes.flatten()

        for idx, (MSE, cell_id, popt) in enumerate(mse_sorted):
            # Get full STE and STA images
            sta_ = sta[cell_id].squeeze()
            ste_ = ste[cell_id].squeeze()

            # Normalize STE to [-1, 1] using min–max scaling
            ptp_ste = np.ptp(ste_)
            if ptp_ste != 0:
                ste_norm = 2 * (ste_ - np.min(ste_)) / ptp_ste - 1
            else:
                ste_norm = ste_

            # Normalize STA to [-1, 1] by dividing with maximum absolute value
            max_abs_sta = np.max(np.abs(sta_))
            if max_abs_sta != 0:
                sta_norm = sta_ / max_abs_sta
            else:
                sta_norm = sta_

            # Plot normalized STE image with ellipse overlay (if fit succeeded)
            ax_ste = axes[2 * idx]
            ax_ste.imshow(ste_norm, cmap='coolwarm', vmin=-1, vmax=1, interpolation='none')
            if popt is not None:
                amplitude, x0, y0, sigma_xx, sigma_xy, sigma_yy, offset = popt
                cov_matrix = np.array([[sigma_xx, sigma_xy],
                                       [sigma_xy, sigma_yy]])
                pos = (x0, y0)
                plot_cov_ellipse(cov_matrix, pos, nstd=2, ax=ax_ste,
                                 edgecolor='red', facecolor='none', linewidth=2)
            else:
                ax_ste.text(0.5, 0.5, 'Fit Failed', transform=ax_ste.transAxes,
                            color='red', fontsize=12, ha='center')
            ax_ste.axis('off')
            ax_ste.set_title(f'MSE: {MSE:.2e}', fontsize=8)

            # Plot normalized STA image
            ax_sta = axes[2 * idx + 1]
            ax_sta.imshow(sta_norm, cmap='coolwarm', vmin=-1, vmax=1, interpolation='none')
            ax_sta.axis('off')
            ax_sta.set_title(f'Cell ID: {cell_id}', fontsize=8)

        # Turn off any remaining empty subplots
        for ax in axes[2 * NC:]:
            ax.axis('off')

        plt.tight_layout()
        if save_fig_path is not None:
            #check if directory exists and make if needed
            folder_dir = os.path.dirname(save_fig_path)
            #if it does not exist, make it
            if not os.path.exists(folder_dir):
                os.makedirs(folder_dir)
            plt.savefig(save_fig_path)
        plt.show()
        plt.close('all')

    # Create a dictionary mapping cell IDs to their Gaussian fit metrics
    cell_id_to_gaussian_fit_metrics = {
        cell_id: {'MSE': MSE, 'popt': popt}
        for MSE, cell_id, popt in mse_sorted
    }

    if save_fig_path is not None:
        folder_dir = os.path.dirname(save_fig_path)
        folder = os.path.join(folder_dir, 'tensors')
        os.makedirs(folder, exist_ok=True)
        file_name = os.path.basename(save_fig_path)
        with open(os.path.join(folder, file_name.replace('.png', '.pkl')), 'wb') as f:
            pickle.dump(cell_id_to_gaussian_fit_metrics, f)

    return cell_id_to_gaussian_fit_metrics
#%%
def get_log_gabor_metrics(sta, ste, cells_in_group_ids=None, show=True, save_fig_path=None, sort = False):
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import pickle
    from tqdm import tqdm

    # Initialize the LogGaborFit object using the default parameters file
    _PARAM_FILE = 'https://raw.githubusercontent.com/bicv/LogGabor/master/default_param.py'
    lg = LogGaborFit(_PARAM_FILE)
    if cells_in_group_ids is None:
        cells_in_group_ids = range(len(sta))
    mse_list = []
    itr = tqdm(cells_in_group_ids) if __name__ == '__main__' else cells_in_group_ids

    # Loop over all cells and compute MSE using the full STA image
    for cell_id in itr:
        # Get STA for this cell (assumed to be a 2D image)
        sta_ = sta[cell_id].squeeze()
        # Convert to numpy array if necessary (e.g. if using torch.Tensor)
        if hasattr(sta_, 'detach'):
            sta_image = sta_.detach().cpu().numpy()
        else:
            sta_image = sta_
        
        #if sta_image is odd, pad it to make it even. pad by using border values
        if sta_image.shape[0] % 2 != 0:
            sta_image = np.pad(sta_image, ((0, 1), (0, 1)), mode='edge')

        # Set the size for the LogGabor fit based on the current STA image dimensions
        lg.set_size((sta_image.shape[1], sta_image.shape[0]))
        
        # Fit the Log-Gabor function and compute MSE
        
        try:
            w_fit, params = lg.LogGaborFit(sta_image)
            MSE = np.mean((sta_image - w_fit) ** 2)
        except Exception as e:
            MSE = np.inf
            w_fit = None
            print(f"Fit failed for cell {cell_id}: {e}")
        
        mse_list.append((MSE, cell_id, w_fit, sta_image))
    
    if sort:
        # Sort cells based on the computed MSE (lowest first)
        mse_sorted = sorted(mse_list, key=lambda x: x[0])
    else:
        mse_sorted = mse_list
    
    if show:
        NC = len(cells_in_group_ids)
        n_total_images = 2 * NC  # Two images per cell: one for STA and one for the fitted image
        
        # Compute grid layout for subplots
        n_cols = int(np.ceil(np.sqrt(n_total_images)))
        if n_cols % 2 != 0:
            n_cols += 1
        n_rows = int(np.ceil(n_total_images / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
        axes = axes.flatten()
        
        for idx, (MSE, cell_id, w_fit, sta_image) in enumerate(mse_sorted):
            # Normalize STA image to the range [-1, 1]
            max_abs_sta = np.max(np.abs(sta_image))
            sta_norm = sta_image / max_abs_sta if max_abs_sta != 0 else sta_image
            
            # Normalize fitted image if available
            if w_fit is not None:
                max_abs_wfit = np.max(np.abs(w_fit))
                w_fit_norm = w_fit / max_abs_wfit if max_abs_wfit != 0 else w_fit
            else:
                w_fit_norm = None
            
            # Plot normalized STA image
            ax_sta = axes[2 * idx]
            ax_sta.imshow(sta_norm, cmap='coolwarm', vmin=-1, vmax=1, interpolation='none')
            ax_sta.axis('off')
            ax_sta.set_title(f'Cell ID: {cell_id}', fontsize=8)
            
            # Plot normalized fitted Log-Gabor image (or show a failure message)
            ax_wfit = axes[2 * idx + 1]
            if w_fit_norm is not None:
                ax_wfit.imshow(w_fit_norm, cmap='coolwarm', vmin=-1, vmax=1, interpolation='none')
                ax_wfit.set_title(f'MSE: {MSE:.2e}', fontsize=8)
            else:
                ax_wfit.text(0.5, 0.5, 'Fit Failed', transform=ax_wfit.transAxes,
                             color='red', fontsize=12, ha='center')
                ax_wfit.set_title('Fit Failed', fontsize=8)
            ax_wfit.axis('off')
        
        # Turn off any extra subplots
        for ax in axes[2 * NC:]:
            ax.axis('off')
        
        plt.tight_layout()
        if save_fig_path is not None:
            #check if directory exists and make if needed
            folder_dir = os.path.dirname(save_fig_path)
            #if it does not exist, make it
            if not os.path.exists(folder_dir):
                os.makedirs(folder_dir)
            plt.savefig(save_fig_path)
        plt.show()
        plt.close('all')
    
    # Create and optionally save a dictionary mapping cell IDs to their Log-Gabor metrics
    cell_id_to_log_gabor_metrics = {
        cell_id: {'MSE': MSE, 'w_fit': w_fit}
        for MSE, cell_id, w_fit, _ in mse_sorted
    }
    
    if save_fig_path is not None:
        folder_dir = os.path.dirname(save_fig_path)
        folder = os.path.join(folder_dir, 'tensors')
        os.makedirs(folder, exist_ok=True)
        file_name = os.path.basename(save_fig_path)
        with open(os.path.join(folder, file_name.replace('.png', '.pkl')), 'wb') as f:
            pickle.dump(cell_id_to_log_gabor_metrics, f)
    
    return cell_id_to_log_gabor_metrics


# %%
def get_corr_metrics(sta, ste, cells_in_group_ids=None, corr_ste_or_sta=None,
                     show=True, save_fig_path_sta=None, save_fig_path_ste=None,
                     sort = False):
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import pickle
    
    if cells_in_group_ids is None:
        cells_in_group_ids = range(len(sta))

    # If no specific mode is provided, compute both 'sta' and 'ste' metrics and combine.
    if corr_ste_or_sta is None:
        sta_metrics = get_corr_metrics(sta, ste, cells_in_group_ids,
                                       corr_ste_or_sta='sta',
                                       show=show,
                                       save_fig_path_sta=save_fig_path_sta,
                                       save_fig_path_ste=save_fig_path_ste)
        ste_metrics = get_corr_metrics(sta, ste, cells_in_group_ids,
                                       corr_ste_or_sta='ste',
                                       show=show,
                                       save_fig_path_sta=save_fig_path_sta,
                                       save_fig_path_ste=save_fig_path_ste)
        # Combine the two dictionaries for each cell.
        for cell_id in cells_in_group_ids:
            sta_metrics[cell_id].update(ste_metrics[cell_id])
        if save_fig_path_sta is not None:
            folder_dir = os.path.dirname(save_fig_path_sta)
            folder = os.path.join(folder_dir, 'tensors')
            os.makedirs(folder, exist_ok=True)
            file_name = os.path.basename(save_fig_path_sta)
            with open(os.path.join(folder, file_name.replace('.png', '.pkl')), 'wb') as f:
                pickle.dump(sta_metrics, f)
        return sta_metrics

    # Main code: compute correlation metric for each cell.
    # Here, the "corr_value" is defined as the maximum absolute value of the chosen image.
    corr_list = []
    for cell_id in cells_in_group_ids:
        sta_ = sta[cell_id].squeeze()  # assumed to be a 2D image
        ste_ = ste[cell_id].squeeze()  # assumed to be a 2D image

        if corr_ste_or_sta == 'sta':
            corr_value = np.max(np.abs(sta_))
        elif corr_ste_or_sta == 'ste':
            corr_value = np.max(np.abs(ste_))
        else:
            raise ValueError("corr_ste_or_sta must be either 'sta' or 'ste'")

        # Append the computed metric along with the cell ID.
        corr_list.append((corr_value, cell_id))

    if sort:
        # Sort cells in descending order based on the computed correlation metric.
        corr_sorted = sorted(corr_list, key=lambda x: x[0], reverse=True)
    else:
        corr_sorted = corr_list

    if show:
        NC = len(cells_in_group_ids)
        n_total_images = 2 * NC  # Each cell will be represented by two images: one for STE and one for STA

        # Compute grid layout for a roughly square figure.
        n_cols = int(np.ceil(np.sqrt(n_total_images)))
        if n_cols % 2 != 0:
            n_cols += 1  # Ensure an even number of columns.
        n_rows = int(np.ceil(n_total_images / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
        axes = axes.flatten()

        for idx, (corr_value, cell_id) in enumerate(corr_sorted):
            # Retrieve the full 2D images.
            sta_ = sta[cell_id].squeeze()
            ste_ = ste[cell_id].squeeze()

            # Normalize STA to [-1, 1] by dividing by its maximum absolute value.
            max_abs_sta = np.max(np.abs(sta_))
            sta_norm = sta_ / max_abs_sta if max_abs_sta != 0 else sta_

            # Normalize STE to [-1, 1] using min–max scaling.
            ptp_ste = np.ptp(ste_)
            ste_norm = 2 * (ste_ - np.min(ste_)) / ptp_ste - 1 if ptp_ste != 0 else ste_

            # Plot the normalized STE image.
            ax_ste = axes[2 * idx]
            ax_ste.imshow(ste_norm, cmap='coolwarm', vmin=-1, vmax=1, interpolation='none')
            ax_ste.axis('off')
            if corr_ste_or_sta == 'sta':
                ax_ste.set_title(f'STA Corr: {corr_value:.3f}', fontsize=16)
            elif corr_ste_or_sta == 'ste':
                ax_ste.set_title(f'STE Corr: {corr_value:.3f}', fontsize=16)

            # Plot the normalized STA image.
            ax_sta = axes[2 * idx + 1]
            ax_sta.imshow(sta_norm, cmap='coolwarm', vmin=-1, vmax=1, interpolation='none')
            ax_sta.axis('off')
            ax_sta.set_title(f'Cell ID: {cell_id}', fontsize=16)

        # Turn off any remaining unused subplots.
        for ax in axes[2 * NC:]:
            ax.axis('off')

        plt.tight_layout()
        if save_fig_path_ste is not None and corr_ste_or_sta == 'ste':
            #check if directory exists and make if needed
            folder_dir = os.path.dirname(save_fig_path_ste)
            #if it does not exist, make it
            if not os.path.exists(folder_dir):
                os.makedirs(folder_dir)
            plt.savefig(save_fig_path_ste)
        elif save_fig_path_sta is not None and corr_ste_or_sta == 'sta':
            #check if directory exists and make if needed
            folder_dir = os.path.dirname(save_fig_path_ste)
            #if it does not exist, make it
            if not os.path.exists(folder_dir):
                os.makedirs(folder_dir)
            plt.savefig(save_fig_path_sta)
        plt.show()
        plt.close('all')

    # Create a dictionary mapping each cell ID to its computed metric.
    cell_id_to_corr_metrics = {
        cell_id: {f'{corr_ste_or_sta}_corr_value': corr_value}
        for corr_value, cell_id in corr_sorted
    }
    return cell_id_to_corr_metrics

def calc_dset_sta(dset, inds, lags, modifier=lambda x: x, batch_size=2048, device='cpu', verbose = 0):
    """
    Calculate the spike-triggered average (STA) for a dataset.
    
    This function computes the STA by correlating neural responses with visual stimuli
    across specified time lags.
    
    Parameters:
    -----------
    dset : DictDataset
        Dataset containing stimuli and neural responses
    inds : torch.Tensor
        Indices of valid samples to use
    lags : int or array-like
        Number of time lags or specific lags to use
    modifier : callable, default=lambda x: x
        Function to modify stimulus before correlation (e.g., squaring for energy)
    batch_size : int, default=2048
        Batch size for processing
    device : str, default='cpu'
        Device to perform computation on ('cpu' or 'cuda:X')
    verbose : int, default=0
        Verbosity level (0=silent, 1=progress bar)
        
    Returns:
    --------
    sta : torch.Tensor
        Spike-triggered average with shape (n_units, n_lags, n_y, n_x)
    """
    # Convert scalar lag value to range if needed
    if np.isscalar(lags):
        lags = np.arange(lags)

    # Define which keys to include and their corresponding lags
    keys_lags = {
        'robs': 0,        # Neural responses at current time
        'stim': lags,     # Stimulus at specified lags
    }

    # Include data filtering status if available
    if 'dfs' in dset:
        keys_lags['dfs'] = 0
    
    # Calculate mean stimulus value for centering
    mu = dset['stim'][inds].mean()

    # Get total spike count for normalization
    n_spikes = dset['robs'][inds].sum()
    # Get stimulus dimensions
    n_y, n_x = dset['stim'].shape[1:3]
    # Get number of neural units
    n_units = dset['robs'].shape[1]
    # Create dataset with embedded time lags

    ce_dset = CombinedEmbeddedDataset(dset, inds, keys_lags)

    # Initialize STA tensor on specified device
    sta = torch.zeros((n_units, len(lags), n_y, n_x), device=device)
    # Create batch iterator
    it = range(0, len(ce_dset), batch_size)
    # Add progress bar if verbose
    if verbose:
        it = tqdm(it, desc='Calculating STA')
    
    # Process data in batches
    for i in it:
        # Get current batch
        batch = ce_dset[i:i+batch_size]
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        # Extract neural responses
        robs = batch['robs']
        # Center and modify stimulus (e.g., for STE calculation)
        stim = modifier(batch['stim'] - mu)
        
        # Compute correlation between responses and stimuli
        # tc: time x channel, tlyx: time x lag x y x x, clyx: channel x lag x y x x
        sta += torch.einsum('tc, tlyx->clyx', robs, stim) / n_spikes
    
    # Move result back to CPU
    sta = sta.to('cpu')
    return sta

def calc_dset_stc(dset, inds, lags, roi = None, sta=None, batch_size=2048, device='cpu', verbose = 0):
    """
    Calculate the spike-triggered covariance (STC) for a dataset.
    
    This function computes the STC matrix for neural responses to visual stimuli,
    which helps identify relevant stimulus dimensions beyond the STA.
    
    Parameters:
    -----------
    dset : DictDataset
        Dataset containing stimuli and neural responses
    inds : torch.Tensor
        Indices of valid samples to use
    lags : int or array-like
        Number of time lags or specific lags to use
    roi : list, optional
        Region of interest as [[y_start, y_end], [x_start, x_end]]
    sta : torch.Tensor, optional
        Pre-computed spike-triggered average
    batch_size : int, default=2048
        Batch size for processing
    device : str, default='cpu'
        Device to perform computation on ('cpu' or 'cuda:X')
    verbose : int, default=0
        Verbosity level (0=silent, 1=progress bar)
        
    Returns:
    --------
    V : torch.Tensor
        Eigenvectors (filters) of the STC matrix
    L : torch.Tensor
        Eigenvalues of the STC matrix
    cov : torch.Tensor
        The full covariance matrix
    """
    
    if np.isscalar(lags):
        lags = np.arange(lags)

    keys_lags = {
        'robs': 0,
        'stim': lags,
    }

    mu = dset['stim'][inds].mean()

    if 'dfs' in dset:
        keys_lags['dfs'] = 0

    n_units = dset['robs'].shape[1]
    n_lags = len(lags)
    if roi is None:
        n_y, n_x = dset['stim'].shape[1:3]
        roi = [[0, n_y], [0, n_x]]
    else:
        n_y = roi[0][1] - roi[0][0]
        n_x = roi[1][1] - roi[1][0]

    if sta is None:
        sta = calc_dset_sta(dset, inds, lags, device=device, verbose=verbose)
    sta = sta[...,roi[0][0]:roi[0][1],roi[1][0]:roi[1][1]].to(device)
    sta = sta.flatten(start_dim=1)

    # Get total spike count for normalization
    n_spikes = dset['robs'][inds].sum().to(device)

    from .data import CombinedEmbeddedDataset # prevents circular import
    ce_dset = CombinedEmbeddedDataset(dset, inds, keys_lags)
    n_dims = n_lags * n_y * n_x
    cov = torch.zeros((n_units, n_dims, n_dims), device=device)
    it = range(0, len(ce_dset), batch_size)
    if verbose:
        it = tqdm(it, desc=f'Calculating STC of dim {n_dims}')
    for i in it:
        batch = ce_dset[i:i+batch_size]
        batch = {k: v.to(device) for k, v in batch.items()}
        robs = batch['robs'] / n_spikes
        stim = batch['stim'][...,roi[0][0]:roi[0][1],roi[1][0]:roi[1][1]]
        stim = stim.flatten(start_dim=1) - mu
        for iU in range(n_units):
            # Compute stimulus deviation from STA, weighted by spike count
            s = (stim - sta[[iU]]) * torch.sqrt(robs[:,[iU]])
            cov[iU] += s.T @ s
        torch.cuda.empty_cache()

    # Eigendecomposition of covariance matrix
    L, V = torch.linalg.eigh(cov)
    L = L.to('cpu')
    L = L.flip(1)  # Sort eigenvalues in descending order
    V = V.to('cpu').reshape(n_units, n_lags, n_y, n_x, n_dims).permute(0, 4, 1, 2, 3)
    V = V.flip(1)  # Sort eigenvectors to match eigenvalues

    cov = cov.to('cpu')
    return V, L, cov

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