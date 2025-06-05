# This file contains functions to sample a stimulus at gaze positions and times given by the grid and shifts tensors.

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np


def transform_grid(grid, extents, inplace=False):
    '''
    Transforms a grid to be used as an input to torch.nn.functional.grid_sample.
    The returned grid extents map onto [-1, 1] for each dimension corresponding
    to the extents of the input tensor's dimensions (defined in the extents variable).

    Parameters:
    -----------
    grid: (N, H, W, 2) or (N, D, H, W, 3) tensor
    extents: list of (min, max) pairs for each dimension. 
        e.g. [(x_min, x_max), (y_min, y_max)] for a grid sampling into a 2D image
        or [(x_min, x_max), (y_min, y_max), (z_min, z_max)] for a grid sampling into a 3D volume
    inplace: bool, whether to transform the grid inplace (default: False)

    Returns:
    --------
    transformed_grid: (N, H, W, 2) or (N, D, H, W, 3) tensor
    '''

    assert len(grid.shape) in [4, 5], 'grid must be a 4D or 5D tensor'
    if len(grid.shape) == 4:
        assert grid.shape[-1] == 2, '4D grid must have 2 channels in the last dimension' 
        grid_dims = 2
    else:
        assert grid.shape[-1] == 3, '5D grid must have 3 channels in the last dimension'
        grid_dims = 3
    assert len(extents) == grid.shape[-1], 'extents must have the same length as the last dimension of grid'

    # transform grid to [-1, 1]
    if not inplace:
        grid = grid.clone()
    for i in range(grid_dims):
        extent = extents[grid_dims - i - 1]
        grid[...,i] = 2 * (grid[...,i] - extent[0]) / (extent[1] - extent[0]) - 1
    
    return grid

def grid_sample_coords(input:torch.tensor, grid:torch.tensor, extent:list, 
                       transform_inplace:bool = False, no_grad:bool = True,
                       mode:str='bilinear', padding_mode:str='zeros', align_corners:bool=False):
    '''
    Wrapper for F.grid_sample that takes a grid of coordinates and samples the input tensor at those coordinates

    Parameters:
    -----------
    input: (N, C, H, W) or (N, C, D, H, W) tensor
    grid: (N, H, W, 2) or (N, D, H, W, 3) tensor
    extent: list of (min, max) pairs for each dimension
    transform_inplace: bool, whether to transform the grid inplace (default: False)
    mode (str):
        interpolation mode 'bilinear' | 'nearest' | 'bicubic'. Default: 'bilinear'
    padding_mode (str): 
        padding mode for outside grid values 'zeros' | 'border' | 'reflection'. Default: 'zeros' 
    align_corners (bool, default False): 
        Geometrically, we consider the pixels of the input as squares rather than points.
        If set to True, the extrema (-1 and 1) are considered as referring to the center points of the input's corner pixels. 
        If set to False, they are instead considered as referring to the corner points of the input's corner pixels, making the sampling more resolution agnostic.
        This option is used in F.interpolate and F.grid_sample.

    Returns:
    --------
    sampled_input: (N, C, H, W) or (N, C, D, H, W) tensor
    '''
    assert grid.dtype == input.dtype, f'input and grid must have the same dtype. Got {input.dtype} (input) and {grid.dtype} (grid)'
    grid = transform_grid(grid, extent, transform_inplace)
    if no_grad:
        with torch.no_grad():
                return F.grid_sample(input, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
    else:
        return F.grid_sample(input, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)

def grid_sample_stim(stim, extents, grid, shifts, frames, mode='bilinear', batch_size=5000, verbose=0, 
                     sample_device='cpu', output_device='cpu'):
    '''
    Sample the stimulus at the gaze positions and times given by the grid and shifts tensors.
    The stimulus is sampled at the time points given by t_bins.

    Parameters:
    -----------
    stim: (n_frames, n_channels, n_y, n_x) tensor
    extents: list of [(x[0], x[-1]), (y[0], y[-1])]
    grid: (2 [x,y], n_y_out, n_x_out) tensor defining the position to sample the stimulus in degrees
    shifts: (n_frames_out, 2) tensor defining the gaze position for each frame in degrees
    frames: (n_frames_out,) tensor defining the frame number for each gaze position (-1 for no frame i.e. zero filled)
    batch_size: int, number of frames to sample at a time

    Returns:
    --------
    sampled_stim: (n_frames_out, n_channels, n_y_out, n_x_out) tensor
    '''
    assert len(stim.shape) == 4, "stim should have shape (n_frames, n_channels, n_y, n_x)"
    assert len(grid.shape) == 3 and grid.shape[0] == 2, "grid should have shape (2, n_y_out, n_x_out)"
    assert len(shifts.shape) == 2 and shifts.shape[1] == 2, "shifts should have shape (n_frames_out, 2)"
    assert len(frames.shape) == 1, "frames should be a 1D tensor"
    assert shifts.shape[0] == frames.shape[0], "shifts and frames dimensions do not match"
    assert len(extents) == 2, "extents should be a list of [(x_min, x_max), (y_min, y_max)]"
    assert len(extents[0]) == 2 and len(extents[1]) == 2, "extents should be a list of [(x_min, x_max), (y_min, y_max)]"

    if isinstance(stim, np.ndarray):
        stim = torch.from_numpy(stim).float()
    stims_tensor = stim.permute((1,0,2,3)).unsqueeze(0) # (1, n_channels, n_frames, n_y, n_x)

    if isinstance(grid, np.ndarray):
        grid = torch.from_numpy(grid)
    
    if isinstance(shifts, np.ndarray):
        shifts = torch.from_numpy(shifts)
    
    if isinstance(frames, np.ndarray):
        frames = torch.from_numpy(frames)
    
    if extents[0][0] > extents[0][1]:
        print(f'Warning: x extents are reversed from typical image format (min, max). Got {extents[0]}.')
    if extents[1][1] > extents[1][0]:
        print(f'Warning: y extents are reversed from typical image format (max, min). Got {extents[1]}.')
    
    n_frames_out, _ = shifts.shape
    _, n_y_out, n_x_out = grid.shape
    n_frames, n_channels_stim, n_y, n_x = stim.shape

    grid = grid.to(sample_device)
    sampled_stim = torch.zeros((n_frames_out, n_channels_stim, n_y_out, n_x_out), device=output_device)
    itr = range(0, n_frames_out, batch_size)
    if verbose:
        itr = tqdm(itr, 'Sampling Stimulus')
    for i in itr:
        bs = slice(i, i+batch_size)

        frames_batch = frames[bs]
        zero_frames = frames_batch == -1
        frames_batch = frames_batch[~zero_frames]
        first_frame = int(frames_batch.min())
        last_frame = int(frames_batch.max())
        stim_slice = slice(first_frame, last_frame+1)
        stim_batch = stims_tensor[:,:,stim_slice,...].to(sample_device)
        shifts_batch = shifts[bs][~zero_frames].to(sample_device)
        frame_extents = [first_frame, last_frame] if last_frame - first_frame > 0 else [first_frame - .1, last_frame + .1]
        batch_extents = [
            frame_extents, # z = frame number
            extents[1], # y = rows
            extents[0]  # x = cols
        ]
        # Construct grid of time points
        # (T, 1, H, W)
        image_grid = frames_batch.repeat(n_y_out,n_x_out,1,1).permute((3,2,0,1)).to(sample_device)

        # Construct grid of gaze shifts
        # (1, 2, H, W) + (T, 2, 1, 1) -> (T, 2, H, W)
        gaze_grid = grid[None,...] + shifts_batch[:, :, None, None]

        # Concatenate gaze and time grids to form the sampling grid
        # (1, 3, T, H, W)
        coord_grid = torch.cat((gaze_grid, image_grid), dim=1
                               ).permute((0,2,3,1)).unsqueeze(0).float()

        sample_batch = grid_sample_coords(
                stim_batch, 
                coord_grid, 
                batch_extents, 
                mode=mode, 
                padding_mode='zeros', 
                align_corners=True
                ).squeeze(dim=0).permute((1,0,2,3))
        
        sampled_stim[bs,...][~zero_frames] = sample_batch.to(output_device)
        torch.cuda.empty_cache()
    return sampled_stim

