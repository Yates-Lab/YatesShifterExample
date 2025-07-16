import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def local_2d(kernel):
    """
    Computes a 2D spatial locality penalty.
    
    This function penalizes filters with broadly distributed energy
    by weighting filter values according to their spatial distance from each other.
    
    Args:
        kernel: Tensor of shape (..., n_y, n_x) representing filter kernels
        
    Returns:
        Scalar regularization term that increases with spatial spread of filter energy
    """
    device = kernel.device
    n_filts = int(np.prod(kernel.shape[:-2]))
    n_y, n_x = kernel.shape[-2:]

    # Create coordinate grid
    y = torch.arange(n_y, dtype=torch.float32, device=device)
    x = torch.arange(n_x, dtype=torch.float32, device=device)
    locations = torch.stack(
        torch.meshgrid(y,x, indexing='ij'),
        dim=-1).reshape(n_y*n_x, 2)
    # Compute squared distance matrix, normalized by grid size
    C = torch.cdist(locations, locations, p=2)**2 / (2 * (n_y + n_x)**2)

    reg = 0
    flat_kernel = kernel.reshape(n_filts, n_y*n_x)
    for i in range(n_filts):
        filt = torch.pow(flat_kernel[i],2)[:,None] # (n_y*n_x, 1)
        # Weight squared filter values by their spatial distances
        filt_reg = (filt.T @ C @ filt).squeeze()
        reg += filt_reg

    return reg

def locality_conv(kernel, dims, independent=True):
    """
    Computes a spatial locality penalty.
    
    This function penalizes filters with broadly distributed energy
    by weighting filter values according to their spatial distance from each other.

    Parameters:
    -----------
    kernel : torch.Tensor
        The kernel to regularize. Should be a 2D or 3D tensor.
    dims : int or list of int
        The dimensions to apply the locality penalty to. If None, applies to last 2 or 3 
        dimensions depending on kernel shape. Can be a single integer or a list.
    independent : bool
        Whether to apply the penalty independently to each filter in the batch. If False,
        applies the penalty to the mean of all filters.
    
    Returns:
    --------
    torch.Tensor
        The locality penalty.
    """
    # Handle default case
    if dims is None:
        if kernel.ndim >= 3:
            dims = [-2, -1]  # Last two dimensions (spatial)
        elif kernel.ndim == 2:
            dims = [0, 1]    # Both dimensions for 2D tensor
    
    # Convert scalar to list
    if np.isscalar(dims):
        dims = [dims]
    
    # Sort dimensions and ensure they're positive
    dims = sorted([d if d >= 0 else kernel.ndim + d for d in dims])
    
    # Calculate batch dimensions (all dimensions except those in dims)
    batch_dims = [i for i in range(kernel.ndim) if i not in dims]
    
    # Calculate batch shape and target shape
    batch_shape = [kernel.shape[i] for i in batch_dims]
    target_shape = [kernel.shape[i] for i in dims]
    
    # Flatten batch dimensions
    batch_size = np.prod(batch_shape)

    # Reshape kernel to (batch_size, 1, *target_shape)
    kernel_flat = torch.pow(kernel.permute(*(batch_dims + dims)).reshape(batch_size, 1, *target_shape), 2)
    if not independent:
        kernel_flat = kernel_flat.mean(dim=0, keepdim=True)
    
    if len(dims) == 1:
        # For 1D dimension, use a 1D Laplacian operator
        n_x = kernel_flat.shape[-1]
        x_d = torch.arange(1-n_x, n_x, dtype=torch.float32, device=kernel.device)
        d = torch.pow(x_d,2) / (2*n_x**2)
        conv = F.conv1d(kernel_flat, d[None,None,:], padding='same')
        return torch.sum(kernel_flat * conv)
        
    elif len(dims) == 2:
        # For 2D spatial dimensions, use a 2D Laplacian operator
        n_y, n_x = kernel_flat.shape[-2:]
        y_d, x_d = torch.meshgrid(torch.arange(1-n_y, n_y, dtype=torch.float32, device=kernel.device),
                                  torch.arange(1-n_x, n_x, dtype=torch.float32, device=kernel.device),
                                  indexing='ij')
        d = torch.pow(y_d,2) + torch.pow(x_d,2)
        d = d / (2*(n_y + n_x)**2)
        conv = F.conv2d(kernel_flat, d[None,None,:,:], padding='same')
        return torch.sum(kernel_flat * conv)
    elif len(dims) == 3:
        # For 3D dimensions, use a 3D Laplacian operator
        n_t, n_y, n_x = kernel_flat.shape[-3:]
        t_d, y_d, x_d = torch.meshgrid(torch.arange(1-n_t, n_t, dtype=torch.float32, device=kernel.device),
                                       torch.arange(1-n_y, n_y, dtype=torch.float32, device=kernel.device),
                                       torch.arange(1-n_x, n_x, dtype=torch.float32, device=kernel.device),
                                       indexing='ij')
        d = torch.pow(t_d,2) + torch.pow(y_d,2) + torch.pow(x_d,2)
        d = d / (2*(n_t + n_y + n_x)**2)
        conv = F.conv3d(kernel_flat, d[None,None,:,:,:], padding='same')
        return torch.sum(kernel_flat * conv)
    else:
        raise ValueError(f"Laplacian only implemented for 1, 2 or 3 dimensions, got {len(dims)}")


def l1(kernel):
    """L1 regularization for a kernel"""
    return torch.norm(kernel, p=1)

def laplacian(kernel, dims=None, padding_mode='reflect'):
    """
    Laplacian regularization for a kernel.
    
    This function applies a Laplacian operator to the kernel to encourage
    smoothness by penalizing high-frequency components. It works for 1D, 2D, and 3D kernels
    of arbitrary batch dimensions.
    
    Args:
        kernel: A tensor representing the kernel of arbitrary shape.
        dims: Dimensions to apply the Laplacian to. If None, applies to last 2 or 3 
              dimensions depending on kernel shape. Can be a single integer or a list.
        padding_mode: Padding mode for convolution, options are 'zeros', 'reflect', 'replicate', or 'circular'.
               
    Returns:
        The L2 norm of the Laplacian of the kernel, which can be used as a 
        regularization term in the loss function.
    """
    # Handle default case
    if dims is None:
        if kernel.ndim >= 3:
            dims = [-2, -1]  # Last two dimensions (spatial)
        elif kernel.ndim == 2:
            dims = [0, 1]    # Both dimensions for 2D tensor
    
    # Convert scalar to list
    if np.isscalar(dims):
        dims = [dims]
    
    # Sort dimensions and ensure they're positive
    dims = sorted([d if d >= 0 else kernel.ndim + d for d in dims])
    
    # Calculate batch dimensions (all dimensions except those in dims)
    batch_dims = [i for i in range(kernel.ndim) if i not in dims]
    
    # Calculate batch shape and target shape
    batch_shape = [kernel.shape[i] for i in batch_dims]
    target_shape = [kernel.shape[i] for i in dims]
    
    # Flatten batch dimensions
    batch_size = np.prod(batch_shape)
    
    # Reshape kernel to (batch_size, 1, *target_shape)
    kernel_flat = kernel.permute(*(batch_dims + dims)).reshape(batch_size, 1, *target_shape)
    
    if len(dims) == 1:
        # For 1D dimension, use a 1D Laplacian operator
        laplacian_op = torch.tensor([1, -2, 1], dtype=kernel.dtype, device=kernel.device).view(1, 1, 3)
        # Apply padding based on the specified mode
        padded_kernel = F.pad(kernel_flat, (1, 1), mode=padding_mode)
        conv = F.conv1d(padded_kernel, laplacian_op)
        
    elif len(dims) == 2:
        # For 2D spatial dimensions, use a 2D Laplacian operator
        laplacian_op = torch.tensor([[0, 1, 0],
                                     [1, -4, 1],
                                     [0, 1, 0]], dtype=kernel.dtype, device=kernel.device).view(1, 1, 3, 3)
        # Apply padding based on the specified mode
        padded_kernel = F.pad(kernel_flat, (1, 1, 1, 1), mode=padding_mode)
        conv = F.conv2d(padded_kernel, laplacian_op)
        
    elif len(dims) == 3:
        # For 3D dimensions, use a 3D Laplacian operator
        laplacian_op = (1/26) * torch.tensor([[[2, 3, 2],
                                              [3, 6, 3],
                                              [2, 3, 2]],
                                             [[3, 6, 3],
                                              [6, -88, 6],
                                              [3, 6, 3]],
                                             [[2, 3, 2],
                                              [3, 6, 3],
                                              [2, 3, 2]]], dtype=kernel.dtype, device=kernel.device).view(1, 1, 3, 3, 3)
        # Apply padding based on the specified mode
        padded_kernel = F.pad(kernel_flat, (1, 1, 1, 1, 1, 1), mode=padding_mode)
        conv = F.conv3d(padded_kernel, laplacian_op)
    else:
        raise ValueError(f"Laplacian only implemented for 1, 2 or 3 dimensions, got {len(dims)}")
    
    # Return the L2 norm of the Laplacian
    return torch.norm(conv, p=2)