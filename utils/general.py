import torch
import numpy as np
import os

def ensure_tensor(x, device=None, dtype=None):
    """
    Ensures that the input is a torch.Tensor. If it is a numpy array, it is converted to a tensor.
    If device is provided, the tensor is moved to the device.

    Parameters:
    ----------
    x : numpy.ndarray, torch.Tensor, int, float, list, or tuple
        The input array or tensor.
    device : torch.device
        The device to move the tensor to.
    dtype : torch.dtype
        The data type to convert the tensor to.

    Returns:
    -------
    torch.Tensor
        The input converted to a tensor.
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if isinstance(x, list):
        x = torch.tensor(x)
    if isinstance(x, int) or isinstance(x, float):
        x = torch.tensor([x])
    if device is not None:
        x = x.to(device)
    if dtype is not None:
        x = x.type(dtype)
    return x

def ensure_ndarray(x, dtype=None):
    """
    Ensures that the input is a numpy.ndarray. If it is a tensor, it is converted to a numpy array.

    Parameters:
    ----------
    x : numpy.ndarray, torch.Tensor, int, float, list, or tuple
        The input array or tensor.

    Returns:
    -------
    numpy.ndarray
        The input converted to a numpy array.
    """
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    if isinstance(x, int) or isinstance(x, float):
        x = [x]
    if isinstance(x, list) or isinstance(x, tuple):
        x = np.array(x)
    if dtype is not None:
        x = x.astype(dtype)
    return x
def set_seeds(seed:int):
    '''
    Set seeds for reproducibility
    '''
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def nd_cut(source_array, position, dest_shape, fill_value=0):
    """
    Extract a region of interest from an n-dimensional array. If the requested region extends
    beyond the source array bounds, the result is padded with fill_value.
    
    Args:
        source_array (np.ndarray): Source array to extract from
        position (tuple): Position in source array where to start extraction (coordinates of [0,0,...])
        dest_shape (tuple): Shape of the output array
        fill_value: Value to use for padding when outside source array bounds
        
    Returns:
        np.ndarray: Result array of shape dest_shape containing the extracted region
    """
    # Create output array filled with the fill value
    result = np.full(dest_shape, dtype=source_array.dtype, fill_value=fill_value)
    
    # Get number of dimensions
    ndim = len(dest_shape)
    
    # Convert inputs to numpy arrays for easier manipulation
    position = np.array(position)
    source_shape = np.array(source_array.shape)
    dest_shape = np.array(dest_shape)
    
    # Calculate effective source region
    source_start = np.maximum(np.zeros(ndim, dtype=int), position)
    source_end = np.minimum(source_shape, position + dest_shape)
    
    # Calculate effective target region
    target_start = np.maximum(np.zeros(ndim, dtype=int), -position)
    target_end = np.minimum(dest_shape, source_shape - position)
    
    # Create slicing tuples
    source_slices = tuple(slice(start, end) for start, end in zip(source_start, source_end))
    target_slices = tuple(slice(start, end) for start, end in zip(target_start, target_end))
    
    # Perform the extraction operation only if there's a valid region to extract
    if all(end > start for start, end in zip(target_start, target_end)):
        result[target_slices] = source_array[source_slices]
    
    return result

def nd_paste(source_array, position, dest_shape, fill_value=0):
    """
    Paste an n-dimensional array into an array of specified shape at given position.
    
    Args:
        source_array (np.ndarray): Source array to paste
        position (tuple): Position where to paste the array (coordinates of [0,0,...])
        dest_shape (tuple): Shape of the output array
        
    Returns:
        np.ndarray: Result array of shape dest_shape with source_array pasted at position
    """
    # Create output array filled with zeros
    result = np.full(dest_shape, dtype=source_array.dtype, fill_value=fill_value)
    
    # Get number of dimensions
    ndim = len(dest_shape)
    
    # Convert inputs to numpy arrays for easier manipulation
    position = np.array(position)
    source_shape = np.array(source_array.shape)
    dest_shape = np.array(dest_shape)
    
    # Calculate effective source region (handling negative positions)
    source_start = np.maximum(np.zeros(ndim, dtype=int), -position)
    source_end = np.minimum(source_shape, dest_shape - position)
    
    # Calculate effective target region
    target_start = np.maximum(np.zeros(ndim, dtype=int), position)
    target_end = np.minimum(dest_shape, position + source_shape)
    
    # Create slicing tuples
    source_slices = tuple(slice(start, end) for start, end in zip(source_start, source_end))
    target_slices = tuple(slice(start, end) for start, end in zip(target_start, target_end))
    
    # Perform the paste operation
    if all(end > start for start, end in zip(source_start, source_end)):
        result[target_slices] = source_array[source_slices]
    
    return result
