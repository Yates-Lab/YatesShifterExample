import numpy as np
from collections import defaultdict
from PIL import Image
from ..general import nd_paste

def get_trial_protocols(exp):
    protocols = []
    fpn_lut = defaultdict(lambda: 'ForageProceduralNoise')
    fpn_lut[1] = 'ForageGrating'
    fpn_lut[3] = 'ForageCSDFlash'
    fpn_lut[4] = 'ForageGabor'
    fpn_lut[5] = 'ForageDots'
    fpn_lut[6] = 'ForageDriftingGrating'
    for iT in range(len(exp['D'])):
        pr_name = exp['D'][iT]['PR']['name']
        if pr_name == 'ForageProceduralNoise':
            noise_type = int(exp['D'][iT]['PR']['noisetype'])
            pr_name = fpn_lut[noise_type]

        protocols.append(pr_name)

    return protocols

def blend_images(I_dest, I_source, alpha=None):
    """
    Blends a destination image with a source (Gabor) image.

    Args:
        I_dest (numpy.ndarray): Destination image (background), MxN, uint8 or float, range [0, 255]
        I_source (numpy.ndarray): Source image (Gabor), MxN, uint8 or float, range [0, 255]
        alpha (float or numpy.ndarray, optional): Alpha value(s), either:
            - Scalar in [0, 1] for uniform transparency
            - MxN matrix in [0, 1] for per-pixel transparency
            Defaults to 'none'.

    Returns:
        numpy.ndarray: Blended image, MxN, uint8, range [0, 255]

    Raises:
        ValueError: If input dimensions don't match or if alpha is invalid
    """
    assert I_dest.ndim == 2, 'Destination image must be 2D.'
    assert I_source.ndim == 2, 'Source image must be 2D.'
    assert np.all(I_dest.shape == I_source.shape), 'Destination and source images must have the same dimensions.'

    # Convert images to float64 (double precision)
    I_dest = I_dest.astype(np.float64)
    I_source = I_source.astype(np.float64)
    
    # Normalize images to [0, 1]
    I_dest_norm = I_dest / 255.0
    I_source_norm = I_source / 255.0
    
    # Handle alpha
    if isinstance(alpha, (int, float)):
        assert 0 <= alpha and alpha <= 1, 'Alpha scalar must be in the range [0, 1].'
        alpha_map = np.full_like(I_dest_norm, alpha)
    elif isinstance(alpha, np.ndarray):
        assert np.all(alpha.shape == I_dest.shape), 'Alpha map must have the same dimensions as the images.'
        if alpha.dtype == np.uint8 or np.max(alpha) > 1 + 1e-6:
            alpha = alpha.astype(np.float64) / 255.0
        alpha_map = np.clip(alpha, 0, 1)    
    else:
        # Calculate alpha from source
        alpha_map = np.abs(I_source_norm - 0.5) >= (1/127)
        alpha_map = alpha_map.astype(np.float64)
    
    # Apply the blending equation:
    # Final = Source * Alpha + Destination * (1 - Alpha)
    I_final_norm = I_source_norm * alpha_map + I_dest_norm * (1 - alpha_map)
    
    # Clip values to [0, 1] to avoid overflow
    I_final_norm = np.clip(I_final_norm, 0, 1)
    
    # Rescale to [0, 255] and convert to uint8
    I_final = np.uint8(I_final_norm * 255 + .5)
    
    return I_final

def gen_gauss_image_texture(im, bkgnd, transparency = .5):
    """
    Generates a Gaussian windowed image texture. 
    """
    assert im.shape[0] == im.shape[1], 'Image must be square'
    d = im.shape[0]
    if im.ndim == 3: 
        im = np.mean(im, axis=2)
    xy = np.arange(1, d+1) - d/2
    X, Y = np.meshgrid(xy, xy)
    g = np.exp(-(X**2 + Y**2) / (2 * (d/6)**2))
    im = g * im + bkgnd * (1 - g)

    if transparency > 0:
        alpha = 255.0 * (g > 0.05)
    else: 
        alpha = 255.0 * g

    return im, alpha

def place_gauss_image_texture(
    im,
    alpha,
    position,
    radius,
    center_pix,
    bkgnd,
    pix_per_deg,
    roi,
    binSize=1,
    ): 
    """
    Places a Gaussian windowed image texture at a given position
    roi format: (2, 2) array where roi[0] is (row_start, row_stop) and roi[1] is (col_start, col_stop)

    Parameters
    ----------
    im : numpy.ndarray
        Image to place.
    alpha : numpy.ndarray
        Image transparency.
    position : numpy.ndarray
        Position of the image in degrees relative to the center.
    radius : float
        Radius of the image in degrees.
    center_pix : numpy.ndarray
        Center of the screen in pixels (col, row).
    bkgnd : float
        Background value.
    pix_per_deg : float
        Pixels per degree.
    roi : numpy.ndarray
        Region of interest.
    binSize : int, optional
        Binning size. The default is 1.
    """
    rad_pix = np.round(radius * pix_per_deg)

    new_dims = (int(rad_pix * 2),int(rad_pix * 2))

    im = Image.fromarray(im)
    im = im.resize(new_dims, resample=2) # 2 = bilinear
    im = np.array(im).astype(np.float64)

    alpha = Image.fromarray(alpha)
    alpha = alpha.resize(new_dims, resample=2) # 2 = bilinear
    alpha = np.array(alpha) / 255

    pos_x = center_pix[0] + np.round(position[0] * pix_per_deg)
    pos_y = center_pix[1] - np.round(position[1] * pix_per_deg)

    x0 = int(pos_x - rad_pix)-1
    y0 = int(pos_y - rad_pix)-1

    # Updated ROI indexing
    src_pos = (y0 - roi[0, 0], x0 - roi[1, 0])
    dest_shape = (roi[0, 1] - roi[0, 0], roi[1, 1] - roi[1, 0])

    im_roi = nd_paste((im-bkgnd)*alpha, src_pos, dest_shape)
    alpha_roi = nd_paste(alpha, src_pos, dest_shape)

    im_roi = im_roi[::binSize, ::binSize]
    alpha_roi = alpha_roi[::binSize, ::binSize]

    return im_roi, alpha_roi

def get_clock_functions(exp):
    '''
    Input: 
        exp dictionary (level that contains ['D'] list)
    Output:
        ptb2ephys: function that converts PTB time to ephys time
        vpx2ephys: function that converts Eyetracker time to ephys time
    '''

    from scipy.interpolate import interp1d

    # Synchronize the clocks
    ephys_clock = []
    ptb_clock = []
    vpx_clock = []
    for i in range(len(exp['D'])):
        keys = ['START_EPHYS', 'END_EPHYS', 'STARTCLOCKTIME', 'ENDCLOCKTIME', 'START_VPX', 'END_VPX']
        key_present = [key in exp['D'][i] for key in keys]

        if not (key_present[0] and key_present[1]):
            # skip if ephys clock is not present
            continue

        # append the ephys clock times
        ephys_clock.append(exp['D'][i][keys[0]])
        ephys_clock.append(exp['D'][i][keys[1]])

        # append the PTB clock times
        # default to the first and last eye data times if not present
        if key_present[2] and key_present[3]:
            ptb_clock.append(exp['D'][i][keys[2]])
            ptb_clock.append(exp['D'][i][keys[3]])
        else:
            ptb_clock.append(exp['D'][i]['eyeData'][0,5])
            ptb_clock.append(exp['D'][i]['eyeData'][-1,5])

        # append the vpx clock times
        # default to the first and last eye data times if not present
        if  key_present[4] and key_present[5]:
            vpx_clock.append(exp['D'][i][keys[4]])
            vpx_clock.append(exp['D'][i][keys[5]])
        else:
            vpx_clock.append(exp['D'][i]['eyeData'][0,5])
            vpx_clock.append(exp['D'][i]['eyeData'][-1,5])

    ephys_clock = np.array(ephys_clock)
    ptb_clock = np.array(ptb_clock)
    vpx_clock = np.array(vpx_clock)

    nan_mask = np.isnan(ephys_clock) | np.isnan(ptb_clock) | np.isnan(vpx_clock)
    ephys_clock = ephys_clock[~nan_mask]
    ptb_clock = ptb_clock[~nan_mask]
    vpx_clock = vpx_clock[~nan_mask]

    ptb2ephys = interp1d(ptb_clock, ephys_clock, kind='linear', fill_value='extrapolate')
    vpx2ephys = interp1d(vpx_clock, ephys_clock, kind='linear', fill_value='extrapolate')
    return ptb2ephys, vpx2ephys
