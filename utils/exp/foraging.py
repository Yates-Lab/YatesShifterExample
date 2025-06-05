import numpy as np
from ..general import nd_paste

def get_general_probe_params(trial_data, exp_settings):
    """
    Returns the probe parameters from a trial data structure.

    Parameters
    ----------
    trial_data : dict
        Trial data from the ExpStruct (ExpStruct['D'][iT])
    exp_settings : dict
        Experiment settings from the ExpStruct (ExpStruct['S'])

    Returns
    -------
    general_probe_params : dict
        Dictionary of general probe parameters
            n_probes (int): Number of probes
            transparent (float): Transparency of the probe
            radius (float): Probe radius
            vrange (float): Probe value range (pixel contrast)
            bkgnd (float): Background value
            phase (float): Probe phase
            cpd (float): Cycles per degree for the preferred orientation
            noncpd (float): Cycles per degree for the non-preferred orientation
            prefori (float): Preferred orientation (degrees)
            nonprefori (float): Non-preferred orientation (degrees)
    """
    pix_per_deg = float(exp_settings['pixPerDeg'])
    keys_to_params = {
        'orinum': ('n_probes', lambda x: int(x)), 
        'probecon' : ('transparent', lambda x: -float(x)),
        'proberadius': ('radius', lambda x: np.round(x * pix_per_deg)),
        'proberange' : ('vrange', lambda x: float(x)),
        'bkgd': ('bkgnd', lambda x: float(x)), # must be a float, not an int
        'phase': ('phase', lambda x: float(x)),
        'cpd': ('cpd', lambda x: float(x)),
        'noncpd': ('noncpd', lambda x: float(x)),
        'prefori': ('prefori', lambda x: float(x)),
        'nonprefori': ('nonprefori', lambda x: float(x)),
    }

    general_params = {}
    for key, (param_name, transform) in keys_to_params.items():
        if key in trial_data['P']:
            general_params[param_name] = transform(trial_data['P'][key])
        else:
            raise KeyError(f'Key "{key}" not found in trial data.')

    general_params['pix_per_deg'] = pix_per_deg

    return general_params

def get_probe_params(trial_data, exp_settings):
    """
    Expands the general probe parameters into a list of probe parameters.

    Parameters
    ----------
    trial_data : dict
        Trial data from the ExpStruct (ExpStruct['D'][iT])
    exp_settings : dict
        Experiment settings from the ExpStruct (ExpStruct['S'])

    Returns
    -------
    probe_params : list of dict
        List of probe parameters
            transparent (float): Transparency of the probe
            radius (float): Probe radius
            vrange (float): Probe value range (pixel contrast)
            bkgnd (float): Background value
            cpd (float): Cycles per degree for the preferred orientation
            ori (float): Orientation of the probe (degrees)
            phase (float): Probe phase
            pix_per_deg (float): Pixels per degree
    """
    gen_params = get_general_probe_params(trial_data, exp_settings)

    keep_keys = ['transparent', 'radius', 'vrange', 'bkgnd', 'pix_per_deg']
    probe_params = []
    n_probes = gen_params['n_probes']
    for iP in range(n_probes):
        params = {k: gen_params[k] for k in keep_keys}
        if iP == 0:
            params['ori'] = gen_params['prefori']
            params['cpd'] = gen_params['cpd']
            params['phase'] = gen_params['phase']
        elif iP < n_probes - 1:
            params['ori'] = gen_params['nonprefori']
            params['cpd'] = gen_params['noncpd']
            params['phase'] = gen_params['phase']
        elif iP == n_probes - 1:
            params['ori'] = 0
            params['cpd'] = 0
            params['phase'] = 0
        probe_params.append(params)
    return probe_params

def gen_probe_texture(
        radius, 
        cpd, 
        phase, 
        ori,
        transparent,
        vrange,
        bkgnd,
        pix_per_deg):
    """
    Generates a Gabor texture for a probe.

    Parameters
    ----------
    radius : float
        Radius of the probe in degrees
    cpd : float
        Cycles per degree for the preferred orientation
    phase : float
        Phase of the probe in degrees
    ori : float
        Orientation of the probe in degrees
    transparent : float
        Transparency of the probe
    vrange : float
        Value range (pixel contrast) of the probe
    bkgnd : float
        Background value of the probe
    pix_per_deg : float
        Pixels per degree

    Returns
    -------
    im : np.ndarray (r, c)
        Image of the probe (uint8)
    alpha : np.ndarray (r, c)
        Alpha channel of the probe (uint8)
    """

    rad_pix = np.floor(radius)
    X, Y = np.meshgrid(np.arange(-rad_pix, rad_pix+1), np.arange(-rad_pix, rad_pix+1))
    sigma = (2 * rad_pix + 1) / 8

    # Exponential envelope (e)
    e = np.exp(-.5*(X**2 + Y**2)/sigma**2)

    # Sinusoidal carrier (s)
    maxRadians = 2 * np.pi * cpd / pix_per_deg
    pha = phase * np.pi / 180
    s = np.cos( np.cos(ori*np.pi/180) * (maxRadians*Y) + \
                np.sin(ori*np.pi/180) * (maxRadians*X) + pha)

    # Gabor (g)
    g = s * e

    # transparency
    t = (255 * abs(transparent)) * e

    # convert to uint8
    im = (g * vrange + bkgnd + .5).astype(np.uint8)
    alpha = (t + .5).astype(np.uint8)
    return im, alpha

def place_probe_texture(
        im, 
        alpha, 
        position, 
        center_pix, 
        bkgnd, 
        pix_per_deg, 
        roi,
        binSize=1
    ):
    """
    Places a probe texture at a given position.
    roi format: (2, 2) array where roi[0] is (row_start, row_stop) and roi[1] is (col_start, col_stop)
    """
    assert im.ndim == 2, 'Image must be 2D.'
    assert im.shape[0] == im.shape[1], 'Image must be square.'
    assert np.all(im.shape == alpha.shape), 'Image and alpha must have the same dimensions.'

    radius = np.floor(im.shape[0] / 2)

    pos_x = center_pix[0] + np.round(position[0] * pix_per_deg)
    pos_y = center_pix[1] - np.round(position[1] * pix_per_deg)

    x0 = int(pos_x - radius)-1
    y0 = int(pos_y - radius)-1

    # Updated ROI indexing
    src_pos = (y0 - roi[0, 0], x0 - roi[1, 0])
    dest_shape = (roi[0, 1] - roi[0, 0], roi[1, 1] - roi[1, 0])
    
    if alpha.dtype == np.uint8 or np.max(alpha) > 1 + 1e-6:
        alpha = alpha / 255
    im_roi = nd_paste((im-bkgnd)*alpha, src_pos, dest_shape)
    alpha_roi = nd_paste(alpha, src_pos, dest_shape)

    im_roi = im_roi[::binSize, ::binSize]
    alpha_roi = alpha_roi[::binSize, ::binSize]

    return im_roi, alpha_roi

def gen_probe_image(
        position,
        center_pix,
        radius, 
        cpd, 
        phase, 
        ori,
        transparent,
        vrange,
        bkgnd,
        pix_per_deg,
        roi
    ):
    """
    Generates a probe image at a given position.
    """
    im, alpha = gen_probe_texture(
            radius, cpd, phase, ori, transparent, vrange, bkgnd, pix_per_deg)
    alpha = alpha / 255
    I_roi, alpha_roi = place_probe_texture(
            im, alpha, position, center_pix, bkgnd, pix_per_deg, roi)
    return I_roi, alpha_roi