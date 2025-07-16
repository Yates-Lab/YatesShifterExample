import numpy as np 
from ..general import ensure_ndarray, ensure_tensor
from .general import blend_images, gen_gauss_image_texture, place_gauss_image_texture
import torch
from tqdm import tqdm
from .foraging import gen_probe_texture, get_probe_params, place_probe_texture
from .support import get_face_library


def get_gaborium_params(trial_data):
    """
    Returns the parameters for a Gaborium trial.
    """

    hNoise_to_params = {
            'rng': ('seed', lambda x: int(x['Seed'])),
            'numGabors': ('n_gabors', int),
            'sfRange': ('sf_range', float),
            'minSF': ('min_sf', float),
            'pixPerDeg': ('pix_per_deg', float),
            'scaleRange': ('scale_range', float),
            'minScale': ('min_scale', float),
            'contrast': ('contrast', float),
            'radius': ('radius', float),
            'position': ('center', lambda x: x),
    }
    trial_hNoise = trial_data['PR']['hNoise']   
    params = {name: func(trial_hNoise[k]) for k, (name, func) in hNoise_to_params.items()}
    
    return params

def gen_gaborium_gabor_params(
        seed, # Exp.D{iTrial}.PR.hNoise.rng.Seed 
        n_gabors, # Exp.D{iTrial}.PR.hNoise.numGabors
        sf_range, # Exp.D{iTrial}.PR.hNoise.sfRange
        min_sf, # Exp.D{iTrial}.PR.hNoise.minSF
        pix_per_deg, # Exp.D{iTrial}.PR.hNoise.Exp.S.pixPerDeg
        scale_range, # Exp.D{iTrial}.PR.hNoise.scaleRange
        min_scale, # Exp.D{iTrial}.PR.hNoise.minScale
        contrast, # Exp.D{iTrial}.PR.hNoise.contrast
        radius, # Exp.D{iTrial}.PR.hNoise.radius
        center, # Exp.D{iTrial}.PR.hNoise.position 
    ):
    """
    Generates the parameters for a each gabor in a Gaborium trial.
    """
    if isinstance(seed, np.random.RandomState):
        rng = seed
    else:
        rng = np.random.RandomState(seed)

    pi = 3.141592654
    vals_1d = rng.rand(6 * n_gabors)
    vals = vals_1d.reshape(6, n_gabors, order='F')

    orientations = vals[0] * 2 * pi

    phases = (.5 - vals[1]) * pi

    frequencies = (vals[2] * sf_range + min_sf) / pix_per_deg * 2 * pi

    scales = (2 * (vals[3] * scale_range + min_scale) * pix_per_deg + 1) / 4

    e_multiplier = -.5 / scales ** 2

    contrasts = np.ones_like(orientations) * contrast

    offset = radius / 2
    x_gabors = vals[4] * radius + center[0] - offset
    y_gabors = vals[5] * radius + center[1] - offset

    params = {
        'orientations': orientations,
        'phases': phases,
        'frequencies': frequencies,
        'e_multiplier': e_multiplier,
        'contrasts': contrasts,
        'x_gabors': x_gabors,
        'y_gabors': y_gabors,
    }

    return params, rng

def gen_gaborium_image_from_params_numpy(
        orientations,
        phases,
        frequencies,
        e_multiplier,
        contrasts,
        x_gabors,
        y_gabors,
        roi, # (2, 2) array where roi[0] is (row_start, row_stop) and roi[1] is (col_start, col_stop)
        binSize=1,
    ):
    """
    Generates a Gaborium image from parameters using numpy (slow).
    """
    x = np.arange(roi[1, 0], roi[1, 1], binSize)
    y = np.arange(roi[0, 0], roi[0, 1], binSize)

    X, Y = np.meshgrid(x, y)
    xx = X.ravel()
    yy = Y.ravel()

    e_dist = 2/np.sqrt(-e_multiplier)
    m = np.logical_and.reduce((
        x_gabors + e_dist > x[0],
        x_gabors - e_dist < x[-1],
        y_gabors + e_dist > y[0],
        y_gabors - e_dist < y[-1]
    ))

    # Computation heavy part
    dx = xx[:,np.newaxis] - x_gabors[np.newaxis,m]
    dy = yy[:,np.newaxis] - y_gabors[np.newaxis,m]

    sin_coeff = np.sin(orientations[m]) * frequencies[m]
    cos_coeff = np.cos(orientations[m]) * frequencies[m]

    sv = np.cos(dx * sin_coeff + dy * cos_coeff + phases[np.newaxis,m])
    ev = np.exp(e_multiplier[np.newaxis,m] * (dx ** 2 + dy ** 2))

    I = np.sum(ev * sv * contrasts[np.newaxis,m], axis=1) * 127
    I = np.clip(I, -127, 127)
    I = I.reshape(len(y), len(x))

    return I

# Deprecated Numba implementation because numba does not support numpy 2.0
#from numba import njit, prange
# Helper function to compute the core calculations for each point
#@njit
#def _compute_gabor_point(x, y, x_gabors, y_gabors, orientations, frequencies, 
#                        phases, e_multiplier, contrasts):
#    """
#    Computes the intensity of a single pixel in a Gaborium image using Numba.
#    """
#    total = 0.0
#    for i in range(len(x_gabors)):
#        dx = x - x_gabors[i]
#        dy = y - y_gabors[i]
#        
#        # Compute sinusoidal component
#        sin_coeff = np.sin(orientations[i]) * frequencies[i]
#        cos_coeff = np.cos(orientations[i]) * frequencies[i]
#        sv = np.cos(dx * sin_coeff + dy * cos_coeff + phases[i])
#        
#        # Compute exponential envelope
#        ev = np.exp(e_multiplier[i] * (dx * dx + dy * dy))
#        
#        # Accumulate contribution
#        total += ev * sv * contrasts[i]
#    
#    return total * 127
#
#@njit(parallel=True)
#def gen_gaborium_image_from_params_numba(
#        orientations, 
#        phases,
#        frequencies,
#        e_multiplier,
#        contrasts,
#        x_gabors, 
#        y_gabors, 
#        roi, # (2, 2) array where roi[0] is (row_start, row_stop) and roi[1] is (col_start, col_stop)
#        binSize=1,
#    ):
#    """
#    Generates a Gaborium image from parameters using Numba (faster).
#    """
#    # Create coordinate grid
#    x = np.arange(roi[1, 0], roi[1, 1], binSize)
#    y = np.arange(roi[0, 0], roi[0, 1], binSize)
#    
#    # Filter gabors within ROI
#    e_dist = 2/np.sqrt(-e_multiplier)
#    mask = x_gabors + e_dist > x[0]
#    mask = np.logical_and(mask, x_gabors - e_dist < x[-1])
#    mask = np.logical_and(mask, y_gabors + e_dist > y[0])
#    mask = np.logical_and(mask, y_gabors - e_dist < y[-1])
#    
#    # Extract valid gabors
#    x_gabors = x_gabors[mask]
#    y_gabors = y_gabors[mask]
#    orientations = orientations[mask]
#    frequencies = frequencies[mask]
#    phases = phases[mask]
#    e_multiplier = e_multiplier[mask]
#    contrasts = contrasts[mask]
#    
#    # Initialize output array
#    result = np.zeros((len(y), len(x)))
#    
#    # Compute intensity for each pixel
#    for i in prange(len(y)):
#        for j in range(len(x)):
#            result[i, j] = _compute_gabor_point(
#                x[j], y[i], x_gabors, y_gabors, orientations, 
#                frequencies, phases, e_multiplier, contrasts
#            )
#    
#    # Clip values
#    return np.clip(result, -127, 127)

def gen_gaborium_image_from_params_gpu(
        orientations, 
        phases,
        frequencies,
        e_multiplier,
        contrasts,
        x_gabors, 
        y_gabors, 
        roi, # (2, 2) array where roi[0] is (row_start, row_stop) and roi[1] is (col_start, col_stop)
        binSize=1):
    """
    Generates a Gaborium image from parameters using GPU (fastest).
    """
    assert torch.cuda.is_available(), "CUDA is not available."

    device = torch.device('cuda')

    x_gabors = ensure_tensor(x_gabors, device)
    y_gabors = ensure_tensor(y_gabors, device)
    e_multiplier = ensure_tensor(e_multiplier, device)
    
    # Create coordinate tensors on GPU
    x = torch.arange(roi[1, 0], roi[1, 1], binSize, device=device)
    y = torch.arange(roi[0, 0], roi[0, 1], binSize, device=device)
    
    # Filter gabors within ROI (include a buffer for the exponential envelope)
    e_dist = 2/torch.sqrt(-e_multiplier)
    mask = (x_gabors + e_dist > x[0]) & (x_gabors - e_dist < x[-1]) & \
           (y_gabors + e_dist > y[0]) & (y_gabors - e_dist < y[-1])
    
    # Move filtered parameters to GPU
    X, Y = torch.meshgrid(x, y, indexing='xy')
    x_gabors = x_gabors[mask]
    y_gabors = y_gabors[mask]
    e_multiplier = e_multiplier[mask]
    orientations = ensure_tensor(orientations[mask], device)
    frequencies = ensure_tensor(frequencies[mask], device)
    phases = ensure_tensor(phases[mask], device)
    contrasts = ensure_tensor(contrasts[mask], device)
    
    n_gabors = len(x_gabors)
    with torch.no_grad(): 
        result = torch.zeros((len(y), len(x)), device=device)
        for iG in range(n_gabors):
            # Compute dx, dy for each gabor
            dx = X - x_gabors[iG]
            dy = Y - y_gabors[iG]
            
            # Compute sinusoidal component
            sin_coeff = torch.sin(orientations[iG]) * frequencies[iG]
            cos_coeff = torch.cos(orientations[iG]) * frequencies[iG]
            sv = torch.cos(dx * sin_coeff + dy * cos_coeff + phases[iG])
            
            # Compute exponential envelope
            ev = torch.exp(e_multiplier[iG] * (dx.pow(2) + dy.pow(2)))
            
            # Accumulate contribution
            result += ev * sv * contrasts[iG]

    result = torch.clamp(result*127, -127, 127)
    return result

def gen_gaborium_images(
        seed,
        n_gabors,
        sf_range,
        min_sf,
        pix_per_deg,
        scale_range,
        min_scale,
        contrast,
        radius,
        center,
        roi,
        binSize=1,
        frame_inds=0,
        method='numba',
        progress=False,
    ):
    """
    Generates Gaborium images from parameters.
    roi format: (n_frames, 2, 2) array where roi[:, 0] is (row_start, row_stop) and roi[:, 1] is (col_start, col_stop)
    """
    assert method in ['numpy', 'numba', 'gpu']

    frame_inds = ensure_ndarray(frame_inds, dtype=int)
    # Need to loop through requested frames sequentially 
    sort = np.argsort(frame_inds)
    frame_inds = frame_inds[sort]
    n_frames = len(frame_inds)

    roi = ensure_ndarray(roi, dtype=int)
    if roi.ndim == 2:
        roi = roi[np.newaxis, :, :]
        roi = np.repeat(roi, n_frames, axis=0)
    roi = roi[sort]

    assert roi.shape == (n_frames, 2, 2), 'ROI must have shape (n_frames, 2, 2)'
    height = roi[0, 0, 1] - roi[0, 0, 0]
    width = roi[0, 1, 1] - roi[0, 1, 0]
    assert np.all(roi[:, 1, 1] - roi[:, 1, 0] == width), 'ROI width must be consistent'
    assert np.all(roi[:, 0, 1] - roi[:, 0, 0] == height), 'ROI height must be consistent'

    if binSize > 1:
        frames = np.zeros((n_frames, height//binSize+1, width//binSize+1))
    else: 
        frames = np.zeros((n_frames, height, width))

    # Generate params in one go
    params = []
    rng = seed
    for i in range(frame_inds[-1]+1):
        p, rng = gen_gaborium_gabor_params(
            seed=rng,
            n_gabors=n_gabors,
            sf_range=sf_range,
            min_sf=min_sf,
            pix_per_deg=pix_per_deg,
            scale_range=scale_range,
            min_scale=min_scale,
            contrast=contrast,
            radius=radius,
            center=center,
        )
        if i in frame_inds:
            params.append(p)

    progress = (lambda x: tqdm(x, desc='Generating Gaborium Images')) if progress else (lambda x: x)

    if method == 'gpu':
        assert torch.cuda.is_available(), "CUDA is not available."
        # Load all relevant parameters onto the GPU
        device = torch.device('cuda')
        params = {k: ensure_tensor(np.stack([p[k] for p in params], axis=0), device) 
                    for k in params[0].keys()}
        roi = ensure_tensor(roi, device)

        iP = 0
        for iF in progress(range(n_frames)):
            # if the frame and the roi is the same as the previous frame, just copy the previous frame
            if iF > 0 and frame_inds[iF] == frame_inds[iF-1] and (roi[iF] == roi[iF-1]).all():
                frames[iF] = frames[iF-1]
                continue

            p = {k: params[k][iP] for k in params.keys()}
            iP += 1
            
            frames[iF] = gen_gaborium_image_from_params_gpu(
                **p,
                roi=roi[iF],
                binSize=binSize,
            ).cpu().numpy()
    else: 
        iP = 0
        for iF in progress(range(n_frames)):
            # if the frame and the roi is the same as the previous frame, just copy the previous frame
            if iF > 0 and frame_inds[iF] == frame_inds[iF-1] and (roi[iF] == roi[iF-1]).all():
                frames[iF] = frames[iF-1]
                continue

            if method == 'numba':
                frames[iF] = gen_gaborium_image_from_params_numba(
                    **params[iP],
                    roi=roi[iF],
                    binSize=binSize,
                )

            elif method == 'numpy':
                frames[iF] = gen_gaborium_image_from_params_numpy(
                    **params[iP],
                    roi=roi[iF],
                    binSize=binSize,
                )
            iP += 1
    # Sort the frames back into the original order
    sort_inv = np.argsort(sort)
    return frames[sort_inv]

class GaboriumTrial:
    '''
    Class to generate the frames from a single Gaborium trial.

    Parameters
    ----------
    trial_data : dict
        Trial data from the ExpStruct (ExpStruct['D'][iT])
    exp_settings : dict
        Experiment settings from the ExpStruct (ExpStruct['S'])
    stride : int
        Stride to use for the movie frames. Default is 1.
    method : str
        Method to use for generating the Gaborium images. Options are 'gpu', 'numba', and 'numpy'.
    draw_latency : float
        Latency between PTB flip times and drawing the stimulus to the screen. Default is 8.3 ms.
    '''
    def __init__(self, trial_data, exp_settings, method='gpu', draw_latency=8.3e-3):
        self.trial_data = trial_data
        self.exp_settings = exp_settings
        self.center_pix = exp_settings['centerPix'] # (x, y)
        self.bkgnd = trial_data['P']['bkgd']
        self.pix_per_deg = float(exp_settings['pixPerDeg'])
        self.screen_rect = np.array([[exp_settings['screenRect'][1], exp_settings['screenRect'][3]], 
                                   [exp_settings['screenRect'][0], exp_settings['screenRect'][2]]], 
                                  dtype=int)  # (2, 2) array
         
        # (n_frames, 3) array
        # columns: PTB flip time, 1st gabor x position, 1st gabor frequency in cpd
        self.noise_history = trial_data['PR']['NoiseHistory']

        # (n_frames, 4) array
        # columns: x, y, probe_index, PTB flip time 
        # If probe_index is > 0 then the probe is a gabor and the index is the index of the probe texture + 1
        # If probe_index is < 0 then the probe is a face and the index is -probe_index - 1
        # If probe_index is 0 then a reward was administered and the probe was the previously shown probe
        # If probe_index is nan, then no probe was shown
        self.probe_history = trial_data['PR']['ProbeHistory']

        assert len(self.noise_history) == len(self.probe_history), 'Noise history and probe history must have the same length'
        assert np.all(self.probe_history[:, 3] == self.noise_history[:, 0]), 'Noise history and probe history must have the same flip times'

        self.draw_latency = draw_latency
        self.flip_times = trial_data['PR']['NoiseHistory'][:, 0] + draw_latency
        self.n_frames = len(self.flip_times)
        self.reward_frames = np.where(self.probe_history[:, 3] == 0)[0]
        self.reward_times = self.flip_times[self.probe_history[:, 3] == 0] 

        # Fixing ProbeHistory to match state updates in 
        # PR_ForageProceduralNoise.state_and_screen_update()

        self.p_positions = self.probe_history[:, :2].copy()
        self.p_index = self.probe_history[:, 2].copy()

        # Fill the last NaN with the subsequent probe index
        # This accounts for the last frame of state 2 (no probe) being
        # actually drawn as the first frame of state 1 (probe).
        for iF in range(len(self.p_index)-1):
            if np.isnan(self.p_index[iF]) and not np.isnan(self.p_index[iF+1]):
                self.p_index[iF] = self.p_index[iF+1]
                self.p_positions[iF] = self.p_positions[iF+1]

        # Remove the face from the last frame before NaNs
        for iF in range(len(self.p_index)-1):
            if self.p_index[iF] < 0 and np.isnan(self.p_index[iF+1]):
                self.p_index[iF] = np.nan
                self.p_positions[iF] = np.nan

        # Fill in any NaN values in the probe positions with the previous position
        for iF in range(1, len(self.p_positions)):
            prev = np.any(np.isnan(self.p_positions[iF]))
            if prev:
                self.p_positions[iF, :] = self.p_positions[iF-1, :]

        # Fill in any 0 values in the probe index with the next index
        for iF in range(len(self.p_index)-2, -1, -1):
            if np.isnan(self.p_index[iF]):
                continue
            if int(self.p_index[iF]) == 0:
                self.p_index[iF] = self.p_index[iF+1]


        # Generate probe textures
        self.p_params = get_probe_params(trial_data, exp_settings)
        self.p_textures = [gen_probe_texture(**p) for p in self.p_params]
        
        # Load face textures
        self.face_library = get_face_library()
        self.f_textures = [gen_gauss_image_texture(
            self.face_library[f'face{i:02}'], self.bkgnd, self.pix_per_deg) for i in range(1, 31)
        ]
        self.f_radius = trial_data['P']['faceradius']
         # Generate gaborium parameters
        self.g_params = get_gaborium_params(trial_data)

        self.g_method = method
        if self.g_method == 'gpu':
            if not torch.cuda.is_available():
                print('GPU not available, using numba optimizated CPU instead')
                self.g_method = 'numba'

        # Validate noise history can be reconstructed
        self.g_iteration = np.zeros(self.n_frames, dtype=int)
        iteration = 0
        sub_g_param_keys = ['n_gabors', 'sf_range', 'min_sf',
                            'pix_per_deg', 'scale_range', 'min_scale', 
                            'contrast', 'radius', 'center']
        g_params_sub = {k: self.g_params[k] for k in sub_g_param_keys}
        params, rng = gen_gaborium_gabor_params(**self.g_params)
        for i in range(self.n_frames):
            found_params = False
            for _ in range(10):
                x_gen = params['x_gabors'][0]
                freq_gen = params['frequencies'][0] / 2 / np.pi
                if np.allclose(x_gen, self.noise_history[i,1]) and np.allclose(freq_gen, self.noise_history[i,2]):
                    found_params = True
                    self.g_iteration[i] = iteration
                    break
                else:
                    params, rng = gen_gaborium_gabor_params(rng, **g_params_sub)
                    iteration += 1

            if not found_params:
                raise ValueError('Gaborium parameters could not be reconstructed. Something is wrong.')

    def get_frames(self, idx, roi=None, stride=1, progress=False):
        """
        Generates and returns the frames for the specified indices.

        Parameters
        ----------
        idx : array-like
            Indices of the frames to generate.
        roi : array-like, optional
            Region of interest for the frames. If None, the entire screen is used.
            Should be of shape (2, 2) or (n_frames, 2, 2).
        stride : int, optional
            Stride to use for the frames. Default is 1.
        progress : bool, optional
            If True, displays a progress bar during frame generation. Default is False.

        Returns
        -------
        frames : np.ndarray
            Array of generated frames with shape (n_frames, height, width).
        """
        
        inds = np.arange(self.n_frames)[idx]
        g_inds = self.g_iteration[inds]
        n_frames = len(inds)

        if roi is None:
            roi = self.screen_rect
        else:
            roi = ensure_ndarray(roi, dtype=int)

        if roi.ndim == 2:
            roi = np.repeat(roi[np.newaxis, :, :], n_frames, axis=0)

        assert roi.shape == (n_frames, 2, 2), 'ROI must have shape (n_frames, 2, 2)'
        height = roi[0, 0, 1] - roi[0, 0, 0]
        width = roi[0, 1, 1] - roi[0, 1, 0]
        assert np.all(roi[:, 1, 1] - roi[:, 1, 0] == width), 'Width of ROI must be the same for all frames'
        assert np.all(roi[:, 0, 1] - roi[:, 0, 0] == height), 'Height of ROI must be the same for all frames'

        assert stride > 0, 'Stride must be positive'
        stride = int(stride)

        gabors = gen_gaborium_images(
            **self.g_params,
            roi=roi,
            binSize=stride,
            frame_inds=g_inds,
            method=self.g_method,
            progress=progress
        )

        if stride > 1:
            frames = np.zeros((n_frames, height//stride+1, width//stride+1), dtype=np.uint8)
        else:
            frames = np.zeros((n_frames, height, width), dtype=np.uint8)
        progress = (lambda x: tqdm(x, desc='Embedding Probes')) if progress else (lambda x: x)
        for iF in progress(range(n_frames)):
            ind = inds[iF]
            # Get the probe image
            p_position = self.p_positions[ind]
            p_index = self.p_index[ind]
            p_index = int(p_index) if not np.isnan(p_index) else p_index
            if p_index > 0:
                # If the probe_ind > 0 then the probe is a gabor
                im_probe, alpha = place_probe_texture(*self.p_textures[p_index-1], p_position, self.center_pix, 
                                                      self.bkgnd, self.pix_per_deg, roi=roi[iF], binSize=stride)

                frames[iF] = blend_images(gabors[iF]+self.bkgnd, im_probe+self.bkgnd, alpha)
            elif p_index < 0:
                # If the probe_ind < 0 then the probe is a face
                face_ind = -p_index - 1
                im_face, alpha = place_gauss_image_texture(*self.f_textures[face_ind], p_position, self.f_radius, self.center_pix, 
                                                           self.bkgnd, self.pix_per_deg, roi=roi[iF], binSize=stride)

                frames[iF] = blend_images(gabors[iF]+self.bkgnd, im_face+self.bkgnd, alpha) 
            else: 
                frames[iF] = (gabors[iF] + self.bkgnd + .5).astype(np.uint8)

        return frames

    def get_frame(self, idx, roi=None, stride=1):
        return self.get_frames([idx], roi=roi, stride=stride)[0]
