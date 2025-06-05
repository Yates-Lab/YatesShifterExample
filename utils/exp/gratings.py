
from .general import gen_gauss_image_texture, place_gauss_image_texture, blend_images
from .foraging import get_probe_params, gen_probe_texture, place_probe_texture
from .support import get_face_library
from ..general import ensure_ndarray
from tqdm import tqdm
import numpy as np

class GratingsTrial:
    def __init__(self, trial_data, exp_settings, draw_latency=8.3e-3):
        self.trial_data = trial_data
        self.exp_settings = exp_settings

        self.bkgnd = trial_data['P']['bkgd']
        self.pix_per_deg = float(exp_settings['pixPerDeg'])
        self.screen_rect = np.array([[exp_settings['screenRect'][1], exp_settings['screenRect'][3]], 
                                   [exp_settings['screenRect'][0], exp_settings['screenRect'][2]]], 
                                  dtype=int)  # (2, 2) array
        self.center_pix = exp_settings['centerPix']
        self.ppd = float(exp_settings['pixPerDeg'])
         
        # (n_frames, 3) array
        # columns: PTB flip time, Orientation, Spatial frequency
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
        self.orientations = self.noise_history[:, 1]
        self.spatial_frequencies = self.noise_history[:, 2]
        self.grating_contrast = float(trial_data['P']['noiseContrast'])
        assert not trial_data['P']['noiseRandomizePhase'], 'Phase randomization not implemented yet'
        assert trial_data['P']['phase'] == 0, 'Phase not implemented yet'


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
    
    def gen_grating(self, sf, ori, contrast, roi, stride=1):
        """
        Generates a grating texture.

        Parameters
        ----------
        sf : float
            Spatial frequency of the grating (cycles per degree)
        ori : float
            Orientation of the grating (degrees)
        contrast : float
            Contrast of the grating (0-1)
        roi : array-like
            Region of interest for the grating. Should be of shape (2, 2).
        stride : int, optional
            Stride to use for the grating. Default is 1.
            
        Returns
        -------
        img : np.ndarray
            Grating texture with shape (height, width).
            Height is (roi[0, 1] - roi[0, 0]) // stride + 1
            Width is (roi[1, 1] - roi[1, 0]) // stride + 1.
        """
        res = 2001 # from stimuli.gratingFFNoise.updateTextures
        x = np.arange(roi[1,0], roi[1,1], stride)
        y = np.arange(roi[0,0], roi[0,1], stride)
        X, Y = np.meshgrid(x-res/2, y-res/2)

        if sf == 0:
            return np.zeros_like(X)

        two_pi = 2 * 3.141592654
        deg2rad = two_pi / 360
        max_rad = two_pi * sf / self.ppd
        ori_rad = (90 - ori) * deg2rad

        gx = np.cos(ori_rad) * (X * max_rad) + np.sin(ori_rad) * (Y * max_rad) 
        img = np.cos(gx) * .5 * contrast * 255
        # Remove pixels outside the screen rect
        x_oob = np.logical_or(x < self.screen_rect[1, 0], x > self.screen_rect[1, 1])
        img[:, x_oob] = 0
        # Remove pixels outside the screen rect
        y_oob = np.logical_or(y < self.screen_rect[0, 0], y > self.screen_rect[0, 1])
        img[y_oob, :] = 0

        return img

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
        should_squeeze = np.isscalar(idx)
        idx = ensure_ndarray(idx, dtype=int)
        inds = np.arange(self.n_frames)[idx]
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

        if stride > 1:
            frames = np.zeros((n_frames, height//stride+1, width//stride+1), dtype=np.uint8)
        else:
            frames = np.zeros((n_frames, height, width), dtype=np.uint8)

        progress = (lambda x: tqdm(x, desc='Generating Frames')) if progress else (lambda x: x)
        for iF in progress(range(n_frames)):
            ind = inds[iF]
            # Get the noise image
            ori = self.orientations[ind]
            sf = self.spatial_frequencies[ind]
            grating = self.gen_grating(sf, ori, self.grating_contrast, roi[iF], stride)

            # Get the probe image
            p_position = self.p_positions[ind]
            p_index = self.p_index[ind]
            p_index = int(p_index) if not np.isnan(p_index) else p_index
            if p_index > 0:
                # If the probe_ind > 0 then the probe is a gabor
                im_probe, alpha = place_probe_texture(*self.p_textures[p_index-1], p_position, self.center_pix, 
                                                      self.bkgnd, self.pix_per_deg, roi=roi[iF], binSize=stride)
                frames[iF] = blend_images(grating+self.bkgnd, im_probe+self.bkgnd, alpha)
            elif p_index < 0:
                # If the probe_ind < 0 then the probe is a face
                face_ind = -p_index - 1
                im_face, alpha = place_gauss_image_texture(*self.f_textures[face_ind], p_position, self.f_radius, self.center_pix, 
                                                           self.bkgnd, self.pix_per_deg, roi=roi[iF], binSize=stride)

                frames[iF] = blend_images(grating+self.bkgnd, im_face+self.bkgnd, alpha) 
            else: 
                frames[iF] = (grating + self.bkgnd + .5).astype(np.uint8)

        if should_squeeze:
            frames = frames.squeeze(0)

        return frames

    def gen_grating_phase(self, sf, ori, roi, stride=1):
        """
        Generates an image where each pixel value corresponds to the phase of the grating.
        """
        res = 2001 # from stimuli.gratingFFNoise.updateTextures
        x = np.arange(roi[1,0], roi[1,1], stride)
        y = np.arange(roi[0,0], roi[0,1], stride)
        X, Y = np.meshgrid(x-res/2, y-res/2)
        if sf == 0:
            return -1 * np.ones_like(X)
        
        two_pi = 2 * 3.141592654
        deg2rad = two_pi / 360
        max_rad = two_pi * sf / self.ppd
        ori_rad = (90 - ori) * deg2rad
        gx = np.cos(ori_rad) * (X * max_rad) + np.sin(ori_rad) * (Y * max_rad)
        img = np.mod(gx, 2*np.pi)
        # Remove pixels outside the screen rect
        x_oob = np.logical_or(x < self.screen_rect[1, 0], x > self.screen_rect[1, 1])
        img[:, x_oob] = -1
        # Remove pixels outside the screen rect
        y_oob = np.logical_or(y < self.screen_rect[0, 0], y > self.screen_rect[0, 1])
        img[y_oob, :] = -1
        return img
    
    def get_frames_phase(self, idx, roi=None, stride=1, progress=False):
        """
        Generates and returns the phase frames for the specified indices.

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
            Array of generated phase frames with shape (n_frames, height, width).
        """
        
        should_squeeze = np.isscalar(idx)
        idx = ensure_ndarray(idx, dtype=int)
        inds = np.arange(self.n_frames)[idx]
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
        if stride > 1:
            frames = np.zeros((n_frames, height//stride+1, width//stride+1), dtype=np.float32)
        else:
            frames = np.zeros((n_frames, height, width), dtype=np.float32)

        progress = (lambda x: tqdm(x, desc='Generating Phase Frames')) if progress else (lambda x: x)
        for iF in progress(range(n_frames)):
            ind = inds[iF]
            # Get the noise image
            ori = self.orientations[ind]
            sf = self.spatial_frequencies[ind]
            phase_img = self.gen_grating_phase(sf, ori, roi[iF], stride)

            # Fill the frames with the phase image
            frames[iF] = phase_img

            # Set phase to -1 if the probe is at least 50% alpha
            p_position = self.p_positions[ind]
            p_index = self.p_index[ind]
            p_index = int(p_index) if not np.isnan(p_index) else p_index
            if p_index > 0:
                # If the probe_ind > 0 then the probe is a gabor
                _, alpha = place_probe_texture(*self.p_textures[p_index-1], p_position, self.center_pix, 
                                                      self.bkgnd, self.pix_per_deg, roi=roi[iF], binSize=stride)
                # If the alpha is greater than 0.5, set the phase image to -1
                mask = alpha > 0.5
                frames[iF][mask] = -1  # Set phase to -1 where the probe is present and alpha > 0.5
            elif p_index < 0:
                # If the probe_ind < 0 then the probe is a face
                face_ind = -p_index - 1
                _, alpha = place_gauss_image_texture(*self.f_textures[face_ind], p_position, self.f_radius, self.center_pix, 
                                                           self.bkgnd, self.pix_per_deg, roi=roi[iF], binSize=stride)

                # If the alpha is greater than 0.5, set the phase image to -1
                mask = alpha > 0.5
                frames[iF][mask] = -1

        if should_squeeze:
            frames = frames.squeeze(0)
        
        return frames
