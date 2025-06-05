import numpy as np
from .support import get_rsvp_fix_stim
from ..general import ensure_ndarray
from .general import place_gauss_image_texture, gen_gauss_image_texture

class FixRsvpTrial:
    def __init__(self, trial, settings, draw_latency=8.3e-3):
        self.trial = trial
        self.settings = settings
        self.bkgnd = trial['P']['bkgd']
        self.screen_rect = settings['screenRect'].astype(int)

        self.center_pix = settings['centerPix']
        hist = trial['PR']['NoiseHistory']
        if hist is None:
            hist = np.zeros((0, 4))
        elif hist.ndim == 1:
            hist = hist[None, :]
        # Last flip is typically delayed with a long latency
        self.flip_times = hist[:-1, 0] + draw_latency
        self.positions = (hist[:-2, 1:3].astype(int) - self.center_pix) / settings['pixPerDeg']
        self.image_ids = hist[:-2, 3].astype(int)
        self.radius = trial['P']['faceRadius']
        self.ppd = settings['pixPerDeg']

    @staticmethod
    def is_valid(trial):
        if 'PR' not in trial:
            return False
        if 'NoiseHistory' not in trial['PR']:
            return False
        hist = trial['PR']['NoiseHistory']
        if hist is None:
            return False
        return hist.ndim == 2 and len(hist) > 3 and hist.shape[1] == 4
          
    def get_rois(self, idx, roi=None, stride=1):
        '''
        Get rois of the image displayed in the trial

        Parameters
        ----------
        idx : int or np.ndarray
            Index of the image to extract. If an array, must be of shape (N,).
        roi : tuple, optional
            Region of interest for the image. If None, the entire image is used.
            Should be of shape (N, 2, 2).
        stride : int, optional
            Stride to use for the image. Default is 1.

        Returns
        -------
        rois : np.ndarray
            Array of extracted rois with shape (N, height, width).
        '''

        # Validate times
        idx = ensure_ndarray(idx)

        should_squeeze = False
        if idx.ndim == 0:
            idx = idx[None]
            should_squeeze = True

        assert idx.ndim == 1, 'Times must be a 1D array'
        n_frames = len(idx)

        # Validate ROI
        if roi is None:
            roi = np.flipud(np.stack([self.screen_rect[:2], self.screen_rect[2:]], axis=1))
        else:
            roi = ensure_ndarray(roi, dtype=int)

        if roi.ndim == 2:
            roi = roi[None, :, :]
            roi = np.repeat(roi, n_frames, axis=0)

        assert roi.shape == (n_frames, 2, 2), 'ROI must have shape (n_frames, 2, 2)'
        height = roi[0, 0, 1] - roi[0, 0, 0]
        width = roi[0, 1, 1] - roi[0, 1, 0]
        assert np.all(roi[:, 1, 1] - roi[:, 1, 0] == width), 'ROI width must be consistent'
        assert np.all(roi[:, 0, 1] - roi[:, 0, 0] == height), 'ROI height must be consistent'
        assert n_frames == len(roi), 'Number of ROIs must match number of times'

        assert stride > 0, 'Stride must be positive'
        stride = int(stride)
        
        images = get_rsvp_fix_stim()
        ims = []
        for i in range(n_frames):
            im_id = self.image_ids[idx[i]]
            im = images[f'im{im_id:02d}'].mean(axis=2).astype(np.uint8)
            pos = self.positions[idx[i]]


            im_tex, alpha_tex = gen_gauss_image_texture(im, self.bkgnd)
            im, _ = place_gauss_image_texture(im_tex, alpha_tex, 
                                              pos, self.radius, self.center_pix,
                                              self.bkgnd, self.ppd, roi=roi[i], binSize=stride)

            im = (im + .5 + self.bkgnd).astype(np.uint8)
            ims.append(im)

        ims = np.stack(ims, axis=0)

        if should_squeeze:
            ims = np.squeeze(ims, axis=0)
        return ims