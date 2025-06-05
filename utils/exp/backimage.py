from PIL import Image
import numpy as np
from ..general import ensure_ndarray, nd_cut
from .support import get_backimage_directory

class BackImageTrial:
    def __init__(self, trial, settings, draw_latency=8.3e-3):
        self.trial = trial
        self.settings = settings
        self.bkgnd = trial['P']['bkgd']
        self.dest_rect = trial['PR']['destRect'].astype(np.int64)
        self.image_file = trial['PR']['imagefile'].split('/')[-1]
        self.image_onset_ptb = trial['PR']['startTime'] + draw_latency
        self.image_offset_ptb = trial['PR']['imageOff'] + draw_latency

    def __repr__(self):
        return f'BackImageTrial({self.image_file}, {self.image_onset_ptb}, {self.image_offset_ptb})'
    
    def get_image(self, reduction='mean'):
        '''
        Get the image displayed in the trial

        Returns
        -------
        image : np.ndarray
            Image displayed in the trial with shape (height, width).
        '''
        image = Image.open(get_backimage_directory() / self.image_file)
        image = image.resize((self.dest_rect[2:] - self.dest_rect[:2]), resample=2)
        image = np.array(image)
        if image.ndim == 3 and reduction == 'mean':
            image = np.mean(image, axis=2).astype(np.uint8)
        return image

    def get_roi(self, roi=None, stride=1):
        '''
        Get rois of the image displayed in the trial

        Parameters
        ----------
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
        # image = Image.open(get_backimage_directory() / self.image_file)
        # image = image.resize((self.dest_rect[2:] - self.dest_rect[:2]), resample=2)
        # image = np.array(image)
        # if image.ndim == 3:
        #     image = np.mean(image, axis=2).astype(np.uint8)
        image = self.get_image()
        
        if roi is None:
            roi = np.flipud(np.stack([self.dest_rect[:2], self.dest_rect[2:]], axis=1))
        else:
            roi = ensure_ndarray(roi, dtype=int)

        should_squeeze = False
        if roi.ndim == 2:
            roi = roi[None, :, :]
            should_squeeze = True

        n_frames = len(roi)
        assert roi.shape == (n_frames, 2, 2), 'ROI must have shape (n_frames, 2, 2)'
        height = roi[0, 0, 1] - roi[0, 0, 0]
        width = roi[0, 1, 1] - roi[0, 1, 0]
        assert np.all(roi[:, 1, 1] - roi[:, 1, 0] == width), 'ROI width must be consistent'
        assert np.all(roi[:, 0, 1] - roi[:, 0, 0] == height), 'ROI height must be consistent'

        assert stride > 0, 'Stride must be positive'
        stride = int(stride)

        ims = []
        src_pos = np.flipud(self.dest_rect[:2])
        
        for i in range(n_frames):
            im_roi = nd_cut(
                image, 
                roi[i,:,0] - src_pos,
                (height, width), 
                fill_value=int(self.bkgnd)
                )
            im_roi = im_roi[::stride, ::stride]
            ims.append(im_roi)
        ims = np.stack(ims)
        if should_squeeze:
            ims = np.squeeze(ims, axis=0)
        return ims