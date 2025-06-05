import numpy as np
import pandas as pd

class FixCalibTrial:
    def __init__(self, trial, settings):

        self.pix_per_deg = settings['pixPerDeg']
        self.center_pix = settings['centerPix'][::-1]

        # (ptb_time, state, x, y)
        # ij = (-y, x) * pixPerDeg + centerPix[::-1]
        # State: 1 = fixation acquired, 2 = fixation completed, 3 = fixation lost
        self.fix_list = trial['PR']['fixList']

        if self.fix_list is None:
            self.fixations = pd.DataFrame(columns=['onset', 'offset', 'x', 'y', 'i', 'j'])
            return

        if self.fix_list.ndim == 1:
            self.fix_list = np.expand_dims(self.fix_list, axis=0)

        fixations = []
        for iF in range(len(self.fix_list)-1):
            fix_state = self.fix_list[iF][1]
            # Only look at fixation acquisitions
            if fix_state != 1:
                continue
            # Skip fixation if the fixation was not completed
            if self.fix_list[iF+1][1] != 2:
                continue

            fix_onset = self.fix_list[iF][0]
            fix_offset = self.fix_list[iF+1][0]
            assert np.all(self.fix_list[iF+1][2:4] == self.fix_list[iF][2:4]), 'Fixation onset and offset have different coordinates'
            fix_x, fix_y = self.fix_list[iF][2:4]
            fix_i = -fix_y * self.pix_per_deg + self.center_pix[0]
            fix_j = fix_x  * self.pix_per_deg + self.center_pix[1]
            fixations.append((fix_onset, fix_offset, fix_x, fix_y, fix_i, fix_j))
        self.fixations = pd.DataFrame(fixations, columns=['onset', 'offset', 'x', 'y', 'i', 'j'])
    
    def get_fix_data(self, t_ptb, signals, reduction=lambda x: np.median(x, axis=0)):
        fix_data = []
        for iF, fix in self.fixations.iterrows():
            i0 = np.searchsorted(t_ptb, fix['onset'])
            i1 = np.searchsorted(t_ptb, fix['offset'])
            fix_data.append(reduction(signals[i0:i1]))
        if fix_data:
            return np.array(fix_data)
        else:
            return np.array([]).reshape(0, signals.shape[1])
