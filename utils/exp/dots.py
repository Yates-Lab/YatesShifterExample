import numpy as np
from .general import get_trial_protocols, get_clock_functions
from ..spikes import bin_spikes
from scipy.interpolate import interp1d
from ..rf import calc_sta, Gaussian2D, thresholded_centroid
from scipy.ndimage import gaussian_filter

class DotsTrial:
    def __init__(self, trial_data, exp_settings, draw_latency=8.3e-3):
        self.trial_data = trial_data
        self.exp_settings = exp_settings
        self.draw_latency = draw_latency
        self.flip_times = trial_data['PR']['NoiseHistory'][:, 0] + draw_latency
        self.n_dots = int(trial_data['P']['numDots'])
        self.center_pix = self.exp_settings['centerPix'][::-1]
        self.dots_pix = np.stack([
            -trial_data['PR']['NoiseHistory'][:, 1+self.n_dots:1+2*self.n_dots],
            trial_data['PR']['NoiseHistory'][:, 1:1+self.n_dots]
            
        ], axis=2) + self.center_pix# (n_frames, n_dots, 2 [i, j])
    
    def __repr__(self):
        return f'DotsTrial(t={self.flip_times[0]:.3f}-{self.flip_times[-1]:.3f}, n_dots={self.n_dots})'

def dots_rf_map_session(exp, dpi, ks_results,
                        dt=1/240, lags=np.arange(7, 14), 
                        roi_deg=np.array([[-4, 4], [-4, 4]]), dxy_deg=.25):
    '''
    Calculate the receptive field of a session using the ForageDots protocol.
    
    Parameters
    ----------
    sess : YatesV1Session
        Session object to process.
    dt : float, optional
        Time bin size in seconds. Default is 1/240.
    lags : np.ndarray, optional
        Lags to use for the receptive field. Default is np.arange(7, 14).
    roi_deg : np.ndarray, optional
        Region of interest in degrees. Default is np.array([[-4, 4], [-4, 4]]).
    dxy_deg : float, optional
        Spatial bin size in degrees. Default is .25.
    
    Returns
    -------
    out_dict : dict or None if no ForageDots trials are found
        Dictionary containing the following keys:
        - rf : np.ndarray
            Receptive field map. [n_i, n_j]
        - i_ax : np.ndarray
            Row indices of the receptive field map.
        - j_ax : np.ndarray
            Column indices of the receptive field map.
        - rf_pix : np.ndarray
            Receptive field center in pixels.
        - rf_deg : np.ndarray
            Receptive field center in degrees.
        - ppd : float
            Pixels per degree.
    '''
    ptb2ephys, vpx2ephys = get_clock_functions(exp)
    protocols = get_trial_protocols(exp)
    unique_protocols = list(set(protocols))
    protocol = 'ForageDots'
    if protocol not in unique_protocols:
        print(f'No trials of {protocol} found')
        return None
    iTs = np.where(np.array(protocols) == protocol)[0]
    exp_settings = exp['S']

    # Load trials
    trials = [DotsTrial(exp['D'][iT], exp_settings) for iT in iTs]
    print(f'Found {len(trials)} {protocol} trials')

    # Load DPI data
    t_dpi = dpi['t_ephys'].values
    dpi_pix = dpi[['dpi_i', 'dpi_j']].values
    dpi_ptb_interp = interp1d(t_dpi, dpi_pix, kind='linear', fill_value='extrapolate', axis=0)

    # Load spikes
    st = ks_results.spike_times
    clu = ks_results.spike_clusters
    cids = np.unique(clu)

    # Bin spikes, DPI, and dots
    t_bins = []
    robs = []
    trial_id = []
    dots_pix = []
    gaze_pix = []
    for iT in range(len(trials)):
        trial = trials[iT]
        flip_times = ptb2ephys(trial.flip_times)
        trial_bin_edges = np.arange(flip_times[0], flip_times[-1], dt)
        trial_bins = trial_bin_edges[:-1] + dt/2
        t_bins.append(trial_bins)

        trial_num = np.ones_like(trial_bins) * iT
        trial_id.append(trial_num)

        trial_bin_inds = np.searchsorted(flip_times, trial_bins) - 1
        dots_pix.append(trial.dots_pix[trial_bin_inds])
        gaze_pix.append(dpi_ptb_interp(trial_bins))
        robs_trial = bin_spikes(st, trial_bin_edges, clu, cids)
        robs.append(robs_trial)

    robs = np.concatenate(robs)
    dots_pix = np.concatenate(dots_pix)
    gaze_pix = np.concatenate(gaze_pix)
    t_bins = np.concatenate(t_bins)
    trial_id = np.concatenate(trial_id)
    dots_rel = dots_pix - gaze_pix[:, None, :]

    ppd = exp_settings['pixPerDeg']
    roi_pix = np.flipud(roi_deg * ppd)
    dxy_pix = dxy_deg * ppd

    i_edges = np.arange(roi_pix[0,0], roi_pix[0,1]+dxy_pix, dxy_pix)
    i_ax = (i_edges[:-1] + i_edges[1:]) / 2

    j_edges = np.arange(roi_pix[1,0], roi_pix[1,1]+dxy_pix, dxy_pix)
    j_ax = (j_edges[:-1] + j_edges[1:]) / 2

    n_frames = len(t_bins)
    stimX = np.zeros((n_frames, len(i_ax), len(j_ax)))
    for iF in range(n_frames):
        dots = dots_rel[iF]
        in_roi = np.logical_and.reduce([
            dots[:, 0] > roi_pix[0, 0], # dots with row > roi row min
            dots[:, 0] < roi_pix[0, 1], # dots with row < roi row max
            dots[:, 1] > roi_pix[1, 0], # dots with col > roi col min
            dots[:, 1] < roi_pix[1, 1], # dots with col < roi col max
        ])
        if np.sum(in_roi) == 0:
            continue
        dots_in_roi = dots[in_roi]
        i = np.searchsorted(i_edges, dots_in_roi[:,0]) - 1
        j = np.searchsorted(j_edges, dots_in_roi[:,1]) - 1
        stimX[iF, i, j] = 1

    robs_all = np.sum(robs, axis=1)
    response = np.zeros_like(robs_all)
    for iLag in lags:
        response += np.roll(robs_all, -iLag)

    rf = calc_sta(stimX[...,None], response, [0], reverse_correlate=False)
    rf = rf.squeeze().numpy() / dt
    rf = gaussian_filter(rf, 1)

    rf_gauss = Gaussian2D(*Gaussian2D.est_p0(rf, j_ax, i_ax))
    rf_gauss.fit(rf, j_ax, i_ax)

    rf_pix = np.round([rf_gauss.y0, rf_gauss.x0]).astype(int)
    rf_deg = np.flipud(rf_pix / ppd) * np.array([1, -1])

    out_dict = {
        'rf': rf,
        'i_edges': i_edges,
        'i_ax': i_ax,
        'j_edges': j_edges,
        'j_ax': j_ax,
        'rf_pix': rf_pix,
        'rf_deg': rf_deg,
    }

    return out_dict