import numpy as np
from tqdm import tqdm
from scipy.stats import poisson
import matplotlib.pyplot as plt
from ..general import ensure_ndarray
from ..spikes import calc_ccgs

def refractory_violation_likelihood(
        n_violations, 
        contam_prop,
        refractory_period,
        firing_rate, 
        n_spikes, 
        ):
    '''
    Calculate the likelihood of an observed number of refractory period violations under a poisson 
    model of refractory violations. likelihood = P(X <= N_v | R_c, T_ref, F_r, N_s), where X is a 
    poisson random variable with rate R_c * T_ref * F_r * N_s and N_v is the observed number of
    refractory period violations in the cluster. R_c is a specified contamination rate,
    T_ref is the refractory period, F_r is the firing rate of the cluster, and N_s is the number of
    spikes in the cluster.

    Parameters
    ----------
    n_violations : array_like
        the observed number of violations
    contam_prop : array_like
        the contamination proportion to test (as a proportion of the firing rate)
    refractory_period : array_like
        the refractory period in seconds
    firing_rate : array_like
        the firing rate of the cluster in Hz
    n_spikes : array_like
        the number of spikes in the cluster

    Returns
    -------
    likelihood : float
        the likelihood of the observing the number of violations or less if the cluster was contaminated at the rate specified

    '''
    # rate of contaminated spikes per second
    contamination_firing_rate = firing_rate * contam_prop

    # expected number of violations in the autocorrelogram
    expected_violations = contamination_firing_rate * refractory_period * n_spikes

    # likelihood of observing the number of violations or less
    likelihood = poisson.cdf(n_violations, expected_violations)

    return likelihood

def binary_search_rv_rate(n_violations, refractory_period, firing_rate, n_spikes, alpha=0.05, 
                         max_contam_prop=1.0, tol=1e-6, max_iter=100):
    """
    Perform binary search to find minimum contamination rate that can be rejected.
    
    Parameters
    ----------
    n_violations : int
        Observed number of violations
    refractory_period : float
        Refractory period in seconds
    firing_rate : float
        Firing rate in Hz
    n_spikes : int
        Number of spikes
    alpha : float
        Significance level
    max_contam_prop : float
        Maximum contamination proption to test (as proportion of firing rate)
    max_iter : int
        Maximum number of iterations
        
    Returns
    -------
    float
        Minimum contamination rate that can be rejected under a poisson model of refractory violations.
    """
    left = 0
    right = max_contam_prop
    mid = 0 
    for _ in range(max_iter):
        mid = (left + right) / 2
        likelihood = refractory_violation_likelihood(
            n_violations, mid, refractory_period, firing_rate, n_spikes)

        if likelihood < alpha and likelihood > alpha - tol:
            return mid
        elif likelihood < alpha - tol:
            right = mid
        else:
            left = mid
    return mid

def compute_min_contam_props(spike_times, spike_clusters=None, cids=None,
                       refractory_periods=np.exp(np.linspace(np.log(0.5e-3), np.log(10e-3), 100)),
                       max_contam_prop=1,
                       fr_est_dur = 1,
                       alpha = 0.05,
                       ref_acg_t_start = .25e-3, 
                       progress = False):
    '''
    Compute the minimum contamination rate that can be rejected for each cluster in the dataset under a poisson model of refractory violations 
    for a range of refractory periods.

    Parameters
    ----------
    spike_times : array-like (n_spikes,)
        Spike times in seconds.
    spike_clusters : array-like (n_spikes,)
        Cluster IDs for each spike. If None, all spikes are assumed to be in the same cluster.
    cids : array-like (n_clusters,)
        Cluster IDs to test. Results returned in order of cids. If None, all clusters are tested.
    refractory_periods : array-like (n_refractory_periods,)
        Refractory periods to test in seconds.
    max_contam_prop : float
        Maximum contamination proportion to test (as a proportion of the firing rate).
    fr_est_dur : float
        Duration of the firing rate estimation window in seconds.
    alpha : float
        Significance level for the test.
    ref_acg_t_start : float
        Start time for the refractory period autocorrelogram in seconds. 
        (necessary because Kilosort removes "duplicate" spikes within a .25 ms window)
    progress : bool
        Show a progress bar.

    Returns
    -------
    min_contam_props : array (n_clusters, n_refractory_periods)
        Minimum contamination rate that can be rejected under a poisson model of refractory violations.
    firing_rates : array (n_clusters,)
        Firing rates for each cluster.
    

    '''
    spike_times = ensure_ndarray(spike_times).squeeze()

    if spike_clusters is None:
        spike_clusters = np.zeros(len(spike_times), dtype=np.int32)
    spike_clusters = ensure_ndarray(spike_clusters, dtype=np.int32).squeeze()
    assert spike_clusters.ndim == 1
    assert len(spike_times) == len(spike_clusters), "Spike times and spike clusters must have the same length."

    if cids is not None:
        cids = ensure_ndarray(cids, dtype=np.int32)
        cids_check = np.unique(spike_clusters)
        assert np.all(np.in1d(cids, cids_check)), "Some clusters are not in spike_clusters."
    else:
        cids = np.unique(spike_clusters)

    assert np.all(refractory_periods > 0), "Refractory periods must be positive."
    assert np.all(np.diff(refractory_periods) > 0), "Refractory periods must be monotonic."
    assert max_contam_prop > 0, "Contamination test proportions must be positive."


    firing_rates = np.zeros(len(cids))
    min_contam_props = np.ones((len(cids), len(refractory_periods))) * max_contam_prop
    for iC in tqdm(range(len(cids)), disable=not progress, desc="Calculating contamination"):
        cid = cids[iC]
        st_clu = spike_times[spike_clusters == cid]
        n_spikes = len(st_clu)
        firing_rate = calc_ccgs(st_clu, [0, fr_est_dur]).squeeze() / fr_est_dur / n_spikes
        firing_rates[iC] = firing_rate
        acg = calc_ccgs(st_clu, np.r_[ref_acg_t_start, refractory_periods]).squeeze()
        n_violations = np.cumsum(acg) # number of refractory violations for each refractory period

        # For each refractory period, find minimum violation rate that can be rejected
        for iR, n_viols in enumerate(n_violations):
            ref_period = refractory_periods[iR] - ref_acg_t_start # adjust by the start time of the acg
            min_contam_props[iC, iR] = binary_search_rv_rate(
                n_viols, ref_period, firing_rate, n_spikes, 
                alpha=alpha, max_contam_prop=max_contam_prop)

    return min_contam_props, firing_rates

def plot_min_contam_prop(spike_times, min_contam_props, refractory_periods, 
                         n_bins = 50, max_contam_prop=1, acg_t_start = .25e-3, axs=None):
    '''
    Utility for plotting the minimum contamination proportion that can be rejected for each cluster in the dataset.
    
    Parameters
    ----------
    spike_times : array-like (n_spikes,)
        Spike times in seconds.
    min_contam_props : array (n_refractory_periods)
        Minimum contamination rate that can be rejected under a poisson model of refractory violations.
    refractory_periods : array-like (n_refractory_periods,)
        Refractory periods to test in seconds.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    axs : list of matplotlib.axes.Axes (2,)
        The axes objects.

    '''

    isis = np.diff(spike_times) * 1000
    max_refrac = refractory_periods.max() * 1000
    min_isi = acg_t_start * 1000
    min_prop = min_contam_props.min()

    if axs is None:
        fig, axs = plt.subplots(1,1)
    else:
        fig = axs.get_figure()
    bins = np.linspace(min_isi, max_refrac, n_bins)
    axs.hist(isis, bins=bins, edgecolor='black', color='black', alpha=0.6)
    axs.set_xlim([min_isi, max_refrac])
    axs.set_ylabel('ISI count (spikes)')
    axs.set_xlabel('ISI / Refractory Period (ms)')
    axs2 = axs.twinx()
    axs2.plot(refractory_periods*1000, min_contam_props, color='red', linewidth=3.5)
    axs2.axhline(min_prop, color='red', linestyle='--', linewidth=2)
    yticks = np.concatenate([np.linspace(0, max_contam_prop, 6), [min_prop]])
    axs2.set_ylim([0, max_contam_prop])
    axs2.set_yticks(yticks)
    axs2.set_yticklabels(['0', '', '', '', '', '1', f'{min_prop:.4g}'])
    axs2.tick_params(axis='y', colors='red')
    axs2.set_ylabel('Minimum Rejected Contamination Proportion', color='red')

    return fig, axs

# Depricated code from Nick Steinmetz's lab (Sliding RP violations)
# https://github.com/SteinmetzLab/slidingRefractory/blob/1.0.0/python/slidingRP/metrics.py
def compute_rvl_tensor(spike_times, spike_clusters=None, cids=None,
                      refractory_periods=np.exp(np.linspace(np.log(0.5e-3), np.log(10e-3), 100)),
                      contamination_test_proportions=np.exp(np.linspace(np.log(5e-3), np.log(.35), 50)),
                      fr_est_dur = 1,
                      ref_acg_t_start = .25e-3, 
                      progress = False):
    '''
    Compute the likelihood of observing the number of refractory period violations or fewer for many clusters, refractory periods, and test comtamination rates.

    Parameters
    ----------
    spike_times : array-like (n_spikes,)
        Spike times in seconds.
    spike_clusters : array-like (n_spikes,)
        The cluster ids for each spike. If None, all spikes are assumed to belong to a single cluster.
    cids : array-like (n_clusters,)
        The list of *all* unique clusters, in any order. That order will be used in the output array. If None, order the clusters by their appearance in `spike_clusters`.
    refractory_periods : array-like (n_refrac,)
        The refractory periods to test, in seconds.
    contamination_test_proportions : array-like (n_contam,)
        The contamination rates to test, as a proportion of the firing rate.
    fr_est_dur : float
        The duration in seconds over which to estimate the firing rate. 
    ref_acg_t_start : float. Default is .25e-3
        The start time in seconds for the refractory period autocorrelogram. 
        Necessary for Kilosort4, which removes duplicate spikes in a .25 ms window, which negatively biases the refractory likelihood estimates.

    Returns
    -------
    rvl_tensor : array
        A `(n_clusters, n_refrac, n_contam)` array with the likelihood of observing the number of refractory period violations or less if the cluster was contaminated at the rate specified.
    '''
    spike_times = ensure_ndarray(spike_times).squeeze()

    if spike_clusters is None:
        spike_clusters = np.zeros(len(spike_times), dtype=np.int32)
    spike_clusters = ensure_ndarray(spike_clusters, dtype=np.int32).squeeze()
    assert spike_clusters.ndim == 1
    assert len(spike_times) == len(spike_clusters), "Spike times and spike clusters must have the same length."

    if cids is not None:
        cids = ensure_ndarray(cids, dtype=np.int32)
        cids_check = np.unique(spike_clusters)
        assert np.all(np.in1d(cids, cids_check)), "Some clusters are not in spike_clusters."
    else:
        cids = np.unique(spike_clusters)

    rvl_tensor = np.ones((len(cids), len(contamination_test_proportions), len(refractory_periods)))

    iter = range(len(cids))
    if progress:
        iter = tqdm(iter, desc="Calculating RVL tensor", position=0, leave=True)
    for iC in iter:
        cid = cids[iC]
        cluster_spikes = spike_times[spike_clusters == cid]
        n_spikes = len(cluster_spikes)
        firing_rate = calc_ccgs(cluster_spikes, [0, fr_est_dur]).squeeze() / fr_est_dur / n_spikes
        acg = calc_ccgs(cluster_spikes, np.r_[ref_acg_t_start, refractory_periods]).squeeze()
        refractory_violations = np.cumsum(acg)

        rvl_tensor[iC] = refractory_violation_likelihood(
                            refractory_violations[None,:], 
                            contamination_test_proportions[:,None],
                            refractory_periods[None,:] - ref_acg_t_start, 
                            firing_rate, 
                            n_spikes)

    return rvl_tensor

def plot_rvl(cluster_spikes, likelihoods, refractory_periods, contamination_test_proportions, likelihood_threshold=0.05):
    min_refrac, max_refrac = refractory_periods.min(), refractory_periods.max()
    min_contam, max_contam = contamination_test_proportions.min(), contamination_test_proportions.max()

    isis = np.diff(cluster_spikes)

    min_likelihood_per_contam = np.min(likelihoods, axis=1)
    min_likelihood_per_contam[min_likelihood_per_contam < likelihood_threshold] = np.inf
    lowest_contam_idx = np.argmin(min_likelihood_per_contam)
    lowest_contam = contamination_test_proportions[lowest_contam_idx]
    lowest_contam_likelihood = likelihoods[lowest_contam_idx]

    fig, axs = plt.subplots(3, 1, figsize=(5, 12), height_ratios=[1, 1.5, 1])
    axs[0].hist(isis * 1000, bins=np.arange(0, max_refrac*1000, .33))
    axs[0].set_title(f'ISI distribution')
    axs[0].set_ylabel('Count')
    axs[0].set_xlabel('ISI (ms)')
    axs[0].set_xlim([0, max_refrac*1000])

    extent = [min_refrac*1000, max_refrac*1000, min_contam, max_contam]
    from matplotlib.image import NonUniformImage
    im = NonUniformImage(axs[1], extent=extent, interpolation='nearest', cmap='viridis')
    im.set_data(refractory_periods*1000, contamination_test_proportions, likelihoods)
    im.set_clim(0, 1)
    axs[1].add_image(im)
    axs[1].axhline(lowest_contam, color='red', linestyle='--')
    axs[1].set_xlim(extent[:2])
    axs[1].set_ylim(extent[2:])
    fig.colorbar(im, ax=axs[1], orientation='horizontal', label='Likelihood')
    axs[1].set_title(f'Likelihood of observed refractory period violations')
    axs[1].set_xlabel('Refractory period (ms)')
    axs[1].set_ylabel('Contamination rate')

    axs[2].semilogy(refractory_periods*1000, lowest_contam_likelihood)
    axs[2].axhline(likelihood_threshold, color='red', linestyle='--')
    axs[2].set_title(f'Highest contamination rate more than {likelihood_threshold*100:.1g}% likely: {lowest_contam*100:.3g}%')
    axs[2].set_xlabel('Refractory period (ms)')
    axs[2].set_ylabel('Likelihood')
    axs[2].set_xlim([min_refrac*1000, max_refrac*1000])

    plt.tight_layout()
    return fig, axs


