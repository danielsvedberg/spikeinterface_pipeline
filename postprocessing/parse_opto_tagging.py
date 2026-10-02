import pynapple as nap
from pathlib import Path
from preprocessing import parse_nidq as pni
from utils.mpl_backend import select_backend
select_backend()  # Qt in the main process, headless Agg in joblib workers
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
from utils import file_structure as fs

def plot_corellograms(ephys_data, bs, opto_events_idx, opto_tagging_folder):
    power_levels = opto_events_idx['power'].unique().tolist()
    correlograms_folder = opto_tagging_folder / "correlograms"
    correlograms_folder.mkdir(exist_ok=True)
    correlograms = []

    # iterate through rows of opto_events_idx
    for index, row in opto_events_idx.iterrows():
        print(index)
        event_corr = nap.compute_eventcorrelogram(
            group=ephys_data, event=bs[index], binsize=0.1, windowsize=0.5
        )
        correlograms.append(event_corr)

    for n, neuron in enumerate(ephys_data.keys()):
        #make a subplot 5 panels, 1 for each power level, and in each panel plot the correlogram for that power level,
        fig, axs = plt.subplots(1, len(power_levels), figsize=(20, 5), sharey=True)
        for p, power in enumerate(power_levels):
            ax = axs[p]
            #get indices of correlograms where power level matches
            chr2_indices = opto_events_idx[
                (opto_events_idx['power'] == power) &
                (opto_events_idx['event'].str.contains('chr2'))
                ].index.tolist()
            chrimson_indices = opto_events_idx[
                (opto_events_idx['power'] == power) &
                (opto_events_idx['event'].str.contains('chrimson'))
                ].index.tolist()
            #get the correlograms for those indices
            #plot chr2 in shades of blue and chrimson in shades of red
            for idx in chrimson_indices:
                event_corr = correlograms[idx]
                ax.plot(
                    event_corr.index,
                    event_corr[neuron].values,
                    color='red',#chrimson_cmap(p),
                    label='chrimson' if idx == chrimson_indices[0] else ""
                    )
            for idx in chr2_indices:
                event_corr = correlograms[idx]
                ax.plot(
                    event_corr.index,
                    event_corr[neuron].values,
                    color='blue',#chr2_cmap(p),
                    label='ChR2' if idx == chr2_indices[0] else ""
                    )
            ax.set_title(f'Power Level: {power}')
            ax.set_xlabel('Time (s)')
        #save
        fig.suptitle(f'Neuron {neuron} Opto Tagging Correlograms')
        axs[0].set_ylabel('correlation')
        axs[-1].legend()
        plt.tight_layout()
        plt.savefig(correlograms_folder / f'neuron_{neuron}_opto_tagging_correlograms.png')
        plt.close(fig)


def count_spikes_in_intervals(spike_times, starts, ends):
    counts = 0
    j = 0
    n = spike_times.shape[0]
    for i in range(starts.shape[0]):
        start = starts[i]
        end = ends[i]
        # Move spike index pointer until spike_times[j] >= start
        while j < n and spike_times[j] < start:
            j += 1
        k = j
        while k < n and spike_times[k] < end:
            counts += 1
            k += 1
    return counts

# ------------------------------------------------------------
# Utility #3: Circular shift spike train
# ------------------------------------------------------------
#@njit(fastmath=True, parallel=True)

def get_auc(tuning_curve, thresh):
    y = tuning_curve.values#[tuning_curve.values > thresh]
    y = y[1:]
    auc = np.sum(y)
    return auc

def shuff_aucs_diff(tuning_curve, thresh):
    curve1 = tuning_curve[0].values
    curve2 = tuning_curve[1].values
    mask = np.random.rand(len(curve1)) < 0.5

    # Use the mask to swap efficiently
    a_swapped = np.where(mask, curve2, curve1)
    b_swapped = np.where(mask, curve1, curve2)
    #compute aucs
    auc1 = np.sum(a_swapped[a_swapped > thresh][1:])
    auc2 = np.sum(b_swapped[b_swapped > thresh][1:])
    #compute difference
    auc_diff = auc1 - auc2
    return auc_diff


def get_tuning_curves(neuron_data, opsins, opsin_traces, power_bins, fs_array):
    tuning_curves = []
    for o, opsin in enumerate(opsins):
        fs = float(fs_array[o])
        trace = opsin_traces[o]
        tuning_curve = nap.compute_tuning_curves(
            data=neuron_data,
            features=trace,
            bins=[power_bins],
            fs=fs)
        #replace nan with 0 in tuning_curve
        tuning_curve = tuning_curve.fillna(0)
        tuning_curves.append(tuning_curve)

    return tuning_curves


def get_firing_rate_percentiles(spikes, n_bins=30, n_iter=10000, bin_size=0.01, conf_level=0.95):
    #verify that spikes is a Ts object
    if not isinstance(spikes, nap.Ts):
        raise ValueError("spikes must be a nap.Ts object")

    sumspikes = spikes.count(bin_size)
    #get percentiles of firing rates
    #bootstrap the percentiles by sampling n_bins with size bin_size 10000 times

    def subsample_rate(counts):
        #pick n_bins random indices
        sample = np.random.choice(counts, size=n_bins, replace=False)
        return np.sum(sample) / (n_bins * bin_size)

    samples = [subsample_rate(sumspikes.values) for _ in range(n_iter)]
    #get 2.5 and 97.5 percentiles
    samples = np.array(samples)
    low_pctl = np.percentile(samples, (1 - conf_level) / 2 * 100)
    high_pctl = np.percentile(samples, (1 + conf_level) / 2 * 100)
    #alternative method:
    percentile_thresholds = [low_pctl, high_pctl]

    return percentile_thresholds


def plot_tuning_curves(neuron, opsins, tuning_curves, pctl_thrsh, auc_diff_pctls, tuning_curves_folder):
    #for tunovth, make any value of tuning curves less than auc_diff_percentiles[1] equal to 0
    tunovth = [tuning_curve.copy() for tuning_curve in tuning_curves]
    for o in range(len(opsins)):
        tunovth[o].values[tunovth[o].values < pctl_thrsh[0]] = 0
        tunovth[o].values[0] = 0  #set the 0 power level to 0
    cusum = [tuning_curve.cumsum() for tuning_curve in tunovth]
    cumulative_difference = cusum[0] - cusum[1]

    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax = axs[0]
    for o, opsin in enumerate(opsins):
        line = tuning_curves[o].plot.line(x='0', add_legend=False, ax=ax)[0]
        line.set_label(opsin)
        if opsin == 'chrimson':
            color = 'red'
        elif opsin == 'chr2':
            color = 'blue'
        else:
            print(f"Unknown opsin: {opsin}, defaulting to black")
            color = 'black'
        line.set_color(color)
        line.set_label(opsin)
    #add horizontal lines for pctl_thrsh
    tiles = ['null 95th %-tile', '99th %-tile']
    linestyles = ['--', '-.']
    for i, thrsh in enumerate(pctl_thrsh):
        ax.axhline(thrsh, color='green',
                   linestyle=linestyles[i],
                   label=tiles[i],
                   alpha=0.7)
    ax.set_xlabel('')
    ax.set_ylabel('Firing Rate (Hz)')
    ax.legend()
    ax.set_title('')

    ax = axs[1]

    colors = ['blue', 'red']
    for o, opsin in enumerate(opsins):
        line = cusum[o].plot.line(x='0', add_legend=False, ax=ax)[0]
        line.set_color(colors[o])
        line.set_label(f'Cumulative {opsin}')

    # plot the 5th and 95th percentiles as horizontal lines
    low_thrsh, high_thrsh = auc_diff_pctls
    ax.axhline(low_thrsh, color='orange', linestyle='--', label='shuff 5th Percentile', alpha=0.7)
    ax.axhline(high_thrsh, color='orange', linestyle='-.', label='shuff 95th Percentile', alpha=0.7)


    #now plot the cumulative difference
    line = cumulative_difference.plot.line(x='0', add_legend=False, ax=ax)[0]
    line.set_color('gray')
    line.set_label('Cumulative Difference')
    #set a dashed horizontal line at y=0
    ax.axhline(0, color='gray', linestyle='--')
    ax.legend()
    ax.set_ylabel('Firing Rate Difference (Hz)')
    ax.set_xlabel(r'Light Intensity ($\mu$W)')
    #remove any vestigal title
    ax.set_title('')

    plt.suptitle(f'Neuron {neuron} Opto Tagging Tuning Curves')
    plt.tight_layout()
    plt.savefig(tuning_curves_folder / f'neuron_{neuron}_opto_tagging_tuning_curves.png')
    plt.close()

def get_opto_epochs(bs, opsin):
    bs_metadata = bs.metadata
    events_idx = bs_metadata[bs_metadata['event'].str.contains(opsin)]
    events_idx = events_idx[~events_idx['event'].str.contains('trace')]
    events_on_idx = events_idx[events_idx['event'].str.contains('_on')]
    off_events_idx = events_idx[events_idx['event'].str.contains('_off')]

    on_events = bs[events_on_idx.index].to_tsd().t  # add 10 ms to account for delay
    off_events = bs[off_events_idx.index].to_tsd().t  # subtract 10 ms to account for delay

    prestim_start = on_events - 1.01
    prestim_end = on_events - 0.01

    opto_on_events = on_events + 0.005
    opto_off_events = off_events - 0.005
    opto_duration = np.round(np.mean(off_events - on_events),3)

    # concat iti_starts and opto_on_events
    opto_on_events = np.concatenate((opto_on_events, prestim_start))
    opto_off_events = np.concatenate((opto_off_events, prestim_end))
    # sort the timestamps
    sorted_indices = np.argsort(opto_on_events)
    opto_on_events = opto_on_events[sorted_indices]
    opto_off_events = opto_off_events[sorted_indices]

    opto_on_events_ts = nap.Ts(opto_on_events, time_units='s')
    opto_off_events_ts = nap.Ts(opto_off_events, time_units='s')
    opto_epochs = nap.IntervalSet(start=opto_on_events_ts.t, end=opto_off_events_ts.t)

    #double check that duration is not zero
    if opto_duration == 0:
        raise ValueError("Opto duration is zero, check opto events")
    return opto_epochs, opto_duration

def get_opto_trace(bs, opsin):
    bs_metadata = bs.metadata
    trace_events_df = bs_metadata[bs_metadata['event'].str.contains('trace')]
    idx = trace_events_df[trace_events_df['event'].str.contains(opsin)].index.tolist()
    trace = bs[idx[0]]
    return trace

def get_powers_and_bins(trace):
    powers = np.unique(trace.values)
    # get midpoints between power levels for binning
    mids = (powers[:-1] + powers[1:]) / 2
    lower = powers[0] - (powers[1] - powers[0]) / 2
    upper = powers[-1] + (powers[-1] - powers[-2]) / 2
    power_bins = np.concatenate(([lower], mids, [upper]))
    return powers, power_bins

def get_sampling_rate(trace):
    fs = int(1 / np.mean(trace.time_diff().values))
    return fs

def get_percentile_thresholds(ephys_data, avg_opto_duration):
    percentile_thresholds95 = Parallel(n_jobs=-1)(delayed(get_firing_rate_percentiles)(
        ephys_data[neuron], n_bins=30, n_iter=10000, bin_size=avg_opto_duration, conf_level=0.95) for neuron in tqdm(ephys_data.keys()))
    percentile_thresholds95 = np.array(percentile_thresholds95)[:,1]

    percentile_thresholds99 = Parallel(n_jobs=-1)(delayed(get_firing_rate_percentiles)(
        ephys_data[neuron], n_bins=30, n_iter=10000, bin_size=avg_opto_duration, conf_level=0.99) for neuron in tqdm(ephys_data.keys()))
    percentile_thresholds99 = np.array(percentile_thresholds99)[:,1]

    return percentile_thresholds95, percentile_thresholds99

def get_auc_diff_percentiles(ephys_data, tun, percentile_thresholds95, n_iter=10000):
    auc_diff_percentiles = []
    for n, neuron in enumerate(ephys_data.keys()):
        diffs = []
        for _ in range(n_iter):
            diffs.append(shuff_aucs_diff(tun[n], percentile_thresholds95[n]))
        #get the 5th and 95th percentiles
        diffs = np.array(diffs)
        low_pctl = np.percentile(diffs, 2.5)
        high_pctl = np.percentile(diffs, 97.5)
        auc_diff_percentiles.append((low_pctl, high_pctl))
    return auc_diff_percentiles

def add_tuning_curve_metadata(ephys_data, tuning_curves, opsins, percentile_thresholds95, percentile_thresholds99, aucs, auc_diff_percentiles):
    #get the neurons that have any part of their tuning curve above the 95th percentile
    metadata = ephys_data.metadata
    significant_95 = []
    significant_99 = []
    for o, opsin in enumerate(opsins):
        tc = tuning_curves[o]
        significant95 = tc.values > percentile_thresholds95[:][:, np.newaxis]
        significant95 = np.any(significant95[:,1:], axis=1)
        significant_95.append(significant95)
        significant99 = tc.values > percentile_thresholds99[:][:, np.newaxis]
        significant99 = np.any(significant99[:,1:], axis=1)
        significant_99.append(significant99)
        metadata[f'significant_{opsin}_95'] = significant95
        metadata[f'significant_{opsin}_99'] = significant99
        metadata[f'auc_{opsin}'] = aucs[o]

    metadata['auc_diff'] = metadata['auc_chr2'] - metadata['auc_chrimson']
    metadata['auc_diff_5th_percentile'] = [auc_diff_percentiles[n][0] for n in range(len(ephys_data.keys()))]
    metadata['auc_diff_95th_percentile'] = [auc_diff_percentiles[n][1] for n in range(len(ephys_data.keys()))]
    metadata['significant_auc_diff'] = (metadata['auc_diff'] <= metadata['auc_diff_5th_percentile']) | (metadata['auc_diff'] >= metadata['auc_diff_95th_percentile'])

    metadata['iSPN'] = (metadata['significant_chrimson_95'] & (metadata['auc_chr2'] < metadata['auc_chrimson']) & metadata['significant_auc_diff'])
    metadata['dSPN'] = (metadata['significant_chr2_95'] & (metadata['auc_chrimson'] < metadata['auc_chr2']) & metadata['significant_auc_diff'])

    #remove rate from metadata if it exists
    if 'rate' in metadata.columns:
        metadata = metadata.drop(columns=['rate'])
    return metadata


def add_ci_to_tuning_curve(tuning_curve, spikes, feature, epochs=None, n_bootstrap=10, ci=95):
    """
    Add bootstrapped confidence intervals to an existing tuning curve xarray.

    Parameters:
    -----------
    tuning_curve : xarray.DataArray
        Pre-computed tuning curve from nap.compute_tuning_curves
    spikes : nap.TsGroup
        Spike times for multiple units
    feature : nap.Tsd
        Feature trace (e.g., opto power)
    epochs : nap.IntervalSet or None
        Epochs to restrict analysis to
    n_bootstrap : int
        Number of bootstrap iterations per bin
    ci : float
        Confidence interval percentage

    Returns:
    --------
    tuning_curve : xarray.DataArray with added 'lower_ci' and 'upper_ci' coordinates
    """

    # Extract bin edges and other attributes
    bin_edges = tuning_curve.bin_edges[0]
    occupancy = tuning_curve.occupancy
    n_bins = len(bin_edges) - 1
    n_units = len(spikes)

    # Restrict feature to epochs if provided
    if epochs is not None:
        feature = feature.restrict(epochs)

    # Initialize CI arrays
    lower_bound = np.zeros((n_units, n_bins))
    upper_bound = np.zeros((n_units, n_bins))

    # For each unit
    for unit_idx, unit_id in enumerate(spikes.keys()):
        spike_times = spikes[unit_id].index

        # Restrict spikes to epochs if provided
        if epochs is not None:
            spike_times_epoch = spikes[unit_id].restrict(epochs).index
        else:
            spike_times_epoch = spike_times

        # For each bin
        for bin_idx in range(n_bins):
            # Find feature values in this bin
            bin_mask = (feature.values >= bin_edges[bin_idx]) & (feature.values < bin_edges[bin_idx + 1])

            if not np.any(bin_mask):
                lower_bound[unit_idx, bin_idx] = 0
                upper_bound[unit_idx, bin_idx] = 0
                continue

            # Get times when feature was in this bin
            bin_times = feature.index[bin_mask]

            # Total time in bin (in seconds)
            total_time = occupancy[bin_idx] / tuning_curve.fs

            if total_time == 0:
                lower_bound[unit_idx, bin_idx] = 0
                upper_bound[unit_idx, bin_idx] = 0
                continue

            # Count spikes in each feature timestamp within this bin
            dt = 1 / tuning_curve.fs / 2

            spike_counts = []
            for t in bin_times:
                n_spikes = np.sum((spike_times_epoch >= t - dt) & (spike_times_epoch < t + dt))
                spike_counts.append(n_spikes)

            spike_counts = np.array(spike_counts)

            # Bootstrap the firing rate
            bootstrap_rates = []
            for _ in range(n_bootstrap):
                resampled_counts = np.random.choice(spike_counts, size=len(spike_counts), replace=True)
                rate = np.sum(resampled_counts) / total_time
                bootstrap_rates.append(rate)

            # Calculate confidence intervals
            alpha = (100 - ci) / 2
            lower_bound[unit_idx, bin_idx] = np.percentile(bootstrap_rates, alpha)
            upper_bound[unit_idx, bin_idx] = np.percentile(bootstrap_rates, 100 - alpha)

    # Add CIs as new coordinates to the xarray
    tuning_curve = tuning_curve.assign_coords({
        'lower_ci': (tuning_curve.dims, lower_bound),
        'upper_ci': (tuning_curve.dims, upper_bound)
    })

    return tuning_curve

def plot_all_tuning_curves(ephys_data, bs, save_folder=None, overwrite=True, njobs=-1):
    # tuning curve approach
    # get indices of bs metadata containing the word 'trace'
    if save_folder is not None:
        opto_tagging_folder = save_folder / "opto_tagging"
        opto_tagging_folder.mkdir(exist_ok=True)
        tuning_curves_folder = opto_tagging_folder / "tuning_curves"
        tuning_curves_folder.mkdir(exist_ok=True)
    else:
        tuning_curves_folder = None


    opsins = ['chr2', 'chrimson']
    opto_durations = []
    tuning_curves = []
    ciLs = []
    ciHs = []
    n_bootstrap = 1000
    ci = 95
    for opsin in opsins:
        opto_epochs, duration = get_opto_epochs(bs, opsin)
        trace = get_opto_trace(bs, opsin)
        powers, bins = get_powers_and_bins(trace)
        fs = get_sampling_rate(trace)

        tuning_curve = nap.compute_tuning_curves(
            data=ephys_data,
            features=trace,
            bins=[bins],
            epochs=opto_epochs,
            fs=fs)

        #tuning_curve = add_ci_to_tuning_curve(tuning_curve,
        #    ephys_data, trace,
        #    epochs=opto_epochs, n_bootstrap=n_bootstrap, ci=ci
        #)

        tuning_curves.append(tuning_curve)
        opto_durations.append(duration)


    percentile_thresholds95, percentile_thresholds99 = get_percentile_thresholds(ephys_data, np.mean(opto_durations))

    aucs = []
    for o, opsin in enumerate(opsins):
        auc = [get_auc(tc, percentile_thresholds95[i]) for i, tc in enumerate(tuning_curves[o])]
        aucs.append(auc)
        print(f"Computed tuning curves and AUCs for opsin: {opsin}")

    #reverse packing of tuning_curves to have list of tuning curves per neuron
    tun = [*zip(*tuning_curves)]

    auc_diff_percentiles = get_auc_diff_percentiles(ephys_data, tun, percentile_thresholds95, n_iter=10000)

    metadata = add_tuning_curve_metadata(ephys_data, tuning_curves, opsins, percentile_thresholds95, percentile_thresholds99, aucs, auc_diff_percentiles)

    #combine percentile thresholds into a list of tuples
    percentile_thresholds = [(percentile_thresholds95[n], percentile_thresholds99[n]) for n in range(len(ephys_data.keys()))]
    Parallel(n_jobs=njobs)(delayed(plot_tuning_curves)(
        neuron, opsins, tun[n], percentile_thresholds[n], auc_diff_percentiles[n], tuning_curves_folder) for n, neuron in tqdm(enumerate(ephys_data.keys())))

    if overwrite:
        ephys_file = save_folder / "spikes.npz"
        ephys_data.set_info(metadata=metadata)
        ephys_data.save(str(ephys_file))

    return metadata

def test():
    base_folder = Path(r"C:\Users\assad\Documents\recording_files\DS13\DS13_20250905")
    pynapple_folder = fs.SessionPaths(base_folder).pynapple_folder
    bs = pni.get_binary_signals(base_folder, overwrite=False)
    ephys_file = pynapple_folder / "spikes.npz"
    ephys_data = nap.load_file(str(ephys_file))
    #edtest = nap.load_file(str(ephys_file))
    opto_tagging_folder = pynapple_folder / "opto_tagging"
    opto_tagging_folder.mkdir(exist_ok=True)

    bs_metadata = bs.metadata
    # get bs metadata row indices where binary_event_name contains 'chrimson_on' or 'chr2_on'
    opto_events_idx = bs_metadata[bs_metadata['event'].str.contains('chrimson|chr2')]
    opto_events_on_idx = opto_events_idx[opto_events_idx['event'].str.contains('_on')]
    # 'event' column entries have a number at the end, extract that number and convert to int
    opto_events_on_idx['power'] = opto_events_on_idx['event'].str.extract(r'(\d+)$')[0].astype(int).tolist()


    metadata = plot_all_tuning_curves(ephys_data, bs, opto_events_idx, opto_tagging_folder)
    #remove the 'rate' column from metadata if it exists
    if 'rate' in metadata.columns:
        metadata = metadata.drop(columns=['rate'])
    ephys_data.set_info(metadata = metadata)
    ephys_data.save(str(ephys_file))


#TODO: plot and compare waveform of evoked spikes and other spikes

#opto_events_idx = opto_events_idx.reset_index(drop=True)

#chr2_events_idx = opto_events_idx[opto_events_idx['event'].str.contains('chr2_on')]
#chrimson_events_idx = opto_events_idx[opto_events_idx['event'].str.contains('chrimson_on')]
#chrimson_events_idx = chrimson_events_idx.index.tolist()


def _peak_count(spike_times, lo, hi, bin_s):
    """Most spikes in any bin_s-long window inside [lo, hi) (spike_times sorted).

    The busiest window can always be taken to start on a spike, so it is the largest count of
    spikes in [s, s + bin_s) over the spikes s in the window; spikes past hi are not counted.
    """
    s = spike_times[np.searchsorted(spike_times, lo):np.searchsorted(spike_times, hi)]
    if s.size == 0:
        return 0
    return int((np.searchsorted(s, s + bin_s, 'left') - np.arange(s.size)).max())


def classify_opsin_response(ch_sig, c2_sig, ch_gt_c2):
    """Cell type from which opsins drive a unit, under the opsins' asymmetric selectivity.

    ChR2 is close to blind to orange light, so a dSPN has to show no orange-light response at
    all. Chrimson is somewhat blue-light sensitive, so an iSPN may also answer blue light, as
    long as it answers orange light more strongly. Anything else driven by both -- equal
    responses, or blue preferred alongside a real orange response -- is muSPN.

    ch_sig, c2_sig : the unit responds to that opsin's light
    ch_gt_c2       : its orange response is larger than its blue one
    """
    if ch_sig and c2_sig:
        return 'iSPN' if ch_gt_c2 else 'muSPN'
    if ch_sig:
        return 'iSPN'
    if c2_sig:
        return 'dSPN'
    return np.nan


def test_stim_firing(ephys_data, bs, opsins=None, response_window_ms=None, peak_bin_ms=10.0,
                     skip_ms=1.0, alpha=0.05, save_folder=None):
    """
    For each unit and opsin, test whether the light pulses drive firing, and label the cell
    type from which opsins do (classify_opsin_response).

    Response: per pulse, the peak firing rate over any *peak_bin_ms* window in the response
    window -- from *skip_ms* after onset (past the light-onset artifact) to the end of the pulse
    by default -- so a unit that adapts during the pulse is scored by its strongest moment, not
    diluted over the whole pulse.

    Baseline: the same statistic over a window of the same length ending *skip_ms* before the
    same pulse. Response and baseline are therefore measured identically, and each pulse is
    its own control (the comparison is paired, so slow changes in the unit's rate cancel).

    Tests, per unit:
      * per opsin, peak during the pulse > peak before it (one-sided Wilcoxon signed-rank)
      * orange response > blue response, the response being peak minus its own baseline peak
        (one-sided Mann-Whitney U), which decides a unit driven by both
      * dose-response: Spearman rho of the peak against power (baseline peaks at power 0),
        one-sided, rho > 0

    A unit responds to an opsin when both the first and the last test say so: the response
    has to rise with power. At low power the orange LED flickers and the flicker can induce a
    noise artifact, strongest at the lowest power and fading as the power goes up; requiring a
    rising dose-response keeps that from reading as an orange-light response, while a real
    response -- even one already saturated at the lowest power -- sits above the baseline at
    every power and passes.

    Parameters
    ----------
    ephys_data : nap.TsGroup-like
        Spike trains keyed by unit id.
    bs : object
        Binary-signals object with ``.metadata`` and ``bs[idx]`` access; pulses are the
        ``<opsin>_on_<power>`` events, their ends the matching ``<opsin>_off_<power>`` events.
    opsins : list of str or None
        Opsin prefixes to test (default ``['chrimson', 'chr2']``).
    response_window_ms : (start, end) or None
        Response window in ms after pulse onset. None: (*skip_ms*, the opsin's median pulse
        duration).
    peak_bin_ms : float
        Width of the window the peak rate is taken over (default 10 ms).
    skip_ms : float
        Time skipped after onset, and the gap between the baseline window and onset.
    alpha : float
        Significance threshold (default 0.05).
    save_folder : Path or None
        If provided, a bar-graph PNG is saved per unit into
        ``save_folder / stim_firing_plots``.

    Returns
    -------
    results_df : pandas.DataFrame
        One row per unit x opsin x power: ``['unit', 'opsin', 'power', 'n_trials',
        'rate_post_hz', 'rate_baseline_hz', 'p_value', 'significant']``, rates being mean peak
        rates.
    summary_df : pandas.DataFrame
        One row per unit: ``p_<opsin>_vs_baseline``, ``resp_<opsin>_hz`` (mean peak minus
        baseline peak), ``peak_rho_<opsin>`` / ``peak_rho_p_<opsin>`` (dose-response),
        ``responds_<opsin>`` (both significant), ``p_chrimson_gt_chr2``,
        ``window_ms_<opsin>``, ``tagged`` and ``cell_type``.
    """
    import re
    import pandas as pd
    from scipy.stats import mannwhitneyu, wilcoxon, spearmanr
    from pathlib import Path

    if opsins is None:
        opsins = ['chrimson', 'chr2']
    bs_metadata = bs.metadata
    bin_s = peak_bin_ms / 1000.0
    skip_s = skip_ms / 1000.0

    def _ts(idx):
        o = bs[idx]
        return np.asarray(o.t if hasattr(o, 't') else o.index, dtype=float)

    if save_folder is not None:
        plot_folder = Path(save_folder) / "stim_firing_plots"
        plot_folder.mkdir(parents=True, exist_ok=True)
    else:
        plot_folder = None

    # ---- per opsin: pulse onsets, their powers, and the response window ----
    opsin_info = {}
    events = dict(zip(bs_metadata['event'], bs_metadata.index))
    for opsin in opsins:
        onsets, powers, durations = [], [], []
        for name, idx in events.items():
            m = re.fullmatch(rf'{opsin}_on_(\d+)', str(name))
            if m is None:
                continue
            on = _ts(idx)
            onsets.append(on)
            powers.append(np.full(len(on), float(m.group(1))))
            off_idx = events.get(f'{opsin}_off_{m.group(1)}')
            if off_idx is not None and len(on):
                off = _ts(off_idx)
                nxt = np.searchsorted(off, on)
                ok = nxt < len(off)
                durations.append(off[nxt[ok]] - on[ok])
        if not onsets or not np.concatenate(onsets).size:
            continue
        on, pw = np.concatenate(onsets), np.concatenate(powers)
        order = np.argsort(on)
        if response_window_ms is not None:
            lo, hi = response_window_ms[0] / 1000.0, response_window_ms[1] / 1000.0
        else:
            dur = np.concatenate(durations) if durations else np.array([])
            if not dur.size:
                raise ValueError(f'no {opsin}_off events to take the pulse duration from; '
                                 'pass response_window_ms')
            lo, hi = skip_s, float(np.median(dur))
        opsin_info[opsin] = {'onsets': on[order], 'powers': pw[order], 'lo': lo, 'hi': hi}

    def _paired_greater(a, b):
        d = np.asarray(a, float) - np.asarray(b, float)
        if len(d) < 2 or not np.any(d):
            return np.nan if len(d) < 2 else 1.0
        try:
            return float(wilcoxon(a, b, alternative='greater', zero_method='zsplit').pvalue)
        except ValueError:
            return np.nan

    def _mwu_greater(a, b):
        if len(a) < 2 or len(b) < 2:
            return np.nan
        try:
            return float(mannwhitneyu(a, b, alternative='greater').pvalue)
        except ValueError:
            return np.nan

    records, summary_records = [], []
    for unit in ephys_data.keys():
        spike_times = np.sort(np.asarray(ephys_data[unit].index, dtype=float))
        row = {'unit': unit}
        resp = {}
        for opsin, info in opsin_info.items():
            lo, hi = info['lo'], info['hi']
            length = hi - lo
            stim = np.array([_peak_count(spike_times, t + lo, t + hi, bin_s)
                             for t in info['onsets']]) / bin_s
            base = np.array([_peak_count(spike_times, t - skip_s - length, t - skip_s, bin_s)
                             for t in info['onsets']]) / bin_s
            resp[opsin] = stim - base
            p = _paired_greater(stim, base)
            row[f'p_{opsin}_vs_baseline'] = p
            row[f'resp_{opsin}_hz'] = float(np.mean(stim - base))
            row[f'window_ms_{opsin}'] = (round(lo * 1000, 2), round(hi * 1000, 2))
            x = np.concatenate([np.zeros(len(base)), info['powers']])
            y = np.concatenate([base, stim])
            if len(np.unique(y)) > 1:
                rho, p_two = spearmanr(x, y)
                row[f'peak_rho_{opsin}'] = float(rho)
                row[f'peak_rho_p_{opsin}'] = float(p_two / 2 if rho > 0 else 1 - p_two / 2)
            else:
                row[f'peak_rho_{opsin}'], row[f'peak_rho_p_{opsin}'] = np.nan, np.nan
            for power in np.unique(info['powers']):
                sel = info['powers'] == power
                pp = _paired_greater(stim[sel], base[sel])
                records.append({'unit': unit, 'opsin': opsin, 'power': power,
                                'n_trials': int(sel.sum()),
                                'rate_post_hz': float(np.mean(stim[sel])),
                                'rate_baseline_hz': float(np.mean(base[sel])),
                                'p_value': pp,
                                'significant': bool(pp < alpha) if not np.isnan(pp) else False})
        for opsin in opsin_info:
            row[f'responds_{opsin}'] = bool(row[f'p_{opsin}_vs_baseline'] < alpha
                                            and row[f'peak_rho_p_{opsin}'] < alpha)
        p_gt = (_mwu_greater(resp['chrimson'], resp['chr2'])
                if 'chrimson' in resp and 'chr2' in resp else np.nan)
        row['p_chrimson_gt_chr2'] = p_gt
        cell_type = classify_opsin_response(row.get('responds_chrimson', False),
                                            row.get('responds_chr2', False), bool(p_gt < alpha))
        row['tagged'] = isinstance(cell_type, str)
        row['cell_type'] = cell_type
        summary_records.append(row)

    results_df = pd.DataFrame(records)
    summary_df = pd.DataFrame(summary_records)
    if len(results_df) > 0:
        _plot_stim_firing_bars(results_df, plot_folder, alpha)
    return results_df, summary_df


def _plot_stim_firing_bars(results_df, plot_folder, alpha=0.05):
    """
    For each unit, draw a grouped bar chart: baseline rate vs post-stim rate
    for every opsin × power combination.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output of ``test_stim_firing``.
    plot_folder : Path or None
        Where to save PNGs.  If None, ``plt.show()`` is used.
    alpha : float
        Significance level for annotating bars with an asterisk.
    """
    import matplotlib.pyplot as plt

    units = results_df['unit'].unique()
    for unit in units:
        udf = results_df[results_df['unit'] == unit].copy()
        udf = udf.sort_values(['opsin', 'power'])

        labels = [f"{row['opsin']}\n{row['power']} µW" for _, row in udf.iterrows()]
        baseline_vals = udf['rate_baseline_hz'].values
        post_vals = udf['rate_post_hz'].values
        sig_flags = udf['significant'].values

        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.4), 5))
        bars_bl = ax.bar(x - width / 2, baseline_vals, width, label='Baseline (peak 10 ms rate)', color='grey', alpha=0.7)
        bars_po = ax.bar(x + width / 2, post_vals, width, label='Pulse (peak 10 ms rate)', color='steelblue')

        # annotate significant comparisons
        y_max = max(np.max(baseline_vals), np.max(post_vals)) if len(baseline_vals) > 0 else 1
        for i, sig in enumerate(sig_flags):
            if sig:
                bar_top = max(baseline_vals[i], post_vals[i])
                ax.text(x[i], bar_top + y_max * 0.02, '*', ha='center', va='bottom',
                        fontsize=14, fontweight='bold', color='red')

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel('Firing rate (Hz)')
        ax.set_title(f'Unit {unit} – Baseline vs Evoked Firing Rate')
        ax.legend()
        fig.tight_layout()

        if plot_folder is not None:
            fig.savefig(plot_folder / f'Unit_{unit}_stim_firing_bar.png', dpi=150)
            plt.close(fig)
        else:
            plt.show()

def plot_stim_firing_bars(ephys_data, bs, save_folder=None, alpha=0.05,
                          post_window_ms=(1.0, 31.0),
                          baseline_duration_s=1.0, baseline_gap_ms=1.0,
                          baseline_bin_ms=10.0):
    """Per-unit bar chart: baseline vs per-power evoked firing rate.

    One subplot per opsin, each 0.8 × 0.8 in.  Orange bars = chrimson,
    blue bars = chr2 (shade varies by power level).  Grey bar = pooled
    baseline.  Error bars are 95 % CI of the mean.  All text 8 pt.
    Saves SVG + PNG under save_folder/stim_firing_plots_v2/.
    """
    import re
    from utils import plot_size_utils as psu

    opsins = ['chrimson', 'chr2']
    _CMAPS  = {'chrimson': plt.cm.Oranges, 'chr2': plt.cm.Blues}
    _LABELS = {'chrimson': 'amber', 'chr2': 'blue'}

    post_start_s   = post_window_ms[0] / 1000.0
    post_end_s     = post_window_ms[1] / 1000.0
    post_bin_s     = post_end_s - post_start_s
    baseline_bin_s = baseline_bin_ms  / 1000.0
    baseline_gap_s = baseline_gap_ms  / 1000.0

    def _extr_power(s):
        m = re.search(r"(\d+)$", str(s))
        return int(m.group(1)) if m else None

    def _ci95(arr):
        arr = np.asarray(arr, dtype=float)
        if len(arr) < 2:
            return 0.0
        return 1.96 * arr.std() / np.sqrt(len(arr))

    bs_metadata = bs.metadata

    # collect per-opsin onset times grouped by power
    opsin_info = {}
    for opsin in opsins:
        on_mask = bs_metadata['event'].str.contains(f'{opsin}_on', na=False)
        on_df = bs_metadata[on_mask].copy()
        if on_df.empty:
            continue
        on_df['power'] = on_df['event'].apply(_extr_power)
        on_df = on_df.dropna(subset=['power'])

        power_groups = {}
        all_onsets, all_powers = [], []
        for idx in on_df.index:
            pw = int(on_df.loc[idx, 'power'])
            ts_obj = bs[idx]
            onsets = ts_obj.t.tolist() if hasattr(ts_obj, 't') else ts_obj.index.tolist()
            power_groups.setdefault(pw, []).extend(onsets)
            all_onsets.extend(onsets)
            all_powers.extend([float(pw)] * len(onsets))

        if not all_onsets:
            continue
        sort_idx = np.argsort(all_onsets)
        opsin_info[opsin] = {
            'all_onsets':   np.array(all_onsets)[sort_idx],
            'power_groups': {k: np.array(v) for k, v in power_groups.items()},
        }

    if not opsin_info:
        print("No opsin events found; skipping plot_stim_firing_bars.")
        return

    if save_folder is not None:
        out_folder = Path(save_folder) / 'stim_firing_plots_v2'
        out_folder.mkdir(parents=True, exist_ok=True)
    else:
        out_folder = None

    units = list(ephys_data.keys())
    present_opsins = [op for op in opsins if op in opsin_info]
    n_cols = len(present_opsins)

    for unit in units:
        spike_times = np.asarray(ephys_data[unit].index)

        fig, axs = plt.subplots(1, n_cols, squeeze=False)

        for ci, opsin in enumerate(present_opsins):
            info          = opsin_info[opsin]
            all_onsets    = info['all_onsets']
            power_groups  = info['power_groups']
            sorted_powers = sorted(power_groups)
            n_pw          = max(len(sorted_powers) - 1, 1)
            cmap          = _CMAPS[opsin]
            ax            = axs[0, ci]

            # pooled baseline for this opsin
            n_bins = int(baseline_duration_s / baseline_bin_s)
            bl_rates = []
            for onset in all_onsets:
                bl_end   = onset - baseline_gap_s
                bl_start = bl_end - baseline_duration_s
                for b in range(n_bins):
                    lo = bl_start + b * baseline_bin_s
                    hi = lo + baseline_bin_s
                    bl_rates.append(
                        int(np.sum((spike_times >= lo) & (spike_times < hi)))
                        / baseline_bin_s
                    )
            bl_rates = np.array(bl_rates)
            bl_mean  = float(bl_rates.mean()) if len(bl_rates) else 0.0
            bl_ci    = _ci95(bl_rates)

            xs      = [0]
            means   = [bl_mean]
            cis     = [bl_ci]
            colors  = ['grey']
            xlabels = ['BL']

            for pi, pw in enumerate(sorted_powers):
                onsets_pw  = power_groups[pw]
                post_rates = np.array([
                    int(np.sum((spike_times >= (on + post_start_s)) &
                               (spike_times <  (on + post_end_s))))
                    / post_bin_s
                    for on in onsets_pw
                ], dtype=float)

                xs.append(pi + 1)
                means.append(float(post_rates.mean()) if len(post_rates) else 0.0)
                cis.append(_ci95(post_rates))
                colors.append(cmap(0.45 + 0.55 * pi / n_pw))
                xlabels.append(f'{pw} µW')

            xs    = np.array(xs)
            means = np.array(means)
            cis   = np.array(cis)

            ax.bar(xs, means, color=colors, width=0.7)
            ax.errorbar(xs, means, yerr=cis,
                        fmt='none', color='black', linewidth=0.8,
                        capsize=2, capthick=0.8)

            ax.set_xticks(xs)
            ax.set_xticklabels(xlabels, fontsize=8, rotation=45, ha='right')
            ax.set_ylabel('Firing rate (Hz)', fontsize=8)
            ax.tick_params(labelsize=8)
            ax.set_title(_LABELS[opsin], fontsize=8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        psu.adjust_figure_for_panel_size_hetero(fig, panel_width=0.8, panel_height=0.8)

        if out_folder is not None:
            fig.savefig(out_folder / f'Unit_{unit}_stim_firing_bar.svg', bbox_inches='tight')
            fig.savefig(out_folder / f'Unit_{unit}_stim_firing_bar.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()


def plot_time_to_first_spike_distribution(ephys_data, bs, save_folder=None, max_latency_ms=100, alpha=0.05):
    """
    Plot the distribution of time to first spike (TTFS) relative to optogenetic stimulus onset
    for each neuron and opsin. The TTFS is calculated as the latency between the stimulus
    onset and the first spike after the onset.

    Parameters
    ----------
    ephys_data : nap.TsGroup-like
        Spike trains keyed by unit id.
    bs : object
        Binary-signals object with ``.metadata`` and ``bs[idx]`` access.
    save_folder : Path or None
        If provided, a summary PNG is saved to ``save_folder / ttfs_summary.png``.
    max_latency_ms : float
        Maximum latency for considering a spike (default 100 ms).
    alpha : float
        Significance threshold used for ``cell_type`` labelling (default 0.05).

    Returns
    -------
    df : pandas.DataFrame
        Per-trial TTFS data with columns:
        ``['unit', 'opsin', 'power', 'onset_s', 'ttfs_ms', 'prestim_ttfs_ms', 'responded']``.
        Trials where no spike occurred within ``max_latency_ms`` are censored
        (``responded=False``, ``ttfs_ms=max_latency_ms``).
    cox_df : pandas.DataFrame
        Per-unit Cox Proportional Hazards results with columns:
        ``['unit', 'n_obs', 'n_events', 'concordance_index',
           'coef_power_chrimson', 'hr_power_chrimson', 'p_power_chrimson',
           'coef_power_chr2',     'hr_power_chr2',     'p_power_chr2',
           'lrt_stat', 'lrt_pval',
           'delta_coef', 'dominant_opsin', 'cell_type']``.
        ``hr`` is the hazard ratio exp(coef); higher means spikes earlier
        per unit increase in power.  Two nested models are compared:
        M1 (separate slopes per opsin) vs M0 (shared slope); the LRT
        p-value tests whether the two dose-response slopes are
        significantly different.  ``delta_coef`` = coef_chrimson −
        coef_chr2 (positive = chrimson steeper).  ``dominant_opsin`` is
        set only when LRT is significant.  ``cell_type`` labels neurons
        hierarchically: 'iSPN' (chrimson-specific), 'dSPN' (chr2-
        specific), 'muSPN' (both significant but indistinguishable), or
        NaN (not tagged).  All numeric columns are NaN if fitting fails.
    ttfs_folder : Path
        Path to the folder containing saved unit figures.
    """
    import re
    import pandas as pd
    from pathlib import Path

    if save_folder is None:
        save_folder = Path.cwd()
    else:
        save_folder = Path(save_folder)
    ttfs_folder = save_folder / 'opto_tagging' / 'ttfs'
    ttfs_folder.mkdir(parents=True, exist_ok=True)

    bs_metadata = bs.metadata

    records = []
    # for each opsin, find all on events and extract spike times in the post-stimulus window
    for opsin in ['chrimson', 'chr2']:
        # find rows corresponding to opsin on events
        mask = bs_metadata['event'].str.contains(f'{opsin}_on', na=False)
        events_idx = bs_metadata[mask]
        if events_idx.shape[0] == 0:
            continue

        # extract power from the event string (expects integer at the end e.g. 'chr2_on_20')
        def _extract_power(s):
            m = re.search(r"(\d+)$", str(s))
            return int(m.group(1)) if m else np.nan

        events_idx = events_idx.copy()
        events_idx['power'] = events_idx['event'].apply(_extract_power)

        for idx, row in events_idx.iterrows():
            ts_obj = bs[idx]
            onsets = ts_obj.t if hasattr(ts_obj, 't') else ts_obj.index
            power = row['power']
            for onset in onsets:
                for unit in ephys_data.keys():
                    spike_times = ephys_data[unit].index
                    # relative spike times (s)
                    rel = spike_times - onset
                    max_lat_s = float(max_latency_ms) / 1000.0
                    mask_spikes = (rel > 0) & (rel <= max_lat_s)
                    if np.any(mask_spikes):
                        latency_s = np.min(rel[mask_spikes])
                        ttfs_ms = float(latency_s * 1000.0)
                        responded = True
                    else:
                        ttfs_ms = float(max_latency_ms)
                        responded = False
                        
                    # Pre-stimulus control: first spike in the 100 ms window before onset.
                    # Trials with no spike are right-censored at max_latency_ms.
                    rel_null = spike_times - (onset - float(max_latency_ms) / 1000.0)
                    mask_null = (rel_null > 0) & (rel_null <= float(max_latency_ms) / 1000.0)
                    prestim_responded = bool(np.any(mask_null))
                    if prestim_responded:
                        prestim_ttfs_ms = float(np.min(rel_null[mask_null]) * 1000.0)
                    else:
                        prestim_ttfs_ms = float(max_latency_ms)

                    records.append({
                        'unit': unit,
                        'opsin': opsin,
                        'power': power,
                        'onset_s': float(onset),
                        'ttfs_ms': ttfs_ms,
                        'responded': responded,
                        'prestim_ttfs_ms': prestim_ttfs_ms,
                        'prestim_responded': prestim_responded,
                    })

    df = pd.DataFrame(records)

    # ---- Cox Proportional Hazards per unit ----
    # Two nested models compared by likelihood ratio test (LRT):
    #
    #   M1 (full):  separate power slopes per opsin
    #               power_chrimson = power × I(chrimson)
    #               power_chr2     = power × I(chr2)
    #               — orthogonal predictors, always identifiable
    #
    #   M0 (null):  single shared power slope
    #               power (same coefficient for both opsins)
    #
    # LRT stat = 2 × (loglik_M1 − loglik_M0) ~ χ²(df=1)
    # Tests whether the two dose-response slopes are significantly different.
    from lifelines import CoxPHFitter, KaplanMeierFitter
    from scipy.stats import chi2 as _chi2

    cox_records = []
    units = sorted(df['unit'].dropna().unique())
    for unit in units:
        unit_df = df[df['unit'] == unit].copy()
        unit_df['opsin_encoded'] = (unit_df['opsin'] == 'chrimson').astype(int)
        unit_df['event'] = unit_df['responded'].astype(int)
        unit_df['power_chrimson'] = unit_df['power'] * unit_df['opsin_encoded']
        unit_df['power_chr2']     = unit_df['power'] * (1 - unit_df['opsin_encoded'])
        stim_df = unit_df[['ttfs_ms', 'event', 'power',
                            'power_chrimson', 'power_chr2']].dropna()

        # Pre-stim rows at power=0 — both per-opsin slopes are 0 here,
        # so they anchor the baseline without affecting slope estimation.
        prestim_df = pd.DataFrame({
            'ttfs_ms':        unit_df['prestim_ttfs_ms'].values,
            'event':          unit_df['prestim_responded'].astype(int).values,
            'power':          0.0,
            'power_chrimson': 0.0,
            'power_chr2':     0.0,
        }).dropna()
        fit_df = pd.concat([stim_df, prestim_df], ignore_index=True)

        n_obs    = len(fit_df)
        n_events = int(fit_df['event'].sum())

        coef_ch = hr_ch = p_ch = np.nan
        coef_c2 = hr_c2 = p_c2 = np.nan
        lrt_stat = lrt_pval = np.nan
        ci = np.nan

        if n_events >= 2:
            try:
                # Full model — separate slopes
                cph_full = CoxPHFitter(penalizer=0.1)
                cph_full.fit(fit_df[['ttfs_ms', 'event', 'power_chrimson', 'power_chr2']],
                             duration_col='ttfs_ms', event_col='event')
                s = cph_full.summary
                if 'power_chrimson' in s.index:
                    coef_ch = float(s.loc['power_chrimson', 'coef'])
                    hr_ch   = float(s.loc['power_chrimson', 'exp(coef)'])
                    p_ch    = float(s.loc['power_chrimson', 'p'])
                if 'power_chr2' in s.index:
                    coef_c2 = float(s.loc['power_chr2', 'coef'])
                    hr_c2   = float(s.loc['power_chr2', 'exp(coef)'])
                    p_c2    = float(s.loc['power_chr2', 'p'])
                ci      = float(cph_full.concordance_index_)
                ll_full = float(cph_full.log_likelihood_)

                # Null model — shared slope
                cph_null = CoxPHFitter(penalizer=0.1)
                cph_null.fit(fit_df[['ttfs_ms', 'event', 'power']],
                             duration_col='ttfs_ms', event_col='event')
                ll_null  = float(cph_null.log_likelihood_)

                lrt_stat = float(max(0.0, 2.0 * (ll_full - ll_null)))
                lrt_pval = float(_chi2.sf(lrt_stat, df=1))
            except Exception:
                pass

        cox_records.append({
            'unit': unit,
            'n_obs': n_obs,
            'n_events': n_events,
            'concordance_index': ci,
            'coef_power_chrimson': coef_ch,
            'hr_power_chrimson':   hr_ch,
            'p_power_chrimson':    p_ch,
            'coef_power_chr2':     coef_c2,
            'hr_power_chr2':       hr_c2,
            'p_power_chr2':        p_c2,
            'lrt_stat':            lrt_stat,
            'lrt_pval':            lrt_pval,
        })

    cox_df = pd.DataFrame(cox_records)

    # ---- derived summary columns ----
    cox_df['delta_coef'] = cox_df['coef_power_chrimson'] - cox_df['coef_power_chr2']

    ch_sig  = cox_df['p_power_chrimson'] < alpha
    c2_sig  = cox_df['p_power_chr2']     < alpha
    lrt_sig = cox_df['lrt_pval']         < alpha

    # dominant_opsin: which opsin has the steeper slope, but only reported
    # when the LRT confirms the difference is significant.
    cox_df['dominant_opsin'] = np.where(
        lrt_sig,
        np.where(cox_df['delta_coef'] > 0, 'chrimson', 'chr2'),
        None,
    )

    # cell_type: hierarchical labelling
    #   (an opsin counts only with a significant POSITIVE coefficient: faster spiking)
    #   1. one opsin significant, other not → label by the significant one
    #   2. both significant: 'iSPN' if the LRT finds the chrimson slope steeper, otherwise
    #      'muSPN' (see classify_opsin_response for the opsins' asymmetric selectivity)
    #   4. neither significant → NaN (not tagged)
    def _cell_type(row):
        # a tagging response makes the unit fire sooner as the power goes up (a positive
        # coefficient); a significant negative one is light delaying its spikes -- inhibition
        ch  = row['p_power_chrimson'] < alpha and row['coef_power_chrimson'] > 0
        c2  = row['p_power_chr2']     < alpha and row['coef_power_chr2']     > 0
        lrt = row['lrt_pval']         < alpha
        # driven by both: orange must win for iSPN (Chrimson answers some blue light); ChR2
        # does not answer orange light, so blue winning is no dSPN
        return classify_opsin_response(bool(ch), bool(c2), bool(lrt and row['delta_coef'] > 0))

    cox_df['cell_type'] = cox_df.apply(_cell_type, axis=1)

    # ---- per-unit TTFS plots (Kaplan-Meier cumulative incidence) ----
    # Censored trials (no spike) contribute to the risk set but do not pull
    # the curve to 1, giving an unbiased estimate of spiking probability.
    opsins = ['chrimson', 'chr2']
    all_powers = sorted(df['power'].dropna().unique())
    n_opsins = len(opsins)
    n_powers = max(len(all_powers), 1)

    cox_lookup = cox_df.set_index('unit') if len(cox_df) > 0 else None

    for unit in units:
        fig, axs = plt.subplots(nrows=n_powers, ncols=n_opsins,
                                figsize=(5 * n_opsins, 3 * n_powers),
                                sharex=True, sharey=True)
        if n_powers == 1 and n_opsins == 1:
            axs = np.array([[axs]])
        elif n_powers == 1:
            axs = axs[np.newaxis, :]
        elif n_opsins == 1:
            axs = axs[:, np.newaxis]

        unit_df = df[df['unit'] == unit]

        # Pre-stim KM fit (pooled across all opsins/powers for this unit)
        prestim_dur = unit_df['prestim_ttfs_ms'].values.astype(float)
        prestim_evt = unit_df['prestim_responded'].values.astype(bool)
        kmf_null = None
        if len(prestim_dur) > 0:
            kmf_null = KaplanMeierFitter()
            kmf_null.fit(prestim_dur, event_observed=prestim_evt)

        for col, op in enumerate(opsins):
            op_df = unit_df[unit_df['opsin'] == op]
            for row_idx, power in enumerate(all_powers):
                ax = axs[row_idx, col]
                power_df = op_df[op_df['power'] == power]
                dur = power_df['ttfs_ms'].values.astype(float)
                evt = power_df['responded'].values.astype(bool)
                pow_label = str(int(power)) if power == int(power) else str(power)

                if kmf_null is not None:
                    t_null = kmf_null.timeline
                    ci_null = 1.0 - kmf_null.survival_function_.values.flatten()
                    ax.step(t_null, ci_null, color='gray', linewidth=2,
                            label='Pre-stim', alpha=0.8, where='post')

                if len(dur) > 0:
                    kmf_op = KaplanMeierFitter()
                    kmf_op.fit(dur, event_observed=evt)
                    t_op = kmf_op.timeline
                    ci_op = 1.0 - kmf_op.survival_function_.values.flatten()
                    ax.step(t_op, ci_op, color='C0', linewidth=2,
                            label=op, alpha=0.8, where='post')
                    ax.set_title(f'{op} @ {pow_label} \u00b5W', fontsize=8)
                else:
                    ax.set_title(f'{op} @ {pow_label} \u00b5W (No Data)', fontsize=8)

                ax.set_xlim(0, max_latency_ms)
                ax.set_ylim(0, 1)
                if row_idx == n_powers - 1:
                    ax.set_xlabel('Time to first spike (ms)')
                if col == 0:
                    ax.set_ylabel('Cumulative incidence')
                if kmf_null is not None or len(dur) > 0:
                    ax.legend(fontsize=8)

        # annotate suptitle with Cox results for this unit
        cox_note = ''
        if cox_lookup is not None and unit in cox_lookup.index:
            crow = cox_lookup.loc[unit]
            p_ch   = float(crow['p_power_chrimson'])
            hr_ch  = float(crow['hr_power_chrimson'])
            p_c2   = float(crow['p_power_chr2'])
            hr_c2  = float(crow['hr_power_chr2'])
            p_lrt  = float(crow['lrt_pval'])
            parts = []
            if not np.isnan(p_ch):
                parts.append(f'chrimson: HR={hr_ch:.2f}, p={p_ch:.3g}')
            if not np.isnan(p_c2):
                parts.append(f'chr2: HR={hr_c2:.2f}, p={p_c2:.3g}')
            if not np.isnan(p_lrt):
                parts.append(f'LRT p={p_lrt:.3g}')
            if parts:
                cox_note = '\nCox PH \u2014 ' + '  |  '.join(parts)

        fig.suptitle(f'Unit {unit} \u2014 TTFS Cumulative Incidence{cox_note}', fontsize=9)
        plt.tight_layout()
        plt.savefig(ttfs_folder / f'Unit_{unit}_ttfs.png')
        plt.close(fig)

    return df, cox_df, ttfs_folder


def plot_ttfs_cdf_single(ephys_data, bs, save_folder=None, max_latency_ms=100):
    """Single-panel CDF variant of plot_time_to_first_spike_distribution.

    All per-power CDF curves are drawn on one 0.8 × 0.8 in axes per unit.
    Chrimson curves are shades of orange, chr2 curves are shades of blue
    (darker = higher power). The pooled pre-stimulus shuffle is a single
    grey step curve. No legend. All text 8 pt.
    Saves SVG + PNG under save_folder/opto_tagging/ttfs_single/.
    """
    import re
    import pandas as pd
    from lifelines import KaplanMeierFitter
    from utils import plot_size_utils as psu

    if save_folder is None:
        save_folder = Path.cwd()
    else:
        save_folder = Path(save_folder)
    out_folder = save_folder / 'opto_tagging' / 'ttfs_single'
    out_folder.mkdir(parents=True, exist_ok=True)

    bs_metadata = bs.metadata
    records = []

    def _extract_power(s):
        m = re.search(r"(\d+)$", str(s))
        return int(m.group(1)) if m else np.nan

    for opsin in ['chrimson', 'chr2']:
        mask = bs_metadata['event'].str.contains(f'{opsin}_on', na=False)
        events_idx = bs_metadata[mask].copy()
        if events_idx.empty:
            continue
        events_idx['power'] = events_idx['event'].apply(_extract_power)
        max_lat_s = float(max_latency_ms) / 1000.0

        for idx, row in events_idx.iterrows():
            ts_obj = bs[idx]
            onsets = ts_obj.t if hasattr(ts_obj, 't') else ts_obj.index
            power = row['power']
            for onset in onsets:
                for unit in ephys_data.keys():
                    spike_times = ephys_data[unit].index
                    rel = spike_times - onset
                    mask_post = (rel > 0) & (rel <= max_lat_s)
                    responded = bool(np.any(mask_post))
                    ttfs_ms = float(np.min(rel[mask_post]) * 1000.0) if responded else float(max_latency_ms)

                    rel_null = spike_times - (onset - max_lat_s)
                    mask_null = (rel_null > 0) & (rel_null <= max_lat_s)
                    prestim_responded = bool(np.any(mask_null))
                    prestim_ttfs_ms = (float(np.min(rel_null[mask_null]) * 1000.0)
                                       if prestim_responded else float(max_latency_ms))

                    records.append({
                        'unit': unit, 'opsin': opsin, 'power': power,
                        'ttfs_ms': ttfs_ms, 'responded': responded,
                        'prestim_ttfs_ms': prestim_ttfs_ms,
                        'prestim_responded': prestim_responded,
                    })

    df = pd.DataFrame(records)
    if df.empty:
        print("No TTFS data found; skipping.")
        return

    _CMAPS  = {'chrimson': plt.cm.Oranges, 'chr2': plt.cm.Blues}

    for unit in sorted(df['unit'].dropna().unique()):
        unit_df = df[df['unit'] == unit]

        fig, ax = plt.subplots(1, 1)

        # pre-stim shuffle — pooled across all opsins/powers, one grey curve
        prestim_dur = unit_df['prestim_ttfs_ms'].values.astype(float)
        prestim_evt = unit_df['prestim_responded'].values.astype(bool)
        if len(prestim_dur) > 0:
            kmf_null = KaplanMeierFitter()
            kmf_null.fit(prestim_dur, event_observed=prestim_evt)
            ci_null = 1.0 - kmf_null.survival_function_.values.flatten()
            ax.step(kmf_null.timeline, ci_null,
                    color='grey', linewidth=1.0, alpha=0.8, where='post')

        # per-opsin × per-power curves
        for opsin in ['chrimson', 'chr2']:
            op_df = unit_df[unit_df['opsin'] == opsin]
            if op_df.empty:
                continue
            cmap = _CMAPS[opsin]
            op_powers = sorted(op_df['power'].dropna().unique())
            n_op = max(len(op_powers) - 1, 1)
            for pi, power in enumerate(op_powers):
                dur = op_df.loc[op_df['power'] == power, 'ttfs_ms'].values.astype(float)
                evt = op_df.loc[op_df['power'] == power, 'responded'].values.astype(bool)
                if len(dur) == 0:
                    continue
                color = cmap(0.45 + 0.55 * pi / n_op)
                kmf = KaplanMeierFitter()
                kmf.fit(dur, event_observed=evt)
                ci_op = 1.0 - kmf.survival_function_.values.flatten()
                ax.step(kmf.timeline, ci_op,
                        color=color, linewidth=1.0, alpha=0.9, where='post')

        ax.set_xlim(0, max_latency_ms)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Time to first spike (ms)', fontsize=8)
        ax.set_ylabel('Cumulative incidence', fontsize=8)
        ax.tick_params(labelsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        psu.adjust_figure_for_panel_size_auto(fig, panel_width=0.8, panel_height=0.8)

        fig.savefig(out_folder / f'Unit_{unit}_ttfs_single.svg', bbox_inches='tight')
        fig.savefig(out_folder / f'Unit_{unit}_ttfs_single.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
