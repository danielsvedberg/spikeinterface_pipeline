import pynapple as nap
from pathlib import Path
import parse_nidq as pni
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy import stats
import numpy as np
import parse_nidq as pni

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
    #base_folder = Path("C:\\Users\\assad\\Documents\\analysis_files\\DS13\\DS13_20250822")
    base_folder = Path("C:\\Users\\assad\\Documents\\analysis_files\\DS13\\DS13_20250905")
    #base_folder = Path(r"C:\Users\assad\Documents\analysis_files\DS13\DS13_20250824")
    pynapple_folder = base_folder / "pynapple"
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
