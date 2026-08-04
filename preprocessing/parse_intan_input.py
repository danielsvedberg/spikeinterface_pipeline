import numpy as np
import spikeinterface.full as si
import pynapple as nap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
import scipy
import pandas as pd
from scipy.signal import find_peaks
from pathlib import Path
from sklearn.mixture import GaussianMixture
import ast
import os
from spikeinterface.extractors import read_split_intan_files
import spikeinterface.preprocessing as spre
from pathlib import Path
from utils import file_structure as fs

default_dinmap = {
    'licks': 1,
    'solenoid': 2,
    'cue': 3,
}

default_adcmap = {
    'blue_1': 0,
    'blue_2': 1,
    'orange_1': 2,
    'orange_2': 3,
}

default_optoidx = {
    'orange': [50,100,500,1000],
    'blue': [50,100,500,1000]
}

#maps physical LED color to the opsin it drives, so opto events are labeled by opsin
#(chr2/chrimson) for cross-compatibility with the nidq pipeline and later analyses
default_opsin_map = {'blue': 'chr2', 'orange': 'chrimson'}

def binarize_trace(trace, threshold=None):
    if threshold is None:
        threshold = (np.max(trace) + np.min(trace)) / 2
    digital = trace > threshold
    digital = digital.astype(int)
    return digital

def get_edge_times(digital_trace, time_array, edge_type='rising', return_idxs=False):
    if edge_type == 'rising':
        edge_idxs = np.where((np.diff(digital_trace) == 1))[0]
    elif edge_type == 'falling':
        edge_idxs = np.where((np.diff(digital_trace) == -1))[0]
    else:
        raise ValueError("edge_type must be 'rising' or 'falling'")
    edge_times = time_array[edge_idxs + 1]  # +1 because diff reduces length by 1
    if return_idxs:
        return edge_times, edge_idxs + 1
    else:
        return edge_times

def get_channel_index(label, idx, digchids):
    """Get array index for a channel label.

    Args:
        label: event label string (e.g., 'licks')
        idx: dict mapping labels to port numbers (e.g., {'licks': 1, 'solenoid': 2})
        digchids: array/list of channel IDs (e.g., ['DIGITAL-IN-01', 'DIGITAL-IN-02', ...])

    Returns:
        int: index of the channel in digchids

    Example:
        idx = {'licks': 1, 'solenoid': 2}
        digchids = ['DIGITAL-IN-01', 'DIGITAL-IN-02', 'DIGITAL-IN-03']
        get_channel_index('licks', idx, digchids) -> 0
    """
    if label not in idx:
        raise ValueError(f"Label '{label}' not found in idx. Available: {list(idx.keys())}")

    port_num = idx[label]
    matching_chids = [i for i, chid in enumerate(digchids)
                     if str(port_num).zfill(2) in chid]

    if not matching_chids:
        raise ValueError(f"No channel found for {label} (port {port_num}) in {list(digchids)}")
    if len(matching_chids) > 1:
        raise ValueError(f"Multiple channels found for {label} (port {port_num}): {[digchids[i] for i in matching_chids]}")

    return matching_chids[0]

def parse_tones_digital(dig_in, solenoid_ts_list, time_support, idx={'cue':3}):
    time_support = nap.IntervalSet(start=time_support.start, end=time_support.end)
    solenoid_onsets = solenoid_ts_list[0].t

    digchids = dig_in.channel_ids
    tone_din = get_channel_index('cue', idx, digchids)

    times = dig_in.get_times()
    dig_trace = dig_in.get_traces(channel_ids=[digchids[tone_din]]).squeeze().astype(np.int32)
    tone_onsets = get_edge_times(dig_trace, times, edge_type='rising')
    tone_offsets = get_edge_times(dig_trace, times, edge_type='falling')

    return parse_tones(tone_onsets, tone_offsets, time_support)

def parse_tones(tone_onsets, tone_offsets, time_support):
    tone_onsets.sort()
    tone_offsets.sort()

    inter_onset_intervals = np.diff(tone_onsets)
    inter_offset_intervals = np.diff(tone_offsets)

    #append infinity to the end of inter_onset_intervals and inter_offset_intervals to make them the same length as tone_onsets and tone_offsets
    inter_onset_intervals = np.append(np.inf, inter_onset_intervals)
    inter_offset_intervals = np.append(inter_offset_intervals, np.inf)

    #keep only tone onsets preceded by an interval of at least 0.001 seconds, and keep only tone offsets followed by an interval of at least 0.01
    min_isi = 0.009  # minimum inter-stimulus interval in seconds
    valid_onset_intervals = inter_onset_intervals >= min_isi
    valid_offset_intervals = inter_offset_intervals >= min_isi

    valid_tone_onsets = tone_onsets[valid_onset_intervals]
    valid_tone_offsets = tone_offsets[valid_offset_intervals]


    #require that the number of valid tone onsets and offsets are the same
    if len(valid_tone_onsets) != len(valid_tone_offsets):
        raise ValueError("Number of valid tone onsets does not match number of valid tone offsets")
    valid_tone_durations = valid_tone_offsets - valid_tone_onsets
    #make sure tone duration are at least 0.05s. anything less should be discarded
    valid_dur_idxs = valid_tone_durations > 0.0000001#> 0.05
    valid_tone_durations = valid_tone_durations[valid_dur_idxs]
    valid_tone_onsets = valid_tone_onsets[valid_dur_idxs]
    valid_tone_offsets = valid_tone_offsets[valid_dur_idxs]

    cue_tone_bool = np.round(valid_tone_durations,1) <= 0.1
    cue_tone_idxs = np.where(cue_tone_bool)[0]
    cue_tone_onsets = valid_tone_onsets[cue_tone_idxs]
    cue_tone_offsets = valid_tone_offsets[cue_tone_idxs]

    cue_tone_on_ts = nap.Ts(t=cue_tone_onsets, time_support=time_support)
    cue_tone_off_ts = nap.Ts(t=cue_tone_offsets, time_support=time_support)

    ts_list = [cue_tone_on_ts, cue_tone_off_ts]
    ts_index = ['start_tone_on', 'start_tone_off']

    return ts_list, ts_index


def get_digital_ts(dig_in, time_support, din=None, event_name='lick'):
    if din is None:
        print("Warning: din not specified, using default channel 0 for lick events.")
        din = 0

    digchids = dig_in.channel_ids
    times = dig_in.get_times()
    n_times = len(times)

    dig_trace = dig_in.get_traces(channel_ids=[digchids[din]]).squeeze().astype(np.int32)
    ontimes = get_edge_times(dig_trace, times, edge_type='rising')
    offtimes = get_edge_times(dig_trace, times, edge_type='falling')

    #check that the first ontime is less than the first offtime
    if ontimes[0] > offtimes[0]:
        #remove the first index from offtimes
        offtimes = offtimes[1:]
    #check that the last ontime is less than the last offtime
    if ontimes[-1] > offtimes[-1]:
        ontimes = ontimes[:-1]

    if (ontimes[0] > offtimes[0] or ontimes[-1] > offtimes[-1]):
        raise ValueError("Onset times and offset times are not properly ordered. Check the digital event times and mapping.")

    on_ts = nap.Ts(t=ontimes, time_support=time_support)
    off_ts = nap.Ts(t=offtimes, time_support=time_support)
    tslist = [on_ts, off_ts]
    indexlist = [f'{event_name}_on', f'{event_name}_off']
    return tslist, indexlist


def _find_thresholds_gmm(data, n_modes=7, plot=False, save_folder=None, plot_label=None):
    """Find thresholds using Gaussian Mixture Model."""
    # Fit Gaussian Mixture Model
    gmm = GaussianMixture(n_components=n_modes, random_state=42)
    gmm.fit(data.reshape(-1, 1))
    vals = gmm.sample(1000)[0].flatten()

    # Get peak locations (means of Gaussians)
    peaks = gmm.means_.flatten()
    sort_idx = np.argsort(peaks)
    peaks = peaks[sort_idx]

    # Calculate thresholds as midpoints between adjacent peaks
    thresholds = []
    for i in range(len(peaks) - 1):
        # Find the minimum probability point between two peaks
        x_range = np.linspace(peaks[i], peaks[i + 1], 1000)
        probs = np.exp(gmm.score_samples(x_range.reshape(-1, 1)))
        threshold = x_range[np.argmin(probs)]
        thresholds.append(threshold)

    if plot:
        nb = 1000
        fig = plt.figure(figsize=(8, 4))
        plt.hist(data, bins=nb, density=True, alpha=0.3, label='Data Histogram (log scale)')
        #xaxis = np.linspace(np.min(data), np.max(data), nb)
        #plt.plot(xaxis, vals)
        for peak in peaks:
            plt.axvline(peak, color='green', linestyle=':', label='Peak' if peak == peaks[0] else "")
        for thresh in thresholds:
            plt.axvline(thresh, color='red', linestyle='-', label='Threshold' if thresh == thresholds[0] else "")
        #log scale y axis
        plt.yscale('log')
        for t in thresholds:
            plt.axvline(t, color='gray', linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        if save_folder is None:
            save_folder = Path.cwd() / 'gmm_threshold_plots'
        save_folder = Path(save_folder)
        save_folder.mkdir(parents=True, exist_ok=True)
        label = plot_label or 'gmm_thresholds'
        save_path = _unique_output_path(save_folder / f'{label}_n{n_modes}.png')
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved GMM threshold plot to {save_path}")
        plt.close(fig)

    return sorted(thresholds)


def find_mode_thresholds(data, n_modes=7, nbin=100, plot=False):
    """
    Identify threshold cutoffs separating modes in a noisy 1D time series.

    Parameters:
        data : array-like
            The input time series values.
        n_modes : int
            Expected number of modes (peaks).
        bandwidth : float or None
            Bandwidth for KDE smoothing (if None, chosen automatically).
        plot : bool
            Whether to plot the estimated density and thresholds.

    Returns:
        thresholds : list of floats
            Sorted values dividing the modes.
    """
    #all data that is less than 0 are set to 1
    #floor all data to 0
    data = np.array(data)
    hist = np.histogram(data, nbin)
    counts = hist[0]
    counts_smooth = scipy.ndimage.gaussian_filter1d(counts, nbin/500)
    #smooth counts with gaussian filte
    bin_edges = hist[1]
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    xs = bin_centers

    # Find peaks (modes)
    peak_idxs, _ = find_peaks(counts_smooth)

    peaks = xs[peak_idxs]
    if len(peaks) < n_modes:
        raise ValueError(f"Found only {len(peaks)} modes, fewer than expected {n_modes} modes.")
    if len(peaks) > n_modes:
        prominences = scipy.signal.peak_prominences(counts_smooth, peak_idxs)[0]
        #get the n_modes most prominent peaks
        top_n_idxs = np.argsort(prominences)[-n_modes:]
        peaks = peaks[top_n_idxs]

    peaks = np.sort(peaks)
    #If COUNTS smooth between peaks is not zero, find the local minima between peaks
    thresholds = []
    for i in range(len(peaks)-1):
        peak1_idx = np.where(xs == peaks[i])[0][0]
        peak2_idx = np.where(xs == peaks[i+1])[0][0]
        segment = counts_smooth[peak1_idx:peak2_idx]
        if np.min(segment) > 0:
            #find local minima in segment
            local_min_idxs, _ = find_peaks(-segment)
            if len(local_min_idxs) > 0:
                #get the global minimum
                global_min_idx = local_min_idxs[np.argmin(segment[local_min_idxs])]
                threshold = xs[peak1_idx + global_min_idx]
                thresholds.append(threshold)
        else:
            #get midpoint between peaks
            threshold = (peaks[i] + peaks[i+1]) / 2.0
            thresholds.append(threshold)
    if len(thresholds) != n_modes - 1:
        raise ValueError(f"Number of thresholds found ({len(thresholds)}) does not match expected ({n_modes - 1}).")

    # Optionally plot
    if plot:
        plt.figure(figsize=(8, 4))
        plt.hist(data, bins=nbin, density=True, alpha=0.3, label='Data Histogram (log scale)')
        plt.plot(xs, counts_smooth)
        for peak in peaks:
            plt.axvline(peak, color='green', linestyle=':', label='Peak' if peak == peaks[0] else "")
        for thresh in thresholds:
            plt.axvline(thresh, color='red', linestyle='-', label='Threshold' if thresh == thresholds[0] else "")
        #log scale y axis
        plt.yscale('log')
        for t in thresholds:
            plt.axvline(t, color='gray', linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()

    return sorted(thresholds)

#get on and off edges for each level
def get_opto_edges(trace, time_array, powers, plot=False, save_folder=None, plot_label=None):
    n_powers = len(powers)
    max_power = max(powers)
    lower_cutoff = 500#np.mean(trace) * 2
    #scaling_factor = (max_trace - mean_trace) ** 0.66
    #lower_cutoff = (scaling_factor * (min_power**0.25))/2 + np.mean(trace)
    filtered_trace = trace[trace >= lower_cutoff]
    #a channel with (almost) no supra-cutoff samples carries no opto pulses; return
    #empty edges so the caller can skip it rather than crashing the GMM fit
    if len(filtered_trace) < n_powers:
        empty = [np.array([]) for _ in powers]
        return empty, empty, np.zeros(len(trace))
    thresholds = _find_thresholds_gmm(np.log(filtered_trace), n_modes=n_powers, plot=plot, save_folder=save_folder, plot_label=plot_label)
    thresholds = np.exp(thresholds)
    thresholds = np.concatenate((np.array([lower_cutoff]), thresholds))

    #get all edges greater than the first threshold
    min_threshold = min(thresholds)
    max_threshold = max(thresholds)
    digital_trace = trace > min_threshold
    digital_trace = digital_trace.astype(int)
    onsets, onidxs = get_edge_times(digital_trace, time_array, edge_type='rising', return_idxs=True)
    offsets, offidxs = get_edge_times(digital_trace, time_array, edge_type='falling', return_idxs=True)
    durations = offsets - onsets
    #filter out any onsets/offsets where duration is less than 0.01 seconds
    valid_durations = durations >= 0.009
    onidxs = np.array(onidxs)[valid_durations]
    offidxs = np.array(offidxs)[valid_durations]

    opto_trace = np.zeros(len(trace))
    for onidx, offidx in zip(onidxs, offidxs):
        section = trace[onidx:offidx]
        section_level = np.mean(section)
        #identify which theshold this section corresponds to
        for pidx, level in enumerate(powers):
            if level == max_power:
                if section_level > max_threshold:
                    opto_trace[onidx:offidx] = level
            else:
                if (section_level > thresholds[pidx]) and (section_level <= thresholds[pidx + 1]):
                    opto_trace[onidx:offidx] = level
    #break optoidx into binary traces for each level
    onset_times = []
    offset_times = []
    for pidx, level in enumerate(powers):
        binary_trace = (opto_trace == level).astype(int)
        ontimes, _ = get_edge_times(binary_trace, time_array, edge_type='rising', return_idxs=True)
        offtimes, _ = get_edge_times(binary_trace, time_array, edge_type='falling', return_idxs=True)
        onset_times.append(ontimes)
        offset_times.append(offtimes)
    return onset_times, offset_times, opto_trace


def get_opto_ts(analog_events, time, idx=None, pwrs=None, opsin_map=None, plot=False, save_folder=None):
    if idx is None:
        print("Warning: idx not specified, using default LED channel map.")
        idx = default_adcmap
    if pwrs is None:
        print("Warning: power levels not specified, using default power levels.")
        pwrs = default_optoidx
    if opsin_map is None:
        opsin_map = default_opsin_map

    #The paired LED drivers (e.g. 'blue_1','blue_2') are hardware-matched, so a single
    #representative channel per color is emitted and labeled by opsin (blue->chr2,
    #orange->chrimson) — not by '_1'/'_2' — for cross-compatibility with the nidq
    #pipeline and later analyses. adcmap is keyed per channel, optoidx per color.
    ts_list = []
    ts_idx = []
    time_support = nap.IntervalSet(start=time[0], end=time[-1])
    adcchids = analog_events.channel_ids
    emitted_colors = set()
    for channel_name, ch in idx.items():
        #color is the prefix carried in the channel name ('blue_1' -> 'blue')
        color = next((c for c in opsin_map if c in channel_name), None)
        if color is None or color in emitted_colors:
            continue
        name = opsin_map[color]  #'blue' -> 'chr2', 'orange' -> 'chrimson'
        #optoidx may be keyed per channel ('blue_1') or per color ('blue')
        pw = pwrs.get(channel_name, pwrs.get(color))
        if pw is None:
            continue
        trace = analog_events.get_traces(channel_ids=[adcchids[ch]]).squeeze()
        on_times, off_times, opto_trace = get_opto_edges(trace, time, pw, plot=plot, save_folder=save_folder, plot_label=name)
        #a flat channel yields no edges; skip it so the pair's live channel can still be used
        if not any(len(o) for o in on_times):
            continue
        emitted_colors.add(color)
        ts_list.append(nap.Tsd(t=time, d=opto_trace, time_support=time_support))
        ts_idx.append(f"{name}_trace")
        for pidx, level in enumerate(pw):
            on = nap.Ts(t=on_times[pidx], time_support=time_support)
            off = nap.Ts(t=off_times[pidx], time_support=time_support)
            ts_list.append(on)
            ts_idx.append(f"{name}_on_{level}")
            ts_list.append(off)
            ts_idx.append(f"{name}_off_{level}")
    return ts_list, ts_idx

def parse_trials(start_tone_ts, solenoid_ts, lick_ts):
    time_support = lick_ts.time_support
    start_times = start_tone_ts.t
    lick_times = lick_ts.t
    solenoid_times = solenoid_ts.t

    reward_trial_starts = []
    early_trial_starts = []
    timeout_trial_starts = []
    rewarded_first_licks = []
    early_first_licks = []
    timeout_outcomes = []
    trial_outcomes = []

    for start_time in start_times:
        end_window = start_time + 7.0

        licks_in_window = lick_times[(lick_times > start_time) & (lick_times <= end_window)]

        if len(licks_in_window) == 0:
            #timeout: no lick within 7s of the cue
            timeout_trial_starts.append(start_time)
            timeout_outcomes.append(end_window)
            trial_outcomes.append(end_window)
        else:
            first_lick = licks_in_window[0]  #nearest lick following the cue
            time_from_cue = first_lick - start_time
            #reward requires a solenoid within 0.1s AFTER the lick
            solenoid_after = np.any((solenoid_times >= first_lick) & (solenoid_times <= first_lick + 0.1))

            if time_from_cue >= 3.333 and solenoid_after:
                #reward: first lick in [3.333, 7]s confirmed by a solenoid
                reward_trial_starts.append(start_time)
                rewarded_first_licks.append(first_lick)
                trial_outcomes.append(first_lick)
            else:
                #early: a first lick that is not a rewarded lick (premature, or no solenoid)
                early_trial_starts.append(start_time)
                early_first_licks.append(first_lick)
                trial_outcomes.append(first_lick)

    #every start tone must produce exactly one outcome (equal-length requirement)
    assert len(trial_outcomes) == len(start_times), \
        f"outcome count ({len(trial_outcomes)}) does not match start tone count ({len(start_times)})"

    reward_trial_starts_ts = nap.Ts(np.array(reward_trial_starts), time_support=time_support) if len(reward_trial_starts) > 0 else nap.Ts(np.array([]), time_support=time_support)
    early_trial_starts_ts = nap.Ts(np.array(early_trial_starts), time_support=time_support) if len(early_trial_starts) > 0 else nap.Ts(np.array([]), time_support=time_support)
    timeout_trial_starts_ts = nap.Ts(np.array(timeout_trial_starts), time_support=time_support) if len(timeout_trial_starts) > 0 else nap.Ts(np.array([]), time_support=time_support)

    rewarded_first_licks_ts = nap.Ts(np.array(rewarded_first_licks), time_support=time_support) if len(rewarded_first_licks) > 0 else nap.Ts(np.array([]), time_support=time_support)
    early_first_licks_ts = nap.Ts(np.array(early_first_licks), time_support=time_support) if len(early_first_licks) > 0 else nap.Ts(np.array([]), time_support=time_support)
    timeout_ts = nap.Ts(np.array(timeout_outcomes), time_support=time_support) if len(timeout_outcomes) > 0 else nap.Ts(np.array([]), time_support=time_support)
    trial_outcome_ts = nap.Ts(np.sort(np.array(trial_outcomes)), time_support=time_support) if len(trial_outcomes) > 0 else nap.Ts(np.array([]), time_support=time_support)

    return (reward_trial_starts_ts, early_trial_starts_ts, timeout_trial_starts_ts,
            rewarded_first_licks_ts, early_first_licks_ts, timeout_ts, trial_outcome_ts)

def parse_lick_bouts(lick_ts, ili_thresh=0.3):
    #thresh from https://pmc.ncbi.nlm.nih.gov/articles/PMC6063358/
    time_support = lick_ts.time_support
    lick_times = lick_ts.t
    if len(lick_times) == 0:
        return [], []
    ilis = np.diff(lick_times)
    bout_starts = [lick_times[0]]
    bout_ends = []
    for idx, ili in enumerate(ilis):
        if ili > ili_thresh:
            bout_ends.append(lick_times[idx])
            bout_starts.append(lick_times[idx + 1])
    bout_ends.append(lick_times[-1])

    bout_start_ts = nap.Ts(np.array(bout_starts), time_support=time_support)
    bout_end_ts = nap.Ts(np.array(bout_ends), time_support=time_support)
    return bout_start_ts, bout_end_ts

def match_starts_and_ends(starts_ts, ends_ts):
    #check if starts and ends are pynapple ts
    if isinstance(starts_ts, nap.Ts) and isinstance(ends_ts, nap.Ts):
        starts = starts_ts.t
        ends = ends_ts.t
    elif isinstance(starts_ts, np.ndarray) and isinstance(ends_ts, np.ndarray):
        starts = starts_ts
        ends = ends_ts
    else:
        raise ValueError("starts_ts and ends_ts must be either pynapple Ts or numpy arrays")

    matched_starts = []
    matched_ends = []
    for s in starts:
        possible_ends = ends[ends > s]
        if len(possible_ends) > 0:
            matched_starts.append(s)
            matched_ends.append(possible_ends[0])

    if isinstance(starts_ts, nap.Ts) and isinstance(ends_ts, nap.Ts):
        time_support = starts_ts.time_support
        matched_starts = nap.Ts(np.array(matched_starts), time_support=time_support)
        matched_ends = nap.Ts(np.array(matched_ends), time_support=time_support)

    return matched_starts, matched_ends

def parse_base_folder(base_folder, test_mode=False):
    #base folder is the rec_folder where the raw recording files live
    if base_folder is None:
        if test_mode:
            base_folder = Path(r"C:\Users\assad\Documents\recording_files\Neptun\Neptun_20260626")
        else:
            #query user to enter base folder in command line
            print(r"base folder example: C:\Users\assad\Documents\recording_files\Neptun\Neptun_20260626")
            base_folder_input = input("Please enter the base folder path: ")
            base_folder = Path(base_folder_input)

    elif not isinstance(base_folder, Path):
        base_folder = Path(base_folder)

    return base_folder


def get_maps_from_txt(base_folder):
    #get 'config.txt' from base_folder
    config_file = base_folder / 'intan_map.txt'
    if not config_file.exists():
        raise FileNotFoundError(f"No intan_map.txt file found in {base_folder}")

    vars_dict = {}
    with open(config_file) as f:
        tree = ast.parse(f.read())
        for node in tree.body:
            if isinstance(node, ast.Assign):
                name = node.targets[0].id
                value = ast.literal_eval(node.value)
                vars_dict[name] = value

    #check that dinmap, ainmap, and optoidx are in vars_dict
    required_keys = ['dinmap', 'adcmap', 'optoidx']
    for key in required_keys:
        if key not in vars_dict.keys():
            raise KeyError(f"{key} not found in intan_map.txt")

    dinmap = vars_dict["dinmap"]
    ainmap = vars_dict["adcmap"]
    optoidx = vars_dict["optoidx"]

    return dinmap, ainmap, optoidx

def parse_channel_maps(base_folder, dinmap, adcmap, optoidx):
    #dinmap should be a dictionary mapping digital channel ids to event names
    #adcmap should be a dictionary mapping analog channel ids to event names
    if (dinmap is not None) and (adcmap is not None) and (optoidx is not None):
        return dinmap, adcmap, optoidx
    elif (dinmap is None) and (adcmap is None) and (optoidx is None):
        #check that intan_map.txt exists in base_folder
        config_file = base_folder / 'intan_map.txt'
        if config_file.exists():
            dinmap, adcmap, optoidx = get_maps_from_txt(base_folder)
        else:
            print("Warning: No intan_map.txt file found in base folder, using default channel maps.")
            dinmap = default_dinmap
            adcmap = default_adcmap
            optoidx = default_optoidx
    else:
        raise ValueError("Either provide all of dinmap, ainmap, and optoidx, or none to use intan_map.txt or defaults.")

    return dinmap, adcmap, optoidx

def get_output_folders(base_folder):
    """Resolve where a binary-signals run reads and writes (see utils/file_structure.py).

    The event tables and threshold plots stay with the rest of the analysis products in the
    pynapple folder; binary_signals.npz itself lives in its own binary_signals folder, because
    sorting needs a copy of it too (kept in step by folders.mirror_binary_signals()).
    """
    folders = fs.SessionPaths(base_folder, create=True)
    pynapple_folder = folders.pynapple_folder
    pynapple_folder.mkdir(parents=True, exist_ok=True)
    binary_signals_file = folders.binary_signals_file()
    binary_signals_file.parent.mkdir(parents=True, exist_ok=True)

    return folders, pynapple_folder, binary_signals_file

def load_pynapple_file(pynapple_file):
    if pynapple_file.exists():
        print(f"Loading existing binary signals from {pynapple_file}, set overwrite=True to re-parse.")
        bs = nap.load_file(str(pynapple_file))
        return bs
    else:
        print(f"No existing binary signals found at {pynapple_file}, parsing new signals.")
        return None


def _split_event_label(event_name):
    import re
    if not isinstance(event_name, str):
        return event_name, None

    # Skip continuous analog trace signals
    if event_name.endswith('_trace'):
        return event_name, None

    # Opto-style: name_on_N or name_off_N (e.g. chrimson_on_10, chr2_off_200)
    m = re.match(r'^(.+)_(on|off)_(\d+)$', event_name)
    if m:
        base = f"{m.group(1)}_{m.group(3)}"
        return base, m.group(2)

    # Standard _on / _off
    if event_name.endswith('_on'):
        return event_name[:-3], 'on'
    if event_name.endswith('_off'):
        return event_name[:-4], 'off'

    # _starts / _ends  (e.g. ITI_starts, lick_bout_ends)
    if event_name.endswith('_starts'):
        return event_name[:-7], 'on'
    if event_name.endswith('_ends'):
        return event_name[:-5], 'off'

    # Everything else is a single-edge timestamp (e.g. rewarded_first_licks, rewarded_cues)
    return event_name, 'single'


def export_event_table(tsg, csv_file):
    metadata = getattr(tsg, 'metadata', None)

    # Collect times grouped by base_event and edge type
    events_dict = {}  # {base_event: {'on': [], 'off': [], 'single': []}}

    if metadata is not None and len(metadata) > 0:
        for key, row in metadata.iterrows():
            event_name = row.get('event')
            base_event, edge = _split_event_label(event_name)
            if edge is None:
                continue  # skip _trace and unrecognised entries

            if base_event not in events_dict:
                events_dict[base_event] = {'on': [], 'off': [], 'single': []}

            times = np.asarray(tsg[key].t)
            events_dict[base_event][edge].extend(times)

    rows = []
    for base_event in sorted(events_dict.keys()):
        on_times     = np.sort(np.array(events_dict[base_event]['on']))
        off_times    = np.sort(np.array(events_dict[base_event]['off']))
        single_times = np.sort(np.array(events_dict[base_event]['single']))

        # Paired on/off rows
        max_idx = max(len(on_times), len(off_times)) if (len(on_times) or len(off_times)) else 0
        for i in range(max_idx):
            on_time  = on_times[i]  if i < len(on_times)  else np.nan
            off_time = off_times[i] if i < len(off_times) else np.nan
            duration = off_time - on_time if not (np.isnan(on_time) or np.isnan(off_time)) else np.nan
            rows.append({'base_event': base_event, 'edge_on': on_time, 'edge_off': off_time, 'duration': duration})

        # Single-edge timestamp rows (no pairable counterpart)
        for t in single_times:
            rows.append({'base_event': base_event, 'edge_on': t, 'edge_off': np.nan, 'duration': np.nan})

    event_table = pd.DataFrame(rows, columns=['base_event', 'edge_on', 'edge_off', 'duration'])
    if not event_table.empty:
        event_table = event_table.sort_values('edge_on', na_position='last').reset_index(drop=True)

    # Write NA string instead of nan for readability
    event_table_out = event_table.copy().astype(str).replace('nan', 'NA')
    event_table_out.to_csv(csv_file, index=False)
    print(f"Saved event table to {csv_file}")

    # Also write summary CSV alongside the event table
    csv_file = Path(csv_file)
    summary_file = csv_file.parent / (csv_file.stem + '_summary.csv')
    export_event_summary(event_table, summary_file)

    return event_table


def export_event_summary(event_table, summary_file):
    """Write a per-base_event summary CSV: count, mean/min/max duration."""
    if event_table.empty:
        summary = pd.DataFrame(columns=['base_event', 'n_events', 'n_with_duration',
                                        'mean_duration', 'min_duration', 'max_duration', 'duration_range'])
        summary.to_csv(summary_file, index=False)
        print(f"Saved event summary to {summary_file}")
        return summary

    rows = []
    for base_event, group in event_table.groupby('base_event', sort=True):
        durations = group['duration'].dropna()
        n_events = len(group)
        n_with_dur = len(durations)
        mean_dur  = durations.mean()          if n_with_dur > 0 else np.nan
        min_dur   = durations.min()           if n_with_dur > 0 else np.nan
        max_dur   = durations.max()           if n_with_dur > 0 else np.nan
        dur_range = max_dur - min_dur         if n_with_dur > 0 else np.nan
        rows.append({
            'base_event':      base_event,
            'n_events':        n_events,
            'n_with_duration': n_with_dur,
            'mean_duration':   mean_dur,
            'min_duration':    min_dur,
            'max_duration':    max_dur,
            'duration_range':  dur_range,
        })

    summary = pd.DataFrame(rows, columns=['base_event', 'n_events', 'n_with_duration',
                                          'mean_duration', 'min_duration', 'max_duration',
                                          'duration_range'])
    summary.to_csv(summary_file, index=False)
    print(f"Saved event summary to {summary_file}")
    return summary


def _unique_output_path(path):
    path = Path(path)
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    counter = 1
    while True:
        candidate = parent / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1

def find_rhd_folder(base_folder):
    rhd_folders = [f for f in Path(base_folder).rglob("*") if f.is_dir() and any(f.glob("*.rhd"))]
    if len(rhd_folders) > 1:
        raise ValueError(f"Multiple .rhd folders found: {rhd_folders}")
    return rhd_folders[0] if rhd_folders else None

def get_binary_signals(base_folder=None, dinmap=None, adcmap=None, optoidx=None, overwrite=False, plot=False, test_mode=False, plot_save_folder=None):
    #section 1: folder handling
    base_folder = parse_base_folder(base_folder, test_mode=test_mode)
    folders, pynapple_folder, pynapple_file = get_output_folders(base_folder)
    #the .rhd files and the channel map are only ever read from the recording folder
    rec_folder = folders.recording_folder
    rhd_folder = find_rhd_folder(rec_folder)
    event_csv_file = pynapple_folder / "binary_signals_events.csv"
    if plot_save_folder is None:
        plot_save_folder = pynapple_folder

    if not overwrite:
        bs = load_pynapple_file(pynapple_file)
        if bs is not None:
            folders.mirror_binary_signals()
            export_event_table(bs, event_csv_file)
            return bs

    #section 2: channel mapping
    dinmap, adcmap, optoidx = parse_channel_maps(rec_folder, dinmap, adcmap, optoidx)

    #section 3: read in intan data
    adc_in = read_split_intan_files(rhd_folder, stream_name='USB board ADC input channel')
    adc_in = spre.unsigned_to_signed(adc_in)
    dig_in = read_split_intan_files(rhd_folder, stream_name='USB board digital input channel')
    dig_in = spre.unsigned_to_signed(dig_in)
    adcchids = adc_in.channel_ids #get all channel ids--channels are indexed using alphanumeric string identifier
    digchids = dig_in.channel_ids

    time = dig_in.get_times()
    time_support = nap.IntervalSet(start=time[0], end=time[-1])

    #analog event index
    ts_list = []
    ts_idx = []
    if 'solenoid' in dinmap.keys():
        solenoid_din = get_channel_index('solenoid', dinmap, digchids)
        solenoid_ts_list, solenoid_ts_idx = get_digital_ts(dig_in, time_support, solenoid_din, event_name='solenoid')
    else:
        raise ValueError("Solenoid channel not found in dinmap.")
    ts_list += solenoid_ts_list
    ts_idx += solenoid_ts_idx

    if 'cue' in dinmap.keys():
        tones_ts_list, tones_ts_idx = parse_tones_digital(dig_in, solenoid_ts_list, time_support, idx=dinmap)
    else:
        raise ValueError("Tone channel not found in either dinmap or ainmap.")
    ts_list += tones_ts_list
    ts_idx += tones_ts_idx

    if 'licks' in dinmap.keys():
        lick_din = dinmap['licks']
        lick_ts_list, lick_ts_idx = get_digital_ts(dig_in, time_support, lick_din)
    else:
        raise ValueError("Lick channel not found in dinmap.")

    ts_list += lick_ts_list
    ts_idx += lick_ts_idx

    led_check = any('orange' in k or 'blue' in k for k in adcmap.keys())
    if led_check:
        opto_ts_list, opto_ts_idx = get_opto_ts(adc_in, time, idx=adcmap, pwrs=optoidx, plot=plot, save_folder=plot_save_folder)
        ts_list += opto_ts_list
        ts_idx += opto_ts_idx

    ts_numbers = [i for i in range(len(ts_list))]
    ts_dict = dict(zip(ts_numbers, ts_list))
    ts_idx_dict = dict(zip(ts_numbers, ts_idx))
    ts_idx_df = pd.DataFrame.from_dict(ts_idx_dict, orient='index', columns=['event'])

    tsg = nap.TsGroup(ts_dict, metadata=ts_idx_df)

    metadata = tsg.metadata
    start_tone_idx = metadata[metadata['event'] == 'start_tone_on'].index[0]
    lick_ts_idx = metadata[metadata['event'] == 'lick_on'].index[0]
    solenoid_ts_idx = metadata[metadata['event'] == 'solenoid_on'].index[0]

    start_tone_ts = tsg[start_tone_idx]
    lick_ts = tsg[lick_ts_idx]
    solenoid_ts = tsg[solenoid_ts_idx]

    reward_trial_starts_ts, early_trial_starts_ts, timeout_trial_starts_ts, rewarded_first_licks_ts, early_first_licks_ts, timeout_ts, trial_outcome_ts = parse_trials(start_tone_ts, solenoid_ts, lick_ts)

    lick_bout_starts_ts, lick_bout_ends_ts = parse_lick_bouts(lick_ts)

    ITI_starts = nap.Ts(trial_outcome_ts.t[:-1] + 3.7, time_support=time_support)
    ITI_ends = nap.Ts(start_tone_ts.t[1:] - 1.5, time_support=time_support)

    first_licks_metadata = pd.DataFrame.from_dict({
        1: 'rewarded_first_licks',
        2: 'early_first_licks',
        3: 'reward_trial_starts',
        4: 'early_trial_starts',
        5: 'timeout_trial_starts',
        6: 'start_tones',
        7: 'reward_outcomes',
        8: 'early_outcomes',
        9: 'timeout_outcomes',
        10: 'trial_outcomes',
        11: 'ITI_starts',
        12: 'ITI_ends',
        13: 'lick_bout_starts',
        14: 'lick_bout_ends'}, orient='index', columns=['event'])
    first_licks_ts = nap.TsGroup({1: rewarded_first_licks_ts,
                                  2: early_first_licks_ts,
                                  3: reward_trial_starts_ts,
                                  4: early_trial_starts_ts,
                                  5: timeout_trial_starts_ts,
                                  6: start_tone_ts,
                                  7: rewarded_first_licks_ts,
                                  8: early_first_licks_ts,
                                  9: timeout_ts,
                                  10: trial_outcome_ts,
                                  11: ITI_starts,
                                  12: ITI_ends,
                                  13: lick_bout_starts_ts,
                                  14: lick_bout_ends_ts
                                  },
                                 metadata=first_licks_metadata, time_support=time_support)




    # mege first_licks_ts into tsg
    tsg = tsg.merge(first_licks_ts, reset_index=True)
    tsg.save(str(pynapple_file))
    folders.mirror_binary_signals()
    export_event_table(tsg, event_csv_file)

    return tsg

def test():
    dinmap = None
    adcmap = None
    optoidx = None
    base_folder = None
    test_mode=False
    overwrite=True
    plot=True
    plot_save_folder = None
    base_folder = Path(r"C:\Users\assad\Documents\recording_files\Neptun\Neptun_20260626")
    tsg = get_binary_signals(base_folder,dinmap,adcmap,optoidx,overwrite,plot,test_mode)