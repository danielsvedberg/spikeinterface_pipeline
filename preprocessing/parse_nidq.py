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
import re
from utils import file_structure as fs
from preprocessing import event_table_licks as etl

default_dinmap = {
    'licks': 0,
    'timing_pulse': 1
}

default_ainmap = {
    'photometry': 0,
    'accelerometer_x': 1,
    'accelerometer_y': 2,
    'accelerometer_z': 3,
    'solenoid': 4,
    'tone': 5,
    'chrimson': 6,
    'chr2': 7
}

# ints = [10,20,40,100,200,400] #light intensities in microwatts
default_ints = [20, 40, 60, 80, 100]  # light intensities in microwatts
default_optoidx = {'chrimson': default_ints,
                   'chr2': default_ints}


def get_nidq_aligner(sync_rec, digital_events, pulse_din=1):

    digchids = digital_events.channel_ids
    #imec_sync = imec_rec.select_channels([imec_rec.channel_ids[-1]])
    imec_time = sync_rec.get_times()
    imec_time = imec_time - imec_time[0]  # zeroing time
    sync_trace = sync_rec.get_traces()
    #convert sync trace to boolean, get the min and max to define a threshold
    threshold = (np.max(sync_trace) + np.min(sync_trace)) / 2
    sync_trace_bool = sync_trace > threshold
    sync_trace_bool = sync_trace_bool.astype(int).squeeze()
    #get the rising/falling edges of the boolean trace
    off_edges = np.where((np.diff(sync_trace_bool) == 1))[0] #| (np.diff(sync_trace_bool) == -1))[0]
    on_edges = np.where((np.diff(sync_trace_bool) == -1))[0]
    imec_onsets = imec_time[on_edges+1]  # +1 because diff reduces length by 1
    imec_offsets = imec_time[off_edges+1]

    nidq_edges = digital_events.get_event_times(digchids[pulse_din])
    #the first half -1 are onsets, the second half are offsets
    nind = len(nidq_edges)
    nidq_offsets = nidq_edges[:nind//2]
    nidq_onsets = nidq_edges[nind//2:]

    #concatenate onsets and offsets
    pulseA = np.sort(np.concatenate((imec_onsets, imec_offsets)))
    pulseB = np.sort(np.concatenate((nidq_onsets, nidq_offsets)))

    # Often, clocks drift linearly. Let's fit tA = slope * tB + intercept
    coef = np.polyfit(pulseB, pulseA, 1)
    slope, intercept = coef
    print(f"Clock alignment fit: tA ~ {slope:.9f} * tB + {intercept:.6f}")

    # Create a function to map any DAQ B timestamp into DAQ A time
    map_B_to_A = np.poly1d(coef)

    return map_B_to_A

#the NIDQ -> imec clock fit, cached beside binary_signals. Computing it reads the imec SYNC
#channel, which SpikeGLX interleaves with every probe channel, so it scans the whole AP binary
#(66-99 GB per session; ~10 min from D:, and its mapped pages fill memory when sessions run in
#parallel). The fit is two numbers that never change for a session's raw data, so it is kept
ALIGNMENT_FILE = 'nidq_imec_alignment.json'


def _raw_signature(spikeglx_folder, imec_folder):
    """Sizes of the raw binaries the alignment was computed from (a re-acquired or edited
    session invalidates the cache)."""
    sig = {}
    for folder, pattern in ((spikeglx_folder, '*.nidq.bin'), (imec_folder, '*.ap.bin')):
        if folder is not None:
            for f in sorted(Path(folder).glob(pattern)):
                sig[f.name] = f.stat().st_size
    return sig


def _alignment_file(paths):
    folder = paths.binary_signals_folder()
    return folder / ALIGNMENT_FILE if folder is not None else None


def load_cached_alignment(paths, spikeglx_folder, imec_folder):
    """The cached NIDQ -> imec map (np.poly1d) if it matches the raw data, else None."""
    import json
    f = _alignment_file(paths)
    if f is None or not f.exists():
        return None
    c = json.loads(f.read_text())
    if c.get('raw') != _raw_signature(spikeglx_folder, imec_folder):
        print(f'{f.name} is for other raw files; recomputing the clock alignment')
        return None
    print(f"Clock alignment from {f.name} ({c.get('source', '')}): "
          f"tA ~ {c['slope']:.9f} * tB + {c['intercept']:.6f}")
    return np.poly1d([c['slope'], c['intercept']])


def save_alignment(paths, spikeglx_folder, imec_folder, coef, source):
    import json
    f = _alignment_file(paths)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({'slope': float(coef[0]), 'intercept': float(coef[1]),
                             'source': source,
                             'raw': _raw_signature(spikeglx_folder, imec_folder)}, indent=2))


def seed_alignment_from_saved(base_folder, max_residual_s=1e-6):
    """Cache the clock alignment of an already parsed session without reading the AP binary.

    Its saved binary_signals holds the raw NIDQ solenoid edges passed through the alignment
    (solenoid_on), so fitting saved against raw recovers the line exactly; the fit is trusted
    only if every edge lands within max_residual_s. Returns the coefficients, or None (no
    saved file, no digital solenoid, or a poor fit) -- the parse then computes it as usual.
    """
    base_folder = parse_base_folder(base_folder)
    paths = fs.SessionPaths(base_folder)
    dinmap, ainmap, optoidx = parse_channel_maps(paths, None, None, None)
    if 'solenoid' not in dinmap:
        return None
    pynapple_folder, pynapple_file = get_output_folders(paths)
    if not Path(pynapple_file).exists():
        return None
    bs = nap.load_file(str(pynapple_file))
    saved = {n: np.asarray(bs[k].t) for k, n in zip(bs.keys(), bs.metadata['event'])}.get('solenoid_on')
    spikeglx_folder = paths.spikeglx_folder()
    imec_folder = paths.imec_folder(fail_quietly=True)
    if saved is None or saved.size < 2 or imec_folder is None:
        return None
    digital_events = si.read_spikeglx_event(spikeglx_folder)
    raw = digital_events.get_event_times(digital_events.channel_ids[dinmap['solenoid']])
    raw_on = np.sort(raw[:len(raw) // 2])
    if raw_on.size != saved.size:
        return None
    coef = np.polyfit(raw_on, np.sort(saved), 1)
    resid = np.abs(np.polyval(coef, raw_on) - np.sort(saved)).max()
    if resid > max_residual_s:
        print(f'saved solenoid_on do not lie on one line with the raw edges (max {resid:.2e} s)')
        return None
    save_alignment(paths, spikeglx_folder, imec_folder, coef,
                   source=f'recovered from the saved binary_signals solenoid_on (max residual {resid:.1e} s)')
    return coef


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

def parse_solenoid_analog(analog_events, nidq_t_aligned, idx={'solenoid':4}, plot=False):
    time_support = nap.IntervalSet(start=nidq_t_aligned[0], end=nidq_t_aligned[-1])
    sol_ch = idx['solenoid']
    solenoid_trace = analog_events[:, sol_ch]
    thresh = find_mode_thresholds(solenoid_trace, n_modes=2, nbin=1000, plot=plot)
    #thresh = _find_thresholds_gmm(solenoid_trace, n_modes=2, plot=plot)
    # threshold solenoid trace to get digital values
    solenoid_digital = binarize_trace(solenoid_trace, threshold=thresh[0])
    solenoid_onsets = get_edge_times(solenoid_digital, nidq_t_aligned, edge_type='rising')
    solenoid_offsets = get_edge_times(solenoid_digital, nidq_t_aligned, edge_type='falling')
    solenoid_onsets = nap.Ts(t=solenoid_onsets, time_support=time_support)
    solenoid_offsets = nap.Ts(t=solenoid_offsets, time_support=time_support)
    solenoid_ts_list = [solenoid_onsets, solenoid_offsets]
    solenoid_ts_index = ['solenoid_on', 'solenoid_off']
    return solenoid_ts_list, solenoid_ts_index

def parse_tones_digital(digital_events, solenoid_ts_list, map2imectime, time_support, idx={'tone':3},
                        tonemap=None, method='auto'):
    """The task's tones from the digital tone line.

    method : 'frequency' (classify by pitch, parse_tones_by_frequency), 'duration' (by length,
    parse_tones), or 'auto': frequency when the line carries the speaker's square wave (many
    cycles per tone), duration when it carries only each tone's envelope.
    """
    time_support = nap.IntervalSet(start=time_support.start, end=time_support.end)
    solenoid_onsets = solenoid_ts_list[0].t

    tone_din = idx['tone']

    digchids = digital_events.channel_ids
    times = digital_events.get_event_times(digchids[tone_din])
    if map2imectime is not None:
        tone_times = map2imectime(times)
    else:
        tone_times = times

    mididx = len(tone_times) // 2
    tone_onsets = tone_times[:mididx]
    tone_offsets = tone_times[mididx:]
    return _parse_tone_edges(tone_onsets, tone_offsets, solenoid_onsets, time_support,
                             tonemap=tonemap, method=method)


def _parse_tone_edges(tone_onsets, tone_offsets, solenoid_onsets, time_support, tonemap=None,
                      method='auto'):
    """The tones from a tone line's rising and falling edges (see parse_tones_digital's method)."""
    if method == 'auto':
        # a square wave has short cycles: most rising edges follow the last within 20 ms
        rises = np.sort(tone_onsets)
        method = 'frequency' if rises.size > 10 and np.median(np.diff(rises)) < 0.02 else 'duration'
    print(f"tone line parsed by {method}")
    if method == 'frequency':
        return parse_tones_by_frequency(tone_onsets, tone_offsets, solenoid_onsets, time_support,
                                        tonemap=tonemap)
    return parse_tones(tone_onsets, tone_offsets, solenoid_onsets, time_support)


def parse_tones_analog(analog_events, solenoid_ts_list, nidq_t_aligned,
                       idx={'tone':5}, tonemap=None, method='auto'):
    """The task's tones from an analog tone channel, binarized; classified like the digital
    line (older rigs, e.g. DS13, record the speaker's square wave on an analog input)."""
    time_support = nap.IntervalSet(start=nidq_t_aligned[0], end=nidq_t_aligned[-1])
    solenoid_onsets = solenoid_ts_list[0].t

    tone_ch = idx['tone']
    tone_trace = analog_events[:,tone_ch]
    tone_digital = binarize_trace(tone_trace)
    tone_onsets = get_edge_times(tone_digital, nidq_t_aligned, edge_type='rising')
    tone_offsets = get_edge_times(tone_digital, nidq_t_aligned, edge_type='falling')
    return _parse_tone_edges(tone_onsets, tone_offsets, solenoid_onsets, time_support,
                             tonemap=tonemap, method=method)

def _near_any(times, targets, tol):
    """Boolean mask of the times that lie within tol of some target time."""
    times = np.asarray(times, dtype=float)
    targets = np.sort(np.asarray(targets, dtype=float))
    if len(targets) == 0:
        return np.zeros(len(times), dtype=bool)
    j = np.searchsorted(targets, times)
    nearest = np.minimum(np.abs(times - targets[np.clip(j - 1, 0, len(targets) - 1)]),
                         np.abs(times - targets[np.clip(j, 0, len(targets) - 1)]))
    return nearest <= tol

#the task's tones, by the frequency the behaviour program plays them at (Hz). The NIDQ tone line
#carries the speaker's square wave, so each tone is told by its pitch, not its length: an early
#lick starts the error tone at once, cutting the start cue off part-way, and the line then shows
#one unbroken pulse whose cue part can be any length (8 ms on DS15_20260702). Measured at 25 kHz
#on the DS15 and DS48 rigs: 3125, ~5000, 439 and 131 Hz -- the period is quantised to whole
#samples, so a 3 kHz tone (8.3 samples) reads as 8. A rig playing other tones sets `tonemap` in
#its nidaq_map.txt (same keys)
default_tonemap = {'start': 3000.0, 'reward': 5000.0, 'error': 440.0, 'timeout': 130.0}
#a cycle is given the nearest tone's label when its frequency is within this factor of it (a
#5 kHz period of 5 samples jitters to 4 or 6, i.e. -17%/+25%)
TONE_FREQ_TOLERANCE = 1.35
#the square waves are 50% duty, so a cycle's high time is half its tone's period too; it has
#to agree within this factor (a 3 kHz half-cycle is 3-4 samples at 25 kHz: up to x1.39). It
#rejects the cycle spanning a change of pitch, whose period can pass for the new tone's but
#which starts on the old tone's short high pulse
TONE_HIGH_TOLERANCE = 1.6
#labels are smoothed over this many cycles (a running mode), so single cycles mis-measured at
#the sample resolution do not split a tone
TONE_SMOOTH_CYCLES = 5


def _tone_segments(rises, falls, tonemap, tol=TONE_FREQ_TOLERANCE, smooth=TONE_SMOOTH_CYCLES,
                   max_gap=0.02):
    """Constant-pitch tones from the square-wave edges of a digital tone line.

    Every cycle (rising edge to rising edge) is labelled with the nearest tone in log frequency,
    or none if it is further than `tol` from all of them; labels are smoothed with a running mode
    over `smooth` cycles; a tone is then a run of one label with no gap over `max_gap` s (the
    20-ms inter-tone rule the duration parser used). Returns a list of (label, onset, offset).
    """
    rises, falls = np.sort(rises), np.sort(falls)
    if rises.size < 2:
        return []
    names = list(tonemap)
    logf = np.log(np.array([tonemap[n] for n in names], dtype=float))
    per = np.diff(rises)
    f = 1.0 / per
    d = np.abs(np.log(f)[:, None] - logf[None, :])
    lab = np.where(d.min(axis=1) <= np.log(tol), d.argmin(axis=1), -1)
    # each cycle's high time: its rising edge to the next falling edge
    nf = np.searchsorted(falls, rises[:-1], side='left')
    high = np.where(nf < falls.size, falls[np.clip(nf, 0, falls.size - 1)] - rises[:-1], np.nan)
    with np.errstate(divide='ignore', invalid='ignore'):
        f_high = 1.0 / (2.0 * high)
    ok = np.abs(np.log(f_high) - logf[np.clip(lab, 0, None)]) <= np.log(TONE_HIGH_TOLERANCE)
    lab = np.where((lab >= 0) & ok, lab, -1)
    raw = lab.copy()
    if smooth > 1:
        half = smooth // 2
        padded = np.pad(lab, half, mode='edge')
        win = np.lib.stride_tricks.sliding_window_view(padded, smooth)
        counts = np.stack([(win == k).sum(axis=1) for k in range(len(names))], axis=1)
        best = counts.argmax(axis=1)
        lab = np.where(counts.max(axis=1) > half, best, lab)
    # a cycle longer than max_gap is the silence between two tones, not a cycle
    lab = np.where(per > max_gap, -1, lab)
    segs = []
    k = 0
    while k < lab.size:
        if lab[k] < 0:
            k += 1
            continue
        j = k
        while j + 1 < lab.size and lab[j + 1] == lab[k]:
            j += 1
        nxt_k = j + 1
        # the smoothing mends mis-measured cycles inside a tone, but must not move its edges:
        # the cycle spanning a change of pitch (last edge of the cue to first edge of the error
        # tone) matches neither, and relabelled it would start the tone at the old tone's last
        # edge (DS2_20260511 at 2291.79 s: 3 ms early, before the lick that triggered it). So a
        # tone starts and ends on cycles that are its own pitch before smoothing
        while k <= j and raw[k] != lab[k]:
            k += 1
        while j >= k and raw[j] != lab[j]:
            j -= 1
        if k > j:
            k = nxt_k
            continue
        # cycles k..j span rises[k] .. rises[j+1]; the tone ends on the last falling edge
        # before the next rising edge (or the run's last edge)
        end_rise = rises[j + 1]
        nxt = np.searchsorted(falls, end_rise, side='left')
        offset = falls[nxt] if nxt < falls.size and (j + 2 >= rises.size or falls[nxt] < rises[j + 2]) \
            else end_rise
        if j - k + 1 >= 3:
            segs.append((names[lab[k]], float(rises[k]), float(offset)))
        k = nxt_k
    return segs


def parse_tones_by_frequency(rises, falls, solenoid_onsets, time_support, tonemap=None,
                             verbose=False, trim_bad_cues=True):
    """The task's tones from a digital tone line's square wave, classified by pitch.

    Every tone's onset is its own first cycle -- an error tone that cut a start cue off begins
    where the pitch changes, with no assumption about its length. Reward tones are also checked
    against the solenoid openings they start with (0.08-0.12 ms after); a mismatch is reported.
    Same outputs as parse_tones.
    """
    tonemap = dict(default_tonemap if tonemap is None else tonemap)
    segs = _tone_segments(rises, falls, tonemap)
    by = {n: (np.array([s[1] for s in segs if s[0] == n]), np.array([s[2] for s in segs if s[0] == n]))
          for n in tonemap}
    rew_on = by['reward'][0]
    near = _near_any(rew_on, solenoid_onsets, tol=0.002)
    print(f"tones by frequency: {', '.join(f'{n} {len(by[n][0])}' for n in tonemap)}; "
          f"{int(near.sum())}/{len(rew_on)} reward tones start with a solenoid opening "
          f"({len(solenoid_onsets)} openings)")
    if near.size and not near.all():
        print(f"Warning: reward tones without a solenoid opening at {rew_on[~near]}")
    return _package_tones(by['start'][0], by['start'][1], by['reward'][0], by['reward'][1],
                          by['timeout'][0], by['timeout'][1], by['error'][0], by['error'][1],
                          time_support, verbose=verbose, trim_bad_cues=trim_bad_cues)


def get_tonemap(paths):
    """The rig's tone frequencies: `tonemap` from its nidaq_map.txt, else default_tonemap."""
    config_file = paths.input_map
    if config_file is not None:
        tree = ast.parse(Path(config_file).read_text())
        for node in tree.body:
            if isinstance(node, ast.Assign) and node.targets[0].id == 'tonemap':
                return {**default_tonemap, **ast.literal_eval(node.value)}
    return dict(default_tonemap)


def parse_tones(tone_onsets, tone_offsets, solenoid_onsets, time_support,
                verbose=False, trim_bad_cues=True):
    tone_onsets.sort()
    tone_offsets.sort()

    inter_tone_intervals = tone_onsets[1:] - tone_offsets[:-1]
    durations = tone_offsets-tone_onsets

    #append infinity to the end to make them the same length as tone_onsets and tone_offsets
    #keep only tone onsets preceded by an interval of at least 0.001 seconds, and keep only tone offsets followed by an interval of at least 0.01
    min_isi = 0.02  # minimum inter-stimulus interval in seconds
    valid_itis = inter_tone_intervals >= min_isi

    valid_onsets = tone_onsets[np.append(True, valid_itis)]
    valid_offsets = tone_offsets[np.append(valid_itis, True)]
    valid_durations = valid_offsets - valid_onsets
    #make sure tone duration are at least 0.05s. anything less should be discarded
    valid_dur_idxs = valid_durations > 0.05
    valid_durations = valid_durations[valid_dur_idxs]
    valid_onsets = valid_onsets[valid_dur_idxs]
    valid_offsets = valid_offsets[valid_dur_idxs]

    #get odd tone indices where the error tone cuts off the cue
    odd_dur_idxs = (valid_durations > 0.205) & (valid_durations < 1.0)
    odd_offsets = valid_offsets[odd_dur_idxs]
    inferred_err_onsets = odd_offsets - 0.20015
    valid_onsets = np.append(valid_onsets, inferred_err_onsets)
    valid_onsets.sort()
    valid_offsets = np.append(valid_offsets, inferred_err_onsets)
    valid_offsets.sort()
    valid_durations = valid_offsets - valid_onsets

    #get the duration of each tone
    rounded_durations = np.round(valid_durations, 2)
    timeout_idx = rounded_durations > 1.0
    early_r_reward_idxs = (rounded_durations >= 0.2) & (rounded_durations < 1.0)
    #get indices of tone onsets that are not timeout or early reward tones
    cue_idx = ~(timeout_idx | early_r_reward_idxs)

    #a reward tone is the tone that starts with a solenoid opening (0.08-0.12 ms after it). match on
    #distance, not on rounded values: rounding to 10 ms splits any pair straddling a rounding
    #boundary, which mislabelled a DS48_20260923 reward tone as an error tone
    reward_tone_onset_idx = _near_any(valid_onsets, solenoid_onsets, tol=0.002)

    reward_tone_onsets = valid_onsets[reward_tone_onset_idx]
    reward_tone_offsets = valid_offsets[reward_tone_onset_idx]


    #if length of reward tone onsets is not equal to length of solenoid onsets, print a warning
    if len(reward_tone_onsets) != len(solenoid_onsets):
        print("Warning: number of reward tone onsets does not match number of solenoid onsets")
        if verbose:
            print('reward_tone_onsets', reward_tone_onsets)
            print('number of reward tones', len(reward_tone_onsets))
            print('solenoid_onsets', solenoid_onsets)
            print('number of solenoid onsets', len(solenoid_onsets))

    timeout_tone_onsets = valid_onsets[timeout_idx]
    timeout_tone_offsets = valid_offsets[timeout_idx]

    #cue tones are tones where tone duration is 0.1 seconds
    cue_tone_onsets = valid_onsets[cue_idx]
    cue_tone_offsets = valid_offsets[cue_idx]

    #error tones are the tones that are not reward, timeout, or cue tones
    #
    non_err_tone_idx = reward_tone_onset_idx + timeout_idx + cue_idx
    error_tone_idx = non_err_tone_idx == 0

    error_tone_onsets = valid_onsets[error_tone_idx]
    error_tone_offsets = valid_offsets[error_tone_idx]

    return _package_tones(cue_tone_onsets, cue_tone_offsets, reward_tone_onsets, reward_tone_offsets,
                          timeout_tone_onsets, timeout_tone_offsets, error_tone_onsets,
                          error_tone_offsets, time_support, verbose=verbose,
                          trim_bad_cues=trim_bad_cues)


def _package_tones(cue_tone_onsets, cue_tone_offsets, reward_tone_onsets, reward_tone_offsets,
                   timeout_tone_onsets, timeout_tone_offsets, error_tone_onsets, error_tone_offsets,
                   time_support, verbose=False, trim_bad_cues=True):
    """Check that cues and outcomes interleave, and package the tones as Ts.

    Shared by the duration (parse_tones) and the frequency (parse_tones_by_frequency) parsers.
    """
    #check that outcomes don't happen between cues
    for i, cue in enumerate(cue_tone_onsets):
        #get the next cue
        next_cue = cue_tone_onsets[i+1] if i+1 < len(cue_tone_onsets) else np.inf
        #check if any reward, timeout, or error tones happen between cue and next cue
        reward_in_between = np.where((reward_tone_onsets > cue) & (reward_tone_onsets < next_cue))[0]
        timeout_in_between = np.where((timeout_tone_onsets > cue) & (timeout_tone_onsets < next_cue))[0]
        error_in_between = np.where((error_tone_onsets > cue) & (error_tone_onsets < next_cue))[0]
        if len(reward_in_between) + len(timeout_in_between) + len(error_in_between) > 1:
            raise ValueError(f"Multiple outcome tones found between cue at index {i} and next cue.")

    outcomes = np.sort(np.concatenate((reward_tone_onsets, timeout_tone_onsets, error_tone_onsets)))
    n_outcomes = len(outcomes)
    n_cues = len(cue_tone_onsets)

    # one trailing unmatched cue is OK (task cut off mid-trial); unmatched outcomes are not
    if n_cues < n_outcomes:
        raise ValueError(
            f"More outcomes ({n_outcomes}) than cues ({n_cues}): "
            f"{n_outcomes - n_cues} outcome(s) have no preceding cue."
        )
    if n_cues > n_outcomes + 1:
        print(f"Warning: {n_cues - n_outcomes} unmatched cue(s)")


    bad_cues_idxs = []
    for i in range(n_outcomes):
        if cue_tone_onsets[i] >= outcomes[i]:
            raise ValueError(
                f"Cue {i} (t={cue_tone_onsets[i]:.4f}) does not precede its outcome (t={outcomes[i]:.4f})."
            )
        if i < n_outcomes - 1 and outcomes[i] >= cue_tone_onsets[i + 1]:
            if trim_bad_cues:
                j=i
                while i < n_outcomes - 1 and outcomes[i] >= cue_tone_onsets[i + 1]:
                    bad_cues_idxs.append(j)
                    cue_tone_onsets = np.delete(cue_tone_onsets, i, 0)
                    j = j+1

            else: raise ValueError(
                f"Outcome {i} (t={outcomes[i]:.4f}) does not precede cue {i+1} (t={cue_tone_onsets[i+1]:.4f}): "
                f"sequences are not properly interleaved."
            )

    if n_cues != n_outcomes:
        print(f"Warning: {n_cues} cues, {n_outcomes} outcomes "
              f"(1 trailing unmatched cue assumed — task cut off mid-trial).")
        if verbose:
            print('reward_tone_onsets', reward_tone_onsets)
            print('number of reward tones', len(reward_tone_onsets))
            print('timeout_tone_onsets', timeout_tone_onsets)
            print('number of timeout tones', len(timeout_tone_onsets))
            print('error_tone_onsets', error_tone_onsets)
            print('number of error tones', len(error_tone_onsets))
            print('cue_tone_onsets', cue_tone_onsets)
            print('number of cue tones', len(cue_tone_onsets))

    cue_tone_on_ts = nap.Ts(t=cue_tone_onsets, time_support=time_support)
    cue_tone_off_ts = nap.Ts(t=cue_tone_offsets, time_support=time_support)
    reward_tone_on_ts = nap.Ts(t=reward_tone_onsets, time_support=time_support)
    reward_tone_off_ts = nap.Ts(t=reward_tone_offsets, time_support=time_support)
    timeout_tone_on_ts = nap.Ts(t=timeout_tone_onsets, time_support=time_support)
    timeout_tone_off_ts = nap.Ts(t=timeout_tone_offsets, time_support=time_support)
    error_tone_on_ts = nap.Ts(t=error_tone_onsets, time_support=time_support)
    error_tone_off_ts = nap.Ts(t=error_tone_offsets, time_support=time_support)

    ts_list = [cue_tone_on_ts, cue_tone_off_ts,
               reward_tone_on_ts, reward_tone_off_ts,
               timeout_tone_on_ts, timeout_tone_off_ts,
               error_tone_on_ts, error_tone_off_ts]
    ts_index = ['start_tone_on', 'start_tone_off',
                'reward_tone_on', 'reward_tone_off',
                'timeout_tone_on', 'timeout_tone_off',
                'error_tone_on', 'error_tone_off']

    return ts_list, ts_index


def get_digital_ts(digital_events, time_support, map2imectime=None, din=None, event_name='lick'):
    if din is None:
        print("Warning: din not specified, using default channel 0 for lick events.")
        din = 0

    #lick_ontimes, lickofftimes = get_licks(digital_events, map2imectime)
    digchids = digital_events.channel_ids
    times = digital_events.get_event_times(digchids[din])
    #check that times is not empty
    if len(times) == 0:
        print('warning, no events found for digital channel', din)
        ontimes = []
        offtimes = []
    else:
        if map2imectime is not None:
            times = map2imectime(times)
        n_times = len(times)
        offtimes = times[n_times//2:]
        ontimes = times[:n_times//2]

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


def get_opto_ts(analog_events, nidq_t_aligned, idx=None, pwrs=None, plot=False, save_folder=None):
    if idx is None:
        print("Warning: idx not specified, using default channels for chrimson and chr2.")
        idx = {'chrimson': 6, 'chr2': 7}
    if pwrs is None:
        print("Warning: power levels not specified, using default power levels.")
        ints = [10,20,40,100,200,400]
        pwrs = {'chrimson': ints, 'chr2': ints}

    #get items from idx that contain 'chrimson' or 'chr2'
    idx = {k: v for k, v in idx.items() if k in ['chrimson', 'chr2']}
    ts_list = []
    ts_idx = []
    time_support = nap.IntervalSet(start=nidq_t_aligned[0], end=nidq_t_aligned[-1])
    for i in idx:
        ch = idx[i]
        pw = pwrs[i]
        trace = analog_events[:,ch]
        on_times, off_times, opto_trace = get_opto_edges(trace, nidq_t_aligned, pw, plot=plot, save_folder=save_folder, plot_label=i)
        ts_list.append(nap.Tsd(t=nidq_t_aligned, d=opto_trace, time_support=time_support))
        ts_idx.append(f"{i}_trace")
        for pidx, level in enumerate(pw):
            on = nap.Ts(t=on_times[pidx], time_support=time_support)
            off = nap.Ts(t=off_times[pidx], time_support=time_support)
            ts_list.append(on)
            ts_idx.append(f"{i}_on_{level}")
            ts_list.append(off)
            ts_idx.append(f"{i}_off_{level}")
    return ts_list, ts_idx

#for each start tone timestamp, find the closest outcome tone timestamp that follows
def parse_trials(start_tone_ts, outcome_tone_ts, lick_ts, require_first_licks=True, anchors=None):
    """Trials from start cues and one kind of outcome tone, with each trial's first lick.

    The first lick is the lick that triggered the outcome: the last lick between the trial's
    cue and its anchor. anchors : per outcome tone, the event the lick triggered (the solenoid
    opening of a rewarded trial, which precedes its reward tone by ~0.1 ms); the outcome tone
    itself by default.
    """
    anchors = (np.asarray(outcome_tone_ts.t) if anchors is None else np.asarray(anchors, dtype=float))
    #for each timestamp in outcome_tone_ts, find the closest timestamp in start_tone_ts that is less than it
    time_support = lick_ts.time_support
    trial_starts = []
    trial_ends = []
    first_licks = []
    start_cues = []
    outcome_tones = []
    for t, anchor in zip(outcome_tone_ts.t, anchors):
        #get timestamps of start_tone_ts.t that are less than outcome_tone.t[t]
        possible_starts = start_tone_ts.t[start_tone_ts.t < t]
        #ensure possible_starts are sorted
        possible_starts.sort()
        if len(possible_starts) > 0:
            trial_start = possible_starts[-1] #get the latest
            trial_starts.append(trial_start)
            start_cues.append(trial_start)
            trial_ends.append(t+7.0)
            outcome_tones.append(t)

            assert trial_start < t, "Trial start {} is not less than trial end {}".format(trial_start, t)
            trial_licks = lick_ts.t[(lick_ts.t >= trial_start) & (lick_ts.t <= anchor)]
            if len(trial_licks) > 0:
                first_lick = trial_licks[-1]
            else:
                first_lick = np.nan
            first_licks.append(first_lick)

        else:
            raise ValueError("No matching start tone found for outcome tone at time {}".format(t))


    #ensure that the lengths of trial_starts and trial_ends are the same
    assert len(trial_starts) == len(trial_ends) == len(first_licks) == len(outcome_tones)

    # if any first licks are nan, raise an error
    if require_first_licks:
        if any(np.isnan(first_licks)):
            #raise ValueError("Some trials are missing first licks: {}".format(first_licks))
            print("warning: some trials are missing first licks: {}".format(first_licks))
            #get the index of the nan
            nanidx = np.argwhere(np.isnan(first_licks))
            #remove the slice of first licks where nanidx
            first_licks = np.delete(first_licks, nanidx)
            #remove slice of trial_starts where nanidx
            trial_starts = np.delete(trial_starts, nanidx)
            trial_ends = np.delete(trial_ends, nanidx)
            start_cues = np.delete(start_cues, nanidx)
            outcome_tones = np.delete(outcome_tones, nanidx)

        first_licks_ts = nap.Ts(np.array(first_licks), time_support=time_support)
    else:
        first_licks_ts = None

    trial_starts = nap.Ts(np.array(trial_starts), time_support=time_support)
    trial_ends = nap.Ts(np.array(trial_ends), time_support=time_support)
    start_cues_ts = nap.Ts(np.array(start_cues), time_support=time_support)
    outcome_tones_ts = nap.Ts(np.array(outcome_tones), time_support=time_support)

    return trial_starts, trial_ends, start_cues_ts, outcome_tones_ts, first_licks_ts

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
            base_folder = Path(r"C:\Users\assad\Documents\recording_files\DS13\DS13_20250905")
        else:
            #query user to enter base folder in command line
            print(r"base folder example: C:\Users\assad\Documents\recording_files\DS13\DS13_20250822")
            base_folder_input = input("Please enter the base folder path: ")
            base_folder = Path(base_folder_input)

    elif not isinstance(base_folder, Path):
        base_folder = Path(base_folder)

    return base_folder

def get_spikeglx_folder(paths, gnum=None):
    #the lookup itself is SessionPaths.spikeglx_folder (see utils/file_structure.py), which always
    #searches the session's recording folder. All this wrapper adds is the console prompt for
    #choosing between several _g# runs, which does not belong in a shared layout module
    if gnum is None and len(paths.spikeglx_folders) > 1:
        candidates = paths.spikeglx_folders
        print(f"Multiple SpikeGLX folders found in {paths.recording_folder}, enter the number corresponding to the desired _g# folder:")
        gnum = input(f"Available folders: {[f.name for f in candidates]} \n Enter g# number: ")
    return paths.spikeglx_folder(gnum)

def get_imec_folder(paths, gnum=None):
    #the folder holding imec[#].ap.bin / imec[#].ap.meta. The run folder goes through the wrapper
    #above so a multi-run session is only asked about once, then the probe folder is found inside
    #whichever run was chosen -- which is why this is not simply paths.imec_folder(gnum)
    return fs.find_imec_folder(get_spikeglx_folder(paths, gnum))

def get_timing_pulse_din(dinmap):
    if 'timing_pulse' not in dinmap.keys():
        print("Warning: 'timing_pulse' not found in dinmap keys. Using default channel 1 for timing pulse.")
        pulse_din = 1
    else:
        pulse_din = dinmap['timing_pulse']
    return pulse_din

def get_maps_from_txt(paths):
    #SessionPaths.input_map finds the channel map beside the raw data, whatever the rig named it,
    #and is None when the session came without one
    config_file = paths.input_map
    if config_file is None:
        raise FileNotFoundError(
            f"No channel map ({' or '.join(fs.INPUT_MAP_FILES)}) found in "
            f"{paths.recording_folder}")

    vars_dict = {}
    with open(config_file) as f:
        tree = ast.parse(f.read())
        for node in tree.body:
            if isinstance(node, ast.Assign):
                name = node.targets[0].id
                value = ast.literal_eval(node.value)
                vars_dict[name] = value

    #check that dinmap, ainmap, and optoidx are in vars_dict
    keys = ['dinmap', 'ainmap', 'optoidx']
    for key in keys:
        if key not in vars_dict.keys():
            vars_dict[key] = None
            print(f"Warning: {key} not found in {config_file}, using None for {key}.")

    dinmap = vars_dict["dinmap"]
    ainmap = vars_dict["ainmap"]
    optoidx = vars_dict["optoidx"]

    return dinmap, ainmap, optoidx

def parse_channel_maps(paths, dinmap, ainmap, optoidx):
    #if all of dinmap, ainmap, and optoidx are None, try to get them from the session's channel map
    #dinmap should be a dictionary mapping digital channel ids to event names
    #ainmap should be a dictionary mapping analog channel ids to event names
    if (dinmap is not None) and (ainmap is not None) and (optoidx is not None):
        return dinmap, ainmap, optoidx
    elif (dinmap is None) and (ainmap is None) and (optoidx is None):
        if paths.input_map is not None:
            dinmap, ainmap, optoidx = get_maps_from_txt(paths)
        else:
            print("Warning: No channel map file found in the recording folder, using default channel maps.")
            dinmap = default_dinmap
            ainmap = default_ainmap
            optoidx = default_optoidx
    else:
        raise ValueError("Either provide all of dinmap, ainmap, and optoidx, or none to use nidaq_map.txt or defaults.")

    return dinmap, ainmap, optoidx

def get_output_folders(paths):
    """Resolve where a binary-signals run writes (see utils/file_structure.py).

    The event tables and threshold plots stay with the rest of the analysis products in the
    pynapple folder; binary_signals.npz itself lives in its own binary_signals folder, because
    sorting needs a copy of it too (kept in step by paths.mirror_binary_signals()).

    Creates the session's output folders first, so this works whether or not the SessionPaths it
    was handed was built with create=True -- without that the accessors below would be None.
    """
    paths.create_output_folders()
    pynapple_folder = paths.pynapple_folder
    pynapple_folder.mkdir(parents=True, exist_ok=True)
    binary_signals_file = paths.binary_signals_file()
    binary_signals_file.parent.mkdir(parents=True, exist_ok=True)

    return pynapple_folder, binary_signals_file

def load_pynapple_file(pynapple_file):
    if pynapple_file.exists():
        print(f"Loading existing binary signals from {pynapple_file}, set overwrite=True to re-parse.")
        bs = nap.load_file(str(pynapple_file))
        return bs
    else:
        print(f"No existing binary signals found at {pynapple_file}, parsing new signals.")
        return None


def _split_event_label(event_name):
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


def pool_task_stim(tsg, opsin, min_dur=0.5):
    """Task-related opto pulses for one opsin, as (on_times, off_times, intensities).

    Pulses shorter than min_dur are the ~100ms calibration flashes fired before and after the
    session rather than stimulation delivered on a trial, so they are dropped. Intensity comes out
    of the event label (chrimson_on_40 -> 40) and is nan on the digital opto path, whose labels
    carry no power level. Sessions with no opto channel give three empty arrays.
    """
    meta = tsg.metadata

    def _pool(edge):
        prefix = f'{opsin}_{edge}'
        times, levels = [], []
        for key, row in meta.iterrows():
            name = row.get('event')
            #startswith rather than == so both label styles are caught: the analog path names each
            #power separately (chrimson_on_40), the digital path has the bare chrimson_on. It also
            #keeps the continuous chrimson_trace out, which has no edges to pool
            if not isinstance(name, str) or not name.startswith(prefix):
                continue
            m = re.match(rf'^{re.escape(prefix)}_(\d+)$', name)
            t = np.asarray(tsg[key].t)
            times.append(t)
            levels.append(np.full(len(t), float(m.group(1)) if m else np.nan))
        if not times:
            return np.array([]), np.array([])
        t = np.concatenate(times)
        lv = np.concatenate(levels)
        order = np.argsort(t)
        return t[order], lv[order]

    on_all, on_levels = _pool('on')
    off_all, _ = _pool('off')
    if len(on_all) == 0 or len(off_all) == 0:
        return np.array([]), np.array([]), np.array([])

    task_on, task_off, task_levels = [], [], []
    for t_on, level in zip(on_all, on_levels):
        following = off_all[off_all > t_on]
        if len(following) > 0 and (following[0] - t_on) > min_dur:
            task_on.append(t_on)
            task_off.append(following[0])
            task_levels.append(level)
    return np.array(task_on), np.array(task_off), np.array(task_levels)


def plot_trial_raster(tsg, ch_on=None, ch_off=None, c2_on=None, c2_off=None,
                      ch_intensities=None, c2_intensities=None,
                      save_path=None, t_start=-1.0, t_end=8.0):
    """Trial raster: one row per start cue, aligned at t=0.

    - Black hash: licks
    - Green dot: reward tone onset
    - Red dot: error tone onset
    - Blue band: chr2 opto stim (alpha scaled by intensity if provided)
    - Orange band: chrimson opto stim (alpha scaled by intensity if provided)

    This is the parse-time QC view of the events that were just extracted -- a session whose tones,
    licks and solenoid were mapped correctly shows licks bunching after the cue and outcome dots
    landing at trial ends. analysis/pres_plot_utils.plot_trial_raster is the figure-quality sibling
    of this, sized for a presentation panel and driven off a saved binary_signals file.
    """
    import matplotlib.patches as mpatches

    meta = tsg.metadata

    def _ts(event_name):
        idx = meta[meta['event'] == event_name].index
        return np.asarray(tsg[idx[0]].t) if len(idx) > 0 else np.array([])

    cue_ts = _ts('start_cues')
    rew_ts = _ts('reward_tones')
    err_ts = _ts('early_tones')
    #the raw lick line only. every other lick-named event (bout edges, per-outcome first licks) is
    #a subset of it, so pooling them would just draw the same licks two and three times over
    lick_ts = _ts('lick_on')

    n_trials = len(cue_ts)
    if n_trials == 0:
        print("Warning: no start cues found, skipping trial raster.")
        return None

    ch_on  = np.asarray(ch_on  if ch_on  is not None else [])
    ch_off = np.asarray(ch_off if ch_off is not None else [])
    c2_on  = np.asarray(c2_on  if c2_on  is not None else [])
    c2_off = np.asarray(c2_off if c2_off is not None else [])

    def _alpha_arr(n_pulses, intensities, default=0.3):
        #darker band = brighter pulse. an all-nan intensity array is the digital opto path, which
        #never recorded a power level, so every band gets the flat default
        if intensities is None or len(intensities) == 0:
            return np.full(n_pulses, default)
        arr = np.asarray(intensities, dtype=float)
        if not np.any(np.isfinite(arr)):
            return np.full(n_pulses, default)
        lo, hi = np.nanmin(arr), np.nanmax(arr)
        norm = (arr - lo) / (hi - lo) if hi > lo else np.ones_like(arr)
        return np.where(np.isfinite(norm), 0.12 + 0.38 * norm, default)

    ch_alphas = _alpha_arr(len(ch_on), ch_intensities)
    c2_alphas = _alpha_arr(len(c2_on), c2_intensities)

    fig_h = max(4, min(n_trials * 0.15 + 1, 40))
    fig, ax = plt.subplots(figsize=(7, fig_h))

    #band colour is the wavelength each opsin is driven at, which is what the legend promises:
    #chrimson orange, chr2 blue. the pres_plot_utils original has these two the other way round
    stim_layers = [
        (ch_on, ch_off, (1.0, 0.5, 0.0), ch_alphas),
        (c2_on, c2_off, (0.1, 0.4, 1.0), c2_alphas),
    ]

    for ti, ct in enumerate(cue_ts):
        ws, we = ct + t_start, ct + t_end

        for son, soff, rgb, alphas in stim_layers:
            #pi, not the si of the pres_plot_utils original: si is spikeinterface in this module
            for pi in range(min(len(son), len(soff))):
                ton, toff = son[pi], soff[pi]
                if toff < ws or ton > we:
                    continue
                x0 = max(ton, ws) - ct
                x1 = min(toff, we) - ct
                ax.add_patch(mpatches.Rectangle(
                    (x0, ti - 0.5), x1 - x0, 1.0,
                    facecolor=rgb, alpha=float(alphas[pi]) if pi < len(alphas) else 0.3,
                    linewidth=0, zorder=1,
                ))

        licks_in = lick_ts[(lick_ts >= ws) & (lick_ts < we)]
        if len(licks_in):
            ax.plot(licks_in - ct, np.full(len(licks_in), ti),
                    '|', color='k', markersize=6, markeredgewidth=1.0, zorder=2)

        rews_in = rew_ts[(rew_ts >= ws) & (rew_ts < we)]
        if len(rews_in):
            ax.plot(rews_in - ct, np.full(len(rews_in), ti),
                    'o', color='limegreen', markersize=4, zorder=3)

        errs_in = err_ts[(err_ts >= ws) & (err_ts < we)]
        if len(errs_in):
            ax.plot(errs_in - ct, np.full(len(errs_in), ti),
                    'o', color='red', markersize=4, zorder=3)

    ax.set_xlim(t_start, t_end)
    ax.set_ylim(n_trials - 0.5, -0.5)
    ax.axvline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axvline(3.33, color='green', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.set_xlabel('t(s) from cue')
    ax.set_ylabel('Trial')
    ax.set_title(f'Trial raster (n={n_trials} trials)')

    legend_handles = [
        plt.Line2D([0], [0], marker='|', color='k', linestyle='none',
                   markersize=8, markeredgewidth=1.5, label='Lick'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='limegreen',
                   markersize=6, label='Reward'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                   markersize=6, label='Early'),
        mpatches.Patch(facecolor=(0.1, 0.4, 1.0), alpha=0.5, label='Blue/ChR2 stim'),
        mpatches.Patch(facecolor=(1.0, 0.5, 0.0), alpha=0.5, label='Orange/ChrimsonR stim'),
    ]
    ax.legend(handles=legend_handles, ncol=1, framealpha=0.9,
              loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)

    if save_path is not None:
        #overwritten rather than uniquified: this is regenerated on every get_binary_signals call,
        #cached loads included, so numbered copies would pile up over repeat runs of one session
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved trial raster to {save_path}")
        plt.close(fig)

    return fig


def _warn_if_licks_recoverable(bs, paths, lick_source):
    """A cached parse with no licks, for a session that has an event table to recover them from."""
    if lick_source == 'nidq' or paths.event_table is None:
        return
    lick_rows = bs.metadata.index[bs.metadata['event'] == 'lick_on']
    if len(lick_rows) and len(bs[lick_rows[0]].t) == 0:
        print(f"Warning: the cached binary signals have no licks, but {paths.event_table.name} "
              f"has them. Back up {paths.binary_signals_file().name} and re-parse with "
              f"overwrite=True to recover them.")


def save_trial_raster(tsg, paths):
    """Write the QC trial raster into the session's analysis-side binary_signals folder.

    Called on every get_binary_signals return, freshly parsed or loaded from cache, and never
    allowed to take the run down with it: by this point the npz is already on disk and the parse
    has succeeded, so a matplotlib failure or an event this session happens not to have is worth a
    warning, not the loss of a good parse.
    """
    try:
        ch_on, ch_off, ch_int = pool_task_stim(tsg, 'chrimson')
        c2_on, c2_off, c2_int = pool_task_stim(tsg, 'chr2')
        plot_trial_raster(tsg,
                          ch_on=ch_on, ch_off=ch_off, c2_on=c2_on, c2_off=c2_off,
                          ch_intensities=ch_int, c2_intensities=c2_int,
                          save_path=paths.binary_signals_folder() / 'trial_raster.png')
    except Exception as e:
        print(f"Warning: could not make trial raster ({type(e).__name__}: {e})")


LICK_SOURCES = ('auto', 'nidq', 'event_table')
#beside binary_signals.npz: the record of where an event-table session's licks came from
ETL_DIAGNOSTICS_STEM = 'lick_event_table_alignment'


def get_lick_ts(digital_events, time_support, map2imectime, dinmap, paths, recorded_events,
                lick_source='auto'):
    """Lick onsets/offsets as [lick_on Ts, lick_off Ts], ['lick_on', 'lick_off'].

    lick_source picks where they come from:
      'nidq'        -- the NIDQ lick line (dinmap['licks']) only, even when it is empty.
      'event_table' -- the behaviour computer's <session>_eventTable.csv, placed on the
                       binary_signals clock by fitting the task events it shares with the NIDQ
                       (recorded_events, from this same parse; see event_table_licks.py).
      'auto'        -- the NIDQ, unless its lick line is absent or empty and the session has an
                       event table, i.e. the BNC copy of the lick line was not recorded.
    The event table logs lick onsets only, so lick_off is left empty there -- the same as an empty
    NIDQ channel gives, which keeps the event layout of every session the same. Nothing derived
    from licks uses the offsets.
    """
    if lick_source not in LICK_SOURCES:
        raise ValueError(f"lick_source must be one of {LICK_SOURCES}, not {lick_source!r}")
    names = ['lick_on', 'lick_off']
    diag_folder = paths.binary_signals_folder()

    nidq_list = None
    if 'licks' in dinmap.keys() and lick_source != 'event_table':
        nidq_list, _ = get_digital_ts(digital_events, time_support, map2imectime, dinmap['licks'])
    elif lick_source == 'nidq':
        raise ValueError("Lick channel not found in dinmap.")

    nidq_empty = nidq_list is None or len(nidq_list[0]) == 0
    use_table = lick_source == 'event_table' or (lick_source == 'auto' and nidq_empty)
    table_file = paths.event_table if use_table else None
    if table_file is None:
        if lick_source == 'event_table':
            raise FileNotFoundError(f"lick_source='event_table' but {paths.recording_folder} has no "
                                    f"*{fs.EVENT_TABLE_SUFFIX}")
        if nidq_list is None:
            raise ValueError("Lick channel not found in dinmap, and no event table to take licks from.")
        if nidq_empty:
            recover = ("lick_source='auto' or 'event_table' would take them from "
                       f"{paths.event_table.name}" if paths.event_table is not None
                       else "there is no event table to recover them from")
            print(f"Warning: the NIDQ lick channel is empty and {recover}; lick-derived events "
                  f"(first licks, cues, bouts) will be empty.")
        #licks from the NIDQ: drop any alignment record a previous event-table parse left, so the
        #binary_signals folder never claims a source its npz no longer has
        for stale in diag_folder.glob(f"{ETL_DIAGNOSTICS_STEM}.*"):
            stale.unlink()
        return nidq_list, names

    if lick_source == 'auto':
        print(f"NIDQ lick channel {'is empty' if nidq_list is not None else 'is not mapped'}; "
              f"taking licks from the event table instead.")
    licks, diagnostics = etl.event_table_licks(table_file, recorded_events, time_support)
    print(etl.summarize(diagnostics))
    try:
        json_file, png_file = etl.save_diagnostics(diagnostics, diag_folder, stem=ETL_DIAGNOSTICS_STEM)
        print(f"Event-table alignment diagnostics saved to {json_file} and {png_file.name}")
    except Exception as e:
        print(f"Warning: could not save event-table alignment diagnostics ({type(e).__name__}: {e})")
    return [nap.Ts(t=licks, time_support=time_support),
            nap.Ts(t=np.array([]), time_support=time_support)], names



def get_binary_signals(base_folder=None, dinmap=None,ainmap=None, optoidx=None, overwrite=False, plot=False, test_mode=False, plot_save_folder=None,
                       lick_source='auto', tone_method='auto', save=True):
    """Parse a session's NIDQ into the binary_signals TsGroup, or load the cached one.

    lick_source: 'auto' | 'nidq' | 'event_table' -- where the licks come from; see get_lick_ts.
    The default takes them from the session's event table only when the NIDQ lick line is empty.
    It applies only when parsing (overwrite=True, or no cached npz): a cached load is returned as
    saved, whatever its lick source.
    tone_method: 'auto' | 'frequency' | 'duration' -- how the digital tone line is classified;
    see parse_tones_digital (tone frequencies from the rig's `tonemap`, see get_tonemap).
    save: False parses into memory only -- nothing is written (no npz, mirror, csv or raster).
    """
    #section 1: folder handling. one SessionPaths resolves every location this run needs -- raw
    #data and the channel map out of recording_files, products into analysis_files -- so nothing
    #below has to re-derive a path from base_folder
    base_folder = parse_base_folder(base_folder, test_mode=test_mode)
    paths = fs.SessionPaths(base_folder, create=True)
    pynapple_folder, pynapple_file = get_output_folders(paths)
    spikeglx_folder =paths.spikeglx_folder()
    imec_folder = paths.imec_folder(fail_quietly=True)
    event_csv_file = pynapple_folder / "binary_signals_events.csv"
    if plot_save_folder is None:
        plot_save_folder = pynapple_folder

    if not overwrite:
        bs = load_pynapple_file(pynapple_file)
        if bs is not None:
            if lick_source != 'auto':
                print(f"Note: lick_source={lick_source!r} is ignored for a cached load; pass "
                      f"overwrite=True to re-parse with it.")
            _warn_if_licks_recoverable(bs, paths, lick_source)
            paths.mirror_binary_signals()
            export_event_table(bs, event_csv_file)
            save_trial_raster(bs, paths)
            return bs

    #section 2: channel mapping
    dinmap, ainmap, optoidx = parse_channel_maps(paths, dinmap, ainmap, optoidx)

    #section 3: read in spikeglx data
    if imec_folder is not None:
        #imec_rec = si.read_spikeglx(imec_folder, stream_name='imec0.ap')#, load_sync_channel=True)
        sync_rec = si.read_spikeglx(imec_folder, stream_name='imec0.ap-SYNC')
    else:
        sync_rec = None
    nidq_rec = si.read_spikeglx(spikeglx_folder, stream_id='nidq')
    stream_names = si.get_neo_streams('spikeglx', spikeglx_folder)

    chids = nidq_rec.channel_ids #get all channel ids--channels are indexed using alphanumeric string identifier
    anachids = [chid for chid in chids if 'D' not in str(chid)] #get analog channel ids (ones without 'D' in them)

    digital_events = si.read_spikeglx_event(spikeglx_folder) #get raw digital events
    analog_events = nidq_rec.get_traces(channel_ids=anachids)

    #section 4: align nidq to imec time
    pulse_din = get_timing_pulse_din(dinmap)
    nidq_t = nidq_rec.get_times()
    nidq_t = nidq_t - nidq_t[0]  # zeroing time
    if sync_rec is not None:
        map2imectime = load_cached_alignment(paths, spikeglx_folder, imec_folder)
        if map2imectime is None:
            map2imectime = get_nidq_aligner(sync_rec, digital_events, pulse_din)
            if save:
                save_alignment(paths, spikeglx_folder, imec_folder, map2imectime.coeffs,
                               source='fitted on the imec SYNC and NIDQ timing pulses')
        nidq_t_aligned = map2imectime(nidq_t)
    else:
        nidq_t_aligned = nidq_t
        map2imectime = None
    time_support = nap.IntervalSet(start=nidq_t_aligned[0], end=nidq_t_aligned[-1])

    #analog event index
    ts_list = []
    ts_idx = []
    if 'solenoid' in dinmap.keys():
        solenoid_din = dinmap['solenoid']
        solenoid_ts_list, solenoid_ts_idx = get_digital_ts(digital_events,time_support, map2imectime, solenoid_din, event_name='solenoid')
    elif 'solenoid' in ainmap.keys():
        solenoid_ts_list, solenoid_ts_idx = parse_solenoid_analog(analog_events, nidq_t_aligned, idx=ainmap, plot=plot)
    else:
        raise ValueError("Solenoid channel not found in either dinmap or ainmap.")
    ts_list += solenoid_ts_list
    ts_idx += solenoid_ts_idx

    if 'tone' in dinmap.keys():
        tones_ts_list, tones_ts_idx = parse_tones_digital(digital_events, solenoid_ts_list, map2imectime, time_support, idx=dinmap,
                                                          tonemap=get_tonemap(paths), method=tone_method)
    elif 'tone' in ainmap.keys():
        tones_ts_list, tones_ts_idx = parse_tones_analog(analog_events, solenoid_ts_list, nidq_t_aligned, idx=ainmap,
                                                         tonemap=get_tonemap(paths), method=tone_method)
    else:
        raise ValueError("Tone channel not found in either dinmap or ainmap.")
    ts_list += tones_ts_list
    ts_idx += tones_ts_idx

    #the task events recorded so far (solenoid, tones) are what an event table is aligned on
    recorded_events = {name: ts.t for ts, name in zip(ts_list, ts_idx)}
    lick_ts_list, lick_ts_idx = get_lick_ts(digital_events, time_support, map2imectime, dinmap, paths,
                                            recorded_events, lick_source=lick_source)

    ts_list += lick_ts_list
    ts_idx += lick_ts_idx

    if ('chrimson' in ainmap.keys()) or ('chr2' in ainmap.keys()):
        opto_ts_list, opto_ts_idx = get_opto_ts(analog_events, nidq_t_aligned, idx=ainmap, pwrs=optoidx, plot=plot, save_folder=plot_save_folder)
        ts_list += opto_ts_list
        ts_idx += opto_ts_idx

    if ('chrimson' in dinmap.keys()):
        chrimson_ts_list, chrimson_ts_idx = get_digital_ts(digital_events, time_support, map2imectime, dinmap['chrimson'], event_name='chrimson')
        ts_list += chrimson_ts_list
        ts_idx += chrimson_ts_idx

    if ('chr2' in dinmap.keys()):
        chr2_ts_list, chr2_ts_idx = get_digital_ts(digital_events, time_support, map2imectime, dinmap['chr2'], event_name='chr2')
        ts_list += chr2_ts_list
        ts_idx += chr2_ts_idx

    ts_numbers = [i for i in range(len(ts_list))]
    ts_dict = dict(zip(ts_numbers, ts_list))
    ts_idx_dict = dict(zip(ts_numbers, ts_idx))
    ts_idx_df = pd.DataFrame.from_dict(ts_idx_dict, orient='index', columns=['event'])

    tsg = nap.TsGroup(ts_dict, metadata=ts_idx_df)

    metadata = tsg.metadata
    # get row of metadata where event is called "start tone on"
    start_tone_idx = metadata[metadata['event'] == 'start_tone_on'].index[0]
    timeout_tone_idx = metadata[metadata['event'] == 'timeout_tone_on'].index[0]
    error_tone_idx = metadata[metadata['event'] == 'error_tone_on'].index[0]
    reward_tone_idx = metadata[metadata['event'] == 'reward_tone_on'].index[0]
    lick_ts_idx = metadata[metadata['event'] == 'lick_on'].index[0]

    start_tone_ts = tsg[start_tone_idx]
    timeout_tone_ts = tsg[timeout_tone_idx]
    error_tone_ts = tsg[error_tone_idx]
    reward_tone_ts = tsg[reward_tone_idx]
    all_outcome_tone_ts = nap.Ts(np.sort(np.concatenate((reward_tone_ts.t, error_tone_ts.t, timeout_tone_ts.t))),
                                 time_support=time_support)
    lick_ts = tsg[lick_ts_idx]


    if len(all_outcome_tone_ts.t) != len(start_tone_ts.t):
        start_tone_ts, all_outcome_tone_ts = match_starts_and_ends(start_tone_ts, all_outcome_tone_ts)

    ITI_starts = nap.Ts(all_outcome_tone_ts.t[:-1] + 3.7)
    ITI_ends = nap.Ts(start_tone_ts.t[1:] - 1.5)

    #a rewarded trial's first lick is the one that opened the solenoid, which the reward tone
    #follows by ~0.1 ms: the lick search ends at the opening (the tone where none is matched)
    sol_on = np.sort(tsg[metadata[metadata['event'] == 'solenoid_on'].index[0]].t)
    j = np.searchsorted(sol_on, reward_tone_ts.t + 1e-9, side='right') - 1
    prev_sol = np.where(j >= 0, sol_on[np.clip(j, 0, None)], -np.inf) if len(sol_on) else np.full(len(reward_tone_ts.t), -np.inf)
    reward_anchors = np.where(reward_tone_ts.t - prev_sol <= 0.002, prev_sol, reward_tone_ts.t)
    reward_trials_starts, reward_trials_ends, rewarded_cues_ts, reward_tones_ts, rewarded_first_licks_ts = parse_trials(start_tone_ts, reward_tone_ts, lick_ts,
                                                                                                                      anchors=reward_anchors)
    error_trials_starts, error_trials_ends, early_cues_ts, early_tones_ts, early_first_licks_ts = parse_trials(start_tone_ts, error_tone_ts, lick_ts)
    timeout_trials_starts, timeout_trials_ends, timeout_cues_ts, timeout_tones_ts, _  = parse_trials(start_tone_ts, timeout_tone_ts, lick_ts, require_first_licks=False)

    #redefine start_tone_ts to be the composite of reward_trials_starts, error_trials_starts, and timeout_trials_starts
    start_cues = np.concatenate((early_cues_ts.t, rewarded_cues_ts.t, timeout_cues_ts.t))
    outcome_tones = np.concatenate((early_tones_ts.t, reward_tones_ts.t, timeout_tones_ts.t))
    start_cue_ts = nap.Ts(np.sort(start_cues), time_support=time_support)
    outcome_tones_ts = nap.Ts(np.sort(outcome_tones), time_support=time_support)


    lick_bout_starts_ts, lick_bout_ends_ts = parse_lick_bouts(lick_ts)

    first_licks_metadata = pd.DataFrame.from_dict({
        1: 'rewarded_first_licks',
        2: 'early_first_licks',
        3: 'rewarded_cues',
        4: 'early_cues',
        5: 'timeout_cues',
        6: 'start_cues',
        7: 'reward_tones',
        8: 'early_tones',
        9: 'timeout_tones',
        10: 'outcome_tones',
        11: 'ITI_starts',
        12: 'ITI_ends',
        13: 'lick_bout_starts',
        14: 'lick_bout_ends'}, orient='index', columns=['event'])
    first_licks_ts = nap.TsGroup({1: rewarded_first_licks_ts,
                                  2: early_first_licks_ts,
                                  3: rewarded_cues_ts,
                                  4: early_cues_ts,
                                  5: timeout_cues_ts,
                                  6: start_cue_ts,
                                  7: reward_tones_ts,
                                  8: early_tones_ts,
                                  9: timeout_tones_ts,
                                  10: outcome_tones_ts,
                                  11: ITI_starts,
                                  12: ITI_ends,
                                  13: lick_bout_starts_ts,
                                  14: lick_bout_ends_ts
                                  },
                                 metadata=first_licks_metadata, time_support=time_support)




    # mege first_licks_ts into tsg
    tsg = tsg.merge(first_licks_ts, reset_index=True)
    if not save:
        return tsg
    tsg.save(str(pynapple_file))
    paths.mirror_binary_signals()
    export_event_table(tsg, event_csv_file)
    save_trial_raster(tsg, paths)

    return tsg

def test():
    dinmap = None
    ainmap = None
    optoidx = None
    base_folder = None
    test_mode=False
    overwrite=True
    plot=True
    plot_save_folder = None
    #base_folder = Path(r'C:\Users\assad\Documents\recording_files\DS13\DS13_20250824')
    #base_folder = Path(r"C:\Users\assad\Desktop\experiment_data\recording_files\DS48\DS48_20260729")
    #base_folder = Path(r"C:\Users\assad\Desktop\experiment_data\recording_files\DS15\DS15_20260703")
    base_folder = Path(r"C:\Users\assad\Desktop\experiment_data\recording_files\DS45\DS45_20260814")
    tsg = get_binary_signals(base_folder,dinmap,ainmap,optoidx,overwrite,plot,test_mode)