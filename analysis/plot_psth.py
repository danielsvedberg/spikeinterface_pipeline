import pynapple as nap
import numpy as np
from pathlib import Path
from utils.mpl_backend import select_backend
select_backend()  # Qt in the main process, headless Agg in joblib workers
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.transforms as mtransforms
import re
#joblib
from joblib import Parallel, delayed
from utils import file_structure as fs

def plot_psth(timestamps, tref, unitID, event_ID, save_folder=None, minmax=(-1, 1), off_event=None):

    peth = nap.compute_perievent(
        timestamps=timestamps,
        tref=tref,
        minmax=minmax,
        time_unit="s")

    #if off_event is a ts or array
    if off_event is not None and hasattr(off_event, 't'):
        off_events = off_event.t-tref.t
        off_event = np.median(off_events)

    spikes = peth.to_tsd()
    #check that spikes is not empty
    if len(spikes) == 0:
        print(f"No spikes found for Unit {unitID} around event {event_ID}")
        return

    fig, axs = plt.subplots(2,1, figsize=(10, 6), sharex=True)

    ax = axs[0]
    ax.plot(np.mean(peth.count(0.01), 1) / 0.01, linewidth=2, color="black")
    ax.set_ylabel("Rate (spikes/sec, 0.01s bins)")

    ax = axs[1]
    #get the number of trials
    ntrials = max(peth) - min(peth)
    ms = -1/1000 * ntrials  + 3
    ax.plot(spikes, "|", markersize=ms, color="black", mew=1)
    ax.set_ylabel("trial/event #")
    ax.set_xlabel("time from event (s)")

    for i in range(2):
        ax = axs[i]
        ax.set_xlim(minmax)
        ax.axvline(0.0)
        if off_event is not None:
            ax.axvline(off_event, color='black', linestyle='--')
    #add a suptitle
    suptitle = f"Unit {unitID} PSTH around event {event_ID}"
    fig.suptitle(suptitle)
    if save_folder is not None:
        fig.savefig(save_folder / f"Unit_{unitID}_PSTH_event_{event_ID}.png")
        plt.close(fig)
    else:
        plt.show()

def plot_on_events_psth(ephys_data, sigs, pynapple_folder):
    sigs_metadata = sigs.metadata
    on_events_idx = sigs_metadata[sigs_metadata['event'].str.contains('_on')].index.tolist()

    neurons_idx = ephys_data.keys()

    psth_folder = pynapple_folder / "psth_plots"
    psth_folder.mkdir(exist_ok=True)
    for event in on_events_idx:
        tref = sigs[event]
        event_name = sigs_metadata.loc[event, 'event']
        parallel_ = Parallel(n_jobs=-1, verbose=5)
        parallel_(
            delayed(plot_psth)(
                ephys_data[neuron],
                tref,
                neuron,
                event_name,
                save_folder=psth_folder
            ) for neuron in neurons_idx)

    cue_events_idx = sigs_metadata[sigs_metadata['event'].str.contains('_cues')].index.tolist()
    for event in cue_events_idx:
        tref = sigs[event]
        event_name = sigs_metadata.loc[event, 'event']
        parallel_ = Parallel(n_jobs=-1, verbose=5)
        parallel_(
            delayed(plot_psth)(
                ephys_data[neuron],
                tref,
                neuron,
                event_name,
                save_folder=psth_folder,
                minmax=(-2, 2)
            ) for neuron in neurons_idx)

    first_licks_idx = sigs_metadata[sigs_metadata['event'].str.contains('_first_licks')].index.tolist()
    for event in first_licks_idx:
        tref = sigs[event]
        event_name = sigs_metadata.loc[event, 'event']
        parallel_ = Parallel(n_jobs=-1, verbose=5)
        parallel_(
            delayed(plot_psth)(
                ephys_data[neuron],
                tref,
                neuron,
                event_name,
                save_folder=psth_folder,
                minmax=(-4, 1)
            ) for neuron in neurons_idx)

    opto_events = sigs_metadata[sigs_metadata['event'].str.contains('chr')]
    opto_off_events_idx = opto_events[opto_events['event'].str.contains('_off')].index.tolist()
    #only get opto events that end with '_on'
    opto_on_events_idx = opto_events[opto_events['event'].str.contains('_on')].index.tolist()
    for i, event in enumerate(opto_on_events_idx):
        tref = sigs[event]
        off_event = sigs[opto_off_events_idx[i]]
        event_name = sigs_metadata.loc[event, 'event']

        parallel_ = Parallel(n_jobs=-1, verbose=5)
        parallel_(
            delayed(plot_psth)(
                ephys_data[neuron],
                tref,
                neuron,
                event_name,
                off_event=off_event,
                save_folder=psth_folder,
                minmax=(-0.5,0.5)
            ) for neuron in neurons_idx)


def _extract_power(event_name):
    """Extract the trailing integer from an event name like 'chrimson_on_40'."""
    m = re.search(r"(\d+)$", str(event_name))
    return int(m.group(1)) if m else None


def _draw_opto_psth_on_axes(timestamps, opsin, power_trefs, power_off_events,
                            ax_psth, ax_raster, minmax=(-0.5, 0.5),
                            fontsize=15, show_legend=True, show_ylabels=True,
                            show_title=True, show_power_labels=True):
    """
    Draw a combined PSTH + raster for one unit / one opsin onto the provided axes.

    Parameters
    ----------
    timestamps : nap.Ts
        Spike timestamps for a single unit.
    opsin : str
        Opsin name (used only for the column title).
    power_trefs : list of (power, tref)
        Sorted by power.
    power_off_events : list of (power, nap.Ts | None)
        Matching off-event timestamps.
    ax_psth : matplotlib.axes.Axes
        Axes for the PSTH (top).
    ax_raster : matplotlib.axes.Axes
        Axes for the raster (bottom).
    minmax : tuple
        Pre/post window in seconds.
    fontsize : int
        Font size for all labels and titles.
    show_legend : bool
        Whether to draw the power legend on the PSTH axes.
    show_ylabels : bool
        Whether to draw y-axis labels on both axes.
    """
    if len(power_trefs) == 0:
        if show_title:
            ax_psth.set_title(opsin, fontsize=fontsize)
        ax_psth.text(0.5, 0.5, 'no events', transform=ax_psth.transAxes,
                     ha='center', va='center', fontsize=fontsize, color='grey')
        return

    _OPSIN_CMAPS = {'chrimson': plt.cm.Oranges, 'chr2': plt.cm.Blues}
    cmap = _OPSIN_CMAPS.get(opsin, plt.cm.viridis)
    n_powers = len(power_trefs)
    colors = [cmap(0.65 + 0.35 * i / max(n_powers - 1, 1)) for i in range(n_powers)]

    # --- compute peri-event for each power level ---
    all_peths = []
    all_spike_tsds = []
    trial_counts = []
    off_medians = []

    for (power, tref), (_, off_ts) in zip(power_trefs, power_off_events):
        peth = nap.compute_perievent(timestamps=timestamps, tref=tref,
                                     minmax=minmax, time_unit="s")
        spikes_tsd = peth.to_tsd()
        all_peths.append(peth)
        all_spike_tsds.append(spikes_tsd)
        n_trials = len(tref)
        trial_counts.append(n_trials)

        if off_ts is not None and hasattr(off_ts, 't') and len(off_ts.t) > 0:
            off_rel = off_ts.t - tref.t
            off_medians.append(float(np.median(off_rel)))
        else:
            off_medians.append(None)

    total_trials = sum(trial_counts)
    if total_trials == 0:
        if show_title:
            ax_psth.set_title(opsin, fontsize=fontsize)
        return

    # ---- PSTH ----
    bin_size = 0.005
    all_rates = []
    for i, (power, _) in enumerate(power_trefs):
        rate = np.mean(all_peths[i].count(bin_size), axis=1) / bin_size
        all_rates.append(rate)
        ax_psth.plot(rate, linewidth=0.75, alpha=0.75, color=colors[i],
                     label=f"{power} µW")
    # grand average
    if len(all_rates) > 0:
        grand_avg = np.mean(
            np.column_stack([r.values if hasattr(r, 'values') else r for r in all_rates]),
            axis=1)
        ref = all_rates[0]
        if hasattr(ref, 'index'):
            ax_psth.plot(ref.index, grand_avg, linewidth=2, color='black', label='mean')
        else:
            ax_psth.plot(grand_avg, linewidth=2, color='black', label='mean')
    if show_ylabels:
        ax_psth.set_ylabel("spks/s", fontsize=fontsize)
    if show_legend:
        ax_psth.legend(fontsize=fontsize, loc='lower left',
                       bbox_to_anchor=(1.01, 0), borderaxespad=0, ncol=1)
    if show_title:
        ax_psth.set_title(opsin, fontsize=fontsize)

    # ---- Raster ----
    trial_offset = 0
    tranche_mids = []
    for i, (power, _) in enumerate(power_trefs):
        start = trial_offset
        spikes_tsd = all_spike_tsds[i]
        if len(spikes_tsd) == 0:
            trial_offset += trial_counts[i]
            tranche_mids.append((start + trial_counts[i] / 2, power, colors[i]))
            continue
        spike_times = spikes_tsd.index
        trial_ids = spikes_tsd.values + trial_offset - min(all_peths[i])
        ms = max(-total_trials / 1000 * 0.8 + 1.5, 0.2)
        ax_raster.plot(spike_times, trial_ids, "|",
                       markersize=ms, color=colors[i], mew=0.3)
        trial_offset += trial_counts[i]
        tranche_mids.append((start + trial_counts[i] / 2, power, colors[i]))

    cum = 0
    for i in range(n_powers - 1):
        cum += trial_counts[i]
        ax_raster.axhline(cum, color='grey', linewidth=0.5, linestyle='-')

    if show_power_labels:
        trans = mtransforms.blended_transform_factory(ax_raster.transAxes, ax_raster.transData)
        for mid_y, power, color in tranche_mids:
            ax_raster.text(1.02, mid_y, f"{power}\nµW", transform=trans,
                           ha='left', va='center', fontsize=fontsize,
                           color=color, clip_on=False)

    if show_ylabels:
        ax_raster.set_ylabel("", fontsize=fontsize)
    ax_raster.tick_params(axis='y', left=False, right=False, labelleft=False, labelright=False)
    ax_raster.spines['left'].set_visible(False)
    ax_raster.spines['right'].set_visible(False)
    ax_raster.spines['top'].set_visible(False)
    ax_raster.set_ylim(-0.5, total_trials - 0.5)
    ax_raster.set_xlabel("t(s) from stim", fontsize=fontsize)
    ax_raster.tick_params(labelsize=fontsize)
    ax_psth.tick_params(labelsize=fontsize)

    # vertical lines
    for ax in (ax_psth, ax_raster):
        ax.set_xlim(minmax)
        ax.axvline(0.0, color='black', linewidth=0.8)
        for i, off_med in enumerate(off_medians):
            if off_med is not None:
                ax.axvline(off_med, color=colors[i], linestyle='--',
                           linewidth=0.6, alpha=0.7)

    ax_psth.set_xticks([-0.3, 0, 0.3])
    ax_psth.set_xticklabels(['-0.3', '0', '0.3'])
    #add the x label
    ax_psth.set_xlabel("t(s)", fontsize=fontsize)


_OPSIN_CMAPS    = {'chrimson': plt.cm.Oranges, 'chr2': plt.cm.Blues}


def _precompute_tagging_data(ephys_data, bs, opsins,
                              max_latency_ms=100,
                              post_window_ms=(1.0, 31.0),
                              baseline_duration_s=1.0,
                              baseline_gap_ms=1.0,
                              baseline_bin_ms=10.0):
    """Per-unit TTFS and evoked-rate data for the CDF and bar panels.

    Returns
    -------
    cdf_data : dict  {unit: {'prestim': (dur, evt), opsin: {power: (dur, evt)}}}
    bars_data: dict  {unit: {opsin: {'baseline': rates, power: rates, ...}}}
    """
    import re

    def _extr_power(s):
        m = re.search(r"(\d+)$", str(s))
        return int(m.group(1)) if m else None

    max_lat_s      = max_latency_ms / 1000.0
    post_start_s   = post_window_ms[0] / 1000.0
    post_end_s     = post_window_ms[1] / 1000.0
    post_bin_s     = post_end_s - post_start_s
    baseline_bin_s = baseline_bin_ms / 1000.0
    baseline_gap_s = baseline_gap_ms / 1000.0
    n_bins_bl      = int(baseline_duration_s / baseline_bin_s)

    bs_metadata = bs.metadata

    # collect per-opsin onset times grouped by power
    opsin_onset_info = {}
    for opsin in opsins:
        mask   = bs_metadata['event'].str.contains(f'{opsin}_on', na=False)
        on_df  = bs_metadata[mask].copy()
        if on_df.empty:
            continue
        on_df['power'] = on_df['event'].apply(_extr_power)
        on_df = on_df.dropna(subset=['power'])

        power_groups, all_onsets = {}, []
        for idx in on_df.index:
            pw     = int(on_df.loc[idx, 'power'])
            ts_obj = bs[idx]
            ons    = ts_obj.t.tolist() if hasattr(ts_obj, 't') else ts_obj.index.tolist()
            power_groups.setdefault(pw, []).extend(ons)
            all_onsets.extend(ons)

        if not all_onsets:
            continue
        opsin_onset_info[opsin] = {
            'power_groups': {k: np.array(v) for k, v in power_groups.items()},
            'all_onsets':   np.array(sorted(all_onsets)),
        }

    cdf_data  = {}
    bars_data = {}

    for unit in ephys_data.keys():
        spk = np.asarray(ephys_data[unit].index)
        unit_cdf  = {}
        unit_bars = {}
        prestim_durs, prestim_evts = [], []

        for opsin, info in opsin_onset_info.items():
            power_groups = info['power_groups']
            all_onsets   = info['all_onsets']
            unit_cdf[opsin]  = {}
            unit_bars[opsin] = {}

            # baseline — all bins (for mean/CI) and per-trial means (for regression)
            bl_rates, bl_trial_means = [], []
            for onset in all_onsets:
                bl_end = onset - baseline_gap_s
                trial_bins = []
                for b in range(n_bins_bl):
                    lo = bl_end - baseline_duration_s + b * baseline_bin_s
                    hi = lo + baseline_bin_s
                    trial_bins.append(int(np.sum((spk >= lo) & (spk < hi))) / baseline_bin_s)
                bl_rates.extend(trial_bins)
                bl_trial_means.append(float(np.mean(trial_bins)))
            unit_bars[opsin]['baseline']           = np.array(bl_rates)
            unit_bars[opsin]['baseline_per_trial'] = np.array(bl_trial_means)

            for pw, onsets_pw in power_groups.items():
                ttfs_ms_list, responded_list, post_rates = [], [], []
                for onset in onsets_pw:
                    # TTFS
                    rel       = spk - onset
                    mask_post = (rel > 0) & (rel <= max_lat_s)
                    responded = bool(np.any(mask_post))
                    ttfs_ms   = float(np.min(rel[mask_post]) * 1000) if responded else float(max_latency_ms)
                    ttfs_ms_list.append(ttfs_ms)
                    responded_list.append(responded)
                    # pre-stim
                    rel_null  = spk - (onset - max_lat_s)
                    mask_null = (rel_null > 0) & (rel_null <= max_lat_s)
                    pre_resp  = bool(np.any(mask_null))
                    pre_ttfs  = float(np.min(rel_null[mask_null]) * 1000) if pre_resp else float(max_latency_ms)
                    prestim_durs.append(pre_ttfs)
                    prestim_evts.append(pre_resp)
                    # post-stim rate
                    cnt = int(np.sum((spk >= onset + post_start_s) & (spk < onset + post_end_s)))
                    post_rates.append(cnt / post_bin_s)

                unit_cdf[opsin][pw]  = (np.array(ttfs_ms_list), np.array(responded_list, dtype=bool))
                unit_bars[opsin][pw] = np.array(post_rates)

        cdf_data[unit]  = {'prestim': (np.array(prestim_durs), np.array(prestim_evts, dtype=bool)),
                            **unit_cdf}
        bars_data[unit] = unit_bars

    return cdf_data, bars_data


def plot_opto_psth_single_unit(timestamps, opsin_data, unitID,
                               waveform_data=None, unit_label=None,
                               save_folder=None, minmax=(-0.5, 0.5), waveform_t_ms=None):
    """
    Plot chrimson and chr2 PSTH + raster side-by-side for one unit.
    When waveform_data is provided a third row of waveform plots is added
    below each raster: the peak-channel waveform of every spike fired during that
    raster's pulses, coloured by power (see _precompute_opto_waveforms).

    Parameters
    ----------
    timestamps : nap.Ts
        Spike timestamps for the unit.
    opsin_data : dict
        ``{opsin_name: (power_trefs, power_off_events)}`` for each opsin.
    unitID : int or str
        Unit identifier.
    waveform_data : dict or None
        ``{opsin_name: (ndarray (n_spikes, n_samples), powers (n_spikes,)) or None}``
        Pre-selected waveforms (peak channel) per opsin.
    waveform_t_ms : ndarray
        Time axis of the waveforms, ms from the spike; required with waveform_data.
    save_folder : Path or None
    minmax : tuple
    """
    opsins = list(opsin_data.keys())
    n_opsins = len(opsins)
    if n_opsins == 0:
        return

    has_wf = waveform_data is not None

    if has_wf:
        fig = plt.figure(figsize=(8 * n_opsins, 10), layout='constrained')
        gs = gridspec.GridSpec(3, n_opsins, height_ratios=[1, 2.5, 2], figure=fig)
        axs = np.empty((3, n_opsins), dtype=object)
        for col in range(n_opsins):
            axs[0, col] = fig.add_subplot(gs[0, col])
            axs[1, col] = fig.add_subplot(gs[1, col], sharex=axs[0, col])
            axs[2, col] = fig.add_subplot(gs[2, col])
        for col in range(1, n_opsins):
            axs[0, col].sharey(axs[0, 0])
            axs[1, col].sharey(axs[1, 0])
            axs[2, col].sharey(axs[2, 0])
            axs[2, col].sharex(axs[2, 0])
    else:
        fig, axs = plt.subplots(2, n_opsins, figsize=(8 * n_opsins, 7),
                                sharex=True, sharey='row',
                                gridspec_kw={'height_ratios': [1, 2.5]})
        if n_opsins == 1:
            axs = axs[:, np.newaxis]

    for col, opsin in enumerate(opsins):
        power_trefs, power_off_events = opsin_data[opsin]
        _draw_opto_psth_on_axes(timestamps, opsin, power_trefs, power_off_events,
                                ax_psth=axs[0, col], ax_raster=axs[1, col],
                                minmax=minmax)

    if has_wf:
        for col, opsin in enumerate(opsins):
            ax = axs[2, col]
            _draw_opto_waveforms(ax, waveform_t_ms, waveform_data.get(opsin), opsin,
                                 fontsize=12)
            ax.set_title(f'{opsin}: every spike during the pulses', fontsize=12)
            ax.set_xlabel('t(ms) from spike', fontsize=15)
            if col == 0:
                ax.set_ylabel('Amplitude (µV)', fontsize=15)

    label_str = f"  [{unit_label}]" if unit_label is not None else ""
    fig.suptitle(f"Unit {unitID}{label_str}  -  opto PSTH (all powers)", fontsize=18)
    if not has_wf:
        fig.tight_layout()

    if save_folder is not None:
        fig.savefig(save_folder / f"Unit_{unitID}_opto_all_powers_psth.png", dpi=150)
        plt.close(fig)
    else:
        plt.show()


def _collect_opsin_events(bs_metadata, bs, opsin, stim_duration=None, duration_tol=0.01):
    """Return (power_trefs, power_off_events) for *opsin*, or None if no events.

    If stim_duration is given (seconds), only power levels whose median
    pulse width is within duration_tol of stim_duration are kept.
    """
    on_mask = bs_metadata['event'].str.contains(f'{opsin}_on', na=False)
    on_events_df = bs_metadata[on_mask].copy()
    if on_events_df.empty:
        return None
    on_events_df['power'] = on_events_df['event'].apply(_extract_power)
    on_events_df = on_events_df.dropna(subset=['power'])
    on_events_df = on_events_df.sort_values('power')

    off_mask = bs_metadata['event'].str.contains(f'{opsin}_off', na=False)
    off_events_df = bs_metadata[off_mask].copy()
    off_events_df['power'] = off_events_df['event'].apply(_extract_power)

    power_trefs = []
    power_off_events = []
    for _, row in on_events_df.iterrows():
        power = int(row['power'])
        tref = bs[row.name]
        power_trefs.append((power, tref))

        off_row = off_events_df[off_events_df['power'] == power]
        off_ts = bs[off_row.index[0]] if not off_row.empty else None
        power_off_events.append((power, off_ts))

    if stim_duration is not None:
        filtered_trefs, filtered_off = [], []
        for (power, tref), (_, off_ts) in zip(power_trefs, power_off_events):
            if off_ts is not None and hasattr(off_ts, 't') and len(off_ts.t) > 0:
                n = min(len(tref.t), len(off_ts.t))
                durations = off_ts.t[:n] - tref.t[:n]
                mask = np.abs(durations - stim_duration) <= duration_tol
                on_times = tref.t[:n][mask]
                off_times = off_ts.t[:n][mask]
                if len(on_times) > 0:
                    filtered_trefs.append((power, nap.Ts(t=on_times)))
                    filtered_off.append((power, nap.Ts(t=off_times)))
        power_trefs, power_off_events = filtered_trefs, filtered_off

    if not power_trefs:
        return None

    return power_trefs, power_off_events


def _pulse_intervals(power_trefs, power_off_events):
    """(on, off, power) arrays of every pulse in an opsin's raster, sorted by onset.

    Built from the same (power, Ts) pairs the raster draws, so the waveforms shown under
    a raster come from exactly the pulses in it -- including any stim_duration filter
    the caller applied -- rather than from every event with the opsin's name.
    """
    on, off, pw = [], [], []
    for (power, tref), (_, off_ts) in zip(power_trefs, power_off_events):
        if off_ts is None:
            continue
        n = min(len(tref.t), len(off_ts.t))
        on.append(tref.t[:n])
        off.append(off_ts.t[:n])
        pw.append(np.full(n, power))
    if not on:
        return np.array([]), np.array([]), np.array([])
    on, off, pw = np.concatenate(on), np.concatenate(off), np.concatenate(pw)
    order = np.argsort(on)
    return on[order], off[order], pw[order]


def _in_pulse(times, on, off):
    """Index of the pulse each time falls in (on <= t <= off), or -1. Pulses must not overlap."""
    k = np.searchsorted(on, times, side='right') - 1
    inside = (k >= 0) & (times <= off[np.clip(k, 0, None)])
    return np.where(inside, k, -1)


def _precompute_opto_waveforms(analyzer, opsin_data, every_spike=True, **job_kwargs):
    """Peak-channel waveforms of the spikes fired during each opsin's pulses.

    every_spike=True reads every such spike straight from the analyzer's recording.
    The analyzer's own waveforms extension only holds its random_spikes subsample
    (max_spikes_per_unit, 500 by default, drawn across the whole session), so the
    handful of those that happen to fall inside a pulse is a tiny, arbitrary share of
    what the light actually evoked. Extracting just the in-pulse spikes, one channel per
    unit, costs a few hundred MB at most where recomputing the extension with
    method='all' would hold every spike of the session on every sparse channel.
    Falls back to the subsample, with a warning, when the recording is not reachable.

    The window and dtype follow the waveforms extension, so the two agree; the
    channel is each unit's template extremum.

    Returns ({unit_id: {opsin: (waveforms (n, n_samples) uV, powers (n,)) or None}},
             t_ms) -- t_ms is the waveform time axis relative to the spike.
    """
    from spikeinterface.core import get_template_extremum_channel
    from spikeinterface.core.waveform_tools import extract_waveforms_to_single_buffer

    fs = analyzer.sampling_frequency
    unit_ids = list(analyzer.unit_ids)
    wf_ext = analyzer.get_extension('waveforms')
    params = wf_ext.params if wf_ext is not None else {'ms_before': 1.0, 'ms_after': 2.0}
    nbefore = int(round(params['ms_before'] * fs / 1000.0))
    nafter = int(round(params['ms_after'] * fs / 1000.0))
    t_ms = (np.arange(nbefore + nafter) - nbefore) / fs * 1000.0
    peak = get_template_extremum_channel(analyzer, outputs='index')

    spikes = analyzer.sorting.to_spike_vector()
    if every_spike and not analyzer.has_recording():
        print('  opto waveforms: analyzer has no recording attached, falling back to '
              'the random_spikes subsample')
        every_spike = False
    if not every_spike:
        pick = analyzer.get_extension('random_spikes').get_data()
        spikes = spikes[pick]
    times = spikes['sample_index'] / fs

    # which pulse of which opsin every candidate spike falls in
    pulses = {o: _pulse_intervals(*d) for o, d in opsin_data.items()}
    hit = {o: _in_pulse(times, on, off) for o, (on, off, _) in pulses.items()}
    keep = np.zeros(len(spikes), bool)
    for k in hit.values():
        keep |= k >= 0

    if every_spike:
        # one channel per unit: the buffer is (n_spikes, n_samples, 1), not 384 wide
        mask = np.zeros((len(unit_ids), analyzer.get_num_channels()), bool)
        mask[np.arange(len(unit_ids)), [peak[u] for u in unit_ids]] = True
        print(f'  opto waveforms: extracting all {int(keep.sum())} in-pulse spikes')
        buf = extract_waveforms_to_single_buffer(
            analyzer.recording, spikes[keep], unit_ids, nbefore, nafter,
            mode='shared_memory', return_in_uV=True, sparsity_mask=mask,
            copy=True, job_name='opto waveforms', **job_kwargs)[:, :, 0]
        # from here on only the in-pulse spikes matter, and buf is aligned with them
        spikes = spikes[keep]
        hit = {o: k[keep] for o, k in hit.items()}
        rows = lambda u, is_unit: buf[is_unit]
    else:
        # a unit's subsample waveforms are stored in its spike order, which is the
        # order its spikes appear in the (sorted) random_spikes pick -- and on the
        # unit's sparse channels, so the peak channel is looked up among those
        sp = analyzer.sparsity
        col = {u: (int(np.flatnonzero(sp.unit_id_to_channel_indices[u] == peak[u])[0])
                   if sp is not None else peak[u]) for u in unit_ids}
        rows = lambda u, is_unit: wf_ext.get_waveforms_one_unit(u)[:, :, col[u]]

    all_wf_data = {u: {} for u in unit_ids}
    for ui, u in enumerate(unit_ids):
        is_unit = spikes['unit_index'] == ui
        wfs_u = rows(u, is_unit) if is_unit.any() else None
        for opsin, k in hit.items():
            ku = k[is_unit]
            if wfs_u is None or not (ku >= 0).any():
                all_wf_data[u][opsin] = None
                continue
            all_wf_data[u][opsin] = (wfs_u[ku >= 0], pulses[opsin][2][ku[ku >= 0]])
    return all_wf_data, t_ms


def _draw_opto_waveforms(ax, t_ms, entry, opsin, fontsize=8, legend=True):
    """Every in-pulse waveform, coloured by pulse power, with their mean in black.

    Power runs light-to-dark in the opsin's colormap, the same order the raster's rows
    run, so a waveform's colour says which block of the raster it came from. The lines
    go in as one LineCollection, which is what keeps thousands of them drawable, and
    their alpha falls with their number so a dense bundle stays a density rather than
    a solid block.
    """
    from matplotlib.collections import LineCollection
    if entry is None:
        ax.text(0.5, 0.5, 'no spikes during pulses', transform=ax.transAxes,
                ha='center', va='center', color='grey', fontsize=fontsize)
        return
    wfs, powers = entry
    cmap = _OPSIN_CMAPS.get(opsin, plt.cm.viridis)
    levels = np.unique(powers)
    shade = {p: 0.45 + 0.55 * i / max(len(levels) - 1, 1) for i, p in enumerate(levels)}
    alpha = float(np.clip(20.0 / len(wfs), 0.02, 0.3))
    segs = np.stack([np.broadcast_to(t_ms, wfs.shape), wfs], axis=-1)
    ax.add_collection(LineCollection(segs, colors=[cmap(shade[p]) for p in powers],
                                     linewidths=0.5, alpha=alpha))
    ax.plot(t_ms, wfs.mean(axis=0), color='black', linewidth=1.5,
            label=f'mean (n={len(wfs)})')
    ax.autoscale_view()
    if legend:
        ax.legend(fontsize=fontsize, loc='upper right', frameon=False)


def plot_opto_psth_single_unit_alt(timestamps, opsin_data, unitID,
                                    waveform_data=None, unit_label=None,
                                    save_folder=None, minmax=(-0.5, 0.5),
                                    cdf_data=None, bars_data=None, waveform_t_ms=None):
    """
    Produces two separate figures per unit:

    Figure 1 — waveforms + rasters (2 rows × 2 cols, sharex within each col):
      Col 0: waveforms (row 0 = opsin 0, row 1 = opsin 1) — sharex + sharey
      Col 1: rasters   (row 0 = opsin 0, row 1 = opsin 1) — sharex

    Figure 2 — tagging panels (1 row × 2 cols):
      Col 0: TTFS CDF   Col 1: evoked-rate bars

    Panel width 0.8 in; spacing ws=hs=0.1.
    """
    from utils import plot_size_utils as psu
    from lifelines import KaplanMeierFitter

    opsins   = list(opsin_data.keys())
    n_opsins = len(opsins)
    if n_opsins == 0:
        return

    wf_h, raster_h = 0.8, 0.8

    # ── Figure 1: waveforms + rasters ────────────────────────────────────────
    fig1, axs1 = plt.subplots(2, 2, squeeze=False)

    axs1[1, 0].sharex(axs1[0, 0])
    axs1[1, 0].sharey(axs1[0, 0])
    axs1[1, 1].sharex(axs1[0, 1])

    # Col 0: waveforms
    for ri, opsin in enumerate(opsins[:2]):
        ax    = axs1[ri, 0]
        _draw_opto_waveforms(ax, waveform_t_ms,
                             waveform_data.get(opsin) if waveform_data else None,
                             opsin, fontsize=8, legend=False)
        ax.set_ylabel('µV', fontsize=8)
        ax.tick_params(labelsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axs1[0, 0].tick_params(labelbottom=False)
    axs1[1, 0].set_xlabel('t(ms)', fontsize=8)

    # Col 1: rasters
    for ri, opsin in enumerate(opsins[:2]):
        ax_dummy = fig1.add_axes([0, 0, 0, 0], label=f'dummy_{ri}')
        ax_dummy.set_visible(False)
        power_trefs, power_off_events = opsin_data[opsin]
        _draw_opto_psth_on_axes(timestamps, opsin, power_trefs, power_off_events,
                                ax_psth=ax_dummy,
                                ax_raster=axs1[ri, 1],
                                minmax=minmax,
                                fontsize=8,
                                show_legend=False,
                                show_ylabels=False,
                                show_title=False,
                                show_power_labels=True)

    axs1[0, 1].tick_params(labelbottom=False)
    axs1[0, 1].set_xlabel('')

    psu.adjust_figure_for_panel_size_hetero(fig1, panel_width=0.75,
                                             panel_height=0.75,
                                             hs=0.1, ws=0.1, r=0.45, b=0.45)

    # ── Figure 2: CDF + bars ─────────────────────────────────────────────────
    fig2, axs2 = plt.subplots(1, 2, squeeze=False)
    ax_cdf  = axs2[0, 0]
    ax_bars = axs2[0, 1]

    # CDF
    if cdf_data is not None:
        pre_dur, pre_evt = cdf_data.get('prestim', (np.array([]), np.array([])))
        if len(pre_dur) > 0:
            kmf = KaplanMeierFitter()
            kmf.fit(pre_dur, event_observed=pre_evt)
            ax_cdf.step(kmf.timeline,
                        1.0 - kmf.survival_function_.values.flatten(),
                        color='grey', linewidth=1.0, alpha=0.8, where='post')
        for opsin in opsins:
            op_cdf = cdf_data.get(opsin, {})
            cmap   = _OPSIN_CMAPS.get(opsin, plt.cm.viridis)
            pws    = sorted(op_cdf.keys())
            n_pw   = max(len(pws) - 1, 1)
            for pi, pw in enumerate(pws):
                dur, evt = op_cdf[pw]
                if len(dur) == 0:
                    continue
                kmf2 = KaplanMeierFitter()
                kmf2.fit(dur, event_observed=evt)
                ax_cdf.step(kmf2.timeline,
                            1.0 - kmf2.survival_function_.values.flatten(),
                            color=cmap(0.45 + 0.55 * pi / n_pw),
                            linewidth=1.0, alpha=0.9, where='post')
    ax_cdf.set_xlim(0, 100)
    ax_cdf.set_ylim(0, 1)
    ax_cdf.set_xlabel('t(ms) to 1st spike', fontsize=8)
    ax_cdf.set_ylabel('CDF', fontsize=8)
    ax_cdf.tick_params(labelsize=8)
    ax_cdf.spines['top'].set_visible(False)
    ax_cdf.spines['right'].set_visible(False)

    # Scatter + linear regression
    from scipy.stats import linregress

    rng = np.random.default_rng(0)
    ann_y = 0.97   # vertical position for first p-value annotation

    if bars_data is not None:
        for opsin in opsins:
            op_bars = bars_data.get(opsin, {})
            if not op_bars:
                continue
            cmap      = _OPSIN_CMAPS.get(opsin, plt.cm.viridis)
            line_color = cmap(0.8)
            bl_trial  = op_bars.get('baseline_per_trial', np.array([]))
            pw_keys   = sorted(k for k in op_bars
                               if k not in ('baseline', 'baseline_per_trial'))
            n_pw      = max(len(pw_keys) - 1, 1)

            reg_x, reg_y = [], []

            # baseline at x=0
            if len(bl_trial) > 0:
                jx = rng.normal(0, 0.3, size=len(bl_trial))
                ax_bars.scatter(jx, bl_trial, color='grey', s=18, alpha=0.6,
                                linewidths=0, zorder=3)
                reg_x.extend([0.0] * len(bl_trial))
                reg_y.extend(bl_trial.tolist())

            # per-power
            for pi, pw in enumerate(pw_keys):
                rates  = np.asarray(op_bars[pw], dtype=float)
                color  = cmap(0.45 + 0.55 * pi / n_pw)
                jx     = rng.normal(float(pw), 0.3, size=len(rates))
                ax_bars.scatter(jx, rates, color=color, s=18, alpha=0.6,
                                linewidths=0, zorder=3)
                reg_x.extend([float(pw)] * len(rates))
                reg_y.extend(rates.tolist())

            reg_x = np.array(reg_x, dtype=float)
            reg_y = np.array(reg_y, dtype=float)
            if len(reg_x) >= 3:
                slope, intercept, _, p, _ = linregress(reg_x, reg_y)
                x_fit = np.array([reg_x.min(), reg_x.max()])
                ax_bars.plot(x_fit, slope * x_fit + intercept,
                             color=line_color, linewidth=1.5, zorder=4)
                p_str = f'p={p:.3f}' if p >= 0.001 else 'p<0.001'
                ax_bars.text(0.97, ann_y, p_str, transform=ax_bars.transAxes,
                             ha='right', va='top', fontsize=8, color=line_color)
                ann_y -= 0.13

    ax_bars.set_ylabel('Rate (Hz)', fontsize=8)
    ax_bars.set_xlabel('Power (µW)', fontsize=8)
    ax_bars.yaxis.set_label_position('right')
    ax_bars.yaxis.tick_right()
    ax_bars.tick_params(labelsize=8)
    ax_bars.spines['top'].set_visible(False)
    ax_bars.spines['left'].set_visible(False)

    psu.adjust_figure_for_panel_size_hetero(fig2, panel_width=0.75, panel_height=0.75,
                                             hs=0.1, ws=0.1, r=0.45, b=0.55)

    # ── save ─────────────────────────────────────────────────────────────────
    if save_folder is not None:
        fig1.savefig(save_folder / f"Unit_{unitID}_opto_psth_alt.png", dpi=150)
        fig1.savefig(save_folder / f"Unit_{unitID}_opto_psth_alt.svg")
        plt.close(fig1)
        fig2.savefig(save_folder / f"Unit_{unitID}_opto_tagging.png", dpi=150)
        fig2.savefig(save_folder / f"Unit_{unitID}_opto_tagging.svg")
        plt.close(fig2)
    else:
        plt.show()


def plot_all_opto_psth_alt(ephys_data, bs, pynapple_folder, minmax=(-0.5, 0.5), analyzer=None,
                           every_spike=True):
    """Alternative-layout version of plot_all_opto_psth (waveforms top, raster middle, PSTH bottom).

    every_spike: see _precompute_opto_waveforms.
    """
    save_folder = pynapple_folder / "opto_psth_plots_alt"
    save_folder.mkdir(exist_ok=True)

    bs_metadata = bs.metadata
    opsin_data = {}
    for opsin in ('chrimson', 'chr2'):
        result = _collect_opsin_events(bs_metadata, bs, opsin, stim_duration=0.1)
        if result is not None:
            opsin_data[opsin] = result

    if len(opsin_data) == 0:
        print("No chrimson or chr2 events found with 100 ms stimulation duration; skipping.")
        return

    all_wf_data, wf_t_ms = None, None
    if analyzer is not None:
        print("Pre-computing opto waveforms...")
        all_wf_data, wf_t_ms = _precompute_opto_waveforms(analyzer, opsin_data,
                                                          every_spike=every_spike)

    print("Pre-computing tagging data (CDF + bars)...")
    cdf_data, bars_data = _precompute_tagging_data(ephys_data, bs, list(opsin_data.keys()))

    metadata = ephys_data.metadata
    ks_col = next((c for c in ('KSLabel', 'ks_label') if c in metadata.columns), None)

    def _unit_label(neuron):
        parts = []
        if ks_col is not None:
            parts.append(f"KS:{metadata.loc[neuron, ks_col]}")
        return '  '.join(parts) if parts else None

    Parallel(n_jobs=-1, verbose=5)(
        delayed(plot_opto_psth_single_unit_alt)(
            timestamps=ephys_data[neuron],
            opsin_data=opsin_data,
            unitID=neuron,
            waveform_data=all_wf_data.get(neuron) if all_wf_data is not None else None,
            unit_label=_unit_label(neuron),
            save_folder=save_folder,
            minmax=minmax,
            cdf_data=cdf_data.get(neuron),
            bars_data=bars_data.get(neuron),
            waveform_t_ms=wf_t_ms,
        ) for neuron in ephys_data.keys()
    )


def plot_all_opto_psth(ephys_data, bs, pynapple_folder, minmax=(-0.5, 0.5), analyzer=None,
                       every_spike=True):
    """
    For each unit, plot chrimson and chr2 PSTH + raster side-by-side in one
    figure, with raster rows ordered by ascending power. With an analyzer, a row
    under each raster shows the waveform of every spike fired during its pulses.

    Parameters
    ----------
    ephys_data : nap.TsGroup-like
        Spike data keyed by unit id.
    bs : nap.TsGroup-like
        Binary-signals object with ``.metadata`` DataFrame containing an
        ``'event'`` column (e.g. ``'chrimson_on_40'``, ``'chr2_off_20'``).
    pynapple_folder : Path
        Base folder; figures are saved under ``opto_psth_plots/``.
    minmax : tuple
        Pre/post window in seconds (default ±0.5 s).
    analyzer : SortingAnalyzer or None
        Source of the waveforms; without it the waveform row is left out.
    every_spike : bool
        True reads every in-pulse spike from the recording; False uses only the
        analyzer's random_spikes subsample (see _precompute_opto_waveforms).
    """
    save_folder = pynapple_folder / "opto_psth_plots"
    save_folder.mkdir(exist_ok=True)

    bs_metadata = bs.metadata
    neurons_idx = ephys_data.keys()

    # collect event data for each opsin once (shared across all units)
    opsin_data = {}
    for opsin in ('chrimson', 'chr2'):
        result = _collect_opsin_events(bs_metadata, bs, opsin)
        if result is not None:
            opsin_data[opsin] = result

    if len(opsin_data) == 0:
        print("No chrimson or chr2 events found in bs.metadata; skipping opto PSTH.")
        return

    all_wf_data, wf_t_ms = None, None
    if analyzer is not None:
        print("Pre-computing opto waveforms...")
        all_wf_data, wf_t_ms = _precompute_opto_waveforms(analyzer, opsin_data,
                                                          every_spike=every_spike)

    metadata = ephys_data.metadata

    ks_col = next((c for c in ('KSLabel', 'ks_label') if c in metadata.columns), None)

    def _unit_label(neuron):
        parts = []
        if ks_col is not None:
            parts.append(f"KS:{metadata.loc[neuron, ks_col]}")
        return '  '.join(parts) if parts else None

    Parallel(n_jobs=-1, verbose=5)(
        delayed(plot_opto_psth_single_unit)(
            timestamps=ephys_data[neuron],
            opsin_data=opsin_data,
            unitID=neuron,
            waveform_data=all_wf_data.get(neuron) if all_wf_data is not None else None,
            unit_label=_unit_label(neuron),
            save_folder=save_folder,
            minmax=minmax,
            waveform_t_ms=wf_t_ms,
        ) for neuron in neurons_idx
    )


def test(base_folder=None):
    if base_folder is None:
        base_folder = Path(r"C:\Users\assad\Documents\recording_files\DS13\DS13_20250822")

    folders = fs.SessionPaths(base_folder)
    pynapple_folder = folders.pynapple_folder

    ephys_file = pynapple_folder / "spikes.npz"
    ephys_data = nap.load_file(ephys_file)
    sigs_file = folders.binary_signals_file()
    sigs = nap.load_file(sigs_file)

    plot_on_events_psth(ephys_data, sigs, pynapple_folder)

def plot_all():
    base_folders = [
        Path(r"C:\Users\assad\Documents\recording_files\DS13\DS13_20250822"),
        Path(r"C:\Users\assad\Documents\recording_files\DS13\DS13_20250905"),
    ]
    for base_folder in base_folders:
        test(base_folder=base_folder)

    #get sigs_metadata rows where binary_event_name contains 'on'

