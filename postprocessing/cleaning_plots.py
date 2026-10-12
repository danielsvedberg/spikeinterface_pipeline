"""One figure per unit of the spikes spike_cleaning removed, against the ones it kept.

    plot_removed_spikes(analyzer, removed, folder, labels=...)

analyzer : the analyzer the cleaning started from (the raw traces are read through it)
removed  : the cleaning's removed_spikes table -- one row per removed spike, with its
           input unit, sample and reason (outlier, stim outlier, nan feature, close
           pair pre- or post-merge)

Each figure shows the unit's strongest channels (the ones the outlier scorer uses),
one row per channel: a random sample of the kept spikes, then one column per removal
reason, then the means of each group with the unit's template. Raw traces, not PCA
reconstructions: whether a removed spike was junk, a neighbour's, or the unit's own is
only visible in the traces themselves. The snippets for every unit come out of one
parallel extraction pass over the recording.
"""
from pathlib import Path

import re

import numpy as np
import pandas as pd

from utils.mpl_backend import select_backend
select_backend()
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from postprocessing import split_outlier_spikes as sos
from preprocessing.preprocess_qc import _save

REASON_COLORS = {'outlier': '#d62728', 'stim outlier': '#9467bd', 'nan feature': '#000000',
                 'close pair (pre-merge)': '#ff7f0e', 'close pair (post-merge)': '#8c564b'}
KEPT_COLOR = '0.45'


def _pick(n, k, rng):
    return np.sort(rng.choice(n, k, replace=False)) if n > k else np.arange(n)


def plot_removed_spikes(analyzer, removed, folder, labels=None, n_kept=200, n_per_reason=150,
                        unit_ids=None, seed=0, names=None, **job_kwargs):
    """Write unit_<id>_removed_spikes.png for every unit that lost spikes; returns the paths.

    labels : optional {input unit id: final label}, shown in each title.
    names  : optional {input unit id: display name}, e.g. 'parent/1' for a split half.
    """
    from spikeinterface.core.waveform_tools import extract_waveforms_to_single_buffer
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    removed = removed.copy()
    removed['unit'] = removed['unit'].astype(np.asarray(analyzer.unit_ids).dtype)
    units = [u for u in analyzer.unit_ids if u in set(removed['unit'])]
    if unit_ids is not None:
        units = [u for u in units if u in set(unit_ids)]
    if not units:
        return []
    rng = np.random.default_rng(seed)
    fs = analyzer.sampling_frequency
    wf = analyzer.get_extension('waveforms').params
    nbefore = int(round(wf['ms_before'] * fs / 1000.0))
    nafter = int(round(wf['ms_after'] * fs / 1000.0))
    t_ms = (np.arange(nbefore + nafter) - nbefore) / fs * 1000.0
    ids = list(analyzer.unit_ids)
    sv = analyzer.sorting.to_spike_vector()
    chans = sos._unit_channels(analyzer, units)
    tpl = analyzer.get_extension('templates')
    t_nb = tpl.nbefore
    by_unit = {u: g for u, g in removed.groupby('unit')}

    # the spikes to draw: (sample, unit index, group label) for every unit, one vector
    picks = []
    for u in units:
        s = sv['sample_index'][sv['unit_index'] == ids.index(u)]
        gone = by_unit[u]
        kept = np.setdiff1d(s, gone['sample'].to_numpy())
        for smp in kept[_pick(kept.size, n_kept, rng)]:
            picks.append((smp, ids.index(u), 'kept'))
        for why, g in gone.groupby('reason'):
            smp = np.sort(g['sample'].to_numpy())
            for x in smp[_pick(smp.size, n_per_reason, rng)]:
                picks.append((x, ids.index(u), why))
    picks.sort(key=lambda p: (p[0], p[1]))
    spikes = np.zeros(len(picks), dtype=sv.dtype)
    spikes['sample_index'] = [p[0] for p in picks]
    spikes['unit_index'] = [p[1] for p in picks]
    group = np.array([p[2] for p in picks])
    mask = np.zeros((len(ids), analyzer.get_num_channels()), bool)
    for u in units:
        mask[ids.index(u), chans[u]] = True
    print(f'  removed-spike plots: extracting {len(picks)} snippets for {len(units)} units')
    buf = extract_waveforms_to_single_buffer(
        analyzer.recording, spikes, ids, nbefore, nafter, mode='shared_memory',
        return_in_uV=True, sparsity_mask=mask, copy=True, job_name='removed-spike plots',
        **job_kwargs)

    jobs = []
    for u in units:
        k = ids.index(u)
        sel = spikes['unit_index'] == k
        w, grp = buf[sel][:, :, :len(chans[u])], group[sel]
        reasons = [r for r in REASON_COLORS if np.any(grp == r)]
        counts = by_unit[u]['reason'].value_counts().to_dict()
        n_total = int(np.sum(sv['unit_index'] == k))
        T = tpl.get_unit_template(u, operator='average')[t_nb - nbefore:t_nb + nafter][:, chans[u]]
        jobs.append(((names or {}).get(u, u), t_ms, w, grp, reasons, counts, n_total, T, chans[u],
                     None if labels is None else labels.get(u), folder))
    # drawing is the slow part (seconds per figure), and each figure is independent;
    # workers get the headless backend (utils.mpl_backend)
    from joblib import Parallel, delayed
    n_jobs = job_kwargs.get('n_jobs', -1)
    return Parallel(n_jobs=n_jobs)(delayed(_unit_figure)(*j) for j in jobs)


def _unit_figure(u, t_ms, w, grp, reasons, counts, n_total, T, chans, label, folder):
    cols = ['kept'] + reasons
    n_ch = len(chans)
    pk = int(np.argmax(np.ptp(T, axis=0)))
    fig, axs = plt.subplots(n_ch, len(cols) + 1, figsize=(3.1 * (len(cols) + 1), 1.9 * n_ch + 0.9),
                            sharex=True, sharey='row', squeeze=False)
    for c, name in enumerate(cols):
        rows = np.flatnonzero(grp == name)
        color = KEPT_COLOR if name == 'kept' else REASON_COLORS[name]
        alpha = float(np.clip(15.0 / max(rows.size, 1), 0.05, 0.5))
        for ci in range(n_ch):
            ax = axs[ci, c]
            if rows.size:
                segs = np.stack([np.broadcast_to(t_ms, (rows.size, t_ms.size)), w[rows, :, ci]], -1)
                ax.add_collection(LineCollection(segs, colors=color, linewidths=0.5, alpha=alpha))
            ax.plot(t_ms, T[:, ci], color='k', lw=1.2)
            ax.autoscale_view()
        n_shown = rows.size
        n_all = n_total - sum(counts.values()) if name == 'kept' else counts.get(name, 0)
        axs[0, c].set_title(f'{name}\n{n_shown} of {n_all} shown', fontsize=8,
                            color='k' if name == 'kept' else color)
    ax_m = axs[:, -1]
    for ci in range(n_ch):
        ax_m[ci].plot(t_ms, T[:, ci], color='k', lw=1.4, label='template')
        for name in cols:
            rows = np.flatnonzero(grp == name)
            if rows.size:
                ax_m[ci].plot(t_ms, w[rows, :, ci].mean(0), lw=1.2,
                              color=KEPT_COLOR if name == 'kept' else REASON_COLORS[name],
                              label=f'{name} mean')
        axs[ci, 0].set_ylabel(f'ch {chans[ci]}' + ('  (peak)' if ci == pk else '') + '\nuV',
                              fontsize=8)
    ax_m[0].set_title('means', fontsize=8)
    ax_m[0].legend(fontsize=6, loc='lower right')
    for ax in axs[-1]:
        ax.set_xlabel('t (ms) from spike', fontsize=8)
    lost = sum(counts.values())
    fig.suptitle(f'unit {u}' + (f'  [{label}]' if label else '') +
                 f'  --  {lost} of {n_total} spikes removed ({100 * lost / max(n_total, 1):.2f}%): ' +
                 ', '.join(f'{r} {counts[r]}' for r in REASON_COLORS if r in counts), fontsize=9)
    fig.tight_layout()
    return _save(fig, folder, f"unit_{str(u).replace('/', '-')}_removed_spikes")


SPLIT_COLORS = ('#2a78d6', '#d62728')


def plot_bimodal_splits(analyzer, labels_by_parent, folder, names=None, table=None, n_show=150,
                        seed=0, **job_kwargs):
    """A review figure per bimodal split: the two clusters in feature space, over time,
    by template scale, and as waveforms on the parent's strongest channels.

    analyzer         : the analyzer the split was made on (the parent's spikes)
    labels_by_parent : {parent unit id: per-spike cluster label 0/1}
    names / table    : display names ('146', '23/1') and the split table for the title stats
    """
    from joblib import Parallel, delayed
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    parents = list(labels_by_parent)
    chans = sos._unit_channels(analyzer, parents)
    pcs, match = sos._pc_features(analyzer, parents, chans, **job_kwargs)
    fs = analyzer.sampling_frequency
    sv = analyzer.sorting.to_spike_vector()
    ids = list(analyzer.unit_ids)
    wf = analyzer.get_extension('waveforms')
    rs = analyzer.get_extension('random_spikes').get_data()
    wp = wf.params
    nb = int(round(wp['ms_before'] * fs / 1000.0))
    t_ms = (np.arange(nb + int(round(wp['ms_after'] * fs / 1000.0))) - nb) / fs * 1000.0
    tpl = analyzer.get_extension('templates')
    jobs = []
    for u in parents:
        k = ids.index(u)
        m = sv['unit_index'] == k
        lab = np.asarray(labels_by_parent[u])
        t = sv['sample_index'][m] / fs
        # the parent's random-spike waveforms, labelled through their position in its train
        pos_in_unit = np.searchsorted(np.flatnonzero(m), rs[sv['unit_index'][rs] == k])
        w = wf.get_waveforms_one_unit(u, force_dense=True)[:, :, chans[u]]
        T = tpl.get_unit_template(u, operator='average')[:, chans[u]]
        pk = int(np.argmax(np.ptp(T, axis=0)))
        name = (names or {}).get(u, str(u))
        row = None
        if table is not None and len(table):
            hit = table[table['unit'] == name]
            row = hit.iloc[0].to_dict() if len(hit) else None
        jobs.append((name, lab, t, pcs[u][:, pk * 5:pk * 5 + 2], match[u], w, lab[pos_in_unit],
                     T, pk, chans[u], t_ms, row, folder, n_show, seed))
    return Parallel(n_jobs=job_kwargs.get('n_jobs', -1))(delayed(_split_figure)(*j) for j in jobs)


def _split_figure(name, lab, t, pc, match, w, wlab, T, pk, chans, t_ms, row, folder, n_show, seed):
    rng = np.random.default_rng(seed)
    r, scale = match
    fig = plt.figure(figsize=(17, 7.5))
    gs = fig.add_gridspec(2, 4)
    ax = fig.add_subplot(gs[0, 0])
    for k in (0, 1):
        ax.scatter(pc[lab == k, 0], pc[lab == k, 1], s=1, color=SPLIT_COLORS[k], alpha=0.3, lw=0)
    ax.set_xlabel('PC1, peak channel'); ax.set_ylabel('PC2, peak channel')
    ax = fig.add_subplot(gs[0, 1:3])
    for k in (0, 1):
        ax.scatter(t[lab == k], pc[lab == k, 0], s=1, color=SPLIT_COLORS[k], alpha=0.3, lw=0)
    ax.set_xlabel('time (s)'); ax.set_ylabel('PC1, peak channel'); ax.set_title('over the session', fontsize=9)
    ax = fig.add_subplot(gs[0, 3])
    bins = np.linspace(np.percentile(scale, 0.5), np.percentile(scale, 99.5), 80)
    for k in (0, 1):
        ax.hist(scale[lab == k], bins=bins, color=SPLIT_COLORS[k], alpha=0.6,
                label=f'cluster {k + 1}: {int(np.sum(lab == k))} spikes, median r {np.median(r[lab == k]):.2f}')
    ax.set_xlabel('template scale (1 = parent template size)'); ax.legend(fontsize=7)
    for ci in range(min(len(chans), 4)):
        ax = fig.add_subplot(gs[1, ci])
        for k in (0, 1):
            rows = np.flatnonzero(wlab == k)
            if rows.size:
                pick = rows if rows.size <= n_show else rng.choice(rows, n_show, replace=False)
                segs = np.stack([np.broadcast_to(t_ms, (pick.size, t_ms.size)), w[pick, :, ci]], -1)
                ax.add_collection(LineCollection(segs, colors=SPLIT_COLORS[k], linewidths=0.4, alpha=0.12))
                ax.plot(t_ms, w[rows, :, ci].mean(0), color=SPLIT_COLORS[k], lw=1.8)
        ax.plot(t_ms, T[:, ci], color='k', lw=1.0, ls='--')
        ax.autoscale_view()
        ax.set_title(f'ch {chans[ci]}' + ('  (peak)' if ci == pk else ''), fontsize=8)
        ax.set_xlabel('t (ms)')
    stats = ''
    if row:
        stats = (f"  |  D {row['D']:.1f}, valley {row['valley']:.2f}, time overlap "
                 f"{row['overlap']:.2f}  ->  {', '.join(row['children'])}")
    fig.suptitle(f'unit {name}: split into two clusters' + stats +
                 '\n(dashed: the parent template, an average of both)', fontsize=10)
    fig.tight_layout()
    return _save(fig, folder, f"unit_{str(name).replace('/', '-')}_bimodal_split")


def plot_noise_units(analyzer, noise_table, folder, n_show=150, seed=0, **job_kwargs):
    """A review figure per dropped noise unit, and one overview page of all of them.

    From the analyzer's own stored random-spike waveforms, so it costs no extraction:
    each unit's strongest channels with its spikes and template, the spike amplitudes
    over the session, and in the title the noise criteria it failed. The overview puts
    every dropped unit's peak-channel template on one page for a first pass.
    """
    from joblib import Parallel, delayed
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    units = noise_table['unit_id'].tolist()
    chans = sos._unit_channels(analyzer, units)
    wf = analyzer.get_extension('waveforms')
    tpl = analyzer.get_extension('templates')
    fs = analyzer.sampling_frequency
    p = wf.params
    nb = int(round(p['ms_before'] * fs / 1000.0))
    t_ms = (np.arange(nb + int(round(p['ms_after'] * fs / 1000.0))) - nb) / fs * 1000.0
    sv = analyzer.sorting.to_spike_vector()
    amps = analyzer.get_extension('spike_amplitudes').get_data()
    ids = list(analyzer.unit_ids)
    jobs, overview = [], []
    for row in noise_table.itertuples():
        u = row.unit_id
        m = sv['unit_index'] == ids.index(u)
        w = wf.get_waveforms_one_unit(u, force_dense=True)[:, :, chans[u]]
        T = tpl.get_unit_template(u, operator='average')[:, chans[u]]
        pk = int(np.argmax(np.ptp(T, axis=0)))
        overview.append((row.name, T[:, pk], row.n_spikes))
        jobs.append((row.name, w, T, pk, chans[u], t_ms, sv['sample_index'][m] / fs, amps[m],
                     row.n_spikes, row.firing_rate, row.failed, folder, n_show, seed))
    paths = Parallel(n_jobs=job_kwargs.get('n_jobs', -1))(delayed(_noise_figure)(*j) for j in jobs)
    paths.append(_noise_overview(overview, t_ms, folder))
    return paths


def _noise_figure(name, w, T, pk, chans, t_ms, t, amp, n, rate, failed, folder, n_show, seed):
    rng = np.random.default_rng(seed)
    fig = plt.figure(figsize=(16, 6.5))
    gs = fig.add_gridspec(2, len(chans))
    rows = np.arange(w.shape[0]) if w.shape[0] <= n_show else rng.choice(w.shape[0], n_show, replace=False)
    for ci in range(len(chans)):
        ax = fig.add_subplot(gs[0, ci])
        if rows.size:
            segs = np.stack([np.broadcast_to(t_ms, (rows.size, t_ms.size)), w[rows, :, ci]], -1)
            ax.add_collection(LineCollection(segs, colors='0.45', linewidths=0.4, alpha=0.15))
        ax.plot(t_ms, T[:, ci], color='k', lw=1.6)
        ax.autoscale_view()
        ax.set_title(f'ch {chans[ci]}' + ('  (peak)' if ci == pk else ''), fontsize=8)
        ax.set_xlabel('t (ms)')
    ax = fig.add_subplot(gs[1, :])
    ax.scatter(t, amp, s=1, color='0.3', alpha=0.4, lw=0)
    ax.set_xlabel('time (s)'); ax.set_ylabel('spike amplitude (uV)')
    fig.suptitle(f'noise unit {name}: {n} spikes, {rate:.2f} Hz -- dropped after splitting\n'
                 f'failed: {failed}', fontsize=9)
    fig.tight_layout()
    return _save(fig, folder, f"noise_unit_{str(name).replace('/', '-')}")


def _noise_overview(items, t_ms, folder, per_row=10):
    n = len(items)
    rows = int(np.ceil(n / per_row))
    fig, axs = plt.subplots(rows, per_row, figsize=(2.0 * per_row, 1.6 * rows), squeeze=False,
                            sharex=True)
    for ax in axs.ravel():
        ax.axis('off')
    for ax, (name, trace, count) in zip(axs.ravel(), items):
        ax.axis('on')
        ax.plot(t_ms, trace, color='k', lw=1.0)
        ax.set_title(f'{name} ({count})', fontsize=7)
        ax.tick_params(labelsize=5)
    fig.suptitle(f'{n} noise units dropped after splitting: peak-channel templates '
                 '(unit, spike count)', fontsize=10)
    fig.tight_layout()
    return _save(fig, folder, 'noise_units_overview')


MERGE_COLORS = ('#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e', '#8c564b')


def _ccg(ta, tb, max_ms=20.0, bin_ms=0.5, exclude_zero=False):
    """Counts of tb - ta lags within +-max_ms (ms bins); ta, tb sorted seconds."""
    edges = np.arange(-max_ms, max_ms + bin_ms / 2, bin_ms) / 1000.0
    lo = np.searchsorted(tb, ta - edges[-1], 'left')
    hi = np.searchsorted(tb, ta + edges[-1], 'right')
    lags = np.concatenate([tb[l:h] - x for x, l, h in zip(ta, lo, hi)]) if len(ta) else np.array([])
    if exclude_zero:
        lags = lags[lags != 0]
    return np.histogram(lags, edges)[0], edges * 1000.0


def plot_merge_report(analyzer, members, folder, title, notes=(), names=None, n_show=150, seed=0,
                      file_name=None):
    """One figure for a merge, drawn from the analyzer *before* it (the members still exist).

    Rows: (1) every member's random-spike waveforms and template on the 4 channels where the
    members' templates are largest; (2) amplitude over time per member, spikes per minute per
    member (whether they take turns), each member's and the pooled autocorrelogram, the
    cross-correlogram of the first two members (the refractory dip, or its absence) and the
    members' locations. `notes`: the numbers that decided the merge, printed in the title.
    """
    rng = np.random.default_rng(seed)
    names = names or {}
    fs = analyzer.sampling_frequency
    T_s = analyzer.get_total_samples() / fs
    tpl = analyzer.get_extension('templates')
    wfe = analyzer.get_extension('waveforms')
    nb = wfe.nbefore
    temps = {u: tpl.get_unit_template(u) for u in members}
    amp = np.max([np.abs(t).max(axis=0) for t in temps.values()], axis=0)
    chans = np.sort(np.argsort(amp)[::-1][:4])
    pk = int(np.argmax(amp[chans]))
    t_ms = (np.arange(next(iter(temps.values())).shape[0]) - nb) / fs * 1000.0
    trains = {u: analyzer.sorting.get_unit_spike_train(u) / fs for u in members}
    amps = analyzer.get_extension('spike_amplitudes').get_data(outputs='by_unit')[0] \
        if analyzer.has_extension('spike_amplitudes') else {}
    loc = analyzer.get_extension('unit_locations').get_data() if analyzer.has_extension('unit_locations') else None
    ids = list(analyzer.unit_ids)
    label = {u: f'{u}' + (f' ({names[u]})' if u in names else '') for u in members}

    fig = plt.figure(figsize=(20, 8.5))
    gs = fig.add_gridspec(2, 6)
    for ci, ch in enumerate(chans):
        ax = fig.add_subplot(gs[0, ci if ci < 4 else 3])
        for k, u in enumerate(members):
            col = MERGE_COLORS[k % len(MERGE_COLORS)]
            w = wfe.get_waveforms_one_unit(u, force_dense=True)
            if len(w):
                pick = _pick(len(w), n_show, rng)
                segs = np.stack([np.broadcast_to(t_ms, (pick.size, t_ms.size)), w[pick, :, ch]], -1)
                ax.add_collection(LineCollection(segs, colors=col, linewidths=0.4, alpha=0.1))
            ax.plot(t_ms, temps[u][:, ch], color=col, lw=1.8, label=label[u] if ci == 0 else None)
        ax.autoscale_view()
        ax.set_title(f'{analyzer.channel_ids[ch]}' + ('  (peak)' if ci == pk else ''), fontsize=8)
        ax.set_xlabel('t (ms)')
        if ci == 0:
            ax.set_ylabel('uV')
            ax.legend(fontsize=7)
    # the members' locations
    ax = fig.add_subplot(gs[0, 4])
    if loc is not None:
        allxy = loc[:, :2]
        ax.scatter(allxy[:, 0], allxy[:, 1], s=4, color='0.8', lw=0)
        # a unit made in the cleaning may not have a location yet (the extension is recomputed
        # after the peeling): the centre of mass of its template's peak-to-peak amplitudes
        cxy = analyzer.get_channel_locations()[:, :2]
        mxy = {}
        for u in members:
            xy = loc[ids.index(u), :2]
            if not np.all(np.isfinite(xy)):
                ptp = np.ptp(temps[u], axis=0)
                xy = (ptp[:, None] * cxy).sum(0) / max(ptp.sum(), 1e-12)
            mxy[u] = xy
        for k, u in enumerate(members):
            ax.scatter(*mxy[u], s=40, color=MERGE_COLORS[k % len(MERGE_COLORS)])
        cy = np.mean([mxy[u][1] for u in members])
        if np.isfinite(cy):
            ax.set_ylim(cy - 150, cy + 150)
    ax.set_xlabel('x (um)'); ax.set_ylabel('y (um)'); ax.set_title('locations', fontsize=8)
    # spikes per minute per member
    ax = fig.add_subplot(gs[0, 5])
    edges = np.arange(0, T_s + 60, 60)
    for k, u in enumerate(members):
        ax.plot(edges[:-1] / 60, np.histogram(trains[u], edges)[0], color=MERGE_COLORS[k % len(MERGE_COLORS)], lw=1.2)
    ax.set_xlabel('time (min)'); ax.set_ylabel('spikes / min'); ax.set_title('activity over the session', fontsize=8)
    # amplitude over time
    ax = fig.add_subplot(gs[1, 0:2])
    for k, u in enumerate(members):
        if u in amps:
            a = np.asarray(amps[u])
            pick = _pick(len(a), 5000, rng)
            ax.scatter(trains[u][pick], a[pick], s=1, color=MERGE_COLORS[k % len(MERGE_COLORS)], alpha=0.4, lw=0)
    ax.set_xlabel('time (s)'); ax.set_ylabel('amplitude (uV)'); ax.set_title('amplitude over the session', fontsize=8)
    # autocorrelograms: each member and pooled
    ax = fig.add_subplot(gs[1, 2])
    for k, u in enumerate(members):
        c, e = _ccg(trains[u], trains[u], exclude_zero=True)
        ax.step(e[:-1], c, where='post', color=MERGE_COLORS[k % len(MERGE_COLORS)], lw=1)
    ax.axvspan(-1.5, 1.5, color='0.9', zorder=0)
    ax.set_title('autocorrelograms (members)', fontsize=8); ax.set_xlabel('lag (ms)')
    ax = fig.add_subplot(gs[1, 3])
    pooled = np.sort(np.concatenate(list(trains.values())))
    c, e = _ccg(pooled, pooled, exclude_zero=True)
    ax.step(e[:-1], c, where='post', color='k', lw=1)
    ax.axvspan(-1.5, 1.5, color='0.9', zorder=0)
    ax.set_title('autocorrelogram (merged)', fontsize=8); ax.set_xlabel('lag (ms)')
    ax = fig.add_subplot(gs[1, 4])
    if len(members) >= 2:
        a_, b_ = members[0], members[1]
        c, e = _ccg(trains[a_], trains[b_])
        ax.step(e[:-1], c, where='post', color='k', lw=1)
        ax.set_title(f'cross-correlogram {a_} x {b_}', fontsize=8)
    ax.axvspan(-1.5, 1.5, color='0.9', zorder=0)
    ax.set_xlabel('lag (ms)')
    ax = fig.add_subplot(gs[1, 5])
    ax.axis('off')
    txt = [f'{label[u]}: {len(trains[u])} spikes' for u in members] + list(notes)
    ax.text(0, 1, '\n'.join(txt), va='top', fontsize=8, family='monospace')
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if file_name is None:
        file_name = title.split(':')[0]
    file_name = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(file_name).replace('/', '-').replace('+', '_'))[:150]
    return _save(fig, folder, file_name)
