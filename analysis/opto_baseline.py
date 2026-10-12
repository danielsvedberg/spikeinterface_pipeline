"""Light-evoked change in firing vs quiet baseline firing, per good unit.

For every session, each animal and the cross-dataset compendium: one dot per good single unit
(sing_tagged_units: bombcell good + non_soma_good), x = quiet-baseline firing rate, y = % change
in firing during the light, one column per light colour (blue = ChR2 pulses, orange =
Chrimson pulses, every intensity pooled), dots coloured by cell type.

Quiet baseline (user, 2026-10-05): every moment of the session where nothing is going on --
no trial (cue onset to outcome tone), no lick or lick bout, no light pulse, no tone or
solenoid -- each event padded by QUIET_PAD_S on both sides, and saturation-blanked time out.
Rate = spikes in that time / that time.

Light response: the peak STIM_WINDOW_S-average firing rate during the light, pooled over every
pulse of a colour: the pulses' spikes binned at 1 ms from LIGHT_ARTIFACT_MS after onset to
LIGHT_ARTIFACT_MS before offset (the tagging rules: parse_opto_tagging), averaged over pulses,
and the highest STIM_WINDOW_S (10 ms, user 2026-10-05; was 100 ms) boxcar mean anywhere in the
trimmed pulse (the whole trimmed pulse when it is shorter: 8 ms of a 10-ms pulse).
% change = 100 * (light rate / baseline - 1).
"""
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap

from analysis.pres_plot_utils import CLASS_COLORS, TAGGED_CLASSES
from postprocessing.parse_opto_tagging import LIGHT_ARTIFACT_MS
from utils import file_structure as fs

QUIET_PAD_S = 0.2
STIM_WINDOW_S = 0.01
Y_BREAK_PCT = 5000.0      # the scatter's y axis is broken above this % change (plot_scatter)
OPSINS = (('chr2', 'blue light (ChR2)'), ('chrimson', 'orange light (Chrimson)'))
CATEGORIES = TAGGED_CLASSES + ('untagged',)
CAT_COLORS = CLASS_COLORS
# spans from an onset event to its offset event, excluded whole (+- the pad); every other
# event is excluded as a moment +- the pad. ITI markers bracket quiet time, and the continuous
# light traces are not events, so neither is used.
SPANS = (('start_cues', 'outcome_tones'), ('lick_bout_starts', 'lick_bout_ends'))
NOT_EVENTS = ('_trace', 'ITI_')


def _merge(iv):
    iv = iv[np.argsort(iv[:, 0])] if len(iv) else iv
    out = []
    for a, b in iv:
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return np.array(out) if out else np.zeros((0, 2))


def _pair(on, off):
    """(n, 2) spans from each onset to the first offset after it (onsets without one dropped)."""
    on, off = np.sort(np.asarray(on, float)), np.sort(np.asarray(off, float))
    i = np.searchsorted(off, on, 'left')
    ok = i < off.size
    return np.column_stack([on[ok], off[i[ok]]])


def quiet_intervals(bs, saturation_file=None, pad=QUIET_PAD_S):
    """(n, 2) seconds of the session with nothing going on (see the module docstring)."""
    ev = {e: np.asarray(bs[i].t, float) for i, e in zip(bs.metadata.index, bs.metadata['event'])
          if not any(s in str(e) for s in NOT_EVENTS)}
    busy = []
    for name, t in ev.items():                        # every moment
        busy.append(np.column_stack([t, t]))
    for name in ev:                                   # X_on .. X_off, e.g. light pulses, tones
        if '_on' in name and name.replace('_on', '_off') in ev:
            busy.append(_pair(ev[name], ev[name.replace('_on', '_off')]))
    for a, b in SPANS:                                # trials, lick bouts
        if a in ev and b in ev:
            busy.append(_pair(ev[a], ev[b]))
    if saturation_file is not None and Path(saturation_file).exists():
        z = np.load(saturation_file, allow_pickle=True)
        busy.append(np.asarray(z['periods'], float) / float(z['sampling_frequency']))
    busy = np.vstack(busy) if busy else np.zeros((0, 2))
    busy = _merge(np.column_stack([busy[:, 0] - pad, busy[:, 1] + pad]))
    start, end = float(bs.time_support.start[0]), float(bs.time_support.end[-1])
    quiet, cur = [], start
    for a, b in busy:
        if a > cur:
            quiet.append([cur, min(a, end)])
        cur = max(cur, b)
        if cur >= end:
            break
    if cur < end:
        quiet.append([cur, end])
    return np.array(quiet) if quiet else np.zeros((0, 2))


def light_pulses(bs, opsin):
    """(n, 2) on/off of every pulse of an opsin, all intensities pooled."""
    md = bs.metadata
    out = []
    for i, e in zip(md.index, md['event']):
        if str(e).startswith(f'{opsin}_on_'):
            off = md.index[md['event'] == str(e).replace('_on_', '_off_')]
            if len(off):
                out.append(_pair(bs[i].t, bs[off[0]].t))
    return np.vstack(out) if out else np.zeros((0, 2))


def peak_light_rate(spikes, pulses, window_s=STIM_WINDOW_S, artifact_ms=LIGHT_ARTIFACT_MS):
    """Peak window_s-average rate (Hz) in the pooled pulse response, the first and last
    artifact_ms of each pulse left out; NaN without pulses."""
    if not len(pulses):
        return np.nan
    art = artifact_ms / 1000.0
    dur = float(np.median(pulses[:, 1] - pulses[:, 0])) - 2 * art
    if dur <= 0:
        return np.nan
    n_bins = max(int(round(dur * 1000)), 1)
    edges = art + np.arange(n_bins + 1) / 1000.0
    s = np.asarray(spikes, float)
    lo = np.searchsorted(s, pulses[:, 0][:, None] + edges[None, :], 'left')
    counts = np.diff(lo, axis=1).sum(axis=0) / len(pulses)        # spikes per pulse per 1-ms bin
    w = min(int(round(window_s * 1000)), n_bins)
    box = np.convolve(counts, np.ones(w), 'valid') / (w / 1000.0)  # Hz
    return float(box.max())


def _category(md, u):
    c = md.loc[u, 'final_classif'] if 'final_classif' in md.columns else None
    return str(c) if isinstance(c, str) and c in CATEGORIES else 'untagged'


def session_table(analysis_folder):
    """One row per good single unit of a session: baseline and light-evoked rates."""
    f = fs.SessionPaths(analysis_folder)
    units = nap.load_file(f.pynapple_folder / 'sing_tagged_units.npz')
    bs = nap.load_file(f.binary_signals_file())
    sat = f.recording_folder / 'saturation' / 'saturation.npz' if f.has_recording else None
    quiet = quiet_intervals(bs, sat)
    quiet_s = float(np.diff(quiet).sum())
    pulses = {op: light_pulses(bs, op) for op, _ in OPSINS}
    md = units.metadata
    rows = []
    for u in units.keys():
        s = np.asarray(units[u].index, float)
        n_quiet = int((np.searchsorted(s, quiet[:, 1]) - np.searchsorted(s, quiet[:, 0])).sum())
        base = n_quiet / quiet_s if quiet_s > 0 else np.nan
        row = {'animal': Path(str(f.an_ID)).name, 'recording': Path(str(f.rec_ID)).name,
               'unit_id': u, 'cell_type': _category(md, u), 'baseline_hz': base,
               'baseline_spikes': n_quiet, 'quiet_s': round(quiet_s, 1)}
        for op, _ in OPSINS:
            light = peak_light_rate(s, pulses[op])
            row[f'{op}_n_pulses'] = len(pulses[op])
            row[f'{op}_peak_hz'] = light
            row[f'{op}_pct_change'] = (100.0 * (light / base - 1.0) if base and base > 0
                                       else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_scatter(table, save_path=None, title=''):
    """Baseline rate (x) vs % change during the light (y), blue | orange columns; linear axes
    (the user's standing preference: no log axes unless asked).

    Near-silent units (a few spikes in the whole quiet time) reach % changes in the millions,
    which on one linear axis flattens everything else. When any unit passes Y_BREAK_PCT the y
    axis is broken there (user's choice, 2026-10-05): the main panel runs -100% .. Y_BREAK_PCT,
    and a short panel above it, also linear, holds the units past the break."""
    t = table[table['baseline_hz'] > 0]
    vals = np.concatenate([t[f'{op}_pct_change'].to_numpy(float) for op, _ in OPSINS])
    vals = vals[np.isfinite(vals)]
    over_vals = vals[vals > Y_BREAK_PCT]
    broken = over_vals.size > 0
    if broken:
        fig, grid = plt.subplots(2, 2, figsize=(5.2, 3.0), sharex=True,
                                 gridspec_kw={'height_ratios': [1, 3], 'hspace': 0.08})
        tops, mains = grid[0], grid[1]
    else:
        fig, grid = plt.subplots(1, 2, figsize=(5.2, 2.4), sharex=True, squeeze=False)
        tops, mains = [None, None], grid[0]
    for top, ax, (op, label) in zip(tops, mains, OPSINS):
        for cat in CATEGORIES:
            d = t[(t['cell_type'] == cat) & np.isfinite(t[f'{op}_pct_change'])]
            kw = dict(color=CAT_COLORS[cat], alpha=0.5 if cat == 'untagged' else 0.9, linewidths=0,
                      zorder=1 if cat == 'untagged' else 2, s=9 if cat == 'untagged' else 14)
            ax.scatter(d['baseline_hz'], d[f'{op}_pct_change'], label=f'{cat} ({len(d)})', **kw)
            if top is not None:
                hi_ = d[d[f'{op}_pct_change'] > Y_BREAK_PCT]
                top.scatter(hi_['baseline_hz'], hi_[f'{op}_pct_change'], **kw)
        ax.set_ylim(-110, Y_BREAK_PCT if broken else max(float(vals.max()) * 1.05, 100.0))
        ax.axhline(0, color='k', lw=0.6)
        ax.set_xlabel('quiet-baseline FR (Hz)', fontsize=8)
        head = top if top is not None else ax
        head.set_title(label, fontsize=8, color='tab:blue' if op == 'chr2' else 'darkorange')
        for a in (ax, top):
            if a is None:
                continue
            a.tick_params(labelsize=7)
            a.spines['top'].set_visible(False)
            a.spines['right'].set_visible(False)
        if top is not None:
            top.set_ylim(Y_BREAK_PCT, float(over_vals.max()) * 1.08)
            top.spines['bottom'].set_visible(False)
            top.tick_params(axis='x', bottom=False)
            # the break marks: short diagonals on the y axis at the gap
            d = 0.015
            kwb = dict(color='k', clip_on=False, lw=0.8)
            top.plot((-d, +d), (-d * 3, +d * 3), transform=top.transAxes, **kwb)
            ax.plot((-d, +d), (1 - d, 1 + d), transform=ax.transAxes, **kwb)
            top.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            top.yaxis.get_offset_text().set_fontsize(6)
    if broken:
        for g in (tops, mains):
            g[1].sharey(g[0])
            g[1].tick_params(labelleft=False)
    else:
        mains[1].sharey(mains[0])
        mains[1].tick_params(labelleft=False)
    mains[-1].legend(fontsize=5, frameon=True, framealpha=0.9, edgecolor='0.6', loc='upper right',
                     markerscale=0.7, handletextpad=0.2, borderpad=0.3, labelspacing=0.2)
    mains[0].set_ylabel('% change in FR during light', fontsize=8)
    n_zero = int((table['baseline_hz'] <= 0).sum())
    notes = []
    if broken:
        notes.append(f'y axis broken at {Y_BREAK_PCT:g}%')
    if n_zero:
        notes.append(f'{n_zero} unit(s) silent at baseline not shown')
    fig.suptitle(title + (' -- ' + '; '.join(notes) if notes else ''), fontsize=8)
    if save_path is not None:
        stem = Path(save_path).parent / Path(save_path).stem
        stem.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(stem.with_suffix('.png'), dpi=150, bbox_inches='tight')
        fig.savefig(stem.with_suffix('.svg'), bbox_inches='tight')
        table.to_csv(stem.with_suffix('.csv'), index=False)
    return fig


if __name__ == '__main__':
    matplotlib.use('Agg')
    import sys
    ana = Path(r'C:\experiment_data\analysis_files')
    subjects = {
        'DS2': ['DS2_20260505', 'DS2_20260506', 'DS2_20260511'],
        'DS13': ['DS13_20250822', 'DS13_20250824', 'DS13_20250905'],
        'DS15': ['DS15_20260701', 'DS15_20260702', 'DS15_20260703', 'DS15_20260707'],
        'DS45': ['DS45_20260811', 'DS45_20260812', 'DS45_20260814', 'DS45_20260819',
                 'DS45_20260820', 'DS45_20260821'],
        'DS48': ['DS48_20260921', 'DS48_20260922', 'DS48_20260923', 'DS48_20260924'],
    }
    only = sys.argv[1:]
    tables = []
    for subj, sessions in subjects.items():
        if only and subj not in only:
            continue
        sub = []
        for s in sessions:
            t = session_table(ana / subj / s)
            t['animal'] = subj
            plot_scatter(t, ana / subj / s / 'ratio_tests' / 'opto_vs_baseline.png', title=s)
            plt.close('all')
            print(f'{s}: {len(t)} units, quiet {t["quiet_s"].iloc[0] if len(t) else 0:.0f} s', flush=True)
            sub.append(t)
        sub = pd.concat(sub, ignore_index=True)
        plot_scatter(sub, ana / subj / 'compendium' / 'opto_vs_baseline.png', title=f'{subj}, all sessions')
        plt.close('all')
        tables.append(sub)
    if not only:
        allt = pd.concat(tables, ignore_index=True)
        plot_scatter(allt, ana / 'compendium' / 'opto_vs_baseline.png', title='all animals, all sessions')
        print('all:', len(allt), 'units', flush=True)
