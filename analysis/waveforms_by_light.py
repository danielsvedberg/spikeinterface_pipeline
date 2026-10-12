"""Every light-driven unit's spike waveforms: spontaneous vs during blue (ChR2) vs orange (Chrimson) light.

    python -m analysis.waveforms_by_light [<analysis session folder> ...]

One figure per unit that the tagging tests call light-driven, whatever its quality (user,
2026-10-08: not only bombcell-good units; light_classif of isolate_single_tagged_units.
tag_all_units, which tags the non-single units of an already tagged session first), into
<analysis session>/waveforms_by_light/unit_<id>_<class>.png, with the quality label appended
(unit_<id>_<class>_mua.png) for a unit that is not a single unit. Columns = spontaneous / blue
light / orange light (a spike counts as light when it falls between an <opsin>_on_<power> event
and its off) / means. Rows:
  1-2  above each waveform column, its event-aligned PSTH (Hz) and raster: the opto PSTH of that
       light (plot_psth._draw_opto_psth_on_axes, rows by power), and for the spontaneous column the
       first licks (every *_first_licks event, +-0.5 s); each PSTH shares its raster's x axis;
       above the means column, the unit's location on the probe (the whole
       probe, and +-PROBE_ZOOM_UM around the unit with the contacts shaded by the template's
       peak-to-peak amplitude, the waveform channels and the neighbouring units marked)
  3-6  the unit's four largest template channels, x axes shared and the rows packed together:
       N_SHOW random snippets per group and the mean of
       *all* the group's spikes in black; in the spontaneous column every spike fired in the
       PRE_LICK_S before a first lick is drawn bright green (all of them, up to N_PRE_LICK), with
       their mean dashed
  7    the whole session, spike amplitude against time (as in the sorting summary), coloured as
       above, across the three waveform columns, and the amplitude distribution beside it
Snippets are from the preprocessed recording the sort ran on (user, 2026-10-08: the PSTH, raster
and session panels).

The recording is read once per session, in chunks, for all of its tagged units together:
reading a few channels of a 384-channel binary still pages the whole file, so a pass per unit
would read 70-200 GB per unit.
"""
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap

from analysis.opto_baseline import light_pulses
from analysis.plot_psth import _collect_opsin_events, _draw_opto_psth_on_axes
from postprocessing.isolate_single_tagged_units import tag_all_units
from utils import file_structure as fs

N_SHOW = 200
N_PRE_LICK = 300
N_CHANNELS = 4
CHUNK_S = 20.0
OUT_FOLDER = 'waveforms_by_light'
GROUPS = (('spontaneous', '0.35'), ('blue light (ChR2)', 'tab:blue'),
          ('orange light (Chrimson)', 'tab:orange'))
OPSINS = (None, 'chr2', 'chrimson')        # the light of each waveform column
OPTO_MINMAX = (-0.5, 0.5)                  # as the opto PSTH plots
LICK_MINMAX = (-0.5, 0.5)
PRE_LICK_S = 0.5
LICK_BIN_S = 0.02
PRE_LICK_COLOR = '#00e000'
LICK_TYPE_COLORS = {'rewarded': '0.1', 'early': '#7b3294'}
PROBE_ZOOM_UM = 150.0                      # half-height of the zoomed probe panel
UNIT_COLOR = 'red'


def _inside(t, p):
    if not len(p):
        return np.zeros(t.size, bool)
    p = p[np.argsort(p[:, 0])]
    j = np.searchsorted(p[:, 0], t, side='right') - 1
    return (j >= 0) & (t <= p[np.clip(j, 0, None), 1])


def _first_licks(bs):
    """(times, type names) of every *_first_licks event, sorted by time."""
    t, kind = [], []
    for i, e in zip(bs.metadata.index, bs.metadata['event'].astype(str)):
        if e.endswith('_first_licks'):
            ti = np.asarray(bs[i].t, float)
            t.append(ti)
            kind += [e[:-len('_first_licks')]] * ti.size
    if not t:
        return np.zeros(0), np.zeros(0, dtype=object)
    t, kind = np.concatenate(t), np.asarray(kind, dtype=object)
    order = np.argsort(t)
    return t[order], kind[order]


def _pre_lick(t, licks, window=PRE_LICK_S):
    """True for spikes in (lick - window, lick) of some first lick."""
    if not licks.size:
        return np.zeros(t.size, bool)
    j = np.searchsorted(licks, t, side='right')     # the next lick after each spike
    nxt = licks[np.clip(j, 0, licks.size - 1)]
    return (j < licks.size) & (nxt - t < window) & (nxt - t > 0)


def _lick_psth(ax_psth, ax_raster, t, licks, kinds):
    """PSTH (Hz) and raster of spike times t around each first lick."""
    lo, hi = LICK_MINMAX
    edges = np.arange(lo, hi + LICK_BIN_S / 2, LICK_BIN_S)
    counts = np.zeros(edges.size - 1)
    for row, (lk, kind) in enumerate(zip(licks, kinds)):
        rel = t[np.searchsorted(t, lk + lo):np.searchsorted(t, lk + hi)] - lk
        counts += np.histogram(rel, edges)[0]
        ax_raster.scatter(rel, np.full(rel.size, row), s=2, color=LICK_TYPE_COLORS.get(kind, '0.4'), lw=0)
    if licks.size:
        ax_psth.plot(edges[:-1] + LICK_BIN_S / 2, counts / licks.size / LICK_BIN_S, color='k', lw=1.2)
    for ax in (ax_psth, ax_raster):
        ax.axvline(0, color='k', lw=0.8)
        ax.set_xlim(lo, hi)
    ax_raster.set_ylim(-0.5, max(licks.size, 1) - 0.5)
    ax_raster.set_xlabel('t (s) from first lick')
    ax_raster.set_ylabel('first lick #')
    ax_psth.set_ylabel('Hz')
    counts_by = {k: int(np.sum(kinds == k)) for k in dict.fromkeys(kinds)}
    ax_psth.set_title('first licks: ' + ', '.join(f'{k} {n}' for k, n in counts_by.items()), fontsize=9)


def _probe(ax_full, ax_zoom, u, v, chan_ids, chan_pos, unit_ids, unit_pos):
    """The unit on the whole probe, and +-PROBE_ZOOM_UM around it."""
    x, y = chan_pos[:, 0], chan_pos[:, 1]
    ux, uy = v['loc']
    other = np.asarray([w != u for w in unit_ids])
    ax_full.scatter(x, y, s=2, marker='s', color='0.85', lw=0)
    ax_full.scatter(unit_pos[other, 0], unit_pos[other, 1], s=3, color='0.45', lw=0)
    ax_full.axhspan(uy - PROBE_ZOOM_UM, uy + PROBE_ZOOM_UM, color=UNIT_COLOR, alpha=0.1, lw=0)
    ax_full.scatter([ux], [uy], s=70, marker='*', color=UNIT_COLOR, zorder=3)
    ax_full.set_xlim(x.min() - 30, x.max() + 30)
    ax_full.set_ylim(y.min() - 20, y.max() + 20)
    ax_full.set_xticks([])
    ax_full.set_ylabel('depth (um)', fontsize=8)
    ax_full.tick_params(labelsize=7)
    ax_full.set_title('probe', fontsize=8)

    near = np.abs(y - uy) <= PROBE_ZOOM_UM + 10
    ax_zoom.scatter(x[near], y[near], c=v['ptp'][near], cmap='Greys', vmin=0, marker='s', s=36,
                    edgecolors='0.6', linewidths=0.3)
    for ch in v['chans']:
        i = chan_ids.index(ch)
        ax_zoom.scatter([x[i]], [y[i]], marker='s', s=36, facecolors='none', edgecolors=UNIT_COLOR, lw=0.8)
        ax_zoom.annotate(str(ch), (x[i], y[i]), xytext=(5, -2), textcoords='offset points', fontsize=6,
                         color=UNIT_COLOR)
    nb_ = other & (np.abs(unit_pos[:, 1] - uy) <= PROBE_ZOOM_UM)
    ax_zoom.scatter(unit_pos[nb_, 0], unit_pos[nb_, 1], s=25, facecolors='none', edgecolors='tab:blue', lw=0.8)
    for w, (wx, wy) in zip(np.asarray(unit_ids)[nb_], unit_pos[nb_]):
        ax_zoom.annotate(str(w), (wx, wy), xytext=(3, 3), textcoords='offset points', fontsize=6, color='tab:blue')
    ax_zoom.scatter([ux], [uy], s=120, marker='*', color=UNIT_COLOR, zorder=3)
    ax_zoom.set_ylim(uy - PROBE_ZOOM_UM, uy + PROBE_ZOOM_UM)
    ax_zoom.set_xlim(x.min() - 30, x.max() + 30)
    ax_zoom.tick_params(labelsize=7)
    ax_zoom.set_title(f'unit at x {ux:.0f}, depth {uy:.0f} um\n'
                      'contacts: template amplitude; blue: other units', fontsize=7)


def session_waveforms(analysis_folder, n_show=N_SHOW, seed=0, units=None, analyzer_name='analyzer_clean',
                      md=None, out_folder=None):
    """Draw the figure for every light-driven unit of one session, of any quality, or for the
    given units (e.g. ones under review that are not tagged: their class reads 'untagged');
    returns the paths written. For a test sort: analyzer_name (a sibling of analyzer_clean),
    md (its tagging table, tag_all_units' columns) and out_folder."""
    from scripts.clean_sessions import load_analyzer
    folders = fs.SessionPaths(analysis_folder)
    md = tag_all_units(analysis_folder) if md is None else md
    tagged = (md.index[md['light_classif'].notna()].tolist() if units is None
              else [u for u in units if u in md.index])
    if not tagged:
        return []
    qcol = next((c for c in ('bombcell_label', 'KSLabel', 'ks_label') if c in md.columns), None)
    a = load_analyzer(folders, analyzer_name)
    chan_ids = list(a.channel_ids)
    chan_pos = np.asarray(a.get_channel_locations(), float)
    unit_ids = list(a.unit_ids)
    unit_pos = (np.asarray(a.get_extension('unit_locations').get_data(), float)[:, :2]
                if a.has_extension('unit_locations') else np.full((len(unit_ids), 2), np.nan))
    rec, fsamp = a.recording, a.sampling_frequency
    n_total = rec.get_num_samples()
    wf = a.get_extension('waveforms')
    nb, na = wf.nbefore, wf.nafter
    tpl = a.get_extension('templates')
    amps = (a.get_extension('spike_amplitudes').get_data(outputs='by_unit')[0]
            if a.has_extension('spike_amplitudes') else {})
    bs = nap.load_file(folders.binary_signals_file())
    blue_p, orange_p = light_pulses(bs, 'chr2'), light_pulses(bs, 'chrimson')
    licks, lick_kinds = _first_licks(bs)
    opsin_data = {o: _collect_opsin_events(bs.metadata, bs, o) for o in OPSINS if o}
    rng = np.random.default_rng(seed)

    units = {}
    for u in tagged:
        t_ = tpl.get_unit_template(u, operator='average')
        ptp = np.ptp(t_, axis=0)
        loc = unit_pos[unit_ids.index(u)]
        if not np.all(np.isfinite(loc)):
            loc = (chan_pos * ptp[:, None]).sum(axis=0) / ptp.sum()   # the template's centre of mass
        order = np.argsort(np.abs(t_).max(axis=0))[::-1][:N_CHANNELS]
        chans = [a.channel_ids[i] for i in sorted(order)]
        s_all = a.sorting.get_unit_spike_train(u)
        keep = (s_all - nb >= 0) & (s_all + na <= n_total)
        s = s_all[keep]
        t = s / fsamp
        blue, orange = _inside(t, blue_p), _inside(t, orange_p)
        masks = [~(blue | orange), blue, orange]
        pre = masks[0] & _pre_lick(t, licks)
        picks = [set(rng.choice(np.flatnonzero(m), min(n_show, int(m.sum())), replace=False).tolist())
                 for m in masks]
        pre_idx = np.flatnonzero(pre)
        picks[0] |= set(rng.choice(pre_idx, min(N_PRE_LICK, pre_idx.size), replace=False).tolist())
        units[u] = {'chans': chans, 'peak': a.channel_ids[order[0]], 'samples': s, 't_all': s_all / fsamp,
                    'amp_all': np.asarray(amps[u]) if u in amps else None, 'keep': keep,
                    'group': np.select(masks, [0, 1, 2], default=0), 'pre': pre, 'picks': picks,
                    'sum': np.zeros((3, nb + na, len(chans))), 'n': np.zeros(3, int),
                    'pre_sum': np.zeros((nb + na, len(chans))), 'pre_n': 0,
                    'shown': [[], [], []], 'shown_pre': [],
                    'cls': md.loc[u, 'light_classif'] if isinstance(md.loc[u, 'light_classif'], str) else 'untagged',
                    'quality': md.loc[u, qcol] if qcol else '', 'single': bool(md.loc[u, 'single_unit']),
                    'ptp': ptp, 'loc': loc}
    all_chans = sorted({c for v in units.values() for c in v['chans']}, key=list(a.channel_ids).index)
    col = {c: i for i, c in enumerate(all_chans)}

    # one sequential pass: each chunk is read with nb / na samples of margin so that every spike
    # whose sample falls in [start, end) has its whole snippet in the block
    step = int(CHUNK_S * fsamp)
    for start in range(0, n_total, step):
        end = min(start + step, n_total)
        lo, hi = max(start - nb, 0), min(end + na, n_total)
        block = rec.get_traces(start_frame=lo, end_frame=hi, channel_ids=all_chans, return_in_uV=True)
        for u, v in units.items():
            s = v['samples']
            i0, i1 = np.searchsorted(s, [start, end])
            if i0 == i1:
                continue
            ci = [col[c] for c in v['chans']]
            idx = np.arange(i0, i1)
            snip = block[(s[idx] - lo)[:, None] + np.arange(-nb, na)[None, :]][:, :, ci]
            g = v['group'][idx]
            p = v['pre'][idx]
            if p.any():
                v['pre_sum'] += snip[p].sum(axis=0)
                v['pre_n'] += int(p.sum())
            for k in range(3):
                m = g == k
                if m.any():
                    v['sum'][k] += snip[m].sum(axis=0)
                    v['n'][k] += int(m.sum())
                    for j in np.flatnonzero(m):
                        if idx[j] in v['picks'][k]:
                            (v['shown_pre'] if k == 0 and p[j] else v['shown'][k]).append(snip[j])

    out = Path(out_folder) if out_folder is not None else Path(folders.analysis_folder) / OUT_FOLDER
    out.mkdir(parents=True, exist_ok=True)
    tms = np.arange(-nb, na) / fsamp * 1e3
    written = []
    for u, v in units.items():
        chans = v['chans']
        nc = len(chans)
        fig = plt.figure(figsize=(20, 1.8 * nc + 10))
        # PSTH + raster rows and the waveform rows are blocks of their own, packed together, inside
        # the looser outer grid
        gs = fig.add_gridspec(3, 4, height_ratios=[2.9, 1.05 * nc, 1.8], hspace=0.22, wspace=0.2,
                              top=0.96, bottom=0.04, left=0.05, right=0.98)
        gt = gs[0, :].subgridspec(2, 4, height_ratios=[1.1, 1.8], hspace=0.05, wspace=0.2)
        gw = gs[1, :].subgridspec(nc, 4, hspace=0.06, wspace=0.2)
        ts = nap.Ts(t=v['t_all'])

        # rows 1-2: event-aligned PSTH + raster above each waveform column, x shared
        for k, opsin in enumerate(OPSINS):
            ax_p = fig.add_subplot(gt[0, k])
            ax_r = fig.add_subplot(gt[1, k], sharex=ax_p)
            if opsin is None:
                _lick_psth(ax_p, ax_r, v['t_all'], licks, lick_kinds)
            elif opsin_data.get(opsin) is not None:
                trefs, offs = opsin_data[opsin]
                _draw_opto_psth_on_axes(ts, opsin, trefs, offs, ax_psth=ax_p, ax_raster=ax_r,
                                        minmax=OPTO_MINMAX, fontsize=9, show_legend=False)
            else:
                ax_p.text(0.5, 0.5, f'no {opsin} pulses', transform=ax_p.transAxes, ha='center')
            ax_p.tick_params(labelbottom=False)
            ax_p.set_xlabel('')
        # above the means column: where the unit sits on the probe
        gi = gs[0, 3].subgridspec(1, 2, width_ratios=[0.35, 1.0], wspace=0.3)
        _probe(fig.add_subplot(gi[0]), fig.add_subplot(gi[1]), u, v, chan_ids, chan_pos, unit_ids, unit_pos)

        # rows 3..: waveforms per channel, x shared by all, y by each row
        axs = np.empty((nc, 4), dtype=object)
        for c in range(nc):
            for k in range(4):
                axs[c, k] = fig.add_subplot(gw[c, k], sharex=axs[0, 0] if c or k else None,
                                            sharey=axs[c, 0] if k else None)
                if c < nc - 1:
                    axs[c, k].tick_params(labelbottom=False)
        for k, (name, color) in enumerate(GROUPS):
            shown = np.array(v['shown'][k]) if v['shown'][k] else None
            for c, ch in enumerate(chans):
                ax = axs[c, k]
                if shown is not None:
                    ax.plot(tms, shown[:, :, c].T, color=color, lw=0.4, alpha=0.25)
                if k == 0 and v['shown_pre']:
                    ax.plot(tms, np.array(v['shown_pre'])[:, :, c].T, color=PRE_LICK_COLOR, lw=0.5, alpha=0.5)
                if v['n'][k]:
                    ax.plot(tms, v['sum'][k][:, c] / v['n'][k], color='k', lw=1.5)
                if k == 0 and v['pre_n']:
                    ax.plot(tms, v['pre_sum'][:, c] / v['pre_n'], color='k', lw=1.2, ls='--')
                if c == 0:
                    extra = (f'; green: {len(v["shown_pre"])} of {v["pre_n"]} pre-lick spikes, their mean dashed'
                             if k == 0 else '')
                    ax.set_title(f'{name}\n{v["n"][k]} spikes ({len(v["shown"][k])} shown, mean of all in black'
                                 f'{extra})', fontsize=8)
                if k == 0:
                    ax.set_ylabel(f'{ch}{" (peak)" if ch == v["peak"] else ""}\nuV', fontsize=8)
        for c in range(nc):
            ax = axs[c, 3]
            for k, (name, color) in enumerate(GROUPS):
                if v['n'][k]:
                    ax.plot(tms, v['sum'][k][:, c] / v['n'][k], color='k' if k == 0 else color, lw=1.5,
                            label=name)
            if v['pre_n']:
                ax.plot(tms, v['pre_sum'][:, c] / v['pre_n'], color=PRE_LICK_COLOR, lw=1.2, ls='--',
                        label='pre lick')
            if c == 0:
                ax.set_title('means', fontsize=9)
                ax.legend(fontsize=7)
        for ax in axs[-1]:
            ax.set_xlabel('t (ms) from spike')

        # last row: the whole session, amplitude against time, and the amplitude distribution
        ax_s = fig.add_subplot(gs[-1, 0:3])
        ax_h = fig.add_subplot(gs[-1, 3], sharey=ax_s)
        if v['amp_all'] is not None:
            amp = v['amp_all'][v['keep']]
            t = v['samples'] / fsamp
            g = np.where(v['pre'], 3, v['group'])
            for k, color in enumerate([c for _, c in GROUPS] + [PRE_LICK_COLOR]):
                m = g == k
                ax_s.scatter(t[m], amp[m], s=2 if k == 0 else 4, color=color, alpha=0.5 if k == 0 else 0.8, lw=0)
            ax_h.hist(amp, bins=80, orientation='horizontal', color='0.4')
            ax_h.set_xlabel('spikes')
        ax_s.set_xlim(0, n_total / fsamp)
        ax_s.set_xlabel('time in session (s)')
        ax_s.set_ylabel('amplitude (uV)')
        ax_s.set_title('whole session: grey spontaneous, blue / orange during light, green before a first lick',
                       fontsize=9)

        fig.suptitle(f'{Path(str(folders.rec_ID)).name} unit {u} ({v["cls"]}, {v["quality"]}): spike waveforms '
                     f'by light condition (preprocessed recording)', y=0.99)
        path = out / (f'unit_{u}_{v["cls"]}.png' if v['single'] else f'unit_{u}_{v["cls"]}_{v["quality"]}.png')
        try:
            fig.savefig(path, dpi=110, bbox_inches='tight')
        except OSError as e:
            # an earlier copy held open by another program (an image viewer) cannot be overwritten
            path = path.with_name(path.stem + '_new.png')
            fig.savefig(path, dpi=110, bbox_inches='tight')
            print(f'{path.with_name(path.name.replace("_new", ""))} could not be written ({e}); '
                  f'wrote {path.name} instead', flush=True)
        plt.close(fig)
        written.append(path)
    return written


PAIR_MAX_UM = 50.0                         # close_pairs: unit locations at most this far apart
PAIR_DUP_MS = 0.3                          # cross-unit coincidences this close: the same spike twice
PAIR_RP_MS = 1.5                           # up to this lag: a refractory violation if one neuron
PAIR_LIGHT_PAD_S = 0.1                     # spontaneous = outside every light pulse + this after it
# merge_candidates tiers: template similarity, and the spontaneous refractory deficit (Poisson
# P(X <= n | expected)) or duplicate excess (P(X >= n | expected))
MERGE_SIM_STRONG = 0.9
MERGE_SIM = 0.8
MERGE_P_STRONG = 0.01
MERGE_P = 0.05
# and a deep enough dip (observed / expected): with thousands of spikes a shallow deficit is
# significant, and any two clusters at one site lose some coincidences to the sorter's overlap
# dead time; one neuron split in two leaves few
MERGE_RATIO_STRONG = 0.5
MERGE_RATIO = 0.75
MERGE_DUP_FRACTION = 0.2                   # duplicates: this share of the smaller unit's spikes
MERGE_MIN_EXPECTED = 3.0                   # below this many expected violations nothing is testable
MERGE_TIERS = ('1 strong: refractory deficit', '2 likely: refractory deficit', '3 duplicate spikes',
               '4 similar, too sparse to test')


def _n_within(a, b, w):
    """The number of (a, b) spike pairs at most w apart (sorted times, s)."""
    return int((np.searchsorted(b, a + w, side='right') - np.searchsorted(b, a - w, side='left')).sum())


def _spontaneous_mask(bs, pad=PAIR_LIGHT_PAD_S):
    """(light intervals padded by pad, their total length in s) of every opsin's pulses."""
    p = np.vstack([light_pulses(bs, o) for o in OPSINS if o] + [np.zeros((0, 2))])
    if not len(p):
        return p, 0.0
    p = p[np.argsort(p[:, 0])] + np.array([0.0, pad])
    merged = [p[0].copy()]
    for lo, hi in p[1:]:
        if lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append(np.array([lo, hi]))
    merged = np.asarray(merged)
    return merged, float((merged[:, 1] - merged[:, 0]).sum())


def merge_candidates(pairs):
    """The strongest merge candidates of a close_pairs table, best first, with a tier and a reason.

    On spontaneous spikes only (light drive synchronises units that are not the same neuron):
      1 strong   template similarity >= MERGE_SIM_STRONG and fewer coincidences at 0.3-1.5 ms than
                 chance (P <= MERGE_P_STRONG): the refractory dip of one neuron split in two
      2 likely   similarity >= MERGE_SIM and the same deficit at P <= MERGE_P
      3 duplicate spikes   similarity >= MERGE_SIM, an excess of coincidences within 0.3 ms
                 (P <= MERGE_P_STRONG) covering >= MERGE_DUP_FRACTION of the smaller unit's spikes:
                 the same spikes in both clusters
      4 similar, too sparse to test   similarity >= MERGE_SIM_STRONG but fewer than
                 MERGE_MIN_EXPECTED violations expected by chance, so no test has power (unless
                 the few there are already exceed chance, P <= MERGE_P: merging would add them)"""
    import pandas as pd
    if not len(pairs):
        return pairs.assign(tier=pd.Series(dtype=object), reason=pd.Series(dtype=object))
    p = pairs
    sim = p['template_similarity']
    testable = p['spont_rp_expected'] >= MERGE_MIN_EXPECTED
    dup_frac = p['spont_dup'] / p[['spont_spikes_a', 'spont_spikes_b']].min(axis=1).clip(lower=1)
    ratio = p['spont_rp'] / p['spont_rp_expected'].where(p['spont_rp_expected'] > 0)
    conds = [(sim >= MERGE_SIM_STRONG) & testable & (p['p_rp_deficit'] <= MERGE_P_STRONG)
             & (ratio <= MERGE_RATIO_STRONG),
             (sim >= MERGE_SIM) & testable & (p['p_rp_deficit'] <= MERGE_P) & (ratio <= MERGE_RATIO),
             (sim >= MERGE_SIM) & (p['p_dup_excess'] <= MERGE_P_STRONG) & (dup_frac >= MERGE_DUP_FRACTION),
             (sim >= MERGE_SIM_STRONG) & ~testable & ~(p['p_rp_excess'] <= MERGE_P)]
    tier = pd.Series(np.select(conds, MERGE_TIERS, default=''), index=p.index)
    out = p.assign(tier=tier, duplicate_fraction=dup_frac.round(3), rp_ratio=ratio.round(3))[tier != '']
    out = out.assign(reason=[
        f"similarity {r.template_similarity:.2f}, {r.distance_um:.0f} um apart; spontaneous: "
        f"{r.spont_rp} coincidences at 0.3-1.5 ms vs {r.spont_rp_expected:.1f} by chance "
        f"(ratio {r.rp_ratio:.2f}, P {r.p_rp_deficit:.1g}), {r.spont_dup} within 0.3 ms ({r.duplicate_fraction:.0%} of the "
        f"smaller unit)" for r in out.itertuples()])
    out = out.sort_values(['tier', 'template_similarity'], ascending=[True, False])
    first = ['tier', 'unit_a', 'unit_b', 'reason']
    return out[first + [c for c in out.columns if c not in first]].reset_index(drop=True)


def _write_csv(table, path):
    try:
        table.to_csv(path, index=False)
    except OSError:
        path = path.with_name(path.stem + '_new.csv')
        table.to_csv(path, index=False)
        print(f'{path.name.replace("_new", "")} is locked; wrote {path.name}', flush=True)
    return path


def close_pairs(analysis_folder, max_um=PAIR_MAX_UM):
    """Pairs of the units session_waveforms draws (light-driven, any quality) whose locations on
    the probe are at most max_um apart: merge candidates to check by eye (user, 2026-10-08).

    Per pair: distance, template cosine similarity (the analyzer's template_similarity), each
    unit's class, quality, spike count, peak channel and template amplitude, and figure name when
    it was drawn. Cross-unit coincidences are counted on spontaneous spikes (outside every light
    pulse + PAIR_LIGHT_PAD_S), against chance (n_a n_b 2 w / T_spontaneous): within PAIR_DUP_MS
    (the same spike in both, `spont_dup`) and at PAIR_DUP_MS-PAIR_RP_MS (`spont_rp`, a refractory
    violation if they were one neuron), with Poisson P of a deficit / an excess; the all-spike
    count within PAIR_RP_MS is kept too (`cross_ratio`, inflated by shared light drive).
    Writes waveforms_by_light/close_pairs.csv (sorted by distance) and merge_candidates.csv
    (merge_candidates) and returns the pairs. Reads the sorting analyzer, not the recording."""
    import pandas as pd
    import spikeinterface.full as si
    from scipy.stats import poisson
    folders = fs.SessionPaths(analysis_folder)
    md = tag_all_units(analysis_folder)
    ids = md.index[md['light_classif'].notna()].tolist()
    qcol = next((c for c in ('bombcell_label', 'KSLabel', 'ks_label') if c in md.columns), None)
    a = si.load_sorting_analyzer(folders.sorting_path('analyzer_clean'))
    unit_ids = list(a.unit_ids)
    pos = np.asarray(a.get_extension('unit_locations').get_data(), float)[:, :2]
    sim = a.get_extension('template_similarity').get_data()
    tpl = a.get_extension('templates')
    fsamp, dur = a.sampling_frequency, a.get_total_duration()
    light, light_s = _spontaneous_mask(nap.load_file(folders.binary_signals_file()))
    spont_s = dur - light_s
    out = Path(folders.analysis_folder) / OUT_FOLDER
    drawn = {int(p.stem.split('_')[1]): p.name for p in out.glob('unit_*.png')} if out.exists() else {}
    info = {}
    for u in ids:
        t_ = tpl.get_unit_template(u, operator='average')
        amp = np.abs(t_).max(axis=0)
        t = np.sort(a.sorting.get_unit_spike_train(u)) / fsamp
        info[u] = {'i': unit_ids.index(u), 'peak': a.channel_ids[int(np.argmax(amp))],
                   'amp': float(np.ptp(t_, axis=0).max()), 't': t, 's': t[~_inside(t, light)]}
    dup_w, rp_w = PAIR_DUP_MS / 1e3, PAIR_RP_MS / 1e3
    rows = []
    for j, ua in enumerate(ids):
        for ub in ids[j + 1:]:
            A, B = info[ua], info[ub]
            d = pos[A['i']] - pos[B['i']]
            dist = float(np.hypot(*d))
            if not dist <= max_um:
                continue
            n_all = _n_within(A['t'], B['t'], rp_w)
            exp_all = A['t'].size * B['t'].size * 2 * rp_w / dur
            n_dup = _n_within(A['s'], B['s'], dup_w)
            n_rp = _n_within(A['s'], B['s'], rp_w) - n_dup
            rate = A['s'].size * B['s'].size / spont_s if spont_s > 0 else np.nan
            exp_dup, exp_rp = rate * 2 * dup_w, rate * 2 * (rp_w - dup_w)
            rows.append({'unit_a': ua, 'unit_b': ub, 'distance_um': round(dist, 1),
                         'dx_um': round(float(abs(d[0])), 1), 'dy_um': round(float(abs(d[1])), 1),
                         'template_similarity': round(float(sim[A['i'], B['i']]), 3),
                         'class_a': md.loc[ua, 'light_classif'], 'class_b': md.loc[ub, 'light_classif'],
                         'quality_a': md.loc[ua, qcol] if qcol else '', 'quality_b': md.loc[ub, qcol] if qcol else '',
                         'n_spikes_a': A['t'].size, 'n_spikes_b': B['t'].size,
                         'spont_spikes_a': A['s'].size, 'spont_spikes_b': B['s'].size,
                         'peak_channel_a': A['peak'], 'peak_channel_b': B['peak'],
                         'amplitude_uv_a': round(A['amp'], 1), 'amplitude_uv_b': round(B['amp'], 1),
                         'spont_rp': n_rp, 'spont_rp_expected': round(exp_rp, 2),
                         'p_rp_deficit': float(poisson.cdf(n_rp, exp_rp)) if exp_rp > 0 else np.nan,
                         'p_rp_excess': float(poisson.sf(n_rp - 1, exp_rp)) if exp_rp > 0 else np.nan,
                         'spont_dup': n_dup, 'spont_dup_expected': round(exp_dup, 2),
                         'p_dup_excess': float(poisson.sf(n_dup - 1, exp_dup)) if exp_dup > 0 else np.nan,
                         f'cross_pairs_{PAIR_RP_MS:g}ms': n_all, 'cross_pairs_expected': round(exp_all, 1),
                         'cross_ratio': round(n_all / exp_all, 3) if exp_all > 0 else np.nan,
                         'figure_a': drawn.get(ua, ''), 'figure_b': drawn.get(ub, '')})
    table = pd.DataFrame(rows)
    if len(table):
        table = table.sort_values('distance_um').reset_index(drop=True)
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(table, out / 'close_pairs.csv')
    _write_csv(merge_candidates(table), out / 'merge_candidates.csv')
    return table


if __name__ == '__main__':
    matplotlib.use('Agg')
    ana = Path(r'C:\experiment_data\analysis_files')
    # default: the sessions of the subjects curated and tagged with the current rules
    subjects = ('DS2', 'DS13', 'DS15', 'DS45', 'DS48')
    folders = [Path(p) for p in sys.argv[1:]] or sorted(
        p for s in subjects for p in (ana / s).glob(f'{s}_20*')
        if (p / 'pynapple' / 'sing_tagged_units.npz').exists())
    for f in folders:
        try:
            paths = session_waveforms(f)
            print(f'=== {f.name}: {len(paths)} units ok', flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f'=== {f.name} FAILED: {type(e).__name__}: {e}', flush=True)
