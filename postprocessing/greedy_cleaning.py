"""Greedy splitting and outlier peeling, the first part of spike_cleaning.clean_and_curate.

The redesign of 2026-10-06 (user): every unit, noise included, is split again and again
while it still holds two clusters, and its outliers are peeled off round after round until
none are left. Nothing is thrown away here: a peel becomes a cluster of its own
('<unit>/out<k>'), which can later merge into a neighbour with the same waveform
(outlier_merge_groups) or stay a unit; light artifacts become '<unit>/art' clusters and
spikes with an undefined feature '<unit>/nan' clusters, which clean_and_curate drops at the
end. The kept part of a peeled unit keeps its name.

The outlier scorer (split_outlier_spikes.find_outlier_spikes) is used as it is: its source is
part of the score cache's key.
"""
import numpy as np
import pandas as pd

from postprocessing import spike_cleaning as scl
from postprocessing import split_outlier_spikes as sos

# a unit is given a fresh random-spike sample when it holds fewer than this share of what
# the analyzer samples per unit: a split hands each part only the random spikes that fall in
# it, so a 40-spike peel of a 100,000-spike unit has none, and no template
RESAMPLE_SHARE = 0.5
# clusters smaller than this are not scored (too few spikes for any of the scorer's tests;
# their close pairs are resolved by order)
MIN_SCORE_SPIKES = 10
# the scorer's reasons for a core that cannot be trusted to judge stimulation spikes
UNTRUSTED = ('REVIEW', 'outlier share', 'too few spikes')
JUNK_KINDS = ('art', 'nan')


def split_partition(analyzer, parts, **job_kwargs):
    """Split units into parts (each a full partition of its train); every spike kept.

    parts : {unit: [indices of part 0, part 1, ...]}, non-empty parts only.
    Returns (analyzer, {unit: [new id of each part]}); the vector is re-sorted with its
    per-spike data (select_units_aligned).
    """
    parts = {u: [np.asarray(p, dtype=np.int64).tolist() for p in ps] for u, ps in parts.items()}
    scl._check_provenance(analyzer)
    # SpikeInterface 0.104.8's templates split fails when every unit is split (see
    # spike_cleaning.drop_spikes), so then one unit goes in a second call
    batches = [parts]
    if len(parts) == len(analyzer.unit_ids) and len(parts) > 1:
        last = list(parts)[-1]
        batches = [{u: v for u, v in parts.items() if u != last}, {last: parts[last]}]
    out, current = {}, analyzer
    for batch in batches:
        new, new_ids = current.split_units(batch, new_id_strategy='append', return_new_unit_ids=True,
                                           format='memory', **job_kwargs)
        current = scl.select_units_aligned(new, list(new.unit_ids))
        out.update({u: list(ids) for u, ids in zip(batch, new_ids)})
    return current, out


def resample_units(analyzer, unit_ids, seed=0, **job_kwargs):
    """Give units with too few random spikes a fresh sample: waveforms, PCs and templates.

    Their old random spikes are replaced by up to max_spikes_per_unit of their spikes, whose
    waveforms are read from the recording (only the chunks holding them), projected on the
    analyzer's PCA models, and averaged into their templates. Modifies the analyzer in place.
    Returns the units resampled.
    """
    from spikeinterface.core.waveform_tools import extract_waveforms_to_single_buffer
    rs_ext = analyzer.get_extension('random_spikes')
    max_n = int(rs_ext.params['max_spikes_per_unit'])
    rs = np.asarray(rs_ext.data['random_spikes_indices'])
    sv = analyzer.sorting.to_spike_vector()
    ids = list(analyzer.unit_ids)
    counts = analyzer.sorting.count_num_spikes_per_unit()
    have = np.bincount(sv['unit_index'][rs], minlength=len(ids))
    todo = [u for u in unit_ids if have[ids.index(u)] < RESAMPLE_SHARE * min(counts[u], max_n)]
    if not todo:
        return []
    rng = np.random.default_rng(seed)
    k_todo = np.array([ids.index(u) for u in todo])
    order = np.argsort(sv['unit_index'], kind='stable')
    bounds = np.searchsorted(sv['unit_index'][order], np.arange(len(ids) + 1))
    picked = []
    for k in k_todo:
        pos = order[bounds[k]:bounds[k + 1]]
        if pos.size > max_n:
            pos = rng.choice(pos, max_n, replace=False)
        picked.append(pos)
    picked = np.sort(np.concatenate(picked))
    keep = ~np.isin(sv['unit_index'][rs], k_todo)
    wf_ext = analyzer.get_extension('waveforms')
    spikes_new = sv[picked]
    mask = analyzer.sparsity.mask if analyzer.sparsity is not None else None
    wfs = extract_waveforms_to_single_buffer(
        analyzer.recording, spikes_new, analyzer.unit_ids, wf_ext.nbefore, wf_ext.nafter,
        mode='shared_memory', return_in_uV=analyzer.return_in_uV, sparsity_mask=mask, copy=True,
        job_name='resample small clusters', **job_kwargs)
    combined = np.concatenate([rs[keep], picked])
    srt = np.argsort(combined, kind='stable')
    new_rows = {'waveforms': {'waveforms': wfs.astype(wf_ext.data['waveforms'].dtype)}}
    if analyzer.has_extension('principal_components'):
        pc = analyzer.get_extension('principal_components')
        new_rows['principal_components'] = {
            'pca_projection': pc.project_new(spikes_new, wfs, progress_bar=False).astype(
                pc.data['pca_projection'].dtype)}
    scl._set_random_spikes(rs_ext, combined[srt])
    for name, keys in new_rows.items():
        data = analyzer.get_extension(name).data
        for key, rows in keys.items():
            old = data[key][keep]
            if rows.ndim == 3 and rows.shape[2] != old.shape[2]:
                # sparse buffers are as wide as the widest unit's channel set
                width = max(rows.shape[2], old.shape[2])
                pad = lambda a: np.pad(a, ((0, 0), (0, 0), (0, width - a.shape[2])))
                old, rows = pad(old), pad(rows)
            data[key] = np.concatenate([old, rows])[srt]
    tpl = analyzer.get_extension('templates')
    for u in todo:
        w = wf_ext.get_waveforms_one_unit(u, force_dense=True)
        k = ids.index(u)
        for op, fn in (('average', np.mean), ('std', np.std), ('median', np.median)):
            if op in tpl.data:
                tpl.data[op][k] = fn(w, axis=0)
    return todo


def _depth_width(sn, nb, fs):
    """Trough depth (baseline-subtracted minimum 0.17 ms before .. 0.33 ms after the spike
    sample) and width (samples under half of it, 0.33 ms before .. 0.67 ms after) of
    snippets (n, samples) aligned with the spike at sample nb."""
    sn = np.atleast_2d(sn)
    b0 = int(round(nb / 3))
    t0, t1 = nb - int(round(0.17e-3 * fs)), nb + int(round(0.33e-3 * fs))
    w0, w1 = nb - int(round(0.33e-3 * fs)), nb + int(round(0.67e-3 * fs))
    sn = sn - np.median(sn[:, :b0], axis=1, keepdims=True)
    tr = sn[:, t0:t1].min(axis=1)
    return tr, (sn[:, w0:w1] < tr[:, None] / 2).sum(axis=1)


def light_artifacts(analyzer, stim_intervals, edge_ms=None, depth_factor=None, width_factor=None):
    """{unit: indices into its train} of light artifacts, and a per-unit table.

    A light artifact is a spike within edge_ms of a light onset or offset that is more than
    depth_factor times as deep as the unit's template (on its peak channel) and at most
    width_factor as wide (at least 2 samples). Only the spikes near an edge are read.
    Units whose template is not trough-dominated are skipped.
    """
    edge_ms = scl.ARTIFACT_EDGE_MS if edge_ms is None else edge_ms
    depth_factor = scl.ARTIFACT_DEPTH_FACTOR if depth_factor is None else depth_factor
    width_factor = scl.ARTIFACT_WIDTH_FACTOR if width_factor is None else width_factor
    if stim_intervals is None or not len(stim_intervals):
        return {}, pd.DataFrame()
    fs = analyzer.sampling_frequency
    rec = analyzer.recording
    n_total = rec.get_num_samples()
    edges = np.sort(np.round(np.asarray(stim_intervals, float).ravel() * fs).astype(np.int64))
    tol = int(round(edge_ms * 1e-3 * fs))
    tpl = analyzer.get_extension('templates')
    nbt = tpl.nbefore
    nb, na = int(round(1e-3 * fs)), int(round(2e-3 * fs))
    trains = scl._unit_samples(analyzer)
    cand, ref, peak = {}, {}, {}
    for u, s in trains.items():
        i = np.searchsorted(edges, s)
        d = np.minimum(np.abs(s - edges[np.clip(i - 1, 0, len(edges) - 1)]),
                       np.abs(s - edges[np.clip(i, 0, len(edges) - 1)]))
        idx = np.flatnonzero((d <= tol) & (s >= nb) & (s < n_total - na))
        if not idx.size:
            continue
        T = tpl.get_unit_template(u)
        c = int(np.argmax(np.abs(T).max(axis=0)))
        tr, w = _depth_width(T[nbt - nb:nbt + na, c], nb, fs)
        if not (tr[0] < 0 and abs(tr[0]) >= 0.5 * T[:, c].max()
                and abs(tr[0]) >= scl.ARTIFACT_MIN_TEMPLATE_UV):
            continue
        cand[u], ref[u], peak[u] = idx, (float(tr[0]), float(w[0])), analyzer.channel_ids[c]
    if not cand:
        return {}, pd.DataFrame()
    # read every window of spikes near edges once: windows closer than 5 ms are joined
    items = sorted((int(trains[u][i]), u, int(i)) for u in cand for i in cand[u])
    chans = sorted(set(peak.values()), key=list(analyzer.channel_ids).index)
    col = {c: k for k, c in enumerate(chans)}
    depth, width = {}, {}
    gap = int(5e-3 * fs)
    g0 = 0
    while g0 < len(items):
        g1 = g0 + 1
        while g1 < len(items) and items[g1][0] - items[g1 - 1][0] <= gap:
            g1 += 1
        lo, hi = items[g0][0] - nb, items[g1 - 1][0] + na
        blk = rec.get_traces(start_frame=lo, end_frame=hi, channel_ids=chans, return_in_uV=True)
        for smp, u, i in items[g0:g1]:
            tr, w = _depth_width(blk[smp - lo - nb:smp - lo + na, col[peak[u]]], nb, fs)
            depth[(u, i)], width[(u, i)] = tr[0], w[0]
        g0 = g1
    out, rows = {}, []
    for u, idx in cand.items():
        d = np.array([depth[(u, i)] for i in idx])
        w = np.array([width[(u, i)] for i in idx])
        art = idx[(d < depth_factor * ref[u][0]) & (w <= max(2.0, width_factor * ref[u][1]))]
        rows.append({'unit_id': u, 'near_light_edge': int(idx.size), 'artifacts': int(art.size),
                     'template_depth_uV': ref[u][0], 'template_width': ref[u][1]})
        if art.size:
            out[u] = art.astype(np.int64)
    return out, pd.DataFrame(rows)


class GreedyState:
    """The working analyzer and what is known about each of its units."""

    def __init__(self, analyzer, result):
        splits_o, audit, scores = result
        self.analyzer = analyzer
        self.splits_o, self.audit, self.scores = dict(splits_o), audit.copy(), dict(scores)
        counts = analyzer.sorting.count_num_spikes_per_unit()
        self.name = {u: str(u) for u in analyzer.unit_ids}
        self.kind = {u: 'unit' for u in analyzer.unit_ids}
        self.depth = {u: 0 for u in analyzer.unit_ids}
        # the peel cap: a unit (or split part) never loses more than MAX_OUTLIER_FRAC of
        # what it had when it came to be
        self.base_n = {u: int(counts[u]) for u in analyzer.unit_ids}
        self.peeled = {u: 0 for u in analyzer.unit_ids}

    def replace(self, u, new_ids, names, kinds, sizes, keep_first=True):
        """Unit u became new_ids (part 0 the kept part when keep_first)."""
        for k, (c, nm, kd) in enumerate(zip(new_ids, names, kinds)):
            self.name[c], self.kind[c] = nm, kd
            self.depth[c] = self.depth[u] + (0 if keep_first and k == 0 else 1)
            if keep_first and k == 0:
                self.base_n[c], self.peeled[c] = self.base_n[u], self.peeled[u]
            else:
                self.base_n[c], self.peeled[c] = int(sizes[k]), 0
        for d in (self.scores, self.splits_o):
            d.pop(u, None)
        if u in self.audit.index:
            self.audit = self.audit.drop(index=u)

    def rescore(self, units, stim_intervals, **job_kwargs):
        counts = self.analyzer.sorting.count_num_spikes_per_unit()
        units = [u for u in units if counts[u] >= MIN_SCORE_SPIKES]
        if not units:
            return []
        s, a, sc = sos.find_outlier_spikes(self.analyzer, unit_ids=units, stim_intervals=stim_intervals,
                                           return_scores=True, **job_kwargs)
        self.splits_o.update(s)
        self.scores.update(sc)
        self.audit = pd.concat([self.audit.drop(index=[u for u in a.index if u in self.audit.index]), a])
        return units


def _peel_sets(st, u, stim_outliers):
    """(outliers, nan) indices of unit u to peel this round."""
    sc = st.scores[u]
    nan = np.flatnonzero(sc['nan'])
    out = np.asarray(st.splits_o[u][1], dtype=np.int64) if u in st.splits_o else np.array([], np.int64)
    reason = str(st.audit['reason'].get(u, '')) if 'reason' in st.audit.columns else ''
    if stim_outliers == 'drop' and sc['d2'] is not None and not reason.startswith(UNTRUSTED):
        # stimulation spikes beyond the cut of a trusted core (the scorer keeps them out of
        # its own outlier set so that they neither define the core nor sway its verdict)
        out = np.union1d(out, np.flatnonzero((sc['d2'] > sc['cut']) & sc['stim'] & ~sc['nan']))
    return np.setdiff1d(out, nan).astype(np.int64), nan.astype(np.int64)


def _peel_plot(cplots, st, members, rnd, plot_folder):
    cplots.plot_merge_report(
        st.analyzer, members, plot_folder / 'peel_plots',
        f'peel round {rnd}: {st.name[members[0]]} -> ' + ', '.join(st.name[m] for m in members[1:]),
        names=dict(st.name), file_name=f'round{rnd}_{st.name[members[0]]}')


def greedy_split_and_peel(analyzer, result, stim_intervals=None, stim_outliers='drop',
                          max_depth=None, max_rounds=None, plot_folder=None, **job_kwargs):
    """Split until no unit holds two clusters, peel outliers until none are left.

    result : score_outliers(analyzer, ...) for every unit of `analyzer`
    Returns (analyzer, state, split table, peel table, artifact table). state.name / kind map
    every unit of the returned analyzer to its name and kind ('unit', 'out', 'art', 'nan');
    state.scores holds the scores of every unit with at least MIN_SCORE_SPIKES spikes.
    plot_folder : review figures of every split (bimodal_split_plots/) and every peel with
    at least scl.OUTLIER_MERGE_MIN_SPIKES outliers (peel_plots/)
    """
    from pathlib import Path
    max_depth = scl.SPLIT_MAX_DEPTH if max_depth is None else max_depth
    max_rounds = scl.PEEL_MAX_ROUNDS if max_rounds is None else max_rounds
    st = GreedyState(analyzer, result)
    split_rows, peel_rows = [], []
    plot_folder = Path(plot_folder) if plot_folder is not None else None

    def do_splits(todo, rnd):
        level = 0
        while todo:
            capped = [u for u in todo if st.depth[u] >= max_depth]
            for u in capped:
                split_rows.append({'unit': st.name[u], 'round': rnd, 'depth': st.depth[u] + 1,
                                   'children': [], 'note': f'depth cap {max_depth}: not split'})
            todo = [u for u in todo if st.depth[u] < max_depth]
            if not todo:
                return
            level += 1
            sp = {u: [np.flatnonzero(st.scores[u]['split'] == k) for k in (0, 1)] for u in todo}
            print(f'    split level {level}: {len(todo)} units')
            if plot_folder is not None:
                from postprocessing import cleaning_plots as cplots
                try:
                    cplots.plot_bimodal_splits(st.analyzer, {u: st.scores[u]['split'] for u in todo},
                                               plot_folder / 'bimodal_split_plots', names=dict(st.name),
                                               table=None, **job_kwargs)
                except Exception as e:  # a figure never stops the cleaning
                    print(f'    split plots failed: {e!r}')
            audit = st.audit
            st.analyzer, new_ids = split_partition(st.analyzer, sp, **job_kwargs)
            children = []
            for u, pair in new_ids.items():
                nm = st.name[u]
                a = audit.loc[u] if u in audit.index else {}
                split_rows.append({'unit': nm, 'round': rnd, 'depth': st.depth[u] + 1,
                                   'children': [f'{nm}/{k + 1}' for k in range(len(pair))],
                                   'sizes': a.get('split_sizes'), 'D': a.get('split_D'),
                                   'valley': a.get('split_valley'), 'overlap': a.get('split_overlap'),
                                   'note': ''})
                st.replace(u, pair, [f'{nm}/{k + 1}' for k in range(len(pair))],
                           [st.kind[u]] * len(pair), [len(p) for p in sp[u]], keep_first=False)
                children.extend(pair)
            resample_units(st.analyzer, children, **job_kwargs)
            scored = st.rescore(children, stim_intervals, **job_kwargs)
            todo = [c for c in scored if st.scores[c].get('split') is not None]

    print('  greedy cleaning: splitting')
    do_splits([u for u, sc in st.scores.items() if sc.get('split') is not None], 0)

    print('  greedy cleaning: light artifacts')
    artifacts, art_table = light_artifacts(st.analyzer, stim_intervals)
    if len(art_table):
        art_table.insert(1, 'name', [st.name[u] for u in art_table['unit_id']])
    for rnd in range(1, max_rounds + 1):
        counts = st.analyzer.sorting.count_num_spikes_per_unit()
        parts, meta = {}, {}
        for u in list(st.analyzer.unit_ids):
            art = artifacts.get(u, np.array([], np.int64)) if rnd == 1 else np.array([], np.int64)
            out = nan = np.array([], np.int64)
            if st.kind[u] == 'unit' and u in st.scores:
                out, nan = _peel_sets(st, u, stim_outliers)
                out = np.setdiff1d(out, art)
                nan = np.setdiff1d(nan, art)
                if out.size and st.peeled[u] + out.size > sos.MAX_OUTLIER_FRAC * st.base_n[u]:
                    peel_rows.append({'unit': st.name[u], 'round': rnd, 'n_spikes': int(counts[u]),
                                      'outliers': 0, 'nan': int(nan.size), 'artifacts': int(art.size),
                                      'note': f'cap: {st.peeled[u] + out.size} would exceed '
                                              f'{sos.MAX_OUTLIER_FRAC:g} of {st.base_n[u]}'})
                    out = np.array([], np.int64)
            if not (out.size or nan.size or art.size):
                continue
            kept = np.setdiff1d(np.arange(counts[u]), np.concatenate([out, nan, art]))
            if not kept.size:
                continue
            groups = [('', 'unit', kept), (f'/out{rnd}', 'out', out), (f'/nan{rnd}', 'nan', nan),
                      ('/art', 'art', art)]
            groups = [g for g in groups if len(g[2])]
            parts[u] = [g[2] for g in groups]
            meta[u] = groups
            peel_rows.append({'unit': st.name[u], 'round': rnd, 'n_spikes': int(counts[u]),
                              'outliers': int(out.size), 'nan': int(nan.size), 'artifacts': int(art.size),
                              'note': ''})
        if not parts:
            print(f'  greedy cleaning: nothing left to peel after {rnd - 1} rounds')
            break
        n_out = sum(len(g[2]) for m in meta.values() for g in m if g[1] == 'out')
        print(f'  greedy cleaning: peel round {rnd}: {len(parts)} units, {n_out} outliers')
        st.analyzer, new_ids = split_partition(st.analyzer, parts, **job_kwargs)
        fresh, plots = [], []
        for u, ids_ in new_ids.items():
            nm = st.name[u]
            groups = meta[u]
            st.replace(u, ids_, [nm + g[0] for g in groups], [g[1] for g in groups],
                       [len(g[2]) for g in groups], keep_first=True)
            st.peeled[ids_[0]] += sum(len(g[2]) for g in groups if g[1] == 'out')
            fresh.append(ids_[0])
            fresh.extend(c for c, g in zip(ids_[1:], groups[1:]) if g[1] == 'out')
            outs = [c for c, g in zip(ids_[1:], groups[1:]) if g[1] == 'out'
                    and len(g[2]) >= scl.OUTLIER_MERGE_MIN_SPIKES]
            if outs:
                plots.append([ids_[0]] + outs)
        resample_units(st.analyzer, [c for ids_ in new_ids.values() for c in ids_[1:]], **job_kwargs)
        if plot_folder is not None and plots:
            from postprocessing import cleaning_plots as cplots
            for members in plots:
                try:
                    _peel_plot(cplots, st, members, rnd, plot_folder)
                except Exception as e:  # a figure never stops the cleaning
                    print(f'    peel plot of {st.name[members[0]]} failed: {e!r}')
        scored = st.rescore(fresh, stim_intervals, **job_kwargs)
        do_splits([c for c in scored if st.scores[c].get('split') is not None], rnd)
    else:
        left = [st.name[u] for u in st.analyzer.unit_ids if st.kind[u] == 'unit' and u in st.scores
                and _peel_sets(st, u, stim_outliers)[0].size]
        if left:
            print(f'  greedy cleaning: round cap {max_rounds} reached with {len(left)} units still '
                  f'holding outliers')
            peel_rows.extend({'unit': n, 'round': max_rounds + 1, 'note': 'round cap: outliers left'}
                             for n in left)
    return (st.analyzer, st, pd.DataFrame(split_rows), pd.DataFrame(peel_rows), art_table)


def _lineage(name):
    """The unit an outlier cluster was peeled from: its name up to '/out'."""
    return name.split('/out')[0] if '/out' in name else None


def outlier_merge_groups(analyzer, names, kinds, exclude=(), w_ms=scl.W_MS, r_ms=scl.R_MS):
    """Merge groups [target, outlier cluster, ...] and a table of every outlier cluster's fate.

    Each outlier cluster of at least OUTLIER_MERGE_MIN_SPIKES spikes goes to the one unit that
    matches it best: a unit (not another outlier cluster, not excluded, not the unit it was
    peeled from or a part of it) within DRIFT_MAX_DISTANCE_UM, with template cosine >=
    MERGE_MIN_SIMILARITY, no spikes of either within (W, R] of the other, and refractory once
    merged. The clusters assigned to one unit join it in order of similarity as long as the
    whole group stays refractory and the clusters do not violate each other.
    """
    fs = analyzer.sampling_frequency
    t_c, t_r = scl._t_c(fs, w_ms), scl._t_c(fs, r_ms)
    ids = list(analyzer.unit_ids)
    counts = analyzer.sorting.count_num_spikes_per_unit()
    loc = analyzer.get_extension('unit_locations').get_data()[:, :2]
    trains = scl._unit_samples(analyzer)
    exclude = set(exclude)
    targets = [u for u in ids if kinds.get(u) == 'unit' and u not in exclude]
    rows, assigned = [], {}
    for o in ids:
        if kinds.get(o) != 'out':
            continue
        row = {'cluster': o, 'name': names.get(o, str(o)), 'n_spikes': int(counts[o]), 'target': None,
               'target_name': '', 'similarity': np.nan, 'note': ''}
        rows.append(row)
        if counts[o] < scl.OUTLIER_MERGE_MIN_SPIKES:
            row['note'] = f'fewer than {scl.OUTLIER_MERGE_MIN_SPIKES} spikes'
            continue
        parent = _lineage(names.get(o, ''))
        lo = loc[ids.index(o)]
        cands = []
        for t in targets:
            nt = names.get(t, str(t))
            if parent is not None and (nt == parent or nt.startswith(parent + '/')):
                continue
            if np.linalg.norm(loc[ids.index(t)] - lo) > scl.DRIFT_MAX_DISTANCE_UM:
                continue
            cands.append((scl._template_cosine(analyzer, o, t), t))
        cands.sort(key=lambda x: -x[0])
        why = 'no unit nearby' if not cands else f'best similarity {cands[0][0]:.2f}'
        for s, t in cands:
            if s < scl.MERGE_MIN_SIMILARITY:
                break
            v = scl._cross_violations(trains[o], trains[t], t_c, t_r)
            if v:
                why = f'{names.get(t, t)}: {v} cross spikes within ({w_ms:g}, {r_ms:g}] ms'
                continue
            veto = scl._merged_rp_veto(analyzer, [t, o], w_ms, r_ms)
            if veto:
                why = veto
                continue
            assigned.setdefault(t, []).append((s, o))
            row.update(target=t, target_name=names.get(t, str(t)), similarity=s)
            why = ''
            break
        row['note'] = why
    by_row = {r['cluster']: r for r in rows}
    groups, notes = [], {}
    for t, lst in assigned.items():
        g = [t]
        for s, o in sorted(lst, key=lambda x: -x[0]):
            clash = next((x for x in g[1:] if scl._cross_violations(trains[o], trains[x], t_c, t_r)), None)
            veto = '' if clash is None else f'cross spikes with {names.get(clash, clash)}'
            veto = veto or scl._merged_rp_veto(analyzer, g + [o], w_ms, r_ms)
            if veto:
                by_row[o].update(target=None, target_name='', note=f'group: {veto}')
                continue
            g.append(o)
        if len(g) > 1:
            groups.append(g)
            notes[tuple(g)] = [f'{names.get(o, o)} -> {names.get(t, t)}: similarity '
                               f'{by_row[o]["similarity"]:.2f}' for o in g[1:]]
    return groups, notes, pd.DataFrame(rows)
