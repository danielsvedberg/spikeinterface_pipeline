"""Spike-level cleaning between sorting and curation: outliers, refractory pairs, merges.

Replaces si.remove_duplicated_spikes. That step resolves every pair of spikes closer
than a censor period by *order* -- it keeps the first and drops the second, whatever
either looks like. When one spike of the pair is contamination from another neuron or
a second detection of the same spike, its waveform, size and location say so, and
which of the two to keep can be decided on that instead.

clean_and_curate(analyzer, stim_intervals) runs, in order:

    1. outlier spikes (postprocessing.split_outlier_spikes): per-unit robust
       Mahalanobis scoring of PCA features, template scaling and location; a unit's
       outliers are dropped when its audit accepts the split. Spikes with an undefined
       feature (NaN location) are dropped from every unit.
    2. refractory pairs, pre-merge: of every pair of the remaining spikes closer than
       W, drop the one that fits the unit worse -- the lower correlation of its raw
       snippet with the unit's template over the unit's strongest channels, i.e. its
       waveform and its spatial footprint (see _badness). method='order' instead drops
       the later spike, as remove_duplicated_spikes does -- the control arm.
    3. merges, in soft mode, with the refractory steps told what W is; each merged
       unit's random-spike sample is then thinned to be uniform over its spikes.
    4. refractory pairs, post-merge: merging two units puts spikes that were never in
       one train side by side, so pairs closer than W reappear. The merged units are
       re-scored against their merged template and resolved the same way. Without this
       step the merged units carry pairs the metric below is told cannot exist.
    5. quality metrics for every unit, with the refractory metrics censored at W, then
       bombcell, then the non-noise units are the clean analyzer.

Steps 1, 2 and 4 drop spikes through one soft split_units + remove_units each: the
per-spike extensions are sliced, and templates, correlograms and the metric
extensions are recomputed for the units that changed -- about a minute, against an
hour for rebuilding the analyzer. Units that lose spikes get new (appended) ids; the
audit tables map every final unit back to its Kilosort units.

Why the refractory metrics must be censored at W: once every pair closer than W is
gone, the Llobet estimate that bombcell reads (rp_contamination) still counts its
full window and would read the missing pairs as missing contamination, biased low by
about (R - W) / R. With censored_period_ms = W it only counts (W, R] and discounts the
dead time -- honest, if and only if no pair closer than W survives, which is asserted
before the metrics are computed.
"""
import copy
import json
import pickle
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import spikeinterface.full as si
import spikeinterface.curation as sic

from postprocessing import split_outlier_spikes as sos

# pairs of spikes at most this far apart are resolved to one spike
W_MS = 1.0
# refractory period the contamination metrics count to, above W. 1.5 ms leaves the
# 1.5-2 ms intervals of bursting units alone (DS45 has units with real short-ISI modes
# there); 2.0 ms doubles the statistical power at the cost of calling those MUA
R_MS = 1.5
# automatic merges, in order; each is one compute_merge_unit_groups preset, except
# REDUNDANT: the sorter's duplicate units (see auto_merge)
REDUNDANT = 'redundant'
DRIFT = 'drift'
MERGE_PRESETS = ('feature_neighbors', 'similarity_correlograms', DRIFT, REDUNDANT)
# DRIFT: one neuron the sorter (or a split) left as two units because drift changed its
# waveform, so their templates miss MERGE_MIN_SIMILARITY. Neighbours within
# DRIFT_MAX_DISTANCE_UM whose trains are mutually refractory (sos.mutually_refractory) and
# whose templates still share DRIFT_MIN_SIMILARITY: the pieces of DS2_20260505 KS 605 sit
# 6 um apart at cosine 0.85-0.89. The similarity floor keeps out pairs that are
# refractory to each other but look nothing alike (DS48_20260922 905/913, cosine -0.39).
DRIFT_MAX_DISTANCE_UM = 40.0
DRIFT_MIN_SIMILARITY = 0.5
# spikeinterface.curation.find_redundant_units: a pair is redundant when more than
# duplicate_threshold of one unit's spikes fall within delta_time (ms) of the other's
# (Kilosort4 emits such pairs, some offset by a fixed few samples -- a time-shifted copy of
# one template -- in every DS15/DS45/DS48 session)
REDUNDANT_PARAMS = {'delta_time': 0.4, 'agreement_threshold': 0.2, 'duplicate_threshold': 0.8}
# a merge must also join units whose templates all have at least this cosine similarity
# and that put no spikes within (W, R] of each other (see _strict_merge_veto)
MERGE_MIN_SIMILARITY = 0.9
# the refractory criterion: a unit is mua when its contamination is over
# RP_MAX_CONTAMINATION with RP_CONFIDENCE -- the one-sided lower confidence bound of the
# Llobet estimate, from the exact Poisson lower bound on its violation count, is over the
# limit -- on more than RP_MAX_FREE_VIOLATIONS violations. The point estimate alone reads
# a slow unit's one or two short intervals as 25-100% contamination (DS45 unit 413: 1
# violation in 450 spikes, 1.0); the bound alone still demotes sparse units on a single
# interval (DS48_20260922 unit 487), and units with 1-3 violations and bounds over 5-10%
# looked well isolated on review (DS48, 2026-09-30), as did those with bounds of 5-10%
RP_MAX_CONTAMINATION, RP_CONFIDENCE, RP_MAX_FREE_VIOLATIONS = 0.10, 0.95, 3
# a split half that is itself two clusters is split again, to this many levels
SPLIT_MAX_DEPTH = 2


# bombcell noise criteria left out. peak_to_trough_duration is measured from the trough to the
# "peak after" it, and when a waveform has no distinct after-peak (a slow repolarisation, as in
# many striatal units) SpikeInterface falls back to the highest point anywhere in the tail --
# a few-uV wobble ~1.4 ms out -- so well-isolated units fail its 1.15 ms limit (DS2_20260505
# KS 542: after-peak 2.2 uV at 1.40 ms; KS 548: 7.9 uV at 1.43 ms). It was the only failed
# criterion of ~950 dropped "noise" units over 17 sessions (user decision, 2026-10-02)
NOISE_EXCLUDED_METRICS = ('peak_to_trough_duration',)
# SNR minimum lowered from bombcell's 5: the noise level SNR divides by is quantised (3.03 uV/bit,
# a 3-5 bit wide background, so the MAD takes values 4.5 / 9.0 / 13.5 uV), and units failing only
# SNR in [4, 5) looked well isolated on review (DS2, user decision, 2026-10-02)
MUA_SNR_MIN = 4


def default_thresholds():
    """bombcell's default thresholds without the excluded noise criteria, with the SNR minimum lowered."""
    th = copy.deepcopy(sic.bombcell_get_default_thresholds())
    for m in NOISE_EXCLUDED_METRICS:
        th['noise'].pop(m, None)
    th['mua']['snr']['greater'] = MUA_SNR_MIN
    return th


def rp_metric_params(w_ms=W_MS, r_ms=R_MS):
    """Refractory metric params that match a train with no pair closer than w_ms.

    rp_violation is exact: its counter takes lags up to t_r inclusive, so with every lag
    <= t_c gone it counts t_c+1..t_r, which is the t_r - t_c the formula divides by.
    isi_violation counts ISIs strictly below its threshold and so covers one lag fewer
    (14 of 15 at 30 kHz, W=1, R=1.5): its ratio reads ~7% low. Bombcell reads only
    rp_contamination.
    """
    return {'rp_violation': {'refractory_period_ms': r_ms, 'censored_period_ms': w_ms},
            'isi_violation': {'isi_threshold_ms': r_ms, 'min_isi_ms': w_ms}}


@contextmanager
def _metric_defaults_restored():
    """Undo SpikeInterface's in-place update of its class-level metric defaults.

    ComputeQualityMetrics._set_params calls .update() on the metric classes' own
    metric_params dicts, so one compute with custom params silently becomes the
    default for every later compute in the process -- including the next session's
    raw analyzer when run_pipeline loops over sessions.
    """
    from spikeinterface.metrics.quality import ComputeQualityMetrics
    # SpikeInterface is handed throwaway copies to mutate; the originals -- which other
    # analyzers' stored params may alias -- are never touched and are put back after
    saved = [(m, m.metric_params) for m in ComputeQualityMetrics.metric_list]
    for m, params in saved:
        m.metric_params = copy.deepcopy(params)
    try:
        yield
    finally:
        for m, params in saved:
            m.metric_params = params


def merge_steps_params(w_ms=W_MS, r_ms=R_MS):
    """The auto-merge refractory steps, told the same W and R as the metrics."""
    return {'remove_contaminated': {'refractory_period_ms': r_ms, 'censored_period_ms': w_ms},
            'quality_score': {'refractory_period_ms': r_ms, 'censored_period_ms': w_ms},
            'cross_contamination': {'refractory_period_ms': r_ms, 'censored_period_ms': w_ms}}


def _t_c(fs, w_ms):
    """W in samples. Pairs with a gap <= this many samples are resolved."""
    return int(round(w_ms * fs / 1000.0))


def resolve_pairs(samples, badness, t_c):
    """Indices to drop so that no two kept spikes are <= t_c samples apart.

    samples : sorted spike frames of one unit; badness : per spike, larger is worse,
    or None to resolve by order (drop the later spike, as remove_duplicated_spikes
    'keep_first_iterative' does).

    One pass in time order is enough. The last kept spike is compared with each next
    spike; if they are too close, the worse of the two goes. If the earlier one goes,
    the one before it is already more than t_c before it, and so more than t_c before
    the new spike -- so no earlier pair is ever reopened. The loop only visits spikes
    inside runs of close spikes.
    """
    samples = np.asarray(samples)
    if samples.size < 2:
        return np.array([], dtype=np.int64)
    close = np.flatnonzero(np.diff(samples) <= t_c)
    if close.size == 0:
        return np.array([], dtype=np.int64)
    drop = []
    # runs of consecutive close gaps: spikes close[k] .. close[k]+1 chained together
    starts = np.r_[0, np.flatnonzero(np.diff(close) > 1) + 1]
    ends = np.r_[starts[1:], close.size]
    for a, b in zip(starts, ends):
        run = np.arange(close[a], close[b - 1] + 2)
        last = run[0]
        for i in run[1:]:
            if samples[i] - samples[last] > t_c:
                last = i
                continue
            if badness is None or badness[i] >= badness[last]:
                drop.append(i)
            else:
                drop.append(last)
                last = i
    return np.asarray(drop, dtype=np.int64)


def _badness(score):
    """Per-spike badness for resolving a close pair: 1 - template match.

    The template match is each spike's raw snippet correlated with its unit's template
    across the unit's strongest channels together, so it carries both the waveform and
    its spatial footprint, and a second detection of the same spike, offset by a
    fraction of a millisecond, matches worse than the aligned one. Not the outlier
    distance: that includes amplitude_scalings, which SpikeInterface fits jointly for
    spikes of different units that collide -- so two sorter units detecting the same
    spikes each get about half the amplitude, and once they are merged the distance
    ranks the two copies of a pair on noise.
    """
    return 1.0 - np.asarray(score['match'], dtype=float)


def _unit_samples(analyzer):
    """{unit_id: sorted sample indices} from the spike vector, one segment."""
    spikes = analyzer.sorting.to_spike_vector()
    assert np.all(spikes['segment_index'] == 0), 'spike cleaning assumes one segment'
    ids = list(analyzer.unit_ids)
    out = {}
    order = np.argsort(spikes['unit_index'], kind='stable')
    bounds = np.searchsorted(spikes['unit_index'][order], np.arange(len(ids) + 1))
    for k, u in enumerate(ids):
        out[u] = spikes['sample_index'][order[bounds[k]:bounds[k + 1]]]
    return out


def _check_provenance(analyzer):
    """A binary-folder analyzer's split rebuilds its sorting from provenance; they must agree.

    The per-spike extensions are copied by position, so a provenance sorting that has
    drifted from the one the analyzer was built on (a re-sort, a phy edit) would
    misalign every per-spike array without an error.
    """
    if analyzer.format == 'memory':
        return
    prov = analyzer.get_sorting_provenance()
    if prov is None:
        return
    a, b = analyzer.sorting.to_spike_vector(), prov.to_spike_vector()
    if a.size != b.size or not (np.array_equal(a['sample_index'], b['sample_index'])
                                and np.array_equal(a['unit_index'], b['unit_index'])):
        raise RuntimeError('the analyzer\'s sorting provenance no longer matches its spikes; '
                           'splitting it would misalign every per-spike extension')


# per-spike extension arrays, indexed like the spike vector
_PER_SPIKE = {'spike_amplitudes': ('amplitudes',), 'spike_locations': ('spike_locations',),
              'amplitude_scalings': ('amplitude_scalings', 'collision_mask')}
# extension arrays with one row per random spike, in random_spikes_indices order
_PER_RANDOM_SPIKE = {'waveforms': ('waveforms',), 'principal_components': ('pca_projection',)}


def _set_random_spikes(rs_ext, indices):
    """Replace the random-spike indices, and drop the vector SpikeInterface caches from them.

    ComputeRandomSpikes.get_random_spikes keeps spikes[random_spikes_indices] in
    _some_spikes the first time it is asked and never refreshes it, and the waveforms
    extension picks each unit's rows through it -- so new indices without clearing it
    hand get_waveforms_one_unit a mask of the old length or the old spikes.
    """
    rs_ext.data['random_spikes_indices'] = np.asarray(indices)
    if hasattr(rs_ext, '_some_spikes'):
        del rs_ext._some_spikes


def select_units_aligned(analyzer, unit_ids):
    """analyzer.select_units(unit_ids) with the per-spike extensions kept on their spikes.

    SpikeInterface 0.104.8 slices the per-spike extension arrays (amplitudes, locations,
    scalings, and the random-spike rows) with a mask taken in the *current* spike-vector
    order, but the selected sorting rebuilds its vector lexsorted by (segment, sample,
    unit index). Whenever the current order is not that -- after a split, which gives
    the split spikes appended unit indices without re-sorting, or a merge -- spikes that
    share a sample swap places and each takes the other's values, in untouched units as
    much as changed ones. On DS45-like sorts that is a few thousand silent swaps per
    split. Here the new order is predicted, checked against what SpikeInterface built,
    and every per-spike array is permuted into it.
    """
    unit_ids = list(unit_ids)
    ids = list(analyzer.unit_ids)
    # SpikeInterface's unit-indexed extensions are sliced in the analyzer's order, not the
    # order asked for, so only an order-preserving selection is safe
    pos = [ids.index(u) for u in unit_ids]
    if pos != sorted(pos):
        raise ValueError('select_units_aligned needs unit_ids in the analyzer\'s own order')
    new_index = np.full(len(ids), -1, dtype=np.int64)
    new_index[[ids.index(u) for u in unit_ids]] = np.arange(len(unit_ids))
    V = analyzer.sorting.to_spike_vector()
    kept = V[new_index[V['unit_index']] >= 0]
    nu = new_index[kept['unit_index']]
    perm = np.lexsort((nu, kept['sample_index'], kept['segment_index']))
    new = analyzer.select_units(unit_ids, format='memory')
    W = new.sorting.to_spike_vector()
    if not (np.array_equal(W['sample_index'], kept['sample_index'][perm])
            and np.array_equal(W['unit_index'], nu[perm])):
        raise RuntimeError('select_units built a spike vector in an order other than the '
                           'predicted lexsort; the per-spike realignment would be wrong')
    if np.array_equal(perm, np.arange(perm.size)):
        return new
    for name, keys in _PER_SPIKE.items():
        if new.has_extension(name):
            data = new.get_extension(name).data
            for k in keys:
                if k in data:
                    data[k] = data[k][perm]
    if new.has_extension('random_spikes'):
        rs_ext = new.get_extension('random_spikes')
        inv = np.empty_like(perm)
        inv[perm] = np.arange(perm.size)
        # SpikeInterface remapped the random spikes to positions in the kept, old order
        pos = inv[np.asarray(rs_ext.data['random_spikes_indices'])]
        order = np.argsort(pos, kind='stable')
        _set_random_spikes(rs_ext, pos[order])
        for name, keys in _PER_RANDOM_SPIKE.items():
            if new.has_extension(name):
                data = new.get_extension(name).data
                for k in keys:
                    if k in data:
                        data[k] = data[k][order]
    return new


def drop_spikes(analyzer, drops, **job_kwargs):
    """Remove spikes by per-unit index. Returns (analyzer, {old_id: new_id}).

    drops : {unit_id: indices into that unit's spike train}. Units with no drops are not
    touched and keep their ids. Implemented as one soft split into [kept, dropped] and
    one remove_units of the dropped halves.
    """
    drops = {u: np.unique(np.asarray(d, dtype=np.int64)) for u, d in drops.items() if len(d)}
    if not drops:
        return analyzer, {}
    _check_provenance(analyzer)
    counts = {u: int(n) for u, n in analyzer.sorting.count_num_spikes_per_unit().items()}
    splits = {}
    for u, d in drops.items():
        keep = np.setdiff1d(np.arange(counts[u]), d)
        if keep.size == 0:
            raise RuntimeError(f'unit {u}: every spike would be dropped')
        splits[u] = [keep.tolist(), d.tolist()]
    # SpikeInterface 0.104.8's templates split indexes with np.array([]) of the unsplit
    # units, a float array when there are none, and raises IndexError. So when every
    # unit loses a spike, one unit is split in a second call.
    batches = [splits]
    if len(splits) == len(analyzer.unit_ids) and len(splits) > 1:
        last = list(splits)[-1]
        batches = [{u: v for u, v in splits.items() if u != last}, {last: splits[last]}]
    idmap, current = {}, analyzer
    for batch in batches:
        split, new_ids = current.split_units(batch, new_id_strategy='append',
                                             return_new_unit_ids=True, format='memory',
                                             **job_kwargs)
        idmap.update({u: ids[0] for u, ids in zip(batch, new_ids)})
        # every unit but the dropped halves, in the split analyzer's own unit order.
        # Never a reordered list: select_units orders the new sorting as asked but
        # slices its unit-indexed extensions (templates, locations, similarity) in the
        # analyzer's existing order, so a reordered selection hands every unit
        # another unit's template.
        drop_halves = {ids[1] for ids in new_ids}
        current = select_units_aligned(split, [u for u in split.unit_ids if u not in drop_halves])
    return current, idmap


def _pair_drops(analyzer, scores, t_c, method, exclude=None, unit_ids=None, prefer=None):
    """{unit: drop indices} resolving every pair <= t_c, and a per-unit count table.

    exclude  : {unit: indices already being dropped}; those spikes are taken out before
               the pairs are found, so an outlier is not counted twice and its removal
               can close no pair it was part of.
    unit_ids : resolve only these units (all by default). In template mode every unit
               resolved must have scores.
    prefer   : {unit: bool per spike}; in a pair with one preferred spike, that spike is
               kept whatever the other test says (the redundant merges' reference copy)
    """
    samples = _unit_samples(analyzer)
    if unit_ids is not None:
        samples = {u: samples[u] for u in unit_ids}
    drops, rows = {}, []
    for u, s in samples.items():
        idx = np.arange(s.size)
        if exclude and u in exclude and len(exclude[u]):
            idx = np.setdiff1d(idx, exclude[u])
        bad = None
        if method == 'template':
            if u not in scores:
                raise KeyError(f'unit {u} has no spike scores to resolve its pairs by')
            bad = _badness(scores[u])[idx]
        if prefer and u in prefer:
            # badness is 1 - a correlation, within [0, 2]: lowering preferred spikes by 2
            # puts every one of them ahead of every other; in order mode they alone count
            pref = np.asarray(prefer[u], dtype=bool)[idx]
            bad = (bad if bad is not None else np.zeros(idx.size)) - 2.0 * pref
        local = resolve_pairs(s[idx], bad, t_c)
        n_pairs = int(np.sum(np.diff(s[idx]) <= t_c)) if idx.size > 1 else 0
        rows.append({'unit_id': u, 'n_spikes': int(s.size), 'close_pairs': n_pairs,
                     'dropped': int(local.size)})
        if local.size:
            drops[u] = idx[local]
    return drops, pd.DataFrame(rows).set_index('unit_id')


def _rp_contamination(train, n_samples, t_c, t_r):
    """SpikeInterface's Llobet contamination of one sorted train, same counter, same formula.

    Every pair within t_r counts (not only adjacent ones, and lag 0 included), exactly
    as misc_metrics' numba counter does.
    """
    from spikeinterface.metrics.quality.misc_metrics import _compute_rp_contamination_one_unit
    n_v = int(np.sum(np.searchsorted(train, train + t_r, side='right') - np.arange(train.size) - 1))
    return _compute_rp_contamination_one_unit(n_v, train.size, n_samples, t_c, t_r)


def _quality_score_pairs(analyzer, pairs, w_ms, r_ms, balance):
    """compute_merge_unit_groups' 'quality_score' step, on the two trains of each pair.

    Same test as SpikeInterface's check_improve_contaminations_score: merge the two
    trains, drop the later spike of every pair within W, and keep the pair only if
    f * (1 - (1 + balance) * c) of the merged unit is at least that of each parent. It
    builds a merged copy of the *whole* sorting per candidate pair to do it, which on
    DS45 (11M spikes, 496 units) ran for over an hour without finishing; the two
    trains are all the test needs. Checked to keep exactly SpikeInterface's pairs.
    """
    from spikeinterface.curation.mergeunitssorting import get_non_duplicated_events
    fs = analyzer.sampling_frequency
    n = analyzer.get_total_samples()
    t_c, t_r = _t_c(fs, w_ms), _t_c(fs, r_ms)
    dur = n / fs
    k = 1.0 + balance
    trains = _unit_samples(analyzer)
    cache = {}

    def score(train):
        return train.size / dur * (1 - k * _rp_contamination(train, n, t_c, t_r))

    keep, dropped = [], []
    for u1, u2 in pairs:
        for u in (u1, u2):
            if u not in cache:
                cache[u] = score(trains[u])
        # the merged train exactly as SpikeInterface's MergeUnitsSorting(delta_time_ms=W)
        # builds it for this test: cross-unit spikes within W are dropped, by order
        merged = get_non_duplicated_events([trains[u1], trains[u2]], int(w_ms / 1000 * fs))
        s_new = score(merged)
        (keep if s_new >= cache[u1] and s_new >= cache[u2] else dropped).append((u1, u2))
    return keep, dropped


def failed_criteria(row, section, nan_fails=False):
    """'metric=value' for every threshold of one bombcell section this unit is outside.

    row: the unit's metrics (quality + template); section: e.g. thresholds['mua'].
    nan_fails: list a NaN metric as a failure -- True for the noise section, where
    bombcell fails a unit on it; its MUA and non-somatic steps ignore NaN.
    """
    out = []
    for k, c in section.items():
        v = row.get(k, np.nan)
        if pd.isna(v):
            if nan_fails:
                out.append(f'{k}=nan')
            continue
        x = abs(v) if c.get('abs') else v
        if (c.get('greater') is not None and x < c['greater']) or \
                (c.get('less') is not None and x > c['less']):
            out.append(f'{k}={v:.3g}')
    return ', '.join(out)


def label_table(analyzer, labels, ks_names, thresholds=None, previous_labels=None,
                ks_labels=None, dropped_noise=None):
    """One row per unit: its labels before and after, and the bombcell criteria it failed.

    analyzer        : the analyzer the labels were computed on (its metric extensions)
    labels          : bombcell labels frame (from _bombcell_with_rp_minimum)
    ks_names        : {unit id: 'ks1+ks2' / 'ks/1'} the Kilosort units it came from
    previous_labels : {Kilosort unit: its label in an earlier curation}, if any
    ks_labels       : {Kilosort unit: Kilosort's own KSLabel}, if any
    dropped_noise   : the noise_units_dropped table, added as rows of their own

    failed_mua lists the refractory criterion only where it counted, the lower
    confidence bound over RP_MAX_CONTAMINATION on more than RP_MAX_FREE_VIOLATIONS
    violations; a unit whose point estimate alone is
    over the limit shows it in rp_not_counted instead. failed_non_somatic lists the shape metrics outside
    their thresholds; bombcell combines these (not every one alone makes a unit
    non-somatic), so read it as the evidence, not the rule.
    """
    th = thresholds or default_thresholds()
    metrics = analyzer.get_metrics_extension_data()
    mua_wo_rp = {k: v for k, v in th['mua'].items() if k != 'rp_contamination'}

    def before(name, table, missing='?'):
        if table is None:
            return ''
        parts = [str(p).split('/')[0] for p in str(name).split('+')]
        return '+'.join(str(table.get(type(next(iter(table)))(p), missing)) for p in parts)

    # an earlier curation kept every unit it did not call noise, so a unit missing from
    # it was noise there
    def prev(name):
        return before(name, previous_labels, missing='noise')

    rows = []
    for u in labels.index:
        m = metrics.loc[u] if u in metrics.index else pd.Series(dtype=float)
        c, c_lo = m.get('rp_contamination', np.nan), labels.loc[u, 'rp_contamination_lo']
        rp_fail = (f'rp_contamination={c:.3g} (lower bound {c_lo:.3g})'
                   if pd.notna(c) and c > RP_MAX_CONTAMINATION else '')
        counted = bool(labels.loc[u, 'rp_counted'])
        mua = failed_criteria(m, mua_wo_rp)
        if counted:
            mua = ', '.join(x for x in (mua, rp_fail) if x)
        name = ks_names.get(u, str(u))
        rows.append({'unit_id': u, 'ks_unit_ids': name, 'KSLabel': before(name, ks_labels),
                     'previous_label': prev(name),
                     'label': labels.loc[u, 'bombcell_label'],
                     'failed_noise': failed_criteria(m, th['noise'], nan_fails=True),
                     'failed_non_somatic': failed_criteria(m, th.get('non-somatic', {})),
                     'failed_mua': mua,
                     'rp_not_counted': f"{rp_fail} ({int(labels.loc[u, 'rp_violations'])} violations)"
                                       if rp_fail and not counted else '',
                     'n_spikes': int(m.get('num_spikes', np.nan)) if pd.notna(m.get('num_spikes', np.nan)) else None,
                     'firing_rate': m.get('firing_rate', np.nan)})
    table = pd.DataFrame(rows)
    if dropped_noise is not None and len(dropped_noise):
        extra = pd.DataFrame({'unit_id': dropped_noise['unit_id'], 'ks_unit_ids': dropped_noise['name'],
                              'KSLabel': [before(n, ks_labels) for n in dropped_noise['name']],
                              'previous_label': [prev(n) for n in dropped_noise['name']],
                              'label': 'noise (dropped after splitting)',
                              'failed_noise': dropped_noise['failed'], 'failed_non_somatic': '',
                              'failed_mua': '', 'rp_not_counted': '',
                              'n_spikes': dropped_noise['n_spikes'],
                              'firing_rate': dropped_noise['firing_rate']})
        table = pd.concat([table, extra], ignore_index=True)
    if previous_labels is not None:
        table['changed'] = [p != '' and p != l.split(' (')[0] for p, l in
                            zip(table['previous_label'], table['label'])]
    return table


def _noise_units(analyzer, thresholds=None, names=None):
    """The units bombcell calls noise, with the waveform-shape criteria each one failed.

    Only bombcell's noise section decides here; its thresholds are on the template's
    shape (peaks and troughs, duration, baseline flatness, spatial decay), which is what
    no spike-level cleaning changes. Returns a table: unit_id, name, n_spikes,
    firing_rate, failed.
    """
    th = copy.deepcopy(thresholds or default_thresholds())
    labels = sic.bombcell_label_units(analyzer, thresholds=th, label_non_somatic=True,
                                      split_non_somatic_good_mua=True)
    noise = labels.index[labels['bombcell_label'] == 'noise'].tolist()
    if not noise:
        return pd.DataFrame()
    metrics = analyzer.get_metrics_extension_data()
    counts = analyzer.sorting.count_num_spikes_per_unit()
    dur = analyzer.get_total_duration()
    rows = []
    for u in noise:
        failed = []
        for k, c in th['noise'].items():
            v = metrics.loc[u, k] if k in metrics.columns else np.nan
            if pd.isna(v):
                failed.append(f'{k}=nan')
                continue
            if (c.get('greater') is not None and v < c['greater']) or \
                    (c.get('less') is not None and v > c['less']):
                failed.append(f'{k}={v:.3g}')
        rows.append({'unit_id': u, 'name': (names or {}).get(u, str(u)),
                     'n_spikes': int(counts[u]), 'firing_rate': counts[u] / dur,
                     'failed': ', '.join(failed)})
    return pd.DataFrame(rows)


def rp_contamination_lower_bound(analyzer, confidence=RP_CONFIDENCE):
    """The one-sided lower confidence bound of each unit's rp_contamination.

    The violation count is Poisson; its exact lower bound (chi-square, Garwood) goes
    through the same Llobet formula, which rises with the count, with the censored and
    refractory periods the quality_metrics extension was computed with.
    """
    qm = analyzer.get_extension('quality_metrics')
    p = qm.params['metric_params']['rp_violation']
    fs = analyzer.sampling_frequency
    t_c = int(round(p['censored_period_ms'] * fs * 1e-3))
    t_r = int(round(p['refractory_period_ms'] * fs * 1e-3))
    n = analyzer.get_total_samples()
    data = qm.get_data()
    return pd.Series({u: _rp_lower_bound(nv, ns, n, t_c, t_r, confidence)
                      for u, nv, ns in zip(data.index, data['rp_violations'], data['num_spikes'])})


def _rp_lower_bound(n_v, n_spikes, n_samples, t_c, t_r, confidence=RP_CONFIDENCE):
    """Lower confidence bound of the Llobet contamination for n_v violations (exact Poisson)."""
    from scipy.stats import chi2
    from spikeinterface.metrics.quality.misc_metrics import _compute_rp_contamination_one_unit
    if n_spikes <= 0:
        return np.nan
    nv_lo = chi2.ppf(1 - confidence, 2 * n_v) / 2 if n_v > 0 else 0.0
    return _compute_rp_contamination_one_unit(nv_lo, n_spikes, n_samples, t_c, t_r)


def _rp_fails(train, n_samples, t_c, t_r):
    """(fails the labels' refractory rule, violations, lower bound) for one sorted train.

    Violations are pairs in (t_c, t_r]: closer pairs are the same spike twice, which the
    pair passes resolve before the labels are made."""
    train = np.sort(train)
    idx = np.arange(train.size)
    n_v = int(np.sum(np.searchsorted(train, train + t_r, 'right') - idx - 1)
              - np.sum(np.searchsorted(train, train + t_c, 'right') - idx - 1))
    lo = _rp_lower_bound(n_v, train.size, n_samples, t_c, t_r)
    return bool(lo > RP_MAX_CONTAMINATION and n_v > RP_MAX_FREE_VIOLATIONS), n_v, lo


def _bombcell_with_rp_minimum(analyzer, thresholds=None):
    """Bombcell labels, with its refractory criterion applied to a confidence bound.

    Bombcell runs with every threshold but rp_contamination; the refractory criterion
    is then applied separately, and demotes a unit (good -> mua, non_soma_good ->
    non_soma_mua) when the RP_CONFIDENCE lower bound of its contamination is over
    RP_MAX_CONTAMINATION (see rp_contamination_lower_bound) and it has more than
    RP_MAX_FREE_VIOLATIONS violations. The labels frame gains
    rp_violations, rp_contamination, rp_contamination_lo, rp_counted (bound over the
    limit) and rp_demoted for the audit.
    """
    th = copy.deepcopy(thresholds or default_thresholds())
    th.get('mua', {}).pop('rp_contamination', None)
    labels = sic.bombcell_label_units(analyzer, thresholds=th, label_non_somatic=True,
                                      split_non_somatic_good_mua=True)
    qm = analyzer.get_extension('quality_metrics').get_data().loc[labels.index]
    lo = rp_contamination_lower_bound(analyzer).loc[labels.index]
    counted = (lo > RP_MAX_CONTAMINATION) & (qm['rp_violations'] > RP_MAX_FREE_VIOLATIONS)
    demote = counted & labels['bombcell_label'].isin(['good', 'non_soma_good'])
    labels.loc[demote, 'bombcell_label'] = labels.loc[demote, 'bombcell_label'].map(
        {'good': 'mua', 'non_soma_good': 'non_soma_mua'})
    return labels.assign(rp_violations=qm['rp_violations'], rp_contamination=qm['rp_contamination'],
                         rp_contamination_lo=lo, rp_counted=counted, rp_demoted=demote)


def _template_cosine(analyzer, u1, u2, max_shift=0):
    """Cosine similarity of two units' average templates, all channels.

    max_shift : also try the templates offset by up to this many samples, and return the
    best. Two sorter units detecting the same spikes a fixed few samples apart have one
    waveform, shifted: DS15_20260702 units 94 and 671 are 0.11 aligned, 0.99 at 8 samples.
    """
    t = analyzer.get_extension('templates')
    a = t.get_unit_template(u1, operator='average')
    b = t.get_unit_template(u2, operator='average')
    best = -1.0
    for s in range(-max_shift, max_shift + 1):
        x = a[max(0, s):a.shape[0] + min(0, s)].ravel()
        y = b[max(0, -s):b.shape[0] + min(0, -s)].ravel()
        best = max(best, float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12)))
    return best


def _cross_violations(train_a, train_b, t_c, t_r):
    """Pairs of a spike from each train at a lag in (t_c, t_r] samples, either order."""
    b = np.sort(train_b)
    hi = np.searchsorted(b, train_a + t_r, 'right') - np.searchsorted(b, train_a - t_r, 'left')
    lo = np.searchsorted(b, train_a + t_c, 'right') - np.searchsorted(b, train_a - t_c, 'left')
    return int((hi - lo).sum())


def _strict_merge_veto(analyzer, group, w_ms, r_ms, max_shift=0, max_cross=0):
    """Why a merge group fails the strict tests, or '' if it passes.

    max_shift : template similarity is taken at the best alignment within this many
    samples (see _template_cosine); 0 for the preset merges.
    max_cross : cross-unit spikes within (W, R] tolerated; 0 for the preset merges. The
    redundant merges take the refractory floor of the labels (RP_MAX_FREE_VIOLATIONS):
    copies of one neuron sharing thousands of spikes are not two neurons for one or two
    near-coincidences (DS48_20260924 344+362+388 failed a zero tolerance on 1).

    Two fragments of one neuron have one waveform, and cannot fire within a refractory
    period of each other. So every pair in the group needs template cosine similarity
    >= MERGE_MIN_SIMILARITY, and no pair may have spikes of one within (W, R] of the
    other. Pairs closer than W are allowed: those are the same spike detected by both,
    which the post-merge pass resolves. On DS45 the feature_neighbors preset (which
    has no template-similarity step) merged units with similarities down to -0.44 and
    up to 55 cross-unit refractory violations; every such unit came out worse.
    """
    fs = analyzer.sampling_frequency
    t_c, t_r = _t_c(fs, w_ms), _t_c(fs, r_ms)
    trains = _unit_samples(analyzer)
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            a, b = group[i], group[j]
            s = _template_cosine(analyzer, a, b, max_shift)
            if s < MERGE_MIN_SIMILARITY:
                return f'template similarity {a}-{b} {s:.2f} < {MERGE_MIN_SIMILARITY:g}'
            v = _cross_violations(trains[a], trains[b], t_c, t_r)
            if v > max_cross:
                return f'{v} cross-unit spikes {a}-{b} within ({w_ms:g}, {r_ms:g}] ms'
    return ''


def _merged_rp_veto(analyzer, group, w_ms=W_MS, r_ms=R_MS):
    """'' if the group's pooled train passes the labels' refractory rule, else why not.

    Being refractory *to each other* says nothing about each member's own contamination:
    on DS2_20260505 good units 250 and 311 were refractory to their MUA neighbours 248
    (26 violations) and 308 (136), and merging made both MUA."""
    fs = analyzer.sampling_frequency
    trains = _unit_samples(analyzer)
    pooled = np.concatenate([trains[u] for u in group])
    fails, n_v, lo = _rp_fails(pooled, analyzer.get_total_samples(), _t_c(fs, w_ms), _t_c(fs, r_ms))
    if fails:
        return (f'merged {"+".join(map(str, group))} not refractory: {n_v} violations, '
                f'contamination lower bound {lo:.2f}')
    return ''


def _drift_pair_test(analyzer, trains_s, a, b):
    """(is a drift pair, note) for two units: mutually refractory, still alike, and
    refractory as one unit."""
    obs, exp = sos.coincidences(trains_s[a], trains_s[b])
    s = _template_cosine(analyzer, a, b)
    if s < DRIFT_MIN_SIMILARITY:
        return False, f'template similarity {a}-{b} {s:.2f} < {DRIFT_MIN_SIMILARITY:g}'
    if not sos.mutually_refractory(obs, exp):
        return False, f'{a}-{b}: {obs} of {exp:.1f} expected coincidences: not mutually refractory'
    why = _merged_rp_veto(analyzer, [a, b])
    if why:
        return False, why
    return True, f'{a}-{b}: {obs} of {exp:.0f} expected coincidences, similarity {s:.2f}'


def _drift_pairs(analyzer):
    """Candidate drift pairs: neighbours that pass _drift_pair_test."""
    fs = analyzer.sampling_frequency
    ids = list(analyzer.unit_ids)
    loc = analyzer.get_extension('unit_locations').get_data()[:, :2]
    trains_s = {u: s / fs for u, s in _unit_samples(analyzer).items()}
    pairs = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if np.linalg.norm(loc[i] - loc[j]) <= DRIFT_MAX_DISTANCE_UM:
                if _drift_pair_test(analyzer, trains_s, ids[i], ids[j])[0]:
                    pairs.append((ids[i], ids[j]))
    return pairs, trains_s


def _drift_merge_veto(analyzer, group, trains_s):
    """Why a drift group fails, or ''. Every pair must look alike; a pair active together
    long enough to test must be mutually refractory, and one that is not (pieces from
    either end of a drift, joined through a middle one) may have no more cross-unit
    coincidences than the refractory floor of the labels."""
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            a, b = group[i], group[j]
            s = _template_cosine(analyzer, a, b)
            if s < DRIFT_MIN_SIMILARITY:
                return f'template similarity {a}-{b} {s:.2f} < {DRIFT_MIN_SIMILARITY:g}'
            obs, exp = sos.coincidences(trains_s[a], trains_s[b])
            if exp >= sos.MUTUAL_MIN_EXPECTED:
                if not sos.mutually_refractory(obs, exp):
                    return f'{a}-{b}: {obs} of {exp:.1f} expected coincidences'
            elif obs > RP_MAX_FREE_VIOLATIONS:
                return f'{a}-{b}: {obs} coincidences within {sos.REFRACTORY_MS:g} ms'
    return _merged_rp_veto(analyzer, group)


def auto_merge(analyzer, presets=MERGE_PRESETS, w_ms=W_MS, r_ms=R_MS, **job_kwargs):
    """Soft merges, one preset at a time. Returns (analyzer, merge table).

    compute_merge_unit_groups takes `preset` (singular); a misspelled keyword falls into
    its **job_kwargs and is silently ignored, which is how the pipeline's merges once
    ran the default preset every time and merged nothing. censor_ms is left None: it
    would resolve the merged units' close pairs by order, which is exactly what the
    post-merge pass does by fit instead.
    """
    from spikeinterface.curation.auto_merge import _compute_merge_presets, _default_step_params
    from spikeinterface.curation.curation_tools import resolve_merging_graph
    cols = ['preset', 'group', 'members', 'new_unit_id', 'merged', 'reference', 'note']
    rows = []
    balance = _default_step_params['quality_score']['firing_contamination_balance']
    for preset in presets:
        max_shift = 0
        trains_s = None
        if preset == DRIFT:
            pairs, trains_s = _drift_pairs(analyzer)
        elif preset == REDUNDANT:
            # the sorter's duplicates: units sharing most of their spikes. The presets miss
            # them -- their cross-correlogram peaks at 0 lag where a split unit's dips -- and
            # the pair passes only resolve pairs within a unit, so both copies survive.
            # Merged, the post-merge pass drops the doubled spikes, keeping the reference
            # member's copy of each (_redundant_preference); unique spikes of either stay.
            pairs = [tuple(p) for p in sic.find_redundant_units(analyzer.sorting, **REDUNDANT_PARAMS)]
            max_shift = _t_c(analyzer.sampling_frequency, w_ms)
            max_cross = RP_MAX_FREE_VIOLATIONS
        else:
            max_cross = 0
            steps = list(_compute_merge_presets[preset])
            own_score = 'quality_score' in steps
            if own_score:
                steps.remove('quality_score')
            # the 'correlogram' step zeroes a lag bin of the stored correlograms in place
            # (get_data hands back the array itself); they are put back after
            ccg_ext = analyzer.get_extension('correlograms')
            ccgs = ccg_ext.data['ccgs'].copy() if ccg_ext is not None else None
            pairs = sic.compute_merge_unit_groups(analyzer, steps=steps, resolve_graph=False,
                                                  steps_params=merge_steps_params(w_ms, r_ms),
                                                  **job_kwargs)
            if ccgs is not None:
                analyzer.get_extension('correlograms').data['ccgs'] = ccgs
            pairs = [tuple(p) for p in pairs]
            if own_score:
                pairs, rejected = _quality_score_pairs(analyzer, pairs, w_ms, r_ms, balance)
                for p in rejected:
                    rows.append({'preset': preset, 'group': '+'.join(map(str, p)),
                                 'members': list(p), 'new_unit_id': None, 'merged': False,
                                 'reference': None,
                                 'note': 'quality score not improved by merging'})
        groups = resolve_merging_graph(analyzer.sorting, pairs) if pairs else []
        # every surviving group has to pass the strict tests between *all* its members
        # (a resolved group joins units that were only proposed through a third)
        strict = []
        for g in groups:
            why = (_drift_merge_veto(analyzer, list(g), trains_s) if preset == DRIFT else
                   _strict_merge_veto(analyzer, list(g), w_ms, r_ms, max_shift, max_cross))
            if why:
                rows.append({'preset': preset, 'group': '+'.join(map(str, g)), 'members': list(g),
                             'new_unit_id': None, 'merged': False, 'reference': None,
                             'note': why})
            else:
                strict.append(g)
        groups = strict
        if not groups:
            rows.append({'preset': preset, 'group': '', 'members': [], 'new_unit_id': None,
                         'merged': False, 'reference': None, 'note': 'no merge passes'})
            continue
        mergeable = analyzer.are_units_mergeable(groups, merging_mode='soft')
        ok = [list(g) for g, m in zip(groups, mergeable.values()) if m]
        for g, m in zip(groups, mergeable.values()):
            if not m:
                rows.append({'preset': preset, 'group': '+'.join(map(str, g)),
                             'members': list(g), 'new_unit_id': None, 'merged': False,
                             'reference': None,
                             'note': 'sparsity overlap below 0.75: not soft-mergeable'})
        if not ok:
            continue
        counts = analyzer.sorting.count_num_spikes_per_unit()
        analyzer, new_ids = analyzer.merge_units(ok, merging_mode='soft', censor_ms=None,
                                                 new_id_strategy='append',
                                                 return_new_unit_ids=True, format='memory',
                                                 raise_error_if_overlap_fails=True,
                                                 **job_kwargs)
        for g, n in zip(ok, new_ids):
            # a redundant group's reference is its largest member, whose copy of every
            # doubled spike the post-merge pass keeps
            ref = max(g, key=lambda u: counts[u]) if preset == REDUNDANT else None
            rows.append({'preset': preset, 'group': '+'.join(map(str, g)), 'members': g,
                         'new_unit_id': n, 'merged': True, 'reference': ref, 'note': ''})
    return analyzer, pd.DataFrame(rows, columns=cols)


def _redundant_preference(before, merged, merged_ids, merges):
    """{merged unit: bool per spike, True for the redundant reference member's spikes}.

    A redundant merge joins copies of one neuron's spikes, possibly offset by a fixed few
    samples; resolving its doubled pairs by template match would keep a mix of both
    alignments and smear the template, so the reference member's copy wins every pair
    (the match still decides within each member). Parents are read by position, as a soft
    merge relabels the spike vector in place. Merged units with no redundant merge in
    their history are left out.
    """
    if merges is None or not len(merges) or 'reference' not in merges.columns:
        return {}
    red = merges[merges['merged'].astype(bool) & (merges['preset'] == REDUNDANT)]
    if not len(red):
        return {}
    sv, bv = merged.sorting.to_spike_vector(), before.sorting.to_spike_vector()
    ids, before_ids = list(merged.unit_ids), list(before.unit_ids)
    out = {}
    for _, r in red.iterrows():
        m = r['new_unit_id']
        if m not in merged_ids:
            continue
        # the reference by identity among the members: a table column holding ids and
        # NaNs (unmerged rows) turns them into floats
        ref = next(x for x in r['members'] if x == r['reference'])
        ref_leaves = {before_ids.index(x) for x in _leaves(ref, merges)}
        pos = np.flatnonzero(sv['unit_index'] == ids.index(m))
        out[m] = np.isin(bv['unit_index'][pos], list(ref_leaves))
    return out


def _leaves(u, merges):
    """The pre-merge units a merged unit is made of, through merges of merged units."""
    made = {r['new_unit_id']: r['members'] for _, r in merges[merges['merged'].astype(bool)].iterrows()}
    out, stack = [], [u]
    while stack:
        x = stack.pop()
        if x in made:
            stack.extend(made[x])
        else:
            out.append(x)
    return out


def _thin_merged_random_spikes(before, merged, merged_ids, merges, seed=0):
    """Give every merged unit a random-spike sample proportional to its parents' sizes.

    A soft merge keeps each parent's own random spikes, each capped at the analyzer's
    max_spikes_per_unit, so a 243-spike fragment merged into a 17,000-spike unit brings
    as many random spikes as its whole train while the big parent is capped. SpikeInterface
    count-weights the merged template correctly, but the sample behind it is not
    uniform, and the next soft split (the post-merge pair pass) rebuilds the template as
    the plain mean of that sample -- the fragment then makes up a third of it. Thinning
    each parent to the same sampling rate makes the sample uniform over the merged unit's
    spikes; its template is then reset to the mean of what is kept, so template and
    waveforms agree whether or not it is split later. Modifies `merged` in place.
    """
    rng = np.random.default_rng(seed)
    rs_ext = merged.get_extension('random_spikes')
    rs = np.asarray(rs_ext.data['random_spikes_indices'])
    sv = merged.sorting.to_spike_vector()
    # a soft merge relabels spikes in place (censor_ms=None drops none), so a merged
    # spike's parent is the unit the same position held before -- by position, not by
    # sample: sorter units that detected the same spikes share samples (DS45 has 748k
    # such cross-unit pairs), and a sample lookup gives every shared spike to one parent
    bv = before.sorting.to_spike_vector()
    if not np.array_equal(sv['sample_index'], bv['sample_index']):
        raise RuntimeError('the merged analyzer\'s spike vector is not the pre-merge one '
                           'relabelled; parents cannot be read by position')
    ids, before_ids = list(merged.unit_ids), list(before.unit_ids)
    counts = before.sorting.count_num_spikes_per_unit()
    drop_pos = []
    for m in merged_ids:
        leaves = _leaves(m, merges)
        pos = np.flatnonzero(sv['unit_index'][rs] == ids.index(m))
        src = bv['unit_index'][rs[pos]]
        n = np.array([counts[leaf] for leaf in leaves], dtype=np.int64)
        r = np.array([np.sum(src == before_ids.index(leaf)) for leaf in leaves], dtype=np.int64)
        valid = (n > 0) & (r > 0)
        if not valid.any():
            continue
        # the parent sampled most sparsely sets the rate, and keeps all of its random
        # spikes; the others keep n_i * r_j / n_j, in integers so no float floor rounds
        # the rate-setting parent itself down by one
        j = np.flatnonzero(valid)[np.argmin(r[valid] / n[valid])]
        for i, leaf in enumerate(leaves):
            mine = pos[src == before_ids.index(leaf)]
            keep = int(n[i] * r[j] // n[j])
            if mine.size > keep:
                drop_pos.extend(rng.choice(mine, mine.size - keep, replace=False).tolist())
    if not drop_pos:
        return
    keep_mask = np.ones(rs.size, bool)
    keep_mask[drop_pos] = False
    _set_random_spikes(rs_ext, rs[keep_mask])
    for name, keys in _PER_RANDOM_SPIKE.items():
        if merged.has_extension(name):
            data = merged.get_extension(name).data
            for k in keys:
                if k in data:
                    data[k] = data[k][keep_mask]
    tpl = merged.get_extension('templates')
    wf = merged.get_extension('waveforms')
    for m in merged_ids:
        w = wf.get_waveforms_one_unit(m, force_dense=True)
        k = ids.index(m)
        for op, fn in (('average', np.mean), ('std', np.std), ('median', np.median)):
            if op in tpl.data:
                tpl.data[op][k] = fn(w, axis=0)


def _trace_ks_units(final_ids, split_maps, merges, origin=None):
    """{final unit: 'ks1+ks2'} through every split rename and merge, for the audit.

    origin: names already given to units before these stages -- the halves of a
    bimodal split are 'parent/1', 'parent/2'.
    """
    origin = dict(origin or {})

    def resolve(u):
        return origin.get(u, str(u))

    # replay the id history in order: pre-merge split, merges, post-merge split
    stages = [('split', split_maps[0])] + [('merge', merges)] + [('split', split_maps[1])]
    for kind, data in stages:
        if kind == 'split':
            for old, new in data.items():
                origin[new] = resolve(old)
        else:
            # the member ids themselves, not the joined string: parsing it back would
            # need to know the ids' type
            for _, r in data[data['merged'].astype(bool)].iterrows():
                origin[r['new_unit_id']] = '+'.join(resolve(p) for p in r['members'])
    return {u: resolve(u) for u in final_ids}


def _score_key(analyzer, stim_intervals):
    """A digest of everything score_outliers' result depends on."""
    import hashlib
    import inspect
    h = hashlib.sha1()
    sv = analyzer.sorting.to_spike_vector()
    for arr in (sv['sample_index'], sv['unit_index'], np.asarray(analyzer.unit_ids).astype(str)):
        h.update(np.ascontiguousarray(arr).tobytes())
    h.update(np.ascontiguousarray(analyzer.get_extension('spike_locations').get_data()).tobytes())
    h.update(np.ascontiguousarray(np.empty((0, 2)) if stim_intervals is None
                                  else np.asarray(stim_intervals, float)).tobytes())
    h.update(repr((sos.OUTLIER_Z, sos.N_CHANNELS, sos.MIN_SPIKES, sos.MAX_FIT_SPIKES,
                   sos.MIN_OUTLIERS, sos.MAX_OUTLIER_FRAC, sos.SHAPE_OUTLIER_SHARE,
                   sos.KEPT_SNR_MIN, sos.N_NULL, sos.REFRACTORY_MS)).encode())
    h.update(inspect.getsource(sos).encode())
    return h.hexdigest()


def score_outliers(analyzer, stim_intervals=None, cache_file=None, **job_kwargs):
    """Step 1's scoring, (splits, audit, scores), optionally cached to a pickle.

    The scoring is the slow part of the cleaning -- one extraction of every spike on its
    unit's strongest channels, a robust fit per unit -- and it depends only on the
    analyzer and the stimulation, so a control arm or a rerun after a later failure
    can reuse it. A cache is trusted only when everything the scores depend on hashes
    the same: the spikes (times and units), the features the scorer reads, the
    stimulation, the scorer's parameters and its source.
    """
    key = _score_key(analyzer, stim_intervals)
    if cache_file is not None and Path(cache_file).exists():
        with open(cache_file, 'rb') as f:
            cached = pickle.load(f)
        if cached.get('key') == key:
            print(f'  outlier scores loaded from {cache_file}')
            return cached['result']
    result = sos.find_outlier_spikes(analyzer, stim_intervals=stim_intervals,
                                     return_scores=True, **job_kwargs)
    if cache_file is not None:
        Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump({'key': key, 'result': result}, f)
    return result


def split_bimodal_units(analyzer, outlier_result, stim_intervals=None, max_depth=SPLIT_MAX_DEPTH,
                        **job_kwargs):
    """Split every unit the scorer found to hold two separate clusters; both halves kept.

    Returns (analyzer, outlier_result, split table, origin, levels): the split analyzer
    (memory), the outlier scores with each split unit replaced by its halves' own
    (re-scored: a mixture's core describes neither population), one row per split,
    {new unit id: 'parent/k' name} for tracing, and [(analyzer, {parent: labels})] per
    level for the review figures. A half that is itself two clusters is split again, to
    max_depth levels. Units that are not split keep their ids, spikes and scores.
    """
    splits_o, audit, scores = outlier_result
    splits_o, scores = dict(splits_o), dict(scores)
    todo = [u for u, sc in scores.items() if sc.get('split') is not None]
    if not todo:
        return analyzer, outlier_result, pd.DataFrame(), {}, []
    _check_provenance(analyzer)
    origin, rows, levels, current = {}, [], [], analyzer
    for depth in range(max_depth):
        if not todo:
            break
        sp = {u: [np.flatnonzero(scores[u]['split'] == k).tolist() for k in (0, 1)] for u in todo}
        levels.append((current, {u: scores[u]['split'] for u in todo}))
        new, new_ids = current.split_units(sp, new_id_strategy='append', return_new_unit_ids=True,
                                           format='memory', **job_kwargs)
        # the split leaves the vector in the old order under new unit indices; the
        # aligned selection re-sorts it with the per-spike data (see select_units_aligned)
        new = select_units_aligned(new, list(new.unit_ids))
        children = []
        for u, pair in zip(sp, new_ids):
            name = origin.get(u, str(u))
            for k, c in enumerate(pair):
                origin[c] = f'{name}/{k + 1}'
                children.append(c)
            a = audit.loc[u]
            rows.append({'unit': name, 'depth': depth + 1, 'children': [origin[c] for c in pair],
                         'child_ids': list(pair), 'sizes': a['split_sizes'], 'D': a['split_D'],
                         'valley': a['split_valley'], 'overlap': a['split_overlap']})
        c_splits, c_audit, c_scores = sos.find_outlier_spikes(
            new, unit_ids=children, stim_intervals=stim_intervals, return_scores=True, **job_kwargs)
        for u in sp:
            scores.pop(u)
            splits_o.pop(u, None)
        audit = pd.concat([audit.drop(index=list(sp)), c_audit])
        scores.update(c_scores)
        splits_o.update(c_splits)
        todo = [c for c in children if scores[c].get('split') is not None]
        current = new
    return current, (splits_o, audit, scores), pd.DataFrame(rows), origin, levels


def _resolve_merged_pairs(a1, a2, merges, merged_ids, map1, stim_intervals, method, t_c,
                          **job_kwargs):
    """Resolve the close pairs that merging created, in the merged units only.

    a1 -> a2 is the soft merge; map1 the {input unit: a1 unit} renames of the split before
    it. Returns (a3, map2, pairs table, [removed-spike frames]); a3 is re-sorted and has
    no pair <= t_c left anywhere.
    """
    map2, pairs_post, removed = {}, pd.DataFrame(), []
    if merged_ids:
        post_scores = {}
        if method == 'template':
            _, _, post_scores = sos.find_outlier_spikes(
                a2, unit_ids=merged_ids, stim_intervals=stim_intervals,
                return_scores=True, **job_kwargs)
        prefer = _redundant_preference(a1, a2, merged_ids, merges)
        sub_drops, pairs_post = _pair_drops(a2, post_scores, t_c, method, unit_ids=merged_ids,
                                            prefer=prefer)
        # a merged spike's input unit: the a1 unit at the same vector position (a soft
        # merge relabels in place), and that unit's input id through the pre-merge split
        sv1, sv2 = a1.sorting.to_spike_vector(), a2.sorting.to_spike_vector()
        inv1 = {new: old for old, new in map1.items()}
        a1_ids, a2_ids = list(a1.unit_ids), list(a2.unit_ids)
        for m, idx in sub_drops.items():
            pos = np.flatnonzero(sv2['unit_index'] == a2_ids.index(m))[idx]
            parents = [inv1.get(a1_ids[k], a1_ids[k]) for k in sv1['unit_index'][pos]]
            why = 'close pair (redundant merge)' if m in prefer else 'close pair (post-merge)'
            removed.append(pd.DataFrame({'unit': parents, 'sample': sv2['sample_index'][pos],
                                         'reason': why}))
        a3, map2 = drop_spikes(a2, sub_drops, **job_kwargs)
    else:
        a3 = a2
    # a merge relabels spikes without re-sorting the vector; an order-preserving aligned
    # selection re-sorts it with the per-spike data, so the analyzer can be saved (a
    # save rebuilds the sorting lexsorted and would copy per-spike arrays by position)
    a3 = select_units_aligned(a3, list(a3.unit_ids))
    # the premise of the censored metric: no pair at or under t_c anywhere
    worst = {u: int(np.diff(s).min()) for u, s in _unit_samples(a3).items() if s.size > 1}
    offenders = {u: g for u, g in worst.items() if g <= t_c}
    if offenders:
        raise AssertionError(f'{len(offenders)} units still have a pair <= {t_c} samples: '
                             f'{list(offenders.items())[:10]}')
    return a3, map2, pairs_post, removed


def _metrics_and_labels(a3, sim_params, loc_params, tm_params, w_ms, r_ms, bombcell_thresholds,
                        **job_kwargs):
    """Template-derived extensions and censored quality metrics for every unit, then labels.

    Merged units' templates were reset by the thinning, and the merge search leaves an l1
    template_similarity behind, so everything derived from templates is recomputed with
    the params it had (None: not computed).
    """
    if sim_params is not None:
        a3.compute('template_similarity', **sim_params, **job_kwargs)
    if loc_params is not None:
        a3.compute('unit_locations', **loc_params, **job_kwargs)
    if tm_params is not None:
        a3.compute('template_metrics', **tm_params, **job_kwargs)
    with _metric_defaults_restored():
        a3.compute('quality_metrics', metric_params=rp_metric_params(w_ms, r_ms),
                   delete_existing_metrics=False, **job_kwargs)
    qm_params = a3.get_extension('quality_metrics').params['metric_params']
    assert qm_params['rp_violation']['censored_period_ms'] == w_ms, qm_params['rp_violation']
    return _bombcell_with_rp_minimum(a3, bombcell_thresholds)


def _extension_params(analyzer, name):
    return (copy.deepcopy(dict(analyzer.get_extension(name).params))
            if analyzer.has_extension(name) else None)


def merge_redundant_units(analyzer, stim_intervals=None, method='template', w_ms=W_MS, r_ms=R_MS,
                          bombcell_thresholds=None, **job_kwargs):
    """Merge the redundant (duplicate) units of an already cleaned analyzer.

    For a session cleaned before the REDUNDANT merge existed. Every step before the merges
    works unit by unit -- outliers are fitted on each unit's own spikes, close pairs are
    resolved within a unit -- so the duplicates in a saved analyzer_clean are already
    cleaned exactly as a full run would clean them; what is left is the part after the
    merge: resolve the merged units' doubled spikes, recompute templates and metrics, and
    relabel. Other units keep their spikes, and their labels (bombcell reads each unit's
    own metrics).

    analyzer : an analyzer_clean (not modified; the result is in memory)
    Returns (clean analyzer, merges table, removed-spike table, labels, report). The clean
    analyzer is the input itself when nothing merges.
    """
    fs = analyzer.sampling_frequency
    t_c = _t_c(fs, w_ms)
    n_in = int(analyzer.sorting.to_spike_vector().size)
    params = {n: _extension_params(analyzer, n)
              for n in ('template_similarity', 'unit_locations', 'template_metrics')}
    names = analyzer.sorting.get_property('ks_unit_ids')
    names = dict(zip(analyzer.unit_ids, names)) if names is not None else {}
    a1 = analyzer.save_as(format='memory')  # the merges modify their input's extensions
    print('redundant units: merging')
    a2, merges = auto_merge(a1, (REDUNDANT,), w_ms, r_ms, **job_kwargs)
    merged = merges['merged'].astype(bool) if len(merges) else pd.Series(dtype=bool)
    merged_ids = [u for u in merges.loc[merged, 'new_unit_id'].tolist() if u in list(a2.unit_ids)]
    report = {'redundant_groups_found': int((merges['group'] != '').sum()) if len(merges) else 0,
              'redundant_merges': len(merged_ids), 'n_units_in': len(analyzer.unit_ids),
              'n_spikes_in': n_in}
    if not merged_ids:
        print('redundant units: nothing passes the strict tests')
        return analyzer, merges, pd.DataFrame(columns=['unit', 'sample', 'reason']), None, report
    _thin_merged_random_spikes(a1, a2, merged_ids, merges)
    print(f'redundant units: resolving pairs in {len(merged_ids)} merged units')
    a3, map2, pairs_post, removed = _resolve_merged_pairs(
        a1, a2, merges, merged_ids, {}, stim_intervals, method, t_c, **job_kwargs)
    print('redundant units: quality metrics and bombcell')
    labels = _metrics_and_labels(a3, params['template_similarity'], params['unit_locations'],
                                 params['template_metrics'], w_ms, r_ms, bombcell_thresholds,
                                 **job_kwargs)
    a3.sorting.set_property('bombcell_label', labels.loc[a3.unit_ids, 'bombcell_label'].values)
    ks = _trace_ks_units(list(a3.unit_ids), [{}, map2], merges,
                         origin={u: str(n) for u, n in names.items()})
    a3.sorting.set_property('ks_unit_ids', np.array([ks[u] for u in a3.unit_ids]))
    removed = (pd.concat(removed, ignore_index=True) if removed else
               pd.DataFrame(columns=['unit', 'sample', 'reason']))
    removed['ks_unit'] = [str(names.get(u, u)) for u in removed['unit']]
    keep = [u for u in a3.unit_ids if labels.loc[u, 'bombcell_label'] != 'noise']
    clean = select_units_aligned(a3, keep)
    n_out = int(a3.sorting.to_spike_vector().size)
    assert n_in - n_out == len(removed), (n_in, n_out, len(removed))
    done = merges[merges['merged'].astype(bool)]
    report.update({'pairs_dropped': int(pairs_post['dropped'].sum()) if len(pairs_post) else 0,
                   'n_units_out': len(a3.unit_ids), 'n_units_clean': len(keep),
                   'n_spikes_out': n_out,
                   # plain ints, so the report can be written as json
                   'merged_units': {int(map2.get(r['new_unit_id'], r['new_unit_id'])): {
                       'members': [int(x) for x in r['members']],
                       'reference': int(next(x for x in r['members'] if x == r['reference'])),
                       'ks_unit_ids': ks[map2.get(r['new_unit_id'], r['new_unit_id'])],
                       'label': labels.loc[map2.get(r['new_unit_id'], r['new_unit_id']), 'bombcell_label']}
                       for _, r in done.iterrows() if r['new_unit_id'] in merged_ids}})
    return clean, merges, removed, labels.loc[list(a3.unit_ids)], report


def clean_and_curate(analyzer, stim_intervals=None, method='template', w_ms=W_MS,
                     r_ms=R_MS, merge_presets=MERGE_PRESETS, audit_folder=None,
                     bombcell_thresholds=None, outlier_result=None, stim_outliers='drop',
                     plots=True, previous_labels=None, **job_kwargs):
    """The whole post-sort cleaning; returns (analyzer_clean, analyzer_curated, report).

    analyzer        : the analyzer of the sort (never modified; everything is in memory)
    stim_intervals  : (n, 2) seconds of stimulation, for the outlier scorer's core fit
    method          : 'template' resolves close pairs by fit, 'order' by keep-first --
                      the control arm, identical otherwise
    outlier_result  : score_outliers(analyzer, ...) if already computed
    stim_outliers   : 'drop' (outliers during stimulation removed too) or 'keep'
    previous_labels : {Kilosort unit: label from an earlier curation}, for the
                      before/after columns of unit_labels.csv
    plots           : with audit_folder, write a figure per unit of its removed spikes
                      (removed_spike_plots/) and per bimodal split (bimodal_split_plots/)
    analyzer_clean  : the non-noise units; analyzer_curated: every unit, with labels
    report          : dict of stage counts; the tables go to audit_folder if given
    """
    assert method in ('template', 'order')
    assert stim_outliers in ('drop', 'keep')
    fs = analyzer.sampling_frequency
    t_c = _t_c(fs, w_ms)
    report = {'method': method, 'w_ms': w_ms, 'r_ms': r_ms, 't_c_samples': t_c,
              'n_units_in': len(analyzer.unit_ids),
              'n_spikes_in': int(analyzer.sorting.to_spike_vector().size)}
    folder = Path(audit_folder) if audit_folder is not None else None
    if folder is not None:
        folder.mkdir(parents=True, exist_ok=True)

    # 1. outliers -- scored on the sort as it came, every unit
    print('spike cleaning 1/5: scoring outlier spikes')
    result = (outlier_result if outlier_result is not None else
              score_outliers(analyzer, stim_intervals, **job_kwargs))
    # 0. units that are two separate clusters become two units, re-scored as such; from
    # here on `analyzer` is the split one, and its new units are named 'parent/k'
    input_analyzer = analyzer
    analyzer, result, split_table, origin, split_levels = split_bimodal_units(
        analyzer, result, stim_intervals, **job_kwargs)
    splits, outlier_audit, scores = result
    report['bimodal_splits'] = len(split_table)
    report['bimodal_split_units'] = split_table['unit'].tolist() if len(split_table) else []

    # 0b. noise units go now. Bombcell's noise test is on the waveform shape, which no
    # removal of outlier spikes or merge can turn into a neuron's; whatever real signal
    # a noise unit carried as a separate cluster has just been split off above.
    # Cleaning them further would only spend time -- and merge candidates -- on units
    # that are thrown out at the end anyway.
    if split_levels and input_analyzer.has_extension('template_metrics'):
        # the split's partial recompute leaves NaNs in the new halves' template metrics,
        # and the noise test fails a unit on any NaN; the split analyzer is in memory
        analyzer.compute('template_metrics',
                         **copy.deepcopy(input_analyzer.get_extension('template_metrics').params),
                         **job_kwargs)
    pre_noise = analyzer
    noise_table = _noise_units(analyzer, bombcell_thresholds, origin)
    noise_ids = noise_table['unit_id'].tolist() if len(noise_table) else []
    report['noise_units_dropped'] = len(noise_ids)
    counts_in = analyzer.sorting.count_num_spikes_per_unit()
    report['noise_unit_spikes_dropped'] = int(sum(counts_in[u] for u in noise_ids))
    if noise_ids:
        analyzer = select_units_aligned(analyzer, [u for u in analyzer.unit_ids if u not in set(noise_ids)])
        scores = {u: s for u, s in scores.items() if u not in set(noise_ids)}
        splits = {u: s for u, s in splits.items() if u not in set(noise_ids)}
        outlier_audit = outlier_audit.drop(index=[u for u in noise_ids if u in outlier_audit.index])
    # restored after the merges: the similarity_correlograms preset recomputes
    # template_similarity with its own method (l1) and leaves it on the analyzer
    sim_params = (dict(analyzer.get_extension('template_similarity').params)
                  if analyzer.has_extension('template_similarity') else None)
    # and template_metrics are recomputed whole in step 5: the split's partial recompute
    # of the changed units leaves their multi-channel 'spread' NaN
    tm_params = (copy.deepcopy(analyzer.get_extension('template_metrics').params)
                 if analyzer.has_extension('template_metrics') else None)
    loc_params = (copy.deepcopy(analyzer.get_extension('unit_locations').params)
                  if analyzer.has_extension('unit_locations') else None)
    pre_drops = {u: np.asarray(sp[1], dtype=np.int64) for u, sp in splits.items()}
    # NaN-feature spikes go from every unit, accepted split or not
    for u, sc in scores.items():
        nan_idx = np.flatnonzero(sc['nan'])
        if nan_idx.size:
            pre_drops[u] = np.union1d(pre_drops.get(u, []), nan_idx).astype(np.int64)
    report['nan_feature_spikes_dropped'] = int(sum(int(np.sum(sc['nan'])) for sc in scores.values()))
    # Spikes fired during stimulation are scored against the unit's spontaneous core
    # like any other; the scorer leaves them out of its own outlier set so that they can
    # neither define the core nor sway the split decision. Here, stim_outliers='drop'
    # removes those beyond the same cut -- light artifacts and foreign spikes -- in
    # every unit whose core can be trusted: fitted (d2 exists) and not rejected as a
    # noise core, a second population or too small. 'keep' leaves them all. The cost
    # to watch: an evoked spike of this unit that overlaps a co-activated neighbour
    # also scores far out, so the audit counts the evoked spikes each unit loses.
    untrusted = ('REVIEW', 'outlier share', 'too few spikes')
    stim_dropped = {}
    if stim_outliers == 'drop':
        reasons = outlier_audit['reason'].astype(str)
        for u, sc in scores.items():
            if sc['d2'] is None or reasons.get(u, '').startswith(untrusted):
                continue
            idx = np.flatnonzero((sc['d2'] > sc['cut']) & sc['stim'] & ~sc['nan'])
            if idx.size:
                stim_dropped[u] = int(idx.size)
                pre_drops[u] = np.union1d(pre_drops.get(u, []), idx).astype(np.int64)
    outlier_audit = outlier_audit.assign(
        n_stim_spikes=[int(np.sum(scores[u]['stim'])) if u in scores else 0 for u in outlier_audit.index],
        stim_outliers_dropped=[stim_dropped.get(u, 0) for u in outlier_audit.index])
    report['stim_outliers'] = stim_outliers
    report['stim_outliers_dropped'] = int(sum(stim_dropped.values()))
    report['outliers_dropped'] = int(sum(len(v) for v in pre_drops.values()))
    report['units_with_outliers_dropped'] = len(pre_drops)
    reasons = outlier_audit['reason'].astype(str)
    # left whole, and worth a look: a second population, or a core that is not the unit
    report['units_over_outlier_share'] = outlier_audit.index[reasons.str.startswith('outlier share')].tolist()
    report['units_for_review'] = outlier_audit.index[reasons.str.startswith('REVIEW')].tolist()

    # 2. refractory pairs among what is left, pre-merge, then one split for both
    print(f'spike cleaning 2/5: resolving pairs <= {w_ms:g} ms ({method})')
    pair_drops, pairs_pre = _pair_drops(analyzer, scores, t_c, method, exclude=pre_drops)
    report['pairs_dropped_pre_merge'] = int(pairs_pre['dropped'].sum())
    all_drops = {u: np.union1d(pre_drops.get(u, []), pair_drops.get(u, [])).astype(np.int64)
                 for u in set(pre_drops) | set(pair_drops)}
    # every removed spike, by (input unit, sample) and why: the audit and the plots
    in_samples = _unit_samples(analyzer)
    removed = []
    for u in all_drops:
        nan = np.flatnonzero(scores[u]['nan']) if u in scores else np.array([], int)
        groups = (('nan feature', nan),
                  ('outlier', np.setdiff1d(splits[u][1], nan) if u in splits else []),
                  ('stim outlier', np.setdiff1d(pre_drops.get(u, []),
                                                np.union1d(nan, splits[u][1] if u in splits else []))),
                  ('close pair (pre-merge)', pair_drops.get(u, [])))
        for why, idx in groups:
            idx = np.asarray(idx, dtype=np.int64)
            if idx.size:
                removed.append(pd.DataFrame({'unit': u, 'sample': in_samples[u][idx], 'reason': why}))
    a1, map1 = drop_spikes(analyzer, all_drops, **job_kwargs)
    if a1 is analyzer:
        # nothing dropped; the merges below modify their input's extensions in memory
        a1 = analyzer.save_as(format='memory')

    # 3. merges
    print('spike cleaning 3/5: merging')
    a2, merges = auto_merge(a1, merge_presets, w_ms, r_ms, **job_kwargs)
    merged_ids = merges.loc[merges['merged'], 'new_unit_id'].tolist() if len(merges) else []
    # a later preset can merge a unit made by an earlier one; only the survivors exist
    merged_ids = [u for u in merged_ids if u in list(a2.unit_ids)]
    report['merges'] = int(merges['merged'].sum()) if len(merges) else 0
    if merged_ids:
        _thin_merged_random_spikes(a1, a2, merged_ids, merges)

    # 4. refractory pairs again, merged units only, re-scored on the merged analyzer
    print(f'spike cleaning 4/5: resolving pairs in {len(merged_ids)} merged units')
    a3, map2, pairs_post, post_removed = _resolve_merged_pairs(
        a1, a2, merges, merged_ids, map1, stim_intervals, method, t_c, **job_kwargs)
    removed.extend(post_removed)
    report['pairs_dropped_post_merge'] = int(pairs_post['dropped'].sum()) if len(pairs_post) else 0

    # 5. metrics for every unit with the censored refractory params, then bombcell
    print('spike cleaning 5/5: quality metrics and bombcell')
    labels = _metrics_and_labels(a3, sim_params, loc_params, tm_params, w_ms, r_ms,
                                 bombcell_thresholds, **job_kwargs)
    a3.sorting.set_property('bombcell_label', labels.loc[a3.unit_ids, 'bombcell_label'].values)
    ks = _trace_ks_units(list(a3.unit_ids), [map1, map2], merges, origin)
    # a string property set after the last merge; merges blank disagreeing strings
    a3.sorting.set_property('ks_unit_ids', np.array([ks[u] for u in a3.unit_ids]))
    keep = [u for u in a3.unit_ids if labels.loc[u, 'bombcell_label'] != 'noise']
    clean = select_units_aligned(a3, keep)
    removed = (pd.concat(removed, ignore_index=True) if removed else
               pd.DataFrame(columns=['unit', 'sample', 'reason']))
    # the Kilosort unit (or split half) each removed spike came from, by name
    removed['ks_unit'] = [origin.get(u, str(u)) for u in removed['unit']]
    report.update({'n_units_out': len(a3.unit_ids), 'n_units_clean': len(keep),
                   'n_spikes_out': int(a3.sorting.to_spike_vector().size),
                   'labels': labels['bombcell_label'].value_counts().to_dict(),
                   'rp_demoted': int(labels['rp_demoted'].sum()),
                   'removed_by_reason': removed['reason'].value_counts().to_dict()})
    # every spike accounted for: what went in is what came out, plus what was removed
    # spike by spike, plus the noise units dropped whole
    assert (report['n_spikes_in'] - report['n_spikes_out']
            == len(removed) + report['noise_unit_spikes_dropped']), \
        (report['n_spikes_in'], report['n_spikes_out'], len(removed), report['noise_unit_spikes_dropped'])

    if folder is not None:
        outlier_audit.to_csv(folder / 'outlier_audit.csv')
        pairs_pre.to_csv(folder / 'pairs_pre_merge.csv')
        pairs_post.to_csv(folder / 'pairs_post_merge.csv')
        merges.to_csv(folder / 'merges.csv', index=False)
        removed.to_csv(folder / 'removed_spikes.csv', index=False)
        pd.DataFrame({'unit_id': list(a3.unit_ids), 'ks_unit_ids': [ks[u] for u in a3.unit_ids],
                      'bombcell_label': labels.loc[a3.unit_ids, 'bombcell_label'].values,
                      'rp_violations': labels.loc[a3.unit_ids, 'rp_violations'].values,
                      'rp_contamination': labels.loc[a3.unit_ids, 'rp_contamination'].values,
                      'rp_contamination_lo': labels.loc[a3.unit_ids, 'rp_contamination_lo'].values,
                      'rp_counted': labels.loc[a3.unit_ids, 'rp_counted'].values,
                      'rp_demoted': labels.loc[a3.unit_ids, 'rp_demoted'].values}
                     ).to_csv(folder / 'unit_ids.csv', index=False)
        (folder / 'report.json').write_text(json.dumps(report, indent=2, default=str))
        if len(split_table):
            split_table.to_csv(folder / 'bimodal_splits.csv', index=False)
        if len(noise_table):
            noise_table.to_csv(folder / 'noise_units_dropped.csv', index=False)
        kl = input_analyzer.sorting.get_property('KSLabel')
        label_table(a3, labels, ks, bombcell_thresholds, previous_labels,
                    dict(zip(input_analyzer.unit_ids, kl)) if kl is not None else None,
                    noise_table).to_csv(folder / 'unit_labels.csv', index=False)
        if plots:
            from postprocessing import cleaning_plots as cplots
            # each pre-merge unit's final label, found through its name in ks_unit_ids
            final_of = {name: labels.loc[u, 'bombcell_label']
                        for u in a3.unit_ids for name in str(ks[u]).split('+')}
            labels_by_unit = {u: final_of.get(origin.get(u, str(u)), '')
                              for u in analyzer.unit_ids}
            cplots.plot_removed_spikes(analyzer, removed, folder / 'removed_spike_plots',
                                       labels=labels_by_unit, names=origin, **job_kwargs)
            if noise_ids:
                cplots.plot_noise_units(pre_noise, noise_table, folder / 'noise_unit_plots',
                                        **job_kwargs)
            for level_analyzer, labels_by_parent in split_levels:
                cplots.plot_bimodal_splits(level_analyzer, labels_by_parent,
                                           folder / 'bimodal_split_plots',
                                           names={**{u: str(u) for u in input_analyzer.unit_ids},
                                                  **origin}, table=split_table)
    return clean, a3, report
