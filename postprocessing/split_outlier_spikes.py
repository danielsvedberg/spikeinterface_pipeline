"""Split the outlying spikes out of each unit, so a unit is scored on its own spikes.

A cluster that is one clear neuron plus a scatter of artifacts, or plus spikes of a
neighbour that fired at the same moment, fails bombcell as MUA on the strength of the
scatter: refractory violations, amplitude spread, SNR. Dropping the unit loses the
neuron; keeping it keeps the scatter. Splitting the scatter off into its own unit lets
each half be judged on what it is.

Every spike of a unit is described by
    - its projection on the analyzer's own PCA models, on the unit's N_CHANNELS
      strongest channels (the analyzer only holds projections for its random_spikes
      subsample, so the unit's spikes are extracted afresh -- on those few channels
      only -- and projected through the fitted models, which reproduces what the
      analyzer would have computed for them),
    - its size: the least-squares scale of the template in its own raw snippet,
    - its estimated location (already per spike),
and scored by its robust Mahalanobis distance from the unit's core. The core is a
Minimum Covariance Determinant fit, which ignores up to half the spikes while fitting,
so the outliers cannot drag the centre toward themselves the way a plain mean and
covariance would.

stim_intervals, when given, restricts the fit to spikes outside them. On a tagging
session that is the point: light-evoked spikes are where neighbours co-fire and merge
into a unit, so they must not define what the unit looks like. They are still scored
against the spontaneous core -- which is also the waveform test tagging needs: an
evoked spike that lands in the spontaneous core is the same neuron.

A split is only proposed when it plausibly helps (see accept_split); every unit gets a
row in the audit table either way, and a diagnostic figure when plot_folder is given.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import MinCovDet

from utils.mpl_backend import select_backend
select_backend()
import matplotlib.pyplot as plt

# retries the intermittent Windows EINVAL on savefig; see its docstring
from preprocessing.preprocess_qc import _save

# PCA projections are taken on this many of the unit's strongest channels
N_CHANNELS = 4
# spikes whose log distance lies this many robust SDs above the spontaneous median are
# outliers. Not a chi2 quantile: spike clouds are heavy-tailed, and on DS45 the chi2
# 0.999 cut took 5-33% of a unit's spikes, nearly all of them the unit's own spikes
# with a neighbour's overlapping in the window. The cut is set against each unit's own
# spread instead, so "outlier" means far out for that unit.
OUTLIER_Z = 6.0
# random stratified removals drawn for the violation null in accept_split
N_NULL = 300
# the core is fitted on at most this many spikes, evenly spread over the session
MAX_FIT_SPIKES = 5000
# a unit needs this many spikes to fit a core in ~23 dimensions at all
MIN_SPIKES = 300
# refractory period for the violation counts in the audit
REFRACTORY_MS = 1.5
# a split needs at least this many outliers, so one stray spike does not become a unit.
# There is deliberately no minimum *share*: splitting a few obviously foreign spikes
# off a large unit costs at most those spikes, which go to a sub-unit labelled noise,
# and removes exactly what should not be in it. A 0.5% floor here once kept unit 19 on
# DS45 (26 of 5765, 0.45%, most carrying a neighbour's -100 to -150 uV spike) whole.
MIN_OUTLIERS = 5
# above this share the "outliers" are more likely a second population or drift than
# contamination, which is a merge/drift question and not this one
MAX_OUTLIER_FRAC = 0.30
# the outliers are a different shape when at least this share of them, one by one,
# match the template worse than the kept spikes' 5th percentile. Per spike rather than
# the group median, because the outliers of one unit are often a mixture -- on unit 19
# a handful of foreign-shaped overlaps (r ~ 0) among larger copies of the unit's own
# spike (r ~ 0.7), whose median says "same shape". By chance alone 5% would fall below
# that percentile; a quarter is five times that.
SHAPE_OUTLIER_SHARE = 0.25
# the spikes a split keeps must still be a spike: the peak of their mean waveform on the
# peak channel at least this many times that channel's noise. The core is fitted to the
# majority of a unit's spikes, and in a unit where noise crossings outnumber the real
# spikes the majority is the noise -- on DS45 unit 23 the kept 74% sat at template
# scale ~0 while the 26% flagged as outliers, at ~4x, were the spikes that make the
# template. Splitting that would keep the noise and throw the neuron away. Judged on the
# kept spikes' own mean against noise, not their scale relative to the whole-unit
# template: a large contaminant inflates that template, so a correctly cleaned unit can
# sit well below scale 1 (1 / (1 - f + f k) for a share f of k-times-larger spikes).
KEPT_SNR_MIN = 1.5
# spikes per waveform extraction batch; the buffer is batch x samples x N_CHANNELS
BATCH_SPIKES = 1_000_000


def _unit_channels(analyzer, unit_ids, n_channels=N_CHANNELS):
    """{unit: channel indices}, the unit's n strongest template channels, ascending."""
    tpl = analyzer.get_extension('templates')
    out = {}
    for u in unit_ids:
        ptp = np.ptp(tpl.get_unit_template(u), axis=0)
        out[u] = np.sort(np.argsort(ptp)[::-1][:n_channels])
    return out


def _template_match(w, tpl):
    """Per-spike shape and size of each raw snippet against the template.

    w: (n_spikes, n_samples, n_ch) raw; tpl: (n_samples, n_ch) on the same channels.
    Returns (r, scale):
      r      Pearson r over all channels concatenated, so a spike that matches on the
             peak channel but carries a neighbour's spike on another channel scores
             low -- the case a peak-channel or mean-waveform comparison cannot see.
      scale  the least-squares amplitude of the template in the snippet (1 = template
             sized). r is blind to size, and larger spikes stand further out of the
             noise and so correlate *better*: on DS45 a second neuron five times the
             unit's amplitude, merged into unit 27, scored r 0.81 against the unit's
             own 0.53. Size has to be judged separately.
    """
    x = w.reshape(w.shape[0], -1).astype(np.float64)
    x -= x.mean(axis=1, keepdims=True)
    t = tpl.reshape(-1).astype(np.float64)
    t -= t.mean()
    tt = t @ t
    xt = x @ t
    return xt / (np.linalg.norm(x, axis=1) * np.sqrt(tt) + 1e-12), xt / (tt + 1e-12)


def _pc_features(analyzer, unit_ids, chans, batch_spikes=BATCH_SPIKES, **job_kwargs):
    """PCA projections and template match of every spike, from one raw extraction.

    Returns ({unit: (n_spikes, n_channels * n_components)},
             {unit: (r, scale) per-spike template match, see _template_match}).

    extract_waveforms_to_single_buffer with a per-unit channel mask returns each unit's
    spikes on exactly its mask channels, in ascending channel order -- the order chans
    holds them in, so column j of the buffer is channel chans[u][j].
    """
    tpl_ext = analyzer.get_extension('templates')
    from spikeinterface.core.waveform_tools import extract_waveforms_to_single_buffer
    fs = analyzer.sampling_frequency
    wf = analyzer.get_extension('waveforms').params
    nbefore = int(round(wf['ms_before'] * fs / 1000.0))
    nafter = int(round(wf['ms_after'] * fs / 1000.0))
    models = analyzer.get_extension('principal_components').get_pca_model()
    all_ids = list(analyzer.unit_ids)
    spikes = analyzer.sorting.to_spike_vector()
    uidx = {u: all_ids.index(u) for u in unit_ids}

    # batches of whole units, so each unit's spikes come out of one buffer
    batches, cur, n = [], [], 0
    counts = {u: int(np.sum(spikes['unit_index'] == uidx[u])) for u in unit_ids}
    for u in unit_ids:
        if cur and n + counts[u] > batch_spikes:
            batches.append(cur)
            cur, n = [], 0
        cur.append(u)
        n += counts[u]
    if cur:
        batches.append(cur)

    # the templates window must be the snippets' window for the match to line up
    t_nb = tpl_ext.nbefore
    tpl_slice = slice(t_nb - nbefore, t_nb + nafter)
    feats, match = {}, {}
    for b, batch in enumerate(batches):
        sel = np.isin(spikes['unit_index'], [uidx[u] for u in batch])
        mask = np.zeros((len(all_ids), analyzer.get_num_channels()), bool)
        for u in batch:
            mask[uidx[u], chans[u]] = True
        print(f'  split: extracting {int(sel.sum())} spikes of {len(batch)} units '
              f'(batch {b + 1}/{len(batches)})')
        buf = extract_waveforms_to_single_buffer(
            analyzer.recording, spikes[sel], all_ids, nbefore, nafter,
            mode='shared_memory', return_in_uV=True, sparsity_mask=mask, copy=True,
            job_name='split features', **job_kwargs)
        sub = spikes[sel]
        for u in batch:
            w = buf[sub['unit_index'] == uidx[u]][:, :, :len(chans[u])]
            feats[u] = np.concatenate(
                [models[c].transform(w[:, :, j]) for j, c in enumerate(chans[u])], axis=1)
            match[u] = _template_match(w, tpl_ext.get_unit_template(u)[tpl_slice][:, chans[u]])
        del buf
    return feats, match


# a unit is split in two when a two-cluster fit of its features finds clusters that are
# clearly apart (Ashman's D along their discriminant, > 2 is bimodal), with a clear
# valley between them in that projection, each holding at least SPLIT_MIN_SPIKES and
# SPLIT_MIN_SHARE of the unit, and both active in the same stretches of the session --
# two clusters that take turns are one neuron drifting, which is flagged, not split.
# On DS45 unit 146 is 6522 spikes at 2.6x template scale (r 0.93) plus 7670 noise
# crossings at scale ~0: D 9.8, and nothing in the valley.
SPLIT_MIN_D = 3.0
SPLIT_MAX_VALLEY = 0.15
SPLIT_MIN_SPIKES, SPLIT_MIN_SHARE = 100, 0.03
SPLIT_MIN_OVERLAP = 0.3
SPLIT_BIN_S = 60.0
SPLIT_MAX_FIT = 20000
# two pieces of one neuron cannot fire within its refractory period of each other: of the
# coincidences within +-REFRACTORY_MS that two independent trains would have (from their
# rates in each SPLIT_BIN_S bin, so a drift handover is judged where both are active), they
# show at most MUTUAL_MAX_RATIO, out of at least MUTUAL_MIN_EXPECTED. Neighbouring
# different neurons sit near 1 (5th percentile 0.74 over 24k pairs within 40 um, all
# sessions); the sorter's missed overlaps leave close high-rate pairs at ~0.25-0.3.
# DS2_20260505 KS 605, drifting 12 um in two steps, was split at both: its pieces show 3
# of 46 expected.
MUTUAL_MAX_RATIO, MUTUAL_MIN_EXPECTED = 0.1, 20.0


def coincidences(ta, tb, r_s=REFRACTORY_MS / 1000.0, bin_s=SPLIT_BIN_S):
    """(observed, expected) pairs of a spike of each train within |lag| <= r_s.

    ta, tb : sorted spike times (s). Expected: what independent trains with the same rate
    in every bin_s bin would have, sum over bins of n_a * n_b * 2 r_s / bin_s.
    """
    ta, tb = np.asarray(ta, float), np.asarray(tb, float)
    if not ta.size or not tb.size:
        return 0, 0.0
    obs = int((np.searchsorted(tb, ta + r_s, 'right') - np.searchsorted(tb, ta - r_s, 'left')).sum())
    edges = np.arange(0.0, max(ta[-1], tb[-1]) + bin_s, bin_s)
    exp = float((np.histogram(ta, edges)[0] * np.histogram(tb, edges)[0]).sum() * 2 * r_s / bin_s)
    return obs, exp


def mutually_refractory(obs, exp):
    """True when the deficit of coincidences says one neuron (see MUTUAL_MAX_RATIO)."""
    return exp >= MUTUAL_MIN_EXPECTED and obs <= MUTUAL_MAX_RATIO * exp


def _bimodal_split(X, t, bad, seed=0):
    """(labels 0/1 or None, stats) for a two-cluster split of one unit's features.

    X is the scorer's feature matrix (PCs on the unit's channels, template scale,
    location), robust-scaled here so no feature dominates by its units. The two-cluster
    Gaussian mixture proposes the partition; the decision rests on the separation along
    its discriminant and the emptiness of the valley between the clusters, not on BIC,
    which prefers two components for almost any real (non-Gaussian) cluster.
    """
    from sklearn.mixture import GaussianMixture
    good = ~bad
    stats = {'split_D': np.nan, 'split_valley': np.nan, 'split_sizes': '', 'split_overlap': np.nan,
             'split_coinc_obs': np.nan, 'split_coinc_exp': np.nan, 'split_reason': ''}
    n = int(good.sum())
    min_n = max(SPLIT_MIN_SPIKES, SPLIT_MIN_SHARE * n)
    if n < 2 * min_n:
        stats['split_reason'] = 'too few spikes'
        return None, stats
    Xg = X[good]
    med = np.median(Xg, axis=0)
    iqr = np.percentile(Xg, 75, axis=0) - np.percentile(Xg, 25, axis=0)
    Z = (Xg - med) / np.where(iqr > 0, iqr, 1.0)
    rng = np.random.default_rng(seed)
    fit = Z if n <= SPLIT_MAX_FIT else Z[rng.choice(n, SPLIT_MAX_FIT, replace=False)]
    gm = GaussianMixture(2, covariance_type='full', n_init=3, random_state=seed).fit(fit)
    lab = gm.predict(Z)
    sizes = np.bincount(lab, minlength=2)
    stats['split_sizes'] = f'{sizes[0]}/{sizes[1]}'
    if sizes.min() < min_n:
        stats['split_reason'] = 'one cluster too small'
        return None, stats
    w = np.linalg.solve(gm.covariances_.mean(axis=0), gm.means_[1] - gm.means_[0])
    p = Z @ w
    p0, p1 = p[lab == 0], p[lab == 1]
    D = abs(p0.mean() - p1.mean()) / np.sqrt((p0.var() + p1.var()) / 2)
    stats['split_D'] = float(D)
    # the valley: the lowest smoothed density between the two cluster medians, against
    # the lower of the two peaks
    lo, hi = sorted((np.median(p0), np.median(p1)))
    hist, edges = np.histogram(p, bins=120)
    dens = np.convolve(hist, np.ones(3) / 3, mode='same')
    centres = (edges[:-1] + edges[1:]) / 2
    between = (centres > lo) & (centres < hi)
    peak = min(dens[np.argmin(abs(centres - np.median(p0)))], dens[np.argmin(abs(centres - np.median(p1)))])
    valley = dens[between].min() / max(peak, 1) if between.any() else 1.0
    stats['split_valley'] = float(valley)
    # temporal overlap: of the bins where the smaller cluster is active, the share where
    # the larger one is active too
    tg = t[good]
    bins = np.arange(0, tg.max() + SPLIT_BIN_S, SPLIT_BIN_S)
    act = [np.histogram(tg[lab == k], bins)[0] >= 3 for k in (0, 1)]
    small = int(np.argmin(sizes))
    overlap = (act[0] & act[1]).sum() / max(act[small].sum(), 1)
    stats['split_overlap'] = float(overlap)
    if D < SPLIT_MIN_D:
        stats['split_reason'] = f'not separated (D {D:.1f} < {SPLIT_MIN_D:g})'
        return None, stats
    if valley > SPLIT_MAX_VALLEY:
        stats['split_reason'] = f'no clear valley ({valley:.2f} > {SPLIT_MAX_VALLEY:g})'
        return None, stats
    if overlap < SPLIT_MIN_OVERLAP:
        stats['split_reason'] = f'REVIEW: clusters take turns in time (overlap {overlap:.2f}); drift?'
        return None, stats
    # clusters that never fire within a refractory period of each other are one neuron,
    # its waveform changed by drift (or another slow change) -- not two neurons
    obs, exp = coincidences(tg[lab == 0], tg[lab == 1])
    stats['split_coinc_obs'], stats['split_coinc_exp'] = obs, exp
    if mutually_refractory(obs, exp):
        stats['split_reason'] = (f'one neuron: clusters mutually refractory ({obs} of {exp:.0f} '
                                 f'expected coincidences within {REFRACTORY_MS:g} ms)')
        return None, stats
    stats['split_reason'] = 'split'
    labels = np.zeros(X.shape[0], dtype=np.int8)
    labels[good] = lab
    # a spike with an undefined feature goes with the larger cluster; it is dropped later
    labels[bad] = int(np.argmax(sizes))
    return labels, stats


def _mahalanobis(X, mu, cov):
    """Squared Mahalanobis distances, through a pseudo-inverse as sklearn does, so a
    near-singular scatter (a tied feature) degrades gracefully instead of raising."""
    from scipy.linalg import pinvh
    d = X - mu
    return np.einsum('ij,jk,ik->i', d, pinvh(cov), d)


def _c_steps(X, support, h, max_iter=50):
    """Concentration steps from a support: (log det, mean, cov) of the h-subset they reach.

    If a later subset turns singular, the last finite step is returned rather than
    nothing: it is still a valid, if not converged, candidate.
    """
    best = (np.inf, None, None)
    for _ in range(max_iter):
        mu = X[support].mean(axis=0)
        cov = np.cov(X[support], rowvar=False)
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            break
        best = (logdet, mu, cov)
        new = np.sort(np.argpartition(_mahalanobis(X, mu, cov), h - 1)[:h])
        if np.array_equal(new, support):
            break
        support = new
    return best


def _robust_fit(X, seed):
    """Minimum Covariance Determinant location and scatter, from several starts.

    sklearn's FastMCD alone can settle on a local optimum whose support mixes two
    populations -- a unit with a quarter of its spikes from a tighter, larger neighbour
    of the same shape and place is the case that matters here: both populations then
    get the same distances and the cut trims the tails of both. So besides its own
    solution, C-steps are also started from MCD fits of the PCA block and of the
    scaling/location block separately (in one of which the two populations separate),
    and the start reaching the lowest determinant wins. That core then gets FastMCD's
    consistency correction and one reweighting step, as MinCovDet does.
    """
    from scipy.stats import chi2
    n, p = X.shape
    h = int(np.ceil(0.5 * (n + p + 1)))
    npc = p - 3
    sk = MinCovDet(random_state=seed).fit(X)
    starts = [sk.raw_support_]
    for block in (slice(0, npc), slice(npc, p)):
        starts.append(MinCovDet(random_state=seed).fit(X[:, block]).raw_support_)
    best = (np.inf, None, None)
    for s in starts:
        s = np.flatnonzero(s)
        mu0, cov0 = X[s].mean(axis=0), np.cov(X[s], rowvar=False)
        if np.linalg.slogdet(cov0)[0] <= 0:
            continue
        s = np.sort(np.argpartition(_mahalanobis(X, mu0, cov0), h - 1)[:h])
        fit = _c_steps(X, s, h)
        if fit[0] < best[0]:
            best = fit
    if best[1] is None:
        # every start singular: sklearn's own fit, which works through pinvh
        return sk.location_, sk.covariance_
    _, mu, cov = best
    d2 = _mahalanobis(X, mu, cov)
    cov = cov * np.median(d2) / chi2.ppf(0.5, p)
    w = _mahalanobis(X, mu, cov) < chi2.ppf(0.975, p)
    mu, cov = X[w].mean(axis=0), np.cov(X[w], rowvar=False)
    d2 = _mahalanobis(X, mu, cov)
    return mu, cov * np.median(d2) / chi2.ppf(0.5, p)


def _violations(times_s, refractory_ms=REFRACTORY_MS):
    return int(np.sum(np.diff(np.sort(times_s)) < refractory_ms / 1000.0))


def _in_intervals(times, intervals):
    if intervals is None or not len(intervals):
        return np.zeros(times.size, bool)
    iv = np.asarray(intervals, float)
    iv = iv[np.argsort(iv[:, 0])]
    k = np.searchsorted(iv[:, 0], times, side='right') - 1
    return (k >= 0) & (times <= iv[np.clip(k, 0, None), 1])


def _violation_null(t, stim, out, rng, n=N_NULL):
    """Violations left after removing as many random spikes as the split does.

    Stratified: the same number from inside and outside the stimulation as the
    outliers hold. Any removal cuts violations, and removing spikes from the pulses,
    where the rate and so the chance coincidences are highest, cuts them fastest --
    so "the kept spikes violate less" only means something against this.
    """
    n_s, n_n = int((out & stim).sum()), int((out & ~stim).sum())
    idx_s, idx_n = np.flatnonzero(stim), np.flatnonzero(~stim)
    null = np.empty(n)
    for i in range(n):
        drop = np.zeros(t.size, bool)
        drop[rng.choice(idx_s, n_s, replace=False)] = True
        drop[rng.choice(idx_n, n_n, replace=False)] = True
        null[i] = _violations(t[~drop])
    return null


def accept_split(row):
    """Whether the proposed split is worth making, and if not, why not.

    Beyond the outlier-share bounds, a split has to buy something measurable: either
    the kept spikes have fewer refractory violations than random removal of as many
    spikes would leave (below the null's 2.5th percentile), or the outliers are
    clearly a different waveform -- a different shape (at least SHAPE_OUTLIER_SHARE of
    them, spike by spike, match the template worse than the kept spikes' 5th
    percentile) or a different size (their median template scale falls outside the
    kept spikes' 1st to 99th percentile). Both bars are the unit's own spike-to-spike variability, so a
    small unit in noise, whose spikes all match loosely, is not held to a number a
    large unit would pass. A broad but clean unit meets neither, and cutting its tail
    would only lose spikes.
    """
    if row['n_spikes'] < MIN_SPIKES:
        return False, 'too few spikes'
    if row['n_outliers'] < MIN_OUTLIERS:
        return False, f"only {row['n_outliers']} outliers (< {MIN_OUTLIERS})"
    if not row['kept_mean_snr'] >= KEPT_SNR_MIN:
        return False, (f"REVIEW: the kept spikes' mean is not a spike (peak "
                       f"{row['kept_mean_snr']:.2f} x noise < {KEPT_SNR_MIN:g}); the core is "
                       'another population, probably noise')
    if row['outlier_frac'] > MAX_OUTLIER_FRAC:
        return False, (f"outlier share {100 * row['outlier_frac']:.2f}% above "
                       f"{100 * MAX_OUTLIER_FRAC:g}%: drift or a second population, not "
                       'contamination')
    fewer_viol = row['viol_kept'] < row['viol_null_p2.5']
    shape = row['removed_below_kept_p5_share'] >= SHAPE_OUTLIER_SHARE
    size = not row['kept_scale_p1'] <= row['removed_scale_median'] <= row['kept_scale_p99']
    why = [w for w, hit in (('fewer violations than random removal', fewer_viol),
                            ('outliers are a different shape', shape),
                            ('outliers are a different size', size)) if hit]
    if why:
        return True, ' + '.join(why)
    return False, 'no fewer violations than random removal, outliers match the waveform'


def find_outlier_spikes(analyzer, unit_ids=None, stim_intervals=None,
                        outlier_z=OUTLIER_Z, n_channels=N_CHANNELS,
                        plot_folder=None, seed=0, return_scores=False, **job_kwargs):
    """Score every spike of every unit; return (splits, audit[, scores]).

    splits : {unit_id: [kept_indices, outlier_indices]} for the accepted units, indices
             into the unit's own spike train -- the form analyzer.split_units takes.
    audit  : one row per unit, accepted or not, with the reason.
    scores : with return_scores, {unit_id: {'t', 'd2', 'cut', 'out', 'stim',
             'channels', 'match', 'scale', 'nan'}} for every unit, accepted or not --
             what plot_kept_vs_removed draws from. d2 is None for units too small to
             fit a core (< MIN_SPIKES); 'nan' marks spikes with an undefined feature.
    """
    unit_ids = list(analyzer.unit_ids if unit_ids is None else unit_ids)
    fs = analyzer.sampling_frequency
    spikes = analyzer.sorting.to_spike_vector()
    all_ids = list(analyzer.unit_ids)
    loc = analyzer.get_extension('spike_locations').get_data()
    noise = analyzer.get_extension('noise_levels').get_data()
    chans = _unit_channels(analyzer, unit_ids, n_channels)
    pcs, match = _pc_features(analyzer, unit_ids, chans, **job_kwargs)
    rng = np.random.default_rng(seed)
    wf = analyzer.get_extension('waveforms').params
    nb = int(round(wf['ms_before'] * fs / 1000.0))
    t_ms = (np.arange(nb + int(round(wf['ms_after'] * fs / 1000.0))) - nb) / fs * 1000.0

    splits, rows, scores = {}, [], {}
    for u in unit_ids:
        m = spikes['unit_index'] == all_ids.index(u)
        t = spikes['sample_index'][m] / fs
        # the size feature is the spike's own least-squares template scale from its raw
        # snippet, not amplitude_scalings: that extension fits colliding spikes of
        # different units jointly with a non-negative regression, and clips the loser to
        # exactly 0 -- on DS45 that is 557k spikes, over half of some units, and a tied
        # half makes the robust covariance singular and becomes its core
        X = np.column_stack([pcs[u], match[u][1], loc['x'][m], loc['y'][m]])
        stim = _in_intervals(t, stim_intervals)
        # a spike with an undefined feature (spike_locations can return NaN) has no
        # distance to anything; it is an outlier by definition and is kept out of the
        # core fit, which sklearn refuses to run on NaN
        bad = ~np.isfinite(X).all(axis=1)
        row = {'unit_id': u, 'n_spikes': int(m.sum()), 'n_features': X.shape[1],
               'n_in_stim': int(stim.sum()), 'outlier_z': outlier_z,
               'n_nan_features': int(bad.sum())}
        # two clusters in one unit: decided first, since the outlier core of a mixture
        # describes neither population (see _bimodal_split)
        split_labels, split_stats = _bimodal_split(X, t, bad, seed)
        row.update(split_stats)
        if m.sum() < MIN_SPIKES:
            rows.append({**row, 'accepted': False, 'reason': 'too few spikes'})
            # no core can be fitted, but the template match still ranks the spikes,
            # which is what resolving refractory violations needs
            scores[u] = {'t': t, 'd2': None, 'cut': np.nan, 'out': bad, 'stim': stim,
                         'channels': chans[u], 'match': match[u][0], 'scale': match[u][1],
                         'nan': bad, 'split': split_labels}
            continue
        ref = np.flatnonzero(~stim & ~bad)
        if ref.size < MIN_SPIKES:
            ref = np.flatnonzero(~bad)
        if ref.size > MAX_FIT_SPIKES:
            ref = np.sort(rng.choice(ref, MAX_FIT_SPIKES, replace=False))
        mu, cov = _robust_fit(X[ref], seed)
        d2 = np.full(m.sum(), np.inf)
        d2[~bad] = _mahalanobis(X[~bad], mu, cov)
        # the cut in d2 units, from the robust spread of the reference spikes' log d2
        lref = np.log(d2[ref])
        med = np.median(lref)
        cut = float(np.exp(med + outlier_z * 1.4826 * np.median(np.abs(lref - med))))
        # every NaN-feature spike (d2 = inf), and no spike fired during stimulation: on
        # a tagging session the light drives neighbours to fire with the tagged neuron,
        # so its evoked spikes overlap theirs and score far from the spontaneous core
        # while being its own. Stimulation is a small share of the session; losing its
        # outliers costs little, losing the evoked spikes would cost the tagging.
        out = (d2 > cut) & ~stim | bad
        kept_t, out_t = t[~out], t[out]
        # the kept spikes the core accepts: the reference for what the unit looks like.
        # Evoked spikes over the cut are kept but are the co-fired overlaps; counting
        # them would widen the shape and size bars until real contamination passes
        core = ~out & ~((d2 > cut) & stim)

        # the peak channel's PCA-reconstructed means, for the diagnostic figure only;
        # whether the outliers are a different waveform is judged per spike, on raw
        # traces across all scoring channels (match, below)
        pk = int(np.argmax([np.ptp(analyzer.get_extension('templates').get_unit_template(u)[:, c])
                            for c in chans[u]]))
        cols = slice(pk * 5, pk * 5 + 5)
        model = analyzer.get_extension('principal_components').get_pca_model()[chans[u][pk]]
        wf_k = model.inverse_transform(X[core, cols].mean(0, keepdims=True))[0]
        wf_o = (model.inverse_transform(X[out, cols].mean(0, keepdims=True))[0]
                if out.any() else wf_k)
        n_v, n_vk = _violations(t), _violations(kept_t)
        r_u, s_u = match[u]
        pk_ch = chans[u][pk]
        row.update({
            'kept_mean_snr': float(np.abs(wf_k).max() / noise[pk_ch]),
            'stim_spikes_over_cut_kept': int(np.sum((d2 > cut) & stim & ~bad)),
            'outlier_frac': float(out.mean()), 'n_outliers': int(out.sum()),
            'outlier_in_stim_frac': float(stim[out].mean()) if out.any() else 0.0,
            'viol_before': n_v, 'viol_kept': n_vk, 'viol_outliers': _violations(out_t),
            'viol_rate_before': n_v / m.sum(), 'viol_rate_kept': n_vk / max((~out).sum(), 1),
            'viol_kept_outside_stim': _violations(t[~out & ~stim]),
            'kept_match_median': float(np.median(r_u[core])),
            'kept_match_p5': float(np.percentile(r_u[core], 5)),
            'removed_match_median': float(np.median(r_u[out])) if out.any() else np.nan,
            'removed_below_kept_p5_share': (float(np.mean(r_u[out] < np.percentile(r_u[core], 5)))
                                            if out.any() else 0.0),
            'kept_scale_median': float(np.median(s_u[core])),
            'kept_scale_p1': float(np.percentile(s_u[core], 1)),
            'kept_scale_p99': float(np.percentile(s_u[core], 99)),
            'removed_scale_median': float(np.median(s_u[out])) if out.any() else np.nan,
            'fit_on': 'spontaneous' if (~stim).sum() >= MIN_SPIKES else 'all',
        })
        null = _violation_null(t, stim, out, rng)
        row.update({'viol_null_mean': float(null.mean()),
                    'viol_null_p2.5': float(np.percentile(null, 2.5))})
        ok, why = accept_split(row)
        row.update({'accepted': ok, 'reason': why})
        rows.append(row)
        scores[u] = {'t': t, 'd2': d2, 'cut': cut, 'out': out, 'stim': stim,
                     'channels': chans[u], 'match': r_u, 'scale': s_u, 'nan': bad,
                     'split': split_labels}
        if ok:
            idx = np.arange(m.sum())
            splits[u] = [idx[~out].tolist(), idx[out].tolist()]
        if plot_folder is not None:
            _plot_unit(u, t, d2, cut, out, stim, X, pk, wf_k, wf_o, model, row,
                       Path(plot_folder), t_ms)
    audit = pd.DataFrame(rows).set_index('unit_id')
    return (splits, audit, scores) if return_scores else (splits, audit)


def plot_kept_vs_removed(analyzer, unit_id, score, row, folder, n_show=200, seed=0,
                         ms_window=(-1.0, 2.0)):
    """Raw traces of the kept and the removed spikes, on the unit's scoring channels.

    Raw rather than the PCA reconstructions the score itself uses: the question this
    figure answers is whether the removed spikes are junk, and an overlapping
    neighbour's spike or an artifact is only visible in the traces themselves.

    Columns: a random sample of kept spikes, the removed ones (most extreme first, up
    to n_show), and the two means with the template. Spikes fired during stimulation
    are drawn in their own colour in both groups. One row per channel, peak channel
    marked, all on one y-scale per row so kept and removed are directly comparable.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    fs = analyzer.sampling_frequency
    rec = analyzer.recording
    nb, na = int(round(-ms_window[0] * fs / 1000)), int(round(ms_window[1] * fs / 1000))
    t_ms = (np.arange(nb + na) - nb) / fs * 1000.0
    tpl_ext = analyzer.get_extension('templates')
    tpl = tpl_ext.get_unit_template(unit_id)
    tnb = tpl_ext.nbefore
    chans = score['channels']
    pk = chans[int(np.argmax([np.ptp(tpl[:, c]) for c in chans]))]
    t, out, stim, d2 = score['t'], score['out'], score['stim'], score['d2']
    rng = np.random.default_rng(seed)

    kept = np.flatnonzero(~out)
    kept = np.sort(rng.choice(kept, min(n_show, kept.size), replace=False))
    removed = np.flatnonzero(out)
    removed = removed[np.argsort(d2[removed])[::-1]][:n_show]

    def snippets(idx):
        frames = np.round(t[idx] * fs).astype(int)
        ok = (frames >= nb) & (frames < rec.get_num_samples() - na)
        return np.stack([rec.get_traces(start_frame=f - nb, end_frame=f + na,
                                        return_in_uV=True)[:, chans]
                         for f in frames[ok]]) if ok.any() else None, idx[ok]

    sk, kept = snippets(kept)
    sr, removed = snippets(removed) if removed.size else (None, removed)
    # the template on the same time axis as the snippets
    tt = (np.arange(tpl.shape[0]) - tnb) / fs * 1000.0

    colors = {'kept': '0.35', 'kept_stim': '#2a78d6', 'removed': '#d62728',
              'removed_stim': '#9467bd'}
    colors_hist = ('0.45', '#d62728')
    fig = plt.figure(figsize=(16.5, 2.1 * len(chans) + 0.8))
    gs = fig.add_gridspec(len(chans), 4, width_ratios=[1, 1, 1, 0.8])
    axs = np.empty((len(chans), 3), dtype=object)
    for ci in range(len(chans)):
        for col in range(3):
            axs[ci, col] = fig.add_subplot(
                gs[ci, col], sharex=axs[0, 0] if ci or col else None,
                sharey=axs[ci, 0] if col else None)
    # the per-spike template match the split decision uses, kept against removed
    hx = fig.add_subplot(gs[:, 3])
    bins = np.linspace(-0.2, 1.0, 49)
    hx.hist(score['match'][~out], bins=bins, color=colors_hist[0], alpha=0.6,
            density=True, label=f'kept (n={int((~out).sum())})')
    if out.any():
        hx.hist(score['match'][out], bins=bins, color=colors_hist[1], alpha=0.6,
                density=True, label=f'removed (n={int(out.sum())})')
    hx.axvline(row['kept_match_p5'], color='k', ls='--', lw=0.8, label='kept 5th pct')
    hx.set_xlabel('per-spike template match (r, all 4 channels)')
    hx.set_ylabel('density')
    hx.legend(fontsize=7, loc='upper left')
    for ci, c in enumerate(chans):
        for col, (s, idx, name) in enumerate(((sk, kept, 'kept'), (sr, removed, 'removed'))):
            ax = axs[ci, col]
            if s is not None:
                for j in range(s.shape[0]):
                    key = name + ('_stim' if stim[idx[j]] else '')
                    ax.plot(t_ms, s[j, :, ci], color=colors[key], lw=0.5,
                            alpha=0.25 if name == 'kept' else 0.45)
            ax.plot(tt, tpl[:, c], color='k', lw=1.4)
        ax = axs[ci, 2]
        ax.plot(tt, tpl[:, c], color='k', lw=1.4, label='template')
        if sk is not None:
            ax.plot(t_ms, sk[:, :, ci].mean(0), color=colors['kept'], lw=1.6,
                    label='kept mean')
        if sr is not None:
            ax.plot(t_ms, sr[:, :, ci].mean(0), color=colors['removed'], lw=1.6,
                    label='removed mean')
        axs[ci, 0].set_ylabel(f'ch {c}' + ('  (peak)' if c == pk else '') + '\nuV',
                              fontsize=8)
    n_out = int(out.sum())
    axs[0, 0].set_title(f'kept: {kept.size} of {int((~out).sum())} shown (random)\n'
                        'grey spontaneous, blue during stim', fontsize=9)
    axs[0, 1].set_title(f'removed: {removed.size} of {n_out} shown (most extreme first)\n'
                        'red spontaneous, purple during stim', fontsize=9)
    axs[0, 2].set_title('means', fontsize=9)
    axs[0, 2].legend(fontsize=7, loc='lower right')
    for ax in axs[-1]:
        ax.set_xlabel('t (ms) from spike')
    verdict = 'would split' if row['accepted'] else 'would NOT split'
    fig.suptitle(
        f"unit {unit_id}: {verdict} -- {row['reason']}\n"
        f"removed {n_out} of {row['n_spikes']} ({100 * row['outlier_frac']:.2f}%), "
        f"{100 * row['outlier_in_stim_frac']:.0f}% of them during stim  |  violations "
        f"<{REFRACTORY_MS:g} ms: {row['viol_before']} -> {row['viol_kept']} kept "
        f"(random removal of as many: {row['viol_null_mean']:.0f})  |  per-spike template "
        f"match, median: removed {row['removed_match_median']:.2f}, kept "
        f"{row['kept_match_median']:.2f}; {100 * row['removed_below_kept_p5_share']:.0f}% of "
        f"removed below the kept 5th pct ({row['kept_match_p5']:.2f})\n"
        f"per-spike size, median: removed {row['removed_scale_median']:.2f}x template, kept "
        f"1st-99th pct {row['kept_scale_p1']:.2f}-{row['kept_scale_p99']:.2f}",
        fontsize=9)
    fig.tight_layout()
    return _save(fig, folder, f'unit_{unit_id}_kept_vs_removed')


def _plot_unit(u, t, d2, cut, out, stim, X, pk, wf_k, wf_o, model, row, folder, t_ms):
    """Four panels: distance over time, PC scatter, waveforms, ISI histograms."""
    folder.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(1, 4, figsize=(19, 4.2), constrained_layout=True)
    kept_c, out_c, stim_c = '0.45', '#d62728', '#2a78d6'

    ax = axs[0]
    ax.scatter(t[~out & ~stim], d2[~out & ~stim], s=1, c=kept_c, alpha=0.3, lw=0,
               label='kept, spontaneous')
    ax.scatter(t[~out & stim], d2[~out & stim], s=2, c=stim_c, alpha=0.5, lw=0,
               label='kept, during stim')
    ax.scatter(t[out], d2[out], s=4, c=out_c, lw=0, label='outlier')
    ax.axhline(cut, color='k', ls='--', lw=0.8, label=f'cut ({row["outlier_z"]:g} robust SD)')
    ax.set_yscale('log')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('robust Mahalanobis distance$^2$')
    ax.legend(fontsize=7, markerscale=4)

    ax = axs[1]
    a, b = pk * 5, pk * 5 + 1
    ax.scatter(X[~out, a], X[~out, b], s=1, c=kept_c, alpha=0.3, lw=0)
    ax.scatter(X[out, a], X[out, b], s=4, c=out_c, lw=0)
    ax.set_xlabel('PC1, peak channel')
    ax.set_ylabel('PC2, peak channel')

    ax = axs[2]
    # individual spikes reconstructed from their PCs: the 5-component view the score
    # itself sees, drawn for a sample of each group
    rng = np.random.default_rng(0)
    for mask, c in ((~out, kept_c), (out, out_c)):
        idx = np.flatnonzero(mask)
        if idx.size:
            idx = rng.choice(idx, min(idx.size, 200), replace=False)
            for w in model.inverse_transform(X[idx, pk * 5:pk * 5 + 5]):
                ax.plot(t_ms, w, color=c, lw=0.4, alpha=0.15)
    ax.plot(t_ms, wf_k, color='k', lw=1.8, label=f'kept mean (n={int((~out).sum())})')
    if out.any():
        ax.plot(t_ms, wf_o, color=out_c, lw=1.8, label=f'outlier mean (n={int(out.sum())})')
    ax.set_xlabel('t (ms), peak channel, PCA reconstruction')
    ax.set_ylabel('uV')
    ax.legend(fontsize=7)

    ax = axs[3]
    bins = np.arange(0, 20.25, 0.25)
    for tt, c, lab in ((t, '0.2', 'whole unit'), (t[~out], stim_c, 'kept')):
        ax.hist(np.diff(np.sort(tt)) * 1000, bins=bins, histtype='step', color=c, label=lab)
    ax.axvline(REFRACTORY_MS, color=out_c, ls=':', lw=1)
    ax.set_xlabel('ISI (ms)')
    ax.set_ylabel('count')
    ax.legend(fontsize=7)

    verdict = 'SPLIT' if row['accepted'] else 'not split'
    fig.suptitle(
        f"unit {u}: {verdict} ({row['reason']})  |  {row['n_outliers']} of "
        f"{row['n_spikes']} outliers ({100 * row['outlier_frac']:.2f}%, "
        f"{100 * row['outlier_in_stim_frac']:.0f}% during stim)  |  violations "
        f"<{REFRACTORY_MS:g} ms {row['viol_before']} -> {row['viol_kept']} kept "
        f"({row['viol_kept_outside_stim']} outside stim)  |  template match, median: "
        f"removed {row['removed_match_median']:.2f}, kept {row['kept_match_median']:.2f} "
        f"(kept 5th pct {row['kept_match_p5']:.2f})  |  size, median: removed "
        f"{row['removed_scale_median']:.2f}x template (kept 1-99th pct {row['kept_scale_p1']:.2f}-"
        f"{row['kept_scale_p99']:.2f})", fontsize=9)
    _save(fig, folder, f'unit_{u}_outlier_split')


def split_outlier_units(analyzer, splits, **job_kwargs):
    """Apply the splits. Returns (new_analyzer, {old_unit: (kept_id, outlier_id)})."""
    if not splits:
        return analyzer, {}
    new, new_ids = analyzer.split_units(splits, new_id_strategy='append',
                                        return_new_unit_ids=True, **job_kwargs)
    return new, {u: tuple(ids) for u, ids in zip(splits, new_ids)}
