"""Close pairs of units after the splits and peels: merge the pieces of one neuron, and delete
the spikes that two units both detected (user, 2026-10-12). Used by spike_cleaning.clean_and_curate:
the merge is the PAIR_MERGE preset of auto_merge (step 5), the deletion step 6b.

Merge (pair_merge_pairs, pair_group_veto). Candidates: units within PAIR_MAX_UM whose templates
have cosine >= PAIR_MIN_COSINE. On spontaneous spikes (outside every light pulse + RP_STIM_PAD_S;
evoked spikes shrink at high rates and fire in bursts):
  refractory  coincidences in (W, R] against the 2-10 ms flank rate. One neuron cannot fire twice
              there; two neurons do at their chance rate. Spikes <= W apart are double detections in
              either case and decide nothing. The pair is judged only with power (expected >=
              REFRACTORY_EXPECTED_MIN): 'merge' when the dip is significant (Poisson p <
              DIP_ALPHA) and as deep as the labels' tolerance (95% lower bound of observed /
              expected <= RP_MAX_CONTAMINATION); 'separate' with no significant dip; 'partial'
              with a shallow one -- close pairs of different neurons sit near 0.74, the sorter
              misses some close detections (2026-10-11); 'unresolved' without power (drift
              pieces, which barely coexist, are always here).
  clusters    the two units' spontaneous random spikes on their shared channels, PCA, Gaussian
              mixtures of 1-3 components (BIC): one neuron when the clusters do not follow the
              units (adjusted Rand index < ARI_MERGE) and the amplitudes overlap (d' < MAX_DPRIME).
A pair merges when the refractory test does not say 'separate'/'partial', there is one-neuron
evidence (refractory 'merge' or clusters), and the merge is worth it (user, 2026-10-12: a good
merge raises presence, above all of tagged units, without costing SNR): the union's presence
ratio >= the better member's + MERGE_MIN_PRESENCE_GAIN, and the merged template's SNR >=
MERGE_MIN_SNR_RATIO x the better member's. Coexisting pieces of one neuron at different amplitudes
gain no presence and lose SNR when merged (2026-10-12: 31 of 42 such merges); they stay apart and
their shared spikes are deleted below. A merge group must pass all of it between every two members
and, as a whole, the labels' refractory rule.

Double detections (double_detections). Two distinct neurons give a flat cross-correlogram; a spike
of one that the other's template also caught gives a narrow excess near zero lag. Units sharing
>= DD_MIN_SHARED_CHANNELS channels within DD_MAX_UM: the DD_PEAK_MS window (sliding within +-DD_CENTER_MS) with the
largest excess over the 2-10 ms flank rate, flagged at Poisson p < DD_P_MAX (Bonferroni over the
windows), excess >= DD_MIN_EXCESS and >= the chance count. Both copies of a double detection are
the same waveform, so the template that explains it better keeps it -- the cleaning's own rule for
close pairs within a unit -- read from each spike's template match (split_outlier_spikes scores;
not Kilosort's pc_features, which each template shapes: every copy sits nearer its own unit's mean,
DS15_20260701 634/656). One side per pair, by the median match difference (copy by copy scatters
one neuron over two units, DS15_20260707 452/623); the excess copies go from it, the clearest first.
Every multiply detected spike keeps one copy; a unit made only of copies goes whole.
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2, poisson

PAIR_MAX_UM = 50.0
PAIR_MIN_COSINE = 0.8
CLUSTER_DUP_MS = 0.3          # spikes this close to the other unit's are left out of the clustering
CLUSTER_MIN_SPIKES = 30
MIN_CLUSTER_SHARE = 0.1
ARI_MERGE = 0.3
MAX_DPRIME = 1.0
REFRACTORY_EXPECTED_MIN = 3.0
DIP_ALPHA = 0.05
FLANK_MS = (2.0, 10.0)
PRESENCE_BIN_S = 60.0
MERGE_MIN_PRESENCE_GAIN = 0.05
MERGE_MIN_SNR_RATIO = 0.8
DD_MIN_SHARED_CHANNELS = 2
# and units this close: a zero-lag peak between distant units is synchrony or a shared artifact, not
# one spike caught twice (DS2_20260511: 108 extra pairs by channels alone, median 146 um apart)
DD_MAX_UM = 60.0
DD_CENTER_MS = 0.5
DD_PEAK_MS = 0.2
DD_P_MAX = 1e-4
DD_MIN_EXCESS = 10
DD_MIN_EXCESS_RATIO = 1.0


def light_samples(stim_intervals, fs, pad_s):
    """Light pulses + pad as merged, sorted (n, 2) sample intervals."""
    iv = np.asarray(stim_intervals if stim_intervals is not None else np.zeros((0, 2)), float).reshape(-1, 2)
    iv = iv[np.argsort(iv[:, 0])] + np.array([0.0, pad_s]) if len(iv) else iv
    out = []
    for lo, hi in iv:
        if out and lo <= out[-1][1]:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return (np.asarray(out) * fs).round().astype(np.int64) if out else np.zeros((0, 2), np.int64)


def in_light(samples, light):
    if not len(light):
        return np.zeros(np.shape(samples), bool)
    k = np.searchsorted(light[:, 0], samples, side='right') - 1
    return (k >= 0) & (samples <= light[np.clip(k, 0, None), 1])


def _count_within(a, b, w0, w1):
    """Pairs with w0 < |b - a| <= w1 (sorted trains)."""
    n1 = (np.searchsorted(b, a + w1, 'right') - np.searchsorted(b, a - w1, 'left')).sum()
    n0 = (np.searchsorted(b, a + w0, 'right') - np.searchsorted(b, a - w0, 'left')).sum()
    return int(n1 - n0)


def refractory_test(sa, sb, fs, w_ms, r_ms, max_contamination):
    """The cross-refractory verdict of two sorted spontaneous trains."""
    t_c, t_r = (int(round(x * 1e-3 * fs)) for x in (w_ms, r_ms))
    f0, f1 = (int(round(x * 1e-3 * fs)) for x in FLANK_MS)
    obs = _count_within(sa, sb, t_c, t_r)
    exp_ = _count_within(sa, sb, f0, f1) / (2 * (f1 - f0)) * 2 * (t_r - t_c)
    p_dip = float(poisson.cdf(obs, exp_)) if exp_ > 0 else 1.0
    ratio_lo = (chi2.ppf(0.05, 2 * obs) / 2 / exp_ if obs else 0.0) if exp_ > 0 else np.nan
    if exp_ < REFRACTORY_EXPECTED_MIN:
        verdict = 'unresolved'
    elif p_dip >= DIP_ALPHA:
        verdict = 'separate'
    elif ratio_lo <= max_contamination:
        verdict = 'merge'
    else:
        verdict = 'partial'
    return {'cross_obs': obs, 'cross_expected': round(float(exp_), 1), 'p_dip': round(p_dip, 4),
            'cross_ratio_lo': round(float(ratio_lo), 3) if np.isfinite(ratio_lo) else np.nan,
            'refractory': verdict}


def _near(t, other, w):
    if not other.size:
        return np.zeros(t.size, bool)
    j = np.searchsorted(other, t)
    return np.minimum(np.abs(t - other[np.clip(j - 1, 0, None)]),
                      np.abs(other[np.clip(j, None, other.size - 1)] - t)) <= w


def cluster_test(analyzer, u, v, trains, light):
    """(ARI of the mixture clusters against the units, number of real clusters, spontaneous
    amplitude d') -- NaNs when either unit has too few spontaneous random spikes."""
    from sklearn.decomposition import PCA
    from sklearn.metrics import adjusted_rand_score
    from sklearn.mixture import GaussianMixture
    fs = analyzer.sampling_frequency
    w = int(round(CLUSTER_DUP_MS * 1e-3 * fs))
    amps = analyzer.get_extension('spike_amplitudes').get_data(outputs='by_unit')[0]
    a_u = np.asarray(amps[u])[~in_light(analyzer.sorting.get_unit_spike_train(u), light)]
    a_v = np.asarray(amps[v])[~in_light(analyzer.sorting.get_unit_spike_train(v), light)]
    dprime = (float(abs(a_u.mean() - a_v.mean()) / np.sqrt((a_u.var() + a_v.var()) / 2))
              if a_u.size > 1 and a_v.size > 1 else np.nan)
    sp = analyzer.sparsity.unit_id_to_channel_ids if analyzer.sparsity is not None else None
    chans = ([c for c in analyzer.channel_ids if c in set(sp[u]) & set(sp[v])] if sp is not None
             else list(analyzer.channel_ids))
    rs = analyzer.get_extension('random_spikes').get_random_spikes()
    wf = analyzer.get_extension('waveforms')
    X, lab = [], []
    for x, other in ((u, trains[v]), (v, trains[u])):
        st = rs['sample_index'][rs['unit_index'] == analyzer.sorting.id_to_index(x)]
        keep = ~in_light(st, light) & ~_near(st, other, w)
        if len(chans) < 2 or keep.sum() < CLUSTER_MIN_SPIKES:
            return np.nan, np.nan, dprime
        wv = wf.get_waveforms_one_unit(x, force_dense=True)[:, :, analyzer.channel_ids_to_indices(chans)]
        X.append(wv[keep].reshape(int(keep.sum()), -1))
        lab.append(np.full(int(keep.sum()), 0 if x == u else 1))
    X, lab = np.vstack(X), np.concatenate(lab)
    Z = PCA(n_components=min(6, X.shape[1], X.shape[0] - 1)).fit_transform(X)
    fits = {k: GaussianMixture(k, covariance_type='full', n_init=2, random_state=0).fit(Z)
            for k in (1, 2, 3) if Z.shape[0] > 10 * k}
    best = min(fits, key=lambda k: fits[k].bic(Z))
    pred = fits[best].predict(Z)
    share = np.bincount(pred, minlength=best) / pred.size
    real = np.flatnonzero(share >= MIN_CLUSTER_SHARE)
    core = np.isin(pred, real)
    ari = adjusted_rand_score(lab[core], pred[core]) if real.size > 1 else 0.0
    return float(ari), int(real.size), dprime


def presence(train, fs, n_bins):
    return np.unique((np.asarray(train) / fs // PRESENCE_BIN_S).astype(int)).size / max(n_bins, 1)


def merged_snr(analyzer, group, counts):
    """(best member's SNR, the merged template's SNR) on the best member's peak channel: the merged
    template is the spike-weighted mean of the members', as a soft merge makes it."""
    tpl = analyzer.get_extension('templates').get_data(operator='average')
    noise = analyzer.get_extension('noise_levels').get_data()
    idx = [analyzer.sorting.id_to_index(u) for u in group]
    peak = [int(np.argmax(np.ptp(tpl[i], axis=0))) for i in idx]
    snr = [float(np.abs(tpl[i][:, p]).max() / noise[p]) for i, p in zip(idx, peak)]
    b = int(np.argmax(snr))
    n = np.array([counts[u] for u in group], float)
    merged = np.tensordot(n / n.sum(), tpl[idx], axes=1)
    return snr[b], float(np.abs(merged[:, peak[b]]).max() / noise[peak[b]])


def pair_merge_pairs(analyzer, exclude, stim_intervals, w_ms, r_ms, pad_s, max_contamination):
    """(pairs to merge, evidence table of every candidate)."""
    fs = analyzer.sampling_frequency
    light = light_samples(stim_intervals, fs, pad_s)
    n_bins = int(np.ceil(analyzer.get_total_duration() / PRESENCE_BIN_S))
    ids = [u for u in analyzer.unit_ids if u not in exclude]
    all_ids = list(analyzer.unit_ids)
    pos = np.asarray(analyzer.get_extension('unit_locations').get_data(), float)[:, :2]
    sim = analyzer.get_extension('template_similarity').get_data()
    counts = analyzer.sorting.count_num_spikes_per_unit()
    trains = {u: np.sort(analyzer.sorting.get_unit_spike_train(u)) for u in ids}
    spont = {u: t[~in_light(t, light)] for u, t in trains.items()}
    rows, pairs = [], []
    for i, u in enumerate(ids):
        iu = all_ids.index(u)
        for v in ids[i + 1:]:
            iv = all_ids.index(v)
            d = float(np.hypot(*(pos[iu] - pos[iv])))
            if sim[iu, iv] < PAIR_MIN_COSINE or d > PAIR_MAX_UM:
                continue
            r = {'unit_a': u, 'unit_b': v, 'distance_um': round(d, 1), 'template_cosine': round(float(sim[iu, iv]), 3)}
            r.update(refractory_test(spont[u], spont[v], fs, w_ms, r_ms, max_contamination))
            ari, ncl, dp = cluster_test(analyzer, u, v, trains, light)
            r.update({'ari': ari, 'n_clusters': ncl, 'amp_dprime': dp})
            pu, pv = presence(trains[u], fs, n_bins), presence(trains[v], fs, n_bins)
            pm = presence(np.concatenate([trains[u], trains[v]]), fs, n_bins)
            snr_best, snr_m = merged_snr(analyzer, [u, v], counts)
            r.update({'presence_best': round(max(pu, pv), 3), 'presence_merged': round(pm, 3),
                      'snr_best': round(snr_best, 2), 'snr_merged': round(snr_m, 2)})
            r['verdict'] = _pair_verdict(r)
            rows.append(r)
            if r['verdict'] == 'merge':
                pairs.append((u, v))
    return pairs, pd.DataFrame(rows)


def _pair_verdict(r):
    if r['refractory'] in ('separate', 'partial'):
        return f"different neurons (refractory {r['refractory']})"
    one = r['refractory'] == 'merge' or (np.isfinite(r['ari']) and r['ari'] < ARI_MERGE
                                         and np.isfinite(r['amp_dprime']) and r['amp_dprime'] < MAX_DPRIME)
    if not one:
        return 'no one-neuron evidence'
    if r['presence_merged'] - r['presence_best'] < MERGE_MIN_PRESENCE_GAIN:
        return 'one neuron, no presence gain (not merged; shared spikes go in step 6b)'
    if r['snr_merged'] < MERGE_MIN_SNR_RATIO * r['snr_best']:
        return 'one neuron, merging costs SNR (not merged)'
    return 'merge'


def pair_group_veto(analyzer, group, stim_intervals, w_ms, r_ms, pad_s, max_contamination,
                    rp_lower_bound, rp_max_free):
    """'' when every two members and the group as a whole pass; else why not."""
    fs = analyzer.sampling_frequency
    light = light_samples(stim_intervals, fs, pad_s)
    trains = {u: np.sort(analyzer.sorting.get_unit_spike_train(u)) for u in group}
    spont = {u: t[~in_light(t, light)] for u, t in trains.items()}
    for i, u in enumerate(group):
        for v in group[i + 1:]:
            r = refractory_test(spont[u], spont[v], fs, w_ms, r_ms, max_contamination)
            if r['refractory'] in ('separate', 'partial'):
                return f"{u}/{v}: refractory {r['refractory']} ({r['cross_obs']} vs {r['cross_expected']})"
    n_bins = int(np.ceil(analyzer.get_total_duration() / PRESENCE_BIN_S))
    counts = analyzer.sorting.count_num_spikes_per_unit()
    best = max(presence(trains[u], fs, n_bins) for u in group)
    union = np.sort(np.concatenate(list(trains.values())))
    if presence(union, fs, n_bins) - best < MERGE_MIN_PRESENCE_GAIN:
        return 'group: no presence gain'
    snr_best, snr_m = merged_snr(analyzer, list(group), counts)
    if snr_m < MERGE_MIN_SNR_RATIO * snr_best:
        return f'group: SNR {snr_best:.1f} -> {snr_m:.1f}'
    t_c, t_r = (int(round(x * 1e-3 * fs)) for x in (w_ms, r_ms))
    m = union[~in_light(union, light)]
    m = m[np.concatenate([[True], np.diff(m) > t_c])] if m.size else m
    i = np.arange(m.size)
    n_v = int(np.sum(np.searchsorted(m, m + t_r, 'right') - i - 1) - np.sum(np.searchsorted(m, m + t_c, 'right') - i - 1))
    n_spont = analyzer.get_total_samples() - int((light[:, 1] - light[:, 0]).sum()) if len(light) else analyzer.get_total_samples()
    lo = rp_lower_bound(n_v, m.size, n_spont, t_c, t_r)
    if lo > max_contamination and n_v > rp_max_free:
        return f'group fails the refractory rule ({n_v} spontaneous violations)'
    return ''


def _excess(ta, tb, fs):
    c = int(round(DD_CENTER_MS * 1e-3 * fs))
    hw = max(1, int(round(DD_PEAK_MS / 2 * 1e-3 * fs)))
    f0, f1 = (int(round(x * 1e-3 * fs)) for x in FLANK_MS)
    rate = _count_within(ta, tb, f0, f1) / (2 * (f1 - f0))
    lo, hi = np.searchsorted(tb, ta - c - hw), np.searchsorted(tb, ta + c + hw, 'right')
    lg = np.concatenate([tb[i:j] - x for x, i, j in zip(ta, lo, hi)] or [np.zeros(0, np.int64)])
    h = np.bincount(lg + c + hw, minlength=2 * (c + hw) + 1)
    win = np.convolve(h, np.ones(2 * hw + 1, int), 'valid')
    expected = rate * (2 * hw + 1)
    k = int(np.argmax(win))
    obs = int(win[k])
    n_tests = win.size / (2 * hw + 1)
    p = float(min(1.0, poisson.sf(obs - 1, expected) * n_tests)) if expected > 0 else 0.0
    return obs, expected, p, k - c, hw


def double_detections(analyzer, stim_intervals, names, score_fn):
    """(drops {unit: indices into the unit's spike train, vector order}, whole units to remove,
    table of the flagged pairs). score_fn(unit_ids) -> {unit: {'match': per spike, vector order}}."""
    fs = analyzer.sampling_frequency
    ids = list(analyzer.unit_ids)
    samples = {u: analyzer.sorting.get_unit_spike_train(u) for u in ids}   # vector order = sorted
    sp = analyzer.sparsity.unit_id_to_channel_ids if analyzer.sparsity is not None else None
    chans = {u: set(sp[u]) if sp is not None else set(analyzer.channel_ids) for u in ids}
    pos = np.asarray(analyzer.get_extension('unit_locations').get_data(), float)[:, :2]
    flagged = []
    for i, u in enumerate(ids):
        for k, v in enumerate(ids[i + 1:], start=i + 1):
            if (len(chans[u] & chans[v]) < DD_MIN_SHARED_CHANNELS
                    or np.hypot(*(pos[i] - pos[k])) > DD_MAX_UM):
                continue
            obs, exp_, p, lag, hw = _excess(samples[u], samples[v], fs)
            ex = obs - exp_
            if p < DD_P_MAX and ex >= DD_MIN_EXCESS and ex >= DD_MIN_EXCESS_RATIO * exp_:
                flagged.append((u, v, obs, exp_, p, lag, hw))
    if not flagged:
        return {}, [], pd.DataFrame()
    scores = score_fn(sorted({x for f in flagged for x in f[:2]}))
    drops, kept, rows = {}, set(), []
    for u, v, obs, exp_, p, lag, hw in flagged:
        ta, tb = samples[u], samples[v]
        j0 = np.searchsorted(tb, ta + lag - hw)
        j1 = np.searchsorted(tb, ta + lag + hw, 'right')
        i = np.flatnonzero(j1 > j0)
        j = j0[i]
        ma, mb = np.asarray(scores[u]['match'], float)[i], np.asarray(scores[v]['match'], float)[j]
        d = np.nan_to_num(mb - ma)                     # > 0: the copy in u matches its template worse
        from_u = bool(np.median(d) >= 0) if d.size else True
        score = d if from_u else -d
        n_del = int(min(round(obs - exp_), d.size))
        order = np.argsort(score)[::-1][:n_del]
        du, dv = [], []
        for k in order:
            (xu, xi), (ku, ki) = ((u, int(i[k])), (v, int(j[k]))) if from_u else ((v, int(j[k])), (u, int(i[k])))
            if (xu, xi) in kept or ki in drops.get(ku, ()):
                continue                               # every multiply detected spike keeps one copy
            drops.setdefault(xu, set()).add(xi)
            kept.add((ku, ki))
            (du if xu == u else dv).append(xi)
        rows.append({'unit_a': u, 'unit_b': v, 'name_a': names.get(u, str(u)), 'name_b': names.get(v, str(v)),
                     'peak_lag_ms': round(lag / fs * 1e3, 3), 'coincident': obs, 'expected': round(exp_, 1),
                     'p': p, 'median_match_a': round(float(np.median(ma)), 3) if ma.size else np.nan,
                     'median_match_b': round(float(np.median(mb)), 3) if mb.size else np.nan,
                     'copies_from': u if from_u else v, 'deleted_from_a': len(du), 'deleted_from_b': len(dv)})
    whole = [u for u, d in drops.items() if len(d) == samples[u].size]
    drops = {u: np.array(sorted(d), dtype=np.int64) for u, d in drops.items() if u not in whole and d}
    return drops, whole, pd.DataFrame(rows)
