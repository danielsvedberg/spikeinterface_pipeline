"""Validation metrics for the preprocessing chain, run before and after preprocessing.

Two routines, both callable on their own:

    validate_raw(raw_rec, save_folder)            -- what is in the data before anything
    validate_preprocessed(final_rec, save_folder) -- what the chain did to it

and one orchestrator, run_preprocessing_qc, that derives the intermediate recordings
the "after" routine needs (the chain without common_reference, and without phase_shift)
straight from the protocol dict, runs both, and writes a combined report.

Every metric is computed from the same randomly-placed chunks within a recording, and
the *same* chunk positions are reused across recordings, so before/after numbers are
directly comparable rather than differing by which second of the session they sampled.
Chunks are read with a margin that is trimmed off, so filter edge effects do not leak in.

Metrics
    noise levels         si.get_noise_levels per channel; the durable number is the
                         pre-CAR -> post-CAR ratio and the across-channel spread.
    bad channels         si.detect_bad_channels labels (dead/noise/out), their spatial
                         contiguity, and a cross-tab against manually supplied channels.
    saturation           fraction of samples at the amplifier rail, per channel. The
                         rail is measured (see detect_saturation), not taken from the
                         dtype: the amplifier clips well below ADC full scale, so an
                         int16-derived threshold never fires and calls every session
                         clean.
    channel correlation  median |r| for spatially distant pairs, pre- vs post-CAR,
                         using the full electrode geometry and split within- against
                         cross-shank.
    PSD vs depth         Welch per channel, stacked by depth.
    line noise           power at 60 Hz and its harmonics relative to a local baseline.
    spectral peaks       narrowband lines found in the raw spectrum and re-measured after
                         the chain, unioned with any the preprocessed spectrum shows on
                         its own: a line can sit under the flag threshold in raw and stand
                         proud once CAR drops the broadband floor out from under it.
    adjacency lag        peak cross-correlation lag between depth-adjacent channels,
                         with and without phase_shift.
    artifact removal     trigger-averaged trace before the step, after it, and after
                         the whole chain. The only step whose success cannot be read
                         off a spectrum, since it is a local, time-domain repair.
    event averages       the same average around behavioural events the caller passes
                         in (licks, cues), raw against fully preprocessed. Not a test
                         of a step: it asks whether the event leaves a trigger-locked
                         signature in the traces, and what the chain does with it.
"""

import json
import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import welch, find_peaks
from scipy.ndimage import median_filter, uniform_filter1d

import spikeinterface.full as si

from preprocessing import detect_saturation as dsat


LINE_FREQ = 60.0
# A harmonic counts as present when its peak sits this far above the local broadband
# baseline. A Welch estimate off a dozen chunks has ~1-2 dB of scatter and the peak is a
# max over three bins, so 3 dB (one doubling) fires on noise; 6 dB does not.
LINE_FLAG_DB = 6.0

DEFAULT_SCAN = dict(n_chunks=16, chunk_duration_s=2.0, margin_s=0.1, seed=0)

# Label of the post-chain stage in the artifact plot. It is both an axis title and a key
# in the stats dict that the text summary reads back, so the two have to agree: a rename
# on one side alone would make the summary report uV for a trace that is not in uV.
FINAL_STAGE = 'fully preprocessed'

# Per-shank colours for the per-channel traces in the event-average plots. Slots 1, 2, 3
# and 7 of the standard categorical palette. Every shank's traces overlap every other
# shank's in the same axes, so the palette has to hold under all-pairs comparison rather
# than only between neighbours: slots 1-4 do not (orange vs yellow is 13.7 in
# normal-vision OKLab dE, under the floor of 15, and yellow is 2.1:1 on white, which for
# a 0.3 pt line is invisible). This set measures worst-pair CVD dE 9.2 against a target
# of 8 and worst normal-vision dE 16.3 against a floor of 15, protan and deutan
# simulated. Every colour still sits under 3:1 against white, so the shank legend is not
# optional -- it is what carries the identity that the contrast does not.
SHANK_COLORS = ('#2a78d6', '#eb6834', '#1baf7a', '#4a3aa7')

# Opacity of one per-channel trace, low on purpose. The lines composite, so ~96 of them
# per shank at alpha a stack to 1 - (1 - a)^96: at any value that makes a single trace
# comfortably visible the bundle is solid, and the last shank drawn simply paints over
# the other three. 0.15 is the point where a trace is still followable on its own and
# the shanks underneath still show through -- which is what makes it visible on DS45
# that shank 1's post-chain spread is wider than shanks 0, 2 and 3. Lower (~0.05) turns
# the bundle into a density cloud and loses the single outlying channel; the plotting
# functions take channel_alpha if a particular plot wants that trade.
CHANNEL_ALPHA = 0.15

# Clipping is a rare-event metric -- opto-driven amplifier saturation can be a handful of
# milliseconds in an hour -- so sampled chunks will miss it. This scans the whole
# recording instead, in large reads; set full_saturation_scan=False to sample instead.
SATURATION_CHUNK_S = 10.0


# ---------------------------------------------------------------------------
# chunk sampling
# ---------------------------------------------------------------------------

def _chunk_starts(rec, n_chunks, chunk_duration_s, margin_s, seed):
    """Random chunk start frames, spaced away from the recording edges by the margin."""
    fs = rec.get_sampling_frequency()
    n_samp = int(round(chunk_duration_s * fs))
    margin = int(round(margin_s * fs))
    total = rec.get_num_samples()
    lo, hi = margin, total - n_samp - margin
    if hi <= lo:
        # recording shorter than one padded chunk: take whatever single chunk fits
        n_samp = max(min(n_samp, total - 2 * margin), 1)
        return np.array([margin]), n_samp, margin
    rng = np.random.default_rng(seed)
    starts = np.sort(rng.integers(lo, hi, size=n_chunks))
    return starts, n_samp, margin


def _read_chunk(rec, start, n_samp, margin, return_in_uV=True):
    """One chunk as (n_samp, n_ch) float32, with the filter margin read then trimmed."""
    total = rec.get_num_samples()
    a = max(start - margin, 0)
    b = min(start + n_samp + margin, total)
    tr = rec.get_traces(start_frame=a, end_frame=b, return_in_uV=return_in_uV)
    off = start - a
    return np.asarray(tr[off:off + n_samp], dtype=np.float32)


def _traces_in_uV(rec):
    """Whether get_traces(return_in_uV=True) will work; preprocessed chains sometimes drop gains."""
    try:
        rec.get_traces(start_frame=0, end_frame=min(10, rec.get_num_samples()),
                       return_in_uV=True)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

def _depths(rec):
    """Channel y positions (depth along the shank)."""
    return np.asarray(rec.get_channel_locations())[:, 1]


def _locations(rec):
    """Full electrode coordinates, for distances that respect shank geometry."""
    return np.asarray(rec.get_channel_locations(), dtype=float)


def _shank_ids(rec):
    """Per-channel shank identity, or None if the probe reports a single shank.

    Used to split the referencing metric within- against cross-shank; a global median
    can only help where the common-mode noise is actually shared between shanks.
    """
    try:
        sh = np.asarray(rec.get_probe().shank_ids)
        if sh.size == rec.get_num_channels() and np.unique(sh).size > 1:
            return sh
    except Exception:
        pass
    g = rec.get_property('group')
    if g is not None and np.unique(np.asarray(g)).size > 1:
        return np.asarray(g)
    return None


def shank_groups(rec):
    """Channel ids grouped by shank, as common_reference's `groups` argument wants.

    common_reference converts each group with ids_to_indices, so these are channel *ids*
    and every one has to exist on the recording being referenced. That is the trap:
    detect_and_remove_bad_channels sits upstream of common_reference and shrinks the
    channel set, so groups built from the raw recording raise once a channel has been
    dropped. Build them from the recording that is actually handed to common_reference
    -- pre_car_rec in the pipeline -- or keep the channel count fixed by interpolating
    bad channels instead of removing them.

    Returns None for a single-shank probe, which is what common_reference already does
    by default, so the caller can pass it through unconditionally.
    """
    sh = _shank_ids(rec)
    if sh is None:
        return None
    ids = np.asarray(rec.channel_ids)
    sh = np.asarray(sh)
    return [ids[sh == s].tolist() for s in np.unique(sh)]


def _adjacent_pairs(rec, max_distance_um=60.0, exclude_inds=()):
    """Index pairs of channels that are neighbours in depth, for the phase_shift check.

    Pairs are consecutive among the channels that are *present*, so when the chain has
    removed bad channels a pair can straddle the gap. Those are dropped: the spacing is
    held to the modal pitch, which is the real row spacing of the probe.

    Neighbours are found within each shank, not across the whole probe. Sorting all
    channels of a multi-shank probe by depth interleaves the shanks -- every shank of an
    NP2.0 4-shank spans the same 1425 um -- so consecutive-in-depth channels are usually
    on *different* shanks, 250 um apart, and the distance cap then rejects all of them.
    That left the metric with zero pairs and reporting nan, which reads as "phase_shift
    untested" rather than as a failure.
    """
    loc = np.asarray(rec.get_channel_locations())
    keep = np.setdiff1d(np.arange(len(loc)), np.asarray(list(exclude_inds), dtype=int))
    shanks = _shank_ids(rec)
    groups = ([keep] if shanks is None
              else [keep[np.asarray(shanks)[keep] == s]
                    for s in np.unique(np.asarray(shanks)[keep])])

    ia_all, ib_all, d_all = [], [], []
    for grp in groups:
        if grp.size < 2:
            continue
        order = grp[np.argsort(loc[grp, 1], kind='stable')]
        ia, ib = order[:-1], order[1:]
        ia_all.append(ia)
        ib_all.append(ib)
        d_all.append(np.linalg.norm(loc[ia] - loc[ib], axis=1))
    if not ia_all:
        return np.array([], dtype=int), np.array([], dtype=int)

    ia = np.concatenate(ia_all)
    ib = np.concatenate(ib_all)
    d = np.concatenate(d_all)
    nz = d[d > 0]
    if nz.size:
        # the modal spacing is the probe's real row pitch; anything wider is a gap left
        # by a removed channel
        pitch = float(np.median(nz))
        max_distance_um = min(max_distance_um, 1.5 * pitch)
    ok = (d > 0) & (d <= max_distance_um)
    return ia[ok], ib[ok]


# ---------------------------------------------------------------------------
# single pass over the sampled chunks
# ---------------------------------------------------------------------------

def _next_pow2_le(n):
    return 1 << (int(n).bit_length() - 1)


def _scan(rec, starts, n_samp, margin, *, do_corr=True, do_psd=True,
          lag_pairs=None, max_lag=5, return_in_uV=True):
    """Accumulate correlation, PSD, RMS and cross-correlation lags in one pass.

    Chunks are processed one at a time and never all held in memory -- a 384-channel
    2 s chunk at 30 kHz is already ~90 MB.
    """
    fs = rec.get_sampling_frequency()
    n_ch = rec.get_num_channels()
    nperseg = min(_next_pow2_le(n_samp), 32768)

    corr_sum = np.zeros((n_ch, n_ch)) if do_corr else None
    psd_sum, freqs = None, None
    ss = np.zeros(n_ch)          # sum of squares, for RMS
    lags = np.arange(-max_lag, max_lag + 1)
    xc_sum = None
    if lag_pairs is not None:
        xc_sum = np.zeros((len(lags), len(lag_pairs[0])))

    n_used = 0
    for s in starts:
        x = _read_chunk(rec, s, n_samp, margin, return_in_uV=return_in_uV)
        if x.shape[0] < n_samp:
            continue
        n_used += 1
        ss += (x.astype(np.float64) ** 2).sum(axis=0)

        if do_corr:
            xc = x - x.mean(axis=0, keepdims=True)
            sd = xc.std(axis=0)
            sd[sd < 1e-12] = 1.0
            xc = xc / sd
            corr_sum += (xc.T @ xc) / x.shape[0]

        if do_psd:
            f, p = welch(x, fs=fs, axis=0, nperseg=nperseg)
            psd_sum = p if psd_sum is None else psd_sum + p
            freqs = f

        if lag_pairs is not None:
            ia, ib = lag_pairs
            a = x[:, ia] - x[:, ia].mean(axis=0, keepdims=True)
            b = x[:, ib] - x[:, ib].mean(axis=0, keepdims=True)
            for k, l in enumerate(lags):
                if l >= 0:
                    seg = (a[:n_samp - l] * b[l:]).mean(axis=0)
                else:
                    seg = (a[-l:] * b[:n_samp + l]).mean(axis=0)
                xc_sum[k] += seg

    if n_used == 0:
        raise ValueError('no full-length chunks could be read from the recording')

    out = {'n_chunks_used': n_used,
           'rms': np.sqrt(ss / (n_used * n_samp))}
    if do_corr:
        out['corr'] = corr_sum / n_used
    if do_psd:
        out['freqs'] = freqs
        out['psd'] = (psd_sum / n_used).T          # (n_ch, n_freq)
    if lag_pairs is not None:
        out['lags'] = lags
        out['xcorr'] = xc_sum / n_used
    return out


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def _distant_pair_corr(corr, locations, min_distance_um=200.0, shank_ids=None):
    """Median |r| over channel pairs far enough apart to share no real signal.

    Common-mode noise correlates distant channels; a working reference collapses this
    toward zero. Nearby pairs are excluded because they legitimately share spikes.

    Distance is the full electrode geometry, not depth alone. On a multi-shank probe two
    channels at the same depth on different shanks sit 250 um or more apart yet zero
    apart in y, so a depth-only test silently drops exactly the pairs that best expose
    shared common-mode noise: on this 4-shank NP2.0 that was 14460 of the 68908
    qualifying pairs, all of them cross-shank.

    With shank_ids the qualifying pairs are also split within- against cross-shank.
    That is the number that says whether one global reference is the right choice: a
    global median subtracts a signal common to all shanks, so it can only help where
    the cross-shank correlation is really there.

    Returns (median_abs_r, vals, groups).
    """
    loc = np.asarray(locations, dtype=float)
    if loc.ndim == 1:
        loc = loc[:, None]
    n = loc.shape[0]
    d = np.linalg.norm(loc[:, None, :] - loc[None, :, :], axis=2)
    iu = np.triu_indices(n, k=1)
    far = d[iu] >= min_distance_um
    vals = np.abs(corr[iu][far])

    groups = {}
    if shank_ids is not None:
        sh = np.asarray(shank_ids)
        cross = (sh[iu[0]] != sh[iu[1]])[far]
        for label, mask in (('within_shank', ~cross), ('cross_shank', cross)):
            v = vals[mask]
            groups[label] = {
                'median_abs_r': float(np.median(v)) if v.size else np.nan,
                'n_pairs': int(v.size),
            }
    if vals.size == 0:
        return np.nan, vals, groups
    return float(np.median(vals)), vals, groups


def _clipping_fraction(rec, starts, n_samp, margin, full_scan=True, verbose=True,
                       rail=None):
    """Fraction of samples at (and near) the amplifier rail, per channel.

    The rail is measured, never taken from the dtype. The amplifier saturates below ADC
    full scale, and the ADC is far narrower than the int16 container it ships in: on
    these NP2.0 recordings the clip sits at +/-1947 and +/-1963 counts while
    np.iinfo(int16).max is 32767. A dtype-derived threshold therefore never fires and
    reports a clipped session as perfectly clean -- which is what this function used to
    do. And since the clipped samples pile up at a single value, a threshold even 1%
    high catches none of them rather than most.

    rail, when supplied (e.g. from detect_saturation, which the pipeline has already
    run), skips the measurement pass.

    full_scan reads the entire recording rather than the sampled chunks. Saturation is
    a rare-event metric: a few milliseconds of amplifier recovery after an opto pulse
    will not land inside a handful of random 2 s chunks, and missing it is exactly the
    failure this metric exists to catch. The read is sequential and I/O bound.

    Returns (at_rail, near_rail, dtype, scanned_s, rail).
    """
    dtype = np.dtype(rec.get_dtype())
    if not np.issubdtype(dtype, np.integer):
        return None, None, dtype, 0.0, None
    n_ch = rec.get_num_channels()
    if rail is None:
        rail, _info = dsat.find_rail(rec, verbose=False)
    if rail is None:
        # find_rail probes the whole recording, so "no rail" means there is nothing
        # here for the expensive pass to count
        return np.zeros(n_ch), np.zeros(n_ch), dtype, 0.0, None
    near = 0.99 * rail
    at_rail = np.zeros(n_ch)
    near_rail = np.zeros(n_ch)
    n_tot = 0

    if full_scan:
        fs = rec.get_sampling_frequency()
        total = rec.get_num_samples()
        step = int(SATURATION_CHUNK_S * fs)
        n_steps = int(np.ceil(total / step))
        for i in range(n_steps):
            a, b = i * step, min((i + 1) * step, total)
            x = rec.get_traces(start_frame=a, end_frame=b, return_in_uV=False)
            at_rail += (np.abs(x) >= rail).sum(axis=0)
            near_rail += (np.abs(x) >= near).sum(axis=0)
            n_tot += x.shape[0]
            if verbose and (i % 50 == 0 or i == n_steps - 1):
                print(f'  saturation scan {i + 1}/{n_steps} chunks', end='\r')
        if verbose:
            print()
    else:
        for s in starts:
            x = _read_chunk(rec, s, n_samp, margin, return_in_uV=False)
            if x.shape[0] < n_samp:
                continue
            at_rail += (np.abs(x) >= rail).sum(axis=0)
            near_rail += (np.abs(x) >= near).sum(axis=0)
            n_tot += x.shape[0]

    if n_tot == 0:
        return None, None, dtype, 0.0, rail
    return (at_rail / n_tot, near_rail / n_tot, dtype,
            n_tot / rec.get_sampling_frequency(), rail)


def _psd_baseline(psd_med, freqs, window_hz=200.0):
    """Smooth broadband floor under a spectrum, as a running median.

    Wide enough that a narrowband line does not pull the floor up under itself, narrow
    enough to follow the real slope of the spectrum.
    """
    df = float(np.median(np.diff(freqs)))
    n = max(int(round(window_hz / max(df, 1e-9))) | 1, 3)   # odd
    return median_filter(psd_med, size=n, mode='nearest')


def _comb_extent(excess_at, spacing, fmax, min_excess_db, min_frac=0.5):
    """How far a comb of the given spacing actually extends.

    Returns the largest k for which at least min_frac of the harmonics 1..k are present.
    Without this a comb is claimed wherever its multiples happen to be dense: 60 Hz
    multiples are 60 Hz apart everywhere, so by 12 kHz a peak from any unrelated source
    lands within a couple of Hz of one and gets called the 200th harmonic. On this
    recording the real 60 Hz comb stops at k=11.
    """
    k_max = int(fmax // spacing)
    best, hits = 0, 0
    for k in range(1, k_max + 1):
        if excess_at(k * spacing) > min_excess_db:
            hits += 1
        if hits >= min_frac * k:
            best = k
    return best


def _classify_peak(f0, combs, fs, tol_hz=2.0):
    """Every family a peak belongs to, as a list; empty means unexplained.

    Peaks can belong to more than one family and saying so is more honest than picking:
    on this probe 6000 Hz is both a member of the 1 kHz comb and exactly fs/5, and there
    is no evidence here for preferring one story.
    """
    labels = []
    for N in range(2, 21):
        if abs(f0 - fs / N) <= tol_hz:
            labels.append(f'fs/{N}')
            break
    for spacing, extent, name in combs:
        if spacing <= 0 or extent <= 0:
            continue
        k = round(f0 / spacing)
        if 1 <= k <= extent and abs(f0 - k * spacing) <= tol_hz:
            labels.append(f'{name} x{k}')
    return labels


def _modulation_series(offsets, tol_hz=6.0):
    """Describe sideband offsets as harmonics of a fundamental where they fit.

    Sidebands from a single modulating frequency land at multiples of it, so reporting
    "+/-56 Hz, harmonics 1-3" says in one line what four separate rows say badly.
    Offsets that do not fit the series are returned separately rather than forced into
    it -- around fs/3 there is a clean 56 Hz series plus one offset at 151 Hz that
    belongs to something else.
    """
    offsets = sorted(float(o) for o in offsets)
    if not offsets:
        return None, [], []
    base = offsets[0]
    fit_k, extra = [], []
    for o in offsets:
        k = max(int(round(o / base)), 1)
        if abs(o - k * base) <= tol_hz:
            fit_k.append(k)
        else:
            extra.append(o)
    return base, sorted(set(fit_k)), extra


def _families(peak):
    """The comb/spur family names a peak belongs to, without the harmonic number."""
    return {k.split(' x')[0] for k in peak.get('kinds', ())}


def group_sidebands(peaks, max_offset_hz=500.0, tol_hz=6.0, margin_db=2.0):
    """Attach symmetric sideband pairs to the carrier they sit around.

    A pair of peaks equidistant either side of a stronger peak is modulation of that
    carrier, not two independent sources. Saying so collapses the report enormously: on
    DS45 fourteen of the twenty otherwise-unexplained peaks are sidebands of just three
    carriers.

    Only symmetric pairs count. A single neighbour on one side is left alone, since one
    offset is equally consistent with an unrelated tone that happens to be nearby.

    Two rules keep this from finding modulation everywhere:

    A member of a comb is not a carrier for its own neighbours. Comb members are equally
    spaced by construction, so every one of them looks like a carrier flanked by
    symmetric sidebands -- the 60 Hz comb would report its 4th harmonic as a carrier
    modulated at 60 Hz. A pair is rejected when carrier and both flanks share a family.

    A flank may be marginally stronger than its carrier (margin_db). Requiring the
    carrier to be strictly strongest broke a real triplet at 11523/11667/11812 Hz, where
    the lower sideband measured 0.2 dB above the carrier -- well inside the scatter of a
    Welch estimate off a dozen chunks.

    Annotates the peak dicts in place and returns them.
    """
    order = sorted(range(len(peaks)), key=lambda i: -peaks[i]['excess_db'])
    taken = set()
    for i in order:
        if i in taken:
            continue
        c = peaks[i]
        c_fam = _families(c)
        cand = [j for j in range(len(peaks))
                if j != i and j not in taken
                and abs(peaks[j]['freq'] - c['freq']) <= max_offset_hz
                and peaks[j]['excess_db'] < c['excess_db'] + margin_db]
        lows = [j for j in cand if peaks[j]['freq'] < c['freq']]
        highs = [j for j in cand if peaks[j]['freq'] > c['freq']]
        offsets = []
        for a in lows:
            da = c['freq'] - peaks[a]['freq']
            for b in highs:
                db = peaks[b]['freq'] - c['freq']
                if abs(da - db) <= tol_hz and b not in taken and a not in taken:
                    if c_fam and (_families(peaks[a]) & c_fam) \
                            and (_families(peaks[b]) & c_fam):
                        continue          # same comb, not modulation
                    off = 0.5 * (da + db)
                    for j, sign in ((a, -1), (b, +1)):
                        peaks[j]['role'] = 'sideband'
                        peaks[j]['carrier_hz'] = c['freq']
                        peaks[j]['offset_hz'] = sign * off
                        taken.add(j)
                    offsets.append(off)
                    break
        if offsets:
            base, ks, extra = _modulation_series(offsets, tol_hz)
            c['role'] = 'carrier'
            c['sidebands'] = {'n': 2 * len(offsets), 'fundamental_hz': base,
                              'harmonics': ks, 'extra_offsets_hz': extra}
            taken.add(i)
    for p in peaks:
        p.setdefault('role', 'isolated')
    return peaks


def find_spectral_peaks(freqs, psd, fs, line_freq=LINE_FREQ, min_excess_db=LINE_FLAG_DB,
                        window_hz=200.0, max_peaks=40, fmin=20.0):
    """Narrowband peaks anywhere in the spectrum, with their absolute power.

    Replaces "measure the height of every multiple of 60 Hz", which could only ever
    report multiples of 60 and so mislabelled every spur as line noise.

    Each peak carries its absolute PSD as well as its height over the local floor. The
    height alone cannot show attenuation: filtering a region pushes a peak and the floor
    beneath it down together, so the ratio barely moves while the power falls by orders
    of magnitude. The absolute number is what makes a before/after comparison mean
    something.
    """
    med = np.asarray(psd).mean(axis=0) if np.asarray(psd).ndim == 2 else np.asarray(psd)
    base = _psd_baseline(med, freqs, window_hz)
    with np.errstate(divide='ignore', invalid='ignore'):
        excess = 10.0 * np.log10(np.maximum(med, 1e-300) / np.maximum(base, 1e-300))
    excess[~np.isfinite(excess)] = 0.0
    excess[freqs < fmin] = 0.0

    df = float(np.median(np.diff(freqs)))
    idx, _ = find_peaks(excess, height=min_excess_db,
                        distance=max(int(round(10.0 / max(df, 1e-9))), 1))
    idx = idx[np.argsort(-excess[idx])][:max_peaks]

    def excess_at(f0, tol=2.0):
        m = np.abs(freqs - f0) <= tol
        return float(excess[m].max()) if m.any() else -np.inf

    # Which combs are actually present, and how far each really runs. Both spacings are
    # worth testing on this rig: mains at 60 Hz, and a 1 kHz comb from switching or
    # clocking that on DS45 is the stronger of the two above ~4 kHz.
    fmax = float(freqs[-1])
    combs = []
    for spacing, name in ((line_freq, f'{line_freq:g}Hz'), (1000.0, '1kHz')):
        extent = _comb_extent(excess_at, spacing, fmax, min_excess_db)
        if extent >= 3:                     # three harmonics before calling it a comb
            combs.append((spacing, extent, name))

    peaks = [{'freq': float(freqs[i]),
              'excess_db': float(excess[i]),
              'psd': float(med[i]),
              'baseline_psd': float(base[i]),
              'kinds': _classify_peak(float(freqs[i]), combs, fs)}
             for i in idx]
    peaks.sort(key=lambda p: p['freq'])
    group_sidebands(peaks)
    combs_out = [{'spacing_hz': s, 'extent_k': int(k), 'name': n} for s, k, n in combs]
    return peaks, med, base, combs_out


def _line_harmonics(freqs, psd, line_freq=LINE_FREQ, peak_halfwidth_hz=1.5,
                    baseline_hz=(5.0, 15.0), max_harmonic_frac=0.9):
    """Power at each multiple of line_freq relative to the local broadband baseline.

    Returns (harmonic_freqs, ratio_db) with ratio_db shaped (n_ch, n_harmonics).
    The baseline is the median of two flanking bands, so a harmonic riding on a sloped
    spectrum is still measured against the floor immediately around it.
    """
    nyq = freqs[-1]
    ks = np.arange(1, int(nyq * max_harmonic_frac / line_freq) + 1)
    hf = ks * line_freq
    ratio_db = np.full((psd.shape[0], len(hf)), np.nan)
    for j, f0 in enumerate(hf):
        peak = np.abs(freqs - f0) <= peak_halfwidth_hz
        base = (((freqs - f0) >= baseline_hz[0]) & ((freqs - f0) <= baseline_hz[1])) | \
               (((f0 - freqs) >= baseline_hz[0]) & ((f0 - freqs) <= baseline_hz[1]))
        if not peak.any() or not base.any():
            continue
        pk = psd[:, peak].max(axis=1)
        bl = np.median(psd[:, base], axis=1)
        bl = np.where(bl <= 0, np.nan, bl)
        ratio_db[:, j] = 10.0 * np.log10(pk / bl)
    return hf, ratio_db


def _peak_lags(xcorr, lags):
    """Sub-sample peak lag per pair, by parabolic interpolation around the argmax."""
    k = np.argmax(xcorr, axis=0)
    out = lags[k].astype(float)
    interior = (k > 0) & (k < len(lags) - 1)
    if interior.any():
        idx = np.flatnonzero(interior)
        y0 = xcorr[k[idx] - 1, idx]
        y1 = xcorr[k[idx], idx]
        y2 = xcorr[k[idx] + 1, idx]
        denom = y0 - 2 * y1 + y2
        shift = np.where(np.abs(denom) > 1e-20, 0.5 * (y0 - y2) / np.where(denom == 0, 1, denom), 0.0)
        out[idx] += np.clip(shift, -1.0, 1.0)
    return out


def _label_contiguity(labels, depths, target='out'):
    """How many separate runs the target label forms, walking channels in depth order.

    Out-of-brain channels are physically contiguous at one end of the shank; a label
    that is scattered into many runs means the detector is being fooled.
    """
    order = np.argsort(depths, kind='stable')
    mask = (np.asarray(labels)[order] == target)
    n = int(mask.sum())
    if n == 0:
        return {'n': 0, 'n_runs': 0, 'contiguous': True, 'at_shank_end': True}
    runs = int(np.sum(mask[1:] & ~mask[:-1])) + int(mask[0])
    at_end = bool(mask[-1] or mask[0])
    return {'n': n, 'n_runs': runs, 'contiguous': runs == 1, 'at_shank_end': at_end}


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------

def _save(fig, folder, name, attempts=4, delay_s=2.0):
    """Write a figure, retrying a transient OSError before giving up.

    This is a mitigation, not a root cause. savefig here intermittently fails with
    OSError EINVAL out of builtins.open, on a path that is valid and writable, and it
    is bursty rather than tied to any one figure: one run hit four different plots and
    the very next run on the same session hit none. Every occurrence so far has
    recovered on the first retry. Ruled out by the diagnostics it prints -- the figure
    (1500x480 px, nothing degenerate), the path (89 characters, parent present, file
    already there), the backend (both Agg and Qt), disk space, memory, and the folder
    itself, which takes 40 consecutive writes of a comparable figure without complaint.
    Scattered across files and clustered in time, it points at another process holding
    a handle -- a virus scanner or the search indexer -- but that part is a guess. What
    is not a guess is that letting it propagate aborts an hour of pipeline work over
    one PNG.
    """
    path = folder / f'{name}.png'
    for attempt in range(attempts):
        try:
            fig.savefig(path, dpi=150, bbox_inches='tight')
            break
        except OSError as e:
            if attempt == attempts - 1:
                plt.close(fig)
                raise
            print(f'  qc: savefig {path.name} failed (errno {e.errno}), retrying '
                  f'{attempt + 1}/{attempts - 1}')
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            time.sleep(delay_s)
    plt.close(fig)
    return path


def _plot_noise(depths, series, folder, name='qc_noise_levels', unit='uV'):
    """RMS/MAD noise per channel vs depth, plus its distribution, for each stage."""
    fig, axs = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    for label, vals in series.items():
        axs[0].plot(vals, depths, '.', markersize=3, alpha=0.7, label=label)
        axs[1].hist(vals, bins=40, histtype='step', label=label)
    axs[0].set_xlabel(f'noise ({unit})')
    axs[0].set_ylabel('depth (um)')
    axs[0].set_title('per-channel noise vs depth')
    axs[0].legend(fontsize=7)
    axs[1].set_xlabel(f'noise ({unit})')
    axs[1].set_ylabel('channels')
    axs[1].set_title('distribution (spread = channel health)')
    axs[1].legend(fontsize=7)
    return _save(fig, folder, name)


def _select_events(n_samples, trig, half, max_events):
    """Triggers whose window fits inside the recording, thinned to max_events.

    Chosen once and reused for every stage, so the panels average the same events and
    any blanked-sample accounting refers to the windows actually drawn.

    max_events=None keeps every event. Thinning is evenly spaced rather than random or
    truncating, so a capped average still spans the whole session instead of its first
    minutes; the caller that wants all of them has to say so, because the cost is
    linear in events and each one is a scattered read of the whole array.
    """
    trig = trig[(trig > half) & (trig < n_samples - half)]
    if max_events is not None and trig.size > max_events:
        trig = trig[np.linspace(0, trig.size - 1, max_events).astype(int)]
    return trig


def _blanked_fraction(trig, half, periods):
    """Share of the averaged window that silence_periods has replaced with fill.

    silence_periods blanks the amplifier-saturation epochs, and on an opto session
    those epochs are caused by the very triggers this plot averages over. When they
    overlap, a flat post-chain panel means the samples were overwritten, not that the
    chain cleaned them -- the opposite reading of the same picture. Periods are merged
    and non-overlapping (see detect_saturation), so the ones touching a window are a
    contiguous run in start order.
    """
    p = np.asarray(periods, dtype=np.int64).reshape(-1, 2)
    if p.size == 0 or trig.size == 0:
        return 0.0
    p = p[np.argsort(p[:, 0])]
    starts, ends = p[:, 0], p[:, 1]
    a, b = trig - half, trig + half
    lo = np.searchsorted(ends, a, side='right')
    hi = np.searchsorted(starts, b, side='left')
    total = 0
    for k in range(len(trig)):
        if hi[k] > lo[k]:
            sl = slice(lo[k], hi[k])
            total += int(np.sum(np.minimum(ends[sl], b[k])
                                - np.maximum(starts[sl], a[k])))
    return float(total) / float(trig.size * 2 * half)


def _event_average(rec, trig, half, baseline, in_uV=True):
    """Mean trace across artifact events, per channel, baseline-subtracted.

    Averaging over events is the whole trick: the artifact is phase-locked to the
    trigger and the neural signal is not, so n events of averaging suppresses
    everything except what the trigger causes. A residual that survives here is real.
    """
    acc = None
    for t in trig:
        x = rec.get_traces(start_frame=int(t) - half, end_frame=int(t) + half,
                           return_in_uV=in_uV).astype(np.float64)
        x -= x[baseline].mean(axis=0, keepdims=True)
        acc = x if acc is None else acc + x
    return acc / trig.size if acc is not None else None


def _draw_event_panel(ax, t_ms, a, unit, per_channel=True, shank_ids=None,
                      channel_alpha=CHANNEL_ALPHA):
    """One event-average panel: the per-channel traces behind the grand average in black.

    Returns the grand average, which is what the metrics are taken from. The per-channel
    lines are the spread the black line is hiding -- a grand average near zero over
    channels that individually swing hundreds of uV means the event moves the array
    incoherently, which is a different fact from the event doing nothing.

    shank_ids colours those lines by shank instead of leaving them grey, which answers
    the question the grey cannot: whether a deflection is the whole probe moving
    together or one shank doing something its neighbours are not. Only shanks are
    coloured, and only up to as many as the palette defines -- past that the lines go
    back to grey rather than reusing a hue for two shanks, since a repeated colour would
    assert an identity that is not there.
    """
    grand = a.mean(axis=1)
    if per_channel:
        groups = ()
        if shank_ids is not None:
            sh = np.asarray(shank_ids)
            if sh.size == a.shape[1]:
                uniq = np.unique(sh)
                if 1 < uniq.size <= len(SHANK_COLORS):
                    groups = tuple((str(u), np.flatnonzero(sh == u)) for u in uniq)
        if groups:
            for (label, cols), color in zip(groups, SHANK_COLORS):
                ax.plot(t_ms, a[:, cols], color=color, lw=0.3, alpha=channel_alpha,
                        zorder=1)
                # the legend entry is a proxy drawn at full weight, not one of the
                # traces themselves: a 0.3 pt line at 45% alpha is a legible trace in a
                # bundle of a hundred and an illegible swatch on its own, and with every
                # hue under 3:1 against white the swatch is what carries the identity
                ax.plot([], [], color=color, lw=2.0,
                        label=f'shank {label} ({cols.size} ch)')
        else:
            ax.plot(t_ms, a, color='0.65', lw=0.3, alpha=channel_alpha, zorder=1)
    ax.plot(t_ms, grand, color='k', lw=1.8, zorder=3, label='average across channels')
    ax.set_xlabel('time from trigger (ms)')
    ax.set_ylabel(f'{unit} (baseline-subtracted)')
    return grand


def _panel_metrics(grand, fs, hp_hz, baseline, inwin=None):
    """Peak, swing and flat-trace floor of one event-average panel.

    The peak is taken above hp_hz, not off the trace as drawn. A raw average is
    dominated by a slow swing of several hundred uV that a local repair like
    remove_artifacts is not meant to touch and the bandpass deletes anyway; scoring
    the trace as plotted reports 0 dB of improvement on a transient that was in fact
    cleanly removed. The swing is kept alongside it, because for a slow, genuinely
    low-frequency event the high-passed number is the one that means nothing.

    The floor is how flat the average is where the trigger has no influence. Any
    trigger-locked structure has to clear it to count as a real effect.
    """
    n_box = max(int(round(fs / max(hp_hz, 1.0))), 3)
    hf = grand - uniform_filter1d(grand, size=n_box, mode='nearest')
    out = {'peak_uV': float(np.abs(hf).max()),
           'swing_uV': float(grand.max() - grand.min()),
           'hf_floor_uV': float(np.abs(hf[baseline]).max())}
    if inwin is not None:
        out['peak_outside_window_uV'] = float(np.abs(hf[~inwin]).max())
    return out


def _plot_artifact_removal(raw_rec, cleaned_rec, triggers, folder,
                           name='qc_artifact_removal', ms_window=5.0,
                           ms_before=0.5, ms_after=0.5, max_events=500,
                           per_channel=True, hp_hz=300.0, final_rec=None,
                           silenced_periods=None, channel_alpha=CHANNEL_ALPHA):
    """Event-averaged trace around the triggers, at each stage of the chain.

    Two panels always: the raw recording, and the raw recording with remove_artifacts
    applied on its own, so they differ by that one step and nothing downstream of it.
    That is the controlled test of whether the step worked.

    final_rec adds a third panel, the trigger-locked average of the fully preprocessed
    recording. It is deliberately *not* a controlled comparison -- it differs from the
    second panel by the whole rest of the chain, not by one step -- and it answers a
    different question: what trigger-locked signal actually reaches the sorter, after
    the bandpass has deleted the slow swing and the reference has taken out whatever
    of the residual was common to every channel.

    That panel has one reading that has to be ruled out before it can be believed, and
    silenced_periods is what rules it out. If the chain contains silence_periods, its
    epochs are the amplifier-saturation spans, which on an opto session are caused by
    these same triggers -- so a flat third panel can mean the samples were replaced
    with fill rather than that anything was cleaned. Passing the periods puts the
    overlap in the panel title and in the stats instead of leaving it to be guessed.

    Panels carry their own y-scale on purpose. Sharing one would render the later
    panels as flat lines at zero, which looks like success but hides the question the
    plot exists to answer: what is left *outside* the interpolated window. A window
    that is too narrow shows up as ringing either side of the shaded span, and that
    number is reported separately from the in-window peak.

    The same hp_hz box-filter metric is used on every panel, including the one that is
    already bandpassed (where it is close to a no-op). Scoring the stages differently
    would make the three numbers incomparable, which is the only reason to print them
    side by side.
    """
    fs = raw_rec.get_sampling_frequency()
    half = int(round(ms_window * fs / 1000.0))
    trig = np.asarray(triggers, dtype=np.int64).ravel()
    t_ms = (np.arange(-half, half) / fs) * 1000.0
    # baseline well clear of the interpolated span, so the DC offset of each raw
    # channel does not scatter the grey lines over the whole axis
    baseline = t_ms < -max(2.0 * ms_before, 1.0)
    if not baseline.any():
        baseline = t_ms < t_ms[len(t_ms) // 4]

    stages = [('before', raw_rec), ('after', cleaned_rec)]
    skipped = None
    if final_rec is not None:
        # the window width and the trigger indices are both in raw frames, so a chain
        # that resamples or slices would index the wrong samples of final_rec without
        # raising. Nothing in the current protocol does, but the check is three lines
        # and the failure would be a plausible-looking wrong panel.
        if (final_rec.get_sampling_frequency() != fs
                or final_rec.get_num_samples() != raw_rec.get_num_samples()):
            skipped = (f'{final_rec.get_sampling_frequency():g} Hz / '
                       f'{final_rec.get_num_samples()} samples vs raw '
                       f'{fs:g} Hz / {raw_rec.get_num_samples()}')
        else:
            stages.append((FINAL_STAGE, final_rec))

    # one event selection for every stage: the guard above has already established that
    # the recordings share a frame base, so the panels average the same windows
    sel = _select_events(raw_rec.get_num_samples(), trig, half, max_events)
    if sel.size == 0:
        return None, {'error': 'no usable triggers'}
    blanked = (_blanked_fraction(sel, half, silenced_periods)
               if silenced_periods is not None else None)

    avgs, units = {}, {}
    for label, rec in stages:
        # preprocessed chains sometimes drop the gains, and then the averages are in
        # whatever the chain left behind rather than in uV
        units[label] = 'uV' if _traces_in_uV(rec) else 'a.u.'
        avgs[label] = _event_average(rec, sel, half, baseline,
                                     in_uV=(units[label] == 'uV'))

    inwin = (t_ms >= -ms_before) & (t_ms <= ms_after)
    n_panels = len(stages) + (1 if skipped else 0)
    stats = {'n_events': int(sel.size), 'ms_window': ms_window,
             'ms_before': ms_before, 'ms_after': ms_after,
             'metric_hp_hz': float(hp_hz), 'units': units}
    if blanked is not None:
        stats['silenced_fraction_of_window'] = blanked
    fig, axs = plt.subplots(1, n_panels, figsize=(6.5 * n_panels, 4.6), sharex=True,
                            constrained_layout=True, squeeze=False)
    axs = axs[0]
    for ax, (label, rec) in zip(axs, stages):
        a = avgs[label]
        key = 'final' if label == FINAL_STAGE else label
        unit = units[label]
        # each stage's own shank ids: the chain can drop channels, so the raw probe's
        # would not line up with a later stage's columns
        grand = _draw_event_panel(ax, t_ms, a, unit, per_channel, _shank_ids(rec),
                                  channel_alpha)
        ax.axvspan(-ms_before, ms_after, color='tab:orange', alpha=0.12, zorder=0,
                   label=f'interpolated (-{ms_before:g} to +{ms_after:g} ms)')
        m = _panel_metrics(grand, fs, hp_hz, baseline, inwin)
        pk, pk_out = m['peak_uV'], m['peak_outside_window_uV']
        stats[f'peak_{key}_uV'] = pk
        stats[f'peak_{key}_outside_window_uV'] = pk_out
        stats[f'swing_{key}_uV'] = m['swing_uV']
        stats[f'hf_floor_{key}_uV'] = m['hf_floor_uV']
        title = label if key == 'final' else f'{label} remove_artifacts'
        # only the post-chain panel can contain silenced samples: silence_periods sits
        # downstream of the step the first two panels compare
        note = (f'\n{100 * blanked:.0f}% of the window is silence_periods fill'
                if key == 'final' and blanked else '')
        ax.set_title(f'{title} ({a.shape[1]} ch)\n'
                     f'peak |mean| above {hp_hz:g} Hz: {pk:.1f} {unit}   '
                     f'(outside the window {pk_out:.1f} {unit}){note}')
        ax.legend(fontsize=7, loc='upper right')
    if skipped:
        ax = axs[-1]
        ax.text(0.5, 0.5, 'fully preprocessed panel skipped:\nframe base differs from '
                f'raw\n{skipped}', transform=ax.transAxes, ha='center', va='center',
                color='grey', fontsize=8)
        stats['final_panel_skipped'] = skipped
    # the channel count lives in each panel title: the chain can drop channels, so the
    # stages do not all average the same number of them
    fig.suptitle(f'artifact-triggered average, {sel.size} events', fontsize=9)
    before_pk = stats['peak_before_uV']
    # before -> after only: the two panels that differ by remove_artifacts alone. The
    # fully preprocessed panel is not a step-for-step comparison, so a dB number
    # against it would be attributing the whole chain to this one step.
    stats['reduction_db'] = float(10.0 * np.log10(
        max(stats['peak_after_uV'], 1e-12) ** 2 / max(before_pk, 1e-12) ** 2))
    return _save(fig, folder, name), stats


def _plot_event_average(raw_rec, final_rec, triggers, folder, event_name,
                        name=None, ms_window=200.0, max_events=None,
                        per_channel=True, hp_hz=300.0, silenced_periods=None,
                        channel_alpha=CHANNEL_ALPHA, marks=()):
    """Event-triggered average before and after the whole chain, for a behavioural event.

    Two panels, raw and fully preprocessed, each drawing every channel's trial-average
    in grey with the channel-and-trial average in black. Unlike the artifact plot this
    is not testing one step: it asks whether the event leaves a trigger-locked
    signature in the traces at all, and whether the chain removes, keeps or creates
    one. A lick is a jaw movement against a headpost, so the honest prior is that it
    does put something in the raw traces; the question is what survives to the sorter.

    ms_window is a *half* window and defaults far wider than the artifact plot's,
    because a behavioural event is slow. +/-200 ms covers a full in-bout interval:
    parse_lick_bouts calls anything under 300 ms one bout and a bout runs near 7 Hz, so
    the neighbouring licks land around +/-143 ms and are inside the window. Seeing them
    is the point -- a rhythmic average says the signature is the licking itself rather
    than one stray movement.

    max_events defaults to None -- every event. That is the right default for a
    behavioural average, where events number in the thousands and thinning would throw
    away the averaging that makes the plot readable. Each event is a scattered read of
    the full window on both recordings, so the count and the duration are printed
    before the loop rather than discovered from the wall clock.

    Read the black line against the grey. A grand average near zero over channels that
    individually swing hundreds of uV means the event moves the array incoherently,
    which is a different fact from the event doing nothing.

    marks: (ms, label) pairs drawn as dashed verticals on every panel, for a time
    inside the window that means something other than the trigger itself.
    """
    fs = raw_rec.get_sampling_frequency()
    half = int(round(ms_window * fs / 1000.0))
    trig = np.asarray(triggers, dtype=np.int64).ravel()
    t_ms = (np.arange(-half, half) / fs) * 1000.0
    # the leading fifth of the window. There is no span that is clean by construction
    # here -- the event is behavioural, not imposed -- so the baseline is simply far
    # enough ahead of it to take the per-channel DC offset off the grey lines.
    baseline = t_ms < t_ms[0] + 0.2 * (t_ms[-1] - t_ms[0])

    stages = [('raw', raw_rec)]
    if (final_rec.get_sampling_frequency() != fs
            or final_rec.get_num_samples() != raw_rec.get_num_samples()):
        # event frames are raw frames; a chain that resampled or sliced would index
        # the wrong samples of final_rec without raising
        return None, {'error': 'final recording does not share the raw frame base',
                      'event': event_name}
    stages.append((FINAL_STAGE, final_rec))

    sel = _select_events(raw_rec.get_num_samples(), trig, half, max_events)
    if sel.size == 0:
        return None, {'error': f'no usable {event_name} events', 'event': event_name}
    # the filtered stage reads a little more than this -- each get_traces pads by the
    # chain's filter margins -- but with no long-impulse step in the chain the padding
    # is a few ms against a window of hundreds
    print(f'  qc: {event_name}-triggered average over {sel.size} events '
          f'(+/-{ms_window:g} ms, {sel.size * 2 * half / fs:.0f} s of traces per stage)')

    blanked = (_blanked_fraction(sel, half, silenced_periods)
               if silenced_periods is not None else None)

    stats = {'event': event_name, 'n_events': int(sel.size), 'ms_window': ms_window,
             'n_events_total': int(trig.size), 'metric_hp_hz': float(hp_hz),
             'units': {}}
    if blanked is not None:
        stats['silenced_fraction_of_window'] = blanked

    fig, axs = plt.subplots(1, len(stages), figsize=(6.5 * len(stages), 4.6),
                            sharex=True, constrained_layout=True, squeeze=False)
    for ax, (label, rec) in zip(axs[0], stages):
        unit = 'uV' if _traces_in_uV(rec) else 'a.u.'
        stats['units'][label] = unit
        a = _event_average(rec, sel, half, baseline, in_uV=(unit == 'uV'))
        key = 'final' if label == FINAL_STAGE else label
        grand = _draw_event_panel(ax, t_ms, a, unit, per_channel, _shank_ids(rec),
                                  channel_alpha)
        for ms, mark_label in marks:
            ax.axvline(ms, color='0.3', ls='--', lw=1.0, zorder=2, label=mark_label)
        m = _panel_metrics(grand, fs, hp_hz, baseline)
        stats[f'peak_{key}_uV'] = m['peak_uV']
        stats[f'swing_{key}_uV'] = m['swing_uV']
        stats[f'hf_floor_{key}_uV'] = m['hf_floor_uV']
        # the swing leads, not the high-passed peak: a lick-triggered average is
        # mostly the slow component that hp_hz is designed to strip, so scoring this
        # panel the way the artifact panel is scored would report a few uV about a
        # trace visibly swinging hundreds. The hp number stays for comparability.
        note = (f'   [{100 * blanked:.0f}% silence_periods fill]'
                if key == 'final' and blanked else '')
        ax.set_title(f'{label} ({a.shape[1]} ch)\n'
                     f'swing of the mean: {m["swing_uV"]:.1f} {unit}   '
                     f'(above {hp_hz:g} Hz: {m["peak_uV"]:.1f} {unit}){note}')
        ax.legend(fontsize=7, loc='upper right')
    fig.suptitle(f'{event_name}-triggered average, {sel.size} of '
                 f'{trig.size} events', fontsize=9)
    return _save(fig, folder, name or f'qc_event_average_{event_name}'), stats


def _plot_bad_channels(depths, labels, manual_ids, channel_ids, folder,
                       name='qc_bad_channels'):
    """Auto labels laid out along the shank, with manually flagged channels marked."""
    cats = ['good', 'dead', 'noise', 'out']
    colors = {'good': '0.8', 'dead': 'tab:blue', 'noise': 'tab:red', 'out': 'tab:orange'}
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for c in cats:
        m = np.asarray(labels) == c
        if m.any():
            ax.plot(np.zeros(m.sum()) + cats.index(c), depths[m], 'o', markersize=3,
                    color=colors[c], label=f'{c} (n={int(m.sum())})')
    if manual_ids is not None and len(manual_ids):
        mm = np.isin(np.asarray(channel_ids), np.asarray(manual_ids))
        if mm.any():
            ax.plot(np.full(mm.sum(), len(cats)), depths[mm], 'x', markersize=5,
                    color='k', label=f'manual (n={int(mm.sum())})')
    ax.set_xticks(range(len(cats) + 1))
    ax.set_xticklabels(cats + ['manual'])
    ax.set_ylabel('depth (um)')
    ax.set_title('bad-channel labels along the shank')
    ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=False)
    return _save(fig, folder, name)


def _plot_saturation(depths, at_rail, near_rail, folder, name='qc_saturation'):
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    ax.plot(np.maximum(near_rail, 1e-9), depths, '.', markersize=4, color='tab:orange',
            label='>=99% full scale')
    ax.plot(np.maximum(at_rail, 1e-9), depths, '.', markersize=4, color='tab:red',
            label='at rail')
    ax.set_xscale('log')
    ax.set_xlabel('fraction of samples')
    ax.set_ylabel('depth (um)')
    ax.set_title(f'clipping (any nonzero at-rail is a red flag)\n'
                 f'total at rail: {at_rail.sum():.3g}')
    ax.legend(fontsize=7)
    return _save(fig, folder, name)


def _plot_correlation(corrs, dist_vals, folder, name='qc_channel_correlation',
                      min_distance_um=200.0):
    """Correlation matrices side by side, plus the distant-pair |r| distributions."""
    n = len(corrs)
    fig, axs = plt.subplots(1, n + 1, figsize=(4.2 * (n + 1), 4), constrained_layout=True)
    axs = np.atleast_1d(axs)
    for ax, (label, c) in zip(axs, corrs.items()):
        im = ax.imshow(c, vmin=-1, vmax=1, cmap='RdBu_r', origin='lower')
        ax.set_title(f'{label}\nchannel x channel r')
        ax.set_xlabel('channel')
        ax.set_ylabel('channel')
        fig.colorbar(im, ax=ax, fraction=0.046)
    ax = axs[-1]
    for label, vals in dist_vals.items():
        if vals.size:
            ax.hist(vals, bins=50, histtype='step',
                    label=f'{label} (median {np.median(vals):.3f})')
    ax.set_xlabel('|r|')
    ax.set_ylabel('pairs')
    ax.set_title(f'distant pairs (>= {min_distance_um:g} um apart)\n'
                 'should collapse toward 0 after referencing')
    ax.legend(fontsize=7)
    return _save(fig, folder, name)


def _plot_psd_depth(freqs, psds, depths, folder, name='qc_psd_depth', fmax=None):
    """Welch PSD as a depth x frequency heatmap, one panel per stage.

    depths is keyed by stage, not shared: a chain containing
    detect_and_remove_bad_channels emits fewer channels than it took in, and indexing
    the raw PSD with the preprocessed depth order would silently draw the wrong
    channels at the wrong depths rather than raise.
    """
    n = len(psds)
    fig, axs = plt.subplots(1, n, figsize=(5.5 * n, 4.2), constrained_layout=True,
                            squeeze=False)
    for ax, (label, psd) in zip(axs[0], psds.items()):
        f = freqs[label]
        d = depths[label]
        if psd.shape[0] != len(d):
            ax.text(0.5, 0.5, f'{label}: {psd.shape[0]} channels but {len(d)} depths',
                    transform=ax.transAxes, ha='center', va='center', color='grey')
            continue
        order = np.argsort(d)
        m = np.ones(len(f), dtype=bool) if fmax is None else (f <= fmax)
        with np.errstate(divide='ignore'):
            db = 10 * np.log10(np.maximum(psd[order][:, m], 1e-20))
        im = ax.pcolormesh(f[m], d[order], db, shading='auto', cmap='viridis')
        ax.set_xlabel('frequency (Hz)')
        ax.set_ylabel('depth (um)')
        ax.set_title(f'{label} PSD ({psd.shape[0]} ch)')
        fig.colorbar(im, ax=ax, label='dB', fraction=0.046)
    return _save(fig, folder, name)


def _plot_spectral_peaks(freqs, psds, peaks, folder, passband=None,
                         name='qc_spectral_peaks', combs=None):
    """Absolute spectra before and after, with the detected peaks named.

    Absolute, because the height of a peak over its own local floor cannot show what
    filtering did to it: attenuating a region pushes the peak and the floor beneath it
    down together, so the ratio barely moves while the power falls by orders of
    magnitude. On DS45 that reported a 231x reduction of the 6 kHz spur as +1 dB of
    growth. Plotting the spectra themselves, and the per-peak change, cannot hide it.
    """
    fig, axs = plt.subplots(1, 2, figsize=(15, 4.8), constrained_layout=True)

    ax = axs[0]
    for label, psd in psds.items():
        ax.semilogy(freqs, np.maximum(psd, 1e-12), lw=0.7, label=label)
    if passband is not None:
        ax.axvspan(passband[0], passband[1], color='tab:green', alpha=0.10,
                   label=f'passband {passband[0]:g}-{passband[1]:g} Hz')
    for p in peaks:
        ax.axvline(p['freq'], color='0.6', lw=0.4, zorder=0)
    # name only the strongest few, or the labels collide
    for p in sorted(peaks, key=lambda q: -q['excess_db'])[:8]:
        j = int(np.argmin(np.abs(freqs - p['freq'])))
        top = max(psd_v[j] for psd_v in psds.values())
        ax.annotate(','.join(p['kinds']) or f"{p['freq']:.0f}Hz",
                    xy=(p['freq'], top), fontsize=6, rotation=90,
                    ha='center', va='bottom')
    ax.set_xlabel('frequency (Hz)')
    ax.set_ylabel('PSD (absolute)')
    ax.set_title('spectrum before and after preprocessing\n'
                 'absolute power, so attenuation is visible')
    ax.legend(fontsize=7)

    ax = axs[1]
    if peaks and 'attenuation_db' in peaks[0]:
        fr = np.array([p['freq'] for p in peaks])
        at = np.array([p['attenuation_db'] for p in peaks])
        inb = np.array([bool(p.get('in_passband')) for p in peaks])
        ax.scatter(fr[~inb], at[~inb], s=22, color='0.5', label='outside passband')
        if inb.any():
            ax.scatter(fr[inb], at[inb], s=34, color='tab:red',
                       label='inside passband -- reaches the sorter')
        for p in sorted(peaks, key=lambda q: q['attenuation_db'])[:5]:
            ax.annotate(','.join(p['kinds']) or f"{p['freq']:.0f}",
                        xy=(p['freq'], p['attenuation_db']), fontsize=6,
                        xytext=(3, 3), textcoords='offset points')
        ax.axhline(0, color='k', lw=0.8)
        ax.set_ylabel('change at the peak (dB)')
        ax.set_title('what preprocessing did to each peak\n(negative = attenuated)')
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, 'no before/after comparison available',
                transform=ax.transAxes, ha='center', va='center', color='grey')
    if passband is not None:
        ax.axvspan(passband[0], passband[1], color='tab:green', alpha=0.10)
    ax.set_xlabel('frequency (Hz)')
    if combs:
        fig.suptitle('combs found: ' + ', '.join(
            f"{c['name']} to x{c['extent_k']}" for c in combs), fontsize=8)
    return _save(fig, folder, name)


def _plot_line_noise(harm, ratios_db, folder, passband=None,
                     name='qc_line_noise', line_freq=LINE_FREQ):
    """60 Hz and harmonics: median excess over baseline, and the depth breakdown.

    The right panel is the preprocessed stage only -- harmonics that fall inside the
    sorting passband are the ones that actually reach the sorter.
    """
    # Harmonics are computed all the way to Nyquist, but plotting 200+ of them buries
    # the handful that matter in the left-hand pixel. Show the passband plus whatever is
    # flagged above it; the full set stays in the JSON.
    hi = 0.0
    for label, rdb in ratios_db.items():
        med = np.nanmedian(rdb, axis=0)
        flag = harm[label][np.isfinite(med) & (med > LINE_FLAG_DB)]
        if len(flag):
            hi = max(hi, float(flag.max()))
    xmax = max(hi * 1.3, (passband[1] * 1.2 if passband is not None else 0.0),
               10 * line_freq)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    ax = axs[0]
    for label, rdb in ratios_db.items():
        med = np.nanmedian(rdb, axis=0)
        m = harm[label] <= xmax
        ax.plot(harm[label][m], med[m], 'o-', markersize=3, label=label)
    ax.axhline(LINE_FLAG_DB, color='k', linestyle='--', linewidth=0.8,
               label=f'{LINE_FLAG_DB:g} dB flag')
    ax.axhline(0, color='0.6', linewidth=0.8)
    if passband is not None:
        ax.axvspan(passband[0], passband[1], color='tab:green', alpha=0.10,
                   label=f'passband {passband[0]:g}-{passband[1]:g} Hz')
    ax.set_xlim(0, xmax)
    ax.set_xlabel(f'frequency (multiples of {line_freq:g} Hz)')
    ax.set_ylabel('peak over local baseline (dB)')
    ax.set_title('line noise and harmonics (median across channels)')
    ax.legend(fontsize=7)

    last = list(ratios_db)[-1]
    rdb = ratios_db[last]
    nshow = max(int(np.sum(harm[last] <= xmax)), 1)
    ax = axs[1]
    im = ax.imshow(rdb[:, :nshow], aspect='auto', origin='lower', cmap='magma',
                   extent=[0.5, nshow + 0.5, 0, rdb.shape[0]])
    ax.set_xlabel(f'harmonic number (x{line_freq:g} Hz)')
    ax.set_ylabel('channel')
    ax.set_title(f'{last}: per-channel harmonic excess')
    fig.colorbar(im, ax=ax, label='dB', fraction=0.046)
    return _save(fig, folder, name)


def _plot_adjacency_lag(lag_sets, folder, name='qc_adjacency_lag', stage=''):
    """Peak cross-correlation lag between depth-adjacent channels.

    Concentrated at 0 means the ADC multiplexing offset has been undone; the
    without-phase_shift trace is the control that shows the metric has teeth.
    """
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    bins = np.linspace(-3, 3, 61)
    for label, lags in lag_sets.items():
        ax.hist(lags, bins=bins, histtype='step',
                label=f'{label} (median |lag| {np.median(np.abs(lags)):.3f} samp)')
    ax.axvline(0, color='k', linewidth=0.8)
    ax.set_xlabel('peak cross-correlation lag (samples)')
    ax.set_ylabel('adjacent channel pairs')
    ax.set_title(f'zero-lag adjacency (tests phase_shift){f" -- {stage}" if stage else ""}')
    ax.legend(fontsize=7)
    return _save(fig, folder, name)


# ---------------------------------------------------------------------------
# public routines
# ---------------------------------------------------------------------------

def validate_raw(raw_rec, save_folder, manual_bad_channel_ids=None,
                 min_distance_um=200.0, scan=None, line_freq=LINE_FREQ,
                 full_saturation_scan=True, rail=None):
    """Pre-preprocessing validation: the state of the data before the chain runs.

    Covers noise levels, bad-channel detection (with contiguity and a cross-tab
    against manually supplied channels), ADC clipping, distant-pair correlation,
    PSD vs depth, and line-noise harmonics.

    Returns a dict of metrics; writes plots and preprocessing_qc_raw.json into
    save_folder.
    """
    save_folder.mkdir(parents=True, exist_ok=True)
    cfg = dict(DEFAULT_SCAN, **(scan or {}))
    depths = _depths(raw_rec)
    locations = _locations(raw_rec)
    shank_ids = _shank_ids(raw_rec)
    chan_ids = np.asarray(raw_rec.channel_ids)
    starts, n_samp, margin = _chunk_starts(raw_rec, cfg['n_chunks'],
                                           cfg['chunk_duration_s'], cfg['margin_s'],
                                           cfg['seed'])
    in_uV = _traces_in_uV(raw_rec)
    rep = {'stage': 'raw', 'n_channels': int(len(chan_ids)),
           'duration_s': float(raw_rec.get_total_duration()),
           'sampling_frequency': float(raw_rec.get_sampling_frequency()),
           'n_chunks': int(len(starts)),
           'chunk_duration_s': float(n_samp / raw_rec.get_sampling_frequency()),
           'traces_in_uV': bool(in_uV)}

    # -- noise levels ------------------------------------------------------
    # Seeded: get_noise_levels draws random chunks, and on an unfiltered, LFP-dominated
    # trace the MAD depends heavily on which seconds it lands on -- three unseeded calls
    # on one recording gave medians of 506, 535 and 374. That variance propagates into
    # the pre-CAR -> post-CAR ratio, which is the durable number this report exists to
    # carry, so it cannot be left to differ run to run on identical data.
    noise = si.get_noise_levels(raw_rec, return_in_uV=in_uV, method='mad',
                                progress_bar=False,
                                random_slices_kwargs=dict(seed=cfg['seed']))
    rep['noise'] = {'median': float(np.median(noise)),
                    'iqr': float(np.subtract(*np.percentile(noise, [75, 25]))),
                    'min': float(noise.min()), 'max': float(noise.max())}
    # raw is unfiltered, so this is LFP-dominated; the meaningful comparison is
    # pre-CAR vs post-CAR, both of which are filtered
    rep['noise']['note'] = 'raw is unfiltered: LFP-dominated, not comparable to the filtered stages'

    # -- bad channels ------------------------------------------------------
    bad_ids, labels = si.detect_bad_channels(raw_rec, seed=cfg['seed'])
    labels = np.asarray(labels)
    rep['bad_channels'] = {
        'counts': {str(c): int((labels == c).sum()) for c in np.unique(labels)},
        'auto_bad_ids': [str(c) for c in np.asarray(bad_ids)],
        'out_contiguity': _label_contiguity(labels, depths, 'out'),
        'dead_contiguity': _label_contiguity(labels, depths, 'dead'),
    }
    if manual_bad_channel_ids is not None and len(manual_bad_channel_ids):
        manual = np.asarray(manual_bad_channel_ids)
        auto = np.asarray(bad_ids)
        rep['bad_channels']['manual_vs_auto'] = {
            'n_manual': int(len(manual)),
            'n_auto': int(len(auto)),
            'in_both': [str(c) for c in np.intersect1d(manual, auto)],
            'manual_only': [str(c) for c in np.setdiff1d(manual, auto)],
            'auto_only': [str(c) for c in np.setdiff1d(auto, manual)],
        }
    else:
        manual = None

    # -- clipping ----------------------------------------------------------
    at_rail, near_rail, dtype, scanned_s, rail = _clipping_fraction(
        raw_rec, starts, n_samp, margin, full_scan=full_saturation_scan, rail=rail)
    uv = raw_rec.get_property('gain_to_uV')
    uv = float(np.asarray(uv)[0]) if uv is not None and np.unique(uv).size == 1 else None
    if at_rail is None:
        rep['saturation'] = {'dtype': str(dtype),
                             'note': 'non-integer dtype, rails undefined'}
    elif rail is None:
        rep['saturation'] = {'dtype': str(dtype), 'rail': None,
                             'note': 'no clipping level found in this recording'}
    else:
        rep['saturation'] = {
            'dtype': str(dtype),
            'rail': int(rail),
            'rail_uV': None if uv is None else rail * uv,
            'dtype_rail': int(np.iinfo(dtype).max),
            'full_scan': bool(full_saturation_scan),
            'scanned_s': float(scanned_s),
            'frac_at_rail_total': float(at_rail.mean()),
            'frac_near_rail_total': float(near_rail.mean()),
            'n_channels_with_clipping': int((at_rail > 0).sum()),
            'worst_channel': str(chan_ids[int(np.argmax(at_rail))]),
            'worst_channel_frac': float(at_rail.max()),
        }
        _plot_saturation(depths, at_rail, near_rail, save_folder)

    # -- correlation, PSD, line noise --------------------------------------
    sc = _scan(raw_rec, starts, n_samp, margin, do_corr=True, do_psd=True,
               return_in_uV=in_uV)
    med_far, far_vals, far_groups = _distant_pair_corr(
        sc['corr'], locations, min_distance_um, shank_ids)
    rep['distant_pair_corr'] = {'median_abs_r': med_far,
                                'min_distance_um': float(min_distance_um),
                                'n_pairs': int(far_vals.size),
                                'by_shank': far_groups}
    harm, rdb = _line_harmonics(sc['freqs'], sc['psd'], line_freq=line_freq)
    med_db = np.nanmedian(rdb, axis=0)
    flagged = [(float(f), float(d)) for f, d in zip(harm, med_db)
               if np.isfinite(d) and d > LINE_FLAG_DB]
    rep['line_noise'] = {
        'line_freq': float(line_freq),
        'flag_db': LINE_FLAG_DB,
        'n_harmonics_tested': int(len(harm)),
        'fundamental_db': float(med_db[0]) if len(med_db) else np.nan,
        'flagged_hz': [f for f, _ in flagged],
        'worst_hz': float(harm[int(np.nanargmax(med_db))]) if np.isfinite(med_db).any() else None,
        'worst_db': float(np.nanmax(med_db)) if np.isfinite(med_db).any() else None,
    }

    _plot_noise(depths, {'raw': noise}, save_folder, 'qc_noise_levels_raw',
                unit='uV' if in_uV else 'a.u.')
    _plot_bad_channels(depths, labels, manual, chan_ids, save_folder)
    _plot_correlation({'raw': sc['corr']}, {'raw': far_vals}, save_folder,
                      'qc_channel_correlation_raw', min_distance_um)
    _plot_psd_depth({'raw': sc['freqs']}, {'raw': sc['psd']}, {'raw': depths},
                    save_folder, 'qc_psd_depth_raw')
    peaks_raw, med_raw, base_raw, combs_raw = find_spectral_peaks(
        sc['freqs'], sc['psd'], raw_rec.get_sampling_frequency(), line_freq=line_freq)
    rep['spectral_peaks'] = {
        'combs': combs_raw,
        'n_peaks': len(peaks_raw),
        'peaks': [{k: v for k, v in pk.items() if k != 'baseline_psd'}
                  for pk in peaks_raw],
    }
    _plot_spectral_peaks(sc['freqs'], {'raw': med_raw}, peaks_raw, save_folder,
                         None, 'qc_spectral_peaks_raw', combs_raw)
    _plot_line_noise({'raw': harm}, {'raw': rdb}, save_folder, None, 'qc_line_noise_raw',
                     line_freq)

    _write_json(save_folder / 'preprocessing_qc_raw.json', rep)
    rep['_internal'] = {'starts': starts, 'n_samp': n_samp, 'margin': margin,
                        'n_total': int(raw_rec.get_num_samples()), 'depths': depths,
                        'labels': labels, 'noise': noise, 'freqs': sc['freqs'],
                        'psd': sc['psd'], 'corr': sc['corr'], 'far_vals': far_vals,
                        'harm': harm, 'ratios_db': rdb,
                        'psd_med': med_raw, 'peaks': peaks_raw, 'combs': combs_raw}
    return rep


def validate_preprocessed(final_rec, save_folder, reference_rec=None,
                          no_phase_shift_rec=None, raw_report=None,
                          passband=None, min_distance_um=200.0, scan=None,
                          line_freq=LINE_FREQ):
    """Post-preprocessing validation: what the chain actually did.

    reference_rec      the same chain *without* common_reference, so the CAR-efficacy
                       numbers compare like with like (both filtered).
    no_phase_shift_rec the same chain without phase_shift, as the control for the
                       zero-lag adjacency test.
    raw_report         the dict from validate_raw; when given, its chunk positions are
                       reused so before/after sample the same seconds of the session,
                       and its line-noise numbers are carried into the comparison plot.
    passband           (low, high) of the bandpass, used to mark which harmonics survive
                       into the sorting band.

    Returns a dict of metrics; writes plots and preprocessing_qc_preprocessed.json.
    """
    save_folder.mkdir(parents=True, exist_ok=True)
    cfg = dict(DEFAULT_SCAN, **(scan or {}))
    depths = _depths(final_rec)
    locations = _locations(final_rec)
    shank_ids = _shank_ids(final_rec)
    chan_ids = np.asarray(final_rec.channel_ids)

    # reuse the raw pass's chunk positions so the comparison is like for like
    internal = (raw_report or {}).get('_internal')
    if internal is not None and final_rec.get_num_samples() == internal['n_total']:
        starts, n_samp, margin = internal['starts'], internal['n_samp'], internal['margin']
    else:
        starts, n_samp, margin = _chunk_starts(final_rec, cfg['n_chunks'],
                                               cfg['chunk_duration_s'], cfg['margin_s'],
                                               cfg['seed'])
    in_uV = _traces_in_uV(final_rec)
    unit = 'uV' if in_uV else 'a.u.'
    rep = {'stage': 'preprocessed', 'n_channels': int(len(chan_ids)),
           'n_chunks': int(len(starts)), 'traces_in_uV': bool(in_uV)}

    # -- noise levels, and the CAR ratio -----------------------------------
    noise_final = si.get_noise_levels(final_rec, return_in_uV=in_uV, method='mad',
                                      progress_bar=False,
                                      random_slices_kwargs=dict(seed=cfg['seed']))
    series = {'post-CAR': noise_final}
    rep['noise'] = {'post_car': {'median': float(np.median(noise_final)),
                                 'iqr': float(np.subtract(*np.percentile(noise_final, [75, 25])))}}
    noise_ref = None
    if reference_rec is not None:
        noise_ref = si.get_noise_levels(reference_rec, return_in_uV=in_uV, method='mad',
                                        progress_bar=False,
                                        random_slices_kwargs=dict(seed=cfg['seed']))
        series = {'pre-CAR': noise_ref, 'post-CAR': noise_final}
        rep['noise']['pre_car'] = {'median': float(np.median(noise_ref)),
                                   'iqr': float(np.subtract(*np.percentile(noise_ref, [75, 25])))}
        rep['noise']['car_ratio'] = float(np.median(noise_final) / np.median(noise_ref))
        rep['noise']['car_iqr_ratio'] = float(
            np.subtract(*np.percentile(noise_final, [75, 25])) /
            max(np.subtract(*np.percentile(noise_ref, [75, 25])), 1e-12))
    _plot_noise(depths, series, save_folder, 'qc_noise_levels_preprocessed', unit=unit)

    # -- residual bad channels ---------------------------------------------
    # Measured pre-CAR, for the same reason the adjacency lag below is. detect_bad_channels
    # scores each channel by its projection onto the median across channels, which is the
    # exact quantity common_reference subtracts. Post-CAR that reference is zero by
    # construction, the projection is 0/0, and the labels scatter at random: on DS45 it
    # condemned all 384 channels (201 dead + 183 noise, not one good) in a session the raw
    # pass called entirely good. Pre-CAR is also the question worth asking -- whether
    # interpolation fixed the channels flagged upstream. The fallback is not a degenerate
    # case: reference_rec is None only when the protocol has no common_reference, and then
    # final_rec is itself the pre-CAR recording.
    resid_rec = reference_rec if reference_rec is not None else final_rec
    resid_stage = 'pre-CAR' if reference_rec is not None else 'post-CAR'
    try:
        bad_ids, labels = si.detect_bad_channels(resid_rec, seed=cfg['seed'])
        labels = np.asarray(labels)
        rep['bad_channels_residual'] = {
            'measured_on': resid_stage,
            'counts': {str(c): int((labels == c).sum()) for c in np.unique(labels)},
            'auto_bad_ids': [str(c) for c in np.asarray(bad_ids)],
        }
    except Exception as e:
        rep['bad_channels_residual'] = {'error': str(e)}

    # -- the main scan of the final recording ------------------------------
    # The zero-lag adjacency test runs on the pre-CAR recording, not the final one:
    # common-mode noise is identical across channels by construction, which makes it the
    # ideal reference signal for measuring a pure ADC timing offset. Common referencing
    # is designed to delete exactly that signal, so measuring the lag after CAR leaves
    # only independent per-channel noise and the peak lag becomes uniform noise.
    lag_pairs = _adjacent_pairs(final_rec)
    lag_rec = reference_rec if reference_rec is not None else final_rec
    lag_stage = 'pre-CAR' if reference_rec is not None else 'post-CAR'
    sc_final = _scan(final_rec, starts, n_samp, margin, do_corr=True, do_psd=True,
                     lag_pairs=None if reference_rec is not None else lag_pairs,
                     return_in_uV=in_uV)

    corrs, dist_vals = {}, {}
    if reference_rec is not None:
        sc_ref = _scan(reference_rec, starts, n_samp, margin, do_corr=True,
                       do_psd=False, lag_pairs=lag_pairs, return_in_uV=in_uV)
        med_ref, vals_ref, grp_ref = _distant_pair_corr(
            sc_ref['corr'], locations, min_distance_um, shank_ids)
        corrs['pre-CAR'] = sc_ref['corr']
        dist_vals['pre-CAR'] = vals_ref
        rep['distant_pair_corr'] = {'pre_car_median_abs_r': med_ref,
                                    'pre_car_by_shank': grp_ref}
    else:
        rep['distant_pair_corr'] = {}
    med_fin, vals_fin, grp_fin = _distant_pair_corr(
        sc_final['corr'], locations, min_distance_um, shank_ids)
    corrs['post-CAR'] = sc_final['corr']
    dist_vals['post-CAR'] = vals_fin
    rep['distant_pair_corr'].update({'post_car_median_abs_r': med_fin,
                                     'post_car_by_shank': grp_fin,
                                     'min_distance_um': float(min_distance_um),
                                     'n_pairs': int(vals_fin.size),
                                     'n_shanks': (0 if shank_ids is None
                                                  else int(np.unique(shank_ids).size))})
    if reference_rec is not None and np.isfinite(rep['distant_pair_corr']['pre_car_median_abs_r']):
        rep['distant_pair_corr']['reduction_factor'] = float(
            rep['distant_pair_corr']['pre_car_median_abs_r'] / max(med_fin, 1e-12))
    _plot_correlation(corrs, dist_vals, save_folder, 'qc_channel_correlation',
                      min_distance_um)

    # -- PSD and line noise -------------------------------------------------
    psds = {'preprocessed': sc_final['psd']}
    freqs = {'preprocessed': sc_final['freqs']}
    pdepths = {'preprocessed': depths}
    harms = {'preprocessed': None}
    rdbs = {}
    if internal is not None:
        psds = {'raw': internal['psd'], 'preprocessed': sc_final['psd']}
        freqs = {'raw': internal['freqs'], 'preprocessed': sc_final['freqs']}
        # the raw stage keeps its own depths: a chain with detect_and_remove_bad_channels
        # returns fewer channels than it was given
        pdepths = {'raw': internal['depths'], 'preprocessed': depths}
        rdbs['raw'] = internal['ratios_db']
        harms = {'raw': internal['harm']}
        rep['n_channels_removed'] = int(len(internal['depths']) - len(depths))
    _plot_psd_depth(freqs, psds, pdepths, save_folder, 'qc_psd_depth')

    harm_f, rdb_f = _line_harmonics(sc_final['freqs'], sc_final['psd'], line_freq=line_freq)
    harms['preprocessed'] = harm_f
    rdbs['preprocessed'] = rdb_f
    med_db = np.nanmedian(rdb_f, axis=0)
    in_band = np.ones(len(harm_f), dtype=bool) if passband is None else \
        (harm_f >= passband[0]) & (harm_f <= passband[1])
    surviving = [(float(f), float(d)) for f, d, b in zip(harm_f, med_db, in_band)
                 if b and np.isfinite(d) and d > LINE_FLAG_DB]
    rep['line_noise'] = {
        'line_freq': float(line_freq),
        'flag_db': LINE_FLAG_DB,
        'passband': list(passband) if passband is not None else None,
        'n_harmonics_tested': int(len(harm_f)),
        'fundamental_db': float(med_db[0]) if len(med_db) else np.nan,
        'flagged_hz': [float(f) for f, d in zip(harm_f, med_db)
                       if np.isfinite(d) and d > LINE_FLAG_DB],
        'flagged_in_passband_hz': [f for f, _ in surviving],
        'worst_in_passband_db': max([d for _, d in surviving], default=None),
    }
    peaks_fin, med_fin_psd, _base, combs_fin = find_spectral_peaks(
        sc_final['freqs'], sc_final['psd'], final_rec.get_sampling_frequency(),
        line_freq=line_freq)
    # The comparison that matters: the peaks found in the RAW spectrum, measured again
    # at the same frequencies after the chain. Detecting peaks separately in each stage
    # would silently drop the ones preprocessing removed, which are the successes.
    compare = peaks_fin
    if internal is not None and len(internal.get('psd_med', [])) == len(med_fin_psd):
        raw_med, raw_peaks = internal['psd_med'], internal['peaks']

        def _measured(pk, source):
            j = int(np.argmin(np.abs(sc_final['freqs'] - pk['freq'])))
            q = dict(pk)
            q['psd_raw'] = float(raw_med[j])
            q['psd'] = float(med_fin_psd[j])
            q['attenuation_db'] = float(10.0 * np.log10(
                max(med_fin_psd[j], 1e-300) / max(raw_med[j], 1e-300)))
            q['in_passband'] = bool(passband is None or
                                    passband[0] <= pk['freq'] <= passband[1])
            q['source'] = source
            return q

        compare = [_measured(pk, 'raw') for pk in raw_peaks]
        # Walking the raw peaks alone cannot see a line that is subthreshold in raw and
        # prominent after the chain -- and those are the ones that matter most, since they
        # reach the sorter with nothing upstream to blame them on. DS45 carries one at
        # 3214 Hz: +5 dB over the raw floor, under the flag threshold, and +19 dB after
        # common_reference drops the broadband floor 11 dB and the line only 0.7 dB. So
        # union the final-spectrum detection in rather than discarding it. Dedupe within
        # three bins, since a peak can land a bin or two apart between the two stages.
        freqs_fin = sc_final['freqs']
        tol = 3 * float(np.median(np.diff(freqs_fin))) if len(freqs_fin) > 1 else 0.0
        seen = np.array([q['freq'] for q in compare], dtype=float)
        for pk in peaks_fin:
            if seen.size and np.abs(seen - pk['freq']).min() <= tol:
                continue
            compare.append(_measured(pk, 'final-only'))
        compare.sort(key=lambda q: q['freq'])
    n_final_only = sum(1 for pk in compare if pk.get('source') == 'final-only')
    rep['spectral_peaks'] = {
        'combs': combs_fin,
        'n_peaks_raw': len(compare) - n_final_only,
        'n_peaks_final_only': n_final_only,
        'peaks': [{k: v for k, v in pk.items() if k != 'baseline_psd'}
                  for pk in compare],
    }
    spec = {'preprocessed': med_fin_psd}
    if internal is not None and len(internal.get('psd_med', [])) == len(med_fin_psd):
        spec = {'raw': internal['psd_med'], 'preprocessed': med_fin_psd}
    _plot_spectral_peaks(sc_final['freqs'], spec, compare, save_folder, passband,
                         'qc_spectral_peaks', combs_fin)
    _plot_line_noise({k: harms[k] for k in rdbs}, rdbs, save_folder, passband,
                     'qc_line_noise', line_freq)

    # -- zero-lag adjacency -------------------------------------------------
    sc_lag = sc_ref if reference_rec is not None else sc_final
    lags_final = _peak_lags(sc_lag['xcorr'], sc_lag['lags'])
    lag_sets = {'with phase_shift': lags_final}
    max_lag = int(sc_lag['lags'].max())
    rep['adjacency_lag'] = {
        'measured_on': lag_stage,
        'n_pairs': int(len(lags_final)),
        'median_abs_lag_samples': float(np.median(np.abs(lags_final))),
        'frac_within_half_sample': float(np.mean(np.abs(lags_final) <= 0.5)),
        # a peak pinned to the search boundary means there is no real peak: the two
        # channels share no signal, so the metric is not measuring timing at all
        'frac_at_search_boundary': float(np.mean(np.abs(lags_final) >= max_lag - 0.5)),
    }
    if no_phase_shift_rec is not None:
        sc_nops = _scan(no_phase_shift_rec, starts, n_samp, margin, do_corr=False,
                        do_psd=False, lag_pairs=lag_pairs, return_in_uV=in_uV)
        lags_nops = _peak_lags(sc_nops['xcorr'], sc_nops['lags'])
        lag_sets['without phase_shift'] = lags_nops
        rep['adjacency_lag']['median_abs_lag_no_phase_shift'] = float(
            np.median(np.abs(lags_nops)))
    _plot_adjacency_lag(lag_sets, save_folder, stage=lag_stage)

    _write_json(save_folder / 'preprocessing_qc_preprocessed.json', rep)
    return rep


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def split_protocol(preprocessing_protocol, at_step='common_reference'):
    """Split a preprocessing dict into the steps before at_step and the rest.

    Applying the two halves in sequence reproduces the full chain exactly, which is
    what lets the QC hold a handle on the recording as it was just before referencing.
    """
    keys = list(preprocessing_protocol)
    if at_step not in keys:
        return dict(preprocessing_protocol), {}
    i = keys.index(at_step)
    before = {k: preprocessing_protocol[k] for k in keys[:i]}
    after = {k: preprocessing_protocol[k] for k in keys[i:]}
    return before, after


def _silenced_periods(preprocessing_protocol):
    """The frame intervals silence_periods blanks, flattened across segments.

    Returned for the artifact plot, which has to be able to say whether a flat
    post-chain panel is a cleaned trace or a blanked one. None when the chain has no
    silence_periods step, which the plot reads as "nothing to account for".
    """
    sp = preprocessing_protocol.get('silence_periods')
    if not sp:
        return None
    # list_periods is one list of intervals per segment. The triggers this is compared
    # against are segment-0 frames, so on a multi-segment recording only the first
    # segment's intervals are on the same clock; the rest would be nonsense here.
    per_segment = sp.get('list_periods') or []
    flat = list(per_segment[0]) if per_segment else []
    return np.asarray(flat, dtype=np.int64).reshape(-1, 2) if flat else None


def _passband_from_protocol(preprocessing_protocol, default=(300.0, 6000.0)):
    """(low, high) of the bandpass step, for marking which line harmonics survive it."""
    bp = preprocessing_protocol.get('bandpass_filter')
    if bp is None:
        return None
    return (float(bp.get('freq_min', default[0])), float(bp.get('freq_max', default[1])))


def run_event_average_qc(raw_rec, final_rec, event_triggers, save_folder, **kwargs):
    """Event-triggered averages, raw against fully preprocessed, one plot per event.

    The standalone entry point for the event panels: run_preprocessing_qc calls it as
    part of a full QC pass, and a script that only wants these plots -- off a session
    that was preprocessed weeks ago, with the saved preprocess folder standing in for
    the chain -- calls it directly with the two recordings and is done.

    event_triggers is {name: frame indices} on the raw recording's clock. Returns
    {name: stats}; a name whose plot could not be made carries an 'error' key rather
    than raising, so one bad event does not lose the others.
    """
    save_folder.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, frames in (event_triggers or {}).items():
        _, out[name] = _plot_event_average(raw_rec, final_rec, frames, save_folder,
                                           name, **kwargs)
    return out


def run_preprocessing_qc(raw_rec, preprocessing_protocol, save_folder,
                         pre_car_rec=None, final_rec=None,
                         manual_bad_channel_ids=None, scan=None,
                         line_freq=LINE_FREQ, run_raw=True,
                         full_saturation_scan=True, rail=None,
                         event_triggers=None, event_ms_window=200.0,
                         max_event_trials=None, silence_ms_window=None):
    """Run both validation routines around a preprocessing protocol.

    pre_car_rec / final_rec: pass the handles the pipeline already built, so the QC
    measures the chain that actually ran. This matters when the protocol contains
    detect_and_remove_bad_channels with no explicit bad_channel_ids: detection samples
    random chunks, so rebuilding the chain here can drop a *different* set of channels
    than the recording that was saved to disk. Split the protocol with split_protocol
    in the caller, build both stages once, and hand them in. When omitted they are
    rebuilt from the protocol, which is fine for a deterministic chain.

    The one recording always built here is the without-phase_shift control for the
    zero-lag adjacency test; it is dropped if channel removal made it inconsistent
    with the final recording.

    event_triggers: {name: frame indices} of behavioural events -- licks, cues -- each
    of which gets a before/after event-triggered average written as
    qc_event_average_<name>.png. Frames, not seconds, and on the raw recording's clock,
    the same convention remove_artifacts' list_triggers uses. Left empty by default
    because the QC has no way to find these events itself: they come from the nidq
    parse, which the caller has already done.

    max_event_trials caps how many events each of those averages uses; None, the
    default, means all of them, which on a session with thousands of licks is tens of
    GB of scattered reads. Set it if the plot is not worth that.

    silence_ms_window: half window of qc_silence_onset.png, the average around the start
    of every silence_periods span. Only drawn when the protocol has that step. None
    sizes it from the padding: each span opens EPOCH_MARGIN_MS ahead of the first
    clipped sample, so a window narrower than that never reaches the clipping. The
    default runs that margin plus 10 ms either side, which shows the clean lead-in, the
    onset of clipping and the first 10 ms of it.

    Returns (raw_report, post_report) and writes preprocessing_qc_summary.txt.
    """
    save_folder.mkdir(parents=True, exist_ok=True)
    before, after = split_protocol(preprocessing_protocol, 'common_reference')

    if pre_car_rec is None:
        pre_car_rec = si.apply_preprocessing_pipeline(raw_rec, before) if before else raw_rec
    if final_rec is None:
        final_rec = si.apply_preprocessing_pipeline(pre_car_rec, after) if after else pre_car_rec
    pre_car, final = pre_car_rec, final_rec

    no_ps = None
    if 'phase_shift' in before:
        without = {k: v for k, v in before.items() if k != 'phase_shift'}
        cand = si.apply_preprocessing_pipeline(raw_rec, without) if without else raw_rec
        # the lag test indexes both recordings with one set of channel positions, so the
        # control is only usable if channel removal left them identical
        if np.array_equal(np.asarray(cand.channel_ids), np.asarray(final.channel_ids)):
            no_ps = cand
        else:
            print('preprocess_qc: skipping the without-phase_shift control -- channel '
                  'removal is not reproducible across rebuilds of the chain')

    raw_report = None
    if run_raw:
        raw_report = validate_raw(raw_rec, save_folder,
                                  manual_bad_channel_ids=manual_bad_channel_ids,
                                  scan=scan, line_freq=line_freq,
                                  full_saturation_scan=full_saturation_scan,
                                  rail=rail)
    post_report = validate_preprocessed(
        final, save_folder, reference_rec=pre_car, no_phase_shift_rec=no_ps,
        raw_report=raw_report, passband=_passband_from_protocol(preprocessing_protocol),
        scan=scan, line_freq=line_freq)

    # remove_artifacts is the one step whose success cannot be read off a spectrum: it is
    # a local, time-domain repair, and averaging over the triggers is the direct test of it.
    # The step is re-applied to the raw recording on its own rather than reusing pre_car,
    # so the first two panels differ by that step alone and nothing downstream of it.
    # The third panel is the same average off the finished recording -- not a controlled
    # comparison, but the one that says what trigger-locked signal reaches the sorter.
    ra = preprocessing_protocol.get('remove_artifacts')
    if ra is not None and np.asarray(ra.get('list_triggers', []), dtype=np.int64).size:
        cleaned = si.apply_preprocessing_pipeline(raw_rec, {'remove_artifacts': ra})
        _, art_stats = _plot_artifact_removal(
            raw_rec, cleaned, ra['list_triggers'], save_folder,
            ms_before=ra.get('ms_before', 0.5), ms_after=ra.get('ms_after', 0.5),
            hp_hz=(_passband_from_protocol(preprocessing_protocol) or (300.0,))[0],
            final_rec=final,
            silenced_periods=_silenced_periods(preprocessing_protocol))
        post_report['artifact_removal'] = art_stats

    # Behavioural events get the same treatment for a different reason: not to score a
    # step, but to see whether the event puts anything in the traces and what the chain
    # does with it. A lick is a jaw movement against a headpost, and a movement artifact
    # that survives to the sorter is one the sorter will try to explain as spikes.
    if event_triggers:
        post_report['event_averages'] = run_event_average_qc(
            raw_rec, final, event_triggers, save_folder,
            ms_window=event_ms_window, max_events=max_event_trials,
            hp_hz=(_passband_from_protocol(preprocessing_protocol) or (300.0,))[0],
            silenced_periods=_silenced_periods(preprocessing_protocol))

    # The silence_periods onsets, averaged the same way: what the traces are doing as
    # each blanked span begins, raw against what reaches the sorter. Kept out of
    # event_averages because the fully preprocessed panel steps to fill at t=0 by
    # construction, so the trigger-locked-residual warning those entries get would
    # fire on every session and mean nothing.
    silenced = _silenced_periods(preprocessing_protocol)
    if silenced is not None:
        # the pad is the detector default; a session scanned with a different
        # margin_ms would put the clipping somewhere other than this line
        pad_ms = dsat.EPOCH_MARGIN_MS
        _, post_report['silence_onset'] = _plot_event_average(
            raw_rec, final, silenced[:, 0], save_folder, 'silence_onset',
            name='qc_silence_onset',
            ms_window=silence_ms_window if silence_ms_window is not None else pad_ms + 10.0,
            max_events=None,
            hp_hz=(_passband_from_protocol(preprocessing_protocol) or (300.0,))[0],
            silenced_periods=silenced,
            marks=((pad_ms, f'clipping begins (+{pad_ms:g} ms pad)'),))

    post_report['final_channel_ids'] = [str(c) for c in np.asarray(final.channel_ids)]
    # validate_preprocessed dumps the json before returning, so anything added to the
    # report out here -- artifact_removal, final_channel_ids -- never reached the file.
    # Rewrite it now that the report is complete; the summary text below already had them.
    _write_json(save_folder / 'preprocessing_qc_preprocessed.json', post_report)
    (save_folder / 'preprocessing_qc_summary.txt').write_text(
        _summary_text(raw_report, post_report, preprocessing_protocol))
    if raw_report is not None:
        raw_report.pop('_internal', None)
    return raw_report, post_report


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items() if k != '_internal'}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def _write_json(path, report):
    path.write_text(json.dumps(_jsonable(report), indent=2))


def _trunc(seq, n=12):
    """Channel-id lists can be hundreds long; keep the summary readable."""
    seq = list(seq)
    if len(seq) <= n:
        return str(seq)
    return f'[{", ".join(map(str, seq[:n]))}, ... +{len(seq) - n} more]'


def _peak_line(pk):
    """One report line for a peak, with its sidebands folded in when it is a carrier."""
    fam = ','.join(pk['kinds']) if pk['kinds'] else 'unidentified'
    line = f"{pk['freq']:8.1f} Hz  {pk['excess_db']:+5.1f} dB  {fam}"
    sb = pk.get('sidebands')
    if sb:
        line += (f"  + {sb['n']} sidebands at +/-{sb['fundamental_hz']:.1f} Hz"
                 + (f" x{sb['harmonics']}" if len(sb['harmonics']) > 1 else ''))
        if sb['extra_offsets_hz']:
            line += (' (plus +/-'
                     + ', '.join(f'{o:.1f}' for o in sb['extra_offsets_hz']) + ')')
    return line


def _shank_lines(groups, indent='  '):
    """One line per within-/cross-shank split of the distant-pair metric."""
    if not groups:
        return []
    out = []
    for label in ('within_shank', 'cross_shank'):
        g = groups.get(label)
        if not g or not g['n_pairs']:
            continue
        out.append(f"{indent}{label.replace('_', '-')}: "
                   f"|r|={g['median_abs_r']:.4f} ({g['n_pairs']} pairs)")
    return out


def _summary_text(raw_report, post_report, protocol):
    """Flat, greppable summary of the numbers that decide whether the chain is healthy."""
    L = ['PREPROCESSING QC', '=' * 60,
         'chain: ' + ' -> '.join(protocol), '']

    if raw_report is not None:
        L += ['-- raw --']
        n = raw_report['noise']
        L.append(f"noise (MAD): median={n['median']:.3g} IQR={n['iqr']:.3g}  [{n['note']}]")
        bc = raw_report['bad_channels']
        L.append(f"bad channels: {bc['counts']}")
        oc = bc['out_contiguity']
        L.append(f"  'out' labels: n={oc['n']} runs={oc['n_runs']} "
                 f"contiguous={oc['contiguous']} at_shank_end={oc['at_shank_end']}"
                 + ('' if oc['contiguous'] else '   <-- scattered: detector may be fooled'))
        mv = bc.get('manual_vs_auto')
        if mv:
            L.append(f"  manual vs auto: both={len(mv['in_both'])} "
                     f"manual_only={_trunc(mv['manual_only'])} "
                     f"auto_only={_trunc(mv['auto_only'])}")
        s = raw_report['saturation']
        if 'frac_at_rail_total' in s:
            flag = '' if s['frac_at_rail_total'] == 0 else '   <-- RED FLAG'
            scope = (f"full {s['scanned_s']:.0f} s" if s.get('full_scan')
                     else f"sampled {s['scanned_s']:.0f} s")
            rail_txt = f"rail=+/-{s['rail']}"
            if s.get('rail_uV'):
                rail_txt += f" ({s['rail_uV'] / 1000:.3f} mV)"
            if s.get('dtype_rail'):
                rail_txt += f", measured not taken from the dtype ({s['dtype_rail']})"
            L.append(f"clipping ({scope}): {rail_txt}")
            L.append(f"  at_rail={s['frac_at_rail_total']:.3g} "
                     f"near_rail={s['frac_near_rail_total']:.3g} "
                     f"channels_affected={s['n_channels_with_clipping']}{flag}")
        else:
            L.append(f"clipping: {s.get('note')}")
        dc = raw_report['distant_pair_corr']
        L.append(f"distant-pair |r|: {dc['median_abs_r']:.4f} "
                 f"({dc['n_pairs']} pairs >= {dc['min_distance_um']:g} um apart, "
                 'full electrode geometry)')
        L += _shank_lines(dc.get('by_shank'), indent='  ')
        ln = raw_report['line_noise']
        L.append(f"line noise: {ln['line_freq']:g} Hz fundamental "
                 f"{ln['fundamental_db']:.1f} dB over baseline; "
                 f"{len(ln['flagged_hz'])}/{ln['n_harmonics_tested']} harmonics "
                 f"> {ln['flag_db']:g} dB")
        sp = raw_report.get('spectral_peaks')
        if sp:
            L.append(f"spectral peaks: {sp['n_peaks']} narrowband peaks"
                     + (f"; combs " + ', '.join(f"{c['name']} to x{c['extent_k']}"
                                                for c in sp['combs'])
                        if sp['combs'] else '; no comb established'))
            shown = [q for q in sp['peaks'] if q.get('role') != 'sideband']
            n_sb = sum(1 for q in sp['peaks'] if q.get('role') == 'sideband')
            if n_sb:
                L.append(f"  {n_sb} of them are modulation sidebands, folded into their "
                         'carriers below')
            for pk in sorted(shown, key=lambda q: -q['excess_db'])[:6]:
                L.append('    ' + _peak_line(pk))
        L.append('')

    p = post_report
    L += ['-- preprocessed --']
    n = p['noise']
    if 'pre_car' in n:
        L.append(f"noise (MAD): pre-CAR median={n['pre_car']['median']:.3g} -> "
                 f"post-CAR {n['post_car']['median']:.3g}  "
                 f"ratio={n['car_ratio']:.3f}  IQR ratio={n['car_iqr_ratio']:.3f}")
    else:
        L.append(f"noise (MAD): post-CAR median={n['post_car']['median']:.3g}")
    dc = p['distant_pair_corr']
    if 'pre_car_median_abs_r' in dc:
        L.append(f"distant-pair |r|: pre-CAR {dc['pre_car_median_abs_r']:.4f} -> "
                 f"post-CAR {dc['post_car_median_abs_r']:.4f}"
                 + (f"  ({dc['reduction_factor']:.1f}x reduction)"
                    if 'reduction_factor' in dc else ''))
    else:
        L.append(f"distant-pair |r|: post-CAR {dc['post_car_median_abs_r']:.4f}")
    if dc.get('n_shanks', 0) > 1:
        # a global median can only remove what the shanks share; if the cross-shank
        # correlation is already near zero before referencing, there is nothing there
        # for it to take out and per-shank groups are the better choice
        L.append(f"  {dc['n_shanks']} shanks -- global CAR is only justified where the "
                 'cross-shank correlation is real:')
        for stage, key in (('pre-CAR', 'pre_car_by_shank'),
                           ('post-CAR', 'post_car_by_shank')):
            for line in _shank_lines(dc.get(key), indent='    '):
                L.append(f'{line}   [{stage}]')
    br = p.get('bad_channels_residual', {})
    if 'counts' in br:
        L.append(f"residual bad channels ({br.get('measured_on', 'post-CAR')}): "
                 f"{br['counts']}")
    ln = p['line_noise']
    L.append(f"line noise: {len(ln['flagged_hz'])}/{ln['n_harmonics_tested']} harmonics "
             f"> {ln['flag_db']:g} dB")
    if ln['passband']:
        L.append(f"  inside passband {ln['passband'][0]:g}-{ln['passband'][1]:g} Hz: "
                 + (', '.join(f'{f:g}' for f in ln['flagged_in_passband_hz']) + ' Hz'
                    if ln['flagged_in_passband_hz'] else 'none')
                 + ('   <-- these reach the sorter' if ln['flagged_in_passband_hz'] else ''))
    sp = p.get('spectral_peaks')
    if sp and sp['peaks']:
        pks = sp['peaks']
        has_att = 'attenuation_db' in pks[0]
        inband = [q for q in pks if q.get('in_passband')]
        n_fo = sp.get('n_peaks_final_only', 0)
        L.append(f"spectral peaks: {len(pks) - n_fo} tracked from the raw spectrum"
                 + (f" + {n_fo} that only the preprocessed spectrum shows" if n_fo else '')
                 + (f", {len(inband)} inside the passband" if has_att else ''))
        if has_att:
            cand = [q for q in (inband or pks) if q.get('role') != 'sideband']
            worst = sorted(cand or (inband or pks),
                           key=lambda q: q['attenuation_db'])[-6:]
            for q in sorted(worst, key=lambda r: r['freq']):
                fam = ','.join(q['kinds']) if q['kinds'] else 'unidentified'
                if q.get('source') == 'final-only':
                    fam += ' [not visible in raw]'
                flag = '   <-- reaches the sorter' if q.get('in_passband') else ''
                L.append(f"    {q['freq']:8.1f} Hz  {q['attenuation_db']:+6.1f} dB "
                         f"({q['psd_raw']:.2e} -> {q['psd']:.2e})  {fam}{flag}")
            L.append('  (attenuation is absolute power; a peak can stay tall relative '
                     'to its own local')
            L.append('   floor while its power drops by orders of magnitude, so the '
                     'ratio is not the test)')
    al = p['adjacency_lag']
    L.append(f"adjacency lag ({al['measured_on']}): "
             f"median |lag|={al['median_abs_lag_samples']:.3f} samples, "
             f"{100 * al['frac_within_half_sample']:.0f}% within half a sample "
             f"({al['n_pairs']} pairs)")
    if 'median_abs_lag_no_phase_shift' in al:
        L.append(f"  without phase_shift: {al['median_abs_lag_no_phase_shift']:.3f} samples "
                 '(control -- should be larger)')
    if al['frac_at_search_boundary'] > 0.2:
        L.append(f"  WARNING {100 * al['frac_at_search_boundary']:.0f}% of pairs peak at the "
                 'search boundary: no shared signal, metric is not measuring timing')
    ar = p.get('artifact_removal')
    if ar and 'peak_before_uV' in ar:
        L.append(f"artifact removal ({ar['n_events']} events averaged): peak |mean| "
                 f"{ar['peak_before_uV']:.1f} -> {ar['peak_after_uV']:.1f} uV "
                 f"({ar['reduction_db']:+.1f} dB)")
        # what is left outside the interpolated span is the number that says whether the
        # window is wide enough: inside it the trace is interpolated and near zero by
        # construction, so a small in-window peak proves nothing on its own
        L.append(f"  outside the -{ar['ms_before']:g}/+{ar['ms_after']:g} ms window: "
                 f"{ar['peak_before_outside_window_uV']:.1f} -> "
                 f"{ar['peak_after_outside_window_uV']:.1f} uV")
        if 'peak_final_uV' in ar:
            # the whole chain, not this step: the bandpass and the reference have had
            # their turn as well, so this is what the sorter sees rather than a score
            # for remove_artifacts
            unit = ar.get('units', {}).get(FINAL_STAGE, 'uV')
            L.append(f"  after the full chain: {ar['peak_final_uV']:.1f} {unit} peak, "
                     f"{ar['peak_final_outside_window_uV']:.1f} {unit} outside the window "
                     f"(floor {ar['hf_floor_final_uV']:.1f} {unit})")
            blanked = ar.get('silenced_fraction_of_window', 0.0)
            if blanked:
                L.append(f"  NOTE {100 * blanked:.0f}% of that window is silence_periods "
                         'fill, not cleaned data: the saturation epochs it blanks are '
                         'caused by these same triggers')
        floor = ar.get('hf_floor_after_uV', 0.0)
        if floor and ar['peak_after_outside_window_uV'] > 4 * floor:
            L.append(f"  WARNING trigger-locked residual outside the window is "
                     f"{ar['peak_after_outside_window_uV'] / floor:.1f}x the flat-trace "
                     f"floor ({floor:.1f} uV): the artifact outlasts "
                     f"-{ar['ms_before']:g}/+{ar['ms_after']:g} ms, widen the window")
    for name, ev in (p.get('event_averages') or {}).items():
        if 'error' in ev:
            L.append(f"{name}-triggered average: skipped ({ev['error']})")
            continue
        unit = ev.get('units', {}).get(FINAL_STAGE, 'uV')
        # the swing, not the high-passed peak: a behavioural event is mostly the slow
        # component the hp metric is built to strip, so the hp number understates it by
        # design. Both are printed; the swing is the one to read first.
        L.append(f"{name}-triggered average ({ev['n_events']} of "
                 f"{ev['n_events_total']} events, +/-{ev['ms_window']:g} ms): "
                 f"swing {ev['swing_raw_uV']:.1f} -> {ev['swing_final_uV']:.1f} {unit}, "
                 f"above {ev['metric_hp_hz']:g} Hz "
                 f"{ev['peak_raw_uV']:.1f} -> {ev['peak_final_uV']:.1f} {unit}")
        floor = ev.get('hf_floor_final_uV', 0.0)
        if floor and ev['peak_final_uV'] > 4 * floor:
            L.append(f"  WARNING what reaches the sorter is {ev['peak_final_uV'] / floor:.1f}x "
                     f"the flat-trace floor ({floor:.1f} {unit}): this event leaves a "
                     'trigger-locked signature the sorter will try to explain')
        blanked = ev.get('silenced_fraction_of_window', 0.0)
        if blanked:
            L.append(f"  NOTE {100 * blanked:.0f}% of that window is silence_periods fill")
    so = p.get('silence_onset')
    if so:
        if 'error' in so:
            L.append(f"silence_periods onset average: skipped ({so['error']})")
        else:
            unit = so.get('units', {}).get(FINAL_STAGE, 'uV')
            L.append(f"silence_periods onset average ({so['n_events']} of "
                     f"{so['n_events_total']} periods, +/-{so['ms_window']:g} ms): "
                     f"swing {so['swing_raw_uV']:.1f} -> {so['swing_final_uV']:.1f} {unit}, "
                     f"{100 * so.get('silenced_fraction_of_window', 0.0):.0f}% of the "
                     'window is fill')
    return '\n'.join(L) + '\n'
