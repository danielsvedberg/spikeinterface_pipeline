"""Find amplifier saturation in a raw recording and turn it into blankable periods.

The amplifier rails before the ADC does, so the clipping level is a property of the
session rather than of the dtype. Two NP2.0 sessions off the same probe model, with
identical meta (imMaxInt=2048, gain 100), clip at different levels:

    DS13_20250822   +/-1947 counts   (5.894 mV)
    DS45_20260811   +/-1963 counts   (5.943 mV)

Neither reaches imMaxInt, and both are two orders of magnitude below the int16 rail, so
a threshold taken from `np.iinfo(dtype)` -- or from the meta file -- finds nothing. And
because clipped samples pile up at a single value, a threshold set even 1% high catches
not most of them but none. The rail is therefore always measured.

A hard clip has an unmistakable signature in the value histogram: counts are flat (or
falling) up to the rail, spike by a factor of ~1000 at it, and are exactly zero above.
`find_rail` looks for that and returns None when it is absent, which is the answer for a
session that does not clip.

Saturation then splits into two problems needing opposite treatments:

    dead channels    railed continuously, independent of what the session is doing --
                     channels to remove, not artifacts to blank. In the per-channel
                     histogram they show as rows that are empty everywhere except the
                     rail.
    saturation       time windows where many channels rail together. These are what
                     `silence_periods` should cover.

The two are separated by asking whether a channel rails *while the rest of the array is
quiet*. That distinguishes a broken channel from one caught in a global event without
reference to how much of the session is affected -- a plain "railed more than half the
time" test would condemn the whole array on a session whose bad block covered most of it.

Typical use, alongside the opto triggers the pipeline already builds:

    sat = detect_saturation(raw_rec)
    print(sat['summary'])
    plot_saturation_diagnostics(raw_rec, sat, out_folder)

    bad_chans = sat['dead_channel_inds']                 # detect_and_remove_bad_channels
    protocol['silence_periods'] = sat['silence_periods_kwargs']

`silence_periods` assumes zero-centred traces, so on a full-band stream (NP2.0 has no
AP/LF split) the step belongs *after* `bandpass_filter`:

    phase_shift -> remove_artifacts -> bandpass_filter -> silence_periods
                -> detect_and_remove_bad_channels -> common_reference

Periods are padded before use so the blanking also covers the ring-down that filtfilt
smears outward from each clipped edge, in both time directions.

The fill defaults to zeros, which is what `silence_periods` supports on an integer
recording -- its `mode="noise"` raises on anything but float32/float64, and
`bandpass_filter` passes the parent dtype through, so an int16 SpikeGLX chain stays
int16 all the way down. Zeros are correct there in any case, since the traces are
zero-centred by that point. Pass silence_mode="noise" only if the chain casts to float
first (`bandpass_filter: {"dtype": "float32"}`); a long zeroed stretch has no variance,
which noise fill avoids.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import spikeinterface.full as si
from spikeinterface.core import BaseRecording


# A rail is a pile-up: the extreme value holds far more samples than the values below
# it. Measured ratios are ~1200x on real clipping, so 20x is a wide margin that an
# ordinary distribution tail cannot reach.
MIN_PILEUP_RATIO = 20.0

# ...and it must hold a non-trivial share of the data. Without this, a clean recording
# whose single largest sample happens to sit above an empty band would divide by zero
# and report a rail.
MIN_RAIL_FRAC = 1e-6

# The reference band, as a fraction of the candidate rail. Kept close to the rail so the
# comparison is against locally similar signal rather than the bulk of the distribution.
REF_BAND_FRAC = 0.95

# A channel is judged dead if it rails through this share of the samples in which the
# array as a whole is quiet.
DEAD_CHANNEL_FRAC = 0.5

# "Quiet" means at most this fraction of channels railed. Above it, a channel sitting at
# the rail says something about the session rather than about the channel.
QUIET_FRAC = 0.02

# A saturation epoch is a run of samples with at least this fraction of channels railed.
EPOCH_FRAC = 0.02

# Runs closer together than this are merged. The gap is too short to sort across, and
# leaving it unblanked would anchor the fill on railed samples.
MERGE_GAP_MS = 10.0

# Padding added to each epoch. The bandpass runs filtfilt, so a clipped edge rings
# outward in both directions; blanking only the railed samples leaves that behind.
# 10 ms comfortably covers the ~5 ms ring-down of a 300 Hz highpass.
EPOCH_MARGIN_MS = 10.0

# Sequential read size for the full scan, matching preprocess_qc.
SCAN_CHUNK_S = 10.0

# How silence_periods fills the blanked spans. "zeros" is the only mode that works on an
# integer recording -- "noise" raises on anything but float32/float64 -- and the traces
# are zero-centred by the time the step runs, so zeros are appropriate there.
DEFAULT_SILENCE_MODE = 'zeros'

# Default sampling for the histogram and dead-channel passes: probes spread evenly over
# the whole recording rather than clustered randomly.
DEFAULT_PROBES = dict(n_probes=300, probe_s=0.05)


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------

def _probe_starts(rec, n_probes, probe_s):
    """Probe start frames spread evenly over the whole recording.

    Evenly spread, not random: clipping is often confined to one block of a session
    (on DS13 it is the last 12%), and a handful of random chunks can miss such a block
    and report the recording clean. Even spacing intersects any block longer than the
    gap between probes.
    """
    n = rec.get_num_samples()
    w = max(min(int(round(probe_s * rec.get_sampling_frequency())), n), 1)
    return np.linspace(0, max(n - w, 0), n_probes).astype(np.int64), w


def max_int_from_meta(rec):
    """imMaxInt from the SpikeGLX meta beside the recording, or None.

    Only ever used for context in the report -- the rail is measured, because it sits
    below ADC full scale. Read from the meta rather than passed in so nothing has to be
    hardcoded per session, and defensive throughout: a missing or unparseable meta must
    degrade the report, never break the scan.
    """
    seen = set()
    stack = [rec]
    while stack:
        r = stack.pop()
        if id(r) in seen:
            continue
        seen.add(id(r))
        folder = (getattr(r, '_kwargs', None) or {}).get('folder_path')
        if folder:
            try:
                for meta in sorted(Path(folder).glob('**/*.ap.meta')):
                    for line in meta.read_text().splitlines():
                        if line.startswith('imMaxInt='):
                            return int(line.split('=', 1)[1])
            except Exception:
                pass
        for attr in ('_parent', 'parent_recording', '_recording'):
            par = getattr(r, attr, None)
            if isinstance(par, BaseRecording):
                stack.append(par)
        kw = getattr(r, '_kwargs', {}) or {}
        for v in kw.values():
            if isinstance(v, BaseRecording):
                stack.append(v)
    return None


def value_histograms(rec, n_probes=None, probe_s=None):
    """Per-channel value histograms, from probes spread across the whole recording.

    Returns (mag_hist, signed_hist, offset, n_sampled_per_channel).

    Both are built in the same pass, because they answer different questions. Rail
    detection uses magnitudes: an analogue rail clips both directions at the same level,
    so folding doubles the evidence for one decision. The report plots the signed
    histogram, because folding hides an asymmetry that is worth seeing -- on one session
    the negative rail held twice as many samples as the positive one.

    signed_hist is indexed by value + offset.
    """
    cfg = dict(DEFAULT_PROBES)
    if n_probes is not None:
        cfg['n_probes'] = n_probes
    if probe_s is not None:
        cfg['probe_s'] = probe_s
    dtype = np.dtype(rec.get_dtype())
    if not np.issubdtype(dtype, np.integer):
        raise TypeError(f'value_histograms needs an integer recording, got {dtype}. '
                        'Rails are a property of the acquisition hardware, so run the '
                        'detection on the raw recording.')
    n_bins = int(np.iinfo(dtype).max) + 1
    n_ch = rec.get_num_channels()

    starts, w = _probe_starts(rec, cfg['n_probes'], cfg['probe_s'])
    n_signed = 2 * n_bins            # value + n_bins keeps every int16 in range
    flat = np.zeros(n_ch * n_bins, dtype=np.int64)
    flat_s = np.zeros(n_ch * n_signed, dtype=np.int64)
    ch = np.arange(n_ch, dtype=np.int64)[None, :]
    n_per_ch = 0
    for s in starts:
        x = np.asarray(rec.get_traces(start_frame=int(s), end_frame=int(s) + w,
                                      return_in_uV=False))
        v = x.astype(np.int64, copy=False)
        a = np.abs(v)
        flat += np.bincount((ch * n_bins + a).ravel(), minlength=n_ch * n_bins)
        flat_s += np.bincount((ch * n_signed + v + n_bins).ravel(),
                              minlength=n_ch * n_signed)
        n_per_ch += x.shape[0]
    hist = flat.reshape(n_ch, n_bins)
    signed = flat_s.reshape(n_ch, n_signed)
    # a full int16 range is (n_ch, 32768) int64 = 100 MB, nearly all of it empty on a
    # 12-bit ADC; keep only up to the largest magnitude actually seen
    occupied = np.flatnonzero(hist.sum(axis=0))
    peak = int(occupied[-1]) if occupied.size else 0
    hist = hist[:, :peak + 1]
    # keep the signed histogram symmetric about zero, over the same range
    lo, hi = n_bins - peak, n_bins + peak + 1
    signed = signed[:, lo:hi]
    return (np.ascontiguousarray(hist), np.ascontiguousarray(signed), peak, n_per_ch)


# ---------------------------------------------------------------------------
# rail detection
# ---------------------------------------------------------------------------

def find_rail(rec, hist=None, n_sampled=None, min_pileup_ratio=MIN_PILEUP_RATIO,
              min_rail_frac=MIN_RAIL_FRAC, ref_band_frac=REF_BAND_FRAC,
              max_int=None, verbose=True, **probe_kwargs):
    """The clipping level in raw counts, or None if the recording does not clip.

    max_int (e.g. imMaxInt from the SpikeGLX meta) is only recorded for context; the
    rail is always measured, because it sits below ADC full scale.

    Returns (rail, info), info carrying the evidence for the decision.
    """
    dtype = np.dtype(rec.get_dtype())
    if not np.issubdtype(dtype, np.integer):
        return None, {'reason': f'non-integer dtype ({dtype}): rails are undefined'}

    if hist is None:
        hist, _signed, _off, n_sampled = value_histograms(rec, **probe_kwargs)
    total = hist.sum(axis=0)
    n_tot = int(total.sum())
    nz = np.flatnonzero(total)
    if nz.size == 0 or n_tot == 0:
        return None, {'reason': 'no samples read'}

    peak = int(nz[-1])                                  # highest magnitude present
    lo = max(int(np.floor(ref_band_frac * peak)), 0)
    band = total[lo:peak]
    band = band[band > 0]
    ref = float(np.median(band)) if band.size else 0.0
    n_at = int(total[peak])
    frac = n_at / n_tot
    ratio = (n_at / ref) if ref > 0 else np.inf

    info = {'peak_value': peak, 'n_at_peak': n_at, 'reference_count': ref,
            'pileup_ratio': float(ratio), 'frac_at_peak': float(frac),
            'min_pileup_ratio': float(min_pileup_ratio),
            'min_rail_frac': float(min_rail_frac),
            'dtype_rail': int(np.iinfo(dtype).max),
            'sampled_s_per_channel': (n_sampled or 0) / rec.get_sampling_frequency()}
    if max_int is not None:
        info['max_int'] = int(max_int)
        info['peak_over_max_int'] = peak / float(max_int)

    # both tests must pass: a large pile-up ratio alone can come from dividing by an
    # empty reference band, and a large share alone can come from a quiet recording
    if ratio < min_pileup_ratio or frac < min_rail_frac:
        info['reason'] = (f'largest magnitude {peak} holds {frac:.2e} of samples at '
                          f'{ratio:.1f}x its neighbours (need {min_rail_frac:.0e} and '
                          f'{min_pileup_ratio:g}x): no clipping')
        if verbose:
            print(f'find_rail: no clipping detected -- {info["reason"]}')
        return None, info

    if verbose:
        print(f'find_rail: rail at +/-{peak} counts ({ratio:.0f}x pile-up, '
              f'{100 * frac:.3f}% of sampled data)')
    return peak, info


# ---------------------------------------------------------------------------
# dead channels
# ---------------------------------------------------------------------------

def find_dead_channels(rec, rail, dead_frac=DEAD_CHANNEL_FRAC, quiet_frac=QUIET_FRAC,
                       verbose=True, **probe_kwargs):
    """Channel indices that rail while the rest of the array is quiet.

    A broken channel rails regardless of what the session is doing; a healthy channel
    rails only inside a global event. Restricting the tally to quiet samples separates
    the two independently of how much of the recording is affected.

    Returns (dead_inds, frac_per_channel, n_quiet_samples).
    """
    cfg = dict(DEFAULT_PROBES)
    cfg.update({k: v for k, v in probe_kwargs.items() if v is not None})
    starts, w = _probe_starts(rec, cfg['n_probes'], cfg['probe_s'])
    n_ch = rec.get_num_channels()
    quiet_max = max(int(np.floor(quiet_frac * n_ch)), 1)

    railed_in_quiet = np.zeros(n_ch, dtype=np.int64)
    n_quiet = 0
    for s in starts:
        x = np.asarray(rec.get_traces(start_frame=int(s), end_frame=int(s) + w,
                                      return_in_uV=False))
        clipped = np.abs(x) >= rail
        quiet = clipped.sum(axis=1) <= quiet_max
        if quiet.any():
            railed_in_quiet += clipped[quiet].sum(axis=0)
            n_quiet += int(quiet.sum())

    if n_quiet == 0:
        # every sampled moment had much of the array railed, so nothing here is
        # attributable to individual channels
        if verbose:
            print('find_dead_channels: no quiet samples found; skipping the test')
        return np.array([], dtype=int), np.zeros(n_ch), 0

    frac = railed_in_quiet / n_quiet
    dead = np.flatnonzero(frac > dead_frac)
    if verbose and dead.size:
        ids = np.asarray(rec.channel_ids)[dead]
        print(f'find_dead_channels: {dead.size} channel(s) railed while the array was '
              f'quiet: {ids.tolist()}')
    return dead, frac, n_quiet


# ---------------------------------------------------------------------------
# saturation epochs
# ---------------------------------------------------------------------------

def find_saturation_epochs(rec, rail, exclude_channels=(), epoch_frac=EPOCH_FRAC,
                           chunk_s=SCAN_CHUNK_S, verbose=True):
    """Frame intervals in which at least epoch_frac of the channels rail together.

    The whole recording is read sequentially rather than sampled: the point of the scan
    is to find every occurrence, not to estimate a rate, and a clipped stretch a few
    milliseconds long will not land in a handful of random chunks. Runs are stitched
    across chunk boundaries so an epoch straddling one is reported once.

    Returns (epochs, clipped_fraction, channel_clipped_frac). The per-channel numbers
    are exact rather than sampled -- this pass already touches every sample, so counting
    per channel while here costs nothing and is what makes the report able to say which
    channels carry the clipping.
    """
    n_ch = rec.get_num_channels()
    keep = np.setdiff1d(np.arange(n_ch), np.asarray(list(exclude_channels), dtype=int))
    if keep.size == 0:
        return np.zeros((0, 2), dtype=np.int64), 0.0, np.zeros(n_ch)
    need = max(int(np.ceil(epoch_frac * keep.size)), 1)

    fs = rec.get_sampling_frequency()
    n_samp = rec.get_num_samples()
    step = max(int(chunk_s * fs), 1)
    n_steps = int(np.ceil(n_samp / step))

    epochs = []
    open_start = None          # start frame of a run still open at a chunk boundary
    n_railed = 0
    ch_railed = np.zeros(n_ch, dtype=np.int64)
    for i in range(n_steps):
        a, b = i * step, min((i + 1) * step, n_samp)
        x = np.asarray(rec.get_traces(start_frame=a, end_frame=b, return_in_uV=False))
        at_rail = np.abs(x) >= rail
        ch_railed += at_rail.sum(axis=0)
        clipped = at_rail[:, keep]
        n_railed += int(clipped.sum())
        hot = (clipped.sum(axis=1) >= need).astype(np.int8)

        # Prepending the carried-over state makes a run that began in an earlier chunk
        # continue rather than re-open here. Nothing is appended: a run still hot at the
        # last sample has not ended, and closing it at the chunk edge would report one
        # epoch as two.
        prev = np.int8(1) if open_start is not None else np.int8(0)
        edges = np.diff(np.concatenate(([prev], hot)))
        # rises and falls strictly alternate, so walking them in time order pairs them
        events = sorted([(int(i), 1) for i in np.flatnonzero(edges == 1)]
                        + [(int(i), 0) for i in np.flatnonzero(edges == -1)])
        for i, is_rise in events:
            if is_rise:
                open_start = a + i
            elif open_start is not None:
                epochs.append((open_start, a + i))
                open_start = None
        if verbose and (i % 50 == 0 or i == n_steps - 1):
            print(f'  saturation scan {i + 1}/{n_steps} chunks, {len(epochs)} epochs',
                  end='\r')
    if open_start is not None:
        epochs.append((open_start, n_samp))
    if verbose:
        print()

    epochs = np.asarray(epochs, dtype=np.int64).reshape(-1, 2)
    return (epochs, n_railed / max(n_samp * keep.size, 1),
            ch_railed / max(n_samp, 1))


def merge_and_pad(epochs, n_samp, fs, merge_gap_ms=MERGE_GAP_MS,
                  margin_ms=EPOCH_MARGIN_MS):
    """Pad each epoch, then coalesce epochs that overlap or nearly touch.

    Padding before merging means a gap the padding closes is treated as one epoch,
    which is what keeps a fill from being anchored on railed samples.
    """
    epochs = np.asarray(epochs, dtype=np.int64).reshape(-1, 2)
    if epochs.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int64)
    margin = int(round(margin_ms * fs / 1000.0))
    gap = int(round(merge_gap_ms * fs / 1000.0))

    e = epochs.copy()
    e[:, 0] = np.maximum(e[:, 0] - margin, 0)
    e[:, 1] = np.minimum(e[:, 1] + margin, n_samp)
    e = e[np.argsort(e[:, 0])]

    out = [[int(e[0, 0]), int(e[0, 1])]]
    for s, t in e[1:]:
        if int(s) - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], int(t))
        else:
            out.append([int(s), int(t)])
    return np.asarray(out, dtype=np.int64)


def residual_clipping(rec, periods, rail, exclude_channels=(), **probe_kwargs):
    """How much clipping is left outside the blanked periods.

    The epoch detector fires on a *fraction* of channels railing together, so a handful
    of channels can rail below that threshold and survive the blanking. This is the one
    number that says whether that leftover matters: if it is negligible the two
    mechanisms -- dropping dead channels and blanking epochs -- have covered the
    problem, and if it is not, epoch_frac is too high or margin_ms too small.

    Sampled rather than exhaustive: an order of magnitude is enough to make the call,
    and the exhaustive pass has already run.

    Returns (residual_fraction, n_samples_examined).
    """
    cfg = dict(DEFAULT_PROBES)
    cfg.update({k: v for k, v in probe_kwargs.items() if v is not None})
    n_ch = rec.get_num_channels()
    keep = np.setdiff1d(np.arange(n_ch), np.asarray(list(exclude_channels), dtype=int))
    if keep.size == 0:
        return 0.0, 0

    periods = np.asarray(periods, dtype=np.int64).reshape(-1, 2)
    starts, w = _probe_starts(rec, cfg['n_probes'], cfg['probe_s'])
    n_clipped = 0
    n_seen = 0
    for s in starts:
        s = int(s)
        frames = np.arange(s, s + w)
        if periods.shape[0]:
            # a frame is blanked if the period starting at or before it also ends after
            i = np.searchsorted(periods[:, 0], frames, side='right') - 1
            inside = (i >= 0) & (frames < periods[np.maximum(i, 0), 1])
        else:
            inside = np.zeros(w, dtype=bool)
        if inside.all():
            continue
        x = np.asarray(rec.get_traces(start_frame=s, end_frame=s + w,
                                      return_in_uV=False))
        x = x[~inside][:, keep]
        n_clipped += int((np.abs(x) >= rail).sum())
        n_seen += x.size
    return (n_clipped / n_seen if n_seen else 0.0), n_seen


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def detect_saturation(rec, rail=None, max_int=None, epoch_frac=EPOCH_FRAC,
                      merge_gap_ms=MERGE_GAP_MS, margin_ms=EPOCH_MARGIN_MS,
                      chunk_s=SCAN_CHUNK_S, silence_mode=DEFAULT_SILENCE_MODE,
                      check_residual=True, verbose=True, **probe_kwargs):
    """Measure the rail, split dead channels from saturation epochs, return both.

    Returns a dict with
        rail                    clipping level in counts, or None
        dead_channel_inds/_ids  channels railed independently of the session
        periods                 (n, 2) merged, padded frame intervals to blank
        silence_periods_kwargs  ready to drop into the preprocessing protocol
        clipped_fraction        share of samples at the rail, dead channels excluded
        blanked_fraction        share of the recording the periods cover
        residual_clipped_frac   clipping left outside the periods (sampled)
        hist                    per-channel magnitude histogram, for plotting
        summary                 greppable text report
    """
    fs = rec.get_sampling_frequency()
    n_samp = rec.get_num_samples()
    chan_ids = np.asarray(rec.channel_ids)
    rep = {'n_channels': int(len(chan_ids)), 'duration_s': n_samp / fs,
           'sampling_frequency': float(fs),
           'channel_ids': [str(c) for c in chan_ids]}
    if max_int is None:
        max_int = max_int_from_meta(rec)

    # The rail is a property of the acquisition hardware, so the scan only means
    # anything on the raw integer traces; a float recording has been scaled or filtered
    # and no longer has one. Checked here rather than deeper down so the answer is a
    # report saying so, not an exception from np.iinfo.
    if not np.issubdtype(np.dtype(rec.get_dtype()), np.integer):
        rep['hist'] = np.zeros((rep['n_channels'], 1), dtype=np.int64)
        rep['signed_hist'] = np.zeros((rep['n_channels'], 1), dtype=np.int64)
        rep['signed_offset'] = 0
        rep['n_sampled_per_channel'] = 0
        rail, rail_info = None, {
            'reason': f'non-integer dtype ({rec.get_dtype()}): rails are undefined, '
                      'run detect_saturation on the raw recording'}
    else:
        hist, signed, offset, n_sampled = value_histograms(rec, **probe_kwargs)
        rep['hist'] = hist
        rep['signed_hist'] = signed
        rep['signed_offset'] = int(offset)
        rep['n_sampled_per_channel'] = n_sampled
        if rail is None:
            rail, rail_info = find_rail(rec, hist=hist, n_sampled=n_sampled,
                                        max_int=max_int, verbose=verbose)
        else:
            rail_info = {'reason': 'rail supplied by caller', 'peak_value': int(rail)}
    rep['rail'] = None if rail is None else int(rail)
    rep['rail_info'] = rail_info

    if rail is None:
        rep.update(dead_channel_inds=np.array([], dtype=int), dead_channel_ids=[],
                   dead_channel_frac=np.zeros(len(chan_ids)),
                   periods=np.zeros((0, 2), dtype=np.int64), n_periods=0,
                   epochs_raw=0, silence_periods_kwargs=None,
                   clipped_fraction=0.0, blanked_fraction=0.0,
                   residual_clipped_frac=0.0,
                   channel_clipped_frac=np.zeros(len(chan_ids)))
        rep['summary'] = _summary(rep, rec)
        return rep

    dead, dead_frac, n_quiet = find_dead_channels(rec, rail, verbose=verbose,
                                                  **probe_kwargs)
    rep['dead_channel_inds'] = dead
    rep['dead_channel_ids'] = [str(c) for c in chan_ids[dead]]
    rep['dead_channel_frac'] = dead_frac
    rep['n_quiet_samples'] = n_quiet

    raw_epochs, clipped_frac, ch_frac = find_saturation_epochs(
        rec, rail, exclude_channels=dead, epoch_frac=epoch_frac, chunk_s=chunk_s,
        verbose=verbose)
    rep['epochs_raw'] = int(len(raw_epochs))
    rep['clipped_fraction'] = float(clipped_frac)
    rep['channel_clipped_frac'] = ch_frac
    if len(raw_epochs):
        d_ms = (raw_epochs[:, 1] - raw_epochs[:, 0]) / fs * 1000.0
        rep['epoch_ms'] = {'median': float(np.median(d_ms)),
                           'p90': float(np.percentile(d_ms, 90)),
                           'p99': float(np.percentile(d_ms, 99)),
                           'max': float(d_ms.max())}

    periods = merge_and_pad(raw_epochs, n_samp, fs, merge_gap_ms, margin_ms)
    rep['periods'] = periods
    rep['n_periods'] = int(len(periods))
    covered = int((periods[:, 1] - periods[:, 0]).sum()) if len(periods) else 0
    rep['blanked_fraction'] = covered / n_samp
    if len(periods):
        # Where the blanking actually sits. First-to-last period is a bad summary --
        # a few isolated events at either end make it span the session regardless of
        # where the problem is. The shortest window holding most of the blanked time
        # answers the question that matters: is there a bad stretch to exclude, or is
        # this spread throughout?
        rep['span_s'] = [float(periods[0, 0] / fs), float(periods[-1, 1] / fs)]
        rep['core_span_s'], rep['core_frac'] = _core_span(periods, fs)

    if check_residual and len(periods):
        resid, n_seen = residual_clipping(rec, periods, rail, dead, **probe_kwargs)
        rep['residual_clipped_frac'] = float(resid)
        rep['residual_n_samples'] = int(n_seen)
    else:
        rep['residual_clipped_frac'] = None

    rep['silence_periods_kwargs'] = (
        {'list_periods': [[(int(s), int(t)) for s, t in periods]],
         'mode': silence_mode}
        if len(periods) else None)
    # the summary is returned rather than printed: the caller prints it, so a cached
    # scan and a fresh one produce identical output
    rep['summary'] = _summary(rep, rec)
    return rep


def detect_saturation_cached(rec, cache_folder, overwrite=False, verbose=True,
                             **kwargs):
    """detect_saturation with the full scan cached, since it reads the whole file.

    The cache holds the periods, the dead channels, the rail and the trimmed histogram
    -- everything the protocol and the plots need -- so a re-run of the pipeline that is
    not redoing preprocessing never touches the raw file again.
    """
    cache_folder = Path(cache_folder)
    npz = cache_folder / 'saturation.npz'
    if npz.exists() and not overwrite:
        try:
            z = np.load(npz, allow_pickle=False)
            rail = None if bool(z['rail_is_none']) else int(z['rail'])
            rep = {'rail': rail,
                   'periods': z['periods'],
                   'n_periods': int(z['periods'].shape[0]),
                   'dead_channel_inds': z['dead_channel_inds'],
                   'dead_channel_ids': [str(c) for c in z['dead_channel_ids']],
                   'hist': z['hist'],
                   'signed_hist': z['signed_hist'],
                   'signed_offset': int(z['signed_offset']),
                   'n_channels': int(z['n_channels']),
                   'duration_s': float(z['duration_s']),
                   'sampling_frequency': float(z['sampling_frequency']),
                   'n_sampled_per_channel': int(z['n_sampled_per_channel']),
                   'channel_clipped_frac': z['channel_clipped_frac'],
                   'channel_ids': [str(c) for c in z['channel_ids']],
                   'clipped_fraction': float(z['clipped_fraction']),
                   'blanked_fraction': float(z['blanked_fraction']),
                   'residual_clipped_frac': float(z['residual_clipped_frac']),
                   'epochs_raw': int(z['epochs_raw']),
                   'span_s': z['span_s'].tolist() if z['span_s'].size else None,
                   'core_span_s': (z['core_span_s'].tolist()
                                   if z['core_span_s'].size else None),
                   'core_frac': float(z['core_frac']),
                   'rail_info': {'reason': 'loaded from cache'},
                   'from_cache': True}
            rep['silence_periods_kwargs'] = (
                {'list_periods': [[(int(a), int(b)) for a, b in rep['periods']]],
                 'mode': str(z['silence_mode'])} if rep['n_periods'] else None)
            for k in ('span_s', 'core_span_s'):
                if rep.get(k) is None:
                    rep.pop(k, None)
            txt = cache_folder / 'saturation_summary.txt'
            rep['summary'] = txt.read_text() if txt.exists() else ''
            if verbose:
                print(f'detect_saturation: loaded cached scan from {npz}')
            return rep
        except Exception as e:
            print(f'detect_saturation: cache at {npz} unreadable ({e}); rescanning')

    rep = detect_saturation(rec, verbose=verbose, **kwargs)
    cache_folder.mkdir(parents=True, exist_ok=True)
    mode = (rep['silence_periods_kwargs'] or {}).get('mode', DEFAULT_SILENCE_MODE)
    np.savez_compressed(
        npz,
        rail=np.int64(rep['rail'] if rep['rail'] is not None else -1),
        rail_is_none=np.bool_(rep['rail'] is None),
        periods=rep['periods'],
        dead_channel_inds=np.asarray(rep['dead_channel_inds'], dtype=np.int64),
        dead_channel_ids=np.asarray(rep['dead_channel_ids'], dtype='U64'),
        hist=rep['hist'],
        signed_hist=rep['signed_hist'],
        signed_offset=np.int64(rep['signed_offset']),
        n_channels=np.int64(rep['n_channels']),
        duration_s=np.float64(rep['duration_s']),
        sampling_frequency=np.float64(rep['sampling_frequency']),
        n_sampled_per_channel=np.int64(rep['n_sampled_per_channel']),
        channel_clipped_frac=np.asarray(rep['channel_clipped_frac'], dtype=np.float64),
        channel_ids=np.asarray(rep['channel_ids'], dtype='U64'),
        clipped_fraction=np.float64(rep['clipped_fraction']),
        blanked_fraction=np.float64(rep['blanked_fraction']),
        residual_clipped_frac=np.float64(rep.get('residual_clipped_frac') or 0.0),
        epochs_raw=np.int64(rep['epochs_raw']),
        span_s=np.asarray(rep.get('span_s') or [], dtype=np.float64),
        core_span_s=np.asarray(rep.get('core_span_s') or [], dtype=np.float64),
        core_frac=np.float64(rep.get('core_frac') or 0.0),
        silence_mode=np.asarray(mode, dtype='U16'),
    )
    (cache_folder / 'saturation_summary.txt').write_text(rep['summary'])
    rep['from_cache'] = False
    return rep


def write_saturation_report(rec, rep, folder, prefix='saturation'):
    """Write the whole raw-data saturation report: text, CSV, JSON and plots.

    Emitted whether or not clipping was found -- "this session is clean" is a result
    worth having on disk, and the absence of a report is otherwise indistinguishable
    from the scan never having run.

    Returns the list of paths written.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    written = []

    txt = folder / f'{prefix}_summary.txt'
    txt.write_text(rep['summary'])
    written.append(txt)

    # scalars, for anything that wants to read this back programmatically
    uv = _uV_per_count(rec)
    js = {k: rep.get(k) for k in
          ('rail', 'n_channels', 'duration_s', 'sampling_frequency',
           'clipped_fraction', 'blanked_fraction', 'residual_clipped_frac',
           'epochs_raw', 'n_periods', 'dead_channel_ids', 'span_s', 'core_span_s',
           'core_frac', 'epoch_ms')}
    js['rail_uV'] = None if (rep['rail'] is None or uv is None) else rep['rail'] * uv
    js['uV_per_count'] = uv
    js['rail_info'] = {k: v for k, v in rep.get('rail_info', {}).items()}
    jsp = folder / f'{prefix}.json'
    jsp.write_text(json.dumps(_jsonable(js), indent=2))
    written.append(jsp)

    # per-channel table: the thing that says whether clipping is a channel problem or a
    # session problem
    cf = np.asarray(rep.get('channel_clipped_frac', np.zeros(rep['n_channels'])))
    ids = np.asarray(rec.channel_ids)
    try:
        depth = np.asarray(rec.get_channel_locations())[:, 1]
    except Exception:
        depth = np.full(len(ids), np.nan)
    dead = set(int(c) for c in np.asarray(rep['dead_channel_inds'], dtype=int))
    lines = ['channel_index,channel_id,depth_um,clipped_fraction,clipped_seconds,is_dead']
    for i in range(len(ids)):
        lines.append(f'{i},{ids[i]},{depth[i]:.1f},{cf[i]:.6e},'
                     f'{cf[i] * rep["duration_s"]:.4f},{int(i in dead)}')
    csv = folder / f'{prefix}_per_channel.csv'
    csv.write_text('\n'.join(lines) + '\n')
    written.append(csv)

    written.append(plot_saturation_diagnostics(rec, rep, folder,
                                               name=f'{prefix}_histogram'))
    if rep['n_periods']:
        written.append(plot_saturation_timeline(rec, rep, folder,
                                                name=f'{prefix}_timeline'))
    return written


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def _core_span(periods, fs, cover=0.9):
    """Shortest contiguous window containing `cover` of the total blanked time.

    Returns ((start_s, end_s), fraction_of_blanked_time_inside).
    """
    dur = (periods[:, 1] - periods[:, 0]).astype(np.float64)
    total = dur.sum()
    if total <= 0:
        return (0.0, 0.0), 0.0
    need = cover * total
    best = None
    acc = 0.0
    lo = 0
    for hi in range(len(periods)):
        acc += dur[hi]
        # shrink from the left while the window still covers enough
        while acc - dur[lo] >= need:
            acc -= dur[lo]
            lo += 1
        if acc >= need:
            width = (periods[hi, 1] - periods[lo, 0]) / fs
            if best is None or width < best[0]:
                best = (width, periods[lo, 0] / fs, periods[hi, 1] / fs, acc / total)
    if best is None:
        return (float(periods[0, 0] / fs), float(periods[-1, 1] / fs)), 1.0
    return (float(best[1]), float(best[2])), float(best[3])


def _uV_per_count(rec):
    """The recording's counts-to-uV scale, if it is uniform across channels."""
    g = rec.get_property('gain_to_uV')
    if g is None:
        return None
    g = np.asarray(g)
    return float(g[0]) if np.unique(g).size == 1 else None


def _summary(rep, rec):
    L = ['SATURATION SCAN', '=' * 62,
         f"recording: {rep['n_channels']} ch, {rep['duration_s']:.0f} s, "
         f"{rep['sampling_frequency']:.0f} Hz"]
    ri = rep['rail_info']

    if rep['rail'] is None:
        L += [f"rail: none detected -- {ri.get('reason', '')}",
              'no clipping: no channels to drop and nothing to blank']
        return '\n'.join(L) + '\n'

    L.append(f"rail: +/-{rep['rail']} counts"
             + (f"   ({ri['pileup_ratio']:.0f}x pile-up vs the values below it, "
                f"{100 * ri['frac_at_peak']:.3f}% of sampled data)"
                if 'pileup_ratio' in ri else ''))
    if 'dtype_rail' in ri:
        L.append(f"  the {rec.get_dtype()} rail would be {ri['dtype_rail']}: a "
                 'dtype-based test finds nothing here')
    if 'max_int' in ri:
        L.append(f"  ADC full scale (imMaxInt) {ri['max_int']}: the amplifier clips "
                 f"{100 * (1 - ri['peak_over_max_int']):.1f}% inside it")
    uv = _uV_per_count(rec)
    if uv is not None:
        L.append(f"  = +/-{rep['rail'] * uv / 1000:.3f} mV at the electrode "
                 f"({uv:.4f} uV/count)")

    n_dead = len(rep['dead_channel_ids'])
    L.append(f"dead channels (railed while the array was quiet): {n_dead}"
             + (f"  {rep['dead_channel_ids']}" if n_dead else ''))
    if n_dead:
        L.append('  -> remove these as channels; they are not artifacts to blank')

    L.append(f"clipped samples (dead channels excluded): {rep['clipped_fraction']:.5f}")

    cf = rep.get('channel_clipped_frac')
    if cf is not None and len(cf):
        cf = np.asarray(cf)
        dur = rep['duration_s']
        live = np.setdiff1d(np.arange(len(cf)),
                            np.asarray(rep['dead_channel_inds'], dtype=int))
        lv = cf[live] if live.size else cf
        L.append('per-channel clipped fraction (exact, from the full scan):')
        L.append('  live channels: ' + '  '.join(
            f'p{q}={np.percentile(lv, q):.2e}' for q in (50, 90, 99)) +
            f'  max={lv.max():.2e}')
        L.append(f"  channels with any clipping: {int((lv > 0).sum())}/{live.size}")
        ids = np.asarray(rep.get('channel_ids', np.arange(len(cf))))
        worst = live[np.argsort(-cf[live])[:5]] if live.size else []
        for c in worst:
            if cf[c] <= 0:
                break
            L.append(f"    {ids[c]}: {cf[c]:.2e} ({cf[c] * dur:.2f} s of {dur:.0f} s)")
        for c in np.asarray(rep['dead_channel_inds'], dtype=int):
            L.append(f"    {ids[c]}: {cf[c]:.3f}  <-- dead, removed as a channel")
    L.append(f"saturation epochs: {rep['epochs_raw']} raw -> {rep['n_periods']} "
             f"after merge+pad")
    if 'epoch_ms' in rep:
        e = rep['epoch_ms']
        L.append(f"  raw epoch duration ms: median={e['median']:.1f} p90={e['p90']:.1f} "
                 f"p99={e['p99']:.1f} max={e['max']:.1f}")
    L.append(f"blanked share of the recording: {rep['blanked_fraction']:.4f} "
             f"({rep['blanked_fraction'] * rep['duration_s']:.0f} s)")
    if 'span_s' in rep:
        s0, s1 = rep['span_s']
        L.append(f"  periods run from t = {s0:.0f} s to t = {s1:.0f} s")
    if 'core_span_s' in rep:
        c0, c1 = rep['core_span_s']
        width = max(c1 - c0, 1e-9)
        blanked_s = rep['blanked_fraction'] * rep['duration_s']
        dens = rep['core_frac'] * blanked_s / width
        L.append(f"  {100 * rep['core_frac']:.0f}% of the blanked time falls in "
                 f"t = {c0:.0f} .. {c1:.0f} s "
                 f"({100 * width / rep['duration_s']:.0f}% of the session, "
                 f"{100 * dens:.0f}% of it blanked)")
        if width < 0.35 * rep['duration_s'] and dens > 0.10:
            L.append('  -> the problem is concentrated: excluding that stretch removes '
                     'most of it, and is worth weighing against blanking')
        else:
            L.append('  -> spread through the session: there is no stretch to exclude, '
                     'so blanking is the option')
    r = rep.get('residual_clipped_frac')
    if r is not None:
        L.append(f"clipping left outside the blanked periods: {r:.2e}"
                 + ('   <-- negligible: dropping the dead channels and blanking these '
                    'periods covers it' if r < 1e-4 else
                    '   <-- still substantial: lower epoch_frac or raise margin_ms'))
    return '\n'.join(L) + '\n'


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------

def plot_saturation_diagnostics(rec, rep, save_folder=None, name='qc_saturation_hist',
                                ref_band_frac=REF_BAND_FRAC):
    """The value-histogram evidence for the rail, and which channels are dead.

    Four panels:
      1  per-channel signed histogram as channel x voltage, log counts. A dead channel
         is an empty row -- every one of its samples sits at the rail, so every other
         bin is unoccupied.
      2  the voltage histogram itself: every channel overlaid plus the array total, log
         counts, signed so an asymmetry between the two rails is visible.
      3  the top of the range, magnitudes folded. This is the decision: a flat or
         falling baseline, a spike at the rail, and exactly zero above it.
      4  per-channel clipped fraction against depth, separating a bad channel from a
         bad session.
    """
    hist = rep['hist']
    n_ch, n_bins = hist.shape
    mag = np.arange(n_bins)
    uv = _uV_per_count(rec)
    x = mag * uv / 1000.0 if uv is not None else mag.astype(float)
    total = hist.sum(axis=0)
    rail = rep['rail']
    max_int = rep['rail_info'].get('max_int')

    # signed histogram for the first two panels; magnitudes only for the decision panel
    signed = rep.get('signed_hist')
    if signed is not None and signed.shape[0] == n_ch and signed.shape[1] > 1:
        off = int(rep.get('signed_offset', (signed.shape[1] - 1) // 2))
        sv = np.arange(signed.shape[1]) - off
    else:
        signed, sv = hist, mag
        off = 0
    sx = sv * uv / 1000.0 if uv is not None else sv.astype(float)
    xlabel = 'voltage (mV at electrode)' if uv is not None else 'value (counts)'

    cf = np.asarray(rep.get('channel_clipped_frac', np.zeros(n_ch)))
    n_panels = 4 if cf.size == n_ch else 3
    fig, axs = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4.6),
                            constrained_layout=True)

    # -- per-channel heatmap ---------------------------------------------------
    ax = axs[0]
    hm = np.ma.masked_where(signed == 0, signed)
    im = ax.pcolormesh(sx, np.arange(n_ch), hm, shading='auto', cmap='magma',
                       norm=LogNorm(vmin=1, vmax=max(signed.max(), 10)))
    ax.set_xlabel(xlabel)
    ax.set_ylabel('channel')
    ax.set_title(f'per-channel voltage histogram (log counts)\n'
                 f'{rep["n_sampled_per_channel"] / rep["sampling_frequency"]:.0f} s '
                 'sampled across the session')
    fig.colorbar(im, ax=ax, label='samples', fraction=0.046)
    for c in rep['dead_channel_inds']:
        ax.axhline(c, color='tab:cyan', lw=0.6, alpha=0.8)
    if len(rep['dead_channel_inds']):
        ax.plot([], [], color='tab:cyan', lw=0.8,
                label=f'dead (n={len(rep["dead_channel_inds"])})')
        ax.legend(fontsize=7, loc='upper right')

    # -- the voltage histogram, signed ----------------------------------------
    ax = axs[1]
    stot = signed.sum(axis=0)
    ax.plot(sx, np.maximum(signed.T, 0.5), lw=0.3, alpha=0.2, color='steelblue')
    ax.plot(sx, np.maximum(stot, 0.5), lw=1.4, color='k', label='all channels')
    if rail is not None:
        n_neg = int(stot[max(off - rail, 0)]) if off - rail >= 0 else 0
        n_pos = int(stot[min(off + rail, len(stot) - 1)])
        ax.set_xlabel(xlabel + f'\nrail counts: {n_neg:,} at -rail, {n_pos:,} at +rail')
    if max_int is not None:
        xm = max_int * uv / 1000.0 if uv is not None else max_int
        ax.axvline(-xm, color='tab:green', ls='--', lw=1)
        ax.axvline(xm, color='tab:green', ls='--', lw=1,
                   label=f'ADC full scale ({max_int})')
    if rail is not None:
        xr = rail * uv / 1000.0 if uv is not None else rail
        ax.axvline(-xr, color='tab:blue', lw=1.2)
        ax.axvline(xr, color='tab:blue', lw=1.2, label=f'rail (+/-{rail})')
    ax.set_yscale('log')
    ax.set_ylim(bottom=0.5)
    ax.set_ylabel('samples (log)')
    ax.set_title('voltage histogram, every channel overlaid')
    ax.legend(fontsize=7)

    # -- the decision, zoomed --
    ax = axs[2]
    peak = rep['rail_info'].get('peak_value', int(np.flatnonzero(total)[-1]))
    m = mag >= int(0.85 * peak)
    ax.plot(mag[m], np.maximum(total[m], 0.5), 'o-', ms=3, lw=1, color='crimson')
    ax.axvspan(int(ref_band_frac * peak), peak, color='0.85', zorder=0,
               label='reference band')
    if max_int is not None:
        ax.axvline(max_int, color='tab:green', ls='--', lw=1,
                   label=f'ADC full scale ({max_int})')
    if rail is not None:
        ri = rep['rail_info']
        ax.axvline(rail, color='tab:blue', lw=1.2,
                   label=f'rail {rail}'
                         + (f' ({rail * uv / 1000:.3f} mV)' if uv is not None else '')
                         + (f'\npile-up {ri["pileup_ratio"]:.0f}x'
                            if 'pileup_ratio' in ri else ''))
    ax.set_yscale('log')
    ax.set_ylim(bottom=0.5)
    ax.set_xlabel('|value| (raw counts)')
    ax.set_ylabel('samples (log)')
    ax.set_title('top of the range: spike at the rail, zero above it')
    ax.legend(fontsize=7)

    # -- per-channel clipping down the shank ------------------------------------
    if n_panels == 4:
        ax = axs[3]
        try:
            depth = np.asarray(rec.get_channel_locations())[:, 1]
        except Exception:
            depth = np.arange(n_ch, dtype=float)
        dead = np.asarray(rep['dead_channel_inds'], dtype=int)
        live = np.setdiff1d(np.arange(n_ch), dead)
        floor = 1.0 / max(rep['duration_s'] * rep['sampling_frequency'], 1.0)
        ax.plot(np.maximum(cf[live], floor), depth[live], '.', ms=4,
                color='tab:blue', label=f'live (n={live.size})')
        if dead.size:
            ax.plot(np.maximum(cf[dead], floor), depth[dead], 'x', ms=7,
                    color='tab:red', label=f'dead (n={dead.size})')
        ax.set_xscale('log')
        ax.set_xlabel('fraction of samples at the rail')
        ax.set_ylabel('depth (um)')
        ax.set_title('per-channel clipping vs depth\n'
                     'a lone outlier is a channel; a broad band is the session')
        ax.legend(fontsize=7)

    if save_folder is None:
        return fig
    save_folder.mkdir(parents=True, exist_ok=True)
    path = save_folder / f'{name}.png'
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_saturation_timeline(rec, rep, save_folder=None, name='qc_saturation_timeline',
                             n_bins=600):
    """Where in the session the blanked periods fall.

    Clipping confined to one block is a different decision from clipping spread
    throughout: the first invites excluding the block, the second has to be blanked.
    """
    fs = rep['sampling_frequency']
    dur = rep['duration_s']
    periods = rep['periods']
    edges = np.linspace(0, dur, n_bins + 1)
    occupancy = np.zeros(n_bins)
    for s, t in periods:
        a, b = s / fs, t / fs
        lo = np.searchsorted(edges, a, 'right') - 1
        hi = np.searchsorted(edges, b, 'left')
        for k in range(max(lo, 0), min(hi, n_bins)):
            occupancy[k] += (min(b, edges[k + 1]) - max(a, edges[k]))
    occupancy /= np.diff(edges)

    fig, ax = plt.subplots(figsize=(11, 3.2), constrained_layout=True)
    ax.fill_between(edges[:-1], 0, occupancy, step='post', color='tab:red', alpha=0.8)
    ax.set_xlim(0, dur)
    ax.set_ylim(0, 1)
    ax.set_xlabel('time in session (s)')
    ax.set_ylabel('blanked fraction')
    ax.set_title(f'saturation blanking over the session '
                 f'({rep["n_periods"]} periods, {rep["blanked_fraction"]:.3%} of the '
                 'recording)')
    if save_folder is None:
        return fig
    save_folder.mkdir(parents=True, exist_ok=True)
    path = save_folder / f'{name}.png'
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return path
