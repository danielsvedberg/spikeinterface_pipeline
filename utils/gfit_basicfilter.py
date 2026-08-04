"""
Python translation of gFitBasicFilter.m (low-pass filter version).

CORRECTED VERSION -- see notes at the bottom of this file for the list of
fixes relative to the first draft, and for the parameter sanity problem
that affects the ORIGINAL MATLAB too.
"""

import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import iirdesign, sosfiltfilt, sosfreqz, sos2zpk


# ----------------------------------------------------------------------
# MATLAB smooth(x, span, 'moving') equivalent
# ----------------------------------------------------------------------
def moving_average(x, span):
    """
    Equivalent to MATLAB's smooth(x, span, 'moving').

    MATLAB's endpoint rule is a SYMMETRIC shrinking window, not a truncated
    one:  yy(1) = y(1), yy(2) = mean(y(1:3)), yy(3) = mean(y(1:5)), ...
    i.e. the half-width at point i is min(i-1, n-i, span//2).

    O(n) via cumulative sums -- the naive per-sample loop is unusable at
    25 kHz (an hour of data is 9e7 samples).
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    span = int(span)
    if span % 2 == 0:
        span += 1              # MATLAB forces an odd span
    half = span // 2
    if n == 0:
        return x.copy()

    idx = np.arange(n)
    h = np.minimum(np.minimum(idx, n - 1 - idx), half)   # half-width per point
    lo = idx - h
    hi = idx + h + 1                                     # exclusive

    csum = np.concatenate(([0.0], np.cumsum(x)))
    return (csum[hi] - csum[lo]) / (hi - lo)


# ----------------------------------------------------------------------
# MATLAB lowpass(..., 'ImpulseResponse','iir','Steepness',s) equivalent
# ----------------------------------------------------------------------
def design_lowpass(cutoff_f, fs, steepness=0.95, stopband_attenuation=60,
                   passband_ripple=0.1, warn=True):
    """
    Approximate MATLAB's
        lowpass(x, fpass, fs, 'ImpulseResponse','iir','Steepness',s)

    MATLAB defines the transition width exactly as

        W = (0.99 - 0.98*s) * (f_nyquist - fpass)
        fstop = fpass + W

    and always uses a 0.1 dB passband ripple with a default 60 dB stopband
    attenuation, designing the minimum-order elliptic IIR filter that meets
    those specs. Returns second-order sections (what MATLAB's digitalFilter
    uses internally); transfer-function (b, a) form is numerically unusable
    at very low normalized cutoffs.
    """
    if not (0.5 <= steepness < 1):
        raise ValueError("steepness must be in [0.5, 1)")
    nyq = fs / 2
    if not (0 < cutoff_f < nyq):
        raise ValueError(f"cutoff_f must be in (0, {nyq})")

    W = (0.99 - 0.98 * steepness) * (nyq - cutoff_f)
    fstop = cutoff_f + W

    sos = iirdesign(cutoff_f / nyq, fstop / nyq,
                    passband_ripple, stopband_attenuation,
                    ftype='ellip', output='sos')

    if warn:
        _, poles, _ = sos2zpk(sos)
        rmax = np.abs(poles).max()
        if rmax > 1 - 1e-5:
            tau = 1.0 / (1.0 - rmax)
            warnings.warn(
                f"Filter poles sit {1 - rmax:.2e} from the unit circle "
                f"(settling time ~{tau:.3e} samples = {tau / fs / 60:.1f} min). "
                f"cutoff_f={cutoff_f:g} Hz at fs={fs:g} Hz is a normalized "
                f"frequency of {cutoff_f / nyq:.2e}. Unless your record is many "
                f"times longer than the settling time, the output will be an "
                f"edge-transient artifact rather than a baseline. "
                f"Consider detrending in the time domain instead.",
                RuntimeWarning, stacklevel=2)
    return sos


def _clean_window_mean(trace, flagged, lo, hi):
    """Mean of trace[lo:hi], ignoring any samples that are themselves flagged as artifact.

    Excluding the flagged samples is what keeps a second artifact lying inside the window --
    or the replacement values already written for one -- out of the anchor.
    Returns nan when the window is empty or holds nothing clean.
    """
    lo = max(0, int(lo))
    hi = min(trace.size, int(hi))
    if hi <= lo:
        return np.nan
    clean = ~flagged[lo:hi]
    if not clean.any():
        return np.nan
    return float(trace[lo:hi][clean].mean())


def interpolate_short_artifacts(ts, flagged, max_run, anchor_window):
    """Interpolate across each unbroken run of flagged samples shorter than max_run.

    Each run is replaced by a straight line drawn between the mean of the anchor_window clean
    samples before it and the mean of the anchor_window after it, so a run of any length lands
    on the local signal level at both ends. A one-sample run reduces to the mean of its
    neighbours, which is what the single-sample rule this generalises did.

    Averaging a window rather than reading the single abutting sample matters because the sample
    next to an artifact is the least trustworthy one available -- it sits on the artifact's own
    rising or falling edge.

    Runs of max_run or longer are left alone and reported back, since interpolating across a
    stretch that long would invent signal rather than repair it. A run reaching either end of
    the trace is filled from whichever side still has clean data.

    Modifies ts in place. Returns (n_fixed, long_runs), where long_runs lists the untouched
    runs as (start, stop_inclusive, length).
    """
    idx = np.flatnonzero(flagged)
    if idx.size == 0:
        return 0, []

    #split the flagged samples into unbroken runs
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[breaks + 1]]
    stops = np.r_[idx[breaks], idx[-1]]          #inclusive

    n_fixed = 0
    long_runs = []
    for start, stop in zip(starts, stops):
        length = stop - start + 1
        if length >= max_run:
            long_runs.append((int(start), int(stop), int(length)))
            continue

        before = _clean_window_mean(ts, flagged, start - anchor_window, start)
        after = _clean_window_mean(ts, flagged, stop + 1, stop + 1 + anchor_window)
        if np.isnan(before) and np.isnan(after):
            continue                              #nothing clean either side; leave it alone
        if np.isnan(before):
            ts[start:stop + 1] = after            #run touches the very start of the trace
        elif np.isnan(after):
            ts[start:stop + 1] = before           #run touches the very end of the trace
        else:
            #the anchors sit just outside the run, so drop the ramp's endpoints
            ts[start:stop + 1] = np.linspace(before, after, length + 2)[1:-1]
        n_fixed += 1

    return n_fixed, long_runs


def g_fit_basic_filter(ts, samples_per_s, cutoff_f=1 / 20000,
                       noise_killing_window=1000, std_thresh=6,
                       artifact_max_ms=1.0, steepness=0.95, set_obj=False, obj=None,
                       make_plots=True):
    """
    Parameters
    ----------
    ts : array_like
        Raw fluorescence timeseries.
    samples_per_ms : float
        Samples per millisecond; fs = samples_per_ms * 1000. Use 25 for a
        25 kHz acquisition.
    cutoff_f : float
        Lowpass passband edge, in Hz.
    noise_killing_window : int
        Moving-average span in SAMPLES (not seconds) -- rescale it yourself
        if you change the sample rate and want to preserve the duration.
    std_thresh : float
        Spike-rejection threshold, in standard deviations of the
        high-passed trace.
    artifact_max_ms : float
        Longest artifact, in milliseconds, that will be interpolated across. Anything longer is
        left in place and warned about, on the grounds that replacing it would be inventing
        signal rather than repairing it.
    steepness : float
        MATLAB 'Steepness' in [0.5, 1).
    set_obj, obj :
        Mirrors the MATLAB `obj.gFitLP.*` side effect.
    make_plots : bool
        Reproduce the MATLAB figures.

    Returns
    -------
    lp : ndarray      low-pass filtered baseline
    sos : ndarray     filter, as second-order sections
    dFF : ndarray     (ts - lp) / lp
    """
    ts = np.asarray(ts, dtype=float).copy()
    sps = samples_per_s

    # 1. Smooth the whole timeseries
    print(f'killing noise with {noise_killing_window} sample window, moving')
    fraw_smooth = moving_average(ts, noise_killing_window)
    print('noise killing complete')

    # 2. Subtract the smoothed curve from the raw trace
    fraw = ts - fraw_smooth

    # 2b. Interpolate across the spikes.
    #     An artifact spans however many samples it lasts -- at 25 kHz even a 0.2 ms one covers
    #     5 -- so the samples flagged for a single spike come in an unbroken run, and the run is
    #     what has to be replaced. Reading the immediately adjacent samples, as the MATLAB does,
    #     therefore reads the artifact itself for every run longer than one sample and leaves it
    #     in place; interpolate_short_artifacts brackets the whole run instead.
    #     NOTE: the MATLAB comment says "above/below" but the code tests only
    #     the positive tail. That one-sided behavior is preserved here; see
    #     the notes at the bottom if you want the symmetric version.
    #     ddof=1 matches MATLAB's nanstd (numpy defaults to ddof=0).
    thresh = std_thresh * np.nanstd(fraw, ddof=1)
    #the threshold is a statement about the raw excursions, so it can only be judged against
    #them; the step below edits ts in place, which would otherwise leave the plot showing a
    #trace from which everything above the line had already been removed
    raw_before_clip = ts.copy() if make_plots else None
    flagged = fraw > thresh
    ms_samples = max(1, int(round(sps / 1000.0)))
    max_run = max(1, int(round(artifact_max_ms * sps / 1000.0)))
    n_fixed, long_runs = interpolate_short_artifacts(ts, flagged, max_run, ms_samples)
    print(f'interpolated {n_fixed} artifact(s) out of {int(flagged.sum())} supra-threshold '
          f'samples ({std_thresh} SD), each shorter than {artifact_max_ms} ms '
          f'({max_run} samples)')
    for start, stop, length in long_runs:
        print(f'  WARNING: left a {length}-sample ({length / sps * 1000:.1f} ms) run at '
              f'{start / sps:.3f}s in place -- longer than artifact_max_ms, so interpolating '
              f'across it would invent signal. Raise artifact_max_ms to include it.')

    # 3. Zero-phase IIR low-pass, then dF/F
    sos = design_lowpass(cutoff_f, sps, steepness)
    lp = sosfiltfilt(sos, ts)
    dFF = (ts - lp) / lp

    if make_plots:
        t = np.arange(ts.size) / sps          # seconds -- uses fs, not /1000
        plt.figure()
        plt.plot(t, raw_before_clip, label='Raw Signal')
        plt.plot(t, lp, label='Low Pass')
        # the spike-rejection threshold in raw-signal units. it is applied to the high-passed
        # trace (ts - fraw_smooth), so on this axis it rides on the smoothed baseline rather
        # than sitting flat, and it is one-sided because the test itself is
        plt.plot(t, fraw_smooth + thresh, linestyle='--', linewidth=1,
                 label=f'spike threshold (+{std_thresh} SD)')
        plt.legend()
        plt.xlabel('session time (s)')

        plt.figure()
        plt.plot(t, dFF, label='dFF')
        plt.legend()
        plt.xlabel('session time (s)')

        w, h = sosfreqz(sos, worN=4096, fs=sps)
        plt.figure()
        plt.semilogx(w[1:], 20 * np.log10(np.abs(h[1:]) + 1e-30))
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (dB)')
        plt.title('Frequency response')

    if set_obj and obj is not None:
        obj.gFitLP = {'dFF': dFF, 'cutoff_f': cutoff_f, 'steepness': steepness}

    return lp, sos, dFF


# ======================================================================
# NOTES
# ======================================================================
#
# Fixes relative to the first draft of this translation:
#
# 1. Steepness formula was wrong. I used
#        fstop = fpass + (1 - s) * (nyq - fpass)
#    MATLAB actually uses
#        fstop = fpass + (0.99 - 0.98*s) * (nyq - fpass)
#    These agree at s = 0.5 but diverge as s -> 1 (at s = 0.95 the true
#    transition band is 18% wider; at s = 0.999 it is 11x wider). My version
#    also collapses to zero width at s = 1, which MATLAB never does -- it
#    floors the transition width at 1% of (nyq - fpass).
#
# 2. Passband ripple was wrong. MATLAB's lowpass ALWAYS uses 0.1 dB, not the
#    1 dB I passed to iirdesign. Stopband 60 dB was correct.
#
# 3. Transfer-function (b, a) form replaced with second-order sections and
#    sosfiltfilt. At these normalized frequencies the (b, a) form is badly
#    conditioned; MATLAB's digitalFilter object is SOS internally.
#
# 4. moving_average had the wrong endpoint rule (truncated asymmetric window
#    instead of MATLAB's symmetric shrinking window) and was O(n*span), which
#    is not runnable on 25 kHz data. Now O(n) via cumsum, correct endpoints.
#
# 5. np.nanstd needs ddof=1 to match MATLAB's nanstd.
#
# 6. Plot time axis divided by 1000 instead of fs. That is a bug in the
#    original MATLAB -- it only gives correct seconds if samples_per_ms == 1.
#    Fixed here to use fs.
#
# 7. freqz was called with fs=1000 rather than the real fs (also inherited
#    from the original MATLAB, where fvtool(f,1000,1000)/freqz(f,1000,1000)
#    passes 1000 as the sample rate).
#
#
# THE PARAMETER PROBLEM (affects the original MATLAB equally)
#
# cutoff_f = 1/20000 Hz at fs = 25000 Hz is a normalized frequency of 4e-9.
# The resulting minimum-order elliptic filter has its pole ~8e-8 from the
# unit circle, i.e. a settling time of ~1.2e7 samples ~ 8 minutes of data
# PER filtering direction. filtfilt's default edge padding is a few samples.
#
# How bad this is depends on how long the record is, so the two measurements
# below disagree and both are worth keeping.
#
# On a 60 s synthetic trace at 25 kHz: changing the value of a SINGLE sample --
# the first one -- by +20 moved the entire computed baseline from 101.3 to
# 139.9, while changing a sample in the middle or at the end changed nothing.
# The "baseline" was flat to within 0.0001 while the injected drift had
# amplitude 5. At this duration the output really is a function of ts[0]
# rather than of the data.
#
# On a 30 min synthetic trace at 25 kHz (offset 500, bleaching 2000*exp(-t/900),
# slow biology, noise) it is milder and the earlier wording overstated it: lp
# tracks the true F0 closely, r = 0.9948. What stays broken is the SCALE: the
# ratio lp/F0_true drifts from 0.860 at the start of the session to 1.335 at the
# end, and ts[0] still has leverage no single sample should have (+20 there moves
# lp.mean() by 5.3, versus 0.045 for a sample in the middle). So the baseline is
# usable in shape but wrong in magnitude, by an amount that varies across the
# session -- which propagates straight into dFF as a time-varying gain error.
#
# This is not a translation artifact. MATLAB's filtfilt uses the same kind of
# short edge padding, so the original code has the same failure mode. The
# cutoff only becomes meaningful for records many hours long -- a 5e-5 Hz
# passband edge corresponds to a ~5.5 hour period, and you cannot resolve
# that in a session shorter than several times its settling time.
#
# WHAT dFF IS AND IS NOT, FOR FIBER PHOTOMETRY
#
# dFF = (ts - lp)/lp is a dF/F in form -- dividing by a low-passed baseline is a
# standard recipe, and the division makes this a fractional change rather than
# merely a filtered signal. Two things stop it being a photometry dF/F, and the
# first is the bigger one because no amount of filter tuning reaches it:
#
# 1. The denominator includes whatever additive offset the analog channel
#    carries -- dark current, amplifier offset, ambient light, tissue
#    autofluorescence -- none of which is indicator fluorescence. F0 has to be
#    the fluorescent part alone, so the offset must be measured (e.g. LED off)
#    and subtracted before dividing. Measured on a synthetic where the true
#    answer is known, an offset of 500 against a 2000-unit fluorescence signal
#    compresses the recovered dF/F amplitude to 36% of truth, and 1000 takes it
#    to 22% -- with a PERFECT baseline. The shape survives; the units do not.
#
# 2. There is no isosbestic (405 nm) control channel here, so motion and
#    hemodynamic artifacts are indistinguishable from indicator transients.
#    The usual treatment regresses the isosbestic trace out of the signal
#    channel before computing dF/F.
#
# Consequence: treat the output as a normalized, detrended signal that is fine
# for asking WHEN activity changed, and not as a dF/F magnitude comparable
# across sessions, animals, or with published values.
#
# If your sessions are on the order of minutes to an hour, consider instead:
#   - fitting and subtracting a low-order polynomial or exponential baseline,
#   - a long moving median / percentile filter (a common dF/F baseline),
#   - decimating first (dF/F baselines rarely need 25 kHz) and then filtering,
#     which puts the cutoff at a sane normalized frequency.
#
#
# STILL PRESENT BY DESIGN (faithful to the MATLAB, flag if unintended)
#
# - Spike rejection is one-sided (only positive excursions), despite the
#   comment saying "above/below". For the symmetric version use
#       np.flatnonzero(np.abs(fraw) > thresh)
# - The threshold compares fraw against std_thresh * std(fraw) without
#   subtracting the mean of fraw. fraw is near zero-mean after the moving
#   average subtraction, so this is usually harmless.
# - Spike replacement no longer matches MATLAB, deliberately. MATLAB replaces
#   each flagged sample with the mean of the two samples touching it, which
#   only repairs an artifact one sample wide. At 25 kHz a real artifact covers
#   tens of samples, so both of those neighbors are inside the artifact and the
#   "correction" writes the artifact's own value back; being sequential and
#   in-place, the left neighbor is a value already written that way, so the run
#   propagates itself. Measured on a 1 ms, 600-unit artifact at 25 kHz: peak
#   excursion 617 before, 605 after. Runs are now bracketed as a whole and
#   interpolated between the mean of the clean millisecond either side, which
#   takes the same case to 75 (the noise floor). See
#   interpolate_short_artifacts.
# - Indices at the very first/last sample are skipped here; MATLAB would
#   error on ts(0).
# - dFF divides by lp with no guard against lp near zero.
# - The original had a dead line `ts = rawF;` overwriting the argument with
#   an undefined variable, and referenced an undeclared `setObj`. Both
#   dropped / made explicit.

