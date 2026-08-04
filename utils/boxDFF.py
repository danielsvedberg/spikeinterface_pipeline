from datetime import datetime
import numpy as np

#the artifact repair is shared with gfit_basicfilter rather than copied: both translations hit
#the same MATLAB single-sample assumption, and one definition cannot drift out of step with the
#other
from utils.gfit_basicfilter import interpolate_short_artifacts


def smooth_moving(y, span):
    """Equivalent of MATLAB's ``smooth(y, span, 'moving')``.

    Centered moving average. Matches MATLAB's two conventions:
      * an even `span` is reduced to `span - 1`
      * near the edges the window shrinks symmetrically, so
        out[k] = mean(y[0 : 2k+1]) and out[-1-k] = mean(y[-(2k+1) :])
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    span = min(int(span), n)
    if span % 2 == 0:                      # MATLAB forces an odd span
        span -= 1
    if span < 3:
        return y.astype(float, copy=True)

    h = (span - 1) // 2

    # Mean-center before the cumsum: differencing two large partial sums to
    # recover a short-window mean is where float64 cancellation would bite on a
    # multi-million-sample trace.
    mu = y.mean()
    cs = np.concatenate(([0.0], np.cumsum(y - mu)))

    out = np.empty(n, dtype=float)
    out[h:n - h] = (cs[span:] - cs[:n - span + 1]) / span       # full windows
    k = np.arange(h)
    out[k] = (cs[2 * k + 1] - cs[0]) / (2 * k + 1)              # leading edge
    out[n - 1 - k] = (cs[n] - cs[n - 2 * k - 1]) / (2 * k + 1)  # trailing edge
    return out + mu


def box_dff(ts, box_win_s, samples_per_s, smoothing_win_s=10, std_level=15,
            artifact_max_ms=1.0, verbose=True):
    """Box-window dF/F.

    Every window is given as a duration, so a call means the same thing whatever the rig sampled
    at. The MATLAB took both windows as sample counts, which silently meant 10 s on a 100 Hz rig
    and 40 ms on a 25 kHz one.

    Parameters
    ----------
    ts : array_like
        Raw fluorescence timeseries.
    box_win_s : float
        Width of the box (baseline) window, in SECONDS.
    samples_per_s : float
        Sampling rate of ts, in Hz. Required, since it is what converts every window into samples.
    smoothing_win_s : float
        Span of the pre-smoothing moving average, in SECONDS. Subtracting this smoothed copy is
        what high-passes the trace for artifact detection, so it wants to be long enough that an
        artifact does not drag it along, and short enough that real slow signal is removed rather
        than left to inflate the threshold. The default of 0.04 is the MATLAB's 1000 samples at
        25 kHz.
    std_level : float
        Artifact threshold, in standard deviations of the high-passed trace.
    artifact_max_ms : float
        Longest artifact, in milliseconds, that step 2b will interpolate across. An artifact
        lasts however long it lasts -- at 25 kHz even a 0.2 ms one covers 5 samples -- so the
        flagged samples come in unbroken runs and the run is what has to be replaced. Runs longer
        than this are reported rather than repaired.
    verbose : bool
        Print progress.
    """
    ts = np.array(ts, dtype=float).ravel()   # copy: never mutate the caller's array
    if samples_per_s is None or samples_per_s <= 0:
        raise ValueError(f"samples_per_s must be a positive sampling rate, got {samples_per_s!r}")

    def _to_samples(seconds, name):
        #both windows are durations now, so a value that used to be a sample count turns into an
        #impossible one here; say so rather than letting it silently clamp
        n_samples = int(round(float(seconds) * samples_per_s))
        if n_samples < 3:
            raise ValueError(
                f"{name}={seconds}s is {n_samples} samples at {samples_per_s} Hz, too short to "
                f"smooth over. Note this argument is in seconds, not samples.")
        if n_samples > ts.size:
            raise ValueError(
                f"{name}={seconds}s is {n_samples} samples at {samples_per_s} Hz, longer than "
                f"the {ts.size}-sample ({ts.size / samples_per_s:.1f}s) trace. Note this "
                f"argument is in seconds, not samples.")
        return n_samples

    box_win = _to_samples(box_win_s, 'box_win_s')
    smoothing_win = _to_samples(smoothing_win_s, 'smoothing_win_s')

    # 1. Smooth the whole day timeseries
    if verbose:
        print(f"killing noise with {smoothing_win_s}s ({smoothing_win} sample) window, moving")
    fraw = smooth_moving(ts, smoothing_win)
    if verbose:
        print("noise killing complete")

    # 2. Subtract the smoothed curve from the raw trace
    fraw = ts - fraw

    # 2b. Eliminate all noise > std_level SD above the whole trace (Frs).
    #     Faithful to the original: the test is one-sided (positive excursions
    #     only), even though the MATLAB comment says "above/below".
    #     MATLAB's `std` normalises by N-1 -> ddof=1.
    flagged = fraw > std_level * np.std(fraw, ddof=1)

    # The MATLAB replaces each flagged sample with the mean of the two samples touching it:
    # `Frs(ig) = mean([Frs(ig-1), Frs(ig+1)], 2)`, one vectorised assignment against the ORIGINAL
    # neighbours. That only repairs an artifact one sample wide. For a run of two or more, both
    # of those neighbours are themselves inside the artifact, so the "correction" writes the
    # artifact's own level straight back. Runs are bracketed as a whole instead, and filled by
    # interpolating between the clean millisecond either side.
    frs = ts.copy()
    anchor_window = max(1, int(round(samples_per_s / 1000.0)))
    max_run = max(1, int(round(artifact_max_ms * samples_per_s / 1000.0)))

    n_fixed, long_runs = interpolate_short_artifacts(frs, flagged, max_run, anchor_window)
    if verbose:
        print(f"interpolated {n_fixed} artifact(s) out of {int(flagged.sum())} supra-threshold "
              f"samples ({std_level} SD)")
    for start, stop, length in long_runs:
        print(f"  WARNING: left a {length}-sample ({length / samples_per_s * 1000:.1f} ms) run at "
              f"{start / samples_per_s:.3f}s in place -- longer than artifact_max_ms, so "
              f"interpolating across it would invent signal. Raise artifact_max_ms to include it.")
    # A run reaching the very first or last sample is filled from whichever side has clean data;
    # MATLAB errors out on that case.

    # 3. Repeat step 1 without the noise points (Fsmooth)
    if verbose:
        print(f"Fitting boxDFF with {box_win_s}s ({box_win} sample) window, moving "
              f"{datetime.now().strftime('%I:%M %p')}")
    fsmooth = smooth_moving(frs, box_win)
    if verbose:
        print("gfitting complete")

    # 4. [Frs(point) - Fsmooth(point)] / Fsmooth(point)
    return (frs - fsmooth) / fsmooth