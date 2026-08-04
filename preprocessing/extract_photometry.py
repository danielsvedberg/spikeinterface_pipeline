import spikeinterface.full as si
from datetime import datetime
from pathlib import Path
from utils import file_structure as fs
from preprocessing import parse_nidq as pni
from utils import parse_input_map as pim
import pynapple as nap
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfiltfilt
#the numerically careful pieces are used from where they already live rather than copied again:
#smooth_moving is MATLAB's edge rule done in O(n), and interpolate_short_artifacts is shared with
#gfit_basicfilter. Only the SEQUENCING is restated here, which is the part that differs
from utils.boxDFF import smooth_moving
from utils.gfit_basicfilter import interpolate_short_artifacts

default_phot_channel = 0
default_lowpass_hz = 300.0


def _win_to_samples(seconds, samples_per_s, n_samples, name):
    """Convert a window given in seconds to samples, refusing the impossible ones by name."""
    win = int(round(float(seconds) * samples_per_s))
    if win < 3:
        raise ValueError(
            f"{name}={seconds}s is {win} samples at {samples_per_s} Hz, too short to smooth "
            f"over. Note this argument is in seconds, not samples.")
    if win > n_samples:
        raise ValueError(
            f"{name}={seconds}s is {win} samples at {samples_per_s} Hz, longer than the "
            f"{n_samples}-sample ({n_samples / samples_per_s:.1f}s) trace. Note this argument "
            f"is in seconds, not samples.")
    return win


def remove_artifacts(ts, samples_per_s, smoothing_win_s=10.0, std_level=15,
                     artifact_max_ms=1.0, verbose=True):
    """Flag supra-threshold excursions and interpolate across them. Steps 1-2b of boxDFF.

    Split out of box_dff so it can run BEFORE the low-pass filter. Order matters: an artifact
    only a few samples wide is sharp and mostly high-frequency, so low-passing first both
    attenuates it and smears it across the filter's impulse response -- a 5-sample spike comes
    out ~46 samples wide at 300 Hz and 25 kHz. It then reads as one long run rather than a spike,
    and gets skipped for exceeding artifact_max_ms. Detecting on the raw trace avoids all of
    that, and removing the spike before filtering keeps the filter from ringing on it.

    Returns a corrected copy; the input is not modified.
    """
    ts = np.asarray(ts, dtype=float).ravel()
    if samples_per_s is None or samples_per_s <= 0:
        raise ValueError(f"samples_per_s must be a positive sampling rate, got {samples_per_s!r}")
    smoothing_win = _win_to_samples(smoothing_win_s, samples_per_s, ts.size, 'smoothing_win_s')

    # 1./2. subtract a smoothed copy, leaving the high-passed trace the threshold is set on
    if verbose:
        print(f"killing noise with {smoothing_win_s}s ({smoothing_win} sample) window, moving")
    fraw = ts - smooth_moving(ts, smoothing_win)
    if verbose:
        print("noise killing complete")

    # 2b. one-sided, as in the MATLAB, even though its comment says "above/below".
    #     MATLAB's `std` normalises by N-1 -> ddof=1
    flagged = fraw > std_level * np.std(fraw, ddof=1)

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
    return frs


def box_dff(ts, box_win_s, samples_per_s, verbose=True):
    """Divide out a box-window baseline: (F - Fsmooth) / Fsmooth. Steps 3-4 of boxDFF.

    This half assumes artifacts are already dealt with -- run remove_artifacts first, and the
    low-pass in between if you want one. Unlike utils.boxDFF.box_dff, which does the whole
    sequence in one call, this does no artifact handling of its own.
    """
    ts = np.asarray(ts, dtype=float).ravel()
    if samples_per_s is None or samples_per_s <= 0:
        raise ValueError(f"samples_per_s must be a positive sampling rate, got {samples_per_s!r}")
    box_win = _win_to_samples(box_win_s, samples_per_s, ts.size, 'box_win_s')

    if verbose:
        print(f"Fitting boxDFF with {box_win_s}s ({box_win} sample) window, moving "
              f"{datetime.now().strftime('%I:%M %p')}")
    fsmooth = smooth_moving(ts, box_win)
    if verbose:
        print("gfitting complete")
    return (ts - fsmooth) / fsmooth


def lowpass_filter(sig, sf, cutoff_hz=default_lowpass_hz, order=4):
    """Zero-phase Butterworth low-pass, for taking the wideband noise off a photometry trace.

    The nidq samples far faster than the indicator can possibly respond, so everything above a
    few hundred Hz is shot and electrical noise. Removing it before the dF/F step means the
    artifact threshold is set by the signal's own variation rather than by that noise.

    Applied forwards and backwards (sosfiltfilt), so there is no phase shift and a transient
    stays where it happened -- which matters here because the trace gets aligned to imec time.
    That doubles the effective order, so `order` is the one-way order. Butterworth is chosen for
    its flat passband: an elliptic or Chebyshev design would ripple across exactly the band the
    fluorescence lives in.

    Parameters
    ----------
    sig : array_like
        The trace to filter.
    sf : float
        Sampling rate, in Hz.
    cutoff_hz : float
        Passband edge, in Hz. -3 dB is at this frequency.
    order : int
        One-way filter order; the zero-phase pass makes the effective rolloff twice this.
    """
    sig = np.asarray(sig, dtype=float)
    nyq = sf / 2.0
    if not 0 < cutoff_hz < nyq:
        raise ValueError(
            f"cutoff_hz must be between 0 and the Nyquist frequency ({nyq} Hz at {sf} Hz "
            f"sampling), got {cutoff_hz}")
    #sosfiltfilt pads by 3*(2*order+1) samples at each end and cannot run on anything shorter
    min_len = 3 * (2 * order + 1)
    if sig.size <= min_len:
        raise ValueError(f"trace is {sig.size} samples, too short to filter at order {order} "
                         f"(needs more than {min_len})")
    sos = butter(order, cutoff_hz / nyq, btype='low', output='sos')
    return sosfiltfilt(sos, sig)

def step_profile(trace, window):
    """How much the trace steps at each point: the mean of the next `window` samples minus the
    mean of the previous `window`.

    This is a matched filter for a step edge, so it answers both of the questions the power
    switch poses -- how big is the biggest step, and where is it -- while averaging away sample
    noise. A slow drift barely registers, because it moves the two windows together.
    """
    trace = np.asarray(trace, dtype=float)
    half = max(1, int(window) // 2)
    profile = np.zeros(len(trace))
    if len(trace) > 2 * half:
        means = uniform_filter1d(trace, size=2 * half, mode='nearest')
        profile[half:len(trace) - half] = means[2 * half:] - means[:len(trace) - 2 * half]
    return profile


def _plateau_edge(coarse, edge_idx, window, backward, tol):
    """Walk from the steepest point of a transition out to where the trace is still on the
    powered plateau.

    step_profile peaks at the CENTRE of a transition, so the steepest point is halfway down a
    comedown, not at its start -- cutting there leaves half the comedown in the signal, and the
    slower the supply falls the more is left. This finds the last (or first) sample still up on
    the powered level, which is where the cut belongs.

    Levels are read locally, from a window either side of the edge, so a bleached session is
    judged against its own end rather than against the bright start of the recording.
    """
    lo = max(0, edge_idx - 2 * window)
    hi = min(len(coarse), edge_idx + 2 * window)
    segment = coarse[lo:hi]
    if segment.size == 0:
        return edge_idx
    powered = np.percentile(segment, 95)
    null = np.percentile(segment, 5)
    threshold = null + (powered - null) * (1 - tol)

    if backward:
        on_plateau = np.flatnonzero(coarse[lo:edge_idx + 1] >= threshold)
        return lo + int(on_plateau[-1]) if on_plateau.size else edge_idx
    on_plateau = np.flatnonzero(coarse[edge_idx:hi] >= threshold)
    return edge_idx + int(on_plateau[0]) if on_plateau.size else edge_idx


def find_powered_span(phot_sig, sf, detect_rate=1000.0, step_window_s=10.0, min_step_frac=0.3,
                      plateau_tol=0.02, margin_s=0.0):
    """Locate the stretch of phot_sig recorded while the excitation power source was on.

    The power supply is switched on at the start of the experiment and off at the end, so the
    trace sits at a near-zero baseline either side of the session and steps up to a much larger
    value in between. Those null ends carry no photometry and would ruin the dF/F fit, whose
    lowpass baseline gets divided into the signal and is ~0 out there, so they are sliced off.

    What marks the switch out is that it is a *step*, arriving far faster than bleaching or
    biology ever moves. Finding it takes two stages, because locating a transition and placing
    a cut at its edge are different questions:

      1. the step profile's largest rise and largest fall say WHERE each switch is, and an edge
         only counts if it covers min_step_frac of the trace's full range;
      2. from each of those points the cut is then walked out to the edge of the powered
         plateau. The profile peaks at the CENTRE of a transition, so stopping at stage 1 leaves
         half of a comedown in the signal -- and all of it once the comedown is slower than
         step_window_s.

    Locating the edges by step rather than by level means a null end lasting a fraction of a
    second is found as reliably as one lasting minutes: a level threshold has to split a
    histogram in which such an end is a rounding error, and lands inside the powered signal
    instead.

    Either edge may be missing, and is then simply not trimmed: a recording already under way
    when the supply came on has no rise, one that ended first has no fall, and one made entirely
    with the power on has neither and is kept whole.

    Detection runs on a copy decimated to detect_rate, which resolves a step lasting many
    milliseconds just as well as the full-rate trace at a fraction of the memory. The returned
    span indexes the original.

    Parameters
    ----------
    phot_sig : array_like
        The raw photometry trace, full sampling rate.
    sf : float
        Sampling rate of phot_sig, in Hz.
    detect_rate : float
        Rate, in Hz, that the trace is decimated to for detection.
    step_window_s : float
        Width, in seconds, of the window the switch is expected to complete its jump within. It
        has to be at least as long as the supply takes to fall, or the step gets averaged away
        and the edge is missed entirely. It cannot simply be made large, though: on simulated
        traces the biggest step bleaching alone can raise grows with it (0.08 of range at 1 s,
        0.19 at 3 s, 0.30 at 5 s), and once that reaches min_step_frac the trace gets trimmed on
        nothing. The default of 3 catches comedowns up to ~4 s while spurious steps stay near
        0.19 -- comfortably under the 0.3 threshold.
    plateau_tol : float
        How far below the local powered level, as a fraction of the powered-to-null difference,
        the trace may sit and still count as "on the plateau" when the cut is walked out to the
        edge of a transition. Smaller cuts closer to the very start of a comedown, at the risk
        of catching ordinary noise on the plateau.
    min_step_frac : float
        The fraction of the trace's full range an edge has to cover to count as a power switch.
        On simulated traces a real switch scores 0.5-0.96 while the largest step bleaching and
        biology can raise on their own is under 0.1, so the default sits in that gap with room
        to spare. Lower it if a switch-off is being missed on a heavily bleached session, where
        the powered level has sunk towards the null one by the end of the recording.
    margin_s : float
        Seconds to step further inside each detected edge, to clear any settling transient at
        switch-on/off. Off by default, so the span ends exactly at the edges.

    Returns
    -------
    slice
        The powered stretch, indexing phot_sig. Apply it to the sample timestamps as well as to
        the signal, so that the two stay the same length.
    """
    phot_sig = np.asarray(phot_sig)
    n = len(phot_sig)

    step = max(1, int(round(sf / detect_rate)))
    coarse = phot_sig[::step].astype(float)
    jump_window = max(2, int(round(step_window_s * detect_rate)))
    profile = step_profile(coarse, jump_window)

    #an edge has to be a big fraction of the whole trace's range to be the power supply rather
    #than a feature of the signal, which makes the test independent of the recording's units
    full_range = np.percentile(coarse, 99) - np.percentile(coarse, 1)
    min_jump = min_step_frac * full_range

    on_idx = int(np.argmax(profile))
    off_idx = int(np.argmin(profile))
    rise = profile[on_idx] if full_range > 0 and profile[on_idx] >= min_jump else None
    fall = -profile[off_idx] if full_range > 0 and -profile[off_idx] >= min_jump else None

    #the supply must come on before it goes off; anything else is not the pattern being looked for
    if rise is not None and fall is not None and on_idx >= off_idx:
        rise = fall = None

    if rise is None and fall is None:
        biggest = np.max(np.abs(profile)) / full_range if full_range > 0 else 0.0
        print(f"photometry: no power switch in the trace (biggest step is {biggest:.0%} of its "
              f"range, needs {min_step_frac:.0%}), keeping all of it")
        return slice(0, n)

    #stage 2: the profile peaked at the middle of each transition, so walk out to the plateau
    #edge. without this the cut sits halfway down the comedown and the rest of it stays in
    half_window = max(1, jump_window // 2)
    if rise is not None:
        start = _plateau_edge(coarse, on_idx, half_window, backward=False, tol=plateau_tol) * step
    else:
        start = 0
    if fall is not None:
        stop = min(n, _plateau_edge(coarse, off_idx, half_window, backward=True,
                                    tol=plateau_tol) * step)
    else:
        stop = n

    margin = int(round(margin_s * sf))
    start = min(start + margin, n)
    stop = max(stop - margin, start)

    on_note = f"switch-on {rise / full_range:.0%} of range" if rise is not None else \
        "already powered at the first sample"
    off_note = f"switch-off {fall / full_range:.0%}" if fall is not None else \
        "still powered at the last sample"
    print(f"photometry: powered from {start / sf:.1f}s to {stop / sf:.1f}s ({on_note}, {off_note}); "
          f"trimming {start / sf:.1f}s off the start and {(n - stop) / sf:.1f}s off the end")
    return slice(start, stop)


def align_to_imec_time(phot_sig, paths):
    # takes in a fs.SessionPaths object
    spikeglx_folder = paths.spikeglx_folder()
    dinmap, _, _ = pim.get_channel_maps(paths.recording_folder)
    digital_events = si.read_spikeglx_event(spikeglx_folder)
    imec_rec = si.read_spikeglx(spikeglx_folder, channel_ids=digital_events.channel_ids)
    pulse_din = pni.get_timing_pulse_din(dinmap)
    aligner = pni.get_nidq_aligner(imec_rec, digital_events, pulse_din)
    phot_sig_aligned = aligner.align(phot_sig)
    return phot_sig_aligned

def get_phot_ch(paths):
    _, ainmap, _ = pim.get_channel_maps(paths.recording_folder)
    if ainmap is None:
        print("photometry channel not in input map .txt file, using default channel")
        return default_phot_channel
    else:
        return ainmap['photometry']


def get_photometry(base_folder, phot_channel=None, align_time=False,
                   lowpass_hz=default_lowpass_hz):
    paths = fs.SessionPaths(base_folder, create=True)
    analysis_folder = paths.analysis_folder
    spikeglx_folder = paths.spikeglx_folder()

    nidq_rec = si.read_spikeglx(spikeglx_folder, stream_id='nidq')
    nidq_t = nidq_rec.get_times()
    sf = nidq_rec.get_sampling_frequency()

    if phot_channel is None:
        phot_channel = get_phot_ch(paths)

    chids = nidq_rec.channel_ids #get all channel ids--channels are indexed using alphanumeric string identifier
    achids = [chid for chid in chids if 'D' not in str(chid)] #get analog channel ids (ones without 'D' in them)
    phot_chid = achids[phot_channel]

    phot_sig = nidq_rec.get_traces(channel_ids=[phot_chid])
    phot_sig = phot_sig.squeeze()

    #the power source is switched on at the start of the experiment and off at the end; drop the
    #null stretches outside that window. the same span slices the timestamps, so the signal and
    #its times stay the same length for the nap.Tsd below
    powered = find_powered_span(phot_sig, sf)
    phot_sig = phot_sig[powered]
    nidq_t = nidq_t[powered]

    #artifact removal -> low-pass -> baseline division, in that order. artifacts are cut out
    #first because they are only a few samples wide and the filter would smear them into
    #something too long to repair; the low-pass then strips the wideband noise the indicator
    #cannot have produced. phot_sig itself is left alone, so the Tsd returned below is still the
    #measured trace
    phot_clean = remove_artifacts(phot_sig, sf, std_level=6)
    phot_lp = lowpass_filter(phot_clean, sf, cutoff_hz=lowpass_hz)
    dFF = box_dff(phot_lp, box_win_s=200, samples_per_s=sf)

    if align_time:
        nidq_t = align_to_imec_time(nidq_t, paths)

    dFF_tsd = nap.Tsd(t=nidq_t, d=dFF)

    #save dFF_tsd to analysis folder / photometry
    # if the analysis_folder\photometry doesn't exist, make it
    phot_folder = analysis_folder / "photometry"
    phot_folder.mkdir(parents=True, exist_ok=True)
    dFF_tsd.save(phot_folder / "photometry_dFF.npz")

    return phot_sig

def test():
    phot_channel=None
    align_time=False
    base_folder = Path(r"C:\Users\assad\Desktop\experiment_data\recording_files\DS48\DS48_20260729")
    get_photometry(base_folder, phot_channel=phot_channel)