"""Licks from the behaviour computer's event table, for sessions whose NIDQ lick line was not recorded.

The Arduino running the task logs every trial event, licks included, to <session>_eventTable.csv
(columns time,label; time in ms on the Arduino's own clock). The NIDQ normally records the licks
too, over a BNC copy of the lick line, but when that copy fails the NIDQ still has every other
task event -- cue, reward and outcome tones, solenoid. Those events are in both records, so they
fix the Arduino-ms -> binary_signals-time mapping, and the logged licks can be placed with it.

The mapping is a straight line: the two clocks are independent crystals, so they differ by an
offset (the Arduino is started at some arbitrary time relative to the NIDQ) and a constant rate
(~13 ppm on the DS48 rig, i.e. ~35 ms over a session -- far too much to ignore). The fit uses only
the anchors in ANCHORS marked fit=True; the others are checked against it, so a table that belongs
to a different session, or a mis-paired fit, fails loudly instead of yielding plausible-looking
licks.

No lick latency correction is applied. The table stamps each event to the Arduino ms tick, so a
mapped lick is within ~1 ms of the real lick edge, which is below everything licks are used for
(PSTH bins, the 0.3 s bout threshold) except one: parse_trials takes the last lick at or before
each outcome tone as the lick that triggered it, and the triggering lick precedes its tone by only
0.2 ms (reward) to 1.3 ms (error) on a healthy NIDQ, and the solenoid it opens by ~0.1 ms. A
global shift would push many trigger licks past their outcomes, so instead each trigger lick is
checked individually against its own NIDQ outcome events (solenoid and tone) and, should the 1 ms
quantization land it at or after the earliest of them, pulled to just before it (see
TRIGGER_OUTCOMES). Mapped licks otherwise carry the table's ~1 ms quantization.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

LICK_LABEL = 'LICK'


@dataclass(frozen=True)
class Anchor:
    """A task event logged both in the event table and in binary_signals.

    latency_ms is the expected range of (binary_signals time - mapped table time), i.e. how long the
    hardware edge the NIDQ sees trails the Arduino's log entry. It is a sanity window for the
    median, measured on DS48_20260923/24: the cue tone fires on the ms tick (~0 ms), the solenoid
    0.2-0.4 ms and the error tone ~1.6 ms after the log entry, and the timeout tone ~3.8 ms.
    """
    label: str
    event: str
    fit: bool = False
    latency_ms: tuple = (-1.0, 1.0)


ANCHORS = (
    #the cue tone: every trial has exactly one, and it fires on the Arduino ms tick, so it alone
    #gives a sub-0.1 ms fit. the others are 1 ms-quantised, and would only add noise to it
    Anchor('CUE_ON', 'start_tone_on', fit=True, latency_ms=(-0.5, 0.5)),
    Anchor('REWARD', 'solenoid_on', latency_ms=(-0.5, 1.5)),
    Anchor('REWARD', 'reward_tone_on', latency_ms=(-0.5, 1.5)),
    Anchor('ABORT', 'error_tone_on', latency_ms=(0.0, 4.0)),
    Anchor('WINDOW_CLOSED', 'timeout_tone_on', latency_ms=(2.0, 6.0)),
)

#outcome rows whose triggering lick downstream code looks for, and the binary_signals events that
#lick physically precedes (it opens the solenoid and triggers the tone). the trigger lick is the
#last LICK row at most TRIGGER_WINDOW_MS before the outcome row (it shares the outcome's ms, or is
#1 ms earlier when the Arduino logged across a tick)
TRIGGER_OUTCOMES = {'REWARD': ('solenoid_on', 'reward_tone_on'), 'ABORT': ('error_tone_on',)}
TRIGGER_WINDOW_MS = 2
#a trigger lick that maps to at/after one of its outcome events is placed this far before it:
#about the lick -> solenoid lead on a healthy NIDQ (0.08-0.12 ms, DS48_20260922)
TRIGGER_LEAD_S = 1e-4

#fit acceptance. the table is 1 ms resolution and the cue fit leaves residuals under 0.1 ms, so a
#2 ms residual means a mis-paired or broken fit, not jitter; 90% of cues allows for a few trials
#cut off at either end of the recording; and any real pair of crystal clocks agrees to well under
#1000 ppm, so a slope beyond that means the pairing is wrong
MIN_MATCH_FRACTION = 0.9
MIN_MATCHES = 10
MAX_RESIDUAL_MS = 2.0
MAX_DRIFT_PPM = 1000.0
#coarse pairing tolerance before the line is known, and the tight one after
COARSE_TOL_S = 0.5
FINE_TOL_S = 0.005
#check anchors are matched within this of the fitted line (their latency is a few ms at most)
CHECK_TOL_S = 0.010


def read_event_table(csv_file):
    """The event table as a DataFrame with float time_s (Arduino seconds) and label columns."""
    table = pd.read_csv(csv_file)
    missing = {'time', 'label'} - set(table.columns)
    if missing:
        raise ValueError(f"{csv_file} is not an event table: missing column(s) {sorted(missing)}")
    table = table.assign(time_s=table['time'].astype(float) / 1000.0,
                         label=table['label'].astype(str).str.strip())
    return table


def label_times(table, label):
    """Distinct times (s) of one label. Some rows are logged twice on the same ms (WINDOW_CLOSED)."""
    return np.unique(table.loc[table['label'] == label, 'time_s'].to_numpy())


def match(x, y, tol):
    """One-to-one nearest match of sorted x into sorted y within tol; returns index arrays (ix, iy).

    An element of y claimed by more than one x is dropped from every pair, rather than given to
    whichever came first, so an ambiguous pairing never enters a fit.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) == 0 or len(y) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    j = np.searchsorted(y, x)
    lo, hi = np.clip(j - 1, 0, len(y) - 1), np.clip(j, 0, len(y) - 1)
    jj = np.where(np.abs(y[lo] - x) <= np.abs(y[hi] - x), lo, hi)
    ok = np.abs(y[jj] - x) < tol
    ii, jj = np.flatnonzero(ok), jj[ok]
    claimed, counts = np.unique(jj, return_counts=True)
    unique = ~np.isin(jj, claimed[counts > 1])
    return ii[unique], jj[unique]


def coarse_offset(x, y, res=0.05):
    """Offset d with x + d ~ y, from the mode of all pairwise differences.

    This makes no assumption about which events pair up -- the first logged cue need not be the
    first recorded one -- since the true offset is the one difference every pair shares.
    """
    d = (y[None, :] - x[:, None]).ravel()
    counts, edges = np.histogram(d, np.arange(d.min(), d.max() + 2 * res, res))
    return edges[np.argmax(counts)] + res / 2


def fit_clock(x, y, n_iter=3, mad_k=5.0):
    """Fit y = slope * x + intercept from two unpaired event-time sets; returns (p, ix, iy).

    Coarse offset -> nearest pairing within COARSE_TOL_S -> linear fit, dropping pairs more than
    mad_k MADs off the line, then re-pairing everything within FINE_TOL_S of it, n_iter times.
    """
    x, y = np.sort(np.asarray(x, dtype=float)), np.sort(np.asarray(y, dtype=float))
    if len(x) < MIN_MATCHES or len(y) < MIN_MATCHES:
        raise ValueError(f"Too few anchor events to fit the clock ({len(x)} logged, {len(y)} "
                         f"recorded; need {MIN_MATCHES}).")
    ix, iy = match(x + coarse_offset(x, y), y, COARSE_TOL_S)
    if len(ix) < 2:
        raise ValueError("No consistent offset between the event table and the recording.")
    p = np.array([1.0, np.median(y[iy] - x[ix])])
    for _ in range(n_iter):
        r = y[iy] - np.polyval(p, x[ix])
        dev = np.abs(r - np.median(r))
        #floor of 1 us: the cue residuals can be so tight the MAD would reject good pairs
        keep = dev <= max(mad_k * 1.4826 * np.median(dev), 1e-6)
        p = np.polyfit(x[ix[keep]], y[iy[keep]], 1)
        ix, iy = match(np.polyval(p, x), y, FINE_TOL_S)
        if len(ix) < 2:
            raise ValueError("Clock fit collapsed: no anchors within "
                             f"{FINE_TOL_S * 1e3:.0f} ms of the fitted line.")
    return np.polyfit(x[ix], y[iy], 1), ix, iy


def _latency_stats(latency_ms):
    if len(latency_ms) == 0:
        return dict(median_ms=None, mad_ms=None, min_ms=None, max_ms=None)
    med = float(np.median(latency_ms))
    return dict(median_ms=med, mad_ms=float(np.median(np.abs(latency_ms - med))),
                min_ms=float(latency_ms.min()), max_ms=float(latency_ms.max()))


def _check_anchors(table, events, p, anchors):
    """Match every anchor under the fitted map; returns per-anchor stats and failure messages."""
    stats, failures = [], []
    for a in anchors:
        x, y = label_times(table, a.label), np.sort(np.asarray(events.get(a.event, []), float))
        tol = FINE_TOL_S if a.fit else CHECK_TOL_S
        ix, iy = match(np.polyval(p, x), y, tol)
        latency = (y[iy] - np.polyval(p, x[ix])) * 1e3
        expected = min(len(x), len(y))
        s = dict(label=a.label, event=a.event, fit=a.fit, n_table=int(len(x)), n_recorded=int(len(y)),
                 n_matched=int(len(ix)), expected_latency_ms=list(a.latency_ms), **_latency_stats(latency),
                 _t=np.polyval(p, x[ix]), _latency=latency)
        stats.append(s)
        if expected == 0:
            continue
        name = f"{a.label} -> {a.event}"
        if len(ix) < MIN_MATCH_FRACTION * expected:
            failures.append(f"{name}: only {len(ix)} of {expected} matched")
        elif a.fit and np.abs(latency).max() > MAX_RESIDUAL_MS:
            failures.append(f"{name}: max residual {np.abs(latency).max():.2f} ms")
        elif not a.latency_ms[0] <= s['median_ms'] <= a.latency_ms[1]:
            failures.append(f"{name}: median latency {s['median_ms']:.2f} ms outside the expected "
                            f"{a.latency_ms[0]}..{a.latency_ms[1]} ms")
    return stats, failures


def _guard_trigger_licks(table, lick_x, licks, events, p):
    """Keep each outcome's triggering lick before its own recorded outcome events (solenoid,
    tone; see module docstring).

    Returns the adjusted licks and how many were moved. Only the table's 1 ms quantization moves
    any: on DS48_20260923 a few reward-trigger licks map tens of us past their solenoid.
    """
    licks = licks.copy()
    moved = set()
    for label, outcome_events in TRIGGER_OUTCOMES.items():
        outcome_x = label_times(table, label)
        for event in outcome_events:
            times = np.sort(np.asarray(events.get(event, []), float))
            ix, iy = match(np.polyval(p, outcome_x), times, CHECK_TOL_S)
            for x_out, t_event in zip(outcome_x[ix], times[iy]):
                k = np.searchsorted(lick_x, x_out + 1e-9) - 1  #last lick logged at or before the outcome
                if k < 0 or x_out - lick_x[k] > TRIGGER_WINDOW_MS / 1e3:
                    continue
                if licks[k] >= t_event:
                    licks[k] = t_event - TRIGGER_LEAD_S
                    moved.add(k)
    return np.sort(licks), len(moved)


def event_table_licks(table, events, time_support=None, anchors=ANCHORS):
    """Lick onsets (binary_signals time) from an event table, aligned on the recorded task events.

    table: DataFrame from read_event_table, or a path to the csv.
    events: {binary_signals event name: times} for the anchors' events, from the same parse.
    time_support: optional pynapple IntervalSet; licks outside it are dropped (and counted).

    Returns (lick_times, diagnostics). Raises ValueError if the table does not align.
    """
    source = None
    if not isinstance(table, pd.DataFrame):
        source = str(table)
        table = read_event_table(table)
    fit_anchors = [a for a in anchors if a.fit]
    x_fit = np.concatenate([label_times(table, a.label) for a in fit_anchors])
    y_fit = np.concatenate([np.asarray(events.get(a.event, []), float) for a in fit_anchors])
    p, ix, iy = fit_clock(x_fit, y_fit)
    drift_ppm = (p[0] - 1) * 1e6

    stats, failures = _check_anchors(table, events, p, anchors)
    if abs(drift_ppm) > MAX_DRIFT_PPM:
        failures.insert(0, f"clock drift {drift_ppm:.0f} ppm is implausible")
    if failures:
        raise ValueError("Event table does not align with the recorded task events "
                         f"(slope {p[0]:.9f}, intercept {p[1]:.4f} s):\n  " + "\n  ".join(failures))

    lick_x = label_times(table, LICK_LABEL)
    licks, n_moved = _guard_trigger_licks(table, lick_x, np.polyval(p, lick_x), events, p)
    n_mapped = len(licks)
    if time_support is not None:
        inside = np.zeros(len(licks), bool)
        for start, end in zip(np.asarray(time_support.start), np.asarray(time_support.end)):
            inside |= (licks >= start) & (licks <= end)
        licks = licks[inside]

    t_table = np.polyval(p, table['time_s'].to_numpy())
    diagnostics = dict(
        lick_source='event_table', event_table=source,
        slope=float(p[0]), intercept_s=float(p[1]), drift_ppm=float(drift_ppm),
        mapping='t_binary_signals = slope * (event_table_ms / 1000) + intercept_s',
        table_window_s=[float(t_table.min()), float(t_table.max())],
        n_licks_table=int(n_mapped), n_licks=int(len(licks)),
        n_licks_outside_support=int(n_mapped - len(licks)),
        n_trigger_licks_moved=int(n_moved),
        anchors=stats,
    )
    return licks, diagnostics


def summarize(diagnostics):
    """A few printable lines on where the licks came from and how well the table aligned."""
    d = diagnostics
    lines = [f"Licks from event table {d['event_table']}",
             f"  clock map: slope {d['slope']:.10f} ({d['drift_ppm']:+.2f} ppm), intercept "
             f"{d['intercept_s']:.6f} s; table covers {d['table_window_s'][0]:.2f}-"
             f"{d['table_window_s'][1]:.2f} s (licks outside it are unknown, not absent)",
             f"  {d['n_licks']} licks placed ({d['n_licks_outside_support']} outside the recording, "
             f"{d['n_trigger_licks_moved']} trigger licks moved before their outcome)"]
    for s in d['anchors']:
        if s['n_matched'] == 0:
            lines.append(f"  {s['label']:>13} -> {s['event']:<16} {s['n_table']} logged, "
                         f"{s['n_recorded']} recorded, none matched")
            continue
        lines.append(f"  {s['label']:>13} -> {s['event']:<16} {s['n_matched']}/{s['n_table']} logged, "
                     f"{s['n_recorded']} recorded; latency {s['median_ms']:+.3f} ms "
                     f"(MAD {s['mad_ms']:.3f}, range {s['min_ms']:+.2f}..{s['max_ms']:+.2f})"
                     + (' [fit]' if s['fit'] else ''))
    return "\n".join(lines)


def save_diagnostics(diagnostics, folder, stem='lick_event_table_alignment'):
    """Write the alignment diagnostics as json, plus a plot of anchor latency against time.

    The json is the record that this session's licks came from the event table, not the NIDQ.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    clean = {k: v for k, v in diagnostics.items() if k != 'anchors'}
    clean['anchors'] = [{k: v for k, v in s.items() if not k.startswith('_')}
                        for s in diagnostics['anchors']]
    json_file = folder / f"{stem}.json"
    json_file.write_text(json.dumps(clean, indent=2))

    import matplotlib.pyplot as plt
    shown = [s for s in diagnostics['anchors'] if s['n_matched'] > 0]
    fig, axes = plt.subplots(len(shown), 1, figsize=(9, 1.8 * len(shown) + 0.6), sharex=True,
                             squeeze=False)
    for ax, s in zip(axes[:, 0], shown):
        ax.axhspan(*s['expected_latency_ms'], color='0.9', zorder=0)
        ax.plot(s['_t'], s['_latency'], '.', ms=4, color='C0' if s['fit'] else 'C1')
        ax.set_ylabel('ms')
        ax.set_title(f"{s['label']} -> {s['event']}  ({s['n_matched']} matched"
                     + (', fit anchor)' if s['fit'] else ')'), fontsize=9, loc='left')
    axes[-1, 0].set_xlabel('binary_signals time (s)')
    fig.suptitle(f"Event table -> binary_signals latency (recorded - mapped); "
                 f"drift {diagnostics['drift_ppm']:+.2f} ppm", fontsize=10)
    fig.tight_layout()
    png_file = folder / f"{stem}.png"
    fig.savefig(png_file, dpi=120)
    plt.close(fig)
    return json_file, png_file
