"""Synthetic checks for preprocessing.event_table_licks (no data needed).

    python -m tests.event_table_licks_test   (or under pytest)

A fake session: cues every ~20 s, rewards and aborts each triggered by a lick, on an Arduino clock
that is offset and runs 13 ppm off the recording clock. The fit must recover the clock and keep
every trigger lick before its tone; a table from a different session must be rejected.
"""
import numpy as np
import pandas as pd

from preprocessing import event_table_licks as etl

SLOPE, INTERCEPT = 1 + 13e-6, 323.14


def _session(seed=0, n_trials=120):
    rng = np.random.default_rng(seed)
    cue_ms = np.round(40000 + np.cumsum(rng.uniform(18000, 24000, n_trials)))
    outcome_ms = cue_ms + np.round(rng.uniform(800, 5000, n_trials))
    rewarded = rng.random(n_trials) < 0.3
    rows = [(t, 'CUE_ON') for t in cue_ms]
    rows += [(t, 'REWARD' if r else 'ABORT') for t, r in zip(outcome_ms, rewarded)]
    licks_ms = np.concatenate([outcome_ms, outcome_ms[rewarded] + 150, outcome_ms[rewarded] + 300])
    rows += [(t, 'LICK') for t in licks_ms]
    table = pd.DataFrame(rows, columns=['time', 'label']).sort_values('time')
    table = table.assign(time_s=table['time'] / 1000.0)

    to_bs = lambda ms: SLOPE * ms / 1000.0 + INTERCEPT
    jitter = rng.uniform(0, 1e-3, n_trials)  #the true edge lies somewhere inside the logged ms
    events = {'start_tone_on': to_bs(cue_ms),
              'solenoid_on': to_bs(outcome_ms[rewarded]) + jitter[rewarded],
              'reward_tone_on': to_bs(outcome_ms[rewarded]) + jitter[rewarded] + 1e-4,
              'error_tone_on': to_bs(outcome_ms[~rewarded]) + jitter[~rewarded] + 1.3e-3}
    return table, events


def test_recovers_clock_and_keeps_trigger_licks_before_tones():
    table, events = _session()
    licks, d = etl.event_table_licks(table, events)
    assert abs(d['slope'] - SLOPE) < 1e-9
    assert abs(d['intercept_s'] - INTERCEPT) < 1e-5
    assert d['n_licks'] == len(etl.label_times(table, 'LICK'))
    for tones in (events['reward_tone_on'], events['error_tone_on']):
        last_before = licks[np.searchsorted(licks, tones) - 1]
        assert np.all(tones - last_before > 0)


def test_late_trigger_lick_is_moved_before_its_tone():
    table, events = _session()
    events['reward_tone_on'] = events['reward_tone_on'] - 0.6e-3  #most tones now precede the mapped lick
    licks, d = etl.event_table_licks(table, events)
    assert d['n_trigger_licks_moved'] > 0
    tones = events['reward_tone_on']
    assert np.all(tones - licks[np.searchsorted(licks, tones) - 1] > 0)


def test_rejects_a_table_from_another_session():
    table, _ = _session(seed=0)
    _, events = _session(seed=1)
    try:
        etl.event_table_licks(table, events)
    except ValueError:
        return
    raise AssertionError('a table from another session was accepted')


def test_licks_outside_time_support_are_dropped():
    import pynapple as nap
    table, events = _session()
    support = nap.IntervalSet(start=0, end=events['start_tone_on'][60])
    licks, d = etl.event_table_licks(table, events, time_support=support)
    assert d['n_licks_outside_support'] > 0
    assert licks.max() <= support.end[0]


def test_trigger_lick_is_kept_before_its_solenoid():
    table, events = _session()
    rewarded_x = etl.label_times(table, 'REWARD')
    #solenoids that fire on the logged ms tick, i.e. a hair before the 1 ms-quantized mapped lick
    events['solenoid_on'] = SLOPE * rewarded_x + INTERCEPT - 2e-5
    events['reward_tone_on'] = events['solenoid_on'] + 1e-4
    licks, d = etl.event_table_licks(table, events)
    assert d['n_trigger_licks_moved'] == len(rewarded_x)
    for t in (events['solenoid_on'], events['reward_tone_on'], events['error_tone_on']):
        assert np.all(t - licks[np.searchsorted(licks, t) - 1] > 0)


def test_get_lick_ts_honours_lick_source():
    """auto falls back to the table only for an empty NIDQ line; nidq never uses the table."""
    import tempfile
    from pathlib import Path
    import pynapple as nap
    from preprocessing import parse_nidq as pn

    tmp = Path(tempfile.mkdtemp())
    table, events = _session()
    table[['time', 'label']].to_csv(tmp / 'S_eventTable.csv', index=False)
    n_table = len(etl.label_times(table, 'LICK'))
    support = nap.IntervalSet(0, 1e5)

    class Paths:
        def __init__(self, has_table): self.has_table = has_table
        recording_folder = tmp
        @property
        def event_table(self): return tmp / 'S_eventTable.csv' if self.has_table else None
        def binary_signals_folder(self): return tmp

    def stub_digital_ts(n):
        def get_digital_ts(digital_events, time_support, map2imectime, channel, event_name='lick'):
            t = np.linspace(500, 600, n)
            return [nap.Ts(t=t, time_support=time_support),
                    nap.Ts(t=t + 0.01, time_support=time_support)], ['lick_on', 'lick_off']
        return get_digital_ts

    expected = {  #(nidq licks, has table, lick_source) -> lick_on count, or the error raised
        (0, True, 'auto'): n_table, (0, True, 'nidq'): 0, (0, True, 'event_table'): n_table,
        (0, False, 'auto'): 0, (0, False, 'nidq'): 0, (0, False, 'event_table'): FileNotFoundError,
        (50, True, 'auto'): 50, (50, True, 'nidq'): 50, (50, True, 'event_table'): n_table,
        (50, False, 'auto'): 50, (50, False, 'nidq'): 50, (50, False, 'event_table'): FileNotFoundError,
    }
    original = pn.get_digital_ts
    try:
        for (n, has_table, source), want in expected.items():
            pn.get_digital_ts = stub_digital_ts(n)
            try:
                (lick_on, _), _ = pn.get_lick_ts(None, support, None, {'licks': 0}, Paths(has_table),
                                                 events, lick_source=source)
                got = len(lick_on)
            except Exception as e:
                got = type(e)
            assert got == want, f"nidq={n} table={has_table} source={source}: {got} != {want}"
    finally:
        pn.get_digital_ts = original


if __name__ == '__main__':
    for name, test in list(globals().items()):
        if name.startswith('test_'):
            test()
            print('ok', name)
