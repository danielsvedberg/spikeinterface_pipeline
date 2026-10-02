import re
from preprocessing import parse_nidq as pni
import numpy as np
import pandas as pd

def get_stimulation_times_df(base_folder):
    bs = pni.get_binary_signals(base_folder, overwrite=True)
    bs_md = bs.metadata
    # get all rows of bs_md where column 'event' contains 'chrimson_on' or 'chr2_on'
    on_events = bs_md[bs_md['event'].str.contains('chrimson_on|chr2_on')]
    off_events = bs_md[bs_md['event'].str.contains('chrimson_off|chr2_off')]
    #make a dataframe with columns 'event', 'time', 'type' where type is 'on' or 'off'
    on_event_times = []
    for index, row in on_events.iterrows():
        ontimes = bs[index].t
        for t in ontimes:
            on_event_times.append({'event': row['event'], 'time': t, 'type': 'on'})
    off_event_times = []
    for index, row in off_events.iterrows():
        offtimes = bs[index].t
        for t in offtimes:
            off_event_times.append({'event': row['event'], 'time': t, 'type': 'off'})
    on_event_times_df = pd.DataFrame(on_event_times)
    off_event_times_df = pd.DataFrame(off_event_times)
    event_times_df = pd.concat([on_event_times_df, off_event_times_df], ignore_index=True)
    #order by time
    event_times_df = event_times_df.sort_values(by='time')
    return event_times_df


def get_stimulation_times(base_folder, nidaq_map=None, overwrite=False):
    # snippet for testing
    #from pathlib import Path
    #base_folder = Path(r"C:\Users\assad\Documents\recording_files\DS21\DS21_20251209")
    #bs = pni.get_binary_signals(base_folder, overwrite=False)

    if nidaq_map is not None:
        dinmap = nidaq_map.get('dinmap', None)
        ainmap = nidaq_map.get('ainmap', None)
        optoidx = nidaq_map.get('optoidx', None)
    else:
        dinmap=None
        ainmap=None
        optoidx=None

    bs = pni.get_binary_signals(base_folder, dinmap=dinmap, ainmap=ainmap, optoidx=optoidx, overwrite=overwrite, plot=True)

    bs_md = bs.metadata
    # get all rows of bs_md where column 'event' contains 'chrimson_on' or 'chr2_on'
    on_events = bs_md[bs_md['event'].str.contains('chrimson_on|chr2_on|solenoid_on')]
    off_events = bs_md[bs_md['event'].str.contains('chrimson_off|chr2_off')]
    # for each row in on_events, make a list of time times
    on_event_times = []
    labels = []
    for index, row in on_events.iterrows():
        ontimes = bs[index].t
        on_event_times.append(ontimes)
        labels.append(row['event'])

    off_event_times = []
    for index, row in off_events.iterrows():
        offtimes = bs[index].t
        off_event_times.append(offtimes)

    #flatten the lists into a single list
    on_event_times = [time for sublist in on_event_times for time in sublist]
    off_event_times = [time for sublist in off_event_times for time in sublist]

    #sort the lists
    on_event_times.sort()
    off_event_times.sort()

    return on_event_times, off_event_times

def get_stimulation_intervals(base_folder, nidaq_map=None, opsins=('chr2', 'chrimson'),
                              overwrite=False):
    """Every light pulse as an (n, 2) array of [on, off] seconds on the imec clock, sorted.

    Pairs each `<opsin>_on_<power>` event with its own `<opsin>_off_<power>` event, so
    the pairing is by name, not by position in one merged list. get_stimulation_times
    cannot be zipped into pulses: its on list also carries solenoid_on, which has no
    off (215 on against 187 off on DS45_20260811).
    """
    if nidaq_map is not None:
        dinmap, ainmap, optoidx = (nidaq_map.get(k, None) for k in ('dinmap', 'ainmap', 'optoidx'))
    else:
        dinmap = ainmap = optoidx = None
    bs = pni.get_binary_signals(base_folder, dinmap=dinmap, ainmap=ainmap, optoidx=optoidx,
                                overwrite=overwrite)
    names = bs.metadata['event'].astype(str)
    index = {n: i for i, n in zip(bs.metadata.index, names)}
    # '<opsin>_on_<power>' from the analog opto channel, bare '<opsin>_on' from a
    # digital one (parse_nidq.get_digital_ts); '<opsin>_trace' matches neither
    pattern = re.compile(r'^(%s)_on(_.+)?$' % '|'.join(map(re.escape, opsins)))
    pulses, found = [], set()
    for name, i in index.items():
        m = pattern.match(name)
        if m is None:
            continue
        opsin, suffix = m.group(1), m.group(2) or ''
        off_name = f'{opsin}_off{suffix}'
        if off_name not in index:
            raise ValueError(f'{name} has no matching {off_name}')
        on, off = np.asarray(bs[i].t), np.asarray(bs[index[off_name]].t)
        if on.size != off.size or np.any(off < on):
            raise ValueError(f'{name}: {on.size} on vs {off.size} off events do not pair')
        pulses.append(np.column_stack([on, off]))
        found.add(opsin)
    missing = [o for o in opsins if o not in found]
    if missing and found:
        print(f'get_stimulation_intervals: no pulses found for {missing}')
    if not pulses:
        print(f'get_stimulation_intervals: no {"/".join(opsins)} pulses in this session')
        return np.empty((0, 2))
    iv = np.concatenate(pulses)
    return iv[np.argsort(iv[:, 0])]


def get_event_times(base_folder, nidaq_map=None, events=('lick_on',), overwrite=False):
    """Timestamps of named binary_signals events, as {event: np.array of seconds}.

    The names are the 'event' column of the binary_signals metadata -- 'lick_on',
    'start_tone_on', 'solenoid_on' and so on -- and the times are already on the imec
    clock, the same one get_stimulation_times returns and the same one the recording's
    frame indices count. An event that is not in the metadata comes back as an empty
    array rather than raising: a session recorded without a lick line is a real thing,
    and it should not take the pipeline down.

    Reads the cached parse by default. get_binary_signals(overwrite=False) does more
    than read the cache -- it also re-mirrors the file and rewrites the event table and
    the trial raster -- so this is not free, just cheap next to reparsing the nidq.
    """
    if nidaq_map is not None:
        dinmap = nidaq_map.get('dinmap', None)
        ainmap = nidaq_map.get('ainmap', None)
        optoidx = nidaq_map.get('optoidx', None)
    else:
        dinmap = ainmap = optoidx = None

    bs = pni.get_binary_signals(base_folder, dinmap=dinmap, ainmap=ainmap,
                                optoidx=optoidx, overwrite=overwrite)
    bs_md = bs.metadata
    out = {}
    for name in events:
        rows = bs_md[bs_md['event'] == name]
        if len(rows) == 0:
            print(f'get_event_times: no {name!r} event in binary_signals')
            out[name] = np.array([], dtype=float)
            continue
        out[name] = np.sort(np.concatenate([np.asarray(bs[i].t) for i in rows.index]))
    return out


def get_default_nidaq_map(): #TODO: fold this into parse_nidq
    dinmap = {
        'licks': 0,
        'timing_pulse': 1,
        'solenoid': 2,
        'tone': 3
    }
    ainmap = {
        'photometry': 0,
        'accelerometer_x': 1,
        'accelerometer_y': 2,
        'accelerometer_z': 3,
        'snc': 4,
        'chrimson': 5,
        'chr2': 6
    }

    optoidx = {
        'chrimson': [20, 40, 60, 80, 100, 200],
        'chr2': [20, 40, 60, 80, 100, 200]
    }
    nidaq_map = {
        'dinmap': dinmap,
        'ainmap': ainmap,
        'optoidx': optoidx
    }
    return nidaq_map

