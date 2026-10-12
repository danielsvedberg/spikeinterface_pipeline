"""Lick-triggered voltage plots for every session of one animal.

    python -m scripts.lick_voltage_plots                     # DS45, the default
    python -m scripts.lick_voltage_plots D:\...\DS46 --events lick_on solenoid_on

For each session it draws the lick-centred average trace before and after
preprocessing -- every channel's trial-average in grey, the channel-and-trial average
in black -- and writes it to that session's preprocess/qc folder.

Nothing is recomputed. The plot reads the raw SpikeGLX stream and the *saved* preprocess
folder, so it shows the recording that was actually sorted rather than a chain rebuilt
now, and it reads lick times out of the cached binary_signals parse. A session that has
not been preprocessed, or whose nidq has not been parsed, is therefore skipped rather
than quietly triggering an hour of pipeline work.

Sessions are independent, so one that fails is reported and the rest carry on: on DS45
some sessions are missing products and one raises on load. The run ends with a table of
what was drawn, what was skipped and why.
"""

import argparse
import traceback
from pathlib import Path

import matplotlib
matplotlib.use('Agg')          # no windows: this is a batch job over whole sessions

import numpy as np
import spikeinterface.full as si

from preprocessing import get_stimulation_frames as gsf
from preprocessing import preprocess_qc as pqc
from utils import file_structure as fs

DEFAULT_ANIMAL = Path(r'C:\experiment_data\recording_files\DS45')


class SessionSkipped(Exception):
    """A session that cannot be plotted, for a reason the preflight checks anticipated.

    Separated from everything else so the two get reported differently: a missing
    preprocess folder is a fact about the session and one line says it, while an
    unexpected exception is a fact about the code and deserves its traceback. Printing
    a traceback for both buries the second in the first.
    """


def session_folders(animal_folder):
    """Session directories under an animal folder, in date order.

    Named <animal>_<date>, which is what excludes the odd folder out -- DS45 has a
    training_data directory beside the sessions.
    """
    animal_folder = Path(animal_folder)
    return sorted(d for d in animal_folder.iterdir()
                  if d.is_dir() and d.name.startswith(f'{animal_folder.name}_'))


def plot_one_session(base_folder, events=('lick_on',), ms_window=200.0,
                     max_events=None):
    """Draw the event-triggered averages for one session. Returns (out_folder, stats).

    Raises SessionSkipped if the session is missing a product it needs; anything else
    that comes out of here is a genuine failure.
    """
    paths = fs.SessionPaths(base_folder)
    if paths.preprocess_folder is None:
        raise SessionSkipped('not preprocessed yet (no preprocess folder)')
    # get_binary_signals falls back to reparsing the nidq when the cache is missing,
    # which is real work and can fail on its own terms. This asks first, so an unparsed
    # session is a one-line skip instead
    try:
        paths.binary_signals_file()
    except FileNotFoundError as e:
        raise SessionSkipped(f'nidq not parsed yet ({e})') from None

    raw_rec = si.read_spikeglx(paths.spikeglx_folder(), stream_id='imec0.ap')
    # the saved chain, not a rebuilt one: detect_and_interpolate_bad_channels samples
    # random chunks, so a chain rebuilt now can differ from the one on disk
    final_rec = si.load(paths.preprocess_folder)

    sf = raw_rec.get_sampling_frequency()
    times = gsf.get_event_times(base_folder, events=events)
    triggers = {name: (t * sf).astype(np.int64) for name, t in times.items() if t.size}
    if not triggers:
        raise SessionSkipped(f'no {" or ".join(events)} events in the binary signals')

    out_folder = paths.preprocess_folder / 'qc'
    stats = pqc.run_event_average_qc(raw_rec, final_rec, triggers, out_folder,
                                     ms_window=ms_window, max_events=max_events)
    return out_folder, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('animal_folder', nargs='?', default=DEFAULT_ANIMAL, type=Path,
                    help=f'animal folder in recording_files (default {DEFAULT_ANIMAL})')
    ap.add_argument('--events', nargs='+', default=['lick_on'],
                    help='binary_signals event names to align on (default lick_on)')
    ap.add_argument('--ms-window', type=float, default=200.0,
                    help='half window around each event, ms (default 200)')
    ap.add_argument('--max-events', type=int, default=None,
                    help='cap the events averaged per session (default: all of them)')
    args = ap.parse_args()

    sessions = session_folders(args.animal_folder)
    print(f'{len(sessions)} sessions under {args.animal_folder}\n')

    done, failed = [], []
    for base_folder in sessions:
        print(f'=== {base_folder.name} ===')
        try:
            out_folder, stats = plot_one_session(
                base_folder, events=tuple(args.events), ms_window=args.ms_window,
                max_events=args.max_events)
        except SessionSkipped as e:
            failed.append((base_folder.name, str(e)))
            print(f'  SKIPPED -- {e}')
            continue
        except Exception as e:
            # one session's unreadable file says nothing about the next one's, and a
            # batch that stops on the first is a batch that has to be restarted by hand
            # around every bad session. This one gets its traceback: it is not a
            # condition the preflight checks knew to look for.
            failed.append((base_folder.name, f'{type(e).__name__}: {e}'))
            print(f'  FAILED -- {type(e).__name__}: {e}')
            traceback.print_exc()
            continue
        for name, st in stats.items():
            if 'error' in st:
                print(f"  {name}: no plot -- {st['error']}")
                continue
            unit = st['units'].get(pqc.FINAL_STAGE, 'uV')
            print(f"  {name}: {st['n_events']} events, swing "
                  f"{st['swing_raw_uV']:.1f} -> {st['swing_final_uV']:.1f} {unit}")
        done.append((base_folder.name, out_folder))
        print(f'  -> {out_folder}')

    print(f'\n{"=" * 60}\ndrawn:   {len(done)}')
    for name, folder in done:
        print(f'  {name}  {folder}')
    print(f'skipped: {len(failed)}')
    for name, why in failed:
        print(f'  {name}  {why}')


if __name__ == '__main__':
    main()
