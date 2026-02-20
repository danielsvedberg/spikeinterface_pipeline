import pynapple as nap
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
#joblib
from joblib import Parallel, delayed

def plot_psth(timestamps, tref, unitID, event_ID, save_folder=None, minmax=(-1, 1), off_event=None):

    peth = nap.compute_perievent(
        timestamps=timestamps,
        tref=tref,
        minmax=minmax,
        time_unit="s")

    #if off_event is a ts or array
    if off_event is not None and hasattr(off_event, 't'):
        off_events = off_event.t-tref.t
        off_event = np.median(off_events)

    spikes = peth.to_tsd()
    #check that spikes is not empty
    if len(spikes) == 0:
        print(f"No spikes found for Unit {unitID} around event {event_ID}")
        return

    fig, axs = plt.subplots(2,1, figsize=(10, 6), sharex=True)

    ax = axs[0]
    ax.plot(np.mean(peth.count(0.01), 1) / 0.01, linewidth=2, color="black")
    ax.set_ylabel("Rate (spikes/sec, 0.01s bins)")

    ax = axs[1]
    #get the number of trials
    ntrials = max(peth) - min(peth)
    ms = -1/1000 * ntrials  + 3
    ax.plot(spikes, "|", markersize=ms, color="black", mew=1)
    ax.set_ylabel("trial/event #")
    ax.set_xlabel("time from event (s)")

    for i in range(2):
        ax = axs[i]
        ax.set_xlim(minmax)
        ax.axvline(0.0)
        if off_event is not None:
            ax.axvline(off_event, color='black', linestyle='--')
    #add a suptitle
    suptitle = f"Unit {unitID} PSTH around event {event_ID}"
    fig.suptitle(suptitle)
    if save_folder is not None:
        fig.savefig(save_folder / f"Unit_{unitID}_PSTH_event_{event_ID}.png")
        plt.close(fig)
    else:
        plt.show()

def plot_on_events_psth(ephys_data, sigs, pynapple_folder):
    sigs_metadata = sigs.metadata
    on_events_idx = sigs_metadata[sigs_metadata['event'].str.contains('_on')].index.tolist()

    neurons_idx = ephys_data.keys()

    psth_folder = pynapple_folder / "psth_plots"
    psth_folder.mkdir(exist_ok=True)
    for event in on_events_idx:
        tref = sigs[event]
        event_name = sigs_metadata.loc[event, 'event']
        parallel_ = Parallel(n_jobs=-1, verbose=5)
        parallel_(
            delayed(plot_psth)(
                ephys_data[neuron],
                tref,
                neuron,
                event_name,
                save_folder=psth_folder
            ) for neuron in neurons_idx)

    cue_events_idx = sigs_metadata[sigs_metadata['event'].str.contains('_cues')].index.tolist()
    for event in cue_events_idx:
        tref = sigs[event]
        event_name = sigs_metadata.loc[event, 'event']
        parallel_ = Parallel(n_jobs=-1, verbose=5)
        parallel_(
            delayed(plot_psth)(
                ephys_data[neuron],
                tref,
                neuron,
                event_name,
                save_folder=psth_folder,
                minmax=(-2, 2)
            ) for neuron in neurons_idx)

    first_licks_idx = sigs_metadata[sigs_metadata['event'].str.contains('_first_licks')].index.tolist()
    for event in first_licks_idx:
        tref = sigs[event]
        event_name = sigs_metadata.loc[event, 'event']
        parallel_ = Parallel(n_jobs=-1, verbose=5)
        parallel_(
            delayed(plot_psth)(
                ephys_data[neuron],
                tref,
                neuron,
                event_name,
                save_folder=psth_folder,
                minmax=(-4, 1)
            ) for neuron in neurons_idx)

    opto_events = sigs_metadata[sigs_metadata['event'].str.contains('chr')]
    opto_off_events_idx = opto_events[opto_events['event'].str.contains('_off')].index.tolist()
    #only get opto events that end with '_on'
    opto_on_events_idx = opto_events[opto_events['event'].str.contains('_on')].index.tolist()
    for i, event in enumerate(opto_on_events_idx):
        tref = sigs[event]
        off_event = sigs[opto_off_events_idx[i]]
        event_name = sigs_metadata.loc[event, 'event']

        parallel_ = Parallel(n_jobs=-1, verbose=5)
        parallel_(
            delayed(plot_psth)(
                ephys_data[neuron],
                tref,
                neuron,
                event_name,
                off_event=off_event,
                save_folder=psth_folder,
                minmax=(-0.5,0.5)
            ) for neuron in neurons_idx)


def test(base_folder=None):
    if base_folder is None:
        base_folder = Path("C:\\Users\\assad\\Documents\\analysis_files\\DS13\\DS13_20250822")
    #    base_folder = Path(r"C:\Users\assad\Documents\analysis_files\DS13\DS13_20250905")

    pynapple_folder = base_folder / "pynapple"

    ephys_file = pynapple_folder / "spikes.npz"
    ephys_data = nap.load_file(ephys_file)
    sigs_file = pynapple_folder / "binary_signals.npz"
    sigs = nap.load_file(sigs_file)

    plot_on_events_psth(ephys_data, sigs, pynapple_folder)

def plot_all():
    base_folders = [
        Path(r"C:\Users\assad\Documents\analysis_files\DS13\DS13_20250822"),
        Path(r"C:\Users\assad\Documents\analysis_files\DS13\DS13_20250905"),
    ]
    for base_folder in base_folders:
        test(base_folder=base_folder)

    #get sigs_metadata rows where binary_event_name contains 'on'

