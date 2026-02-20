import matplotlib.pyplot as plt
import spikeinterface.full as si
from pathlib import Path
import matplotlib
matplotlib.use('Qt5Agg')
#make a dataframe from temp_metrics
import pandas as pd
import numpy as np
import spikeinterface.curation as sic
import metrics_curation as mc
import get_stimulation_frames as gsf
import shutil
from spikeinterface.widgets import plot_sorting_summary
import parse_nidq as pni
import pynapple as nap
import parse_opto_tagging as pot
import plot_psth as pp

def run_pipeline(base_folder, nidaq_map=None, bad_chans=None, export_raw_summary=False, plot_psth=False, plot_tuning_curves=False):
    #base folder needs to be a Path object
    # Usually, you would read in your raw recording
    spikeglx_folder = list(base_folder.glob('*_g*'))[0]

    on_event_times, off_event_times = gsf.get_stimulation_times(base_folder, nidaq_map, overwrite=True)
    #concatenate into one list
    artifact_times = on_event_times + off_event_times
    artifact_times.sort()

    stream_names, stream_ids = si.get_neo_streams('spikeglx', spikeglx_folder)
    raw_rec = si.read_spikeglx(spikeglx_folder, stream_id='imec0.ap')
    raw_rec.get_probe().to_dataframe()
    sf = raw_rec.get_sampling_frequency()
    artifact_idxs = [int(t * sf) for t in artifact_times]

    chids = raw_rec.channel_ids
    if bad_chans is None:
        bad_chans = []
    bad_chids = chids[bad_chans]

    my_protocol = {
        'preprocessing': {
            'notch_filter': {'freq': 60, 'q': 300},
            'bandpass_filter': {},
            'detect_and_remove_bad_channels': {'bad_channel_ids' : bad_chids},
            'phase_shift': {},
            'common_reference': {'operator': 'median'},
            'remove_artifacts': {'list_triggers': artifact_idxs, 'ms_before': 1, 'ms_after': 1.5},
        },
        'sorting': {
            'sorter_name': 'kilosort4',
            'verbose': False,
            'folder': base_folder / 'kilosort4_output',
            'remove_existing_folder': True,
            'progress_bar': False
        },
        'postprocessing': {
            'random_spikes': {},
            'isi_histograms': {},
            'correlograms': {},
            'noise_levels': {},
            'principal_components': {},
            'waveforms': {},
            'templates': {},
            'spike_amplitudes': {},
            'amplitude_scalings': {},
            'spike_locations': {},
            'template_metrics': {'include_multi_channel_metrics':True},
            'template_similarity': {},
            'unit_locations': {'method': 'center_of_mass'},
            'quality_metrics': {},
        }
    }

    preprocessed_rec = si.apply_preprocessing_pipeline(raw_rec, my_protocol['preprocessing'])
    preprocessed_rec.save(folder=base_folder / 'preprocess', format='binary', n_jobs=23, progress_bar=True, overwrite=True)
    preprocessed_rec = si.load(base_folder / 'preprocess')
    sorting = si.run_sorter(recording=preprocessed_rec, **my_protocol['sorting'])

    sorting = si.load(base_folder / 'kilosort4_output' )
    preprocessed_rec = si.load(base_folder / 'preprocess')
    sorting_clean = si.remove_duplicated_spikes(sorting, method='keep_first_iterative')
    sorting_clean = si.remove_excess_spikes(sorting_clean, preprocessed_rec)

    analyzer = si.create_sorting_analyzer(recording=preprocessed_rec, sorting=sorting_clean, folder=base_folder / 'analyzer', format='binary_folder', n_jobs=-1, overwrite=True)

    job_kwargs=dict(n_jobs=23, progress_bar=True)
    analyzer.compute(my_protocol['postprocessing'], **job_kwargs)
    analyzer = si.load_sorting_analyzer(folder=base_folder / 'analyzer')

    if export_raw_summary:
        si.export_report(sorting_analyzer=analyzer, output_folder=base_folder / 'sorting_summary_raw', remove_if_exists=True)

    template_diff_thresh = [0.05, 0.15, 0.25]
    presets = ["x_contaminations"] * len(template_diff_thresh)
    steps_params = [
        {"template_similarity": {"template_diff_thresh": i}}
        for i in template_diff_thresh
    ]

    analyzer_merged = sic.auto_merge_units(
        analyzer,
        presets=presets,
        steps_params=steps_params,
        recursive=True,
        merging_mode='hard',
        **job_kwargs,
    )

    nonred_units = sic.remove_redundant_units(analyzer_merged, remove_strategy="minimum_shift")
    analyzer_clean = analyzer_merged.select_units(nonred_units.unit_ids)

    keep_unit_ids = mc.curate_units(analyzer_clean)
    analyzer_clean = analyzer_clean.select_units(keep_unit_ids)
    #plot_sorting_summary(sorting_analyzer=analyzer_clean, curation=True, backend='spikeinterface_gui')

    analyzer_path = base_folder / 'analyzer_clean'
    if analyzer_path.exists():
        shutil.rmtree(analyzer_path)
    analyzer_clean.save_as(folder=base_folder / 'analyzer_clean', format='binary_folder')
    analyzer_clean = si.load_sorting_analyzer(folder=analyzer_path)

    #export to pynapple
    from spikeinterface.exporters import to_pynapple_tsgroup
    my_tsgroup = to_pynapple_tsgroup(analyzer_clean,
        attach_unit_metadata=True)
    pynapple_folder = base_folder / 'pynapple'
    pynapple_folder.mkdir(exist_ok=True)
    my_tsgroup.save(pynapple_folder / 'spikes.npz')

    if plot_psth:
        plot_psth(base_folder, overwrite=True)
    if plot_tuning_curves:
        plot_tuning_curves(base_folder, overwrite=True)


def parse_nidq_kwargs(nidq_map_kwargs):
    if nidq_map_kwargs is not None:
        dinmap = nidq_map_kwargs.get('dinmap', None)
        ainmap = nidq_map_kwargs.get('ainmap', None)
        optoidx = nidq_map_kwargs.get('optoidx', None)
    else:
        dinmap=None
        ainmap=None
        optoidx=None
    return dinmap, ainmap, optoidx


def get_pynapple_data(base_folder, nidq_map_kwargs=None, overwrite=False):
    dinmap, ainmap, optoidx = parse_nidq_kwargs(nidq_map_kwargs)

    pynapple_folder = base_folder / "pynapple"
    bs = pni.get_binary_signals(base_folder, dinmap=dinmap, ainmap=ainmap, optoidx=optoidx, overwrite=overwrite,
                                plot=True)
    ephys_file = pynapple_folder / "spikes.npz"
    ephys_data = nap.load_file(ephys_file)
    return ephys_data, bs, pynapple_folder


def plot_psth(base_folder, overwrite=False, nidq_map_kwargs=None):
    ephys_data, bs, pynapple_folder = get_pynapple_data(base_folder, nidq_map_kwargs, overwrite)
    pp.plot_on_events_psth(ephys_data, bs, pynapple_folder)


def plot_tuning_curves(base_folder, overwrite=False, nidq_map_kwargs=None):
    ephys_data, bs, pynapple_folder = get_pynapple_data(base_folder, nidq_map_kwargs, overwrite)
    pot.plot_all_tuning_curves(ephys_data, bs, pynapple_folder, overwrite=overwrite)


def launch_sorting_gui(base_folder): #convenience function to launch the sorting GUI on the cleaned analyzer
    analyzer_clean = si.load_sorting_analyzer(folder=base_folder / 'analyzer_clean')
    si.export_report(sorting_analyzer=analyzer_clean, output_folder=base_folder / 'sorting_summary_clean', remove_if_exists=True)

########################################################################################################################
########################################################################################################################
########################################################################################################################

# pathlist = [Path(r"C:\Users\assad\Documents\recording_files\DS21\DS21_20251209"),
#             Path(r"C:\Users\assad\Documents\recording_files\DS21\DS21_20251210"),#]
#             Path(r"C:\Users\assad\Documents\recording_files\DS21\DS21_20251211"),
#             Path(r"C:\Users\assad\Documents\recording_files\DS21\DS21_20251212"),
#             Path(r"C:\Users\assad\Documents\recording_files\DS21\DS21_20251213"),
#             Path(r"C:\Users\assad\Documents\recording_files\DS1\DS1_20251117"),
#             Path(r"C:\Users\assad\Documents\recording_files\DS1\DS1_20251118"),
#             Path(r"C:\Users\assad\Documents\recording_files\DS1\DS1_20251119"),]
pathlist = [Path(r"C:\Users\assad\Documents\recording_files\DS23\DS23_20260211")] #modify the paths as needed
bad_chans = [44, 142, 154, 181, 198, 199, 228, 383]  # add 11?
for folder in pathlist:
    run_pipeline(folder, bad_chans=bad_chans)

