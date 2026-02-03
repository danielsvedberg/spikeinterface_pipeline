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

def run_pipeline(base_folder):
    #base folder needs to be a Path object
    # Usually, you would read in your raw recording
    spikeglx_folder = list(base_folder.glob('*_g*'))[0]

    on_event_times, off_event_times = gsf.get_stimulation_frames(base_folder, overwrite=True)
    #concatenate into one list
    artifact_times = on_event_times + off_event_times
    artifact_times.sort()

    stream_names, stream_ids = si.get_neo_streams('spikeglx', spikeglx_folder)
    raw_rec = si.read_spikeglx(spikeglx_folder, stream_id='imec0.ap')
    raw_rec.get_probe().to_dataframe()

    chids = raw_rec.channel_ids
    bad_chans = [44,142,154,181,198,199,228,383]
    bad_chids = chids[bad_chans]

    my_protocol = {
        'preprocessing': {
            'notch_filter': {'freq': 60, 'q': 300},
            'bandpass_filter': {},
            'detect_and_remove_bad_channels': {'bad_channel_ids' : bad_chids},
            'phase_shift': {},
            'common_reference': {'operator': 'median'},
            'remove_artifacts': {'list_triggers': artifact_times, 'ms_before': 0.25, 'ms_after': 2},
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
    preprocessed_rec.save(folder=base_folder / 'preprocess', format='binary', n_jobs=23, progress_bar=True)
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

    #si.export_report(sorting_analyzer=analyzer, output_folder=base_folder / 'sorting_summary')

    nonred_units = sic.remove_redundant_units(analyzer)
    analyzer_clean = analyzer.select_units(nonred_units.unit_ids)

    keep_unit_ids = mc.curate_units(analyzer_clean)
    analyzer_clean = analyzer.select_units(keep_unit_ids)

    analyzer_path = base_folder / 'analyzer_clean'
    analyzer_clean.save_as(folder=base_folder / 'analyzer_clean', format='binary_folder')
    analyzer_clean = si.load_sorting_analyzer(folder=analyzer_path)

    si.export_report(sorting_analyzer=analyzer_clean, output_folder=base_folder / 'sorting_summary_clean')

    #export to pynapple
    from spikeinterface.exporters import to_pynapple_tsgroup
    my_tsgroup = to_pynapple_tsgroup(analyzer_clean,
        attach_unit_metadata=True)
    pynapple_folder = base_folder / 'pynapple'
    pynapple_folder.mkdir(exist_ok=True)
    my_tsgroup.save(pynapple_folder / 'spikes.npz')



#base_folder = Path("C:\\Users\\assad\\Documents\\analysis_files\\DS13\\DS13_20250822")
#base_folder = Path(r"C:\Users\assad\Documents\analysis_files\DS13\DS13_20250905")
#base_folder = Path(r"C:\Users\assad\Documents\analysis_files\DS13\DS13_20250824")
#base_folder = Path(r"C:\Users\assad\Documents\recording_files\DS1\DS1_20251117")
#base_folder =  Path(r"C:\Users\assad\Documents\recording_files\DS1\DS1_20251118")
#base_folder = Path(r"C:\Users\assad\Documents\recording_files\DS21\DS21_20251209")
#base_folder = Path(r"C:\Users\assad\Documents\recording_files\DS21\DS21_20251211")
#base_folder = Path(r"C:\Users\assad\Documents\recording_files\DS21\DS21_20251212")
#base_folder = Path(r"C:\Users\assad\Documents\recording_files\DS21\DS21_20251213")
pathlist = [Path(r"C:\Users\assad\Documents\recording_files\DS21\DS21_20251209"),
            Path(r"C:\Users\assad\Documents\recording_files\DS21\DS21_20251210"),#]
            Path(r"C:\Users\assad\Documents\recording_files\DS21\DS21_20251211")]
            #Path(r"C:\Users\assad\Documents\recording_files\DS21\DS21_20251213"),
            #Path(r"C:\Users\assad\Documents\recording_files\DS1\DS1_20251117"),
            #Path(r"C:\Users\assad\Documents\recording_files\DS1\DS1_20251118"),
            #Path(r"C:\Users\assad\Documents\recording_files\DS1\DS1_20251119")]
for folder in pathlist:
    run_pipeline(folder)