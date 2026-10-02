from pathlib import Path
import spikeinterface.full as si
from spikeinterface.extractors import read_split_intan_files
from utils.ucla128dn_probe.ucla128DN import get_probe
import spikeinterface.preprocessing as spre
import spikeinterface.curation as sic
import spikeinterface.sorters as se
import pynapple as nap
from utils import file_structure as fs


def find_rhd_folder(base_folder):
    rhd_folders = [f for f in Path(base_folder).rglob("*") if f.is_dir() and any(f.glob("*.rhd"))]
    if len(rhd_folders) > 1:
        raise ValueError(f"Multiple .rhd folders found: {rhd_folders}")
    return rhd_folders[0] if rhd_folders else None

if __name__ == "__main__":
    probe = get_probe()

    base_folder = Path(r"C:\Users\assad\Documents\recording_files\Neptun\Neptun_20260626")
    #sorting products go to sorting_files, pynapple products to analysis_files, and raw .rhd data is
    #only read from the recording folder (see utils/file_structure.py)
    folders = fs.SessionPaths(base_folder).create_output_folders()


    rhd_folder = find_rhd_folder(folders.recording_folder)

    ephys = read_split_intan_files(rhd_folder, stream_name='RHD2000 amplifier channel')
    ephys = spre.unsigned_to_signed(ephys)
    aux_in = read_split_intan_files(rhd_folder, stream_name='RHD2000 auxiliary input channel')
    adc_in = read_split_intan_files(rhd_folder, stream_name='USB board ADC input channel')
    dig_in = read_split_intan_files(rhd_folder, stream_name='USB board digital input channel')
    fs = dig_in.sampling_frequency
    dig_in = dig_in.get_traces()
    adc_in = adc_in.get_traces()

    ephys = ephys.set_probe(probe)


    ephys.get_probe().to_dataframe()
    sf = ephys.get_sampling_frequency()


    chids = ephys.channel_ids
    bad_chans = []
    bad_chids = chids[bad_chans]

    my_protocol = {
        'preprocessing': {
            'notch_filter': {'freq': 60, 'q': 300},
            'bandpass_filter': {},
            'detect_and_remove_bad_channels': {'bad_channel_ids': bad_chids},
            #'phase_shift': {},
            'common_reference': {'operator': 'median'},
        },
        'sorting': {
            'sorter_name': 'kilosort4',
            'verbose': True,
            'folder': folders.sorting_path(fs.KS_FOLDER),
            'remove_existing_folder': True,
            'progress_bar': True
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
            'template_metrics': {'include_multi_channel_metrics': True},
            'template_similarity': {},
            'unit_locations': {'method': 'center_of_mass'},
            'quality_metrics': {},
        }
    }

    preprocessed_rec = si.apply_preprocessing_pipeline(ephys, my_protocol['preprocessing'])
    #a write names its destination raw: folders.preprocess_folder is None until this line has run
    preprocessed_rec.save(folder=folders.recording_folder / fs.PREPROCESS_FOLDER, format='binary', n_jobs=-1, progress_bar=True, overwrite=True)
    # preprocessed_rec = si.load(folders.preprocess_folder)
    sorting = si.run_sorter(recording=preprocessed_rec, **my_protocol['sorting'])

    # sorting = si.load(folders.ks_folder)
    # preprocessed_rec = si.load(folders.preprocess_folder)
    sorting_clean = si.remove_duplicated_spikes(sorting, method='keep_first_iterative')

    analyzer = si.create_sorting_analyzer(recording=preprocessed_rec, sorting=sorting_clean,
                                          folder=folders.sorting_path('analyzer'), format='binary_folder', n_jobs=-1,
                                          overwrite=True)

    job_kwargs = dict(n_jobs=23, progress_bar=True)
    analyzer.compute(my_protocol['postprocessing'], **job_kwargs)
    analyzer = si.load_sorting_analyzer(folder=folders.sorting_path('analyzer'))

    from spikeinterface_gui import run_mainwindow
    # open and run the Qt app
    run_mainwindow(analyzer, mode="desktop", curation=True)

    si.export_report(sorting_analyzer=analyzer, output_folder=folders.sorting_path('sorting_summary_raw'), remove_if_exists=True)
    template_diff_thresh = [0.05, 0.1]

    to_merge = sic.compute_merge_unit_groups(
        analyzer,
        presets="feature_neighbors",
        template_similarity_threshold=template_diff_thresh[0],
        merging_mode='hard',
        **job_kwargs
    )
    if to_merge != []:
        analyzer_merged = analyzer.merge_units(to_merge,
                                               merging_mode='hard',
                                               **job_kwargs)
    else:
        analyzer_merged = analyzer

    to_merge = sic.compute_merge_unit_groups(
        analyzer_merged,
        presets="feature_neighbors",
        template_similarity_threshold=template_diff_thresh[1],
        **job_kwargs
    )

    if to_merge != []:
        analyzer_merged = analyzer_merged.merge_units(to_merge,
                                                      merging_mode='hard',
                                                      **job_kwargs)

    to_merge = sic.compute_merge_unit_groups(
        analyzer_merged,
        steps=["unit_locations", "template_similarity"],
        merging_mode='hard',
        **job_kwargs,
    )
    if to_merge != []:
        analyzer_merged = analyzer_merged.merge_units(to_merge,
                                                      merging_mode='hard',
                                                      **job_kwargs)

    bombcell_default_thresholds = sic.bombcell_get_default_thresholds()
    bombcell_labels = sic.bombcell_label_units(analyzer_merged, thresholds=bombcell_default_thresholds,
                                               label_non_somatic=True, split_non_somatic_good_mua=True)

    analyzer_merged.sorting.set_property(
        'bombcell_label',
        bombcell_labels.loc[analyzer_merged.unit_ids, 'bombcell_label'].values
    )
    # keep all labels that are not 'noise'
    keep_unit_ids = bombcell_labels[bombcell_labels['bombcell_label'] != 'noise'].index.tolist()
    analyzer_clean = analyzer_merged.select_units(keep_unit_ids)
    #


    analyzer_path = folders.sorting_path('analyzer_clean')
    import shutil
    if analyzer_path.exists():
        shutil.rmtree(analyzer_path)
    analyzer_clean.save_as(folder=analyzer_path, format='binary_folder')
    # analyzer_clean = si.load_sorting_analyzer(folder=analyzer_path)

    # export to pynapple
    from spikeinterface.exporters import to_pynapple_tsgroup

    my_tsgroup = to_pynapple_tsgroup(analyzer_clean,
                                     attach_unit_metadata=True)
    pynapple_folder = folders.pynapple_folder
    pynapple_folder.mkdir(parents=True, exist_ok=True)
    my_tsgroup.save(pynapple_folder / 'spikes.npz')

    si.export_report(sorting_analyzer=analyzer_clean, output_folder=folders.sorting_path('sorting_summary_clean'),
                     remove_if_exists=True)
    # plot_sorting_summary(sorting_analyzer=analyzer_clean, curation=True, backend='spikeinterface_gui')
