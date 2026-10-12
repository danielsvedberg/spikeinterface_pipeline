import matplotlib.pyplot as plt
import spikeinterface.full as si
from pathlib import Path
import matplotlib
matplotlib.use('Qt5Agg')
#make a dataframe from temp_metrics
import pandas as pd
import numpy as np
import spikeinterface.curation as sic
from postprocessing import metrics_curation as mc
from postprocessing import spike_cleaning as scl
from preprocessing import get_stimulation_frames as gsf
from preprocessing import preprocess_qc as pqc
from preprocessing import detect_saturation as dsat
import shutil
from spikeinterface.widgets import plot_sorting_summary
from preprocessing import parse_nidq as pni
import pynapple as nap
from postprocessing import parse_opto_tagging as pot
from analysis import plot_psth as pp
from utils import file_structure as fs

def test():
    base_folder = Path(r"C:\experiment_data\recording_files\DS45\DS45_20260811")
    nidaq_map=None
    bad_chans=None
    export_raw_summary=True
    ow_bs=True
    ow_pre=True
    ow_ks=True
    ow_post=True
    run_qc=True
    ow_sat=True
    full_saturation_scan=True
    run_pipeline(base_folder, nidaq_map=nidaq_map, bad_chans=bad_chans, export_raw_summary=export_raw_summary,
                 ow_bs=ow_bs, ow_pre=ow_pre, ow_ks=ow_ks, ow_post=ow_post, run_qc=run_qc,
                 full_saturation_scan=full_saturation_scan, ow_sat=ow_sat)


def run_pipeline(base_folder, nidaq_map=None, bad_chans=None, export_raw_summary=False, ow_bs=False,
                 ow_pre=False, ow_ks=False, ow_post=False, run_qc=True,
                 full_saturation_scan=True, ow_sat=False, stop_after_qc=False):
    #base folder needs to be a Path object
    #sorting/analysis products go to their own category folders, raw data is only read from
    #the recording folder (see utils/file_structure.py)
    global_job_kwargs = dict(n_jobs=-1, chunk_duration="10s", progress_bar=True)
    si.set_global_job_kwargs(**global_job_kwargs)

    folders = fs.SessionPaths(base_folder).create_output_folders()
    # Usually, you would read in your raw recording
    spikeglx_folder = list(folders.recording_folder.glob('*_g*'))[0]

    on_event_times, off_event_times = gsf.get_stimulation_times(base_folder, nidaq_map, overwrite=ow_bs)
    #concatenate into one list
    artifact_times = on_event_times + off_event_times
    artifact_times.sort()

    stream_names, stream_ids = si.get_neo_streams('spikeglx', spikeglx_folder)
    raw_rec = si.read_spikeglx(spikeglx_folder, stream_id='imec0.ap')
    raw_rec.get_probe().to_dataframe()
    sf = raw_rec.get_sampling_frequency()
    artifact_idxs = [int(t * sf) for t in artifact_times]


    my_protocol = {
        'sorting': {
            'sorter_name': 'kilosort4',
            'verbose': True,
            'folder': folders.sorting_path(fs.KS_FOLDER),
            'remove_existing_folder': True,
            'do_CAR': False,
            #Kilosort always builds a highpass filter -- there is no off switch short of
            #skip_kilosort_preprocessing, which also disables whitening. Setting the
            #cutoff to 1 Hz makes it all-pass instead (|H| = 1.00000 at 300 Hz and
            #above, impulse response within 1e-4 of a delta), so the 300 Hz bandpass
            #above is the only highpass in the chain. At the 300 Hz default the two
            #filters stack and cost ~6 dB extra right at the bottom of the passband.
            #0 is not usable despite the parameter docs claiming min=0: scipy rejects it.
            'highpass_cutoff': 1.0,
            'progress_bar': True,
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


    #preprocess_folder is None until this session has been preprocessed; it is also None-ed here
    #if what is on disk cannot be loaded, so the two reasons to redo the work share one test

    if not ow_pre or folders.preprocess_folder is None:
        try:
            preprocessed_rec = si.load(folders.preprocess_folder)
        except Exception:
            ow_pre = True

    if ow_pre:
        #Saturation is a property of the raw traces, so it is measured before the chain is
        #built. The clipping level is per-session and sits below the ADC full scale the
        #meta advertises -- two NP2.0 sessions off the same probe model clipped at 1947
        #and 1963 counts against an imMaxInt of 2048 -- so it cannot be taken from the
        #dtype or the meta, and because clipped samples pile up at a single value a
        #threshold even 1% high catches none of them. The scan answers two questions at
        #once: which channels are railed continuously (channels to drop) and which spans
        #rail across the array (spans to blank). It reads the whole file, so it is cached.
        #beside preprocess, not in it: the save below deletes preprocess first, so a
        #cache kept there never survived to be reused. A cache from before the move is
        #carried over once, before that save can delete it.
        sat_folder = folders.recording_folder / fs.SATURATION_FOLDER
        old_cache = folders.recording_folder / fs.PREPROCESS_FOLDER / 'qc' / 'saturation.npz'
        if not (sat_folder / 'saturation.npz').exists() and old_cache.exists():
            sat_folder.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old_cache, sat_folder / 'saturation.npz')
        sat = dsat.detect_saturation_cached(raw_rec, sat_folder, overwrite=ow_sat)
        #written whether or not clipping was found, and independently of run_qc: a
        #raw-data report that only appears for bad sessions cannot be distinguished from
        #one that never ran. Emits the text summary, a JSON, a per-channel CSV and the
        #log-scale voltage histogram.
        for f in dsat.write_saturation_report(raw_rec, sat, sat_folder):
            print(f'  saturation report: {f.name}')
        print(sat['summary'])

        my_protocol['preprocessing'], bad_chids = build_preprocessing_protocol(
            raw_rec, artifact_idxs, sat, bad_chans)

        #the chain is built in two stages rather than one call so the QC can hold a handle
        #on the recording as it was immediately before common_reference. Applying the two
        #halves in sequence is identical to applying the whole dict at once
        pre_car_steps, car_steps = pqc.split_protocol(my_protocol['preprocessing'],
                                                      'common_reference')
        pre_car_rec = si.apply_preprocessing_pipeline(raw_rec, pre_car_steps)
        preprocessed_rec = si.apply_preprocessing_pipeline(pre_car_rec, car_steps)

        #A misconfigured bad-channel detector fails silently and expensively: one run of
        #this pipeline saved a 33-channel recording out of 384 and sorted it, with
        #nothing in the log to say so. Losing a fifth of the array is not something a
        #healthy probe does, so say it loudly before spending an hour writing and sorting
        #the result. A warning rather than a raise: on a genuinely bad probe this is the
        #correct outcome and the run should continue.
        n_in, n_out = raw_rec.get_num_channels(), preprocessed_rec.get_num_channels()
        if n_out < 0.8 * n_in:
            print(f'*** WARNING: preprocessing kept {n_out}/{n_in} channels '
                  f'({100 * (n_in - n_out) / n_in:.0f}% removed). Check the '
                  "detect_and_remove_bad_channels method against the saturation report's "
                  'dead-channel list before trusting the sort. ***')
        else:
            print(f'channels: {n_in} -> {n_out}')
        #the destination is named raw rather than through folders, since the accessor answers None
        #for exactly the folder we are about to write (and .save refuses an existing one anyway)
        #
        #The saved recording replaces the lazy chain from here on, which matters for two
        #reasons beyond tidiness. run_sorter serialises whatever it is handed to JSON and
        #rebuilds it inside the sorter process; a silence_periods step does not survive
        #that round trip (its structured `periods` array lands in JSON as a plain nested
        #list, and the constructor rejects a list), so sorting the lazy chain fails with
        #"periods must be a np.array with dtype ...". And a lazy chain is not binary
        #compatible, so run_sorter re-materialises the whole thing to a second temporary
        #binary -- an extra full-length write on top of the one just done here. A
        #BinaryFolderRecording round-trips trivially and Kilosort reads it in place.
        preprocessed_rec = preprocessed_rec.save(
            folder=folders.recording_folder / fs.PREPROCESS_FOLDER,
            format='binary', overwrite=True, **global_job_kwargs)
        pm = si.plot_probe_map(preprocessed_rec)
        pltpth = folders.preprocess_folder / 'probe_map.png'
        plt.savefig(pltpth)
        plt.close()
        print('probe map saved')

        if run_qc:
            #validation of the chain itself: what the data looked like going in, and what
            #preprocessing did to it. The two stages built above are handed in rather than
            #rebuilt, so the QC measures the same channel set that was just saved --
            #detect_and_remove_bad_channels runs its own random-chunk detection when
            #bad_channel_ids is None, so a rebuilt chain can drop different channels.
            #The saturation check reads the whole raw file rather than sampled chunks,
            #since opto-driven amplifier saturation is milliseconds long and sampling
            #misses it; run_pipeline(full_saturation_scan=False) trades that for speed
            qc_folder = folders.preprocess_folder / 'qc'
            #licking is a jaw movement against a headpost, so it has every opportunity to
            #put a movement artifact in the traces, and one that survives the chain is
            #something the sorter will try to explain as spikes. The event-triggered
            #average before and after preprocessing is the direct look at that. The nidq
            #parse is already cached by now, so this reads it rather than redoing it.
            event_triggers = {name: (t * sf).astype(int) for name, t in
                              gsf.get_event_times(base_folder, nidaq_map,
                                                  events=('lick_on',)).items()
                              if t.size}
            raw_qc, post_qc = pqc.run_preprocessing_qc(
                raw_rec, my_protocol['preprocessing'], qc_folder,
                pre_car_rec=pre_car_rec, final_rec=preprocessed_rec,
                manual_bad_channel_ids=bad_chids,
                full_saturation_scan=full_saturation_scan,
                event_triggers=event_triggers,
                #the rail was measured above; handing it over saves the QC re-measuring
                #it, and guarantees both reports describe the same threshold
                rail=sat['rail'],
            )
            print((qc_folder / 'preprocessing_qc_summary.txt').read_text())



    #A preprocessing-only rerun has to return here rather than rely on the ow_ flags.
    #Everything below has side effects that do not ask permission: if the sort or its
    #analyzer fails to load, ow_ks flips to True and run_sorter(remove_existing_folder=True)
    #deletes and redoes the sort, and the exports and plots always overwrite theirs.
    if stop_after_qc:
        return preprocessed_rec

    #A sort is made on one preprocessed recording; redoing the preprocessing redoes it,
    #rather than leaving a sort (and analyzer, and clean analyzer) that index a binary
    #that has since been overwritten. For a QC-only rerun use stop_after_qc.
    if ow_pre and folders.ks_folder is not None and not ow_ks:
        print('preprocessing was redone, so the sort made on the old recording is redone too')
        ow_ks = True

    #The sort and its analyzer are loaded separately, and nothing here falls through to
    #run_sorter by accident: run_sorter's remove_existing_folder=True deletes the sort
    #first, so only ow_ks=True or a session that was never sorted gets a new one. A sort
    #that fails to load raises. It is read without its own recording JSON, whose
    #absolute path a moved session no longer has, and given the recording loaded above.
    sorting, analyzer = None, None
    if not ow_ks and folders.ks_folder is not None:
        sorting = si.read_sorter_folder(folders.ks_folder, register_recording=False)
        sorting.register_recording(preprocessed_rec)
        try:
            analyzer = si.load_sorting_analyzer(folder=folders.sorting_path('analyzer'))
        except Exception as e:
            print(f'analyzer did not load ({e!r}); rebuilding it from the existing sort')
        else:
            if not analyzer.has_recording():
                analyzer.set_temporary_recording(preprocessed_rec)
            #an analyzer of another sort -- left by an interrupted run, or built on the
            #old deduplicated sort -- does not describe this one, and is rebuilt
            a, b = analyzer.sorting.to_spike_vector(), sorting.to_spike_vector()
            if a.size != b.size or not (np.array_equal(a['sample_index'], b['sample_index'])
                                        and np.array_equal(a['unit_index'], b['unit_index'])):
                print(f'the analyzer ({a.size} spikes) is not of this sort ({b.size} spikes); '
                      'rebuilding it')
                analyzer = None
            else:
                #an interrupted compute leaves the analyzer loadable with a tail of
                #extensions missing (a run that did not complete is not loaded); the
                #cleaning needs them all, so the missing ones are computed now
                missing = {k: v for k, v in my_protocol['postprocessing'].items()
                           if not analyzer.has_extension(k)}
                if missing:
                    print(f'analyzer is missing {list(missing)}; computing them')
                    analyzer.compute(missing, **global_job_kwargs)
                    stale = folders.sorting_path('analyzer_clean')
                    if stale.exists():
                        shutil.rmtree(stale)
                    ow_post = True
    new_sort = ow_ks or sorting is None
    if new_sort or analyzer is None:
        #whatever was derived from the sort or analyzer being replaced goes now, so an
        #interrupted run cannot leave it to be exported against the new one
        stale = folders.sorting_path('analyzer_clean')
        if stale.exists():
            shutil.rmtree(stale)
    if new_sort:
        sorting = si.run_sorter(recording=preprocessed_rec, **my_protocol['sorting'], n_jobs=-1)
    if analyzer is None:
        #The analyzer is built on the sort as it came, with no remove_duplicated_spikes.
        #That step resolves every pair of spikes closer than its censor period by order,
        #keeping the first whatever either looks like; spike_cleaning below resolves the
        #same pairs (and a wider window) by which spike fits its unit worse instead.
        analyzer = si.create_sorting_analyzer(recording=preprocessed_rec, sorting=sorting, folder=folders.sorting_path('analyzer'), format='binary_folder', overwrite=True, **global_job_kwargs)
        analyzer.compute(my_protocol['postprocessing'], **global_job_kwargs)
        #an analyzer that is new, or rebuilt, has no clean analyzer that describes it
        ow_post = True

    if export_raw_summary:
        si.export_report(sorting_analyzer=analyzer, output_folder=folders.sorting_path('sorting_summary_raw'), remove_if_exists=True)

    analyzer_path = folders.sorting_path('analyzer_clean')
    if not analyzer_path.exists():
        ow_post = True
    if ow_post:
        #Outlier spikes, then close pairs resolved by fit, then merges, then the merged
        #units' new close pairs, then censored refractory metrics and bombcell -- see
        #postprocessing/spike_cleaning.py. The stimulation intervals keep light-evoked
        #spikes, where co-firing neighbours merge into a unit, out of each unit's core fit.
        stim_intervals = gsf.get_stimulation_intervals(base_folder, nidaq_map)
        analyzer_clean, _, report = scl.clean_and_curate(
            analyzer, stim_intervals=stim_intervals,
            audit_folder=folders.sorting_path('spike_cleaning'), **global_job_kwargs)
        print(f'spike cleaning: {report}')
        #the audit folder also gets a figure per unit of the spikes the cleaning removed
        #(removed_spike_plots/) and one per bimodal split (bimodal_split_plots/)
        #the old analyzer_clean is removed only now, with its replacement in hand
        if analyzer_path.exists():
            shutil.rmtree(analyzer_path)
        analyzer_clean = analyzer_clean.save_as(folder=analyzer_path, format='binary_folder')
    else:
        analyzer_clean = si.load_sorting_analyzer(folder=analyzer_path)

    #export to pynapple
    from spikeinterface.exporters import to_pynapple_tsgroup
    my_tsgroup = to_pynapple_tsgroup(analyzer_clean,
        attach_unit_metadata=True)
    pynapple_folder = folders.pynapple_folder
    pynapple_folder.mkdir(parents=True, exist_ok=True)
    my_tsgroup.save(pynapple_folder / 'spikes.npz')

    si.export_report(sorting_analyzer=analyzer_clean, output_folder=folders.sorting_path('sorting_summary_clean'),
                     remove_if_exists=True)
    #plot_sorting_summary(sorting_analyzer=analyzer_clean, curation=True, backend='spikeinterface_gui')

    plot_opto_psth(base_folder)
    plot_psth(base_folder)
    plot_time_to_first_spike(base_folder)
    plot_tuning_curves(base_folder)


def build_preprocessing_protocol(raw_rec, artifact_idxs, sat, bad_chans=None):
    """The preprocessing chain for one session, as a spikeinterface protocol dict.

    Built from the raw recording, the opto trigger frames and the saturation scan, so
    run_pipeline and anything that wants to test a variant of the chain (a reordering,
    a step dropped) start from the same dict. Returns (protocol, bad_chids).
    """
    chids = raw_rec.channel_ids
    #Dead channels are railed regardless of what the session is doing, so they belong
    #with the manually flagged ones. Supplying bad_channel_ids makes
    #detect_and_remove_bad_channels skip its own detection, so this list has to be
    #complete rather than additive -- 'method' below is inert whenever it is non-empty.
    manual = np.asarray(bad_chans, dtype=int) if bad_chans else np.array([], dtype=int)
    all_bad = np.union1d(manual, np.asarray(sat['dead_channel_inds'], dtype=int))
    bad_chids = chids[all_bad] if all_bad.size else None
    if all_bad.size:
        print(f'bad channels: {len(manual)} manual + '
              f'{len(sat["dead_channel_inds"])} railed -> {chids[all_bad].tolist()}')

    protocol = {
        #remove_artifacts goes first, ahead of every filter, and has to stay there. Run
        #first it hands the bandpass an already-interpolated trace, so the filter has no
        #transient to ring on; run anywhere after the bandpass it can only overwrite
        #samples inside its own window, while the ringing the filter has already spread
        #beyond it survives untouched. Measured on DS45_20260811 by the paired
        #injection test (tests/artifact_removal_test.py, experiment 'order'): first cuts
        #the residual outside the window 2.5-3x, and last is identical to no artifact
        #removal at all outside it.
        'remove_artifacts': {'list_triggers': artifact_idxs, 'ms_before': 0.5,
                             'ms_after': 1.0, 'mode': 'linear'},
        #Line-noise notch, ahead of the bandpass so the notch's long impulse
        #response never sees the opto transients (remove_artifacts has already
        #interpolated over them). Note this is belt-and-braces for the sorting
        #chain: the 300 Hz bandpass below already puts 60 Hz at -144 dB. It earns
        #its place if freq_min is ever lowered, or if the saved preprocessed file
        #gets used for anything below 300 Hz.
        #q=30 (2 Hz wide), not 300: margin_ms scales with q, so q=300 needs 8 s of
        #padding either side of every chunk -- 26 s read per 10 s kept, 2.6x the
        #I/O -- while q=30 needs 0.8 s (1.2x) and is still far narrower than any
        #real line-noise peak.
        #'notch_filter': {'freq': 60, 'q': 30},
        #freq_max is 5000 rather than the 6000 default: these recordings carry an
        #isolated narrowband tone at exactly 6000 Hz on every channel (fs/5, a clock
        #or ADC spur, +13 dB over baseline and not a 60 Hz harmonic -- 5940 and 6060
        #are flat). At the 6000 default it sits on the -3 dB corner, is only half
        #attenuated, survives common_reference and reaches the sorter. Spikes carry
        #little energy above 5 kHz, so moving the corner costs nothing and also
        #attenuates the fs/4 spur at 7500 Hz far harder.
        'bandpass_filter': {'freq_max': 5000},
        'phase_shift': {},
        #silence_periods assumes zero-centred traces, and NP2.0 has no AP/LF split, so
        #the blanking sits after the bandpass. The periods already carry padding for
        #the ring-down filtfilt smears out of each clipped edge in both directions.
        **({'silence_periods': sat['silence_periods_kwargs']}
           if sat['silence_periods_kwargs'] is not None else {}),
        #coherence+psd, not neighborhood_r2. On this probe neighborhood_r2 condemned
        #357 of 384 channels at its defaults, and 382 of 384 on unfiltered traces --
        #so it is not a filtering-regime problem, the method simply does not suit
        #this geometry (a wider radius makes it worse, flagging all 384).
        #coherence+psd is SI's default and the IBL method; it finds 0 bad channels
        #here, agreeing with the saturation scan, which also found none.
        'detect_and_interpolate_bad_channels': {'bad_channel_ids': bad_chids,
                                                'method': 'coherence+psd'},
        #Global CAR: one median across all four shanks, deliberately. Per-shank
        #referencing (pqc.shank_groups(pre_car_rec) -> the `groups` argument) cuts
        #cross-shank residual correlation 4x, but a 96-channel median is a noisier
        #reference than a 384-channel one, so it leaves 3.6x more *within*-shank
        #correlation -- and within-shank is what bears on sorting, since templates
        #span neighbouring channels on one shank and none span shanks. Measured on
        #DS45: within 0.037 global vs 0.133 per-shank, cross 0.063 vs 0.016.
        #Revisit if the shanks ever sit in different structures, where a global
        #median would carry one region's population activity into another's.
        'common_reference': {'operator': 'median'},
    }
    return protocol, bad_chids


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

    pynapple_folder = fs.SessionPaths(base_folder).pynapple_folder
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


def plot_opto_psth(base_folder, overwrite=False, nidq_map_kwargs=None):
    ephys_data, bs, pynapple_folder = get_pynapple_data(base_folder, nidq_map_kwargs, overwrite)
    analyzer = si.load_sorting_analyzer(folder=fs.SessionPaths(base_folder).sorting_path('analyzer_clean'))
    #the analyzer puts the waveform of every spike fired during the pulses under each raster
    pp.plot_all_opto_psth(ephys_data, bs, pynapple_folder, analyzer=analyzer)
    pp.plot_all_opto_psth_alt(ephys_data, bs, pynapple_folder, analyzer=analyzer, minmax=(-0.1, 0.2))

def plot_time_to_first_spike(base_folder, overwrite=False, nidq_map_kwargs=None):
    """Convenience wrapper: load pynapple data and call the TTFS plotting function."""
    ephys_data, bs, pynapple_folder = get_pynapple_data(base_folder, nidq_map_kwargs, overwrite)
    save_folder = pynapple_folder / "ttfs_plots"
    opsins = ['chrimson', 'chr2']
    df, cox_df, figpath = pot.plot_time_to_first_spike_distribution(ephys_data, bs, save_folder=pynapple_folder)
    return df, cox_df, figpath


def test_stim_firing(base_folder, overwrite_bs=False, nidq_map_kwargs=None, **kwargs):
    """Convenience wrapper: load pynapple data and run the stim-firing Mann-Whitney U test."""
    ephys_data, bs, pynapple_folder = get_pynapple_data(base_folder, nidq_map_kwargs, overwrite_bs)
    results_df, summary_df = pot.test_stim_firing(ephys_data, bs, save_folder=pynapple_folder, **kwargs)
    return results_df, summary_df

def test_time_to_first_spike():
    base_folder = Path(r"C:\Users\assad\Documents\recording_files\DS23\DS23_20260211")#modify the paths as needed
    overwrite=False
    nidq_map_kwargs=None
    post_window_ms = 100.0
    min_latency_ms = 0.5
    per_unit = False
    plot_time_to_first_spike(base_folder, overwrite, nidq_map_kwargs)


def launch_sorting_gui(base_folder): #convenience function to launch the sorting GUI on the cleaned analyzer
    folders = fs.SessionPaths(base_folder)
    analyzer_clean = si.load_sorting_analyzer(folder=folders.sorting_path('analyzer_clean'))
    si.export_report(sorting_analyzer=analyzer_clean, output_folder=folders.sorting_path('sorting_summary_clean'), remove_if_exists=True)


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
#pathlist = [Path(r""C:\Users\assad\Documents\recording_files\DS37\DS37_031726\DS37_g0"C:\Users\assad\Documents\recording_files\DS23\DS23_20260211")] #modify the paths as needed

