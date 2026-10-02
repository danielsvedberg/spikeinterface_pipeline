import spikeinterface as si
import pynapple as nap
from pathlib import Path
import matplotlib.pyplot as plt
from analysis import pres_plot_utils as pu
from utils import file_structure as fs


def load_session(folders):
    """(ephys, binary_signals, trial_events) of one session, its units cut to the lick-bout
    modulated ones -- what every compendium plot is drawn from."""
    pynapple_folder = folders.pynapple_folder
    ep = nap.load_file(pynapple_folder / "sing_tagged_units.npz")
    bs = nap.load_file(folders.binary_signals_file())
    bs = pu.add_all_licks(bs)

    ch_on, ch_off = pu.pool_task_stim(bs, 'chrimson')
    c2_on, c2_off = pu.pool_task_stim(bs, 'chr2')
    pu.compute_ramp_scores(ep, bs, ch_on, c2_on)

    te = pu.categorize_outcome_trials(bs, ch_on, ch_off, c2_on, c2_off)

    lick_mod = pu.id_lick_bout_modulated(ep, bs, te)
    sig_units = lick_mod.loc[lick_mod['significant'], 'unit_id'].tolist()
    ep = ep[sig_units]
    print(f"{folders.rec_ID.name}: {len(sig_units)} / {len(lick_mod)} lick-bout modulated units")
    return ep, bs, te


def load_sessions(ana_folders):
    return [load_session(folders) for folders in ana_folders]


def plot_compendium(sessions, ana_folders, comp_folder):
    """Every compendium plot of the given sessions, merged into one pseudo-session, into comp_folder.

    One session is merged the same way as many (merge_sessions drops its lickless trials), so a
    single session's plots show exactly what it contributes to the pooled compendium.
    """
    comp_folder = Path(comp_folder)
    comp_folder.mkdir(parents=True, exist_ok=True)

    # --- merge all sessions into one pseudo-session ---
    ephys, binary_signals, trial_events = pu.merge_sessions(sessions)
    print(f"Total merged units: {len(ephys.keys())}")

    # --- validation steps: unit yield and firing rate per category, per recording.
    # Both work off the pre-merge sessions list so each unit's rate uses its own
    # session's duration rather than the merged pseudo-session's.
    pu.plot_units_per_category(
        sessions, ana_folders,
        save_path=comp_folder / 'units_per_category'
    )

    pu.plot_firing_rate_by_category(
        sessions, ana_folders,
        save_path=comp_folder / 'firing_rate_by_category'
    )

    # --- plots ---
    pu.plot_spn_lick_bout_psth(ephys, binary_signals, trial_events, comp_folder /
                                "lick_bout_psth")

    pu.plot_spn_summary(
        ephys, binary_signals, trial_events,
        save_path=comp_folder / 'spn_summary.png',
        ms_censored=500,
        n_quintiles=7,
    )

    pu.plot_outcome_heatmap_nostim(
        ephys, binary_signals, trial_events,
        save_path=comp_folder / 'first_lick_heatmap_nostim.png',
        clim=(None,None)
    )

    pu.plot_cue_outcome_heatmap(
        ephys, binary_signals, trial_events,
        save_path=comp_folder / 'cue_outcome_heatmap.png',
        ms_censored=50, clim=(-0.25, 4),
    )

    pu.plot_cue_outcome_heatmap(
        ephys, binary_signals, trial_events,
        save_path=comp_folder / 'cue_outcome_heatmap.png',
        ms_censored=50, clim=(0, 0.1),
        zscore=False
    )

    pu.plot_spn_heatmap(
        ephys, binary_signals, trial_events,
        save_path=comp_folder / 'spn_heatmap.png',
        ms_censored=500, clim=(-0.25, 2.5)
    )

    pu.plot_outcome_heatmap_nostim(
        ephys, binary_signals, trial_events,
        save_path=comp_folder / 'first_lick_heatmap_nostim.png',
        clim=(0, None), zscore=False,
    )

    pu.plot_cue_aligned_nostim(
        ephys, binary_signals, trial_events, n_quintiles=10,
            save_path=comp_folder / 'cue_aligned_nostim.png', ms_censored=50
    )

    pu.plot_spn_cue_aligned_nostim(
        ephys, binary_signals, trial_events, n_quintiles=6,
        save_path=comp_folder / 'spn_cue_aligned_nostim.png', ms_censored=1000
    )

    pu.plot_spn_mean_psth(
        ephys, binary_signals, trial_events,
        save_path=comp_folder / 'spn_mean_psth_rewarded.png', ms_censored=3333
    )

    pu.plot_spn_mean_psth(
        ephys, binary_signals, trial_events,
        save_path=comp_folder / 'spn_mean_psth_unrewarded.png', outcome_range = [500,3332]
    )

    pu.plot_spn_rew_unrew_psth(
        ephys, binary_signals, trial_events,
        save_path=comp_folder / 'spn_rew_unrew_psth.png',
        rewarded_range_ms=(3333, None),
        unrewarded_range_ms=(100, 3332),
    )



    pu.plot_dspn_isPN_diff(
        ephys, binary_signals, trial_events,
        save_path=comp_folder / 'dspn_ispn_diff.png',
        n_quintiles=10, ms_censored=1000
    )

    fig, summary = pu.plot_dspn_ispn_lick_bout_bar(
        ephys, binary_signals, trial_events,
        save_path=comp_folder / "dspn_ispn_lick_bout_bar"
    )

    pu.plot_dspn_isPN_bar(
        ephys, binary_signals, trial_events,
        save_path=comp_folder / 'dspn_ispn_bar.svg',
        ms_censored=500,
    )


if __name__ == "__main__":
    #subj_folder = Path(r"C:\Users\assad\Desktop\experiment_data\analysis_files\DS45")
    #this cross-session compendium is an analysis product, so it is saved under the subject's
    #analysis_files folder rather than next to the raw recordings
    #save_folder = fs.category_path(subj_folder, fs.ANALYSIS_ROOT) / "compendium"
    #ana_folder = Path(r"C:\Users\assad\Desktop\experiment_data\analysis_files")
    ana_folder = Path(r"C:\Users\assad\Desktop\experiment_data\analysis_files")
    comp_folder = ana_folder / "compendium"

    ana_folders = [
        #Path(r"C:\Users\assad\Desktop\experiment_data\analysis_files\DS15\DS15_20260701"),
        Path(r"C:\Users\assad\Desktop\experiment_data\analysis_files\DS15\DS15_20260702"),
        Path(r"C:\Users\assad\Desktop\experiment_data\analysis_files\DS15\DS15_20260703"),
        Path(r"C:\Users\assad\Desktop\experiment_data\analysis_files\DS45\DS45_20260811"),
        Path(r"C:\Users\assad\Desktop\experiment_data\analysis_files\DS45\DS45_20260812"),
        Path(r"C:\Users\assad\Desktop\experiment_data\analysis_files\DS45\DS45_20260814"),
        Path(r"C:\Users\assad\Desktop\experiment_data\analysis_files\DS2\DS2_20260505"),
        Path(r"C:\Users\assad\Desktop\experiment_data\analysis_files\DS2\DS2_20260506"),
        Path(r"C:\Users\assad\Desktop\experiment_data\analysis_files\DS2\DS2_20260511")


    ]
    ana_folders = [fs.SessionPaths(f) for f in ana_folders]
    # --- per-session processing ---
    sessions = load_sessions(ana_folders)
    plot_compendium(sessions, ana_folders, comp_folder)

    plt.show()
