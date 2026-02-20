import pynapple as nap
from pathlib import Path
import parse_nidq as pni
import plot_psth as pp
import parse_opto_tagging as pot
#import get_ramp_modulation as grm

def run_analysis(base_folder, dinmap=None, ainmap=None, optoidx=None, overwrite=False):
        pynapple_folder = base_folder / "pynapple"
        bs = pni.get_binary_signals(base_folder, dinmap=dinmap, ainmap=ainmap, optoidx=optoidx, overwrite=overwrite, plot=True)
        ephys_file = pynapple_folder / "spikes.npz"
        ephys_data = nap.load_file(ephys_file)

        pot.plot_all_tuning_curves(ephys_data, bs, pynapple_folder, overwrite=overwrite)
        pp.plot_on_events_psth(ephys_data, bs, pynapple_folder)
        #grm.get_ramp_modulation(base_folder)

# pathlist = [#Path(r"C:\Users\assad\Documents\recording_files\DS21\DS21_20251209"),
#             #Path(r"C:\Users\assad\Documents\recording_files\DS21\DS21_20251210"),#]
#             Path(r"C:\Users\assad\Documents\recording_files\DS21\DS21_20251211"),
#             Path(r"C:\Users\assad\Documents\recording_files\DS21\DS21_20251212"),
#             Path(r"C:\Users\assad\Documents\recording_files\DS21\DS21_20251213")]
pathlist = [Path(r"C:\Users\assad\Documents\recording_files\DS23\DS23_20260211")]
for base_folder in pathlist:
    run_analysis(base_folder, overwrite=True)

base_folder = Path(r"C:\Users\assad\Documents\analysis_files\DS13\DS13_20250905")
dinmap = {
    'licks': 0,
    'timing_pulse': 1,
    'solenoid': 2
}
ainmap = {
    'photometry': 0,
    'accelerometer_x': 1,
    'accelerometer_y': 2,
    'accelerometer_z': 3,
    'tone': 5,
    'chrimson': 6,
    'chr2': 7
}

intensities = [20,40,60,80,100]
optoidx = {
    'chrimson': intensities,
    'chr2': intensities
}

#get correlations

base_folder = Path("C:\\Users\\assad\\Documents\\analysis_files\\DS13\\DS13_20250822")


dinmap = {
    'licks': 0,
    'timing_pulse': 1,
}

ainmap = {
    'photometry': 0,
    'accelerometer_x': 1,
    'accelerometer_y': 2,
    'accelerometer_z': 3,
    'solenoid': 4,
    'tone': 5,
    'chrimson': 6,
    'chr2': 7
}

intensities = [10,20,40,100,200,400]
optoidx = {
    'chrimson': intensities,
    'chr2': intensities
}


base_folder = Path(r"C:\Users\assad\Documents\analysis_files\DS13\DS13_20250824")

dinmap = {
    'licks': 0,
    'timing_pulse': 1
}

ainmap = {
    'photometry': 0,
    'accelerometer_x': 1,
    'accelerometer_y': 2,
    'accelerometer_z': 3,
    'solenoid': 4,
    'tone': 5,
    'chrimson': 6,
    'chr2': 7
}

intensities = [10,20,40,100,200,400]
optoidx = {
    'chrimson': intensities,
    'chr2': intensities
}

run_analysis(base_folder, dinmap, ainmap, optoidx, overwrite=True)


base_folder = Path(r"C:\Users\assad\Documents\recording_files\DS1\DS1_20251117")
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

intensities = [10,20,40,100,200,400]
optoidx = {
    'chrimson': intensities,
    'chr2': intensities
}

run_analysis(base_folder, dinmap, ainmap, optoidx, overwrite=False)