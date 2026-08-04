import spikeinterface as si
from scripts import si_pipeline as sip
from analysis import plot_psth as pp
from pathlib import Path
from utils import file_structure as fs


#base_folder = Path(r"C:\Users\assad\Documents\recording_files\DS2\DS2_050526")
    #r"C:\Users\assad\Documents\recording_files\DS2\DS2_20260506")
#base_folder = Path(r"C:\Users\assad\Documents\recording_files\DS2\DS2_20260511")
base_folder = Path(r"C:\Users\assad\Documents\recording_files\DS15\DS15_20260702")
folders = fs.SessionPaths(base_folder)
pynapple_folder = folders.pynapple_folder

ephys_data, bin_sigs, _ = sip.get_pynapple_data(base_folder, overwrite=False)
analyzer = si.load_sorting_analyzer(folder=folders.sorting_path('analyzer_clean'))

pp.plot_all_opto_psth_alt(ephys_data, bin_sigs, pynapple_folder,
  analyzer=analyzer, minmax=(-0.1, 0.2))

pp.plot_all_opto_psth(ephys_data, bin_sigs, pynapple_folder, analyzer=analyzer)
