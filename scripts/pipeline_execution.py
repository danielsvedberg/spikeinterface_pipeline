from scripts import si_pipeline as sip
from pathlib import Path

pathlist = [#Path(r"C:\experiment_data\recording_files\DS37\DS37_031926"),
            #Path(r"C:\experiment_data\recording_files\DS37\DS37_032126"),
            #Path(r"C:\experiment_data\recording_files\DS37\DS37_031726"),
            #Path(r"C:\experiment_data\recording_files\DS37\DS37_031826"),#
            #Path(r"C:\Users\assad\Documents\recording_files\DS15\DS15_20260702"),
            #Path(r"C:\Users\assad\Documents\recording_files\DS15\DS15_20260707"),
            #Path(r"C:\Users\assad\Documents\recording_files\DS15\DS15_20260708")]
            #Path(r"C:\Users\assad\Desktop\experiment_data\recording_files\DS45\DS45_20260811"),
            #Path(r"C:\Users\assad\Desktop\experiment_data\recording_files\DS45\DS45_20260812")]
            #Path(r"C:\Users\assad\Desktop\experiment_data\recording_files\DS45\DS45_20260814")]
            #Path(r"C:\Users\assad\Desktop\experiment_data\recording_files\DS45\DS45_20260819")]
            #Path(r"C:\Users\assad\Desktop\experiment_data\recording_files\DS48\DS48_20260921"),
            #Path(r"C:\Users\assad\Desktop\experiment_data\recording_files\DS48\DS48_20260922"),
            #Path(r"C:\Users\assad\Desktop\experiment_data\recording_files\DS48\DS48_20260923"),
            #Path(r"C:\Users\assad\Desktop\experiment_data\recording_files\DS48\DS48_20260924"),
            #Path(r"C:\experiment_data\recording_files\DS23\DS23_20260208"), #triggered error after qc
            #Path(r"C:\experiment_data\recording_files\DS23\DS23_20260209"), #triggered error after qc
            #Path(r"C:\experiment_data\recording_files\DS45\DS45_20260811"),
            #Path(r"C:\experiment_data\recording_files\DS45\DS45_20260812"),
            #Path(r"C:\experiment_data\recording_files\DS45\DS45_20260813"),
            Path(r"C:\experiment_data\recording_files\DS45\DS45_20260814"),
            Path(r"C:\experiment_data\recording_files\DS45\DS45_20260819"),
            Path(r"C:\experiment_data\recording_files\DS45\DS45_20260820"),
            Path(r"C:\experiment_data\recording_files\DS45\DS45_20260821"),
            Path(r"C:\experiment_data\recording_files\DS23\DS23_20260210"),
            Path(r"C:\experiment_data\recording_files\DS23\DS23_20260211"),
]
#[Path(r"C:\Users\assad\Documents\recording_files\DS2\DS2_20260511")]

#base_folder = pathlist[0]

#sip.plot_time_to_first_spike(base_folder)


bad_chans = []#44, 142, 154, 181, 198, 199, 228, 383]  # add 11?

if __name__ == "__main__":  # required on Windows: workers re-import this module
    for folder in pathlist:
        sip.run_pipeline(folder, bad_chans=bad_chans,
                         ow_pre=True, ow_ks=True, ow_post=True)