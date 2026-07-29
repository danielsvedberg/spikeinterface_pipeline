"""
file_moving.py will take a (recording) rec_folder, which contains raw recording files, preprocessed signals files, spike-sorting
files, and post-hoc analysis that all correspond to a single recording session--spread across various folders, within
the rec_folder--and moves them to separate folders in 3 categories: 1) (raw) recording files, which will contain
just the raw recorded data, 2) sorting_files, which will contain files related to the spike sorting pipeline, including
pre-processed data, extracted binary signals, the kilosort spike sorting output, and files related to curation, and 3)
analysis files, which will contain the pynapple files that are the product of the final curation, posthoc analysis plots,
and a copy of the extracted binary signals. Raw recording files include any file under ANID_g# folders as well as
the ANID_YYYYMMDD_Timestamp.csv, ...FrameID.csv, .avi, .mat, and .txt files. Sorting files include any analyzer_*,
kilosort4_output, preprocess, and sorting_summary_* folders, as well as the binary_signals.npz file in the pynapple folder
Analysis files include all original contents of the pynapple folder. A significant catch is that before files get moved
around, the binary_signals.npz file should be moved to a new subfolder of rec_folder called "binary_signals" so that
the binary signals can be easily copied to both the sorting and anlaysis folders.
The pynapple folder will be moved wholesale to the analysis folder, and the binary_signals.npz file will be copied to
the sorting folder. The pynapple folder will no longer contain the binary_signals.npz file, but will still contain all
other files that were originally in the pynapple folder.
"""

from pathlib import Path
import os
import shutil

rec_folder = Path(r"C:\Users\assad\Documents\recording_files\DS15\DS15_20260701")

"""
Example original filestructure (rec_folder = recording_files/DS15/DS15_20260701):

recording_files/
└── DS15/                                     <- animal folder (an_ID)
    └── DS15_20260701/                        <- rec_folder (this recording session)
        ├── DS15_20260701_g0/                 <- raw recording (ANID_g# folder)
        │   └── ...                               raw acquisition files
        ├── DS15_20260701_142530.csv          <- ANID_YYYYMMDD_Timestamp.csv
        ├── DS15_20260701_142530_FrameID.csv
        ├── DS15_20260701.avi
        ├── DS15_20260701.mat
        ├── DS15_20260701.txt
        ├── preprocess/                        <- sorting: preprocessed data
        ├── kilosort4_output/                  <- sorting: kilosort spike sorting output
        ├── analyzer_DS15_20260701/            <- sorting: sorting analyzer (analyzer_*)
        ├── sorting_summary_DS15_20260701/     <- sorting: curation summary (sorting_summary_*)
        └── pynapple/                          <- analysis: final curation products
            ├── binary_signals.npz            <- extracted binary signals (needed by both sorting & analysis)
            └── ...                               other pynapple analysis files

Example target filestructure (root_folder holds the 3 category folders, each mirroring an_ID/rec_ID):

root_folder/
├── recording_files/                          <- raw recorded data only
│   └── DS15/
│       └── DS15_20260701/
│           ├── DS15_20260701_g0/             <- ANID_g# raw acquisition folder
│           │   └── ...
│           ├── DS15_20260701_142530.csv      <- ANID_YYYYMMDD_Timestamp.csv
│           ├── DS15_20260701_142530_FrameID.csv
│           ├── DS15_20260701.avi
│           ├── DS15_20260701.mat
│           └── DS15_20260701.txt
├── sorting_files/                            <- spike-sorting pipeline products
│   └── DS15/
│       └── DS15_20260701/
│           ├── preprocess/
│           ├── kilosort4_output/
│           ├── analyzer_DS15_20260701/
│           ├── sorting_summary_DS15_20260701/
│           └── binary_signals/               <- copy of the binary_signals folder
│               └── binary_signals.npz
└── analysis_files/                           <- final curation + posthoc analysis
    └── DS15/
        └── DS15_20260701/
            ├── pynapple/                      <- moved wholesale (minus binary_signals.npz)
            │   └── ...                            other original pynapple files
            └── binary_signals/               <- copy of the binary_signals folder
                └── binary_signals.npz
"""



# Part 1: declare all folders in scope of the program, and relevant file parts
# 1.1: get all parts of the rec_folder path that precede 'recording_files' and assign to variable root_folder
rec_folder_parts = rec_folder.parts
recording_files_index = rec_folder_parts.index("recording_files")
root_folder = Path(*rec_folder_parts[:recording_files_index])

# 1.2: derive the recording's identity within recording_files so the sorting/analysis
#      category folders can mirror the same animal/session nesting (an_ID/rec_ID)
rf_folder = Path(*rec_folder_parts[:recording_files_index + 1])  # "rf": recording_files root
rec_ID = rec_folder.relative_to(rf_folder)                       # e.g. DS15/DS15_20260701
an_ID = rec_ID.parts[0]                                          # e.g. DS15

# 1.3: category roots. Note recording_files needs no destination of its own: rec_folder IS the
#      raw-recording folder, so the raw files simply stay put once sorting/analysis are moved out
sf_folder = root_folder / "sorting_files"   # "sf": sorting files category root
af_folder = root_folder / "analysis_files"  # "af": analysis files category root

# 1.4: binary signals staging. binary_signals.npz originally lives inside the pynapple (analysis)
#      folder; before anything else it is staged into its own bs_folder so it can be copied to
#      both the sorting and analysis destinations (programs doing both need it)
bs_folder = rec_folder / "binary_signals"  # "bs": staging folder inside rec_folder
pynapple_folder = rec_folder / "pynapple"
bsf_pynapple = pynapple_folder / "binary_signals.npz"  # original location
bsf_staged = bs_folder / "binary_signals.npz"          # staged location inside bs_folder

# 1.5: destination folders, each mirroring rec_ID under its category root
sort_dest = sf_folder / rec_ID   # root/sorting_files/DS15/DS15_20260701
ana_dest = af_folder / rec_ID    # root/analysis_files/DS15/DS15_20260701
sort_bs_dest = sort_dest / "binary_signals"
ana_bs_dest = ana_dest / "binary_signals"

# 1.6: prefixes identifying which of rec_folder's subfolders belong to each category.
#      binary_signals is intentionally excluded here: it is copied to both destinations
#      (handled separately) rather than moved to one
sort_subfolder_prefixes = ["analyzer", "sorting_summary", "kilosort", "preprocess"]
analysis_subfolder_prefixes = ["pynapple"]


# Part 2 / Phase 1: identify the current filestructure of rec_folder
rec_subfolders = [f for f in rec_folder.iterdir() if f.is_dir()]
sort_origin_subfolders = [f for f in rec_subfolders if any(f.name.startswith(p) for p in sort_subfolder_prefixes)]
analysis_origin_subfolders = [f for f in rec_subfolders if any(f.name.startswith(p) for p in analysis_subfolder_prefixes)]
# rec_folder still "contains sorting/analysis files" if any such subfolder is present, or if the
# binary signals have not yet been distributed (still in pynapple, or only staged in bs_folder)
origin_exists = (bool(sort_origin_subfolders) or bool(analysis_origin_subfolders)
                 or bsf_pynapple.exists() or bs_folder.exists())

print("Phase 1 - current contents of rec_folder:")
print(f"  sorting subfolders            : {[f.name for f in sort_origin_subfolders]}")
print(f"  analysis subfolders           : {[f.name for f in analysis_origin_subfolders]}")
print(f"  binary_signals.npz in pynapple: {bsf_pynapple.exists()}")
print(f"  binary_signals staged         : {bs_folder.exists()}")


# Phase 2: check whether the target filestructure already exists
dest_exists = sort_dest.exists() or ana_dest.exists()

print("Phase 2 - target destination status:")
print(f"  sorting dest exists : {sort_dest.exists()} ({sort_dest})")
print(f"  analysis dest exists: {ana_dest.exists()} ({ana_dest})")


# Phase 3: decide what to do based on origin vs destination state
if not origin_exists:
    # nothing sorting/analysis-related remains in rec_folder
    if dest_exists:
        print("Phase 3 - files have already been moved for this recording; nothing to do.")
    else:
        print("Phase 3 - no sorting/analysis files in rec_folder and no destination exists; nothing to move.")
    raise SystemExit(0)

if dest_exists:
    # origin still present AND a destination already exists -> ambiguous/partial state.
    # do not risk overwriting sensitive files; abort and let the user resolve it
    print("Phase 3 - WARNING: rec_folder still contains sorting/analysis files but a destination")
    print("          already exists. Aborting to avoid overwriting. Please resolve manually.")
    raise SystemExit(1)

# origin exists and destinations do not -> safe to create destinations and move files
print("Phase 3 - moving files...")

# 3.1: stage binary_signals.npz out of the pynapple folder into its own bs_folder
if not bs_folder.exists():
    os.makedirs(bs_folder)
if bsf_pynapple.exists():
    shutil.move(str(bsf_pynapple), str(bsf_staged))
elif not bsf_staged.exists():
    print(f"  note: binary_signals.npz not found in {bsf_pynapple} or {bsf_staged}")

# 3.2: create the sorting and analysis destination folders (parents included)
os.makedirs(sort_dest, exist_ok=True)
os.makedirs(ana_dest, exist_ok=True)

# 3.3: move the sorting subfolders (analyzer_*, sorting_summary_*, kilosort*, preprocess) into sorting_files
for f in sort_origin_subfolders:
    print(f"  moving {f.name} -> {sort_dest}")
    shutil.move(str(f), str(sort_dest / f.name))

# 3.4: move the pynapple (analysis) folder wholesale into analysis_files
for f in analysis_origin_subfolders:
    print(f"  moving {f.name} -> {ana_dest}")
    shutil.move(str(f), str(ana_dest / f.name))

# 3.5: distribute the binary_signals folder: copy into analysis, move the original into sorting,
#      leaving rec_folder holding only raw recording files
if bs_folder.exists():
    print(f"  copying binary_signals -> {ana_bs_dest}")
    shutil.copytree(bs_folder, ana_bs_dest)
    print(f"  moving  binary_signals -> {sort_bs_dest}")
    shutil.move(str(bs_folder), str(sort_bs_dest))

print("Phase 3 - done.")



