"""
file_moving.py will take a (recording) rec_folder, which contains raw recording files, preprocessed signals files, spike-sorting
files, and post-hoc analysis that all correspond to a single recording session--spread across various folders, within
the rec_folder--and moves them to separate folders in 3 categories: 1) (raw) recording files, which will contain
the recorded data and the preprocess folder derived from it, 2) sorting_files, which will contain files related to the
spike sorting pipeline, including extracted binary signals, the kilosort spike sorting output, and files related to
curation, and 3)
analysis files, which will contain the pynapple files that are the product of the final curation, posthoc analysis plots,
and a copy of the extracted binary signals. Raw recording files include any file under ANID_g# folders as well as
the ANID_YYYYMMDD_Timestamp.csv, ...FrameID.csv, .avi, .mat, and .txt files, and the preprocess folder--which is
already in rec_folder and so is simply left there. Sorting files include any analyzer_*,
kilosort4_output, and sorting_summary_* folders, as well as the binary_signals.npz file in the pynapple folder
Analysis files include all original contents of the pynapple folder. A significant catch is that before files get moved
around, the binary_signals.npz file should be moved to a new subfolder of rec_folder called "binary_signals" so that
the binary signals can be easily copied to both the sorting and anlaysis folders.
The pynapple folder will be moved wholesale to the analysis folder, and the binary_signals.npz file will be copied to
the sorting folder. The pynapple folder will no longer contain the binary_signals.npz file, but will still contain all
other files that were originally in the pynapple folder.

Some analysis products span several recording sessions of one animal rather than belonging to any single one--the
compendium folder is the example--and so they sit beside the rec_folders in the an_ID folder instead of inside one.
These are moved to the same place relative to the an_ID folder: analysis_files/ANID/compendium, a sibling of the
session folders. The subject-level .mat files next to them (HSOMBS_ANID_YYYYMMDD.mat) are raw recording files and
stay put, like the rest of the raw data.

move_rec_folder() does this for one rec_folder and move_subject_folder() for one an_ID folder; both report what they
did, so a whole list of rec_folders can be moved in one run: a recording that is already moved, or in a partially-moved
state, is skipped without aborting the others. move_rec_folders() takes just the list of rec_folders and moves each
one's an_ID-level products too, once per animal. Since preprocess now stays put, a rec_folder whose only non-raw
subfolder is preprocess has nothing this program needs to do, and reports 'nothing_to_move' rather than 'moved'.
Sessions split under the older layout, with preprocess already in sorting_files, are put right by move_preprocess.py.
The target layout itself is defined once in utils/file_structure.py, which is also what the pipeline now writes to.
"""

from pathlib import Path
import os
import shutil

from utils import file_structure as fs

#folders whose name contains any of these are left completely alone -- not moved, not scanned, not
#reported on. Matching is case-insensitive and applies to every path component below the category
#root, so an animal folder called "ignore" excludes everything under it. The rule lives in
#file_structure.py so that every mover obeys the same one; pass skip_substrings=() to any of the
#movers here to turn it off
SKIP_SUBSTRINGS = fs.SKIP_SUBSTRINGS


#every session folder under D:/experiment_data/recording_files, one per line. Folders
#matched by SKIP_SUBSTRINGS are left out (recording_files/ignore/), as is DS2/compendium,
#which is an an_ID-level product and gets moved by move_subject_folder instead.
#The ones marked "nothing to move" hold no sorting/analysis products today; they are kept
#so the list stays a complete record, and simply report nothing_to_move.
r'''
rec_folders = [
    # DS1 -- 2 of 6 with files to move
    Path(r"D:\experiment_data\recording_files\DS1\DS1_20251117"),
    Path(r"D:\experiment_data\recording_files\DS1\DS1_20251118"),
    Path(r"D:\experiment_data\recording_files\DS1\DS1_20251119"),  # nothing to move
    Path(r"D:\experiment_data\recording_files\DS1\DS1_20251120"),  # nothing to move
    Path(r"D:\experiment_data\recording_files\DS1\DS1_20251122"),  # nothing to move
    Path(r"D:\experiment_data\recording_files\DS1\DS1_training"),  # nothing to move
    # DS13 -- 0 of 6 with files to move
    Path(r"D:\experiment_data\recording_files\DS13\DS13_20250822"),  # nothing to move
    Path(r"D:\experiment_data\recording_files\DS13\DS13_20250824"),  # nothing to move
    Path(r"D:\experiment_data\recording_files\DS13\DS13_20250829"),  # nothing to move
    Path(r"D:\experiment_data\recording_files\DS13\DS13_20250902"),  # nothing to move
    Path(r"D:\experiment_data\recording_files\DS13\DS13_20250905"),  # nothing to move
    Path(r"D:\experiment_data\recording_files\DS13\training"),  # nothing to move
    # DS15 -- 6 of 7 with files to move
    Path(r"D:\experiment_data\recording_files\DS15\DS15_20260630"),
    Path(r"D:\experiment_data\recording_files\DS15\DS15_20260701"),
    Path(r"D:\experiment_data\recording_files\DS15\DS15_20260702"),
    Path(r"D:\experiment_data\recording_files\DS15\DS15_20260703"),
    Path(r"D:\experiment_data\recording_files\DS15\DS15_20260707"),
    Path(r"D:\experiment_data\recording_files\DS15\DS15_20260708"),
    Path(r"D:\experiment_data\recording_files\DS15\training"),  # nothing to move
    # DS2 -- 3 of 3 with files to move
    Path(r"D:\experiment_data\recording_files\DS2\DS2_20260505"),
    Path(r"D:\experiment_data\recording_files\DS2\DS2_20260506"),
    Path(r"D:\experiment_data\recording_files\DS2\DS2_20260511"),
    # DS21 -- 5 of 6 with files to move
    Path(r"D:\experiment_data\recording_files\DS21\DS21_20251209"),
    Path(r"D:\experiment_data\recording_files\DS21\DS21_20251210"),
    Path(r"D:\experiment_data\recording_files\DS21\DS21_20251211"),
    Path(r"D:\experiment_data\recording_files\DS21\DS21_20251212"),
    Path(r"D:\experiment_data\recording_files\DS21\DS21_20251213"),
    Path(r"D:\experiment_data\recording_files\DS21\training"),  # nothing to move
    # DS23 -- 4 of 11 with files to move
    Path(r"D:\experiment_data\recording_files\DS23\DS23_20251228"),  # nothing to move
    Path(r"D:\experiment_data\recording_files\DS23\DS23_20251229"),  # nothing to move
    Path(r"D:\experiment_data\recording_files\DS23\DS23_20260103"),  # nothing to move
    Path(r"D:\experiment_data\recording_files\DS23\DS23_20260104"),  # nothing to move
    Path(r"D:\experiment_data\recording_files\DS23\DS23_20260105"),  # nothing to move
    Path(r"D:\experiment_data\recording_files\DS23\DS23_20260205"),  # nothing to move
    Path(r"D:\experiment_data\recording_files\DS23\DS23_20260206"),  # nothing to move
    Path(r"D:\experiment_data\recording_files\DS23\DS23_20260208"),
    Path(r"D:\experiment_data\recording_files\DS23\DS23_20260209"),
    Path(r"D:\experiment_data\recording_files\DS23\DS23_20260210"),
    Path(r"D:\experiment_data\recording_files\DS23\DS23_20260211"),
    # DS37 -- 4 of 5 with files to move
    Path(r"D:\experiment_data\recording_files\DS37\DS37_031726"),
    Path(r"D:\experiment_data\recording_files\DS37\DS37_031826"),
    Path(r"D:\experiment_data\recording_files\DS37\DS37_031926"),
    Path(r"D:\experiment_data\recording_files\DS37\DS37_032026"),  # nothing to move
    Path(r"D:\experiment_data\recording_files\DS37\DS37_032126"),
    # DS45 -- 0 of 11 with files to move
    Path(r"D:\experiment_data\recording_files\DS45\DS45_20260629"),  # nothing to move
    Path(r"D:\experiment_data\recording_files\DS45\DS45_20260701"),  # nothing to move
    Path(r"D:\experiment_data\recording_files\DS45\DS45_20260702"),  # nothing to move
    Path(r"D:\experiment_data\recording_files\DS45\DS45_20260707"),  # nothing to move
    Path(r"D:\experiment_data\recording_files\DS45\DS45_20260708"),  # nothing to move
    Path(r"D:\experiment_data\recording_files\DS45\DS45_20260709"),  # nothing to move
    Path(r"D:\experiment_data\recording_files\DS45\DS45_20260710"),  # nothing to move
    Path(r"D:\experiment_data\recording_files\DS45\DS45_20260717"),  # nothing to move
    Path(r"D:\experiment_data\recording_files\DS45\DS45_20260720"),  # nothing to move
    Path(r"D:\experiment_data\recording_files\DS45\DS45_20260721"),  # nothing to move
    Path(r"D:\experiment_data\recording_files\DS45\DS45_20260722"),  # nothing to move
    # DS48 -- 0 of 4 with files to move
    Path(r"D:\experiment_data\recording_files\DS48\DS48_20260717"),  # nothing to move
    Path(r"D:\experiment_data\recording_files\DS48\DS48_20260720"),  # nothing to move
    Path(r"D:\experiment_data\recording_files\DS48\DS48_20260721"),  # nothing to move
    Path(r"D:\experiment_data\recording_files\DS48\DS48_20260722"),  # nothing to move
    # Neptun -- 1 of 1 with files to move
    Path(r"D:\experiment_data\recording_files\Neptun\Neptun_20260625"),
]
'''

#Every session folder under C:/Users/assad/Desktop/experiment_data/recording_files. Its own
#root, so these resolve to Desktop/experiment_data/{sorting,analysis}_files -- separate from
#the D: tree, whose list is kept in the r''' block above.
#Skipped by SKIP_SUBSTRINGS: nothing here matches. Left out as they hold no sessions:
#DS45/, test/ (both empty) and CalSRate_2025-11-20T12.32.06_g0/ (a stray _g0 folder sitting
#at the animal level).
rec_folders = [
    # Bhad_Bhabie -- 0 of 1 with files to move
    Path(r"C:\Users\assad\Desktop\experiment_data\recording_files\Bhad_Bhabie\Bhad_Bhabie_test"),  # nothing to move
    # DS15 -- 1 of 6 with files to move
    Path(r"C:\Users\assad\Desktop\experiment_data\recording_files\DS15\DS15_20260630"),  # nothing to move
    Path(r"C:\Users\assad\Desktop\experiment_data\recording_files\DS15\DS15_20260701"),  # already moved
    Path(r"C:\Users\assad\Desktop\experiment_data\recording_files\DS15\DS15_20260702"),
    Path(r"C:\Users\assad\Desktop\experiment_data\recording_files\DS15\DS15_20260703"),  # already moved
    Path(r"C:\Users\assad\Desktop\experiment_data\recording_files\DS15\DS15_20260707"),  # already moved
    Path(r"C:\Users\assad\Desktop\experiment_data\recording_files\DS15\DS15_20260708"),  # already moved
    # DS48 -- 0 of 5 with files to move
    Path(r"C:\Users\assad\Desktop\experiment_data\recording_files\DS48\DS48_20260723"),  # nothing to move
    Path(r"C:\Users\assad\Desktop\experiment_data\recording_files\DS48\DS48_20260724"),  # nothing to move
    Path(r"C:\Users\assad\Desktop\experiment_data\recording_files\DS48\DS48_20260727"),  # nothing to move
    Path(r"C:\Users\assad\Desktop\experiment_data\recording_files\DS48\DS48_20260728"),  # nothing to move
    Path(r"C:\Users\assad\Desktop\experiment_data\recording_files\DS48\DS48_20260729"),  # nothing to move
    # Neptun -- 1 of 1 with files to move
    Path(r"C:\Users\assad\Desktop\experiment_data\recording_files\Neptun\Neptun_20260626"),
]

"""
Example original filestructure (rec_folder = recording_files/DS15/DS15_20260701):

recording_files/
└── DS15/                                     <- animal folder (an_ID)
    ├── DS15_20260701/                        <- rec_folder (this recording session)
    │   ├── DS15_20260701_g0/                 <- raw recording (ANID_g# folder)
    │   │   └── ...                               raw acquisition files
    │   ├── DS15_20260701_142530.csv          <- ANID_YYYYMMDD_Timestamp.csv
    │   ├── DS15_20260701_142530_FrameID.csv
    │   ├── DS15_20260701.avi
    │   ├── DS15_20260701.mat
    │   ├── DS15_20260701.txt
    │   ├── preprocess/                        <- recording: preprocessed data (stays put)
    │   ├── kilosort4_output/                  <- sorting: kilosort spike sorting output
    │   ├── analyzer_DS15_20260701/            <- sorting: sorting analyzer (analyzer_*)
    │   ├── sorting_summary_DS15_20260701/     <- sorting: curation summary (sorting_summary_*)
    │   └── pynapple/                          <- analysis: final curation products
    │       ├── binary_signals.npz            <- extracted binary signals (needed by both sorting & analysis)
    │       └── ...                               other pynapple analysis files
    ├── HSOMBS_DS15_20260701.mat              <- an_ID-level raw recording file (stays put)
    └── compendium/                            <- an_ID-level analysis: spans several sessions
        └── ...                                   cross-session plots and tables

Example target filestructure (root_folder holds the 3 category folders, each mirroring an_ID/rec_ID):

root_folder/
├── recording_files/                          <- raw recorded data only
│   └── DS15/
│       ├── DS15_20260701/
│       │   ├── DS15_20260701_g0/             <- ANID_g# raw acquisition folder
│       │   │   └── ...
│       │   ├── DS15_20260701_142530.csv      <- ANID_YYYYMMDD_Timestamp.csv
│       │   ├── DS15_20260701_142530_FrameID.csv
│       │   ├── DS15_20260701.avi
│       │   ├── DS15_20260701.mat
│       │   ├── DS15_20260701.txt
│       │   └── preprocess/                   <- preprocessed data, left where it already was
│       └── HSOMBS_DS15_20260701.mat          <- an_ID-level raw file, left in place
├── sorting_files/                            <- spike-sorting pipeline products
│   └── DS15/
│       └── DS15_20260701/
│           ├── kilosort4_output/
│           ├── analyzer_DS15_20260701/
│           ├── sorting_summary_DS15_20260701/
│           └── binary_signals/               <- copy of the binary_signals folder
│               └── binary_signals.npz
└── analysis_files/                           <- final curation + posthoc analysis
    └── DS15/
        ├── DS15_20260701/
        │   ├── pynapple/                      <- moved wholesale (minus binary_signals.npz)
        │   │   └── ...                            other original pynapple files
        │   └── binary_signals/               <- copy of the binary_signals folder
        │       └── binary_signals.npz
        └── compendium/                        <- an_ID-level: sibling of the session folders
            └── ...                                cross-session plots and tables
"""


def move_rec_folder(rec_folder, skip_substrings=SKIP_SUBSTRINGS):
    """Move one rec_folder's sorting and analysis files out to their category folders.

    Returns a short status string ('moved', 'already_moved', 'nothing_to_move', 'ambiguous' or
    'skipped') so a batch run can be summarised, and so an unresolvable recording is skipped
    rather than aborting the rest of the list.
    """
    rec_folder = Path(rec_folder)
    print(f"\n=== {rec_folder}")

    # Part 1: declare all folders in scope of the program, and relevant file parts
    # 1.1: resolve this session's identity (root_folder + an_ID/rec_ID) and its category folders.
    #      Note recording_files needs no destination of its own: rec_folder IS the raw-recording
    #      folder, so the raw files simply stay put once sorting/analysis are moved out
    folders = fs.SessionPaths(rec_folder)

    # 1.1b: a session parked under a name like "ignore" is off limits, as is one whose animal
    #       folder is. checking rec_ID rather than the whole path keeps the test to the part of
    #       the tree this program owns, so a root folder that happens to contain the word is fine
    skipped = fs.skip_reason(folders.rec_ID.parts, skip_substrings)
    if skipped:
        print(f"Skipping: path contains {skipped!r}.")
        return 'skipped'

    # 1.2: binary signals staging. binary_signals.npz originally lives inside the pynapple (analysis)
    #      folder; before anything else it is staged into its own bs_folder so it can be copied to
    #      both the sorting and analysis destinations (programs doing both need it)
    bs_folder = rec_folder / fs.BINARY_SIGNALS_FOLDER  # "bs": staging folder inside rec_folder
    pynapple_folder = rec_folder / fs.PYNAPPLE_FOLDER
    bsf_pynapple = pynapple_folder / fs.BINARY_SIGNALS_FILE  # original location
    bsf_staged = bs_folder / fs.BINARY_SIGNALS_FILE          # staged location inside bs_folder

    # 1.3: destination folders, each mirroring rec_ID under its category root. these are named
    #      through category_path rather than off the SessionPaths object, because a destination
    #      that does not exist yet is exactly what this program is about to create -- whereas
    #      SessionPaths.sorting_folder reports None for one that is not there, and would both
    #      hide the path and break the "has it already been moved?" test below
    sort_dest = fs.category_path(rec_folder, fs.SORTING_ROOT)   # root/sorting_files/DS15/DS15_...
    ana_dest = fs.category_path(rec_folder, fs.ANALYSIS_ROOT)   # root/analysis_files/DS15/DS15_...
    sort_bs_dest = sort_dest / fs.BINARY_SIGNALS_FOLDER
    ana_bs_dest = ana_dest / fs.BINARY_SIGNALS_FOLDER

    # Part 2 / Phase 1: identify the current filestructure of rec_folder. a subfolder whose own
    # name is marked stays put, so e.g. preprocess_ignore is left where it is
    rec_subfolders = [f for f in rec_folder.iterdir()
                      if f.is_dir() and not fs.skip_reason([f.name], skip_substrings)]
    sort_origin_subfolders = [f for f in rec_subfolders
                              if any(f.name.startswith(p) for p in fs.SORTING_SUBFOLDER_PREFIXES)]
    analysis_origin_subfolders = [f for f in rec_subfolders
                                  if any(f.name.startswith(p) for p in fs.ANALYSIS_SUBFOLDER_PREFIXES)]
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
            return 'already_moved'
        print("Phase 3 - no sorting/analysis files in rec_folder and no destination exists; nothing to move.")
        return 'nothing_to_move'

    if dest_exists:
        # origin still present AND a destination already exists -> ambiguous/partial state.
        # do not risk overwriting sensitive files; skip and let the user resolve it.
        # the common cause is running the pipeline on this session before moving it: the pipeline
        # now creates these destinations itself, so they can exist (even empty) alongside the
        # legacy copies still sitting in rec_folder
        print("Phase 3 - WARNING: rec_folder still contains sorting/analysis files but a destination")
        print("          already exists. Skipping to avoid overwriting. Please resolve manually")
        print("          (if the destination is an empty folder left by a pipeline run, delete it")
        print("           and re-run this mover).")
        return 'ambiguous'

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
    folders.create_output_folders()

    # 3.3: move the sorting subfolders (analyzer_*, sorting_summary_*, kilosort*) into sorting_files.
    #      preprocess is not among them: it is a recording file, and rec_folder is already the
    #      recording folder, so it stays exactly where it is
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
    return 'moved'


def move_subject_folder(an_folder, skip_substrings=SKIP_SUBSTRINGS):
    """Move one an_ID folder's cross-session analysis products (compendium) into analysis_files.

    These span several recordings, so they belong beside the session folders under
    analysis_files/<an_ID>/ rather than inside any one of them. Everything else in the an_ID folder
    -- the rec_folders themselves and the subject-level .mat files -- is left untouched.
    Returns the same status strings as move_rec_folder().
    """
    an_folder = Path(an_folder)
    print(f"\n=== {an_folder} (an_ID level)")

    _, _, rel_ID = fs.split_category_path(an_folder)
    skipped = fs.skip_reason(rel_ID.parts, skip_substrings)
    if skipped:
        print(f"Skipping: path contains {skipped!r}.")
        return 'skipped'

    # 1: the an_ID folder mirrored under analysis_files, e.g. analysis_files/DS2
    ana_dest = fs.category_path(an_folder, fs.ANALYSIS_ROOT)

    # 2: which subfolders here are cross-session analysis products
    an_subfolders = [f for f in an_folder.iterdir()
                     if f.is_dir() and not fs.skip_reason([f.name], skip_substrings)]
    subject_origin_subfolders = [f for f in an_subfolders
                                 if any(f.name.startswith(p) for p in fs.SUBJECT_ANALYSIS_SUBFOLDER_PREFIXES)]
    origin_exists = bool(subject_origin_subfolders)

    # 3: whether such products already exist at the destination. note this checks the product
    #    folders themselves, not ana_dest: analysis_files/<an_ID> is created as soon as any one of
    #    this animal's sessions is moved, so its mere existence says nothing about the compendium
    existing_dests = [f for f in (ana_dest.iterdir() if ana_dest.is_dir() else [])
                      if f.is_dir() and any(f.name.startswith(p) for p in fs.SUBJECT_ANALYSIS_SUBFOLDER_PREFIXES)]

    print("Phase 1 - an_ID-level analysis subfolders:")
    print(f"  in an_folder      : {[f.name for f in subject_origin_subfolders]}")
    print(f"  already at dest   : {[d.name for d in existing_dests]} ({ana_dest})")

    if not origin_exists:
        if existing_dests:
            print("Phase 2 - an_ID-level files have already been moved; nothing to do.")
            return 'already_moved'
        print("Phase 2 - no an_ID-level analysis files to move.")
        return 'nothing_to_move'

    if existing_dests:
        # a destination already holds a folder of this name -- most likely compendium_plots.py was
        # run after the layout change and wrote there directly. moving on top of it would nest the
        # old folder inside the new one, so stop and let the user reconcile the two
        print("Phase 2 - WARNING: an_folder still contains an_ID-level analysis files but a destination")
        print("          already exists. Skipping to avoid overwriting. Please resolve manually.")
        return 'ambiguous'

    print("Phase 2 - moving an_ID-level files...")
    ana_dest.mkdir(parents=True, exist_ok=True)
    for f in subject_origin_subfolders:
        print(f"  moving {f.name} -> {ana_dest}")
        shutil.move(str(f), str(ana_dest / f.name))

    print("Phase 2 - done.")
    return 'moved'


def move_rec_folders(rec_folders, skip_substrings=SKIP_SUBSTRINGS):
    """Move a list of rec_folders, plus the an_ID-level products of each one's animal.

    The an_ID folders are handled first and only once per animal, however many of that animal's
    sessions appear in the list. Anything whose path is marked by skip_substrings is reported as
    'skipped' and otherwise left alone.
    """
    results = {}
    for an_folder in dict.fromkeys(Path(f).parent for f in rec_folders):
        results[an_folder] = move_subject_folder(an_folder, skip_substrings)

    for rec_folder in rec_folders:
        results[Path(rec_folder)] = move_rec_folder(rec_folder, skip_substrings)

    print("\n=== summary")
    for folder, status in results.items():
        print(f"  {status:<16} {folder}")
    return results


if __name__ == "__main__":
    move_rec_folders(rec_folders)