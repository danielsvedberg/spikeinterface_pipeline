"""
move_preprocess.py migrates one folder that changed sides: preprocess.

It used to be written to sorting_files/ANID/ANID_YYYYMMDD/preprocess, and now belongs in
recording_files/ANID/ANID_YYYYMMDD/preprocess -- a preprocessed recording is a filtered rendering
of the acquisition rather than the product of any one sorting run, so it outlives the sortings
that read it and belongs beside the raw data they were derived from. utils/file_structure.py has
been changed to say so (SessionPaths.preprocess_folder), which leaves already-sorted sessions
pointing at the old location; this script puts them right.

Give it session folders, one per line in the `sorting_folders` list below, and run the file. Each
entry may name the session's sorting_files copy or its recording_files copy -- both resolve to the
same session -- so a list assembled for file_moving.py can be handed over unchanged.

Like file_moving.py, each session reports a status rather than raising, so one unresolvable
session does not abort the rest of the list:

    moved                -> preprocess is now in recording_files
    already_moved        -> it was already there and gone from sorting_files; nothing to do
    nothing_to_move      -> this session has no preprocess folder on either side
    ambiguous            -> BOTH sides have one. moving would nest preprocess inside preprocess,
                            so it is left alone for you to compare the two and delete one
    no_recording_folder  -> there is no recording_files folder to move it into. the raw data has
                            been moved or was never there, and creating the folder here would
                            strand the preprocess folder in an otherwise empty session
    skipped              -> the path is marked by fs.SKIP_SUBSTRINGS ("ignore", "hidden")

The emptied sorting_files folder is left in place; a session whose only remaining content was
preprocess simply ends up with an empty sorting folder, which the pipeline reuses on its next run.
"""

from pathlib import Path
import shutil

#SKIP_SUBSTRINGS and skip_reason come from file_structure too, so this mover and file_moving.py
#cannot disagree about which folders are off limits
from utils import file_structure as fs

#the sessions to migrate. these are the ones under C:/Users/assad/Desktop/experiment_data; the
#comments record what each held when the list was written, so it stays a complete account of the
#tree rather than only of the folders that needed work


sorting_folders = [
    # DS1 -- 2 of 6 with files to move
    Path(r"D:\experiment_data\sorting_files\DS1\DS1_20251117"),
    Path(r"D:\experiment_data\sorting_files\DS1\DS1_20251118"),
    Path(r"D:\experiment_data\sorting_files\DS1\DS1_20251119"),  # nothing to move
    Path(r"D:\experiment_data\sorting_files\DS1\DS1_20251120"),  # nothing to move
    Path(r"D:\experiment_data\sorting_files\DS1\DS1_20251122"),  # nothing to move
    Path(r"D:\experiment_data\sorting_files\DS1\DS1_training"),  # nothing to move
    # DS13 -- 0 of 6 with files to move
    Path(r"D:\experiment_data\sorting_files\DS13\DS13_20250822"),  # nothing to move
    Path(r"D:\experiment_data\sorting_files\DS13\DS13_20250824"),  # nothing to move
    Path(r"D:\experiment_data\sorting_files\DS13\DS13_20250829"),  # nothing to move
    Path(r"D:\experiment_data\sorting_files\DS13\DS13_20250902"),  # nothing to move
    Path(r"D:\experiment_data\sorting_files\DS13\DS13_20250905"),  # nothing to move
    Path(r"D:\experiment_data\sorting_files\DS13\training"),  # nothing to move
    # DS15 -- 6 of 7 with files to move
    Path(r"D:\experiment_data\sorting_files\DS15\DS15_20260630"),
    Path(r"D:\experiment_data\sorting_files\DS15\DS15_20260701"),
    Path(r"D:\experiment_data\sorting_files\DS15\DS15_20260702"),
    Path(r"D:\experiment_data\sorting_files\DS15\DS15_20260703"),
    Path(r"D:\experiment_data\sorting_files\DS15\DS15_20260707"),
    Path(r"D:\experiment_data\sorting_files\DS15\DS15_20260708"),
    Path(r"D:\experiment_data\sorting_files\DS15\training"),  # nothing to move
    # DS2 -- 3 of 3 with files to move
    Path(r"D:\experiment_data\sorting_files\DS2\DS2_20260505"),
    Path(r"D:\experiment_data\sorting_files\DS2\DS2_20260506"),
    Path(r"D:\experiment_data\sorting_files\DS2\DS2_20260511"),
    # DS21 -- 5 of 6 with files to move
    Path(r"D:\experiment_data\sorting_files\DS21\DS21_20251209"),
    Path(r"D:\experiment_data\sorting_files\DS21\DS21_20251210"),
    Path(r"D:\experiment_data\sorting_files\DS21\DS21_20251211"),
    Path(r"D:\experiment_data\sorting_files\DS21\DS21_20251212"),
    Path(r"D:\experiment_data\sorting_files\DS21\DS21_20251213"),
    Path(r"D:\experiment_data\sorting_files\DS21\training"),  # nothing to move
    # DS23 -- 4 of 11 with files to move
    Path(r"D:\experiment_data\sorting_files\DS23\DS23_20251228"),  # nothing to move
    Path(r"D:\experiment_data\sorting_files\DS23\DS23_20251229"),  # nothing to move
    Path(r"D:\experiment_data\sorting_files\DS23\DS23_20260103"),  # nothing to move
    Path(r"D:\experiment_data\sorting_files\DS23\DS23_20260104"),  # nothing to move
    Path(r"D:\experiment_data\sorting_files\DS23\DS23_20260105"),  # nothing to move
    Path(r"D:\experiment_data\sorting_files\DS23\DS23_20260205"),  # nothing to move
    Path(r"D:\experiment_data\sorting_files\DS23\DS23_20260206"),  # nothing to move
    Path(r"D:\experiment_data\sorting_files\DS23\DS23_20260208"),
    Path(r"D:\experiment_data\sorting_files\DS23\DS23_20260209"),
    Path(r"D:\experiment_data\sorting_files\DS23\DS23_20260210"),
    Path(r"D:\experiment_data\sorting_files\DS23\DS23_20260211"),
    # DS37 -- 4 of 5 with files to move
    Path(r"D:\experiment_data\sorting_files\DS37\DS37_031726"),
    Path(r"D:\experiment_data\sorting_files\DS37\DS37_031826"),
    Path(r"D:\experiment_data\sorting_files\DS37\DS37_031926"),
    Path(r"D:\experiment_data\sorting_files\DS37\DS37_032026"),  # nothing to move
    Path(r"D:\experiment_data\sorting_files\DS37\DS37_032126"),
    # DS45 -- 0 of 11 with files to move
    Path(r"D:\experiment_data\sorting_files\DS45\DS45_20260629"),  # nothing to move
    Path(r"D:\experiment_data\sorting_files\DS45\DS45_20260701"),  # nothing to move
    Path(r"D:\experiment_data\sorting_files\DS45\DS45_20260702"),  # nothing to move
    Path(r"D:\experiment_data\sorting_files\DS45\DS45_20260707"),  # nothing to move
    Path(r"D:\experiment_data\sorting_files\DS45\DS45_20260708"),  # nothing to move
    Path(r"D:\experiment_data\sorting_files\DS45\DS45_20260709"),  # nothing to move
    Path(r"D:\experiment_data\sorting_files\DS45\DS45_20260710"),  # nothing to move
    Path(r"D:\experiment_data\sorting_files\DS45\DS45_20260717"),  # nothing to move
    Path(r"D:\experiment_data\sorting_files\DS45\DS45_20260720"),  # nothing to move
    Path(r"D:\experiment_data\sorting_files\DS45\DS45_20260721"),  # nothing to move
    Path(r"D:\experiment_data\sorting_files\DS45\DS45_20260722"),  # nothing to move
    # DS48 -- 0 of 4 with files to move
    Path(r"D:\experiment_data\sorting_files\DS48\DS48_20260717"),  # nothing to move
    Path(r"D:\experiment_data\sorting_files\DS48\DS48_20260720"),  # nothing to move
    Path(r"D:\experiment_data\sorting_files\DS48\DS48_20260721"),  # nothing to move
    Path(r"D:\experiment_data\sorting_files\DS48\DS48_20260722"),  # nothing to move
    # Neptun -- 1 of 1 with files to move
    Path(r"D:\experiment_data\sorting_files\Neptun\Neptun_20260625"),
]


def move_preprocess_folder(session_folder, skip_substrings=fs.SKIP_SUBSTRINGS):
    """Move one session's preprocess folder from sorting_files to recording_files.

    session_folder may be either category's copy of the session. Returns one of the status
    strings documented at the top of this module.
    """
    session_folder = Path(session_folder)
    print(f"\n=== {session_folder}")

    #both ends are named through category_path rather than off a SessionPaths object, for the
    #reason file_moving.py gives at its step 1.3: the checked accessors answer None for a folder
    #that is not there, which is exactly the state this program has to be able to see and report
    _, _, rel_ID = fs.split_category_path(session_folder)
    skipped = fs.skip_reason(rel_ID.parts, skip_substrings)
    if skipped:
        print(f"Skipping: path contains {skipped!r}.")
        return 'skipped'

    rec_folder = fs.category_path(session_folder, fs.RECORDING_ROOT)
    sort_folder = fs.category_path(session_folder, fs.SORTING_ROOT)
    source = sort_folder / fs.PREPROCESS_FOLDER
    dest = rec_folder / fs.PREPROCESS_FOLDER

    print(f"  in sorting_files  : {source.is_dir()} ({source})")
    print(f"  in recording_files: {dest.is_dir()} ({dest})")

    if not source.is_dir():
        if dest.is_dir():
            print("  already in recording_files; nothing to do.")
            return 'already_moved'
        print("  no preprocess folder on either side; nothing to move.")
        return 'nothing_to_move'

    if dest.is_dir():
        #shutil.move onto an existing directory puts the source *inside* it, so this would
        #silently produce recording_files/<session>/preprocess/preprocess. the likely cause is a
        #pipeline run after the layout change, which wrote a fresh preprocess folder beside the
        #raw data while the old one still sat in sorting_files
        print("  WARNING: both sides have a preprocess folder. Skipping to avoid nesting one")
        print("           inside the other -- compare them and delete whichever is stale, then")
        print("           re-run this mover.")
        return 'ambiguous'

    if not rec_folder.is_dir():
        print(f"  WARNING: no recording folder for this session ({rec_folder}). Skipping rather")
        print("           than creating one, which would leave preprocess in an empty session.")
        return 'no_recording_folder'

    print(f"  moving preprocess -> {dest}")
    shutil.move(str(source), str(dest))
    return 'moved'


def move_preprocess_folders(session_folders, skip_substrings=fs.SKIP_SUBSTRINGS):
    """Migrate a list of sessions' preprocess folders, reporting on each one."""
    results = {}
    for session_folder in session_folders:
        results[Path(session_folder)] = move_preprocess_folder(session_folder, skip_substrings)

    print("\n=== summary")
    for folder, status in results.items():
        print(f"  {status:<20} {folder}")
    return results


if __name__ == "__main__":
    move_preprocess_folders(sorting_folders)
