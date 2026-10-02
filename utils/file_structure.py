"""
file_structure.py is the single definition of where a recording session's files live.

Every session is identified by a rec_ID (e.g. DS15/DS15_20260701, i.e. an_ID/session) that is
mirrored under three sibling category roots inside one root_folder:

root_folder/
├── recording_files/DS15/DS15_20260701/   <- acquisition data (_g# folders, csv/avi/mat/txt, .rhd
│                                            folders, nidaq_map.txt / intan_map.txt) plus preprocess/,
│                                            which is a filtered rendering of that same raw signal
├── sorting_files/DS15/DS15_20260701/     <- kilosort4_output, analyzer*, sorting_summary*,
│                                            binary_signals/binary_signals.npz (copy)
└── analysis_files/DS15/DS15_20260701/    <- pynapple/ (curation products, event tables, plots),
                                             binary_signals/binary_signals.npz (primary)

SessionPaths is the accessor for all of it: given any one of a session's three category folders it
returns every other path the pipeline needs, folders and individual files alike (analysis_folder,
pynapple_folder, binary_signals_file(), sorting_path(...), input_map, spikeglx_folder(), ...).
Programs are handed a session folder (any category's copy of the session) and resolve the rest from
it, so raw data is always read from recording_files while products are always written to
sorting_files/analysis_files. Old sessions are often archived -- recording_files and sorting_files
moved off the drive to save space, analysis_files left behind -- so posthoc programs are handed the
analysis_folder instead and SessionPaths resolves everything that does not depend on the raw data.
file_moving.py uses the same definitions to relocate legacy sessions that still keep everything
inside rec_folder.
"""

from pathlib import Path
import shutil

#the three category roots. these names are the vocabulary used for the `category` argument
#throughout this module, so there is no second set of aliases to keep in sync
RECORDING_ROOT = "recording_files"
SORTING_ROOT = "sorting_files"
ANALYSIS_ROOT = "analysis_files"
CATEGORY_ROOTS = (RECORDING_ROOT, SORTING_ROOT, ANALYSIS_ROOT)

#names of the standard products inside a session's category folders
PREPROCESS_FOLDER = "preprocess"
#the raw-data saturation scan's cache and report, beside preprocess rather than inside it:
#saving the preprocessed recording with overwrite=True deletes its folder first
SATURATION_FOLDER = "saturation"
KS_FOLDER = "kilosort4_output"
PYNAPPLE_FOLDER = "pynapple"
BINARY_SIGNALS_FOLDER = "binary_signals"
BINARY_SIGNALS_FILE = "binary_signals.npz"

#the acquisition channel map, written beside the raw data. its name depends on the rig, so both
#are listed and whichever a session actually has is the one SessionPaths.input_map returns
INPUT_MAP_FILES = ("nidaq_map.txt", "intan_map.txt")

#the behaviour computer's per-session log of every task event (time in Arduino ms, label), written
#beside the raw data as <session>_eventTable.csv. preprocessing/event_table_licks.py reads its licks
#when the NIDQ lick line was not recorded
EVENT_TABLE_SUFFIX = "_eventTable.csv"

#which of a legacy rec_folder's subfolders belong to which category (used by file_moving.py, and
#kept here so the pipeline's write locations and the mover's expectations cannot drift apart).
#preprocess is deliberately absent: it belongs to recording_files, which is where a legacy
#rec_folder already keeps it, so the mover leaves it exactly where it is
SORTING_SUBFOLDER_PREFIXES = ("analyzer", "sorting_summary", "kilosort")
ANALYSIS_SUBFOLDER_PREFIXES = (PYNAPPLE_FOLDER,)

#an_ID-level analysis products: folders that sit beside the session folders and span several
#sessions of one animal, so they belong under analysis_files/<an_ID>/ rather than any one session
SUBJECT_ANALYSIS_SUBFOLDER_PREFIXES = ("compendium",)

#folders whose name contains any of these are off limits to the movers -- not moved, not scanned,
#not reported on. Kept here beside the layout itself so every program that relocates files obeys
#the same rule; pass skip_substrings=() to any mover to turn it off
SKIP_SUBSTRINGS = ("hidden", "ignore")


def skip_reason(names, skip_substrings=SKIP_SUBSTRINGS):
    """The first skip substring found in any of `names`, or None if none of them match.

    Matching is case-insensitive. Callers hand it the path components below the category root
    -- rec_ID.parts, or a single subfolder name -- so a marked animal folder excludes everything
    under it while a root folder that happens to contain the word is fine.
    """
    for name in names:
        lowered = str(name).lower()
        for substring in skip_substrings:
            if substring.lower() in lowered:
                return substring
    return None


def split_category_path(folder):
    """Split a path into (root_folder, category, rel_ID).

    rel_ID is the part of the path below the category root, e.g. DS15/DS15_20260701 for a session
    folder or DS15 for a subject folder. The innermost category root wins, so a path that happens
    to nest one root inside another resolves to the one closest to the session.
    """
    folder = Path(folder)
    parts = folder.parts
    idxs = [i for i, p in enumerate(parts) if p in CATEGORY_ROOTS]
    if not idxs:
        raise ValueError(
            f"{folder} does not live under one of the category roots {CATEGORY_ROOTS}. "
            f"Paths must look like <root_folder>/{RECORDING_ROOT}/<an_ID>/<rec_ID>."
        )
    idx = idxs[-1]
    return Path(*parts[:idx]), parts[idx], Path(*parts[idx + 1:])


def category_path(folder, category):
    """Map a folder under one category root to the same relative location under another.

    Works for session folders and subject (an_ID) folders alike:
    recording_files/DS2 -> analysis_files/DS2, recording_files/DS2/DS2_20260511 ->
    analysis_files/DS2/DS2_20260511.
    """
    if category not in CATEGORY_ROOTS:
        raise ValueError(f"category must be one of {CATEGORY_ROOTS}, got {category}")
    root_folder, _, rel_ID = split_category_path(folder)
    return root_folder / category / rel_ID


def find_imec_folder(spikeglx_folder, fail_quietly=False):
    """The folder holding a SpikeGLX run's imec[#].ap.bin / imec[#].ap.meta pair.

    Normally a probe subfolder of the run folder; some recordings keep the files in the run folder
    itself, which is the fallback. Takes the run folder directly so a caller that has already
    chosen between several runs does not have to re-resolve it.
    """
    spikeglx_folder = Path(spikeglx_folder)

    #the probe subfolder case: files are named <run>_t0.imec0.ap.bin, hence the leading '*.'
    for subfolder in (f for f in spikeglx_folder.iterdir() if f.is_dir()):
        bin_files = list(subfolder.glob("*.imec[0-9]*.ap.bin"))
        meta_files = list(subfolder.glob("*.imec[0-9]*.ap.meta"))
        if len(bin_files) > 0 and len(meta_files) > 0:
            return subfolder

    #the flat case: the probe files sit in the run folder, named imec0.ap.bin (no run prefix)
    bin_files = list(spikeglx_folder.glob("imec[0-9]*.ap.bin"))
    meta_files = list(spikeglx_folder.glob("imec[0-9]*.ap.meta"))
    if len(bin_files) > 0 and len(meta_files) > 0:
        return spikeglx_folder
    elif fail_quietly:
        return None
    else:
        raise FileNotFoundError(
            f"No IMEC probe files (imec[#].ap.bin and imec[#].ap.meta) found in {spikeglx_folder}")


class SessionPaths:
    """Every path belonging to one recording session -- folders and individual files alike.

    Construct it from any of the session's three category folders -- normally the raw rec_folder,
    but the analysis_folder does just as well; every other location is derived from the rec_ID,
    not from the folder handed in, so no caller needs to know the layout.

    The sorting and analysis folders report what is actually on disk, and the two kinds of
    accessor answer a missing one differently:

      * FOLDER accessors -- sorting_folder, analysis_folder, sorting_path, analysis_path,
        pynapple_folder, binary_signals_folder, preprocess_folder, ks_folder -- return None.
        "Has this session been sorted yet?" is then a question you can ask the object, rather
        than one every caller answers with its own .exists() call.
      * FILE accessors -- binary_signals_file -- raise FileNotFoundError instead. They are asked
        for something to open, and None is never a useful answer to that: it defers the failure
        to whatever tries to read it, which then reports a TypeError about NoneType and says
        nothing about which session or which stage is missing.

    Pass create=True to have the folders made at construction, after which they exist and every
    accessor returns a path -- except the two product accessors noted below, preprocess_folder
    and ks_folder, which stay None until the stage that fills them has actually run.

    recording_folder is deliberately not treated that way: it always returns its path, never
    None. There is no not-yet state for raw data, and the accessors above it -- preprocess_folder,
    input_map -- read it as a path, so a None would only resurface as the TypeError this design
    exists to avoid.

    That path can still be absent, though, and legitimately so: an ARCHIVED session is one whose
    recording_files (and normally sorting_files with it) have been moved off this drive to
    reclaim space, leaving analysis_files behind. Such a session is built from its analysis_folder
    and everything not derived from the raw data keeps working -- analysis_path, pynapple_folder,
    binary_signals_file() -- while preprocess_folder and input_map return None and the
    raw-acquisition accessors raise, naming the folder that is gone. has_recording is the question
    to ask about it, and it is what stops create=True from rebuilding the sorting tree that was
    deleted; see _creatable.

    preprocess_folder is the exception to create=True: it stays None until something has
    actually been preprocessed, however this object was built, because creating it empty is not
    a harmless head start -- SpikeInterface's recording.save(folder=...) refuses to write into a
    folder that already exists, so pre-making it would break the very step that fills it.

    preprocess_folder and ks_folder are also the two accessors that check for the product itself
    rather than for the category folder around it, because in both cases that folder is always
    there by the time anyone asks -- recording_folder never has a not-yet state, and the sorting
    folder is normally made up front by create_output_folders. Testing the category folder would
    make `is None` answer "does this session have somewhere to put a sorting?" when what the
    caller means is "has this session been sorted?".

    To name a destination that does not exist yet and should NOT be created -- what a writer
    needs, and what file_moving.py needs -- use the module-level category_path(),
    recording_folder / PREPROCESS_FOLDER for the preprocess case, or sorting_path(KS_FOLDER)
    for the sorting one.

    The other members that touch the filesystem are the ones whose name is not fixed by the
    layout: spikeglx_folder and imec_folder raise when there is nothing to find, and input_map
    returns None when the session has no channel map.
    """

    def __init__(self, session_folder, create=False):
        #the folder this object was built from, kept as handed in. it is whichever category's copy
        #the caller happened to have -- a rec_folder normally, an analysis_folder for an archived
        #session -- so it is the right thing to quote back in a message but not the right thing to
        #read raw data from; use recording_folder for that
        self.base_folder = Path(session_folder)
        self.root_folder, _, self.rec_ID = split_category_path(self.base_folder)
        if not self.rec_ID.parts:
            raise ValueError(
                f"{session_folder} names a category root but no session below it; expected "
                f"<root_folder>/{RECORDING_ROOT}/<an_ID>/<rec_ID>."
            )
        self.an_ID = self.rec_ID.parts[0]
        self.create = bool(create)
        if self.create:
            self.create_output_folders()

    def __repr__(self):
        made = [c for c in (SORTING_ROOT, ANALYSIS_ROOT) if self._raw_category_folder(c).is_dir()]
        state = ', '.join(made) if made else 'no outputs yet'
        if not self.has_recording:
            state += '; archived, no raw data'
        return f"SessionPaths({self.rec_ID.as_posix()} under {self.root_folder}; {state})"

    def _raw_category_folder(self, category):
        """Where a category's folder would be, whether or not anything is there."""
        if category not in CATEGORY_ROOTS:
            raise ValueError(f"category must be one of {CATEGORY_ROOTS}, got {category}")
        return self.root_folder / category / self.rec_ID

    @property
    def has_recording(self):
        """Whether this session's raw acquisition data is still on this drive.

        False means the session has been archived -- recording_files (and normally sorting_files
        with it) moved off to reclaim space, leaving only analysis_files behind. Everything
        derived from the raw data is then unavailable, but the analysis products are all still
        there, so posthoc work runs unchanged.
        """
        return self._raw_category_folder(RECORDING_ROOT).is_dir()

    @property
    def has_sorting(self):
        """Whether this session's sorting outputs are still on this drive.

        False either because the session has not been sorted yet or because it has been archived;
        has_recording is what tells those apart.
        """
        return self._raw_category_folder(SORTING_ROOT).is_dir()

    def _creatable(self, category):
        """Whether a missing `category` folder for this session may be made on demand.

        Only sorting_files is ever withheld, and only from an archived session: with the raw data
        gone there is nothing left to sort, so making the folder would rebuild part of the very
        tree that was deleted to reclaim the space -- and would leave behind an empty sorting
        folder that later reads as "sorted, but the outputs vanished".

        This gates creation only, never reads. A session whose recording moved out but whose
        sorting stayed keeps working in full: category_folder returns any folder that is actually
        there before it ever consults this, so the half-archived case needs no special casing.
        """
        return category != SORTING_ROOT or self.has_recording

    def category_folder(self, category):
        """This session's folder under the given category root, or None if it is not there yet.

        recording_files always comes back as a path; sorting_files and analysis_files come back
        as None until they exist, or are created first if this object was built with create=True
        -- except that an archived session's sorting folder is never created, see _creatable.
        """
        folder = self._raw_category_folder(category)
        if category == RECORDING_ROOT or folder.is_dir():
            return folder
        if self.create and self._creatable(category):
            folder.mkdir(parents=True, exist_ok=True)
            return folder
        return None

    #the three category folders. raw data is only ever read from recording_folder; everything the
    #pipeline produces goes under sorting_folder or analysis_folder
    @property
    def recording_folder(self):
        return self.category_folder(RECORDING_ROOT)

    @property
    def sorting_folder(self):
        return self.category_folder(SORTING_ROOT)

    @property
    def analysis_folder(self):
        return self.category_folder(ANALYSIS_ROOT)

    @property
    def preprocess_folder(self):
        """The filtered copy of this session's raw signal, beside the raw data, or None if this
        session has not been preprocessed yet.

        It lives in recording_files rather than sorting_files because it is a rendering of the
        acquisition, not a product of any one sorting run: several sortings of a session read the
        same preprocess folder, and it stays valid when their outputs are thrown away.

        Unlike the other folder accessors this one checks for the product itself rather than for
        the category folder around it, since recording_folder is always there -- so
        `folders.preprocess_folder is None` is the test for whether this session has been
        preprocessed. It is also never created, not even under create=True: see the class
        docstring for why an empty preprocess folder is worse than a missing one. A writer names
        its destination as recording_folder / PREPROCESS_FOLDER.
        """
        folder = self.recording_folder / PREPROCESS_FOLDER
        return folder if folder.is_dir() else None

    def sorting_path(self, *names):
        """A named product inside this session's sorting folder, e.g. sorting_path('analyzer').

        None if the session has no sorting folder, so that a missing product and a missing
        folder do not have to be distinguished by the caller.
        """
        folder = self.sorting_folder
        return folder.joinpath(*names) if folder is not None else None

    def analysis_path(self, *names):
        """A named product inside this session's analysis folder, or None if there is no such
        folder yet."""
        folder = self.analysis_folder
        return folder.joinpath(*names) if folder is not None else None

    @property
    def ks_folder(self):
        """This session's KiloSort output, or None if it has not been sorted yet.

        Like preprocess_folder, this checks for the product rather than for the category folder
        around it: the sorting folder is normally created up front, so `folders.ks_folder is
        None` is the test for whether the sorter has actually run. A writer names its
        destination as sorting_path(KS_FOLDER), which is a path whether or not it exists.
        """
        folder = self.sorting_path(KS_FOLDER)
        return folder if folder is not None and folder.is_dir() else None

    @property
    def pynapple_folder(self):
        """Curation products, event tables and posthoc plots all live here."""
        return self.analysis_path(PYNAPPLE_FOLDER)

    @property
    def input_map(self):
        """This session's acquisition channel-map file, beside its raw data.

        The name depends on the rig -- nidaq_map.txt on the SpikeGLX rig, intan_map.txt on the
        Intan one -- so whichever the session actually has is returned. A session with neither
        gets None, so `if folders.input_map is not None` is the test for whether this session
        came with a map at all, rather than an .exists() check on a speculative path.
        """
        rec_folder = self.recording_folder
        for name in INPUT_MAP_FILES:
            candidate = rec_folder / name
            if candidate.exists():
                return candidate
        return None

    @property
    def event_table(self):
        """The behaviour computer's event table for this session, beside its raw data, or None.

        Named <session>_eventTable.csv; failing that, the folder's only *_eventTable.csv is taken,
        so a session renamed after the behaviour program wrote it is still found. More than one
        candidate and no exact name is ambiguous, and gets None rather than a guess.
        """
        rec_folder = self.recording_folder
        exact = rec_folder / f"{self.rec_ID.name}{EVENT_TABLE_SUFFIX}"
        if exact.exists():
            return exact
        candidates = sorted(rec_folder.glob(f"*{EVENT_TABLE_SUFFIX}"))
        return candidates[0] if len(candidates) == 1 else None

    def _require_folder(self, category, wanted):
        """The category folder, or a FileNotFoundError naming what was being looked for.

        The folder accessors answer "has this session been sorted yet?", so None is a useful
        answer there. The file accessors are asked for something to open, and None is never a
        useful answer to that -- it just defers the failure to whatever tries to read it, which
        reports a TypeError about NoneType and says nothing about the session. So the file
        accessors come through here instead.
        """
        folder = self.category_folder(category)
        if folder is None:
            #what to do about it differs, so say which case this is. suggesting create=True to an
            #archived session would be advice that quietly does nothing: _creatable refuses to
            #rebuild a sorting folder whose raw data is gone
            if not self._creatable(category):
                remedy = ("This session has been archived -- its raw data is gone, and its "
                          f"{category} went with it -- so nothing can be written there. Its "
                          f"{ANALYSIS_ROOT} products are still reachable.")
            else:
                remedy = ("It has probably not been through that stage of the pipeline yet; "
                          "pass create=True to SessionPaths if you are about to write there.")
            raise FileNotFoundError(
                f"cannot give a path to {wanted}: session {self.rec_ID.as_posix()} has no "
                f"{category} folder ({self._raw_category_folder(category)} does not exist). "
                f"{remedy}")
        return folder

    def binary_signals_folder(self, category=ANALYSIS_ROOT):
        """The binary_signals folder, which exists in both sorting and analysis files, or None
        if that category folder has not been made yet."""
        folder = self.category_folder(category)
        return folder / BINARY_SIGNALS_FOLDER if folder is not None else None

    def binary_signals_file(self, category=ANALYSIS_ROOT):
        """The extracted binary signals file.

        Raises FileNotFoundError if this session has no such category folder, rather than
        handing back a None that would only surface later as a TypeError inside whatever tried
        to open it. See _require_folder for why the file accessors raise where the folder ones
        return None.
        """
        folder = self._require_folder(category, f"{BINARY_SIGNALS_FOLDER}/{BINARY_SIGNALS_FILE}")
        return folder / BINARY_SIGNALS_FOLDER / BINARY_SIGNALS_FILE

    #the raw acquisition folders. unlike everything above, these are found by looking at the
    #filesystem rather than derived from the layout -- hence methods, not properties -- but they
    #always search recording_folder, so they resolve the same whichever category folder this
    #object was built from
    @property
    def spikeglx_folders(self):
        """Every SpikeGLX run folder (ANID_g#) in this session's recording folder, sorted by name."""
        rec_folder = self.recording_folder
        if not rec_folder.is_dir():
            raise FileNotFoundError(
                f"No recording folder for session {self.rec_ID.as_posix()}: {rec_folder} does "
                f"not exist. This session has probably been archived -- its raw data moved off "
                f"this drive -- in which case only its analysis products can be reached. Check "
                f"folders.has_recording before asking for anything derived from the raw signal.")
        return sorted((f for f in rec_folder.iterdir() if f.is_dir() and '_g' in f.name),
                      key=lambda f: f.name)

    def spikeglx_folder(self, gnum=None):
        """This session's SpikeGLX run folder (ANID_g#), inside recording_files.

        gnum picks a run when the session recorded more than one (gnum=0 -> the _g0 folder). With
        several runs and no gnum this raises, naming the candidates, rather than asking on the
        console -- so it stays safe inside a batch run. parse_nidq.get_spikeglx_folder is the
        interactive wrapper for when a human is there to answer.
        """
        candidates = self.spikeglx_folders
        if gnum is not None:
            matches = [f for f in candidates if f"_g{gnum}" in f.name]
            if not matches:
                raise FileNotFoundError(
                    f"No SpikeGLX folder matching _g{gnum} in {self.recording_folder} "
                    f"(found {[f.name for f in candidates]})")
            return matches[0]
        if not candidates:
            raise FileNotFoundError(f"No SpikeGLX folder found in {self.recording_folder}")
        if len(candidates) > 1:
            raise ValueError(
                f"Multiple SpikeGLX folders in {self.recording_folder}: "
                f"{[f.name for f in candidates]}. Pass gnum to choose one.")
        return candidates[0]

    def imec_folder(self, gnum=None, fail_quietly=False):
        """The folder holding this session's imec[#].ap.bin / imec[#].ap.meta pair."""
        return find_imec_folder(self.spikeglx_folder(gnum), fail_quietly=fail_quietly)

    def create_output_folders(self):
        """Create the sorting and analysis folders (and their parents) for this session.

        Equivalent to having constructed with create=True; use whichever reads better at the
        call site. Goes through the raw paths, since the checked ones are None precisely when
        there is something to create.

        An archived session gets only its analysis folder: see _creatable for why the sorting
        folder is not rebuilt once the raw data it belongs to is gone.
        """
        if self._creatable(SORTING_ROOT):
            self._raw_category_folder(SORTING_ROOT).mkdir(parents=True, exist_ok=True)
        self._raw_category_folder(ANALYSIS_ROOT).mkdir(parents=True, exist_ok=True)
        return self

    def mirror_binary_signals(self):
        """Keep the sorting-side copy of binary_signals.npz in step with the analysis-side one.

        Sorting and analysis both need the extracted binary signals, so the file is duplicated
        rather than shared. The analysis copy is the one the parsers write, and this is called on
        every binary-signals run -- freshly parsed or loaded from cache -- so the sorting copy is
        never left behind.
        """
        #raw paths on both sides. this is the one place that must tolerate either folder being
        #absent: it is called on every binary-signals run to top up a copy that may simply not be
        #due yet, so a missing folder is a "nothing to do", not the error the file accessors raise
        source = (self._raw_category_folder(ANALYSIS_ROOT)
                  / BINARY_SIGNALS_FOLDER / BINARY_SIGNALS_FILE)
        dest = (self._raw_category_folder(SORTING_ROOT)
                / BINARY_SIGNALS_FOLDER / BINARY_SIGNALS_FILE)
        if not source.exists():
            return None
        #the session folder itself, not dest.parent: a sorted session that has never had binary
        #signals copied over is missing only the binary_signals/ subfolder and should still get
        #it, while an archived one is missing the whole sorting folder and must not have it made
        if not self.has_sorting:
            return None
        if dest.exists() and dest.stat().st_mtime >= source.stat().st_mtime:
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        return dest