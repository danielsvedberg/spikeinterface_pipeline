"""Re-parse sessions' binary_signals with the current parser, backing up and diffing each.

    python -m scripts.reparse_binary_signals <backup_root> <session_folder> [...] [--jobs=N]

--jobs=N re-parses N sessions at once, each in its own process (default 1). A session's parse
is mostly single-threaded and its file copies are I/O, so several at a time use the machine
far better; output from parallel sessions interleaves, each line prefixed with its session.

For each session the current binary_signals.npz is copied to
<backup_root>/<animal>/<session>/binary_signals_<tag>.npz (one copy: the analysis-side file
and its sorting-side mirror are checked to be byte-identical first; if they differ both are
kept), then the session is re-parsed (get_binary_signals(overwrite=True)) and every event whose
times changed is reported. A session that fails is logged and the rest still run.
"""
import filecmp
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pynapple as nap

from preprocessing import parse_nidq as pni
from utils import file_structure as fs

TAG = 'pre_tonefreq'


def _events(tsg):
    return {n: np.asarray(tsg[k].t) for k, n in zip(tsg.keys(), tsg.metadata['event'])}


def _copy(src, dst):
    """Copy under a temporary name, renamed only when complete."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    tmp = dst.with_suffix('.partial')
    shutil.copy2(src, tmp)
    tmp.rename(dst)


def reparse(base, backup_root, tag=TAG):
    paths = fs.SessionPaths(base)
    primary = paths.binary_signals_file()
    mirror = paths.binary_signals_file(fs.SORTING_ROOT)
    _, _, rel = fs.split_category_path(base)
    dest = Path(backup_root) / rel
    _copy(primary, dest / f'binary_signals_{tag}.npz')
    if mirror is not None and mirror.exists() and not filecmp.cmp(primary, mirror, shallow=False):
        _copy(mirror, dest / f'binary_signals_{tag}_sorting_copy.npz')
    old = _events(nap.load_file(dest / f'binary_signals_{tag}.npz'))
    # the clock alignment is recovered from the saved file rather than recomputed: computing it
    # scans the whole AP binary (66-99 GB), which is what made re-parses slow and memory-bound
    if not (paths.binary_signals_folder() / pni.ALIGNMENT_FILE).exists():
        pni.seed_alignment_from_saved(base)
    new = _events(pni.get_binary_signals(base, overwrite=True, plot=True))
    changed = {}
    for n in sorted(set(old) | set(new)):
        a, b = old.get(n, np.array([])), new.get(n, np.array([]))
        if a.size != b.size:
            changed[n] = f'{a.size} -> {b.size}'
        elif a.size and not np.array_equal(a, b):
            d = np.abs(a - b)
            changed[n] = f'{int((d > 1e-6).sum())} times moved (max {d.max() * 1e3:.2f} ms)'
    return changed


def _run_one(session, backup_root):
    """One session, for the process pool: its result line (never raises)."""
    import matplotlib
    matplotlib.use('Agg')
    s, t0 = Path(session), time.time()
    try:
        ch = reparse(s, backup_root)
        return f'=== {s.name}: ok, {(time.time() - t0) / 60:.1f} min; changed: {ch or "nothing"}'
    except Exception as e:
        return f'=== {s.name}: FAILED {e!r}\n{traceback.format_exc()}'


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--jobs=')]
    jobs = next((int(a.split('=', 1)[1]) for a in sys.argv[1:] if a.startswith('--jobs=')), 1)
    backup_root, sessions = Path(args[0]), args[1:]
    if jobs <= 1:
        for s in sessions:
            print(_run_one(s, backup_root), flush=True)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(_run_one, s, backup_root) for s in sessions]
            for f in as_completed(futures):
                print(f.result(), flush=True)
