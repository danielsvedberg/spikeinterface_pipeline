"""Reorganize to the file_structure.py layout and keep only the latest (heavy Kilosort 4.1.7) sorts
(user, 2026-10-09): sorting_files and analysis_files on C: (C:\\experiment_data), recording_files on D:
(D:\\experiment_data\\recording_files, junctioned in as C:\\experiment_data\\recording_files so one root
serves SessionPaths). Old sorts, old preprocessing and their products are deleted. Raw data (anything in
an <animal>_g# folder, any SpikeGLX .bin/.meta) is never deleted: rmtree_guarded refuses it.

    python reorganize.py <phase> [--dry]
phases: layout     C: internal moves of the 17 latest sessions into C:\\experiment_data (instant)
        recordings new-chain DS15/DS2 preprocess (+ saturation) C: -> D:, verified, old ones deleted
        old        the 17 sessions' old D: sorting/analysis deleted (binary_signals checked first),
                   the 4 animals' other sorting/analysis folders moved to C:
        others     the other animals' sorting_files / analysis_files D: -> C:
        cleanup    C:\\si_stage scratch layout
"""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

C_ROOT = Path(r'C:\experiment_data')
D_ROOT = Path(r'D:\experiment_data')
STAGE = Path(r'C:\si_stage')
RESORT = STAGE / 'resort'
SCRATCH_ANA = STAGE / 'root' / 'analysis_files'
LATEST = ['DS15_20260701', 'DS15_20260702', 'DS15_20260703', 'DS15_20260707', 'DS2_20260505',
          'DS2_20260506', 'DS2_20260511', 'DS45_20260811', 'DS45_20260812', 'DS45_20260814',
          'DS45_20260819', 'DS45_20260820', 'DS45_20260821', 'DS48_20260921', 'DS48_20260922',
          'DS48_20260923', 'DS48_20260924']
RESORT_NAME = {'DS48_20260922': 'DS48_20260922_ks417'}
NEW_CHAIN_ON_C = [s for s in LATEST if s.startswith(('DS15', 'DS2_'))]
SORT_MOVES = {'ks': 'kilosort4_output', 'analyzer': 'analyzer', 'analyzer_clean': 'analyzer_clean',
              'spike_cleaning': 'spike_cleaning', 'sorting_summary_clean': 'sorting_summary_clean',
              'comparison.json': 'comparison_with_previous_sort.json',
              'tagged_comparison.csv': 'tagged_comparison_with_previous_sort.csv'}
RAW_FILE = re.compile(r'\.(ap|lf|nidq)\.(bin|meta)$', re.I)
RAW_DIR = re.compile(r'_g\d+$', re.I)
DRY = '--dry' in sys.argv
LOG = Path(__file__).parent / 'reorganize.log'


def log(msg):
    print(msg, flush=True)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')


def is_junction(p):
    try:
        return bool(os.readlink(p))
    except OSError:
        return False


def unlink_junction(p):
    """Remove a directory junction itself, never its target."""
    if not DRY:
        subprocess.run(['cmd', '/c', 'rmdir', str(p)], check=True, capture_output=True)
    log(f'  junction removed {p}')


def make_junction(link, target):
    if not DRY:
        subprocess.run(['cmd', '/c', 'mklink', '/J', str(link), str(target)], check=True, capture_output=True)
    log(f'  junction {link} -> {target}')


def raw_inside(p):
    p = Path(p)
    if any(RAW_DIR.search(part) for part in p.parts):
        return str(p)
    for dirpath, dirnames, filenames in os.walk(p):
        # never descend through junctions: their contents are somewhere else
        dirnames[:] = [d for d in dirnames if not is_junction(Path(dirpath) / d)]
        for d in dirnames:
            if RAW_DIR.search(d):
                return str(Path(dirpath) / d)
        for f in filenames:
            if RAW_FILE.search(f):
                return str(Path(dirpath) / f)
    return None


def rmtree_guarded(p):
    p = Path(p)
    if not p.exists() and not is_junction(p):
        return
    if is_junction(p):
        unlink_junction(p)
        return
    bad = raw_inside(p)
    if bad:
        raise RuntimeError(f'REFUSED to delete {p}: raw data inside ({bad})')
    if not DRY:
        for dirpath, dirnames, _ in os.walk(p, topdown=False):
            for d in dirnames:
                q = Path(dirpath) / d
                if is_junction(q):
                    unlink_junction(q)
        shutil.rmtree(p)
    log(f'  deleted {p}')


def tree_stats(p):
    p = Path(p)
    if p.is_file():
        return 1, p.stat().st_size
    files = [Path(dp) / f for dp, _, fs in os.walk(p) for f in fs]
    return len(files), sum(f.stat().st_size for f in files)


def move_same_drive(src, dst):
    if not src.exists():
        return
    if dst.exists():
        raise RuntimeError(f'{dst} exists already; not moving {src}')
    if not DRY:
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.rename(src, dst)
    log(f'  moved {src} -> {dst}')


def copy_verified(src, dst):
    """Copy a tree to another drive and check file count and bytes; returns True when they match."""
    if DRY:
        log(f'  would copy {src} -> {dst}')
        return True
    if dst.exists():
        raise RuntimeError(f'{dst} exists already')
    shutil.copytree(src, dst, symlinks=True)
    a, b = tree_stats(src), tree_stats(dst)
    log(f'  copied {src} -> {dst}: {a[0]} files {a[1] / 1e9:.2f} GB; copy {b[0]} files {b[1] / 1e9:.2f} GB')
    return a == b


def move_cross_drive(src, dst):
    if not src.exists():
        return
    if src.is_file():
        if not DRY:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            if dst.stat().st_size != src.stat().st_size:
                raise RuntimeError(f'copy of {src} does not match; source kept')
            src.unlink()
        log(f'  moved file {src} -> {dst}')
        return
    if copy_verified(src, dst):
        rmtree_guarded(src)
    else:
        raise RuntimeError(f'copy of {src} does not match; source kept')


def phase_layout():
    for s in LATEST:
        a, name = s.split('_')[0], RESORT_NAME.get(s, s)
        log(f'=== layout {s}')
        sort_dst = C_ROOT / 'sorting_files' / a / s
        for old, new in SORT_MOVES.items():
            move_same_drive(RESORT / name / old, sort_dst / new)
        ana_dst = C_ROOT / 'analysis_files' / a / s
        move_same_drive(SCRATCH_ANA / a / name, ana_dst)
        # the sorting folder's binary_signals copy (file_structure: a mirror of the primary)
        if not DRY and (ana_dst / 'binary_signals').exists() and not (sort_dst / 'binary_signals').exists():
            shutil.copytree(ana_dst / 'binary_signals', sort_dst / 'binary_signals')
            log(f'  binary_signals mirror -> {sort_dst}')
    link = C_ROOT / 'recording_files'
    if not link.exists():
        make_junction(link, D_ROOT / 'recording_files')


def phase_recordings():
    for s in NEW_CHAIN_ON_C:
        a = s.split('_')[0]
        log(f'=== recording {s}')
        rec = D_ROOT / 'recording_files' / a / s
        new = STAGE / s / 'preprocess_new'
        if not (new / 'traces_cached_seg0.raw').exists():
            log('  no new-chain copy on C:, skipped')
            continue
        tmp = rec / 'preprocess_newchain_incoming'
        if not copy_verified(new, tmp):
            raise RuntimeError(f'{tmp} does not match {new}; nothing deleted')
        rmtree_guarded(rec / 'preprocess')          # the old-chain preprocessing
        if not DRY:
            os.rename(tmp, rec / 'preprocess')
        log(f'  {tmp.name} -> preprocess')
        sat_new, sat_old = STAGE / s / 'saturation', rec / 'saturation'
        if sat_new.exists():
            rmtree_guarded(sat_old)
            move_cross_drive(sat_new, sat_old)
        rmtree_guarded(new)
        if not DRY and STAGE.joinpath(s).exists() and not any(STAGE.joinpath(s).iterdir()):
            STAGE.joinpath(s).rmdir()


def same_files(src, dst):
    """Every file of src is in dst with the same size."""
    for dp, _, fs in os.walk(src):
        for f in fs:
            q = Path(dst) / Path(dp).relative_to(src) / f
            if not q.exists() or q.stat().st_size != (Path(dp) / f).stat().st_size:
                return False
    return True


def phase_old():
    animals = sorted({s.split('_')[0] for s in LATEST})
    for s in LATEST:
        a = s.split('_')[0]
        log(f'=== old products {s}')
        d_ana, c_ana = D_ROOT / 'analysis_files' / a / s, C_ROOT / 'analysis_files' / a / s
        if d_ana.exists():
            if (d_ana / 'binary_signals').exists() and not same_files(d_ana / 'binary_signals', c_ana / 'binary_signals'):
                # keep anything the C: copy lacks (e.g. backups) before the D: folder goes
                for dp, _, fs in os.walk(d_ana / 'binary_signals'):
                    for f in fs:
                        q = c_ana / 'binary_signals' / Path(dp).relative_to(d_ana / 'binary_signals') / f
                        if not q.exists():
                            if not DRY:
                                q.parent.mkdir(parents=True, exist_ok=True)
                                shutil.copy2(Path(dp) / f, q)
                            log(f'  kept {q.name} (missing from the C: binary_signals)')
            rmtree_guarded(d_ana)
        rmtree_guarded(D_ROOT / 'sorting_files' / a / s)
    for a in animals:
        log(f'=== {a}: subject-level old products and unsorted sessions')
        rmtree_guarded(D_ROOT / 'analysis_files' / a / 'compendium')
        for cat in ('sorting_files', 'analysis_files'):
            src = D_ROOT / cat / a
            if not src.exists():
                continue
            for item in sorted(src.iterdir()):
                move_cross_drive(item, C_ROOT / cat / a / item.name)
            if not DRY and not any(src.iterdir()):
                src.rmdir()


def phase_others():
    for cat in ('sorting_files', 'analysis_files'):
        for src in sorted((D_ROOT / cat).iterdir()):
            free = shutil.disk_usage('C:\\').free
            n, size = tree_stats(src)
            log(f'=== {cat}/{src.name}: {n} files {size / 1e9:.1f} GB; C: free {free / 1e9:.0f} GB')
            if size > free - 50e9:
                log('  not enough room on C: (keeping 50 GB free); left on D:')
                continue
            move_cross_drive(src, C_ROOT / cat / src.name)


def phase_cleanup():
    for d in (STAGE / 'root' / 'sorting_files', STAGE / 'root' / 'recording_files'):
        if d.exists():
            for dp, dirnames, _ in os.walk(d, topdown=False):
                for x in dirnames:
                    q = Path(dp) / x
                    if is_junction(q):
                        unlink_junction(q)
            rmtree_guarded(d)
    log('=== cross-session tables and the old-vs-new tagging review')
    for f in sorted(SCRATCH_ANA.glob('*.csv')):
        move_same_drive(f, C_ROOT / 'analysis_files' / f.name)
    move_same_drive(RESORT / 'tagging_review', C_ROOT / 'analysis_files' / 'tagging_review_vs_previous_sort')
    rmtree_guarded(RESORT / 'DS48_20260922')     # the Kilosort 4.1.4 test sort
    for d in (SCRATCH_ANA, STAGE / 'root', STAGE / 'claims', *sorted(RESORT.glob('*'))):
        if d.exists():
            left = [str(p) for p in d.rglob('*') if p.is_file()]
            if left:
                log(f'  {d} still holds {len(left)} files, kept: {left[:5]}')
            else:
                rmtree_guarded(d)


if __name__ == '__main__':
    phase = sys.argv[1]
    log(f'##### {phase}{" (dry run)" if DRY else ""}')
    {'layout': phase_layout, 'recordings': phase_recordings, 'old': phase_old,
     'others': phase_others, 'cleanup': phase_cleanup}[phase]()
    log(f'##### {phase} done')
