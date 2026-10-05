"""Full curation + analysis of every sorted session of one subject, as was done for DS48.

    python -m scripts.curate_subject <recording_files/<subject> folder> [session ...] [--steps=a,b]

--steps runs only the named steps (binary_signals, clean, summary, pynapple, isolate,
ratio_tests); the subject compendium is built only when isolate runs.

Sessions given on the command line are the only ones run AND the only ones pooled into the
subject compendium; to resume a partly done subject, run the rest, then rebuild the compendium
with analysis.compendium_plots.plot_compendium over every session.

Per session, in order (a failing step is logged and that session's later steps are skipped;
the other sessions still run):
  0. binary_signals re-parsed with the current parse_nidq (the reward-tone matching fix),
     the old npz (and its sorting-side mirror) kept as binary_signals_pre_tonefix.npz,
     and every event whose times changed reported
  1. spike cleaning + bombcell curation from the existing analyzer (old analyzer_clean kept
     as analyzer_clean_prev)
  2. sorting summary + rp_review
  3. pynapple: previous curation's outputs -> pynapple/prev_curation, fresh spikes.npz,
     opto PSTH / PSTH / TTFS / tuning curves
  4. isolate_single_tagged_units (bombcell labels)
  5. per-session compendium plots (ratio_tests)
Then the pooled DS15 compendium into analysis_files/DS15/compendium.
"""
import shutil
import sys
import time
import traceback
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
import spikeinterface.full as si
from joblib import Parallel
from spikeinterface.exporters import to_pynapple_tsgroup

from analysis import compendium_plots as cp
from analysis import plot_psth as pp
from analysis import session_compendium_plots as scp
from postprocessing.isolate_single_tagged_units import isolate_single_tagged_units
from preprocessing import parse_nidq as pni
from scripts import clean_sessions as cs
from scripts import si_pipeline as sp
from utils import file_structure as fs

REC = None  # set from the command line
JOB = dict(n_jobs=-1, chunk_duration='10s', progress_bar=False)
OLD_PYNAPPLE = ('opto_psth_plots', 'opto_psth_plots_alt', 'opto_tagging', 'psth_plots', 'spikes.npz',
                'sing_tagged_units.npz', 'singU_metadata.csv')


def _events(tsg):
    return {n: tsg[k].t for k, n in zip(tsg.keys(), tsg.metadata['event'])}


def reparse_binary_signals(base):
    paths = fs.SessionPaths(base)
    f = paths.binary_signals_file()
    mirror = paths.binary_signals_file(fs.SORTING_ROOT)
    backups = []
    for p in (f, mirror):
        if p is not None and p.exists():
            b = p.with_name('binary_signals_pre_tonefix.npz')
            if not b.exists():  # copied under a temporary name, so an interrupted copy is never taken for a backup
                tmp = b.with_suffix('.partial')
                shutil.copy2(p, tmp)
                tmp.rename(b)
            backups.append(b)
    old = _events(nap.load_file(backups[0])) if backups else {}
    new = _events(pni.get_binary_signals(base, overwrite=True, plot=True))
    changed = {n: (len(old.get(n, [])), len(t)) for n, t in new.items()
               if n not in old or len(old[n]) != len(t) or not np.allclose(old[n], t, atol=1e-6)}
    gone = sorted(set(old) - set(new))
    print(f'  binary_signals re-parsed; changed events (old n, new n): {changed or "none"}'
          + (f'; events no longer present: {gone}' if gone else ''), flush=True)
    return changed


def pynapple_analysis(base):
    folders = fs.SessionPaths(base)
    pyn = folders.pynapple_folder
    prev = pyn / 'prev_curation'
    if not prev.exists():
        prev.mkdir(parents=True)
        for name in OLD_PYNAPPLE:
            if (pyn / name).exists():
                shutil.move(str(pyn / name), str(prev / name))
    else:
        # prev_curation already holds the original curation's outputs; what is here now was
        # made from a later curation (e.g. before the redundant merges), so plots of units that
        # no longer exist would linger among the new ones -- clear them, they are regenerated
        for name in OLD_PYNAPPLE + ('stim_firing_plots',):
            q = pyn / name
            try:
                if q.is_dir():
                    shutil.rmtree(q)
                elif q.exists():
                    q.unlink()
            except PermissionError as e:  # a csv left open in Excel; isolate writes a _new copy
                print(f'  could not clear {q} ({e.strerror}); left in place', flush=True)
    clean = si.load_sorting_analyzer(folders.sorting_path('analyzer_clean'))
    to_pynapple_tsgroup(clean, attach_unit_metadata=True).save(pyn / 'spikes.npz')
    for step in (sp.plot_opto_psth, sp.plot_psth, sp.plot_time_to_first_spike, sp.plot_tuning_curves):
        t0 = time.time()
        try:
            step(base)
        except Exception as e:
            # plot_on_events_psth's loky workers can die in pynapple's numba count (an
            # intermittent access violation never seen in-process); redo the step in-process
            if 'TerminatedWorkerError' not in type(e).__name__:
                raise
            print(f'  {step.__name__}: worker crashed ({e!r}); redoing it in-process', flush=True)
            saved = pp.Parallel
            pp.Parallel = lambda *a, **k: Parallel(n_jobs=1)
            try:
                step(base)
            finally:
                pp.Parallel = saved
        plt.close('all')
        print(f'  {step.__name__}: {(time.time() - t0) / 60:.1f} min', flush=True)


STEPS = ('binary_signals', 'clean', 'summary', 'pynapple', 'isolate', 'ratio_tests')


def run_session(name, only=STEPS):
    base = REC / name
    ana = fs.SessionPaths(base).analysis_folder
    steps = [('binary_signals', lambda: reparse_binary_signals(base)),
             ('clean', lambda: cs.clean_session(base, JOB)),
             ('summary', lambda: cs.export_summary(base)),
             ('pynapple', lambda: pynapple_analysis(base)),
             ('isolate', lambda: isolate_single_tagged_units(ana, quality_label='bombcell')),
             ('ratio_tests', lambda: scp.plot_session(ana))]
    out = {}
    for step, fn in steps:
        if step not in only:
            continue
        t0 = time.time()
        print(f'--- {name}: {step}', flush=True)
        try:
            r = fn()
        except Exception as e:
            traceback.print_exc()
            out[step] = f'FAILED {type(e).__name__}: {e}'
            print(f'=== {name}: {step} FAILED: {e!r}', flush=True)
            return out
        if step == 'clean':
            r = {k: r[k] for k in ('n_units_in', 'bimodal_splits', 'noise_units_dropped', 'merges',
                                   'n_units_clean', 'labels')}
        elif step == 'isolate':
            r = {'single_units': len(r['singU_metadata']),
                 'tagged': r['best_tagged']['final_classif'].value_counts().to_dict(),
                 'possible_false_negatives': len(r['poss_fn'])}
        elif step not in ('binary_signals', 'summary'):
            r = None
        out[step] = r
        plt.close('all')
        print(f'=== {name}: {step} ok, {(time.time() - t0) / 60:.1f} min: {r}', flush=True)
    return out


if __name__ == '__main__':
    si.set_global_job_kwargs(**JOB)
    # --steps=binary_signals,clean,summary runs only those steps (in their usual order)
    args = [a for a in sys.argv[1:] if not a.startswith('--steps=')]
    only = next((a.split('=', 1)[1].split(',') for a in sys.argv[1:] if a.startswith('--steps=')),
                list(STEPS))
    unknown = set(only) - set(STEPS)
    if unknown:
        raise SystemExit(f'unknown steps {sorted(unknown)}; choose from {STEPS}')
    REC = Path(args[0])
    # every session with a sort and its analyzer; the unsorted ones are left alone
    sessions = args[1:] or sorted(d.name for d in REC.iterdir() if d.is_dir()
                                  and (fs.category_path(d, fs.SORTING_ROOT) / 'analyzer').exists())
    print(f'=== {REC.name}: sessions {sessions}, steps {only}', flush=True)
    results = {s: run_session(s, only) for s in sessions}
    if 'isolate' not in only:
        for s, r in results.items():
            print(s, r)
        sys.exit(0)
    # the pooled compendium reads sing_tagged_units.npz, so every session whose isolate step ran
    good = [s for s, r in results.items() if 'isolate' in r and not str(r['isolate']).startswith('FAILED')]
    print(f'=== pooled {REC.name} compendium over {good}', flush=True)
    try:
        # a session that cannot be categorised (no licks, no rewards: DS45_20260821) is left out
        # of the compendium rather than failing it
        sessions_data, folders = [], []
        for s in good:
            f = fs.SessionPaths(fs.SessionPaths(REC / s).analysis_folder)
            try:
                sessions_data.append(cp.load_session(f))
                folders.append(f)
            except Exception as e:
                print(f'=== compendium: {s} left out: {type(e).__name__}: {e}', flush=True)
        cp.plot_compendium(sessions_data, folders,
                           fs.category_path(REC, fs.ANALYSIS_ROOT) / 'compendium')
        print('=== compendium ok', flush=True)
    except Exception as e:
        traceback.print_exc()
        print(f'=== compendium FAILED: {e!r}', flush=True)
    for s, r in results.items():
        print(s, r)
