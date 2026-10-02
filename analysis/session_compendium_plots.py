"""The compendium plots of each session on its own, to see what it contributes to the average.

    python -m analysis.session_compendium_plots <analysis_folder> [<analysis_folder> ...]

Each session is analysed and plotted exactly as analysis/compendium_plots.py does for a pooled
set -- same unit selection, same lickless-trial filter, same plots -- and saved to that session's
analysis folder under ratio_tests/. Sessions are independent: one that fails (e.g. no licks
recorded, so no trials to categorise) is reported and the rest still run.
"""
import shutil
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # a batch over sessions: write the figures, open no windows
import matplotlib.pyplot as plt

from analysis import compendium_plots as cp
from utils import file_structure as fs

RATIO_TESTS_FOLDER = 'ratio_tests'


def plot_session(analysis_folder):
    folders = fs.SessionPaths(analysis_folder)
    out = folders.analysis_path(RATIO_TESTS_FOLDER)
    # the folder holds only this function's figures; a re-run starts it empty, so a figure the
    # session no longer produces (e.g. no tagged SPN left) is not left behind from an old run
    if out.exists():
        shutil.rmtree(out)
    cp.plot_compendium([cp.load_session(folders)], [folders], out)
    plt.close('all')
    return out


def plot_sessions(analysis_folders):
    done, failed = {}, {}
    for f in map(Path, analysis_folders):
        print(f'\n=== {f.name} ===', flush=True)
        try:
            done[f.name] = plot_session(f)
        except Exception as e:
            traceback.print_exc()
            failed[f.name] = f'{type(e).__name__}: {e}'
    print(f'\ndone: {len(done)}/{len(done) + len(failed)} sessions')
    for name, out in done.items():
        print(f'  {name}: {out}')
    for name, err in failed.items():
        print(f'  {name} FAILED: {err}')
    return done, failed


if __name__ == '__main__':
    plot_sessions(sys.argv[1:])
