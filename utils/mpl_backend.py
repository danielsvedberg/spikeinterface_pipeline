"""Pick a matplotlib backend that matches the process we are running in.

Interactive figures only make sense in the process the user is sitting in front
of.  joblib/loky spawns worker processes that re-import the plotting modules;
bringing up a Qt canvas there has no event loop behind it, and on Windows that
surfaces as a sporadic access violation in whatever native code the worker
happens to be running at the time (often somewhere entirely unrelated, such as
a numba kernel doing the next large allocation).  Workers only ever save
figures to disk, so hand them the headless backend instead.

Call `select_backend()` at import time in any module that both draws figures
and is imported inside a worker, in place of a bare `matplotlib.use(...)`.
"""

import multiprocessing

import matplotlib


def in_worker_process():
    """True when this process was spawned by multiprocessing/loky.

    Two windows have to be covered, because module imports happen in both:

    - while the child is unpickling its setup payload, before
      ``BaseProcess._bootstrap`` has recorded the parent.  ``spawn.prepare()``
      re-imports ``__main__`` here, so a pipeline script pulls in its plotting
      modules during this window.  The stdlib marks it with ``_inheriting``.
    - after ``_bootstrap``, when the pool unpickles the actual task and imports
      the module holding it.  ``parent_process()`` is set by then.
    """
    return (multiprocessing.parent_process() is not None
            or getattr(multiprocessing.current_process(), "_inheriting", False))


def select_backend(interactive="Qt5Agg", headless="Agg"):
    """Use `interactive` in the main process and `headless` in workers."""
    backend = headless if in_worker_process() else interactive
    matplotlib.use(backend)
    return backend
