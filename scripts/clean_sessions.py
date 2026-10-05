"""Run the post-sort spike cleaning on already-sorted sessions, from their existing analyzer.

    python -m scripts.clean_sessions <session_folder> [<session_folder> ...]
    python -m scripts.clean_sessions --relabel <session_folder> [...]

--relabel re-applies the bombcell labels to a session's saved analyzer_clean (after a
change to the labelling rules, e.g. the refractory criterion) without re-cleaning it,
and rewrites unit_ids.csv and unit_labels.csv.

    python -m scripts.clean_sessions --merge-redundant <session_folder> [...]

--merge-redundant merges the duplicate units of a session cleaned before the redundant
merge existed, from its saved analyzer_clean (the old one kept as
analyzer_clean_pre_redundant); see merge_redundant_session.

    python -m scripts.clean_sessions --summary <session_folder> [...]

--summary exports the SpikeInterface sorting summary of analyzer_clean to
sorting_summary_clean (an older one, of the previous curation, is kept as
sorting_summary_clean_prev), and copies the summaries of every unit whose refractory
contamination estimate is over the limit into spike_cleaning/rp_review, named by label
and statistics, for judging the refractory rule by eye.

For each session: the old analyzer_clean is kept as analyzer_clean_prev (a session whose
_prev already exists is skipped rather than risk overwriting it), the existing
'analyzer' is cleaned with postprocessing.spike_cleaning.clean_and_curate, and the
result is saved as analyzer_clean. Audit tables, the outlier-score cache and every
review figure go to the session's spike_cleaning/ folder. No re-sort, no analyzer
rebuild, no pynapple export or plots: this is the cleaning step alone, for sessions
sorted before it existed. One failing session is logged and the rest still run.

Sessions whose analyzer was built after the old 0.3 ms order-based dedup (everything
sorted before the cleaning) are cleaned from that analyzer; the dedup had removed about
0.1-0.2% of their spikes.
"""
import json
import sys
import time
import traceback
from pathlib import Path

import pandas as pd
import spikeinterface.full as si
import spikeinterface.curation as sic

from postprocessing import spike_cleaning as scl
from preprocessing import get_stimulation_frames as gsf
from utils import file_structure as fs


def clean_session(base_folder, job_kwargs):
    folders = fs.SessionPaths(base_folder)
    clean_path = folders.sorting_path('analyzer_clean')
    prev_path = folders.sorting_path('analyzer_clean_prev')
    # a session cleaned before (analyzer_clean_prev = the original curation, analyzer_clean = an
    # earlier cleaning) is re-cleaned with both kept: the earlier cleaning is moved to
    # analyzer_clean_superseded once its replacement is in hand
    superseded = folders.sorting_path('analyzer_clean_superseded')
    reclean = clean_path.exists() and prev_path.exists()
    if reclean and superseded.exists():
        raise RuntimeError(f'{superseded} already exists; not overwriting it -- delete it (an '
                           'earlier re-clean) or move it out of the way first')
    audit = folders.sorting_path('spike_cleaning')
    analyzer = load_analyzer(folders, 'analyzer')
    # the original curation's labels, for the before/after columns of unit_labels.csv
    previous = None
    old_path = prev_path if prev_path.exists() else clean_path
    if old_path.exists():
        old = si.load_sorting_analyzer(old_path, load_extensions=False)
        if old.sorting.get_property('bombcell_label') is not None:
            previous = dict(zip(old.unit_ids, old.sorting.get_property('bombcell_label')))
        old = None
    stim = gsf.get_stimulation_intervals(base_folder)
    print(f'{base_folder.name}: {len(analyzer.unit_ids)} units, '
          f'{analyzer.sorting.to_spike_vector().size} spikes, {len(stim)} stimulation pulses')
    scores = scl.score_outliers(analyzer, stim, cache_file=audit / 'outlier_scores.pkl', **job_kwargs)
    clean, _, report = scl.clean_and_curate(analyzer, stim_intervals=stim, audit_folder=audit,
                                            outlier_result=scores, previous_labels=previous,
                                            **job_kwargs)
    # the old clean analyzer is moved aside only now, with its replacement in hand
    if clean_path.exists():
        clean_path.rename(superseded if reclean else prev_path)
    clean.save_as(folder=clean_path, format='binary_folder')
    return report


def relabel_session(base_folder):
    """Re-apply the bombcell labels to the saved analyzer_clean, in place.

    Only the labels change: the spikes, units and metrics are the cleaning's. The
    labelling can only move units among good / mua (and their non_soma kin) -- the
    units the final pass called noise are not in analyzer_clean and stay noise.
    """
    folders = fs.SessionPaths(base_folder)
    audit = folders.sorting_path('spike_cleaning')
    clean = si.load_sorting_analyzer(folders.sorting_path('analyzer_clean'))
    old = dict(zip(clean.unit_ids, clean.sorting.get_property('bombcell_label')))
    labels = scl._bombcell_with_rp_minimum(clean)
    new = labels.loc[clean.unit_ids, 'bombcell_label'].values
    assert not (new == 'noise').any(), 'relabel turned a kept unit into noise'
    clean.set_sorting_property('bombcell_label', new)
    ids = pd.read_csv(audit / 'unit_ids.csv').set_index('unit_id')
    for col in ('bombcell_label', 'rp_violations', 'rp_contamination', 'rp_contamination_lo',
                'rp_counted', 'rp_demoted'):
        ids.loc[list(clean.unit_ids), col] = labels.loc[clean.unit_ids, col].values
    ids.reset_index().to_csv(audit / 'unit_ids.csv', index=False)
    write_label_table(base_folder, labels=labels)
    changes = {(old[u], l) for u, l in zip(clean.unit_ids, new) if old[u] != l}
    return {'labels': pd.Series(new).value_counts().to_dict(),
            'changed': {f'{a}->{b}': int(sum(1 for u, l in zip(clean.unit_ids, new)
                                               if (old[u], l) == (a, b))) for a, b in changes}}


def merge_redundant_session(base_folder, job_kwargs):
    """Merge the redundant (duplicate) units of a session cleaned before that step existed.

    Runs spike_cleaning.merge_redundant_units on the saved analyzer_clean: only the merged
    units' spikes, templates and metrics change (see its docstring for why no re-clean is
    needed). The old analyzer is kept as analyzer_clean_pre_redundant until the result is
    accepted; the audit gains redundant_merges.csv, removed_spikes_redundant.csv and
    report_redundant.json, and unit_ids.csv / unit_labels.csv are rewritten.
    """
    folders = fs.SessionPaths(base_folder)
    audit = folders.sorting_path('spike_cleaning')
    clean_path = folders.sorting_path('analyzer_clean')
    backup = folders.sorting_path('analyzer_clean_pre_redundant')
    if backup.exists():
        raise RuntimeError(f'{backup} already exists: this session was merged already '
                           '(or a run was interrupted) -- check it before running again')
    clean = load_analyzer(folders, 'analyzer_clean')
    stim = gsf.get_stimulation_intervals(base_folder)
    merged, merges, removed, labels, report = scl.merge_redundant_units(
        clean, stim_intervals=stim, **job_kwargs)
    merges.to_csv(audit / 'redundant_merges.csv', index=False)
    (audit / 'report_redundant.json').write_text(json.dumps(report, indent=2, default=str))
    if merged is clean:
        return report
    removed.to_csv(audit / 'removed_spikes_redundant.csv', index=False)
    # unit_ids.csv: the final-noise rows (units never in analyzer_clean) stay; every unit
    # that was in it is replaced by the merged analyzer's units, so a merged-away member is
    # not mistaken for a noise unit by write_label_table
    old = pd.read_csv(audit / 'unit_ids.csv')
    old = old[~old['unit_id'].isin(clean.unit_ids)]
    cols = ['bombcell_label', 'rp_violations', 'rp_contamination', 'rp_contamination_lo',
            'rp_counted', 'rp_demoted']
    names = dict(zip(merged.unit_ids, merged.sorting.get_property('ks_unit_ids')))
    new = pd.DataFrame({'unit_id': labels.index, 'ks_unit_ids': [names.get(u, '') for u in labels.index],
                        **{c: labels[c].values for c in cols}})
    # the merged analyzer is saved before anything is renamed; the old one is moved aside
    # only with its replacement on disk
    tmp = folders.sorting_path('analyzer_clean_redundant_tmp')
    merged.save_as(folder=tmp, format='binary_folder')
    # Windows will not rename a folder whose files are still mapped: drop every handle on
    # the old analyzer first
    import gc
    clean = merged = None
    gc.collect()
    clean_path.rename(backup)
    tmp.rename(clean_path)
    pd.concat([old, new], ignore_index=True).to_csv(audit / 'unit_ids.csv', index=False)
    write_label_table(base_folder)
    return report


def load_analyzer(folders, name):
    """A sorting analyzer of the session, with its preprocessed recording attached when it has
    none (an analyzer saved with the absolute path of a recording that has since moved, or a
    lazy chain from an older version), after checking it is the recording it was built on."""
    analyzer = si.load_sorting_analyzer(folders.sorting_path(name))
    if not analyzer.has_recording():
        rec = si.load(folders.preprocess_folder)
        assert (rec.get_num_samples() == analyzer.get_num_samples()
                and rec.get_num_channels() == analyzer.get_num_channels()), \
            f'{folders.preprocess_folder} does not match {name}'
        analyzer.set_temporary_recording(rec)
    return analyzer


def write_label_table(base_folder, labels=None):
    """unit_labels.csv for a session cleaned before the cleaning wrote it, from what it saved.

    Built from the saved analyzer_clean (the metrics of every unit it kept), the audit's
    unit_ids.csv and noise_units_dropped.csv, the old labels in analyzer_clean_prev and
    Kilosort's KSLabel in the analyzer. Units the final bombcell pass called noise are
    not in analyzer_clean, so their metrics are gone; they get a row saying so.
    """
    folders = fs.SessionPaths(base_folder)
    audit = folders.sorting_path('spike_cleaning')
    clean = si.load_sorting_analyzer(folders.sorting_path('analyzer_clean'))
    ids = pd.read_csv(audit / 'unit_ids.csv')
    kept = ids[ids['unit_id'].isin(clean.unit_ids)].set_index('unit_id')
    if labels is None:
        if 'rp_contamination_lo' not in kept.columns:  # saved before the confidence rule
            labels = scl._bombcell_with_rp_minimum(clean)
        else:
            labels = kept[['bombcell_label', 'rp_violations', 'rp_contamination_lo', 'rp_counted']]
    prev_path = folders.sorting_path('analyzer_clean_prev')
    previous = None
    if prev_path.exists():
        old = si.load_sorting_analyzer(prev_path, load_extensions=False)
        old_labels = old.sorting.get_property('bombcell_label')
        if old_labels is not None:  # a previous curation saved without labels has nothing to compare
            previous = dict(zip(old.unit_ids, old_labels))
    raw = si.load_sorting_analyzer(folders.sorting_path('analyzer'), load_extensions=False)
    kl = raw.sorting.get_property('KSLabel')
    noise_path = audit / 'noise_units_dropped.csv'
    table = scl.label_table(clean, labels, dict(zip(ids['unit_id'], ids['ks_unit_ids'])),
                            previous_labels=previous,
                            ks_labels=dict(zip(raw.unit_ids, kl)) if kl is not None else None,
                            dropped_noise=pd.read_csv(noise_path) if noise_path.exists() else None)
    final_noise = ids[~ids['unit_id'].isin(clean.unit_ids)]
    if len(final_noise):
        table = pd.concat([table, pd.DataFrame({
            'unit_id': final_noise['unit_id'], 'ks_unit_ids': final_noise['ks_unit_ids'],
            'label': final_noise['bombcell_label'],
            'previous_label': [('+'.join(str(previous.get(int(str(q).split('/')[0]), 'noise'))
                                         for q in str(n).split('+')) if previous else '')
                               for n in final_noise['ks_unit_ids']],
            'failed_noise': '(not recoverable: noise units are not saved in analyzer_clean)'})],
            ignore_index=True)
    try:
        table.to_csv(audit / 'unit_labels.csv', index=False)
    except PermissionError:  # open in Excel
        table.to_csv(audit / 'unit_labels_new.csv', index=False)
        print(f'{audit / "unit_labels.csv"} is locked; wrote unit_labels_new.csv')
    return table


def main(sessions):
    job = dict(n_jobs=-1, chunk_duration='10s', progress_bar=False)
    si.set_global_job_kwargs(**job)
    summary = {}
    for s in sessions:
        s = Path(s)
        t0 = time.time()
        try:
            rep = clean_session(s, job)
            summary[s.name] = {'ok': True, 'minutes': round((time.time() - t0) / 60, 1),
                               **{k: rep[k] for k in ('n_units_in', 'bimodal_splits', 'noise_units_dropped',
                                                      'outliers_dropped', 'pairs_dropped_pre_merge',
                                                      'merges', 'n_units_clean', 'labels')}}
        except Exception as e:
            traceback.print_exc()
            summary[s.name] = {'ok': False, 'error': repr(e)}
        print(f'=== {s.name}: {summary[s.name]}', flush=True)
    print(json.dumps(summary, indent=2, default=str))
    return summary


def export_summary(base_folder):
    import shutil
    folders = fs.SessionPaths(base_folder)
    audit = folders.sorting_path('spike_cleaning')
    out = folders.sorting_path('sorting_summary_clean')
    prev = folders.sorting_path('sorting_summary_clean_prev')
    if out.exists() and not prev.exists():
        out.rename(prev)
    clean = si.load_sorting_analyzer(folders.sorting_path('analyzer_clean'))
    si.export_report(sorting_analyzer=clean, output_folder=out, remove_if_exists=True)
    return fill_rp_review(base_folder, clean)


def fill_rp_review(base_folder, clean=None):
    """spike_cleaning/rp_review: the summaries of the units the refractory rule decides.

    Those are the units over the contamination limit (point estimate) that pass every
    other bombcell criterion -- good or mua on the refractory rule alone. Named
    <label>_unit<id>_nv<violations>_c<estimate>_lo<lower bound>.png, most certain first.
    """
    import shutil
    folders = fs.SessionPaths(base_folder)
    audit = folders.sorting_path('spike_cleaning')
    out = folders.sorting_path('sorting_summary_clean')
    clean = clean or si.load_sorting_analyzer(folders.sorting_path('analyzer_clean'))
    th = scl.default_thresholds()
    metrics = clean.get_metrics_extension_data()
    others = {k: v for k, v in th['mua'].items() if k != 'rp_contamination'}
    ids = pd.read_csv(audit / 'unit_ids.csv').set_index('unit_id')
    ids = ids[ids.index.isin(clean.unit_ids) & (ids['rp_contamination'] > scl.RP_MAX_CONTAMINATION)]
    ids = ids[[scl.failed_criteria(metrics.loc[u], others) == '' for u in ids.index]]
    review = audit / 'rp_review'
    if review.exists():
        shutil.rmtree(review)
    review.mkdir()
    for u, r in ids.sort_values('rp_contamination_lo', ascending=False).iterrows():
        shutil.copy(out / 'units' / f'{u}.png',
                    review / (f"{r['bombcell_label']}_unit{u}_nv{int(r['rp_violations'])}"
                              f"_c{r['rp_contamination']:.3f}_lo{r['rp_contamination_lo']:.3f}.png"))
    ids.to_csv(review / 'rp_review.csv')
    return len(ids)


def relabel(sessions):
    for s in sessions:
        print(f'=== {Path(s).name}: {relabel_session(Path(s))}', flush=True)


if __name__ == '__main__':  # required on Windows: job workers re-import this module
    if sys.argv[1:2] == ['--relabel']:
        relabel(sys.argv[2:])
    elif sys.argv[1:2] == ['--merge-redundant']:
        job = dict(n_jobs=-1, chunk_duration='10s', progress_bar=False)
        si.set_global_job_kwargs(**job)
        for s in sys.argv[2:]:
            t0 = time.time()
            try:
                rep = merge_redundant_session(Path(s), job)
                print(f"=== {Path(s).name}: {rep.get('redundant_merges', 0)} merged of "
                      f"{rep.get('redundant_groups_found', 0)} redundant groups, "
                      f"{rep.get('pairs_dropped', 0)} doubled spikes dropped, "
                      f"{(time.time() - t0) / 60:.1f} min: {rep.get('merged_units', {})}", flush=True)
            except Exception as e:
                traceback.print_exc()
                print(f'=== {Path(s).name}: FAILED {e!r}', flush=True)
    elif sys.argv[1:2] == ['--summary']:
        for s in sys.argv[2:]:
            print(f'=== {Path(s).name}: {export_summary(Path(s))} units in rp_review', flush=True)
    else:
        main(sys.argv[1:])
