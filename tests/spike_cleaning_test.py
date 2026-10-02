"""Test postprocessing.spike_cleaning on one session, template arm against order arm.

    python -m tests.spike_cleaning_test smoke|full [session_folder] [output_tag]

Never calls run_pipeline: its load-or-resort fallback would re-sort, deleting the
Kilosort folder, if a load failed. The session's existing pre-merge 'analyzer' is
opened read-only and every product is written to new names.

Two arms, identical except for how refractory pairs are resolved:
    template  drop the spike of each close pair that fits its unit worse
    order     drop the later spike (remove_duplicated_spikes' keep_first_iterative)
Both get the same outlier removal, the same (now actually running) merges and the same
censored refractory metrics, so a difference between them is the pair rule's alone.

smoke runs ~20 units -- the ones inspected by hand, a NaN-location unit, a high-rate
duplicate-heavy unit, a small unit and the neighbours of two tagged units, so merges
have candidates -- in memory, with audits only. full runs every unit and saves both
clean analyzers (analyzer_clean_v2, analyzer_clean_v2_order).

Caveat, stated in the report: the existing analyzer was built after the old 0.3 ms
order-based dedup, which removed 5,865 of 11,187,349 spikes (0.05%). The pipeline now
builds its analyzer from the raw sort; that one-line change is not exercised here.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import spikeinterface.full as si

from postprocessing import spike_cleaning as scl
from preprocessing import get_stimulation_frames as gsf
from utils import file_structure as fs

DEFAULT_FOLDER = Path(r"D:\experiment_data\recording_files\DS45\DS45_20260811")
HAND_CHECKED = [28, 27, 19, 320, 271]
# two separate clusters in one unit (a waveform and noise crossings): must be split
BIMODAL_UNIT = 146
NAN_UNIT, DUPLICATE_UNIT = 325, 416


def tagged_units(analyzer, iv, ratio=5.0, min_rate=10.0):
    """Units firing >= ratio x their pre-pulse rate and >= min_rate Hz during the pulses."""
    fs_ = analyzer.sampling_frequency
    dur = float((iv[:, 1] - iv[:, 0]).sum())
    pre = np.column_stack([iv[:, 0] - 1.0, iv[:, 0] - 0.9])
    out = {}
    for u, s in scl._unit_samples(analyzer).items():
        t = s / fs_
        n_in = int(np.sum(_inside(t, iv)))
        n_pre = int(np.sum(_inside(t, pre)))
        r_in, r_pre = n_in / dur, n_pre / (0.1 * len(iv))
        if r_in >= min_rate and (r_in + 0.1) / (r_pre + 0.1) >= ratio:
            out[u] = r_in / max(r_pre, 0.1)
    return out


def _inside(t, iv):
    k = np.searchsorted(iv[:, 0], t, side='right') - 1
    return (k >= 0) & (t <= iv[np.clip(k, 0, None), 1])


def check_alignment(before, after, names=('spike_amplitudes', 'spike_locations',
                                           'amplitude_scalings')):
    """Every per-spike value of `after` equals the input's for the same spike.

    A spike is identified by (sample, Kilosort unit); a merged unit's spike by its
    sample and any of its parents. The values themselves are never recomputed by the
    cleaning, only carried, so any mismatch is a spike that took another's value.
    Returns {extension: number of mismatched spikes}.
    """
    b_samples = scl._unit_samples(before)
    sv_b = before.sorting.to_spike_vector()
    ids_b = list(before.unit_ids)
    sv_a = after.sorting.to_spike_vector()
    ks = dict(zip(after.unit_ids, after.sorting.get_property('ks_unit_ids')))
    out = {}
    for name in names:
        vb, va = before.get_extension(name).get_data(), after.get_extension(name).get_data()
        # the input's values per unit, in each unit's time order
        order_b = np.argsort(sv_b['unit_index'], kind='stable')
        bounds = np.searchsorted(sv_b['unit_index'][order_b], np.arange(len(ids_b) + 1))
        per_unit = {u: vb[order_b[bounds[k]:bounds[k + 1]]] for k, u in enumerate(ids_b)}
        bad = 0
        for k, u in enumerate(after.unit_ids):
            rows = np.flatnonzero(sv_a['unit_index'] == k)
            s, v = sv_a['sample_index'][rows], va[rows]
            found = np.zeros(rows.size, bool)
            for p in str(ks[u]).split('+'):
                p = type(ids_b[0])(str(p).split('/')[0])
                sp, vp = b_samples[p], per_unit[p]
                j = np.clip(np.searchsorted(sp, s), 0, sp.size - 1)
                same = (sp[j] == s) & (vp[j] == v if vp.dtype.names is None else
                                       np.all([vp[j][f] == v[f] for f in vp.dtype.names], axis=0))
                found |= same
            bad += int((~found).sum())
        out[name] = bad
    return out


def check_templates(before, after, min_parent_corr=0.8):
    """Each unit's template is its own: its waveforms' mean, and like its parent's.

    Two failure modes, both silent: a template that is not the average of the unit's
    own stored waveforms (templates and waveforms out of step), and one that does not
    resemble the template of the Kilosort unit it came from (unit-indexed data landing
    on the wrong unit -- a selection that reorders units does exactly this). Dropping a
    few percent of a unit's spikes cannot turn its template into another unit's, so a
    single-parent unit whose template correlates below min_parent_corr with its
    parent's is a bookkeeping error. Returns a dict of counts and the worst offenders.
    """
    ta, tb = after.get_extension('templates'), before.get_extension('templates')
    wf = after.get_extension('waveforms')
    ks = dict(zip(after.unit_ids, after.sorting.get_property('ks_unit_ids')))
    ids_b = list(before.unit_ids)
    own_bad, parent_low, split_halves = [], [], []
    for u in after.unit_ids:
        t = ta.get_unit_template(u, operator='average')
        w = wf.get_waveforms_one_unit(u, force_dense=True)
        if w.shape[0] and not np.allclose(t, w.mean(axis=0), atol=1e-3, rtol=1e-3):
            own_bad.append(u)
        if '/' in str(ks[u]):
            # a bimodal split half is not meant to resemble its parent's template, which
            # averaged both halves; its own consistency is still checked above
            split_halves.append(u)
            continue
        parents = [type(ids_b[0])(p) for p in str(ks[u]).split('+')]
        if len(parents) == 1:
            ref = tb.get_unit_template(parents[0], operator='average')
        else:
            # a merged unit's template is its parents' templates weighted by spike count;
            # a random-spike sample that is not uniform over the merged unit breaks this
            # while still passing the waveform-mean check above
            n = before.sorting.count_num_spikes_per_unit()
            w = np.array([n[p] for p in parents], dtype=float)
            ref = sum(wi * tb.get_unit_template(p, operator='average')
                      for wi, p in zip(w, parents)) / w.sum()
        r = np.corrcoef(t.ravel(), ref.ravel())[0, 1]
        if not r >= min_parent_corr:
            parent_low.append((u, ks[u], round(float(r), 3)))
    return {'template_not_mean_of_own_waveforms': len(own_bad),
            'template_corr_to_parents_below_%g' % min_parent_corr: len(parent_low),
            'split_halves_not_compared_to_parent': len(split_halves),
            'examples': (own_bad[:5], parent_low[:5])}


def _final_labels(audit_folder):
    """{input unit: final bombcell label}, a merged unit's label for each of its parents."""
    ids = pd.read_csv(Path(audit_folder) / 'unit_ids.csv')
    return {int(p): row.bombcell_label + (f' (merged {row.ks_unit_ids})' if '+' in str(row.ks_unit_ids) else '')
            for row in ids.itertuples() for p in str(row.ks_unit_ids).split('+')}


def smoke_units(analyzer, tagged, noise_examples=()):
    """The hand-checked units plus edge cases plus spatial neighbours for merges."""
    counts = analyzer.sorting.count_num_spikes_per_unit()
    small = next(u for u, n in sorted(counts.items(), key=lambda kv: kv[1]) if n >= 50)
    loc = analyzer.get_extension('unit_locations').get_data()[:, :2]
    ids = list(analyzer.unit_ids)
    picked = list(dict.fromkeys(HAND_CHECKED + [NAN_UNIT, DUPLICATE_UNIT, BIMODAL_UNIT, small]
                                + list(noise_examples)))
    for anchor in (28, 27):
        d = np.linalg.norm(loc - loc[ids.index(anchor)], axis=1)
        for j in np.argsort(d)[1:7]:
            if ids[j] not in picked:
                picked.append(ids[j])
    return picked[:25], small


def summarize(arm_reports, finals, before_labels, tagged, out):
    """Stage counts per arm, and the tagged units' labels before and after."""
    lines = ['stage counts', json.dumps(arm_reports, indent=2, default=str), '']
    rows = []
    for arm, a in finals.items():
        audit = pd.read_csv(out / arm / 'outlier_audit.csv', index_col=0)
        qm = a.get_extension('quality_metrics').get_data()
        ks = a.sorting.get_property('ks_unit_ids')
        lab = a.sorting.get_property('bombcell_label')
        for u, k, l in zip(a.unit_ids, ks, lab):
            parts = [int(str(p).split('/')[0]) for p in str(k).split('+')]
            hit = [p for p in parts if p in tagged]
            if not hit:
                continue
            rows.append({'arm': arm, 'unit_id': u, 'ks_unit_ids': k, 'tagged_ks': hit,
                         'label_before': ','.join(str(before_labels.get(p, 'n/a')) for p in hit),
                         'label_after': l, 'rp_contamination': qm.loc[u, 'rp_contamination'],
                         'rp_violations': qm.loc[u, 'rp_violations'], 'snr': qm.loc[u, 'snr'],
                         'amplitude_median': qm.loc[u, 'amplitude_median'],
                         'num_spikes': qm.loc[u, 'num_spikes'],
                         # what dropping stimulation outliers cost the tagging
                         'evoked_spikes': int(sum(audit['n_stim_spikes'].get(p, 0) for p in hit)),
                         'evoked_dropped': int(sum(audit['stim_outliers_dropped'].get(p, 0) for p in hit))})
    table = pd.DataFrame(rows)
    try:
        table.to_csv(out / 'tagged_units_before_after.csv', index=False)
    except PermissionError:  # open in Excel, which locks it
        table.to_csv(out / f'tagged_units_before_after_{time.strftime("%H%M%S")}.csv', index=False)
    lines.append('tagged units (ratio >= 5 during pulses), labels before -> after')
    lines.append(table.to_string(index=False) if len(table) else '(none)')
    (out / 'summary.txt').write_text('\n'.join(lines))
    print('\n'.join(lines))


def main(mode='smoke', base_folder=DEFAULT_FOLDER, tag=None):
    job = dict(n_jobs=-1, chunk_duration='10s', progress_bar=False)
    si.set_global_job_kwargs(**job)
    folders = fs.SessionPaths(base_folder)
    out = folders.sorting_path('spike_cleaning_test') / (tag or mode)
    out.mkdir(parents=True, exist_ok=True)

    analyzer = si.load_sorting_analyzer(folders.sorting_path('analyzer'))
    iv = gsf.get_stimulation_intervals(base_folder)
    tagged = tagged_units(analyzer, iv)
    old = si.load_sorting_analyzer(folders.sorting_path('analyzer_clean'), load_extensions=False)
    before = dict(zip(old.unit_ids, old.sorting.get_property('bombcell_label')))
    print(f'{len(tagged)} tagged units; stim pulses {len(iv)}')

    if mode == 'smoke':
        # units the old pipeline called noise, so the noise-drop path runs
        noise_examples = [u for u in analyzer.unit_ids if u not in set(old.unit_ids)][:3]
        units, small = smoke_units(analyzer, tagged, noise_examples)
        print(f'smoke units ({len(units)}): {units}  (small unit {small})')
        # in the analyzer's own unit order: a reordered select_units gives every unit
        # another unit's template (see spike_cleaning.select_units_aligned)
        units = [u for u in analyzer.unit_ids if u in set(units)]
        analyzer = scl.select_units_aligned(analyzer, units)

    # step 1's scoring is identical for both arms and is the slow part: once, cached
    t0 = time.time()
    outliers = scl.score_outliers(analyzer, iv, cache_file=out / 'outlier_scores.pkl', **job)
    print(f'outlier scoring: {time.time() - t0:.0f} s')
    reports, finals = {}, {}
    for arm in ('template', 'order'):
        t0 = time.time()
        clean, curated, rep = scl.clean_and_curate(
            analyzer, stim_intervals=iv, method=arm, audit_folder=out / arm,
            outlier_result=outliers, **job)
        rep['runtime_s'] = round(time.time() - t0)
        rep['misaligned_per_spike_values'] = mis = check_alignment(analyzer, curated)
        print(f'{arm}: per-spike values checked against the input by (sample, ks unit): {mis}')
        if any(mis.values()):
            raise AssertionError(f'{arm}: per-spike extension values moved to other spikes: {mis}')
        rep['template_checks'] = tc = check_templates(analyzer, curated)
        print(f'{arm}: template checks {tc}')
        if tc['template_not_mean_of_own_waveforms'] or tc['template_corr_to_parents_below_0.8']:
            raise AssertionError(f'{arm}: templates do not belong to their units: {tc}')
        reports[arm], finals[arm] = rep, curated
        print(f'{arm}: {rep}')
        if mode == 'full':
            name = 'analyzer_clean_v2' + ('' if arm == 'template' else '_order')
            clean.save_as(folder=folders.sorting_path(name), format='binary_folder')
    summarize(reports, finals, before, tagged, out)


if __name__ == '__main__':  # required on Windows: job workers re-import this module
    main(sys.argv[1] if len(sys.argv) > 1 else 'smoke',
         Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_FOLDER,
         sys.argv[3] if len(sys.argv) > 3 else None)
