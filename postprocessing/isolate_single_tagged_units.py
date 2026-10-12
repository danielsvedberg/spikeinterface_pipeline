import re
import shutil
import numpy as np
import pandas as pd
import pynapple as nap
from pathlib import Path
from postprocessing import parse_opto_tagging as pot
from utils import file_structure as fs

# the unit label that decides quality, and the values of it that count as a good unit.
# 'kilosort' is KiloSort's own KSLabel (older sortings spell it ks_label); 'bombcell' is the
# spike-cleaning curation's bombcell_label, where a non-somatic unit that passes every quality
# criterion is as well isolated as a somatic one
QUALITY_LABELS = {'kilosort': (('KSLabel', 'ks_label'), {'good'}),
                  'bombcell': (('bombcell_label',), {'good', 'non_soma_good'})}

# SPNs fire slowly at rest. A tagged unit whose quiet-baseline rate (analysis.opto_baseline:
# every moment with no trial, lick, light, tone or solenoid, +-200 ms) is above
# TAGGED_MAX_BASELINE_HZ is kept as light-driven but not called an SPN: dSPN -> bU, iSPN -> oU,
# muSPN -> mU (blue / orange / multi unit). 8.42 Hz is the equal-posterior boundary of an
# exponential + Gaussian mixture (maximum likelihood, the Gaussian truncated at 0) fitted to the
# quiet-baseline rates of all 292 tagged units (2026-10-06): an exponential of mean 1.94 Hz for
# the slow, SPN-like 70%, and a broad 0 +- 51 Hz tail. Compared with two Gaussians (6.64 Hz,
# AIC 1827), two lognormals (40.1 Hz, AIC 1746, the only fit to the whole distribution, but
# its slow component spans ~0.03-50 Hz) and exponential + lognormal (no SPN / fast split); the
# user chose this one (AIC 1770.5): anything above the dense slow cluster is suspect. Was 7.8 Hz
# (an untruncated 2-Gaussian fit). Scratchpad exp_gauss_cutoff.py; issues.md 22c.
TAGGED_MAX_BASELINE_HZ = 8.42
HIGH_BASELINE_CLASS = {'dSPN': 'bU', 'iSPN': 'oU', 'muSPN': 'mU'}
# significance of the tagging tests (parse_opto_tagging's alpha), and the statistics of each kept
# per unit as fire_<stat> / ttfs_<stat> (the firing test's and the latency model's)
TAG_ALPHA = 0.05
_FIRE_STATS = ['p_chrimson_vs_baseline', 'peak_rho_chrimson', 'peak_rho_p_chrimson', 'resp_chrimson_hz',
               'responds_chrimson', 'p_chr2_vs_baseline', 'peak_rho_chr2', 'peak_rho_p_chr2',
               'resp_chr2_hz', 'responds_chr2', 'p_chrimson_gt_chr2']
_TTFS_STATS = ['coef_power_chrimson', 'p_power_chrimson', 'coef_power_chr2', 'p_power_chr2',
               'lrt_pval', 'delta_coef', 'n_events']


# -- load manual curation: units judged by eye, one label per line --
#     good= [17,164,228,263]
#     bad= [25,37,67]
#  good units are kept whatever KiloSort called them, bad ones dropped whatever it called them.
#  Either line may be missing or empty. The file is hand-typed, so spacing and capitalisation are
#  tolerated and an unrecognised label is reported rather than ignored -- a typo'd 'god= [...]'
#  otherwise curates nothing and looks exactly like an empty list.
def parse_manual_units(path):
    """{label: [unit_id, ...]} for every `label= [id, id, ...]` entry of a manual curation file.

    Scans the whole file rather than line by line so that a long list wrapped across several
    lines still reads as one entry -- which is what a hand-typed list of thirty units turns into.
    """
    labelled = {}
    for label, ids in re.findall(r'(\w+)\s*=\s*\[([^\]]*)\]', Path(path).read_text()):
        labelled[label.lower()] = [int(x) for x in re.findall(r'\d+', ids)]
    return labelled


def _classify(ephys, binary_signals, folders, pynapple_folder, quality_label, tag, plot_folder=None):
    """The tagging tests and the per-unit classification of isolate_single_tagged_units, for the
    units of `ephys` (a TsGroup); the tests' plots go to plot_folder (default pynapple_folder).
    Returns (ephys_metadata, singU_metadata, best_tagged, poss_fn)."""
    plot_folder = Path(plot_folder) if plot_folder is not None else pynapple_folder
    kst_df, cox_df, figpath = pot.plot_time_to_first_spike_distribution(
        ephys, binary_signals, save_folder=plot_folder)
    mwu_df, mwu_summary_df = pot.test_stim_firing(ephys, binary_signals, save_folder=plot_folder)

    ephys_metadata = ephys.metadata.copy()
    # -- merge per-unit classifications into ephys_metadata --
    ephys_metadata = ephys_metadata.join(
        cox_df.set_index('unit')['cell_type'].rename('ttfs_classif'),
        how='left'
    )
    ephys_metadata = ephys_metadata.join(
        mwu_summary_df.set_index('unit')['cell_type'].rename('firing_classif'),
        how='left'
    )
    # whether the firing test finds a real (power-dependent) orange-light response
    ephys_metadata = ephys_metadata.join(
        mwu_summary_df.set_index('unit')['responds_chrimson'].rename('orange_response'),
        how='left'
    )
    # every test statistic behind the classification, for review (tag_diagnostics)
    ephys_metadata = ephys_metadata.join(
        mwu_summary_df.set_index('unit')[[c for c in _FIRE_STATS if c in mwu_summary_df.columns]]
        .add_prefix('fire_'), how='left')
    ephys_metadata = ephys_metadata.join(
        cox_df.set_index('unit')[[c for c in _TTFS_STATS if c in cox_df.columns]].add_prefix('ttfs_'),
        how='left')

    manual_units_path = pynapple_folder / "manual_units.txt"
    manual_good_path = pynapple_folder / "manual_good_units.txt"
    if manual_units_path.exists():
        manual_labels = parse_manual_units(manual_units_path)
        manual_source = manual_units_path.name
    elif manual_good_path.exists():
        #older sessions: a bare list of good IDs, no labels, so everything in it is 'good'
        manual_labels = {'good': [int(x) for x in re.findall(r'\d+', manual_good_path.read_text())]}
        manual_source = manual_good_path.name
    else:
        manual_labels = {}
        manual_source = "no manual curation file"

    manual_good_ids = manual_labels.get('good', [])
    manual_bad_ids = manual_labels.get('bad', [])
    manual_ok = ephys_metadata.index.isin(manual_good_ids)
    manual_bad = ephys_metadata.index.isin(manual_bad_ids)

    #the index dtype is printed because isin() silently matches nothing when the IDs are ints and the
    #index is strings -- which looks identical to "the curation file did nothing"
    print(f"[{tag}] {manual_source}: {len(manual_good_ids)} good {manual_good_ids}, "
          f"{len(manual_bad_ids)} bad {manual_bad_ids}")
    print(f"[{tag}]   matched {manual_ok.sum()} good and {manual_bad.sum()} bad units in "
          f"ephys_metadata (index dtype: {ephys_metadata.index.dtype})")
    unknown_labels = [k for k in manual_labels if k not in ('good', 'bad')]
    if unknown_labels:
        print(f"[{tag}]   WARNING: ignoring unrecognised label(s) {unknown_labels}; "
              f"expected 'good' and 'bad'")
    both_labelled = sorted(set(manual_good_ids) & set(manual_bad_ids))
    if both_labelled:
        print(f"[{tag}]   WARNING: units {both_labelled} are listed as both good and bad; bad wins")

    # -- quality: label good/empty OR manually curated --
    #  resolved per session: older and newer sortings spell the KiloSort label column differently
    label_cols, good_values = QUALITY_LABELS[quality_label]
    ks_col = next((c for c in label_cols if c in ephys_metadata.columns), None)
    print(f"[{tag}] quality label: {ks_col or 'none (every unit counts as good)'}, good = {sorted(good_values)}")
    if ks_col is not None:
        ks_ok = (ephys_metadata[ks_col].isin(good_values) | ephys_metadata[ks_col].isna()
                 | (ephys_metadata[ks_col] == '') | manual_ok)
    else:
        ks_ok = pd.Series(True, index=ephys_metadata.index)

    # A manual rejection is absolute: it beats a good KiloSort label and a good automatic
    # classification alike. It is subtracted from every selection below rather than folded into
    # ks_ok, because ks_ok is also read inverted (~ks_ok = "KiloSort called this bad"), so putting a
    # manual rejection inside it would pull those units into poss_fn instead of out of it.

    spn_types = {'iSPN', 'dSPN'}

    # -- cell type: which opsins drive the unit is the firing test's call alone (a power-
    #  dependent rise of the peak rate during the pulse, parse_opto_tagging.test_stim_firing):
    #  a latency (time-to-first-spike) effect on its own made units with poor light responses
    #  tagged (user decision, 2026-10-08; until then the latency label decided and the firing
    #  test only filled in). Driven by both, the unit is iSPN when its orange response is the
    #  stronger by either test -- the firing test's orange > blue response, or the latency
    #  model's steeper orange slope -- and muSPN otherwise (classify_opsin_response: Chrimson
    #  answers some blue light, ChR2 next to no orange light)
    def _sig(col):
        x = ephys_metadata.get(col)
        return (x < TAG_ALPHA).fillna(False) if x is not None else pd.Series(False, index=ephys_metadata.index)
    fire_orange = ephys_metadata.get('fire_responds_chrimson', pd.Series(False, index=ephys_metadata.index))
    fire_blue = ephys_metadata.get('fire_responds_chr2', pd.Series(False, index=ephys_metadata.index))
    fire_orange, fire_blue = fire_orange.fillna(False).astype(bool), fire_blue.fillna(False).astype(bool)
    orange_stronger = _sig('fire_p_chrimson_gt_chr2') | (
        _sig('ttfs_lrt_pval') & (ephys_metadata.get('ttfs_delta_coef', pd.Series(np.nan, index=ephys_metadata.index)) > 0))
    cell_type = pd.Series([pot.classify_opsin_response(o, b, s) for o, b, s
                           in zip(fire_orange, fire_blue, orange_stronger)],
                          index=ephys_metadata.index, dtype=object)
    ephys_metadata['cell_type'] = cell_type

    # a manually curated good unit was judged tagged by eye; with no automatic label it is muSPN
    cell_type = cell_type.mask(manual_ok & cell_type.isna(), 'muSPN')

    ephys_metadata['final_classif'] = cell_type.where(ks_ok & ~manual_bad)

    # -- quiet-baseline rate; a fast tagged unit is no SPN (TAGGED_MAX_BASELINE_HZ) --
    #  tag_classif keeps the opsin-based label, final_classif carries the relabelled one
    from analysis.opto_baseline import quiet_intervals
    sat = (folders.recording_folder / 'saturation' / 'saturation.npz') if folders.has_recording else None
    quiet = quiet_intervals(binary_signals, sat)
    quiet_s = float((quiet[:, 1] - quiet[:, 0]).sum())
    ephys_metadata['baseline_hz'] = [
        (int((np.searchsorted(t, quiet[:, 1]) - np.searchsorted(t, quiet[:, 0])).sum()) / quiet_s
         if quiet_s > 0 else np.nan)
        for t in (np.asarray(ephys[u].index, float) for u in ephys_metadata.index)]
    ephys_metadata['tag_classif'] = ephys_metadata['final_classif']
    fast = ephys_metadata['baseline_hz'] > TAGGED_MAX_BASELINE_HZ
    ephys_metadata['final_classif'] = ephys_metadata['final_classif'].where(
        ~fast, ephys_metadata['final_classif'].map(HIGH_BASELINE_CLASS))
    # the light response of every unit whatever its quality (final_classif is good units only),
    # with the same fast-unit relabelling
    light = cell_type.where(~manual_bad)
    ephys_metadata['light_classif'] = light.where(~fast, light.map(HIGH_BASELINE_CLASS))
    print(f"[{tag}] quiet baseline: {quiet_s:.0f} s; tagged units above "
          f"{TAGGED_MAX_BASELINE_HZ:g} Hz relabelled: "
          f"{ephys_metadata.loc[fast & ephys_metadata['tag_classif'].notna(), 'final_classif'].value_counts().to_dict()}")

    # tagged iSPN/dSPN excluded by the quality label (possible false negatives). A manually
    # rejected unit is not a false negative: it was looked at and turned down
    poss_fn = ephys_metadata[cell_type.isin(spn_types) & ~ks_ok & ~manual_bad].copy()

    # backstop for the invariant: a manual rejection holds whatever any rule above assigned
    ephys_metadata.loc[manual_bad, 'final_classif'] = pd.NA

    # units with a cell type that pass the quality check
    best_tagged = ephys_metadata[ephys_metadata['final_classif'].notna()].copy()

    # all good-quality units regardless of classification (KS label or manual override)
    ks_good = (ephys_metadata[ks_col].isin(good_values) if ks_col is not None
               else pd.Series(True, index=ephys_metadata.index))
    singU_metadata = ephys_metadata[(ks_good | manual_ok) & ~manual_bad].copy()


    return ephys_metadata, singU_metadata, best_tagged, poss_fn


def isolate_single_tagged_units(analysis_folder, quality_label='kilosort'):
    """Classify and save the single tagged units of one session's analysis folder.

    Works from the analysis folder alone, so it also runs on archived sessions whose recording
    and sorting folders have been moved off the drive. Writes `sing_tagged_units.npz` and, beside
    it, `singU_metadata.csv` into the session's pynapple folder, and returns the frames it was
    built from.

    Parameters
    ----------
    analysis_folder : str or Path
        A session analysis folder, e.g. .../analysis_files/DS37/DS37_031926
    quality_label : {'kilosort', 'bombcell'}
        Which unit label decides quality (see QUALITY_LABELS). A session without that label's
        column treats every unit as good, as before.

    Returns
    -------
    dict with keys 'ephys_metadata' (every unit, with the three classification columns),
    'singU_metadata' (the good-quality units whose metadata was attached to the saved TsGroup),
    'best_tagged' (units carrying a final_classif) and 'poss_fn' (possible false negatives).
    The saved TsGroup itself is not returned -- it holds the spike times, which a batch would
    then retain for every session, and it is already on disk as sing_tagged_units.npz.
    """
    analysis_folder = Path(analysis_folder)
    tag = analysis_folder.name  # prefixes every print so a batch of sessions stays readable

    folders = fs.SessionPaths(analysis_folder)
    pynapple_folder = folders.pynapple_folder
    ephys_file = pynapple_folder / "spikes.npz"
    binary_signals_file = folders.binary_signals_file()

    ephys = nap.load_file(ephys_file)
    binary_signals = nap.load_file(binary_signals_file)

    ephys_metadata, singU_metadata, best_tagged, poss_fn = _classify(
        ephys, binary_signals, folders, pynapple_folder, quality_label, tag)

    # subset TsGroup to good units, attach enriched metadata, and save
    sing_units = ephys[singU_metadata.index.tolist()]
    new_cols = ['ttfs_classif', 'firing_classif', 'orange_response', 'cell_type', 'tag_classif',
                'baseline_hz', 'final_classif']
    sing_units.set_info(singU_metadata[[c for c in new_cols if c in singU_metadata.columns]])
    out_file = pynapple_folder / 'sing_tagged_units'
    sing_units.save(str(out_file))
    print(f"[{tag}] saved {len(sing_units)} single units "
          f"({len(best_tagged)} tagged, {len(poss_fn)} possible false negatives) to {out_file}")
    # after the npz: tag_all_units takes a table older than it for stale
    _write_all_units(ephys_metadata, singU_metadata.index, pynapple_folder, tag)

    # -- the same metadata as a csv, beside the npz --
    #  the npz above is already written, so a csv that cannot be opened (usually because the last
    #  run's copy is still open in Excel) is reported and does not throw the session's work away
    csv_file = out_file.with_name('singU_metadata.csv')
    try:
        singU_metadata.rename_axis('unit').to_csv(csv_file)
        print(f"[{tag}] wrote metadata for {len(singU_metadata)} units to {csv_file}")
    except OSError as e:
        alt = csv_file.with_name('singU_metadata_new.csv')
        singU_metadata.rename_axis('unit').to_csv(alt)
        print(f"[{tag}] could not write {csv_file} ({type(e).__name__}: {e}); wrote {alt} instead")

    return {'ephys_metadata': ephys_metadata, 'singU_metadata': singU_metadata,
            'best_tagged': best_tagged, 'poss_fn': poss_fn}


# the columns isolate_single_tagged_units adds to the single units' metadata
_TAG_COLUMNS = ['ttfs_classif', 'firing_classif', 'orange_response', 'cell_type', 'tag_classif',
                'baseline_hz', 'final_classif']


def tag_new_single_units(analysis_folder, quality_label='bombcell', backup_suffix='_prev'):
    """Bring a session's saved single units up to date after its quality labels changed, without
    re-running the tagging tests on the units that were already single units.

    The labels are read from pynapple/spikes.npz (refresh them first: refresh_quality_labels).
    Units that are now good but are not in sing_tagged_units get the tagging tests and the
    classification of isolate_single_tagged_units, run on them alone, and are added; units that
    are no longer good are removed. The previous npz and csv are kept with backup_suffix.
    Returns {'added': [...], 'removed': [...], 'added_classes': {...}}.
    """
    analysis_folder = Path(analysis_folder)
    tag = analysis_folder.name
    folders = fs.SessionPaths(analysis_folder)
    pyn = folders.pynapple_folder
    ephys = nap.load_file(pyn / 'spikes.npz')
    binary_signals = nap.load_file(folders.binary_signals_file())
    npz = pyn / 'sing_tagged_units.npz'
    old = nap.load_file(npz)
    old_md = old.metadata.copy()
    label_cols, good_values = QUALITY_LABELS[quality_label]
    col = next(c for c in label_cols if c in ephys.metadata.columns)
    manual = parse_manual_units(pyn / 'manual_units.txt') if (pyn / 'manual_units.txt').exists() else {}
    good = set(ephys.metadata.index[ephys.metadata[col].isin(good_values)]) | set(manual.get('good', []))
    good -= set(manual.get('bad', []))
    added = sorted(good - set(old_md.index))
    removed = sorted(set(old_md.index) - good)
    keep = [u for u in old_md.index if u not in set(removed)]
    md = old_md.loc[keep]
    # the stored quality label of the units already there follows the refreshed one
    md[col] = ephys.metadata.loc[keep, col]
    added_classes = {}
    if added:
        print(f"[{tag}] tagging {len(added)} new single units: {added}")
        _, new_md, _, _ = _classify(ephys[added], binary_signals, folders, pyn, quality_label, tag)
        added_classes = new_md['final_classif'].value_counts().to_dict()
        md = pd.concat([md, new_md[[c for c in md.columns if c in new_md.columns]]])
    for f in (npz, npz.with_name('singU_metadata.csv')):
        if f.exists():
            shutil.copy2(f, f.with_name(f.stem + backup_suffix + f.suffix))
    md = md.sort_index()
    units = ephys[md.index.tolist()]
    # a TsGroup subset comes back in its own index order, which set_info requires
    units.set_info(md.loc[units.index, [c for c in _TAG_COLUMNS if c in md.columns]])
    units.save(str(npz.with_suffix('')))
    csv = npz.with_name('singU_metadata.csv')
    full = ephys.metadata.loc[md.index].copy()
    for c in _TAG_COLUMNS:
        if c in md.columns:
            full[c] = md[c]
    try:
        full.rename_axis('unit').to_csv(csv)
    except OSError:
        full.rename_axis('unit').to_csv(csv.with_name('singU_metadata_new.csv'))
        print(f"[{tag}] {csv} is locked; wrote singU_metadata_new.csv")
    print(f"[{tag}] single units: +{len(added)} {added_classes}, -{len(removed)}")
    return {'added': added, 'removed': removed, 'added_classes': added_classes}


# every unit's tagging result, whatever its quality (pynapple folder), and where tag_all_units
# puts the tests' plots of the units that are not single units
ALL_UNITS_CSV = 'all_units_tagging.csv'
ALL_UNITS_PLOT_FOLDER = 'all_units_tagging'


def _write_all_units(ephys_metadata, single_ids, pynapple_folder, tag):
    """Write every unit's quality labels and tagging columns to ALL_UNITS_CSV; returns the table."""
    quality = [c for cols, _ in QUALITY_LABELS.values() for c in cols if c in ephys_metadata.columns]
    cols = quality + [c for c in _TAG_COLUMNS + ['light_classif'] if c in ephys_metadata.columns]
    cols += [c for c in ephys_metadata.columns
             if c.startswith(('fire_', 'ttfs_')) and c != 'ttfs_classif' and c not in cols]
    table = ephys_metadata[cols].copy()
    table['single_unit'] = table.index.isin(list(single_ids))
    table = table.sort_index().rename_axis('unit')
    csv = Path(pynapple_folder) / ALL_UNITS_CSV
    try:
        table.to_csv(csv)
    except OSError:
        csv = csv.with_name(csv.stem + '_new.csv')
        table.to_csv(csv)
        print(f"[{tag}] {ALL_UNITS_CSV} is locked; wrote {csv.name}")
    print(f"[{tag}] light-responsive units, all qualities: "
          f"{table['light_classif'].value_counts().to_dict()} -> {csv.name}")
    return table


def tag_all_units(analysis_folder, quality_label='bombcell', overwrite=False):
    """Every unit's tagging result, whatever its quality: the table of ALL_UNITS_CSV, indexed by
    unit, with `light_classif` (the tagging class, fast units relabelled bU / oU / mU, for any
    quality) and `single_unit`.

    isolate_single_tagged_units writes the table. For a session tagged before it did, the
    single units' results are taken from sing_tagged_units and the tagging tests are run on the
    other units alone (their plots go to pynapple/ALL_UNITS_PLOT_FOLDER, so the single units'
    summaries are left alone); the tests are per unit, so this gives what a full run would.
    A table older than spikes.npz or sing_tagged_units.npz is rebuilt."""
    analysis_folder = Path(analysis_folder)
    tag = analysis_folder.name
    folders = fs.SessionPaths(analysis_folder)
    pyn = folders.pynapple_folder
    csv = pyn / ALL_UNITS_CSV
    npz, spikes = pyn / 'sing_tagged_units.npz', pyn / 'spikes.npz'
    if (csv.exists() and not overwrite
            and csv.stat().st_mtime >= max(npz.stat().st_mtime, spikes.stat().st_mtime)):
        return pd.read_csv(csv, index_col='unit')
    ephys = nap.load_file(spikes)
    singles = nap.load_file(npz).metadata
    rest = [u for u in ephys.metadata.index if u not in set(singles.index)]
    # a single unit is good, so its tagging class is its final class
    tags = [singles[[c for c in _TAG_COLUMNS if c in singles.columns]]
            .assign(light_classif=singles['final_classif'])]
    if rest:
        print(f"[{tag}] tagging the {len(rest)} units that are not single units")
        rest_md, _, _, _ = _classify(ephys[rest], nap.load_file(folders.binary_signals_file()), folders,
                                     pyn, quality_label, tag, plot_folder=pyn / ALL_UNITS_PLOT_FOLDER)
        tags.append(rest_md[[c for c in _TAG_COLUMNS + ['light_classif'] if c in rest_md.columns]])
    md = ephys.metadata.drop(columns=_TAG_COLUMNS + ['light_classif'], errors='ignore').join(pd.concat(tags))
    return _write_all_units(md, singles.index, pyn, tag)


def ispn_false_negatives(analysis_folder, unit_labels=None):
    """Every unit with any sign of an orange-light (Chrimson) response that is not a final iSPN,
    with each step of the classification that kept it from being one (user, 2026-10-08: iSPN
    yield matters most). Signs: the firing test's peak rate above its pre-pulse baseline, or
    rising with power, or the latency model's significant positive orange coefficient.

    Reasons, all that apply: the firing test's two conditions (above baseline, rising with
    power) not both met; driven by blue light as well without the orange response being the
    stronger (muSPN); a quiet-baseline rate over TAGGED_MAX_BASELINE_HZ (oU / mU); not a single
    unit (its quality label, and the bombcell criteria it failed when unit_labels -- the
    cleaning's unit_labels.csv -- is given). `n_reasons` 1 is a near miss: one step alone.
    Needs the test statistics in ALL_UNITS_CSV (isolate_single_tagged_units writes them).
    Writes pynapple/ispn_false_negatives.csv and returns the table."""
    folders = fs.SessionPaths(analysis_folder)
    t = tag_all_units(analysis_folder)
    if 'fire_p_chrimson_vs_baseline' not in t.columns:
        raise ValueError(f'{ALL_UNITS_CSV} has no test statistics; rerun isolate_single_tagged_units')
    nan = pd.Series(np.nan, index=t.index)
    a = TAG_ALPHA
    above = (t.get('fire_p_chrimson_vs_baseline', nan) < a)
    rising = (t.get('fire_peak_rho_p_chrimson', nan) < a)
    latency = (t.get('ttfs_p_power_chrimson', nan) < a) & (t.get('ttfs_coef_power_chrimson', nan) > 0)
    cand = t[(above | rising | latency) & (t['light_classif'] != 'iSPN')].copy()
    qcol = next((c for c in ('bombcell_label', 'KSLabel', 'ks_label') if c in t.columns), None)
    failed = {}
    if unit_labels is not None and Path(unit_labels).exists():
        ul = pd.read_csv(unit_labels).set_index('unit_id')
        failed = {u: ', '.join(str(x) for x in ul.loc[u, ['failed_mua', 'failed_non_somatic']]
                               if isinstance(x, str) and x) for u in ul.index}
    reasons = []
    for u, r in cand.iterrows():
        why = []
        if not (above[u] and rising[u]):
            if above[u]:
                why.append(f"orange peak rate above baseline but not rising with power "
                           f"(rho {r.get('fire_peak_rho_chrimson', np.nan):.2f}, P {r.get('fire_peak_rho_p_chrimson', np.nan):.2g})")
            elif rising[u]:
                why.append(f"orange peak rate rises with power but not above baseline "
                           f"(P {r.get('fire_p_chrimson_vs_baseline', np.nan):.2g})")
            else:
                why.append('orange: latency effect only, no rise in firing')
        elif r.get('cell_type') == 'muSPN':
            why.append(f"driven by blue too, orange not the stronger (P orange > blue "
                       f"{r.get('fire_p_chrimson_gt_chr2', np.nan):.2g}) -> muSPN")
        if r['light_classif'] in ('oU', 'mU'):
            why.append(f"quiet-baseline rate {r['baseline_hz']:.1f} Hz > {TAGGED_MAX_BASELINE_HZ:g} -> {r['light_classif']}")
        if not r['single_unit']:
            q = r[qcol] if qcol else 'not single'
            why.append(f"quality {q}" + (f": {failed[u]}" if failed.get(u) else ''))
        reasons.append('; '.join(why))
    cand['reasons'] = reasons
    cand['n_reasons'] = [len(x.split('; ')) if x else 0 for x in reasons]
    first = ['light_classif', qcol, 'single_unit', 'baseline_hz', 'n_reasons', 'reasons']
    cand = cand[[c for c in first if c] + [c for c in cand.columns if c not in first]]
    cand = cand.sort_values(['n_reasons', 'single_unit'], ascending=[True, False])
    out = folders.pynapple_folder / 'ispn_false_negatives.csv'
    try:
        cand.to_csv(out)
    except OSError:
        cand.to_csv(out.with_name('ispn_false_negatives_new.csv'))
    return cand


def refresh_quality_labels(analysis_folder, analyzer_folder, column='bombcell_label', backup_suffix='_prev'):
    """Copy the quality labels of a (relabelled) analyzer into the session's pynapple/spikes.npz.

    Refuses when the npz was not exported from that analyzer (other unit ids or spike counts).
    The previous npz is kept with backup_suffix. Returns the number of labels that changed."""
    import spikeinterface.full as si
    folders = fs.SessionPaths(analysis_folder)
    f = folders.pynapple_folder / 'spikes.npz'
    ephys = nap.load_file(f)
    a = si.load_sorting_analyzer(analyzer_folder, load_extensions=False)
    counts = a.sorting.count_num_spikes_per_unit()
    if sorted(ephys.index) != sorted(a.unit_ids) or any(len(ephys[u]) != counts[u] for u in a.unit_ids):
        raise ValueError(f'{f} was not exported from {analyzer_folder}')
    new = pd.Series(a.sorting.get_property(column), index=a.unit_ids).reindex(ephys.index)
    changed = int((ephys.metadata[column].astype(str) != new.astype(str)).sum())
    if changed:
        shutil.copy2(f, f.with_name(f.stem + backup_suffix + f.suffix))
        ephys.set_info(**{column: new})
        ephys.save(str(f.with_suffix('')))
    return changed


def relabel_fast_units(analysis_folder):
    """Re-apply TAGGED_MAX_BASELINE_HZ to a session's saved single units, without re-running the
    tagging tests: final_classif from the saved tag_classif and baseline_hz. For a change of the
    cutoff only; anything else needs isolate_single_tagged_units. Returns {class: count}."""
    folders = fs.SessionPaths(analysis_folder)
    npz = folders.pynapple_folder / 'sing_tagged_units.npz'
    units = nap.load_file(npz)
    md = units.metadata
    if not {'tag_classif', 'baseline_hz'} <= set(md.columns):
        raise ValueError(f'{npz} has no tag_classif/baseline_hz; run isolate_single_tagged_units')
    tagc = md['tag_classif'].where(md['tag_classif'].isin(list(HIGH_BASELINE_CLASS)))
    fast = md['baseline_hz'].astype(float) > TAGGED_MAX_BASELINE_HZ
    final = tagc.where(~fast, tagc.map(HIGH_BASELINE_CLASS))
    units.set_info(final_classif=final)
    units.save(str(npz.with_suffix('')))
    csv = npz.with_name('singU_metadata.csv')
    if csv.exists():
        t = pd.read_csv(csv, index_col='unit')
        t['final_classif'] = final.reindex(t.index)
        try:
            t.to_csv(csv)
        except OSError:
            t.to_csv(csv.with_name('singU_metadata_new.csv'))
    return final.value_counts().to_dict()


def isolate_batch(analysis_folders, skip_errors=True, quality_label='kilosort'):
    """Run isolate_single_tagged_units over a list of analysis folders.

    Each session writes its own products -- including its singU_metadata.csv -- into its own
    pynapple folder, so nothing is compiled across sessions here.

    Sessions are independent, and an archived one can be missing its pynapple products
    altogether, so by default a failure is reported and the remaining folders still run; the
    exceptions are reported in a summary at the end so nothing fails silently. Pass
    skip_errors=False to stop on the first failure instead.

    Returns
    -------
    {Path: result_dict} for the sessions that succeeded.
    """
    analysis_folders = [Path(f) for f in analysis_folders]
    results, failures = {}, []
    for folder in analysis_folders:
        print(f"\n=== {folder.name} ({folder}) ===")
        try:
            results[folder] = isolate_single_tagged_units(folder, quality_label=quality_label)
        except Exception as e:
            if not skip_errors:
                raise
            failures.append((folder, e))
            print(f"[{folder.name}] FAILED: {type(e).__name__}: {e}")
    print(f"\ndone: {len(results)}/{len(analysis_folders)} sessions succeeded")
    if failures:
        print(f"failed: {[f.name for f, _ in failures]}")
        for folder, e in failures:
            print(f"  {folder.name}: {type(e).__name__}: {e}")
    return results


if __name__ == "__main__":
    #run from the repo root as a module, so that `postprocessing`/`utils` import:
    #    python -m postprocessing.isolate_single_tagged_units
    analysis_folders = [
        #Path(r"C:\experiment_data\analysis_files\DS37\DS37_031926"),
        #Path(r"C:\Users\assad\Desktop\experiment_data\analysis_files\DS45\DS45_20260814"),
        #Path(r"C:\Users\assad\Desktop\experiment_data\analysis_files\DS45\DS45_20260811"),
        #Path(r"C:\Users\assad\Desktop\experiment_data\analysis_files\DS15\DS15_20260703"),
        Path(r"C:\experiment_data\analysis_files\DS23\DS23_20260208"),
        Path(r"C:\experiment_data\analysis_files\DS23\DS23_20260209"),
        Path(r"C:\experiment_data\analysis_files\DS23\DS23_20260210"),
        Path(r"C:\experiment_data\analysis_files\DS23\DS23_20260211")
    ]
    results = isolate_batch(analysis_folders)
