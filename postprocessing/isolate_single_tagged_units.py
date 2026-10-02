import re
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

    kst_df, cox_df, figpath = pot.plot_time_to_first_spike_distribution(
        ephys, binary_signals, save_folder=pynapple_folder)
    mwu_df, mwu_summary_df = pot.test_stim_firing(ephys, binary_signals, save_folder=pynapple_folder)

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
    ttfs = ephys_metadata['ttfs_classif']
    firing = ephys_metadata['firing_classif']
    orange = ephys_metadata['orange_response'].fillna(False).astype(bool)

    # -- cell type: the time-to-first-spike label decides; the firing test fills in where it is
    #  silent. Both apply the opsins' asymmetric selectivity (parse_opto_tagging.
    #  classify_opsin_response): ChR2 is close to blind to orange light, so a unit the latency
    #  test calls dSPN but that the firing test finds driven by orange light -- by its peak rate,
    #  which the latency test can miss in a unit that fires late or adapts -- is driven by both
    cell_type = ttfs.where(ttfs.notna(), firing)
    cell_type = cell_type.mask((cell_type == 'dSPN') & orange, 'muSPN')
    ephys_metadata['cell_type'] = cell_type

    # a manually curated good unit was judged tagged by eye; with no automatic label it is muSPN
    cell_type = cell_type.mask(manual_ok & cell_type.isna(), 'muSPN')

    ephys_metadata['final_classif'] = cell_type.where(ks_ok & ~manual_bad)

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

    # subset TsGroup to good units, attach enriched metadata, and save
    sing_units = ephys[singU_metadata.index.tolist()]
    new_cols = ['ttfs_classif', 'firing_classif', 'orange_response', 'cell_type', 'final_classif']
    sing_units.set_info(singU_metadata[[c for c in new_cols if c in singU_metadata.columns]])
    out_file = pynapple_folder / 'sing_tagged_units'
    sing_units.save(str(out_file))
    print(f"[{tag}] saved {len(sing_units)} single units "
          f"({len(best_tagged)} tagged, {len(poss_fn)} possible false negatives) to {out_file}")

    # -- the same metadata as a csv, beside the npz --
    #  the npz above is already written, so a csv that cannot be opened (usually because the last
    #  run's copy is still open in Excel) is reported and does not throw the session's work away
    csv_file = out_file.with_name('singU_metadata.csv')
    try:
        singU_metadata.rename_axis('unit').to_csv(csv_file)
        print(f"[{tag}] wrote metadata for {len(singU_metadata)} units to {csv_file}")
    except OSError as e:
        print(f"[{tag}] could not write {csv_file}: {type(e).__name__}: {e}")

    return {'ephys_metadata': ephys_metadata, 'singU_metadata': singU_metadata,
            'best_tagged': best_tagged, 'poss_fn': poss_fn}


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
        #Path(r"D:\experiment_data\analysis_files\DS37\DS37_031926"),
        #Path(r"C:\Users\assad\Desktop\experiment_data\analysis_files\DS45\DS45_20260814"),
        #Path(r"C:\Users\assad\Desktop\experiment_data\analysis_files\DS45\DS45_20260811"),
        #Path(r"C:\Users\assad\Desktop\experiment_data\analysis_files\DS15\DS15_20260703"),
        Path(r"D:\experiment_data\analysis_files\DS23\DS23_20260208"),
        Path(r"D:\experiment_data\analysis_files\DS23\DS23_20260209"),
        Path(r"D:\experiment_data\analysis_files\DS23\DS23_20260210"),
        Path(r"D:\experiment_data\analysis_files\DS23\DS23_20260211")
    ]
    results = isolate_batch(analysis_folders)
