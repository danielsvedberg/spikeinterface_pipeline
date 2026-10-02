# spikeinterface_pipeline

**Before starting any work, read `issues.md`, and keep it current while you work:** add every
new problem the moment you find it (even ones you will not fix now), record progress and
decisions under each issue, and move an issue to *Resolved* only once the user has accepted
the result. The user relies on it so that no issue raised along the way is forgotten.

## What this is

Daniel Svedberg's (Assad lab, HMS) pipeline for Neuropixels recordings of striatum (mostly
VLS) in mice doing a cue -> lick task, with dual-opsin opto-tagging: **ChR2 (blue light) tags
dSPNs, Chrimson (orange/red light) tags iSPNs**; a unit driven by both is **muSPN**.
SpikeInterface 0.104.8 + Kilosort4 for sorting, bombcell (via SpikeInterface) for curation,
pynapple for analysis. Acquisition is SpikeGLX with a NIDQ for task/opto signals (some rigs use
Intan — see `preprocessing/intan_vs_nidq.md`).

## Environment

- Windows 11; use the Bash tool (Git Bash) or PowerShell. Conda env `si_env` (active).
- Run from the repo root as modules: `PYTHONPATH=. python -m scripts.clean_sessions ...`.
- Every script that parallelises (joblib/loky, SpikeInterface job_kwargs) needs an
  `if __name__ == '__main__':` guard — Windows spawns workers that re-import the module.
- Batch runs: `matplotlib.use('Agg')`, and set `PYTHONIOENCODING=utf-8` when redirecting output
  to a file (non-ASCII prints otherwise crash with UnicodeEncodeError).
- Some files are CRLF; preserve line endings when editing (Edit tool, or read/write with
  `newline=''`). Bash heredocs mangle backslash escapes — write a .py file instead.
- Much of the newer code is **untracked in git** (see issues.md); be careful with block edits —
  a range replacement once silently deleted functions from `spike_cleaning.py`. After editing a
  module, run `python -m pyflakes <file>` for undefined names.

## Data layout (`utils/file_structure.py`, `SessionPaths`)

`<root>/{recording_files,sorting_files,analysis_files}/<animal>/<session>/`, e.g.
`DS15/DS15_20260702`. Roots: `C:\Users\assad\Desktop\experiment_data` (DS48, current) and
`D:\experiment_data` (DS15, DS45, DS2, archive). C: is nearly full — keep big outputs on D:.
- recording_files: SpikeGLX `*_g0` (nidq + imec), `nidaq_map.txt` (dinmap/ainmap/optoidx),
  behaviour `.mat`, `<session>_eventTable.csv` (Arduino event log, ms), `preprocess/`
  (filtered recording the sort was run on), `saturation/` (raw saturation scan cache).
- sorting_files: `kilosort4_output/`, `analyzer/` (raw sort), `analyzer_clean/` (cleaned +
  curated), `analyzer_clean_prev/` (previous curation), `spike_cleaning/` (audit + review
  plots), `sorting_summary_clean[_prev]/`, `binary_signals/` (mirror).
- analysis_files: `binary_signals/binary_signals.npz` (primary), `pynapple/` (`spikes.npz`,
  `sing_tagged_units.npz`, `singU_metadata.csv`, PSTH/opto/ttfs/tuning plots,
  `prev_curation/`), `ratio_tests/` (per-session compendium); `<animal>/compendium/`.
`SessionPaths(any category folder)` resolves every other path; posthoc code is handed the
analysis folder.

## Pipeline

1. **Binary signals** — `preprocessing/parse_nidq.get_binary_signals(base, overwrite=...)`:
   NIDQ -> imec time; a pynapple TsGroup whose `metadata['event']` names each event (tones,
   solenoid, `lick_on/off`, `<opsin>_on_<power>`/`_off_<power>` pulses, derived trial events:
   rewarded/early/timeout cues, `*_first_licks`, lick bouts, ITIs). Cached; `overwrite=True`
   re-parses. `lick_source='auto'|'nidq'|'event_table'`: when the NIDQ lick line is empty the
   licks come from the event table (`preprocessing/event_table_licks.py`, CUE_ON-aligned fit).
   Tones are classified by pitch from the tone line's square wave (`parse_tones_by_frequency`,
   `default_tonemap`, optional `tonemap` in nidaq_map.txt); first licks are the last lick
   before the solenoid opening / error-tone onset. The NIDQ -> imec clock fit is cached in
   `binary_signals/nidq_imec_alignment.json`: computing it scans the whole AP binary
   (66–99 GB), so never re-parse many sessions without the cache
   (`seed_alignment_from_saved`); `get_binary_signals(save=False)` parses without writing.
   Batch re-parse: `scripts/reparse_binary_signals.py <backup_root> <sessions> --jobs=3`.
2. **Preprocessing** — `scripts/si_pipeline.build_preprocessing_protocol`; `remove_artifacts`
   is first in the chain (decided after ringing tests); saturation detection
   (`preprocessing/detect_saturation.py`), QC in `preprocessing/preprocess_qc.py`.
3. **Sorting** — Kilosort4 via `si_pipeline.run_pipeline`.
4. **Spike cleaning + curation** — `postprocessing/spike_cleaning.clean_and_curate` (replaced
   `remove_duplicated_spikes`), in order: bimodal PCA splits (GMM, up to 2 levels;
   `split_outlier_spikes`) -> drop bombcell-noise units (waveforms plotted for review) ->
   outlier spikes (robust MCD on PCs + template-match scale + location, log-d2 z > 6,
   stimulation outliers dropped too) -> resolve spike pairs <= 1 ms within a unit (worse
   template match dropped) -> strict merges (presets + cosine >= 0.9 + no cross spikes in
   (1, 1.5] ms) -> pairs again -> metrics with refractory windows censored at 1.0 ms, R = 1.5 ms
   -> bombcell labels with the rp rule (MUA if the 95% lower bound of contamination > 10% and
   > 3 violations). Audit tables (`unit_labels.csv`, `unit_ids.csv`, `removed_spikes.csv`,
   `merges.csv`, ...) and review plots go to `sorting_files/.../spike_cleaning/`.
   - `scripts/clean_sessions.py`: cleaning from an existing analyzer; `--relabel` (re-apply
     labels to analyzer_clean), `--summary` (sorting summary + `rp_review/`).
5. **Analysis** — pynapple export (`to_pynapple_tsgroup`) + `si_pipeline.plot_opto_psth`,
   `plot_psth`, `plot_time_to_first_spike`, `plot_tuning_curves`.
6. **Tagging + single units** — `postprocessing/isolate_single_tagged_units(analysis_folder,
   quality_label='bombcell')` (`'kilosort'` = KSLabel, the old default). Two tests in
   `postprocessing/parse_opto_tagging.py`: time-to-first-spike Cox model (decides; an opsin
   counts only with a significant *positive* power coefficient) and `test_stim_firing` (per
   pulse peak 10-ms rate vs the same statistic just before the pulse; a response must also rise
   with power, which rejects the low-power orange-LED flicker artifact). Shared rule
   `classify_opsin_response`: blue only -> dSPN; orange only -> iSPN; both -> iSPN only if
   orange is significantly stronger (Chrimson is somewhat blue-sensitive), else muSPN (ChR2 is
   blind to orange). A TTFS dSPN with a real orange response becomes muSPN. Manual curation:
   `pynapple/manual_units.txt` (`good= [...]`, `bad= [...]`).
7. **Compendium** — `analysis/compendium_plots.py` (`load_session`: lick-bout modulated units
   only; `plot_compendium`), `analysis/session_compendium_plots.py` (one session ->
   `ratio_tests/`).
8. **Batch** — `scripts/curate_subject.py <recording_files/<animal>> [sessions]` runs steps
   1 (re-parse, backup `binary_signals_pre_tonefix.npz`), 4, summary, 5, 6, 7 per session.

## Rules (from the user and from past mistakes)

- Never call `run_pipeline` from tests or batch re-analysis of sorted sessions: its fallbacks
  can re-sort and delete a sort. Never modify an existing `analyzer`; a new curation is saved as
  `analyzer_clean` with the old one kept as `analyzer_clean_prev`.
- Back up before overwriting any data product (copy to a temporary name, then rename). Check
  the target before deleting or overwriting.
- Do not sort, re-preprocess, or start multi-hour runs without asking; say what a run will
  replace before starting it.
- CSVs the user has open in Excel are locked — write an alternate name and say so.
- SpikeInterface gotchas: `select_units` re-sorts the spike vector, so only order-preserving
  selection is safe for per-spike extensions (`spike_cleaning.select_units_aligned`);
  `metric_params` mutates class defaults (use `_metric_defaults_restored`); `random_spikes`
  caches `_some_spikes`.
- User preferences: elegant, general solutions over hard-coding; repair sample-scale artifacts at
  full resolution; spikes during stimulation are not exempt from cleaning (report evoked-spike
  loss instead); when the user names a data location, use it. Explain results with numbers
  and per-unit evidence; the user checks units by eye in the sorting summaries and PSTHs.
