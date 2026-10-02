# Issues

Every problem found while working on the pipeline, with its status and what happens next.
Check this file before starting work, and update it as you go: add new issues the moment they
are found, record progress and decisions under the issue, and move an issue to *Resolved* only
when the user has accepted the result. Dates are when the entry was written or last updated.

Status: **open** (not started) · **investigating** · **proposed** (fix designed, awaiting the
user) · **in progress** · **blocked** (waiting on the user or data) · **resolved**

---

## Open

### 1. Duplicated (redundant) units from Kilosort — applied, awaiting acceptance (2026-10-01)
Kilosort4 emits pairs (and groups) of units that are the same neuron: they share most spikes at
a lag of 0–9 samples. The spike cleaning only resolves close spike pairs *within* a unit, and
the merge presets do not propose these (their cross-correlogram peaks at 0 lag instead of
dipping), so the duplicates survive curation and are counted twice in every analysis.
- DS15_20260702 raw sort: 20 redundant pairs (>= 20% of spikes coincident within 0.4 ms,
  `spikeinterface.curation.find_redundant_units`), incl. a mutual 6-unit group (KS 96, 118,
  124, 179, 184, 186). After cleaning, 5 remain: 94/671 (KS 94/99: every spike of 671 has a
  partner in 94 at a fixed 7–9 sample lag — a time-shifted template), 759/1025 (KS 187/452: lag
  0–1 sample), 303/308, 757/1029, 580/1135.
- Every cleaned session has them (redundant pairs / pairs involving a good unit):
  DS48 0921 1/1, 0922 1/0, 0923 6/2, 0924 3/3; DS15 0701 2/1, 0702 5/4, 0703 1/0, 0707 7/4;
  DS45 0811 6/4.
- Proposed fix: in `spike_cleaning.clean_and_curate`, take `find_redundant_units` groups as
  merge candidates (they pass the strict veto: cosine >= 0.9, no cross spikes in (1, 1.5] ms),
  merge them, and let the existing post-merge pair step (<= 1 ms, worse spike dropped by
  template match) remove the doubled spikes. Unique spikes of either unit are kept.
- 2026-10-01, user approved; asked why a full re-clean would be needed — it is not: every step
  before the merges is per unit, so a saved analyzer_clean only needs the merge and what
  follows it. Implemented (status: in progress):
  - `spike_cleaning.py`: `REDUNDANT` merge preset (last in `MERGE_PRESETS`), candidates from
    `find_redundant_units` (`REDUNDANT_PARAMS`); strict veto with templates compared at the
    best alignment within W (94/671: cosine 0.11 aligned, 0.99 shifted 8 samples) and up to
    `RP_MAX_FREE_VIOLATIONS` (3) cross spikes in (1, 1.5] ms tolerated; the merged unit's
    doubled spikes resolve in favour of the reference (largest) member
    (`_redundant_preference`), so its alignment is kept. Steps 4-5 factored into
    `_resolve_merged_pairs` / `_metrics_and_labels`; `merge_redundant_units()` runs them on a
    saved analyzer_clean.
  - `scripts/clean_sessions.py --merge-redundant <sessions>`: keeps the old analyzer as
    `analyzer_clean_pre_redundant`, writes `redundant_merges.csv`,
    `removed_spikes_redundant.csv`, `report_redundant.json`, rewrites `unit_ids.csv` /
    `unit_labels.csv`. Shared `load_analyzer()` attaches the preprocess recording when needed.
  - In-memory trial on DS15_20260702: 4 merged (94+671 -> exactly 94's 18,277 spikes, template
    r = 1.000 with 94; 759+1025 -> 37,938), 757+1029 refused (9 cross spikes); 57,835 doubled
    spikes dropped, spike accounting exact, no other unit's label changed.
  - DS48_20260921: its one pair (360+835) refused, 40 cross spikes — nothing changed.
  - Applied 2026-10-01 to every cleaned session (`--merge-redundant`; merged / doubled spikes
    dropped; refused with reason):
    DS48 0921 0 (360+835 refused, 40 cross spikes); 0922 1 / 73; 0923 5 / 13,164;
    0924 1 / 4,222 (344+362+388 -> 914 good; 872+910 refused, similarity 0.88);
    DS15 0701 2 / 1,712; 0702 4 / 57,835 (757+1029 refused, 9 cross spikes); 0703 1 / 53;
    0707 3 / 10,825 (842+850 refused, 6 cross spikes; 980+988 refused, 71);
    DS45 0811 4 / 14,164 (incl. 36+39/2+23/2+43/2 -> 1099 good).
    Old analyzers kept as `analyzer_clean_pre_redundant` (~3 GB each; DS48 ones are on C:).
  - Refused pairs (above) are left as two units: either a duplicate that also holds another
    neuron's spikes, or two distinct neurons with overlapping detections. Worth a look by eye.
- Next: refresh the downstream analysis of these sessions together with issue 2; delete the
  `_pre_redundant` backups once the user accepts; future cleanings run the step
  automatically (it is in `MERGE_PRESETS`).

### 2. Opto-tagging classification rewrite — accepted by the user, being applied (2026-10-01)
The user found obvious iSPNs unclassified in DS15_20260702 (e.g. 649, 90). Causes found:
- `test_stim_firing` compared windows of different length: a Spearman of 1-s baseline means vs
  30-ms post counts (biased negative, so the firing test almost never called a unit) and a
  Mann-Whitney of 10-ms baseline bins vs 30-ms post (biased toward significance).
- `final_classif` required both tests to agree; on disagreement a non-muSPN label overrode muSPN
  (621 became dSPN).
- The time-to-first-spike Cox rule ignored the coefficient's sign, so units *inhibited* by blue
  light were called dSPN/muSPN (94, 671, 855, 759, 1025, 795).
- Low orange powers flicker and induce a noise artifact (user, 2026-10-01): responses at the
  lowest orange power that decline with power are untrustworthy (usually MUA).

Implemented (user's rules): firing test = per-pulse peak 10-ms rate from 1 ms to pulse end vs
the same statistic in an equal pre-pulse window (paired); a response must also rise with power;
shared `classify_opsin_response` (blue only -> dSPN; orange only -> iSPN; both -> iSPN only if
orange significantly stronger, else muSPN); Cox counts an opsin only with a positive
coefficient; the time-to-first-spike label decides, a dSPN with a real orange response becomes
muSPN, the firing test fills in where TTFS is silent. Files: `postprocessing/
parse_opto_tagging.py`, `postprocessing/isolate_single_tagged_units.py`.
Dry run on DS15_20260702 (scratch only): good units iSPN 32, 90, 601, 649, 1089; dSPN 167,
749, 905; muSPN 617, 621 — matches the user's calls on 649, 621, 601, 90.
Next: user accepts -> rerun isolate + ratio_tests + compendia for DS48 and DS15 (and DS45_0811),
then resume the paused batch (issue 5).
- 2026-10-01 16:30: the user accepted the tagging rules and the add_all_licks fix and asked to
  run everything: finish the cleaning of DS45 0814/0819/0820/0821 and DS2, then pynapple +
  isolate + ratio_tests for every sorted session and each subject's compendium (issue 5).

### 3. Many tagged units fail quality ("possible false negatives") — open (2026-10-01)
With the new rules DS15_20260702 has 88 tagged units that fail bombcell (79 dSPN, 9 iSPN),
mostly MUA in the id block 581–905. Older sessions show the same pattern (e.g. DS48_20260923:
116). Decide whether these are real multi-unit clusters, splittable units, or over-strict
quality criteria (failed criteria are in `spike_cleaning/unit_labels.csv`). Note: some of them
may be duplicates (issue 1).

### 4. `add_all_licks` reads events by position — fixed, awaiting re-analysis (2026-10-01)
`analysis/pres_plot_utils.py:add_all_licks` takes `binary_signals[1]` and `[2]` as the first
licks; in saved files those are `solenoid_off` and `start_tone_on`. So `all_first_licks` is the
solenoid-close times and `compute_ramp_scores` (used by `compendium_plots.load_session` and
`plot_ramping.py`) is wrong in every session. Fix: look up `rewarded_first_licks` and
`early_first_licks` by name. Waiting for the user's OK (it changes ramp-score results).
- 2026-10-01, checked: keys 1/2 are `solenoid_off` / `start_tone_on` in DS48_20260922 and
  DS15_20260702 (the real first licks are keys 26/27 and 34/35). The 100-ms cue filter removes
  the start tones (each is its own cue), so `all_first_licks` = solenoid-close times (+ any
  start tone without an outcome); error trials are missing. Only `compute_ramp_scores` reads
  it, so the damage is the `first_lick_ramp_score` column (attached by
  `compendium_plots.load_session`, ranked by `plot_ramping.py`); no compendium/ratio_tests
  figure uses it — their lick/outcome alignment comes from `categorize_outcome_trials`
  (outcome times), which is correct. The user aligns on outcome times because the triggering
  lick precedes a lick-triggered outcome by ~0.1–3 ms; `*_first_licks` are exactly those
  triggering licks, so either marker gives the same ramp windows.
- 2026-10-01, fixed (user OK): looks `rewarded_first_licks` / `early_first_licks` up by name
  and raises if either is missing. DS15_20260702: all_first_licks = 31 + 50 first licks, 77
  after the 100-ms cue filter. Ramp scores change in every session once its analysis is
  re-run (issue 5).

### 13. Tone classification by frequency; first licks anchored on outcomes — in progress (2026-10-01)
The user wants first licks aligned exactly on the lick immediately preceding `solenoid_on`
(rewarded; the solenoid is open only 10-30 ms) or the error-tone onset (early). Error tones are
hard: an early lick starts the error tone at once, cutting the start cue off, and the digital
line then shows one long pulse. `parse_tones` classifies by duration and infers the error onset
as pulse offset − 0.20015 s for 0.205–1 s pulses.
Found: the NIDQ tone line (din 3, 25 kHz) carries the speaker's square wave, so frequency is
measurable from rising-edge periods. DS48_20260922: cue ~3.13 kHz (171, 100 ms, shortest 21 ms
= a cut-off cue), error ~440 Hz (86, 200 ms), timeout ~130 Hz (64, 2 s), reward ~5 kHz
(fragmented at 5 samples/period; reward tones are found via the solenoid anyway). Merged
pulses switch 3,125 -> 439 Hz at the true error onset (e.g. after < 20 ms at 873.0 s).
Proposal: classify constant-frequency segments by frequency (exact onsets, no duration
heuristics, cut-off cues of any length), keep the solenoid cross-check for reward tones; check
the frequencies on the DS15/DS45/DS2 rigs first (make them configurable in nidaq_map.txt if they
differ). Then `parse_trials`: rewarded first lick = last lick <= solenoid_on, early first lick
= last lick < error onset; `add_all_licks` reads them by name (issue 4). Awaiting the user.
- 2026-10-01, implemented in `preprocessing/parse_nidq.py` (status: in progress):
  `_tone_segments` + `parse_tones_by_frequency` (each cycle labelled by nearest nominal tone in
  log frequency, within x1.35, 5-cycle running mode, runs split at label changes or gaps
  > 20 ms); `default_tonemap = {start 3000, reward 5000, error 440, timeout 130}` overridable
  by `tonemap` in nidaq_map.txt (`get_tonemap`); `parse_tones_digital(method='auto')` uses it
  when the line carries the square wave, else the old duration rules; `parse_tones`' checks
  shared as `_package_tones`; `parse_trials(anchors=)`: a rewarded first lick is the last lick
  <= its solenoid opening; `get_binary_signals(tone_method=, save=False)` for dry runs.
- Tested on DS15_20260702: same frequencies as DS48 (3125/~5000/439/131 Hz measured); dry run
  identical to the saved file except the 6 error tones that cut a cue — their onsets are now
  measured, ~1.0 ms earlier than the old inference for 4 (the old rule assumed a 200.15 ms
  error tone; standalone error tones last 201.12–201.16 ms, and the measured merged ones match
  exactly) — and those cues' offsets. Cut-off cues lasted 8.3, 25.3, 30.5, 45.2 ms. First
  licks unchanged (NIDQ licks lead the solenoid by 0.08–0.12 ms). Written to disk; previous
  file kept as `binary_signals_pre_tonefreq.npz` (both copies). event_table_licks tests pass.
- Next: re-parse the other sessions (the curate_subject `binary_signals` step does it, keeping
  `binary_signals_pre_tonefix.npz` only once — sessions already re-parsed need their own backup
  name); fix `add_all_licks` (issue 4).
- 2026-10-01: re-parsing every other session with `scripts/reparse_binary_signals.py`
  (DS48 0921-0924, DS15 0701/0703/0707/0708, DS45 0811/0812/0814/0819/0820/0821, DS2 0505/0506/
  0511). Backups go to `D:/experiment_data/backups/<animal>/<session>/
  binary_signals_pre_tonefreq.npz` (one copy each: the sorting-side mirror is checked
  byte-identical; C: had no room for them). DS48_20260921: 3 error onsets moved ~1 ms earlier
  and those cues' offsets corrected; nothing else changed.
- 2026-10-01: all 17 re-parsed, every reward tone matched a solenoid, no counts changed except
  DS2_20260511 — which exposed a bug: the cycle spanning a change of pitch (last cue edge ->
  first error edge, 2-3 ms) could pass for an error cycle, starting the error tone ~2-3 ms
  early, before its triggering lick (2291.79 s), so that early trial was dropped as lickless.
  Fixed in `_tone_segments`: a cycle's high time must also match its tone (50% duty,
  `TONE_HIGH_TOLERANCE` 1.6), and a tone's first/last cycles must be its own pitch before
  smoothing. Checked: every error tone now lasts 201.12–201.16 ms, merged or not (DS2_0511,
  DS15_0702, DS48_0922). DS2_0511's extra start cue is real (a 43-ms cue at 3406.36 s with no
  outcome — the task stopped mid-cue; the old parser dropped it as < 50 ms).
  Re-parsing all 18 sessions again (0702 included; its backup moved into
  `D:/experiment_data/backups`; a duplicate pre_tonefreq copy remains in
  `sorting_files/DS15/DS15_20260702/binary_signals/`).
- 2026-10-01 15:00, status of the second (fixed-parser) re-parse: done — DS48 0921, 0922, 0923,
  0924 (sequential, ~2.5 min each on C:) and DS15_20260701. The user asked about speed;
  `reparse_binary_signals.py` gained `--jobs=N` (a process per session), but 5 jobs on D: were
  slower, not faster (DS15_0701 took 22 min vs ~9.7 alone: the sessions compete for the D:
  drive), and Claude Code stopped the run for low system memory at ~15:00. Not done:
  DS15 0702, 0703, 0707, 0708; DS45 0811, 0812, 0814, 0819, 0820, 0821; DS2 0505, 0506, 0511.
  Their originals are safe in `D:/experiment_data/backups`. Restart only when the user asks;
  run them sequentially (or 2 at a time at most).
- 2026-10-01 15:40: the user asked for fewer parallel jobs; 3 jobs also grew without bound
  (~6.6 GB/min, 105 GB in 17 min, no session past tone parsing) and were stopped. Cause: every
  parse computed the NIDQ -> imec clock fit with `sync_rec.get_traces()`, and the imec SYNC
  channel is interleaved with all probe channels, so it scans the whole AP binary (66 GB
  DS15_0702, 99 GB DS48_0922) — slow on D:, and its mapped pages fill memory.
  Fix (`parse_nidq.py`): the fit is cached as `binary_signals/nidq_imec_alignment.json` (keyed
  to the raw binaries' sizes) and reused; `seed_alignment_from_saved()` recovers it exactly
  from a saved binary_signals (raw NIDQ solenoid edges vs saved `solenoid_on`, residual
  check). DS48_20260922: recovered vs computed differ by <= 0.23 ns; 21 s vs 95 s.
  `reparse_binary_signals.py` seeds the cache before each parse. Re-run with 3 jobs: memory
  flat at ~34 GB total, ~8 min per session in parallel. (Diffs now list every event with
  "0 times moved": sub-ns differences between the recovered and original fit.)

### 5. Curation + re-analysis batch on D: — done, awaiting the user's review (2026-10-02)
Overnight run finished 03:55. Cleaning (clean units / good + non_soma_good): DS45 0814 269 /
47+4, 0819 291 / 53+4, 0820 384 / 88+4, 0821 296 / 57+6; DS2 0505 429 / 68+3, 0506 328 / 30+0
(487 noise units dropped of 792), 0511 385 / 49+0. Analysis (pynapple + isolate with the new tagging + ratio_tests) ok for all
17 sorted sessions; tagged single units (final_classif):
DS48 0921 dSPN 7, mu 1 | 0922 iSPN 16, dSPN 10, mu 10 | 0923 iSPN 21, dSPN 16, mu 10 |
0924 dSPN 10, iSPN 7, mu 1; DS15 0701 dSPN 2 | 0702 iSPN 5, dSPN 3, mu 2 | 0703 dSPN 5, iSPN 1 |
0707 iSPN 4; DS45 0811 dSPN 8, iSPN 4, mu 1 | 0812 dSPN 5, mu 2, iSPN 1 | 0814 dSPN 5, iSPN 1 |
0819 iSPN 4 | 0820 iSPN 6, mu 2 | 0821 iSPN 2; DS2 0505 dSPN 12, mu 1 | 0506 dSPN 2 |
0511 dSPN 6, iSPN 2, mu 2. Possible false negatives (tagged, failing quality) are 33–206 per
session (issue 3). DS48 compendium ok (164 units). The single compendium process then
segfaulted starting DS15's (native crash, right after its sessions merged; same plots ran fine
in their own process on 2026-10-01) — rebuilding DS15, DS45, DS2 one process each with
faulthandler on (scratchpad `one_compendium.py`). All four compendia now done: DS48 (164 lick-bout
modulated units), DS15 (148), DS45 (229, after clearing the numba cache — issue 7), DS2 (114).

Overnight run notes (2026-10-01 22:20):
2026-10-01 overnight run (scratchpad `run_parallel.sh`, logs `par_<session>_<step>.log`,
summary `parallel.log`): cleaning 3 sessions at a time, then pynapple+isolate+ratio_tests for
all 17 sorted sessions 2 at a time (DS48, DS15 first), then the DS48/DS15/DS45/DS2 compendia.
Done by 22:17: DS45 0814 (sequential, 60 min), 0819 (141 min), 0820 (163 min), 0821 (56 min) —
parallel cleaning on D: is barely faster than sequential (disk contention). DS2 0505/0506/0511
cleaning in progress. If found interrupted: check each `par_*` log; cleaning that finished has
`analyzer_clean_prev` + a new `analyzer_clean`; rerun only what is missing.

Earlier pause:
2026-10-01: the user asked to keep the spike removal going while issues 2/4 are pending.
`curate_subject.py` gained `--steps=`; ran `--steps=binary_signals,clean,summary` on DS45, then
paused by the user at 11:31.
**Progress of the cleaning-only run (binary_signals + clean + summary):**
- DS45_20260812: done (19.3 min clean): 489 units in, 18 bimodal splits, 159 noise dropped,
  2 merges (redundant step included), 341 clean (82 good, 7 non_soma_good, 227 mua,
  25 non_soma_mua); sorting summary + rp_review (15 units) written. Old analyzer_clean kept as
  analyzer_clean_prev.
- DS45_20260814: stopped during the binary_signals backup copy. The original
  `binary_signals.npz` (both copies) is untouched and loads; the incomplete
  `binary_signals_pre_tonefix.partial` was deleted. Nothing else of 0814 was touched.
- Not started: DS45 0814 (from the start), 0819, 0820, 0821; DS2 0505, 0506, 0511.
**Resume:** `PYTHONPATH=. python -m scripts.curate_subject D:/experiment_data/recording_files/DS45
DS45_20260814 DS45_20260819 DS45_20260820 DS45_20260821 --steps=binary_signals,clean,summary`,
then `... curate_subject D:/experiment_data/recording_files/DS2 --steps=binary_signals,clean,summary`.
Still to do for every session once issues 2 and 4 are settled: the `pynapple`, `isolate`,
`ratio_tests` steps and the subject compendia (DS48, DS15, DS45, DS2).
Original state when paused the first time:
`scripts/curate_subject.py` per subject (binary_signals re-parse -> clean -> summary/rp_review
-> pynapple plots -> isolate (bombcell) -> ratio_tests; then subject compendium).
- DS15: 0701, 0702, 0703, 0707 done; compendium over all four. 0708 skipped (never sorted,
  preprocess incomplete; user's choice).
- DS45: 0811 done. 0812: binary_signals re-parsed, outlier_scores.pkl cached, cleaning stopped
  mid-run (old analyzer_clean untouched). 0814, 0819, 0820, 0821 not started. 0813 never sorted.
- DS2: 0505, 0506, 0511 not started.
Resume: `PYTHONPATH=. python -m scripts.curate_subject D:/experiment_data/recording_files/DS45
DS45_20260812 DS45_20260814 DS45_20260819 DS45_20260820 DS45_20260821`, rebuild the DS45
compendium over all six sessions, then `... curate_subject D:/experiment_data/recording_files/DS2`.
Issues 1 and 2 change the cleaning and tagging, so DS48/DS15/DS45_0811 will need re-running too.

### 6. Event-table licks only cover part of DS48 0923/0924 — open (2026-09-30)
Licks recovered from the Arduino event table are known only inside the table's span
(0923: 367–2706 s; 0924: 323–3203 s); the opto blocks run ~10 s to the end of the recording.
Outside the span licks are unknown, not absent. Lick-based opto analyses of these sessions
should be restricted to the span — nothing in the code does this yet.

### 7. Parallel PSTH workers crash in pynapple — open (2026-09-30)
`analysis/plot_psth.plot_on_events_psth` (joblib loky, `n_jobs=-1`) intermittently dies with a
Windows access violation inside pynapple's numba `jitcount`, never in a single process; seen on
lick-onset PSTHs of DS48 0923. Probably concurrent workers on numba's `cache=True` files.
Workaround: `curate_subject.py` redoes the step in-process on `TerminatedWorkerError`.
`plot_psth.py` itself is unchanged.
- 2026-10-01: likely cause found in the repo's own `utils/mpl_backend.py` (an earlier fix):
  loky workers that import a Qt matplotlib backend have no event loop, and on Windows that
  surfaces as sporadic access violations in unrelated native code such as numba kernels. Check
  whether `analysis/plot_psth.py` (or what it imports) sets a Qt backend instead of calling
  `mpl_backend.select_backend()`; if so, that is the fix.
- 2026-10-02: the same access violation in pynapple's `jitcount` hit a *single* process (no
  workers, Agg backend): the DS45 compendium, in `plot_dspn_ispn_lick_bout_bar` ->
  `_zscore_diff` -> `TsGroup.count`, and it killed the combined compendium run at DS15. Inputs
  checked and valid (merged time support sorted, non-overlapping; spikes sorted, none outside).
  pynapple's numba cache (`site-packages/pynapple/core/__pycache__/*.nbi|*.nbc`, cache=True)
  dated from March–June — compiled before the env's current numba 0.61.2 / numpy 2.2.6, and
  numba does not invalidate a cache on dependency upgrades. Moved aside (23 files, to the
  scratchpad `pynapple_numba_cache_backup/`) so numba recompiles; rebuilding the DS45
  compendium to test. If it recurs, the Qt-backend lead above is next.
  Result: the DS45 compendium then completed (6 sessions, 25 figures) where it had crashed
  twice. Consistent with the stale cache, but one clean run is not proof — keep an eye out.

### 8. Compendium session list points at C: copies — open (2026-10-01)
`analysis/compendium_plots.py`'s `__main__` list uses `C:\Users\assad\Desktop\experiment_data\
analysis_files\{DS15,DS45,DS2}`, but the re-curation writes to the D: copies. The pooled
cross-subject compendium is stale until the list points at D: (or the C: copies are refreshed).

### 9. DS48_20260924 drops many close spike pairs — open (2026-09-30)
Its cleaning removed 186,757 spikes as pairs <= 1 ms (~0.6% of 31M), far more than the other
sessions. Not investigated; may be noise, a dense cluster, or duplicate units (issue 1).

### 10. Data-hygiene questions for the user — open (2026-09-30)
- DS48_20260923 behaviour files are named `HSOMBS_DS40_20260923.mat` / `opto_DS40_20260923.mat`
  (DS40); the event table matches this recording exactly, so probably misnamed. Confirm.
- `recording_files/DS45/DS45_20260813` holds `DS45_20260820_eventTable.csv` and
  `ANID_20260820_eventTable.csv` (misplaced; 0820's NIDQ licks are fine, so unused).
- DS45_20260821 has no rewards (29 licks, 0 solenoid): its ratio_tests will fail / be skipped.
- DS15_20260703 has two probes (imec1 = ALM, broke during recording); only imec0 (VLS) is sorted.
- Old sessions whose reward tone was mislabelled by the parse_tones bug (issue R5) are fixed on
  re-parse; Neptun_20260626 has no reward_tone row at all (not looked at).

### 11. Housekeeping — open (2026-10-01)
- Most of the cleaning code is untracked in git (`postprocessing/spike_cleaning.py`,
  `split_outlier_spikes.py`, `cleaning_plots.py`, `scripts/clean_sessions.py`,
  `curate_subject.py`, `preprocessing/event_table_licks.py`, ...). An edit deleted functions
  from the untracked `spike_cleaning.py` on 2026-09-30 and they had to be recovered from Claude
  Code's file history. Commit them.
- Stray files in the repo root: `X.npy`, `big.npy`, `real1.pkl` (older), `cuefit_DS48_*.npy`,
  `fit_DS48_*.npy` (scratch outputs of the lick-alignment work).
- `DS48_20260921/spike_cleaning/unit_labels_new.csv` was written because `unit_labels.csv` was
  open in Excel; replace the old file with it.
- C: had ~36 GB free on 2026-10-01; large outputs should go to D:.

### 12. Offered, never answered — open
- Rename the final units back to their Kilosort numbers, the better half of a split inheriting
  the parent's number (now final units get appended ids; `ks_unit_ids` maps them back).

---

## Resolved

### R1. Refractory (rp) rule for good -> MUA — resolved 2026-09-30
After the 5% point-estimate and 5% lower-bound trials, the user reviewed the units by eye and
chose: MUA when the 95% lower bound of the Llobet contamination is over 10% *and* the unit has
more than 3 violations (`RP_MAX_CONTAMINATION, RP_CONFIDENCE, RP_MAX_FREE_VIOLATIONS = 0.10,
0.95, 3` in `spike_cleaning.py`). Applied to DS48 by `clean_sessions --relabel`.

### R2. Licks missing on the NIDQ for DS48 0923/0924 — resolved 2026-09-30
Broken BNC; licks recovered from the behaviour computer's event table, aligned on CUE_ON ->
start_tone_on (linear fit, ~13 ppm drift, residuals < 0.08 ms), via
`preprocessing/event_table_licks.py` and `get_binary_signals(lick_source='auto'|'nidq'|
'event_table')`. Originals kept as `binary_signals_nidq_no_licks.npz`. `lick_off` is left
empty. (Limitation: issue 6.)

### R3. Spike-cleaning merge helpers deleted — resolved 2026-10-01
`_template_cosine`, `_cross_violations`, `_strict_merge_veto` were cut by a block edit on
2026-09-30, so every DS15 clean failed; restored from file history (v5), verified nothing else
changed, pyflakes clean.

### R4. Interrupted backup copy — resolved 2026-10-01
Stopping a run cut DS45_20260811's `binary_signals_pre_tonefix.npz` short (original intact);
the partial file was deleted and `curate_subject.py` now copies to `.partial` and renames.

### R5. Reward tone mislabelled as error tone — resolved 2026-09-30
`parse_nidq.parse_tones` matched reward tones to solenoid openings by rounding to 10 ms; now the
nearest solenoid within 2 ms. DS15_20260702 (30 -> 31 rewarded) and DS45_20260811 (26 -> 27)
re-parsed; old files kept as `binary_signals_pre_tonefix.npz`.

### R6. Analyzers without a recording — resolved 2026-10-01
DS15/DS2 analyzers have no loadable recording; `clean_sessions.clean_session` now attaches the
session's preprocess folder after checking samples and channels match.

### R7. Single-session compendium plots crash without both SPN types — resolved 2026-10-01
`_spn_rows` helper and guards in `plot_cue_outcome_heatmap` / `plot_spn_cue_aligned_nostim`
skip panels or figures with no dSPN/iSPN; `curate_subject.py` pools every session whose isolate
step ran (not only those whose ratio_tests did).

### R8. `≈` in a print crashed redirected runs on Windows — resolved 2026-09-30
`parse_nidq.py` alignment print uses `~`.
