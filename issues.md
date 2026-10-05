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
- 2026-10-02, DS2 traced (user: "I've lost basically all the iSPNs from the old pipeline";
  scratchpad `trace_ispn.py`, tables `trace/<session>/trace_<session>.csv`). The old pipeline's
  10 DS2 iSPNs: 1 kept (0511 KS 205), 1 dropped at cleaning (0505 KS 542, bombcell noise:
  peak_to_trough_duration 1.23 ms), 8 lost at QUALITY (bombcell mua / non_soma_mua). Tagging
  lost none (new rules: 8 of the 9 surviving units still iSPN, 1 muSPN; all lick-bout
  modulated). Criteria failed by all 40 old-tagged units lost at quality: rp_contamination 27,
  snr 25, amplitude_median 17, presence_ratio 10, num_spikes 2. DS2 cannot meet bombcell's
  snr >= 5 / amplitude >= 30 µV: session medians snr 3.6 / 2.7 / 3.2, amplitude 30 / 15 / 18 µV
  (0505 / 0506 / 0511); snr < 5 for 277/429, 219/328, 279/385 units. The old pipeline used
  KSLabel (refractory only). Odd: amplitude_median values are quantised in ~3.03 µV steps.
  Two old 0505 iSPNs (KS 4, 545) had KSLabel mua — let in some other way (manual list?).
  Options for the user: rig/probe-specific snr/amplitude thresholds (e.g. relative to each
  session's noise), snr/amplitude as review flags only, or review the 8 by eye first (several
  also have high refractory contamination, lower bounds 0.29–1.0).

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

### 14. Many "noise" units dropped on peak_to_trough_duration alone — in progress: DS2 re-cleaning, other datasets TODO (2026-10-02)
Found tracing the DS2 iSPNs: DS2_20260505 KS 542 (an old iSPN, 4,447 spikes) was dropped as
bombcell noise only for peak_to_trough_duration = 1.23 ms (noise limit < 1.15 ms), yet its
waveform is a clean positive-peaked spike whose peak and trough are ~0.2 ms apart on every
channel (`spike_cleaning/noise_unit_plots/noise_unit_542.png`). Suspected: for a positive-
polarity waveform the metric runs from the trough to the *next* peak, i.e. the slow recovery
hump ~1.2 ms later. Scale (noise units dropped for this criterion alone): DS15 38/81/52/105;
DS2 121/35/57; DS45 33/59/34/28/43/45; DS48 42/51/89/35 — ~950 units over 17 sessions, often
half of each session's noise drops. They are only out of analyzer_clean (the raw analyzer has
them), so a re-clean recovers them. Next: verify the metric on a sample (trough -> recovery-hump
hypothesis, share of positive-peaked units), review a sheet of their noise plots with the user,
then a polarity-aware duration (or exempt positive-peaked units) and re-clean.
- 2026-10-02, user: "bombcell is way too conservative with classifying noise, unit 542, 548 look
  like well isolated units". 548 (19,264 spikes) is a classic negative spike with a slow,
  peakless recovery: duration 1.42 ms is the trough to wherever the flat tail peaks — so the
  metric misfires on any waveform without a distinct after-peak, not just positive ones.
  All 17 sessions: 2,696 noise units dropped; 1,775 failed exactly 1 criterion, 445 two,
  476 three+. Criterion counts: peak_to_trough_duration 1,090, num_negative_peaks 1,065,
  exp_decay 917, num_positive_peaks 586, peak_after_to_trough_ratio 469, baseline flatness 152.
  Options put to the user: (1) noise only on >= 2 failed criteria; (2) fix the metrics (duration
  to a prominent after-peak only, count only prominent peaks); (3) no noise drop, label as a
  flag. Proposed first: a review sheet, ~20 random single-criterion drops per criterion.
  Recovery needs a re-clean (outlier-score caches cover these units).
- 2026-10-02, decided (user): peak_to_trough_duration removed from the noise criteria
  (`spike_cleaning.NOISE_EXCLUDED_METRICS`, `default_thresholds()`, used everywhere the
  bombcell defaults were). Why the metric fails, verified on the templates: SpikeInterface looks
  for an after-peak of >= 40% (then 20%) of the waveform's largest deflection and, finding none,
  takes the tail's global maximum — a 2–8 µV wobble ~1.4 ms out on slowly repolarising units.
  The user preferred recovering units over a full re-clean; a re-clean with the cached outlier
  scores is cheaper (SpikeInterface cannot combine two analyzers, so an incremental merge would
  rebuild every unit's extensions) and leaves untouched units identical — agreed for DS2.
  `clean_session` now re-cleans a session that already has analyzer_clean + _prev: the earlier
  cleaning goes to `analyzer_clean_superseded` (delete once accepted).
- Review sheets for the other single-criterion drops (user to decide):
  `D:/experiment_data/noise_review/{exp_decay,peak_after_to_trough_ratio,num_negative_peaks}_review.png`
  (+ `_sample.csv`), 20 random units each from all 17 sessions (351 / 253 / 214 such units).
  First impression of the num_negative_peaks sheet: mostly real noise (flat, ringing).
- 2026-10-02 ~19:15: **rerun started** (user: "Start it") for DS48, DS15 and DS45, run one
  subject after another (scratchpad `rerun_noise_snr.sh`, logs `rerun_<subj>.log`):
  - Clears the review-plot folders, then clean, summary, pynapple, isolate, ratio_tests and the
    subject compendium.
  - The clean step uses the new noise rule plus SNR 4 (covers issue 15 too) and keeps the
    earlier cleaning as `analyzer_clean_superseded`.
  - It was started before the user had reviewed the noise review sheets.
- 2026-10-03, results:
  - **DS48** (done 23:31, compendium ok). Good + non_soma_good / tagged:
    - 0921: 60+3 / dSPN 12, iSPN 1, mu 1
    - 0922: 91+4 / iSPN 18, dSPN 12, mu 15
    - 0923: 109+20 / iSPN 27, dSPN 20, mu 12
    - 0924: 92+1 / dSPN 17, iSPN 7, mu 2
    - 144 tagged in total, was 109.
  - **DS45** (done 06:50, compendium ok). Good + non_soma_good / tagged:
    - 0811: 123+12 / dSPN 15, iSPN 5, mu 2
    - 0812: 106+13 / dSPN 6, iSPN 2, mu 2
    - 0814: 67+9 / dSPN 13, iSPN 2
    - 0819: 71+6 / iSPN 4
    - 0820: 108+11 / iSPN 9, mu 4
    - 0821: 75+10 / iSPN 3
  - **DS15:** the run died silently during 0702's ratio_tests plotting (no traceback; a native
    crash like issue 7), so 0703 and 0707 never ran.
    - 0701: 31 good / dSPN 2. 0702: 105+23 / iSPN 6, dSPN 6, mu 4.
    - Finished on the second run (logs `rerun_DS15b/c.log`):
      - 0703: 48+14 / dSPN 12, iSPN 1 (was dSPN 5, iSPN 1).
      - 0707: 82+6 / iSPN 4, dSPN 1 (was iSPN 4).
      - DS15 compendium over all four: ok.
    - All datasets now use the same rules. Cross-dataset compendium (`compendium_plots.py`
      __main__) queued after it.
  - The "noise: 1–3" in the clean reports are units that the *final* bombcell pass calls noise.
    Spike cleaning or a merge changed their waveform (mostly num_negative_peaks=2 or
    exp_decay), and they were then dropped from analyzer_clean. They were mua or non_soma_mua
    before. Not a bug.
- **TODO: rerun the other datasets too** (DS48 0921–0924, DS15 0701/0702/0703/0707, DS45
  0811–0821): `curate_subject --steps=clean,summary,pynapple,isolate,ratio_tests` per subject
  (clear `spike_cleaning/{noise_unit_plots,removed_spike_plots,bimodal_split_plots,rp_review}`
  first), ideally after the user has decided on the review sheets, so the noise rules change
  once. DS2 is being re-cleaned now (2026-10-02 ~09:20).
- 2026-10-02, DS2 done (re-clean + summary + pynapple + isolate + ratio_tests + compendium):
  noise dropped 221->100 / 487->452 / 169->112, clean units 429->553 / 328->364 / 385->446
  (0505 / 0506 / 0511); good+non_soma_good 73+5 / 31 / 52; tagged single units 0505 dSPN 12,
  mu 1 | 0506 dSPN 2 | 0511 dSPN 7, iSPN 2, mu 2 (unchanged: the recovered units mostly fail
  amplitude/snr, so possible false negatives rose to 294 / 206 / 167). DS2 compendium over all
  three: 125 lick-bout modulated units. Old cleanings kept as `analyzer_clean_superseded`.

### 15. SNR threshold 5 -> 4 — queued (2026-10-02)
User agreed (dry run on DS2: good + non_soma_good 78->92, 31->34, 52->56; tagged units gained:
0505 dSPN 695, 719, 833, 1283, 1359, 1393; 0511 iSPN 790 (old KS 157), dSPN 978, 1022 — each
failed only SNR in [4, 5)). Reason: the noise level that SNR divides by is quantised (the probe
stores 3.03 µV/bit and the background is only 3-5 bits wide, so the MAD estimate takes values
4.5 / 9.0 / 13.5 µV) and DS2's units are smaller (median 15-27 µV vs 30-48 elsewhere).
Applied: `spike_cleaning.MUA_SNR_MIN = 4` in `default_thresholds()`. DS2 relabel + summary,
pynapple, isolate, ratio_tests + pooled compendium done (2026-10-02).
- good + non_soma_good: 85+7 / 34 / 56 (was 73+5 / 31 / 52).
- Tagged single units: 0505 dSPN 18, mu 1 (was dSPN 12); 0506 dSPN 2; 0511 dSPN 9, iSPN 3
  (now includes 790), mu 2 (was dSPN 7, iSPN 2).
- Awaiting the user's review. TODO: relabel DS48, DS15, DS45 too (can go
with the issue-14 rerun). The refractory rule stays (685: spontaneous violations on a flat 1-3 ms
floor = ~15-20% foreign spikes).
Re-sort test 683/685 (scratchpad `pair_resort_test.py`, results in
`sorting_files/DS2/DS2_20260505/unit_685_pca/resort_685_683.*`): no help. 685's lower bound is
0.150 as sorted; it rises to 0.222 with hard reassignment and stays at 0.154 with a 0.05 margin.
Dropping the ambiguous spikes gives 0.093, but that removes 31.6k of 128k spikes, and removing
the same number of 685's spikes at random gives 0.130 (0.089-0.182), so it is no better than chance.
683 gets worse under every reassignment. The two templates barely separate spikes by correlation.
The foreign spikes in 685 are not 683's.

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

### 8. Compendium session list points at C: copies — fixed (2026-10-02), now points at D: (issue 16)
`analysis/compendium_plots.py`'s `__main__` list uses `C:\Users\assad\Desktop\experiment_data\
analysis_files\{DS15,DS45,DS2}`, but the re-curation writes to the D: copies. The pooled
cross-subject compendium is stale until the list points at D: (or the C: copies are refreshed).

### 16. Move DS48 to D: for a cross-dataset compendium — done, awaiting acceptance (2026-10-02)
**Done 19:05:**
- Copied, and verified with the scratchpad `verify_ds48.py`: 61,870 + 11,647 + 208 files,
  checked on size, mtime and content hash (whole file under 50 MB, first and last 64 MB above).
  No file was missing or different apart from the known 0729 clash.
- The C: copy of the clashing 0729 npz is kept on D: as `binary_signals_from_C.npz`.
- Deleted `recording_files/DS48` and `sorting_files/DS48` from C: (C: free 25 -> 1049 GB).
  `analysis_files/DS48` stays on C: as a copy.
- All 4 sessions load from D: (analyzer_clean with recording, pynapple, binary_signals).
- `compendium_plots.py` now points at `D:\experiment_data\analysis_files` with DS48 0921–0924
  added; CLAUDE.md updated.
- Caveat: DS48 / DS15 / DS45 are still curated under the old noise rule and SNR 5 (issues 14
  and 15), so a cross-dataset compendium mixes criteria until they are rerun.

Original request and plan:
User asked for all of DS48 (recording 928 GB, sorting 97 GB, analysis 33 GB) to be moved from
`C:\Users\assad\Desktop\experiment_data` to `D:\experiment_data`, keeping a copy of
`analysis_files/DS48` on C:.
- **Copy:** robocopy (scratchpad `copy_ds48.ps1`, logs in `copy_ds48/`), never overwriting
  files already on D:. D: already had DS48 0717–0727 recordings and `DS48_20260729`
  sorting/analysis. Both drives have a `sorting_files/DS48/DS48_20260729/binary_signals/binary_signals.npz`
  and the two differ; the C: copy is not copied over.
- **Paths:** analyzers and `preprocess` store relative paths, and `clean_sessions.load_analyzer`
  re-attaches a moved recording, so the sessions should load from D:.
- **Then:** verify; delete `recording_files/DS48` and `sorting_files/DS48` on C:; point
  `compendium_plots.py` and the CLAUDE.md data layout at D: (see also issue 8).

### 19. Units split by drift — in progress (2026-10-03)
User: DS2_20260505 1599 and 1607 "are the same unit split in time" (then 1411/1417).
**1599 / 1607 / 1609:**
- Our bimodal split cut KS 605 at Kilosort's two ~6 um drift steps (~2000 s, ~2650 s).
- The pieces are mutually refractory: pooled 54,792 spikes with 4 violations; 3 of 46 expected
  coincidences across the 1599->1609 handover.
- The merge step could not rejoin them: their template cosine is 0.67, under the 0.9 floor.

**1411 / 1417** (KS 506 / 509, 23 um apart): not supported as one neuron. Both are active all
session and coincide at chance in every 10-min window. Both are small and contaminated; 1411's
violations rise over the session. Told the user; left as MUA.

**Neighbour-pair scan** (scratchpad `neighbor_refractory.py`, 24k pairs within 40 um, expected >= 10):
- Coincidence ratio median 1.47, 5th percentile 0.74, 1st percentile 0.36.
- Close high-rate pairs sit at ~0.25-0.3, which looks like the sorter missing overlapping spikes.
- Some pairs with unrelated templates are near 0. So refractoriness alone is not enough.

**User chose option 1** (no re-sort). Implemented:
- `sos.coincidences` / `sos.mutually_refractory`: at least 20 expected, at most 10% observed,
  with the expectation time-resolved in 60-s bins.
- `_bimodal_split` refuses a split whose clusters are mutually refractory (reason "one neuron:
  ...").
- New merge preset `DRIFT` (before REDUNDANT) for neighbours within 40 um that are mutually
  refractory with template cosine >= 0.5. `_drift_merge_veto` checks every pair of a group;
  pairs that never overlap may have at most 3 coincidences.
- Kilosort drift per session (ops `dshift`): DS2 0506 130 um with a 50 um jump in 1 min (re-sort
  candidate, option 3); DS2 0505 14 um; the rest 1-12 um.
- Dry run 1 on DS2_20260505 (scratchpad `test_drift_0505.py`, nothing written). The 605 split
  was refused (4 of 46.5 expected), so it is one non_soma_good unit (54,931 spikes, 4
  violations). But two drift merges turned good units into MUA: 250+248 (lower bound 0.26) and
  311+308 (138 violations). Their partners were contaminated themselves.
  Added `_merged_rp_veto`: the pooled train must pass the labels' refractory rule
  (`_rp_fails`; it agrees with the stored labels on all 553 units of 0505).
- Dry run 2: good 85 + non_soma_good 7, the same as now. Changes:
  - 605 is one unit again.
  - 229+230/2 drift-merged (mua + mua -> mua, 5 violations; KS 230 is one of the user's
    "two clusters" units).
  - 1411/1417 (KS 506/509) unchanged.
- 2026-10-03 21:14: **full rerun started** (user: "Rename it and let it rip").
  - The 17 `analyzer_clean_superseded` folders were renamed `_superseded_2`.
  - Script: scratchpad `rerun_drift.sh`, logs in `rerun_drift/<session>.log`.
  - Per session: clean, summary, pynapple, each retried once on a crash.
  - Then per subject: isolate, ratio_tests and the compendium. Then the cross-dataset compendium.
  - DS2, DS48, DS15, DS45, in that order.
- 2026-10-04 20:13, **rerun done**: all steps ok for 16 sessions, except DS2_20260505.
  - **DS2_20260505:** its pynapple step failed because `singU_metadata.csv` was open in Excel,
    after `spikes.npz` had already been cleared.
    - Fixed: `curate_subject.pynapple_analysis` now skips a locked file with a warning, and
      `isolate` writes `singU_metadata_new.csv` when the csv is locked.
    - 0505's pynapple and isolate are being rerun, then the cross-dataset compendium.
  - **Splits refused as one neuron:** DS2 0505 KS 605 (4 of 46 expected), DS2 0506 KS 442
    (0 of 23), DS15 0702 KS 496 (0 of 26).
  - **Drift merges:** DS2 0505 1027+1573 (229+230/2); DS2 0506 KS 402+413/1 (mua);
    DS15 0702 496+487/2 (good, 10 violations); DS45 0811 1130+1152 (134/2+134/1/2);
    DS45 0812 327+321/2 (good); DS45 0814 382+666; DS45 0820 756+772. The last four were
    merged again by a later preset.
  - **Tagged counts:** unchanged everywhere except:
    - DS15 0702: dSPN 6 -> 7.
    - DS2 0505: KS 605, whole again (unit 1521, non_soma_good), is a new **iSPN**.
  - 21:40: DS2_20260505 fixed; the DS2 compendium and the cross-dataset compendium are rebuilt
    (668 lick-bout modulated units, 25 figures). Awaiting the user's review.
- Then
  re-clean all 17 sessions. Note that `clean_session` refuses while `analyzer_clean_superseded`
  exists (it does for every session after the noise rerun); the user must decide whether to
  delete or rotate them.

### 18. Cross-dataset compendium hung in the PSTH loops — fixed (2026-10-03)
The 12-session pooled compendium (672 units) sat at 49 GB for over 10 min in
`plot_spn_lick_bout_psth`.
- **Cause:** a `np.histogram(spk - t)` over each unit's *whole* spike train for every
  (event, unit) pair. The same loop was in 4 places in `analysis/pres_plot_utils.py`.
- **Fix:** a new `event_counts()` that counts by binary search. It gives identical counts:
  0 mismatching bins over DS2_20260505's 92 units x 439 bouts.
- Separately: these PSTHs scaled counts as `h * bin_width` (count x 0.05), not a rate, though
  their labels said spikes/s. User: "I would like psth bin counts/averages to be scaled to Hz".
  **Fixed 2026-10-03:**
  - All 18 sites in `pres_plot_utils.py` now use `/ bin_width`, including `compute_unit_z`,
    so z-scores are unchanged. Checked: binned means equal spikes/s exactly.
  - The 3 remaining per-trial histogram loops now use `event_counts` too.
  - The raw `cue_outcome_heatmap` clim `(0, 0.1)` -> `(0, 40)` Hz (the same colour scale).
  - `plot_psth.py` was already in Hz.
  - Regenerated, all ok (12:02): every ratio_tests folder, the 4 subject compendia and the
    cross-dataset compendium (`D:\experiment_data\analysis_files\compendium`, 25 figures).
    Done with the scratchpad `replot_hz.py`.
  - Still slow on the pooled data (41 min in all): `dspn_ispn_lick_bout_bar` and the
    `dspn_ispn_bar_*` plots took ~25 min between them. Not looked at yet.

### 17. Other datasets: re-sort readiness — surveyed, awaiting the user (2026-10-03)
**How it was checked:**
- Scratchpad `inventory.py` lists sort and curation dates; `inventory.csv` holds the table.
- Scratchpad `check_parse.py` runs `get_binary_signals(overwrite=True, save=False)` with the
  current parser, using an identity map in place of the imec clock fit unless a fit is cached.
  Log: `check_parse.log`.

**Parse OK with the current parser:**
- DS13 0822, 0824 (no nidaq_map; the defaults work).
- DS21 1209 1210 1211 1213. Sorted 2025-12-16 / 2026-03-17.
- DS23 0208–0211. Sorted 2026-09-20..26.
- DS37 031726 031826 031926 032126. Sorted 2026-09-20.
- DS45 0813 (never sorted), DS15 0708 (never sorted).
- DS1 1122 (no nidaq_map; never sorted).

**Fail:**
- DS1 1117–1120 (no nidaq_map): "Cue 0 does not precede its outcome". The defaults read the
  tone and solenoid from analog channels, but DS1 has them on digital lines 3 and 2. With that
  dinmap the tones parse, but opto fails (the GMM gets 0 samples on default analog channels
  6/7). DS1 needs a nidaq_map.txt from the user.
- DS13 0905 (no nidaq_map): "Multiple outcome tones found between cue at index 0".
- DS21 1212: two SpikeGLX runs (g0, g1); the parser needs `gnum`.
- Neptun 0625: no SpikeGLX folder.
- DS48 0717–0729: behaviour/NIDQ only, no imec (0729 parses), so these cannot be sorted.

**DS45_20260813 "always fails re-sort" — explained:**
- The old duration-based tone classification raises `_package_tones` "Multiple outcome tones
  found between cue at index 91 and next cue" (reproduced with `tone_method='duration'`).
- At cue 91 (t = 2203.9 s) the timeout tone lasted 1.0 s instead of the usual 2.0 s (same
  130 Hz pitch, 262 vs 526 edges, no gap).
- The frequency parser (default since 2026-10-01) classifies it by pitch. The full call the
  re-sort makes (`plot=True`, real clock fit 0.999997298 / +0.29 ms) now parses: 118 trials,
  8 rewards, 10 errors, 100 timeouts.
- The session's notes call it a "complete disaster" (mouse not engaged), with a second run
  `g1` parked in `hidden/`.

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
