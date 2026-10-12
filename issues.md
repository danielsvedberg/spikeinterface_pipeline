# Issues

Every problem found while working on the pipeline, with its status and what happens next.
Check this file before starting work, and update it as you go: add new issues the moment they
are found, record progress and decisions under the issue, and move an issue to *Resolved* only
when the user has accepted the result. Dates are when the entry was written or last updated.

Status: **open** (not started) · **investigating** · **proposed** (fix designed, awaiting the
user) · **in progress** · **blocked** (waiting on the user or data) · **resolved**

---

## Open

### 32. Data reorganized; only the latest sorts kept — done except cleanup (2026-10-09)
User: files were hard to find in the scratch layout; reorganize to the file_structure.py layout,
sorting_files + analysis_files on C:, recording_files on D:, and delete all old sorting runs,
preprocessings and their products ("work exclusively off the latest sortings").
- Root `C:\experiment_data`: real sorting_files / analysis_files, recording_files = junction to
  `D:\experiment_data\recording_files`. CLAUDE.md data layout updated; active scripts' default
  paths moved from D:\experiment_data to C:\experiment_data (scratchpad `repoint_paths.py`);
  file_moving.py / move_preprocess.py left as historical migration lists.
- Script `scripts/reorganize_data.py` (copy of scratchpad `reorganize.py`; log in the
  scratchpad `reorganize.log`), phases: layout (C: renames of the 17 heavy sessions; DS48 0922
  `_ks417` suffix dropped; `ks` -> kilosort4_output; comparison files renamed
  *_with_previous_sort), recordings (new-chain DS15/DS2 preprocess + saturation C: -> D:,
  each verified by file count and bytes before the old-chain preprocess was deleted; 7 sessions
  57-111 GB each), old (17 sessions' old D: sorting/analysis + DS15/DS2/DS45/DS48 compendia
  deleted; never-sorted DS15 0630/0708, DS45 0813, DS48 0729, DS48 photometry moved to C:),
  others (DS1, DS13, DS21, DS23, DS37, Neptun sorting + analysis, and the cross-dataset
  analysis_files/compendium, moved to C:, verified). Raw-data guard: refuses any path in a
  `_g#` folder or holding SpikeGLX .bin/.meta; never triggered. The auto-mode permission
  check blocked running the destructive phases; the user ran recordings/old/others with `!`.
- Left: `cleanup` (C:\si_stage scratch junctions/folders, the 4.1.4 test sort
  resort/DS48_20260922, cross-session CSVs + tagging_review into C:\experiment_data\analysis_files).
  The cross-dataset compendium (old sorts) was moved, not deleted -- user to decide.
  D:\experiment_data also holds backups/, noise_review/, olympus fluoro images (untouched).

### 34. Which curation approach maximises well-isolated SPN yield — answered (2026-10-12)
User asked (ultracode) for every post-sorting curation approach tried and which yields the most
well-isolated iSPNs/dSPNs. Workflow wf_2a1af3b7-876 (inventory -> per-approach yield -> independent
recomputation; all agreed). Scripts/CSVs: scratchpad yield_compare/. Yield = bombcell good or
non_soma_good and light_classif dSPN/iSPN (bU/oU/mU excluded), 17 sessions:
- Current (applied): dSPN 109 / iSPN 87 / muSPN 33 (strict good 95 / 82). Best applied.
  Stimulation-free rp relabel was the big gain (+17 / +8 / +3 vs pre_stimrp 100 / 79 / 42).
- Merge test 111 / 86 / 32 (the -1 iSPN is the bad 803+805 merge; without it 111 / 87 / 32);
  dedup test and resplit: no change; overlap template-match deletion 110 / 88 / 32, but the +1
  iSPN (DS48 0923 2407, muSPN -> iSPN) is a near-arbitrary side (match 0.966 vs 0.960) and a
  duplicate of iSPN 1723 (582/598 spikes coincide); +1 dSPN (DS2 0505 1911) sits on the
  3-violation floor. Old pre-heavy sorts (TTFS tagging) 107 / 73 / 40.
- User's manual iSPN review: +8 iSPN confident verdicts (95), up to 108 with weak ones -- the only
  real iSPN gain; needs a class-override mechanism (manual_units.txt does only good/bad); 2
  confident verdicts (793, 1754) break the 8.42 Hz rule.
- Tag rule A: 104 / 91 / 29; rule B: 98 / 83 / 29 (not applied).
- Estimated best combination (not computed): current + confident review verdicts + 723+1839
  merge + one unit for 932/1878/1882/557 + 2407/1723 counted once ~ 109 / 94 / ~23.
- Untested: dual-driven = iSPN unless blue stronger; false-positive review of the 87 iSPNs;
  7 iSPN+iSPN pairs that may double count; wider firing-test window.
- User asked to test all of them (2026-10-12). Double counting done (scratchpad
  tagging_tests/spn_pairs.py; analysis_files/tagging_tests/spn_pairs/spn_pairs.csv + one figure per
  pair): all 276 pairs of counted SPNs within 60 um: 8 same, 142 different, 126 unresolved (mostly
  no power: sparse units, < 3 expected cross-refractory coincidences). Same: DS48 0922
  557/932/1878/1882 (user verdict; keep 1882; but the rule finds no dip for 1878/1882, 16 vs 12.8,
  or 932/1882, 4 vs 6.1), 996/1908 dSPN (0 vs 3.0, borderline power, size difference 0.71),
  DS48 0923 1723/2407 (2407 88% zero-lag copies of 1723; keep 1723). Double counted: 0 iSPN,
  1 dSPN, 4 muSPN. Of the 7 iSPN+iSPN review pairs, 3 are different (1203/1749, 1048/1180,
  7/1627: no dip with 3.4-8.6 expected) and 4 unresolved (1203/1669, 46/1477, 526/1475,
  544/2175: similar templates 0.83-0.93, 0 copies, < 2.4 expected); up to 5 iSPN (+408/412)
  could still be double counted -- judge by eye.
- False-positive review of the 87 iSPNs done (2026-10-12; user asked to test all of the above):
  analysis_files/tagging_tests/ispn_fp_review/ispn_fp_review.csv (12 flags with numbers, score =
  flag count, empty verdict/notes columns) + the existing waveforms_by_light figure of each copied
  into <session>/ (session_waveforms reads the preprocessed recording on D:, so not regenerated).
  Scores: 0 flags 41, 1 22, 2 12, 3 5, 4 5, 5 2. Flags: latency 20 (first spike no earlier than
  chance), sparse 17, weak orange 13, no dose 8, top power only 7, duplicate 7, rp 7, onset-locked 5,
  baseline > 6 Hz 4, blue equal 2, waveform change 1. Notable: DS48 0922 146 and 1890 respond at
  1.3-1.4 ms with 82-86% of the excess in 1-3 ms after onset, to blue as well -- artifact-like; a
  weak-response cluster (orange p 0.002-0.03, rho < 0.2, one power only: DS48 0924 61, DS15 0702 850,
  DS45 0811 56, DS2 0506 380, DS15 0707 1499). None of the 87 were in the user's earlier review.
  Script: scratchpad ispn_fp_review.py.
- Tagging-rule tests run (2026-10-12, user asked; read-only; scratchpad tagging_tests/rule_tests.py
  -> analysis_files/tagging_tests/rule_tests_units.csv, rule_tests_totals.csv). Firing test
  recomputed per unit (10-ms window reproduces the saved light_classif 100% in all 17 sessions) +
  a one-sided blue > orange MWU. Well-isolated dSPN / iSPN / muSPN:
  baseline 109 / 87 / 33; test1 (both -> iSPN unless blue significantly stronger) 109 / 103 / 17;
  30-ms window 111 / 89 / 32; whole-pulse window 116 / 90 / 32; test1 + 30 ms 111 / 103 / 18;
  test1 + whole pulse 116 / 104 / 18. Test1: 17 muSPN -> iSPN, 1 iSPN -> muSPN (DS48 0922 1890,
  blue > orange p 0.001); 15 of the 17 gains in DS45 0820 / DS48 0922 / 0923, orange and blue
  responses about equal (e.g. 0922 1876 157 vs 152 Hz); 4 of them are the 0922 932/1878/1882/557
  group the user called one neuron. Wider windows add weak dSPNs (+1-4 Hz responses).
- Combined (scratchpad tagging_tests/combined_yield.py -> tagging_tests/combined_yield.csv):
  the user had already reviewed 17 of the 18 units the dual-driven rule changes and called only 5
  iSPN (0820 12, 0923 630 / 1103 / 1805 / 2407), 9 muSPN, 2 not tagged, 1 weak muSPN -- so with
  the review applied the rule adds nothing (95 iSPN either way). Best-case well-isolated counts:
  current rule + confident review - duplicates (0922 557/932/1878 -> 1882, 996 -> 1908, 0923
  2407 -> 1723) - onset-locked artifacts (DS2 0511 130, DS48 0922 146 / 1890) = 107 / 91 / 23;
  with the 30-ms window and dual-driven rule 109 / 94 / 21; minus the 7 FP-review units with
  >= 4 flags 107 / 85 / 23. Awaiting the user's verdicts (ispn_fp_review.csv, spn_pairs).

### 35. Combined merge + dedup policy test — running (2026-10-12)
User: merging and deduplication still matter, above all for drifting units that get split; asked
to run the best combination. Scratchpad combined_test.py (copies: analyzer_clean_combtest_merged,
analyzer_clean_combtest; outputs analysis_files/<a>/<s>/combined_test/): on top of the cleaning's
own merges, merge_scan 'merge' pairs + pair_rule 'merge' pairs, vetoed where the refractory rule can
judge and says separate/partial, groups grown greedily with a whole-group refractory check, then
template-match double-detection deletion, then every unit re-tagged. Sessions DS48 0923, DS2 0511,
DS48 0922, DS45 0821, DS15 0702.
- User criterion (2026-10-12): a good merge + dedup policy raises presence ratio, above all of tagged
  units, without costing SNR (or isolation). combined_eval.py -> merge_scan/combined_eval_
  {merges,sessions}.csv: per merged unit presence/SNR of best member vs after, amplitude cutoff,
  rp contamination, l_ratio; per session presence/SNR of the well-isolated SPNs before vs after.
- Result, 5 sessions (2026-10-12): 42 merges. Only 7 raise presence by > 0.05, all handoffs: 0923
  41+42 0.47 -> 0.69 (iSPN, 297 spikes, mua), 0511 230+231 0.66 -> 0.85 (good dSPN, the one SPN
  gain), 0922 294+296, 0821 162+490 0.70 -> 0.86, 0702 276+1178, 327+330, 47+51. 31 merge units
  already present all session (best member presence ~1), no gain; 15 of 42 lose > 20% SNR vs the
  best member (0923 907+2397 17.2 -> 8.8, 1050+2261 12.0 -> 4.0; 0511 1364+1366 9.0 -> 5.4).
  SPN yield: +1 dSPN, -2 muSPN; SPN median presence unchanged; DS2 0511 SPN median SNR 7.6 -> 6.8.
  Conclusion: merge candidates are mostly coexisting amplitude splits, not drift handoffs.
  Proposed: presence-gated merges (union presence >= best member + 0.05, SNR >= 0.8 x best
  member's, refractory checks as now); coexisting same-neuron pairs counted once (keep the better
  unit) instead of merged. Awaiting the user.
- User (2026-10-12): convinced re-merging + dedup belongs AFTER all splits and peels (no
  merge-before-split test wanted). Asked what is thrown out before that stage. Cleaning reports,
  17 sessions: 943 final-noise units (16.0M spikes, 9.5%) dropped; close pairs 2.56M pre-merge +
  0.17M post; small outlier clusters 263k; light artifacts 842; nan 114.
- Lost drift pieces among the dropped noise units? (scratchpad noise_pieces.py ->
  merge_scan/noise_pieces*.csv; 879 whole-KS-cluster noise units vs kept units within 60 um:
  template cosine, presence of kept vs union, minute overlap, union refractory rule, union SNR).
  544 pairs; 79 with cosine >= 0.8, of which only 9 look like handoffs and 16 pass the refractory
  rule. One candidate meets every test: DS2 0505 noise 1055 (KS 403, 2,458 spikes, failed
  exp_decay) -> good unit 1059 (untagged), 7 um, cosine 0.96, presence 0.81 -> 1.0, SNR 5.8 ->
  5.4, 0 union violations. No tagged SPN is missing a piece. The big presence "gains" come from
  large noise clusters present all session that would wreck SNR and fail the refractory rule.
  64 noise split/peel pieces (411k spikes) cannot be rebuilt (membership not saved).
- User (2026-10-12): DS2 0505 noise unit 17 should not have been thrown out. It failed only
  bombcell's spatial decay (exp_decay 0.0072; noise unless 0.01 < exp_decay < 0.1 /um, SI default,
  probes NP2014). Without that criterion it is good: SNR 5.4, -70 uV, presence 0.97, 0 spontaneous
  violations, 0.4% shared spikes (untagged). 1055 (exp_decay 0.0084) would be a good dSPN. Kept
  good units sit at exp_decay 0.016-0.081 (1st-99th pct). 112 units were dropped for exp_decay
  alone (29 low side, 82 high side). In DS2 0505 the high-side ones are junk (6-24 uV, SNR 1-3,
  thousands of violations). scratchpad decay_rescue.py -> tagging_tests/decay_rescue/ running on
  all 17 sessions (label without the criterion, tag on the KS train, duplicate check).
  - Result (111 whole-cluster units, 14 sessions): the DS2 0505 pattern does NOT generalise. Low
    side (< 0.01, 25 units) median SNR 1.9, -9 uV, contamination lower bound 1.0 -- mostly junk;
    only 17 and 1055 are good. High side (> 0.1, 83 units) median SNR 9.8, -58 uV, presence 0.46 --
    mostly large short-lived units; 4 would be good, all just above the cut (exp_decay
    0.101-0.12: DS15 0702 1616, DS45 0814 732, DS45 0820 798, DS48 0923 2365). Without the criterion:
    6 more good units (one good dSPN, DS2 0505 1055), 25 tagged MUA (not counted). Junk dropped by it
    would mostly fail SNR/amplitude/rp anyway (-> mua, not counted). Proposed: exp_decay as a flag,
    not a noise criterion (or widen to ~0.005-0.15). Awaiting the user.
- User decision (2026-10-12): re-clean with exp_decay widened to 0.005-0.15, and keep small
  outlier clusters (< OUTLIER_KEEP_MIN_SPIKES) when their shape passes bombcell's noise criteria or
  they have < OUTLIER_EVAL_MIN_SPIKES = 50 spikes (too few to judge; kept as mua). Reverses the
  2026-10-06 drop (those clusters had lowered their parents' PCA isolation metrics -- check).
  Code: spike_cleaning.EXP_DECAY_RANGE in default_thresholds; step 6 no longer drops small '/out'
  clusters, step 7 relabels too-small noise ones to mua (report keys small_outlier_clusters*).
  Then re-test the merges: a graded candidate list (my verdict), the user grades it, and we derive
  merge criteria together.
  - Running (scratchpad reclean_all.sh, log reclean_all.log): per session spike_cleaning ->
    spike_cleaning_pre_reclean (score cache copied back), curate_subject --steps=clean,pynapple,
    isolate (analyzer_clean -> analyzer_clean_prev; tagging outputs -> pynapple/prev_curation),
    staged to C:\si_stage\reclean. Test analyzers (analyzer_clean_*test*) now refer to the OLD
    curation's unit ids.
  - DEFERRED (user, 2026-10-12: forecast ~25-30 h, too long for now; "make a note to test/run this
    later"). Cancelled during DS2 0505's split step: nothing saved; the three DS2 spike_cleaning
    folders restored from spike_cleaning_pre_reclean; analyzer_clean untouched everywhere; the
    staged copy deleted. TODO later: run scratchpad reclean_all.sh (or curate_subject
    --steps=clean,pynapple,isolate per subject, with the spike_cleaning backup). The code changes
    (EXP_DECAY_RANGE, small-outlier handling) ARE in spike_cleaning.py now: any relabel /
    _metrics_and_labels call uses the widened decay range, so labels recomputed on test copies can
    differ from the saved ones for units with exp_decay in 0.005-0.01 or 0.1-0.15.
- Next (user): graded merge-candidate list on the current analyzer_clean.
  - Done (2026-10-12; scratchpad merge_candidates.py): analysis_files/merge_candidates/
    merge_candidates.csv, 627 pairs (merge_scan merge/review + pair_rule merge), one figure each in
    figures/. My grades: A merge 23 (2 with a tagged SPN: DS48 0923 41+42 iSPN, DS2 0511 230+231
    dSPN), B same neuron / count once 104, C uncertain 413, D different 87; 157 involve a tagged SPN.
    suggested_review = 99 (all A, SPN-involved B and D, top 20 SPN-involved C by presence gain).
    Empty user_grade / user_verdict / user_notes columns. Awaiting the user's grades.
- User (2026-10-12): put the merge and dedup into the pipeline, at the end of splitting and
  peeling. Done in code, NOT applied (needs the deferred re-clean):
  - postprocessing/pair_merge.py (new): refractory test, mixture-cluster test, presence/SNR gate,
    group veto; double_detections (excess test, template-match side, one copy kept).
  - spike_cleaning: PAIR_MERGE preset in MERGE_PRESETS (after DRIFT, before REDUNDANT; auto_merge
    takes stim_intervals; evidence -> spike_cleaning/pair_merge_candidates.csv); step 6b
    (DELETE_DOUBLE_DETECTIONS) after the junk removal: involved units re-scored with
    find_outlier_spikes, copies dropped (removed_spikes reason 'double detection'),
    double_detections.csv, report keys double_detection_*.
  - Test on DS2 0511 analyzer_clean in memory (scratchpad test_pair_merge.py; scores from the cache):
    PAIR_MERGE merges exactly 230+231 (the A handoff); 209 candidates. Dedup first flagged 462
    pairs (55.6k spikes) by shared channels alone -- 108 extra pairs at median 146 um (synchrony /
    shared artifacts, not double detections) -> DD_MAX_UM = 60 um added: 184 pairs, 38.3k spikes
    (earlier overlap test 30.5k). Spike accounting and no pairs <= W after drop_spikes: OK.
  - Side effect of the test: get_stimulation_intervals re-wrote DS2 0511's pynapple
    binary_signals_events*.csv and binary_signals/trial_raster.png (regenerated from the same npz).
  - Merge criteria (thresholds in pair_merge.py) are provisional until the user grades
    merge_candidates.csv.

### 33. Close pairs: overlap deletion and a merge / keep-separate rule — test running (2026-10-11)
User: close pairs that contaminate each other (DS45 0811 877/879) need their overlapping spikes
found and deleted; then a way to decide whether a close pair is merged (and decontaminated) or
kept separate (and decontaminated). User: shared refractory violations between two close,
similar units are duplicate spikes of one neuron, not evidence of two.
- Overlap deletion (scratchpad `overlap_dedup.py`, test copies `analyzer_clean_overlaptest`,
  outputs `analysis_files/<a>/<s>/overlap_test/`): pairs sharing >= 2 KS PC channels; a narrow
  (0.2 ms) cross-correlogram peak within +-0.5 ms, Poisson p < 1e-4 after Bonferroni, excess >= 10
  and >= the chance count; the excess copies deleted, the worse-fitting copy (Gaussian per unit on KS
  PC features of spontaneous spikes) at a simultaneous peak (|lag| <= 0.1 ms), only copies that fit
  the other unit 10x better at an offset peak. Every multiply detected spike keeps one copy.
  - DS15 0701: 213 of 1136 pairs flagged, 38k spikes from 146 units (median 0.5% of a unit,
    90th pct 5%); mostly mua-mua. Evoked spikes not over-deleted (median light share of deleted /
    overall 0.68). 782 lost its dSPN tag: its 17 Hz response was double detections of 774's
    63 Hz one (774 bU), so the tag was borrowed. 576 mua -> good.
- Rule (scratchpad `pair_rule.py` -> `merge_scan/pair_rule.csv`): spikes <= 1 ms apart are
  double detections in either outcome and decide nothing; the verdict is the cross-refractory
  window (1, 1.5] ms on spontaneous spikes vs the 2-10 ms flank. merge = powered (expected >= 3),
  significant dip (Poisson p < 0.05), the dip as deep as the labels' tolerance (95% lower bound
  of observed/expected <= 0.10) and the union passes the rp rule; else partial / separate /
  unresolved (no power).
  - Even pairs 20-80 um apart sit at a median ratio ~0.74, not 1 (sorter suppression of close
    detections), so a significant but shallow dip is not one neuron: 877/879 0.86 on 1245
    expected -> separate.
  - Reviewed pairs: 723/1839 merge; 877/879, 803/805 separate; 932/1882 (4 vs 6.1) and
    1878/1882 (16 vs 12.8, ratio CI 0.70-1.87) separate, against the user's note; 932/1878 and
    53/1695 unresolved. Of the 73 merge-test merges 7 pass (5 good); 61 other pairs pass.
- Scan of all 17 sessions with the rule (merge-verdict pairs skipped), 2026-10-11:
  `merge_scan/overlap_pairs.csv` (49,395 pairs, 6,133 flagged: 4,301 separate, 1,367 partial,
  407 unresolved, 58 merge) and `merge_scan/overlap_units.csv`. Median unit loses ~0.1% of its
  spikes. All 17 done after the fix below. Labels (corrected): 5 mua -> good, 3 good -> mua,
  8 -> noise, 2 units removed whole; 28 tag changes (9 dSPN -> untagged, 3 untagged -> dSPN,
  2 untagged -> iSPN, ...).
  - DS2 0505 and DS48 0923 failed: a unit whose every spike was a copy (drop_spikes refuses to
    empty a unit). Now such a unit is removed whole (0505 511, 0923 73); both re-run.
  - Bug found: `analyzer.save_as(format='memory')` returns STALE sorting properties (the
    pre-2026-10-08 bombcell_label, before the stimulation-free relabel) although the analyzer on
    disk has the new ones. The scan's "label before" was wrong for those units (it showed 19 mua ->
    good); corrected in merge_scan/overlap_units.csv and overlap_pairs.csv (the per-session
    overlap_test/pairs.csv keep the stale labels) and overlap_dedup now reads labels from disk.
    The merge test's before-labels were checked: 0 of 153 stale. Any code that reads
    properties after save_as(memory) is suspect (dedup_apply, resplit_test, merge_test use it).
  - OPEN problem: when one unit is mostly a copy of another (75 simultaneous pairs where > 50% of the
    smaller unit is copies), the per-spike "worse fit" choice splits the deletions between both
    units (15 pairs > 10% each way), scattering one neuron's spikes over two units: DS15 0707
    452/623 (15,926 of 452's 15,944 spikes copied; 531 deleted from 623), 1731/2123 (26,317
    shared, ~13k deleted from each). Proposed: at a simultaneous peak delete the copies from the one
    unit they make up the larger share of, unless the features clearly say otherwise.
    User agreed (2026-10-11). Implemented: copies go from that unit unless the other unit's copy
    fits it 10x better (MIN_SCORE). Check on 0702 212/216: 805 of 216's 809 deleted, none from
    212 (stays good). Full 17-session re-run started; previous tables kept as
    merge_scan/overlap_{units,pairs}_v1.csv.
  - Re-run done (2026-10-11): 393k spikes from 4,051 units (v1 394k); pairs split > 10% both
    ways 1 of 46 (was 15). Labels: good -> mua 7, good -> noise 2, mua -> good 5, -> noise 16
    (mostly near-pure duplicate units: 0707 452 15,926/15,944, 2123 26,241/33,909, 0702 216),
    removed 2; 32 tag changes.
  - NEW problem: the larger-share side disagrees with the per-spike-fit majority in 935 of 2,008
    simultaneous pairs. DS15 0701 634 (mua, 49k) / 656 (good, 2k): 1,792 shared; the fit took all
    1,777 from 634 (v1, 656 stayed good), the share rule takes them from 656 -> noise. DS48 0921
    945 (mua 18k) / 1119 (good 244k): fit 11,839 from 1119, share 12,887 from 945 -> noise.
    Proposed: decide the side per pair by the summed per-spike fit (the v1 evidence, applied to
    the whole pair so it does not scatter). User agreed (2026-10-11); implemented (side = sign
    of the summed sa - sb), check 0701 634/656: 1,777 from 634, 656 stays good. Third full re-run
    started; the second run's tables kept as overlap_{units,pairs}_v2.csv.
  - User question (2026-10-11): do merges + deduplication belong before the cleaning? Proposed
    order: bimodal splits -> noise-unit drop -> pair decision (merge / overlap deletion, replacing
    the strict-merge step) -> outlier spikes -> within-unit pairs (absorbs merged duplicates) ->
    labels. To test on 1-2 sessions (copies) after the third scan; awaiting the user.
  - Multi-agent review of that plan (user asked, "ultracode", 2026-10-11; 6 lenses + skeptics,
    workflow wf_fedab258-c92; reviewer scratch in scratchpad review_stat/, review_order/):
    - Not before all cleaning. On raw KS trains pair_rule loses 12 of 46 merges (26%), gains none:
      <= 1 ms within-unit doublets inflate the (1, 1.5] ms counts ~13%. Merge decisions need
      cleaned trains.
    - The plan described an outdated order: the cleaning is now greedy split + peel in one loop
      (greedy_cleaning.py:325-388; re-split after every peel), noise units are only flagged
      (spike_cleaning.py:1539-1561) and the <= 1 ms pair step already precedes the merges
      (:1549). CLAUDE.md step 4 is out of date (to fix).
    - Recommended order: <= 1 ms within-unit pairs BEFORE any split (the GMM splits double
      detections apart: DS2 0506 KS 309 2223/2223, KS 316 982/982 close pairs straddle the halves)
      -> greedy split+peel, light artifacts -> noise flag -> pair decision (pair_rule + dedup)
      replacing the strict merges, keeping the OUTLIERS re-homing (35) and DRIFT (45) merges
      (pair_rule has no candidate generator; drift handoffs come out unresolved) -> <= 1 ms pairs
      in merged units with a `prefer` mask -> labels. Check merge groups (pair_rule merges form 6
      non-clique groups) with resolve_merging_graph -> _merged_rp_veto -> are_units_mergeable.
    - BIGGER problem than the order: the dedup copy scoring was ill-conditioned (np.cov + 1e-3
      ridge on PC dimensions with zero variance: likelihood ratios 1e4-1e7; the side flipped
      between np.cov and MCD in 19 of 105 DS15 0701 simultaneous pairs, between clean and raw
      trains in only 4). The third scan (v3) was stopped mid-run (5 sessions done) for this.
    - Evoked: do not drop in-light coincidences from detection (real zero-lag duplicates of evoked
      spikes, e.g. 155/1501: 110 in one 0.2 ms bin vs 17-37 around); a stimulus-aware null (shift
      predictor) would be better. Log dedup deletions in removed_spikes.csv ('double detection').
    - Refuted: the split guard's +-1.5 ms window caused split halves to be merged back (it was low
      power: 5 of 6 had < 20 expected).
    - pair_rule still disagrees with the user on the 3 DS48 0922 pairs; needs footprint/handoff
      evidence, not order.
  - Scoring fix (2026-10-11, overlap_dedup.py; old version overlap_dedup_v3.py): dimensions with no
    variance in either unit dropped, MCD (Ledoit-Wolf when MCD's support is degenerate), ridge 1%
    of the median feature variance, side by the MEDIAN score, a copy that fits its own unit 10x
    better stays (both stay if both do), a unit with < 50 spikes outside the coincidences is all
    copies. Validating on DS15 0701.
    - Result: 18k deletions (was 38k) and 28k coincidences with BOTH copies kept -- because
      Kilosort pc_features are shaped by the template that claimed the spike: the same spike at the
      same sample has different features in the two units, each nearer its own unit's mean (634/656:
      100% of copies; 736/746 79%/66%). So pc_features cannot arbitrate copies at all; every scoring
      version so far (v1-v3 and this fix, kept as overlap_dedup_ksfeat.py) rested on them.
  - Template-match scoring (2026-10-11): each spike's raw-waveform correlation with its own raw KS
    unit's template, from the cleaning's score cache (spike_cleaning/outlier_scores.pkl, coverage
    100%). The template that explains the same waveform better keeps it -- the cleaning's own rule for
    pairs within a unit. Not normalised per unit (656: copies match 0.43, its other spikes 0.16; 634's
    copies of the same spikes 0.82). One side per pair (median), the excess copies deleted from it,
    simultaneous and offset peaks alike. Validating on DS15 0701.
    - Full run, all 17 sessions (finished 2026-10-12 02:18; user approved; no staging needed, it
      reads no recording): 895k spikes from 4,051 units (v1 393k: offset peaks now lose their full
      excess); median unit 0.2%, 90th pct 3.4%, 99th 28%. Labels: mua -> good 6, good -> mua 4,
      good -> noise 2 (0701 656, 0812 410: near-pure copies), 0707 452 non_soma_good -> noise;
      44 tag changes (9 dSPN -> untagged, 5 dSPN -> muSPN, 5 muSPN -> iSPN, 4 untagged -> dSPN,
      ...). 1,051 of 6,133 flagged pairs have templates matching the copies about equally
      (median difference < 0.05; 95 involve a good unit): side close to arbitrary there, listed in
      merge_scan/overlap_ambiguous_pairs.csv. Tables: merge_scan/overlap_{pairs,units}.csv.
  - The overlap_all.sh units aggregation wrote an empty csv (doubled backslashes in a heredoc
    path); rebuilt by hand.

### 31. Refractory rule without stimulation; firing-test-only tagging; iSPN false negatives — running (2026-10-08)
User: opto-driven SPNs can fire near 1 kHz (well-isolated, otherwise quiet units driven that
hard by blue light), so leave the stimulated periods out of the rp/ISI criterion of the
labels, without re-cleaning; reclassify every dataset once the last two finish cleaning and
replot all tagging plots. Some units with poor light responses are tagged: (1) if the latency
(TTFS) test alone can tag a unit, stop that; (2) otherwise widen the 10-ms peak window to
max(30 ms, the pulse). iSPN yield matters most: find the causes of iSPN false negatives.
- (1) was true: `_classify` took the TTFS label and the firing test only filled in
  (`ttfs.where(ttfs.notna(), firing)`). Now which opsins drive a unit is the firing test's
  call alone (`fire_responds_<opsin>`: peak above the pre-pulse baseline and rising with
  power); driven by both, iSPN when orange is the stronger by either test (firing test's
  orange > blue, or TTFS LRT with a steeper orange slope), else muSPN. (2) not done, as the
  user made it conditional on (1) being false; still open as an option.
- Labels: `spike_cleaning.spontaneous_rp` + `_bombcell_with_rp_minimum(stim_intervals=...)`:
  violations, spike count and duration outside every light pulse + RP_STIM_PAD_S = 100 ms.
  Used by relabel_session / write_label_table (scripts/clean_sessions.py) and by future
  cleanings (`_metrics_and_labels`). The cleaning's merge veto still counts every spike.
  Audit columns rp_violations_all, rp_spikes_counted.
  - Dry run on the 15 finished heavy sessions (scratchpad `rp_stim_dryrun.csv`): 31 units
    mua -> good, none the other way. Caveat: some fire almost only during light (DS2 0511
    1562: 17 of 3445 spikes spontaneous; 1648: 140 of 10668); their refractory test has no
    power. The re-run prints newly good units with < 100 spontaneous spikes.
- Every unit's test statistics are now kept (`fire_*`, `ttfs_*` columns of
  singU_metadata.csv / all_units_tagging.csv). `ispn_false_negatives(analysis_folder,
  unit_labels)` -> `pynapple/ispn_false_negatives.csv`: units with any orange evidence that
  are not iSPN, with every blocking step (firing test half-met, muSPN, fast baseline, quality
  with the bombcell criteria failed); `n_reasons` 1 = near miss.
- Run (scratchpad): `retag_heavy.py` (labels -> unit_ids/unit_labels.csv, spikes.npz,
  isolate_single_tagged_units, iSPN table; backups `*_pre_stimrp`) and `replot_heavy.py`
  (all tagging plots; old waveforms folder kept as `waveforms_by_light_pre_stimrp`). The
  workers' relabel step (`relabel_heavy.py`) now runs retag_heavy, so DS45 0819 / DS48 0924
  get it directly. Tested first on DS2 0506.
- Only newly good unit with < 100 spontaneous spikes so far: DS2 0511 1562 (17 of 3445
  spikes outside light, 0.004 Hz baseline, +158 Hz blue / +51 Hz orange, muSPN). User
  reviewed the figure: "looks like a real neuron" -- kept as good.
- Run finished 2026-10-09 00:29, all 17 heavy sessions. Good-unit tags before -> after:
  iSPN 79 -> 87, dSPN 100 -> 109, muSPN 42 -> 33, bU 47 -> 48, oU 27 -> 18, mU 8 -> 6
  (DS45 0819 / DS48 0924 "before" already had the firing-test-only tagging); 31 units
  mua -> good. Summary: `C:\si_stage\root\analysis_files\ispn_false_negatives_all.csv`
  (scratchpad `ispn_fn_summary.py`). 866 units with orange evidence are not iSPN, 648 of them
  MUA; 113 good units are blocked by one step only: 42 orange rate rising with power but not
  above baseline (all powers pooled), 33 muSPN (blue too, orange not significantly stronger),
  18 oU (fast), 15 latency only, 5 above baseline but not rising.
- Evaluated (scratchpad `baseline_rule_eval.py/.csv`), not applied: "above baseline" judged
  A at the strongest power, or B at any power Holm-corrected, instead of pooled. Good units:
  A iSPN 87 -> 91, dSPN 109 -> 104, muSPN 33 -> 29, oU 18 -> 15; B iSPN 87 -> 83, dSPN
  109 -> 98. Of the 42 pooled-baseline near misses only 8 pass at the strongest power and 3
  with Holm: most are weak responses (pooled P median 0.16), not low-power dilution. Neither
  rule is a net gain. Individual dilution cases A recovers: DS2 0506 135, DS15 0703 372,
  DS48 0923 1491, DS48 0922 27, DS48 0924 one unit.
- muSPN near misses (39 good units): 21 clearly blue-dominated (P(orange > blue) >= 0.95);
  18 with orange about equal to or above blue without significance. Option offered, not
  applied: dual-driven = iSPN unless blue is significantly stronger (+18 iSPN).
- 2026-10-09, manual review list (user): `C:\si_stage\root\analysis_files\ispn_review_list.csv`
  (scratchpad `ispn_review_list.py`), 68 good units, ordered by category then dose-response P:
  1a above baseline at the strongest power only (8), 1b rising with power but weak (34),
  2 muSPN with orange ~ blue (17), 4 latency only, slow (1), 5 above baseline not rising,
  slow (3), 6 iSPN-like at 8.42-15 Hz -> oU (5). No MUA unit with an iSPN response failed
  only one criterion. The 36 without a waveforms_by_light figure were drawn
  (`session_waveforms(..., units=[...])`, `unit_<id>_untagged.png`; scratchpad
  `review_waveforms.py`); paths in ispn_review_list_new.csv (the original was open in Excel).
  Awaiting the user's review.
- 2026-10-09, merge-then-resplit test (user), in memory only, scratchpad
  `merge_resplit_test.py` -> `C:\experiment_data\analysis_files\merge_resplit_test\`
  (summary.csv + one figure per group): pool a group's random spikes (uV waveforms on shared
  channels; the whitened PCs hid the amplitude difference and gave spurious "re-split"
  verdicts), drop cross-unit duplicates, GMM 1-3 components by BIC (components < 10% = outlier
  tail), compare with the units (ARI); merged unit's spontaneous refractory stats.
  - 17 groups: 10 keep separate (ARI 0.95-1.00, amplitude d' 1.5-5.6, mostly a weak iSPN
    beside a low-amplitude MUA cluster); DS48 0922 1876/1880/1896 share no channels.
  - DS48 0923 53 (weak iSPN) + 1695: 1695's spikes at 0-300 s and 2700-3100 s sit in 53's
    amplitude band (~-120 uV), its middle-session spikes at ~-200 uV: probably two neurons
    swapped between the clusters by drift -> resplit candidate that would make 53 less sparse.
  - DS48 0922 932+1878+1882+557: waveforms overlap, merged unit refractory-clean (26
    spontaneous violations, contamination lower bound 0.04): consistent with one neuron (the
    user's note); the mixture's split follows the amplitude drift after ~2500 s.
- 2026-10-09, applied as a test on copies (scratchpad `merge_test.py`; saved sorts untouched;
  `sorting_files/DS48/<s>/analyzer_clean_mergetest`, `analysis_files/DS48/<s>/merge_test/
  {before,after,report.json}`):
  - DS48 0922 932+1878+1882+557 -> unit 2027: 44542 spikes, good, 26 spontaneous violations
    (contamination lower bound 0.043), orange 194 Hz / blue 184 Hz -> muSPN, but its quiet
    baseline (8.58 Hz, the four summed) is just over TAGGED_MAX_BASELINE_HZ 8.42 -> mU.
  - DS48 0923: 1695 split by amplitude (GMM means -186 / -104 uV) and the -104 band merged
    into 53 (cosine 0.83) -> 2428 "iSPN". REJECTED: the -104 band is 97% light-evoked spikes
    (3% between the stimulation blocks at 10-313 / 2717-3092 s), the -186 band 83%
    spontaneous: 1695 is one iSPN whose evoked spikes are smaller at high rates, and the merge
    gave 1695's light response to 53. 53 and 1695 stay as they are. Lesson: an amplitude
    comparison between units must use spontaneous spikes (evoked spikes attenuate).
- 2026-10-10, merge test on all 17 latest sessions (user: "run the merge test on all datasets ...
  just run it so I can review it all"): scratchpad `merge_scan.py` -> `C:\experiment_data\
  analysis_files\merge_scan\{pairs.csv, groups.csv, figures/}`. Candidates: pairs within 50 um
  with template cosine >= 0.8, any quality. On SPONTANEOUS spikes: duplicates, GMM on uV waveform
  PCs of shared channels (ARI vs units), amplitude d', merged refractory test. Merge when ARI <
  0.3 (or one cluster), d' < 1 and the merged unit passes; keep separate when ARI >= 0.8 or the
  merged unit fails; review otherwise. Groups = connected merge pairs, re-checked.
  - DS48 0922 on spontaneous spikes: 932+1878 merge (d' 0.37); 1882 differs in spontaneous
    amplitude (d' ~2 vs 932/1878) -> review; 557 separate. The earlier 4-way merge leaned on
    evoked spikes.
  - `merge_apply.py`: merges applied to copies (`analyzer_clean_mergetest`), merged units
    re-tagged and redrawn, `merge_test/{before,after,report.json}`, overall
    `merge_scan/applied.csv`. The first tests' outputs keep the suffix `_first`.
  - Scan: 2915 candidate pairs; 83 merge groups (79 pass, 4 fail the group refractory check).
    Applied (one session per process; the scratchpad Lock now records its pid and clears a dead
    owner's lock; faulthandler on): 73 merged units in 17 sessions; 6 groups skipped because
    SpikeInterface's soft merge needs >= 75% channel-footprint overlap (DS2 0506 320+326+328,
    DS45 0811 877+879, DS45 0821 181+186, DS48 0923 1829+1833, DS48 0924 427+431, 801+807).
    After: 48 good (39 untagged), 22 mua, 3 non_soma_mua; tags dSPN 6, iSPN 2, muSPN 1, bU 4, oU 1.
  - To review first: DS45 0821 803 (good, untagged, ~125 Hz) + 805 (good iSPN) -> 1369, tagged
    oU (fast baseline) with 967 spontaneous violations: this merge removes an iSPN and looks
    wrong. Fast units merged with 805-1058 violations still pass the rp rule (huge spike
    counts): DS45 0814 935, DS48 0921 1287/1289 (bU).
  - Possible gains: DS2 0511 230 (good) + 231 (mua dSPN) -> good dSPN; DS48 0923 723 (mua dSPN)
    + 1839 (good) -> good dSPN; DS48 0923 389 (good) + 2039 (good dSPN) -> good dSPN;
    DS48 0922 412 (good iSPN) + 5 (mua) -> good iSPN, 1068 spikes; 932+1878 -> good muSPN.
  - Dedup-only alternative (user: test "when you clean out duplicate spikes"; scratchpad
    `dedup_apply.py`, copies `analyzer_clean_deduptest`, per session `merge_test/dedup_report.json`,
    all in `merge_scan/dedup.csv`): every group's units kept separate, the smaller member's spikes
    within W_MS (1 ms) of a larger member's removed, labels recomputed, units re-tagged. 83 groups,
    178 units: duplicates in 49 groups, 83,195 spikes in all, but mostly in fast units (DS48 0924
    91+529+537 alone 59,519; median 0.3% of a group's spikes). No label changed; one tag (DS45 0811
    879 dSPN -> untagged once its 2279 duplicates of 877 went). So deduplication alone does little;
    the merges are what change labels (MUA fragments -> good units).
    DS48 0923's run needed the test copy's templates trimmed to average/std (SpikeInterface's
    percentile split fails when a dropped half holds no random spikes).
  - merge_scan figures (`merge_scan_figures.py`): amplitude over time, waveforms on the group's 4
    largest channels (shared y), the probe (members' stars, the group's shank), the cross-
    correlogram of the two largest members (1 ms duplicate window shaded) and the merged unit's
    autocorrelogram after the merge's censoring; title: both outcomes. E.g. DS48 0923 723+1839: 723
    (mua dSPN) fires 0-480 s, 1839 (good) from 480 s at a similar waveform, 7 um apart, 0 merged
    violations -> merge gives a good dSPN of 6554 spikes; dedup alone removes 3 spikes.
  - Merge rule weaknesses seen in the figures, to fix before any real merge: DS45 0821 803+805
    (different spatial footprints, 967 violations allowed by the lower bound on 290k spikes);
    needs an absolute violation cap and a footprint-similarity check.

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

### 22c. Which mixture sets the bU/oU/mU cutoff — awaiting the user (2026-10-06)
User: why not exponential + Gaussian ("quiet SPNs seem to have exponentially distributed
baseline"); then also two lognormals. Scratchpad `exp_gauss_cutoff.py`, figure
`compendium/spn_baseline_rate/baseline_mixture_models.png`. 292 tagged units, quiet baseline,
maximum likelihood, Gaussians truncated at 0:

| model | AIC | BIC | whole-mixture KS p | boundary | units above |
|---|---|---|---|---|---|
| exponential (mean 1.94 Hz, 70%) + Gaussian (0 +- 51 Hz, i.e. a broad tail) | 1770.5 | 1785.2 | 0.001 | 8.42 Hz | 80 |
| two Gaussians (the truncated fit; the current 7.8 Hz came from an untruncated GMM) | 1827.3 | 1845.7 | <0.001 | 6.64 Hz | 91 |
| two lognormals (90%: median 1.58 Hz, sigma 2.0; 10%: median 62 Hz, sigma 0.33) | **1746.0** | **1764.4** | **0.78** | **40.1 Hz** | 38 |

- The two-lognormal model is the only one that fits the whole distribution (KS p 0.78) and has
  the best AIC/BIC. But its slow component is very wide (sigma 2 spans ~0.03-50 Hz), so it puts
  the 8-40 Hz tagged units in the SPN population, and only a compact ~45-90 Hz cluster is
  "fast".
- Exponential + Gaussian at 8.42 Hz would move only 4 units back from U to SPN.
- User: "What about exponential + log normal?"
  - Unconstrained, the exponential takes the near-silent units (mean 0.32 Hz, 24%) and one
    broad lognormal (median 5 Hz, sigma 1.73) takes the rest, SPNs included: AIC 1760.3, KS
    p 0.32, boundary 0.52 Hz (213 units above). Not an SPN / fast split.
  - With the lognormal held to median >= 20 Hz, it sits on that bound (20 Hz, sigma 1.11),
    exponential mean 1.37 Hz: boundary 4.91 Hz, 108 units above, AIC 1772.4, KS p 0.021.
    No better than exponential + Gaussian.
- Not applied; the current constant is still 7.8 Hz. The user to choose.
- **User: "Ok let's do 8.4hz"** (2026-10-06).
  - `TAGGED_MAX_BASELINE_HZ = 8.42`, the model's boundary; DS48 0922 unit 705 at 8.41 Hz
    becomes muSPN again. Provenance comment updated.
  - Expected: 80 units bU/oU/mU (4 fewer: DS45 0814 694, DS48 0923 962 -> dSPN; DS15 0707 768
    -> iSPN; DS48 0922 705 -> muSPN).
  - Rerun: `rerun_classes.sh` (isolate, ratio_tests, subject + cross compendia, opto scatters).
  - User: "why are you re-tagging? isn't it just reapplying the labels?" Correct: the isolate
    step re-ran the tagging tests needlessly (~75 min). By then (11:09) every subject was done;
    only the cross-dataset compendium and the opto scatters, which are needed anyway, remained.
  - **Done:** all subjects, the cross-dataset compendium and the opto scatters at 8.42 Hz.
    Totals: dSPN 104, iSPN 69, muSPN 39 (212 SPNs); bU 44, oU 29, mU 7 (80), as expected.
  - Added `isolate_single_tagged_units.relabel_fast_units` and a `curate_subject` step
    `relabel` (`--steps=relabel,ratio_tests`). They re-apply the cutoff to the saved
    tag_classif/baseline_hz in seconds, and the subject compendium is built after relabel too.
    Check: DS45 0814 -> bU 7, dSPN 6, oU 2, iSPN 1 (unit 694 back to dSPN), identical to the
    isolate run.

### 22b. bU / oU / mU split applied — running (2026-10-05)
User asked why so many tagged units have weak light-evoked firing, "especially those with
substantially higher baseline firing rates".
- Finding (scratchpad `weak_tagged.py`): of 292 tagged units, 38 rise < +100% with the
  intensities pooled (median baseline 33 Hz vs 2.4 Hz for all tagged); 19 stay < +100% at their
  best intensity. Most of these are DS48/DS45, tagged by TTFS or the firing test alone with
  tiny effects (e.g. DS48 0924 iSPN 732: 114 Hz, +20%).
- **User chose the split** (writing "mU"). The cutoff is recomputed on the quiet baseline: a
  2-Gaussian fit to the 292 tagged units, equal posterior at **7.8 Hz** (components 1.9 / 40.5
  Hz, D 1.70, Silverman p 0.20). 84 units go above it (DS45 31, DS48 46, DS15 4, DS2 3).
- **`isolate_single_tagged_units`:** `TAGGED_MAX_BASELINE_HZ = 7.8`, `HIGH_BASELINE_CLASS`.
  Each unit's quiet `baseline_hz` (`opto_baseline.quiet_intervals`). `tag_classif` = the
  opsin label; `final_classif` = bU/oU/mU above the cutoff.
- **`pres_plot_utils`:**
  - `TAGGED_CLASSES = (dSPN, iSPN, muSPN, bU, oU, mU)` and `CLASS_COLORS`.
  - 10 per-function `_get_classif` and 6 row orders use them. `_class_cols` + `_spn_rows(class_cols)`
    in spn_mean_psth, spn_rew_unrew_psth, spn_summary and spn_lick_bout_psth (a row per class
    present, then dSPN-iSPN).
  - The heatmaps get a row per class; muSPN used to be pooled into Unlabeled.
  - `_CATEGORY_ORDER` (unit counts, rate, binned distributions) and the opto scatter
    categories include the new classes.
  - Unchanged, dSPN vs iSPN only: `plot_dspn_isPN_diff`, `plot_dspn_isPN_bar`,
    `plot_dspn_ispn_lick_bout_bar`.
  - Backup of the module before the edit: scratchpad `pres_plot_utils_before_classes.py`.
- Test DS48 0923: dSPN 17, iSPN 24, muSPN 12, oU 4, bU 3; spn_summary shows the rows.
- Running (scratchpad `rerun_classes.sh`): isolate + ratio_tests + compendium per subject, the
  cross-dataset compendium, then the opto scatters. Not updated: `spn_table.csv` (SPNs only).
- **Done 2026-10-06 03:09.** All subjects, the cross-dataset compendium and the opto scatters are
  ok; pynapple segfaults were retried (DS15 once, DS48 twice, cross once).
  - Totals: dSPN 102, iSPN 68, muSPN 38 (208 SPNs); bU 46, oU 30, mU 8 (84), matching the
    prediction.
  - By animal (SPN / U): DS2 34/3, DS13 2/0, DS15 33/4, DS45 37/31, DS48 102/46.
  - Awaiting the user's review.

### 27. Rare artifacts left in units; a wrong drift merge — implemented, testing (2026-10-06)
User (looking at the waveforms_by_light plots): DS15_20260702 units 32, 625, 661, 917, 927 and
1261 contain high-amplitude noise; 1292 contains low-amplitude noise. Why did cleaning not
remove them?
- Finding (scratchpad `ds15_0702_artifacts.py`):
  - The sharp, deep, narrow events are only 2-4 per unit (0.01-0.8%), 14 of 17 during light,
    3 near licks: light / lick transients.
  - The outlier scorer acted on none of them: under MIN_OUTLIERS = 5, and they barely move its
    features (PCs + template-match scale + location).
- 1292 = drift merge of KS 496 + 487/2. All 6,595 487/2 spikes are flat on 496's peak channel
  (16% of 1292), so it is a different neuron. It passed the coincidence test (0 of 26: a big
  unit masking the small one's spikes) and the 0.5 similarity floor (~0.6). The two are active
  together all session.
- **User: take the artifacts out algorithmically; implement and test.**
  - **`spike_cleaning.artifact_spikes`:** one chunked pass. Per spike on the unit's peak
    channel: trough depth and width. Dropped if > ARTIFACT_DEPTH_FACTOR (2) x the median depth
    and <= ARTIFACT_WIDTH_FACTOR (0.5) x the median width (>= 2 samples), with no minimum count.
    It runs in clean_and_curate step 1b (the outlier cache is unaffected). Reason 'artifact' in
    removed_spikes.csv; `n_artifacts_dropped` in outlier_audit.
  - **Drift merge:** `_time_overlap` (sum over minutes of min(share of each unit's spikes)) <=
    DRIFT_MAX_TIME_OVERLAP 0.3, and DRIFT_MIN_SIMILARITY 0.5 -> 0.8.
    - For pairs active together too little to test (expected < 20), at most 3 coincidences:
      pieces that hand over with no overlap can now merge.
    - A group checks only its non-adjacent members (no similarity floor; refractory to each
      other), plus the pooled refractory rule. KS 605's chain ends are at 0.67.
    - Measured: KS 605 pieces overlap 0.03-0.05 (similarity 0.85-0.89); 1292's parts 0.73.
  - Dry run DS15 0702 (scratchpad `test_drift2_0702.py`), then the user has two more pipeline
    changes before the full re-clean.
- Dry run results: the shape-only artifact rule flagged 23,129 spikes in 283 units (far too
  broad); light artifacts sit +0.31..+0.44 ms after light onset, a few within 1 ms of offset;
  lick transients are not ms-locked (0% within 2 ms). KS 62 + 63 drift-merged wrongly: 62 is
  ~3x larger and fires in bursts inside 63's active span.
- **User (2026-10-06): "yes add both drift rules. For artifacts, just separate them out into
  their own clusters, don't throw them until the very end, and don't remove all spikes within
  1ms of light onset/offset. the greedy split peel redesign is everything I have in mind for
  the next re-clean, so please set that up, test it, and run it".** -> issue 28.

### 29. Could Kilosort parameters replace the cleaning? — answered, awaiting the user (2026-10-06)
User: "is any of this doing anything I couldn't better accomplish using different kilosort
parameters?"
- Ultracode workflow: 9 agents (KS 4.1.7 source, the sort params and drift of all 20 sessions,
  audit evidence, literature, synthesis, 3 adversarial verifiers). Full report:
  scratchpad `ks_question/FINAL_REPORT.md`; working notes in the same folder.
- **Answer: mostly no.** These stages have no KS counterpart, and KS 4.1.7 never re-examines a
  unit after its final merge:
  - splits: 99 of 1,685 final good/nsg units contain a split child;
  - redundant merges: 43, of 183k doubled spikes, 94% in good/nsg;
  - bombcell noise: 2,025 units, 14% of spikes;
  - the confidence-bound rp rule.

  The merge floors are hard-coded (similarity 0.5, Q12 < 0.05).
- **KS bug (verified by me): `swarmsplitter.py:57` `len(st2 == 0)`.** It is always true, so
  check_CCG always returns False and the refractory veto on splits never fires. KS over-splits
  same-neuron clusters (bursty or drifting units). Present since 4.1.5 per the report; 4.1.4
  has a working veto.
- The stages with a KS counterpart barely change labels:
  - Close pairs: 3.47M spikes, 97.6% outside good units. Every lag is 0.23–1.0 ms, above KS's
    0.25 ms window. Raising `duplicate_spike_ms` would be worse: it keeps the first copy, and
    the template-match choice differs from that in 41–48% of pairs.
  - Outlier peeling: 86% in mua/nsm.
  - Together they decided 2 good labels in the 3 sessions tested.
- Light artifacts (`/art`): probably a legacy of the old preprocessing chain (blanking after
  filtering, then KS's 300 Hz zero-phase high-pass). The new chain already fixes this
  upstream. To check: run the detector on one new-chain tagging session.
- **Bugs found in our cleaning:**
  - REDUNDANT merges units on different shanks ~1 mm apart (0702 187+452 = unit 826: 99.8%
    shared spikes at 0–1 samples, i.e. crosstalk or a channel-map fault). Needs a
    shank/distance check.
  - The outlier-cluster merges undo the peeling, net good ≈ 0, and caused the 0707 loss.
- **Proposal (not run, needs the user's yes):** three re-sorts of DS15_20260702 into new folders
  on C:, ~22 min each.
  - A0: 4.1.7 as is (run-to-run floor).
  - A1: 4.1.4 in a cloned env (live split veto).
  - A2: `cluster_downsampling` 20 -> 10.

  Each is cleaned identically under a scratch root and compared per stage and per problem unit
  (496/487, 62/63, 449, 487, 66, 264, 187/452, 197).

### 30. Heavy Kilosort test sort of DS15_20260702 — running (2026-10-06)
User: trade the cleaning's CPU time for Kilosort GPU time ("try more templates or throw more
processing power to perform sharper clustering"). User: "stop the greedy curation and run the
heavy sort test... just do heavy sort though and compare to old sort".
- **The DS15 re-clean 2 was stopped for this** (0702 and 0707 mid-run).
  - 0701 and 0703 finished with the 1000-spike rule: their `analyzer_clean` is the new one, and
    the greedy run is kept as `*_superseded_4`.
  - 0702 and 0707 still have the greedy-run `analyzer_clean` (the 424-outlier-cluster version).
    Their audit `spike_cleaning_superseded_4` was moved back to `spike_cleaning` so that it
    matches. The stopped run's partial audit is in `spike_cleaning_interrupted` (plots + score
    cache only; deletable).
  - DS15 is therefore inconsistent across sessions until the re-clean is finished.
- **Sort setup:**
  - Kilosort 4.1.4 (working split veto, issue 29), loaded from `C:\si_stage\ks414`
    (`pip --target`); si_env is untouched.
  - The old sort's exact `sorter_params`, except `cluster_downsampling` 20 -> 5,
    `max_cluster_subset` 25000 -> 50000, `cluster_neighbors` 10 -> 20, `n_pcs`/`n_templates`
    6 -> 8, and `max_peels` 100 -> 300.
  - Input: the staged copy `C:\si_stage\DS15_20260702\preprocess` (the file the old sort read).
  - Output: `C:\si_stage\resort\heavy`. Script: scratchpad `heavy_sort_0702.py`.
- **User: re-preprocess too, since DS15 is on the old chain.** Agreed, as a second arm, so that
  the KS-params effect stays separable from the preprocessing effect.
  - scratchpad `prep_new_0702.py` runs the current `build_preprocessing_protocol` (as
    run_pipeline does, no QC). Output: `C:\si_stage\DS15_20260702\preprocess_new`; saturation
    cache in `C:\si_stage\DS15_20260702\saturation`.
  - Then `heavy_sort_0702.py newchain` (heavy + highpass_cutoff 1, do_CAR False) ->
    `C:\si_stage\resort\heavy_newchain`.
  - The three sorts are compared: old; heavy on the old chain; heavy on the new chain.
  - DS15 needs re-preprocessing anyway for the final pipeline (light-artifact legacy, issue 29).
- **Comparison:** scratchpad `heavy_compare_0702.py` -> `heavy_compare/`.
- **Results (2026-10-07):** raw sorts, no cleaning, identical metrics and labels.
  - Sort times: heavy sort 46 min (first clustering 641 s vs 46 s), newchain 40 min.
    Re-preprocessing took 22 min.
  - The parallel PCA fit deadlocked in the analyzer build; with n_jobs=1 it took < 1 min.

  | | old | heavy (old chain) | heavy (new chain) |
  |---|---|---|---|
  | units | 535 | 479 | 392 |
  | KS good | 236 | 246 | 213 |
  | good+nsg (bombcell, rp rule) | 113 | 118 | 103 |
  | noise | 74 | 103 | 52 |
  | pairs 0.25-1 ms per 1k spikes | 5.7 | 7.0 | 8.4 |
  | good median l_ratio | 0.084 | 0.056 | 0.026 |
  | good median iso distance | 169 | 168 | 182 |
  | good median d' | 5.9 | 5.6 | 6.3 |
  | good median nn_hit | 0.58 | 0.60 | 0.67 |

  For reference, the previous cleaning of the old sort gave 127 good+nsg.
  - Problem units:
    - 496/487 stay separate in both heavy sorts.
    - 197 is good without any peeling.
    - 62 is absorbed into 63 (still good).
    - 187/452 is still duplicated, and both copies absorb 449.
    - 272 and 32 are lost in both heavy sorts.
    - 66 becomes MUA in the new chain (41 violations).
    - The new chain has a 102k-spike noise unit (SNR 1).
  - Next, offered: run the cleaning on heavy_newchain under a scratch root. Does it end above
    127, and with how much less cleaning work?
- **User: yes, and then the rest of the animal's recordings.**
  - 0701, 0703 and 0707. 0708 is excluded: never sorted, preprocess incomplete, the user's
    earlier choice.
  - scratchpad `heavy_session.py <session>`: prep (current chain) -> heavy sort (4.1.4 + heavy +
    new chain) -> analyzer (old extension params) -> clean_and_curate (current code) ->
    `comparison.json` against the current analyzer_clean.
  - All on C: (`C:\si_stage\<session>`, `C:\si_stage\resort\<session>`). Locks give one D: prep
    and one GPU sort at a time.
  - All 4 launched 2026-10-07 (0702 starts at the clean step). Logs: scratchpad
    `heavy_session_DS15_<date>.log`.
  - **0702 result:** heavy new-chain sort + current cleaning vs the old sort.

    | | heavy + clean | old + previous cleaning | old + greedy |
    |---|---|---|---|
    | good+nsg | 111 | 127 | 134 |
    | good spikes | 692k | 804k | 845k |
    | l_ratio | 0.027 | 0.057 | 0.092 |
    | iso dist | 187 | 172 | 163 |
    | d' | 6.0 | 5.7 | 5.3 |
    | nn_hit | 0.65 | 0.55 | 0.52 |

    Cleaning workload on the heavy sort vs the old sort's greedy run:
    - 3 light artifacts (vs 616);
    - 52 noise units (vs 150);
    - 7 merges, no drift or feature merges (vs 16);
    - 19.9k peeled (vs 26.2k);
    - 37 splits (vs 30);
    - 40.7k close pairs (vs 25.7k);
    - 58k redundant-duplicate spikes in 4 merges.

    In short: fewer but cleaner good units; the merge and artifact work disappears, and the
    split, peel and duplicate work stays.
  - **0701: 41 vs 31 good+nsg; 0703: 48 vs 62.** The heavy sort's good units are better isolated
    in all sessions (l_ratio 2-4x lower).
  - Good-unit fates (scratchpad `heavy_lost_units.py` -> `heavy_compare/good_unit_fates.csv`):
    - 152 are good in both.
    - Current-only: 75. The 59 unmatched have median SNR 5.1, l_ratio 0.57, nn_hit 0.21 —
      marginal.
    - Heavy-only: 48, well isolated and low-rate.
  - Tagged units (`heavy_tagged_fates.py`): of 29 mappable tagged units, 18 are good in heavy and
    11 lost.
    - Most of the lost are marginal (SNR 4-5).
    - Two are real losses: 0702 KS 32 iSPN (SNR 10.8, l_ratio 0.007; split into two MUA) and
      0703 KS 59/1 dSPN (SNR 10.3; unmatched).
- **User: tag the heavy sorts too; and run the heavy KS step on DS48** (already on the current
  preprocessing, with a `qc/` folder).
  - Sessions with `preprocess/qc` are on the current chain: DS13, DS23, DS37, DS45, DS48. DS15
    and DS2 lack it.
  - `heavy_session.py` now stages a copy of a qc'd D: preprocess instead of re-preprocessing.
    DS48 0921-0924 were launched 2026-10-07.
  - scratchpad `heavy_tagging.py <session>`:
    - builds a scratch root `C:\si_stage\root\analysis_files\...`: binary_signals copied in,
      spikes.npz from the heavy analyzer_clean;
    - runs isolate (bombcell);
    - matches tagged units to the last real tagging. singU_metadata was made from
      analyzer_clean_superseded_3, verified by ids and names.
  - Running for DS15 0701-0703; 0707 follows when its cleaning finishes.
  - **Tagging results, 0701-0703 (old = last real tagging on superseded_3; heavy = heavy new-chain
    + clean):**
    - Tagged units: 32 old vs 27 heavy. 16 are the same neuron with the same class.
    - 0701: 2 vs 2 (1 shared). 0702: 17 vs 13 (10 shared). 0703: 13 vs 12 (5 shared; old dSPN
      603 = heavy bU 771).
    - Lost in heavy:
      - mostly marginal units (SNR 4.1-4.7, l_ratio 0.19-2.4);
      - plus KS 32 iSPN (0702, SNR 10.8) and KS 59/1 dSPN (0703, SNR 10.3, 340 spikes);
      - the 0702 bU 1292 is the bad 496+487/2 merge, not a real loss.
    - Gained in heavy: 11. Example 0701 unit 242 dSPN is weak: only chr2 700 uW, late 30-50 ms
      first spikes, Cox p=0.03, LRT p=0.16.
    - Review figures for all lost/gained units: `C:\si_stage\resort\tagging_review\`
      (ttfs, stim firing, old waveforms_by_light).
    - Each heavy run also wrote its own figures: `C:\si_stage\root\analysis_files\DS15\<s>\pynapple\`
      (opto_tagging/ttfs, stim_firing_plots).
  - **0702 iSPN KS 32 breakdown (user: "the kind of unit we shouldn't miss"):**
    - The old unit 32 (439 spikes, presence 0.85) mixed two parts:
      - heavy 26: 370 spikes, -91 uV, bursty, SNR 10.5, 0 violations;
      - heavy 24: 71 spikes, -148 uV, mostly in the last 10 min, with a different footprint (a
        strong trough on AP21/AP23 that 26 lacks). Another neuron, or collisions.
    - Orange response: 26 has all of it (spikes in pulse vs pre-pulse, summed over 30 pulses:
      0/0, 0/0, 7/0, 42/0, 195/0 over 100-1000 uW); 24 has none. Neither responds to blue.
    - The heavy split was right. 26 was lost at labelling: presence_ratio 0.688 < 0.7 (bombcell),
      because it is quiet outside stimulation (53% of its spikes are evoked).
    - The pieces were not merged back: overlap 0.45, inside 0.76, amplitude ratio 1.59 fail the
      drift rule, and cosine 0.86 < 0.9.
    - Figure: `C:\si_stage\resort\tagging_review\0702_KS32_split_26_24.png`.
    - Units that are MUA for presence_ratio alone: heavy 13/4/2, current 12/4/2 (0701/0702/0703).
    - **Proposed:** drop presence_ratio as an MUA criterion (keep it as a flag), after reviewing
      those units. It is a label-only change. Awaiting the user.
  - `plot_merge_report` now accepts a str folder (Path(folder)).
  - Follow-up: 26 vs 24 are separate (user agrees).
    - 24 is not light-driven, and pulses span the whole session.
    - 26 was never split, peeled or pair-dropped. KS gave 370 spikes; the old sort gave 439,
      including 24's.
    - Template scan (`template_scan_0702_26.py`): 0 unassigned matching events, so 26 is
      genuinely sparse (126 spontaneous spikes in 48 min plus 244 evoked).
- **User 2026-10-07: presence_ratio minimum 0.7 -> 0.6** (`MUA_PRESENCE_MIN`, `default_thresholds`).
  Reclassify only; tag only the units that become good.
  - New in `isolate_single_tagged_units.py`:
    - `_classify`, the tests + classification, extracted verbatim; the old file is backed up in
      the scratchpad;
    - `refresh_quality_labels(analysis_folder, analyzer_folder)` copies labels into spikes.npz and
      refuses an npz from another analyzer;
    - `tag_new_single_units(analysis_folder)` tests only the new single units and merges them in,
      with backups.
  - Heavy (C:): 0701 +2 good (untagged), 0702 +1 (unit 26 -> iSPN by both tests, baseline
    0.04 Hz), 0703 0. Script: scratchpad `relabel_heavy.py`.
  - D: (`relabel_d_pr06.py`, backups `*_pre_pr06`): 72 units mua -> good over 20 sessions;
    newly tagged 4 dSPN, 4 iSPN, 2 muSPN.
    - DS15: relabelled, but tagging not updated, because its spikes.npz is from
      superseded_3/greedy.
  - Still to relabel when they finish: DS15 0707 heavy and DS48 heavy (started with 0.7).
- **User: full tagging-evaluation plots for the heavy DS15 sorts only** ("not the old sorts and
  curation").
  - scratchpad `heavy_tag_plots.py`: opto PSTH (both styles, with pulse-spike waveforms), TTFS,
    tuning curves, waveforms_by_light, in `C:\si_stage\root\analysis_files\DS15\<s>\`.
  - Junctions in the scratch root point to the heavy analyzer_clean and the new-chain recording
    (C:). No raw-data folder is linked: get_pynapple_data is replaced by a cached-file loader.
  - **DS15 done (2026-10-07):** all 4 sessions have tagging, the 0.6 relabel and the full plot set.
    - 0707 heavy vs current: 99 vs 99 good+nsg (heavy +8 at 0.6). Tagged units are identical:
      5 = 2 iSPN, 2 oU, 1 dSPN.
    - DS15 totals at 0.6, good+nsg: heavy 302 vs current 328 (0702 current = greedy).
    - DS15 tagged: heavy 33 vs old 37 (old = last real tagging, pre-0.6).
    - Heavy good units are better isolated in every session.
  - DS48: follow-ups (tagging -> 0.6 relabel -> plots) are queued per session to start when its
    comparison.json appears. Logs: scratchpad `heavy_followup_DS48_<date>.log`.
    `heavy_tagging.py` now picks the saved cleaning that the last real tagging matches.
  - DS48 is large: 8-31M spikes per session (old and heavy similar), against 3-7.5M for DS15.
    - The amplitude_scalings extension took more than 3 h on 0921.
    - 0924 (29M spikes) was restarted with `SKIP_SCALINGS=1`; nothing in the comparison or the
      cleaning uses that extension.
    - Expect DS48's cleanings to run overnight.
- **User: get DS45 onto Kilosort while the GPU is free** (preprocessing already current).
  - C: has only 375 GB free (DS45 would need 370 GB staged), so `heavy_session.py` with
    `SORT_ONLY=1` sorts straight from D:. Kilosort reads in order (~120 MB/s), and the D: and GPU
    locks are held for the whole sort.
  - Six sessions: 0811, 0812, 0814, 0819, 0820, 0821 (0813 never sorted). Logs: scratchpad
    `heavy_sort_DS45_<date>.log`.
  - **DS48 results (heavy + clean vs current clean, good+nsg):**

    | session | heavy | current | heavy l_ratio vs current |
    |---|---|---|---|
    | 0923 | 133 (137 at 0.6) | 132 | 0.038 vs 0.094 |
    | 0924 | 90 | 95 | 0.10 vs 0.085 (no gain) |
    | 0922 | 96 | 98 | 0.033 vs 0.069 |

    - Tagged: 0923 heavy 67 vs old 63 (46 shared). 0924: 28 vs 27, near identical. 0922:
      43 vs 47 (27 shared, 13 old with no heavy match).
    - **0922 finding: the heavy sort lumps sparse tagged iSPNs into busy units.**
      - old 533 -> 99.6% in heavy KS 26 (8,753 spikes);
      - old 699 -> 99.7% in heavy KS 112 (48,712);
      - old 723 -> 99.9% in heavy KS 119 (18,594);
      - old 797 -> 97% in heavy KS 150 (14,189);
      - all of those heavy units are MUA.
    - The cleaning's bimodality test cannot recover them (D 0.7-2.4 < 3): each is a 2-17%
      minority with overlapping features, and only the light response separates it.
    - Suspect: the 4.1.4 working split veto keeps a 0.2 Hz neuron merged with a busy one, because
      too few CCG counts to fail it. The heavy params are a confound.
    - Proposed test: DS48 0922 heavy params on 4.1.7 (~1 h GPU). **User: yes** — scratchpad
      `ks417_test_0922.py` -> `C:\si_stage\resort\DS48_20260922_ks417\ks`; it reports where old
      533/699/723/797/635 land.
    - DS48 0921: heavy 60 vs 63, isolation no better; tagged 19 vs 16. All four DS48 sessions
      are done, with plots.
    - The DS48 staged copies of 0921/0923/0924 were deleted (originals on D: verified by size).
      0922 is kept until the 4.1.7 test is done.
- **0922 test result (heavy params, 4.1.4 vs 4.1.7):** units old 422 (clean) | 4.1.4 353 | 4.1.7 382.

  | old iSPN | heavy 4.1.4 | heavy 4.1.7 |
  |---|---|---|
  | 533 | lumped in KS 26 (8,753 spikes) | own unit (1,370 of 1,520) |
  | 699 | lumped in KS 112 (48,712) | own unit (1,044 of 1,427) |
  | 723 | lumped in KS 119 (18,594) | own unit (1,602 of 1,763) |
  | 797 | lumped in a 14k-spike unit | lumped in a 14k-spike unit |
  | 635 | spread | spread |

  - The 4.1.4 working split veto caused most of the lumping. With few coincidences, the CCG
    test cannot reject a split between a sparse neuron and a busy one, so the veto keeps them
    merged. 4.1.7's bug disables the veto. The over-splitting it causes is what our merges undo.
  - **User 2026-10-07: switch to Kilosort 4.1.7.**
    - `heavy_session.py`: `KS_VERSION` (default 4.1.7; 4.1.4 = side install) and
      `RESORT_SUFFIX` (separate output folder). The tagging, relabel and plot scripts take the
      suffix too.
    - Stopped the 4.1.4 runs: DS45 0821 (running) and DS2 0505/0506/0511 (queued; prep kept).
    - The finished DS45 4.1.4 sorts are kept as `C:\si_stage\resort\DS45_<date>\ks_414`.
    - Relaunched on 4.1.7: all 6 DS45 sorts (SORT_ONLY, from D:), the 3 DS2 full runs, and the
      0922-4.1.7 clean, compare, tagging, relabel and plots (`RESORT_SUFFIX=_ks417`, log
      `ks417_0922_full.log`).
    - DS15 and DS48 heavy results so far are 4.1.4; re-sort them on 4.1.7 if wanted.
  - Overnight (2026-10-07): scratchpad `ds45_overnight.sh` (log `ds45_overnight.log`) runs, per
    DS45 session after its 4.1.7 sort, one at a time: stage, analyzer, clean, compare, tagging,
    0.6 relabel, plots, then deletes the staged copy (original verified).
  - **User: re-sort everything with 4.1.7; the 4.1.4 outputs need not be kept** (overwrite to save
    space).
    - Deleted on C: all 4.1.4 heavy outputs for DS15 and DS48 (`resort\` sort, analyzer, clean,
      compare folders; scratch-root `analysis_files` folders). The scratch-root junctions and
      every C:-only new-chain recording were verified intact. C: has 387 GB free.
    - DS15 0701/0702/0703/0707: full runs on 4.1.7 (0702's special-case paths removed) plus
      the queued follow-ups.
    - DS48: 0922 = the `_ks417` run in progress. 0921/0923/0924 are sorted from D:
      (SORT_ONLY); their analysis is chained after the DS45 chain (`ds48_after_ds45.sh`), one
      at a time, deleting each staged copy. The 0922 copy is deleted when its run finishes.
    - DS45: 4.1.7 sorts plus `ds45_overnight.sh`. DS2: 4.1.7 full runs plus follow-ups.
    - The 4.1.4 DS45 sorts are still in `resort\DS45_<date>\ks_414` (unanalysed; can be deleted).
  - The question of moving the C:-only new-chain recordings (DS15, DS2; ~540 GB) to D: is still
    open; for now they stay on C:.
  - 2026-10-08 03:40: the fixed-order chains (ds45_overnight, ds48_after_ds45) were waiting on
    0811 while 0820 was already sorted. They are replaced by:
    - `staged_queue.sh`: DS45 + DS48 0921/0923/0924, in sort-completion order, one at a time;
    - `followup_queue.sh`: follow-ups for DS15 0702/0707 and DS2 0506/0511.

    Mishap while switching: the stop command also matched the launcher shells, whose command
    lines named the chain script. All heavy_session processes survived (verified). Two waiters
    that could not be identified were stopped, and the follow-up queue covers their sessions.
  - **4.1.7 results so far:**

    | | heavy 4.1.7 | heavy 4.1.4 | current |
    |---|---|---|---|
    | DS48 0922 good+nsg | 99 | 96 | 98 |
    | DS48 0922 tagged | 51 (36 shared with old) | 43 (27) | 47 |
    | DS2 0505 good+nsg | 100 | — | 103 |
    | DS2 0505 tagged | 20 | — | 20 |
    | DS15 0701 good+nsg | 39 | 41 | 33 |
    | DS15 0703 good+nsg | 52 | 48 | 62 |
    | DS15 0703 tagged | 9 | 12 | 13 |

    - 0922 on 4.1.7 recovers iSPNs 533 (0.94), 699 (0.75) and 723 (0.90); 635 and 797 are still
      lost.
    - 0922 isolation, 4.1.7 between 4.1.4 and current: l_ratio 0.060 vs 0.033 vs 0.069.
- 2026-10-08 11:23 (user: two analyses at a time): `staged_queue.sh` was replaced by two
  `staged_worker.sh` workers (A, B), which claim sessions through `C:\si_stage\claims\<session>`.
  - The 0812 analysis (pid 15140) kept running; `finish_0812.log` runs its follow-ups and the
    copy deletion.
  - At the switch: A took DS45 0814, B took DS45 0821.
- **waveforms_by_light, new layout (user 2026-10-08):**
  - Above the blue and orange waveform columns: their opto PSTH and raster
    (`plot_psth._draw_opto_psth_on_axes`, rows by power, +-0.5 s).
  - Above the spontaneous column: PSTH and raster around every first lick (all `*_first_licks`,
    +-0.5 s, rows coloured by type), with a dashed line 100 ms before the lick.
  - Spontaneous spikes in the 500 ms before a first lick are bright green (all of them, up to
    300), with their mean dashed.
  - Bottom: the whole session, amplitude against time, coloured the same way, across the three
    columns, plus an amplitude histogram.
  - Unit info text above the means column.
  - Tested on DS15 0701 heavy unit 828.
  - New follow-ups use it automatically. Finished sessions are being redrawn
    (`regen_waveforms.py`; DS45 0811/0812/0820 read the D: preprocess under the D: lock, because
    their staged copies are gone).
- **Deferred (user 2026-10-08: "let's worry about it later"):** in `split_outlier_spikes`, the
  outlier core is fitted on non-stim spikes, but falls back to all spikes when a unit has fewer
  than 300 spontaneous spikes (`fit_on='all'`; quiet SPNs, e.g. 0702 iSPN 26). The bimodal split
  test uses all spikes, stim included. Options, not applied:
  - lower that minimum;
  - exclude evoked spikes from the split test.

  Either one changes the score cache, so all sessions would need re-scoring.
- **User: then DS2** (no heavy re-sort yet; needs full re-preprocessing).
  - `heavy_session.py` DS2 0505/0506/0511 with `SKIP_SCALINGS=1`, plus the queued follow-ups
    (tagging -> 0.6 relabel -> plots).
  - Re-preprocessing waits for the D: lock, which is held by the DS45 sorts.
  - Logs: scratchpad `heavy_session_DS2_<date>.log`, `heavy_followup_DS2_<date>.log`.
  - Later, once the DS48 staged copies are freed: a full `heavy_session.py` run per DS45 session
    (stages, skips the sort, analyzer with `SKIP_SCALINGS=1`, clean, compare), then the follow-ups.
- **User rule (2026-10-07): never delete raw data** (anything under `<ANID>_g#`, especially
  .bin). Deleting old copies of preprocessed recordings is fine once the original is verified.
  The rule is in CLAUDE.md and in memory. The verified C: copy of 0702's old-chain preprocess
  was deleted.
  - KS level: units, KSLabel, close pairs.
  - Unit matching, including the problem units.
  - Raw-sort quality metrics and bombcell labels, computed identically for both sorts with no
    cleaning. The heavy analyzer is in `C:\si_stage\resort\heavy_analyzer`.

### 28. Greedy split / peel re-clean — implemented, testing on DS15 0702 (2026-10-06)
User: run the PCA split and the outlier removal on every unit, noise included; repeat each until
nothing is left to split or peel; keep peeled outliers as clusters and merge them into nearby
units with a similar waveform; a merge report plot for every merge, in the sorting folder.
- **New module `postprocessing/greedy_cleaning.py`.** `split_outlier_spikes.py` is unchanged, so
  the score caches stay valid.
  - Splits repeat until no unit is bimodal (SPLIT_MAX_DEPTH = 6, a safety cap).
  - Peels repeat for up to PEEL_MAX_ROUNDS = 5 rounds. Accepted outliers, plus stimulation
    outliers of trusted cores, become `<unit>/out<k>` clusters; the kept part keeps its name.
  - A unit never loses more than MAX_OUTLIER_FRAC (0.3) of its size at birth.
  - Kept parts and outlier clusters are re-scored after every peel. Outlier clusters can be
    split, never peeled.
  - Small clusters get a fresh random-spike sample, so they have real templates
    (`resample_units`: waveforms, PCA projections, templates).
  - Light artifacts (`light_artifacts`) become `<unit>/art` clusters. Rule: within
    ARTIFACT_EDGE_MS = 1 ms of a light on/off edge, deeper than 2x the unit's template trough,
    and at most 0.5x as wide. Only spikes near edges are read; other spikes near an edge are
    left alone.
  - Spikes with an undefined feature become `<unit>/nan` clusters.
- **`clean_and_curate`, now in 7 stages:**
  - Noise units are found after the peeling and kept out of every merge. Final labels decide
    whether a unit is noise.
  - `/nan` clusters are dropped before the merges, because their NaN locations break the knn
    merge search.
  - `/art` clusters, and unmerged outlier clusters under OUTLIER_MERGE_MIN_SPIKES = 20, are
    dropped after the post-merge pairs, before the metrics.
  - Accounting: spikes in − spikes out = removed_spikes.csv. Noise units are not removed.
- **Merges:**
  - New first preset OUTLIERS (`outlier_merge_groups`). Each outlier cluster goes to its best
    unit: not its own lineage, not noise, within 40 µm, cosine ≥ 0.9, 0 cross spikes in
    (1, 1.5] ms, and refractory once merged.
  - The other presets leave outlier clusters alone.
  - Drift adds DRIFT_MAX_AMPLITUDE_RATIO 1.5 and DRIFT_MAX_INSIDE 0.5: the share of either
    piece inside the other's 5–95% time span.
  - `plot_merge_report` for every merge -> `spike_cleaning/merge_plots/`. Every peel with ≥ 20
    outliers gets a figure in `peel_plots/`; every split level in `bimodal_split_plots/`.
- New audit tables: `peels.csv`, `light_artifacts.csv`, `outlier_clusters.csv`,
  `noise_units.csv` (final noise units; replaces `noise_units_dropped.csv`).
- Test: scratchpad `test_greedy_0702.py`, writing to `scratchpad/test_greedy_0702/`. Then the
  full re-clean of all 20 sessions, after asking. The old spike_cleaning folders are kept
  aside, the outlier score caches are reused, and `analyzer_clean_superseded` must be handled.
- Dry-run fixes: plot file names were not unique (peel and merge figures); the folder was not
  created; NaN unit locations of new clusters in plots; figures can no longer stop a cleaning
  run; `/art` clusters must go through the pre-merge pair pass (post-merge assert: DS15 0702
  unit 609, a pair 15 samples apart).
- Dry run DS15 0702 (one process, until the assert):
  - 30 splits in about 10 min.
  - Peel rounds: 23,883 -> 1,789 -> 361 -> 100 -> 39 outliers. 26,172 peeled from 337 units;
    the round cap was hit with 3 units still holding outliers.
  - 436 outlier clusters, 616 light artifacts.
  - Merges made only 15 merged units.
  - Greedy stage about 61 min, the whole run about 70 min. Re-scoring after the peel (the
    per-unit fits in split_outlier_spikes, which run serially) dominates.
  - To check: 616 light artifacts vs the 17 found by hand; 15 merges from 436 clusters.
- **User 2026-10-06: run it on the rest of DS15; parallelize?** Running DS15 0701, 0702, 0703
  and 0707 side by side: 4 processes of `curate_subject --steps=clean,summary`, with
  `SI_N_JOBS=6` (new env override of JOB). The machine has 191 GB RAM and 24 CPUs; a process
  peaks around 16 GB. 0702 runs for real too (the dry run was stopped).
  - Old folders kept as `analyzer_clean_superseded_3` and `spike_cleaning_superseded_3`.
  - pynapple/isolate/ratio_tests and the DS15 compendium run afterwards in one process: each
    curate_subject run rebuilds the subject compendium from only its own sessions.
  - Logs: scratchpad `reclean_DS15_<date>.log`.
- The first parallel launch stalled: D: is one 11 TB SATA hard disk. It was at 180 MB/s and 0%
  idle, and three of the four sessions sat in their first split level for 30 min. Every peel
  round re-reads the whole preprocessed recording (54–82 GB per session).
  - Fix: `clean_sessions._stage_recording`. With `SI_STAGE_DIR` set, each session's
    `preprocess/` is copied to that fast disk (C: NVMe, `C:\si_stage`, 1.1 TB free). Copies go
    one at a time across processes (a lock file). The copy replaces the analyzer's
    `_recording`, because split and select copies drop a temporary recording.
  - The saved analyzer_clean points back at D:, and the staged copy is deleted after the save.
  - An interrupted re-clean now clears its own unfinished audit folder instead of failing on
    the rename.
  - All four relaunched, staged.
- **DS15 re-clean done (2026-10-06).** Staged copies removed. Old cleanings kept as
  `analyzer_clean_superseded_3` / `spike_cleaning_superseded_3`.

  | Session | Time | Clean units | Good before -> after | Merges | Light artifacts |
  |---|---|---|---|---|---|
  | 0701 | 73 min | 329 | 31 -> 31 | 7 | 18 |
  | 0702 | 97 min | 617 | 104 -> 109 | 16 | 616 |
  | 0703 | 73 min | 368 | 48 -> 48 | 3 | 72 |
  | 0707 | 122 min | 691 | 82 -> 81 | 10 | 310 |

  Notes on the table: 0702's 109 good units include 2 outlier clusters, and 0707's 81 include 2
  more.
  - 0702 checks:
    - 1292 (496+487/2) is fixed: 496 alone, good, 0 rp violations; 487/2 is MUA.
    - 62/63 are not merged.
    - Light artifacts were split off in the flagged units: KS 42 (7), 66 (10), 264 (8), 272 (4)
      and 8/2 (3). KS 32: none, because its sharp events are near licks, not light.
    - A peel of 264 became a good unit (264/out1/2).
  - Labels of the units shared with the old cleaning barely moved (0702: one mua -> noise).
  - Outlier clusters:
    - Few merge. In 0701, 3 of 215 merged; 131 were under 20 spikes, and 81 had a nearest
      similar unit at a median cosine of only 0.48.
    - The unmerged clusters with >= 20 spikes stay as units: 424 non-noise across DS15, almost
      all MUA, 4 good.
  - 0702's light artifacts are dominated by near-flat templates (-1 to -5 uV): any spike there
    is "2x deeper".
  - **Awaiting the user:**
    - (1) Drop unmerged outlier clusters unless good (offered as a post-hoc drop from
      analyzer_clean, no re-clean).
    - (2) Minimum template depth (~20 uV) for the light-artifact rule. Offered for the next
      datasets; for DS15 it would need a re-run.
  - Then pynapple / isolate / ratio_tests and the DS15 compendium, in one process.
- **User: did quality metrics improve for isolated units, and do isolated counts go up?**
  (scratchpad `compare_cleanings.py`, vs analyzer_clean_superseded_3)
  - Good + non_soma_good: 308 -> 316 (+8; 4 of them outlier clusters).
  - 294 units are good in both cleanings:
    - 255 have identical spikes.
    - Spike-based metrics are unchanged (medians identical), and no labels changed.
    - PCA isolation metrics are slightly worse for most of them: iso distance 154 -> 152,
      L-ratio 0.16 -> 0.19, d' 5.1 -> 5.0, nn hit 0.51 -> 0.50. The kept outlier-cluster units
      sit next to their parents in feature space; bombcell does not use these metrics.
  - Regression: 0707 good 231+243 (snr 30.6, 0 viol) became MUA (5 viol). The redundant preset
    chained it with 234+241/out1/1/2 and 237+241/out1/1/1. Redundant groups have no
    pooled-refractory veto.
  - **User: drop outlier clusters with fewer than 1000 spikes.** Now
    OUTLIER_KEEP_MIN_SPIKES = 1000: an unmerged outlier cluster stays a unit only with
    >= 1000 spikes. The merge candidates still start at 20 spikes.
    - Fixes (a) and (c) were applied too (ARTIFACT_MIN_TEMPLATE_UV = 20).
    - DS15 re-clean 2 started 2026-10-06, 4 sessions staged in parallel. Logs: scratchpad
      `reclean2_DS15_<date>.log`. The greedy cleaning is kept as `*_superseded_4`.
  - Proposed earlier, before the user's answer (needs a DS15 re-clean, ~2 h):
    - (a) `_merged_rp_veto` for every preset.
    - (b) Drop unmerged non-good outlier clusters, then recompute the metrics.
    - (c) Artifact rule only for templates >= ~20 uV deep.
- Speed-up options to offer:
  - Parallel per-unit fits in `find_outlier_spikes` (joblib over units). This changes
    split_outlier_spikes.py, so every score cache is invalidated: one extra full scoring per
    session.
  - GPU: the fits are many small per-unit problems (GMM, MCD, 300-shuffle null). GPU overhead
    would dominate and the code would need rewriting, so CPU parallelism is the better lever.

### 26. Waveforms by light condition for every tagged unit — running (2026-10-06)
User: for each tagged unit, a plot like `unit_685_waveforms_by_light.png` (spontaneous / blue /
orange snippets + means, 4 largest channels), saved in each session's analysis folder.
- New module `analysis/waveforms_by_light.py` (`session_waveforms`;
  `python -m analysis.waveforms_by_light [folders]`).
  - It covers every unit whose final_classif is in TAGGED_CLASSES (dSPN, iSPN, muSPN, bU, oU, mU).
  - Output: `<analysis session>/waveforms_by_light/unit_<id>_<class>.png`.
  - One chunked pass over the preprocessed recording per session for all its tagged units
    (running sums for the means of all spikes, plus 200 random snippets per group). The
    one-off scratchpad script read the whole recording per unit.
- Test DS2 0506: 2 units, ~9 min (recording read). The default list is the 20 current
  sessions (DS2, DS13, DS15, DS45, DS48); older DS21/23/37 tagging is skipped.
- Running all 20 (log scratchpad `waveforms_by_light.log`), ~3 h.
- 2026-10-08, new layout (user): first-lick PSTH + raster above the spontaneous column, opto
  PSTH + raster above blue / orange, pre-lick (500 ms) spontaneous spikes in green, a
  whole-session amplitude-vs-time row. Redrawn for the finished heavy 4.1.7 sessions (DS15
  0701/0703/0707, DS2 0505/0506). An image open in a viewer cannot be overwritten (Errno 22):
  the figure is then written as `<name>_new.png`.
- 2026-10-08, second round (user), for the next batch only ("don't regenerate any plots"):
  - Plot every unit the tagging tests call light-driven, whatever its quality, not only
    bombcell-good. `isolate_single_tagged_units` now writes `pynapple/all_units_tagging.csv`
    (every unit: quality labels, tagging columns, `light_classif` = the tagging class with the
    fast-unit relabel for any quality, `single_unit`). `tag_all_units(analysis_folder)` returns
    it, and for a session tagged before this builds it: single units from sing_tagged_units,
    tests run on the rest only (plots to `pynapple/all_units_tagging/`, so the single units'
    summaries are left alone; the tests are per unit, no cross-unit correction). Rebuilt when
    older than spikes.npz / sing_tagged_units.npz. Non-single figures are named
    `unit_<id>_<class>_<quality>.png`.
  - This adds many figures: DS2 0506 has 253 light-driven units among its 433 non-single
    units (173 of 370 MUA called dSPN); DS15 0702 has 112 figures (14 min).
  - Waveform rows packed (hspace 0.06) with shared, single-labelled x axes; PSTH and raster
    packed with shared x; the lick panels keep only the t = 0 line (no -100 ms line);
    legend "pre lick"; no metadata text box; above the means column the whole probe and
    +-150 um around the unit (contacts shaded by template amplitude, waveform channels and
    neighbouring units marked).
  - Layout checked on DS15 0702 units 25 and 1080 (600 s of recording, scratchpad
    `wfl_layout_test/`). DS15 0702's real figures were drawn mid-change (probe zoom + text
    box version, all light-driven units).
- 2026-10-08, merge candidates (user: "a csv listing pairs of units in waveforms_by_light that
  are close together on the probe"): `waveforms_by_light.close_pairs(analysis_folder)` ->
  `waveforms_by_light/close_pairs.csv`. Pairs of light-driven units (any quality) with unit
  locations <= PAIR_MAX_UM = 50 um apart, sorted by distance: dx/dy, template cosine
  similarity, class / quality / spikes / peak channel / template amplitude of each, spike pairs
  within +-1.5 ms vs chance (n_a n_b 2 R / T; a split neuron shows a deficit, ratio << 1),
  figure names. Sorting analyzer only, no recording read. All finished heavy sessions in one
  table: `C:\si_stage\root\analysis_files\close_pairs_all.csv` (scratchpad
  `close_pairs_all.py`); `heavy_tag_plots.py` runs it for the sessions still to finish.
  - DS2 0511: 479 pairs among 218 units. Many pairs have strongly negative template
    similarity at the same location (opposite-polarity templates, checked against a direct
    cosine), and most have *more* +-1.5 ms coincidences than chance (shared light drive, or
    duplicates) rather than a refractory deficit.
- 2026-10-08, strongest merge candidates (user): `close_pairs` now counts cross-unit
  coincidences on spontaneous spikes only (outside every light pulse + 100 ms), split into
  <= 0.3 ms (same spike in both: `spont_dup`) and 0.3-1.5 ms (`spont_rp`, a refractory
  violation if one neuron), with Poisson P of a deficit / excess, and also writes
  `merge_candidates.csv` (`merge_candidates`), tiers:
  1 strong (similarity >= 0.9, deficit P <= 0.01 and ratio <= 0.5), 2 likely (>= 0.8,
  P <= 0.05, ratio <= 0.75), 3 duplicate spikes (>= 0.8, <= 0.3 ms excess P <= 0.01 covering
  >= 20% of the smaller unit), 4 similar but too sparse to test (>= 0.9, < 3 violations
  expected, no significant excess). The ratio cut was added after a first pass: with thousands
  of spikes, shallow deficits (0.85-0.9) were significant; the sorter's overlap dead time
  produces those between any two clusters at one site.
  - 13 finished heavy sessions: 63 candidates (3 strong, 26 likely, 8 duplicate, 26 sparse);
    `C:\si_stage\root\analysis_files\merge_candidates_all.csv`. Mostly MUA-MUA; good-good
    examples: DS48 0922_ks417 932/1882 (ratio 0.16), 33/1388 (0.49).
  - User, 2026-10-08, after reviewing the tables: no strong reasons to do any merges. No
    merges made; the csvs stay as a review record (still written for new sessions).

### 25. Light-evoked % change vs quiet-baseline rate scatter — done, awaiting review (2026-10-05)
User: per session, per animal and for the compendium, a dot per good unit: x = baseline firing
rate, y = % change in firing during the light (peak 100-ms average during stimulation /
baseline). "Baseline" = every moment with nothing going on: no lick, light, trial
(cue..outcome), with 200 ms either side of each event. Then: separate columns for orange and
blue light, the intensities pooled.
- New module `analysis/opto_baseline.py` (`quiet_intervals`, `light_pulses`,
  `peak_light_rate`, `session_table`, `plot_scatter`; `python -m analysis.opto_baseline`).
  - Quiet = the session minus every event moment, every X_on..X_off span (pulses, tones,
    solenoid), cue..outcome-tone trials and lick bouts, each padded by 200 ms, and
    saturation-blanked time. ITI markers and light traces are not events.
    DS48 0923: 2216 of 3115 s quiet, in 737 stretches (median 2.5 s).
  - Light rate: the pooled pulses, 1-ms bins from 1 ms after onset to 1 ms before offset (the
    tagging rules), the highest 100-ms boxcar mean (the whole trimmed pulse when shorter: 98 ms
    / 8 ms). % change = 100 (light / baseline - 1).
- Outputs: `opto_vs_baseline.png/.svg/.csv` in every session's ratio_tests, each
  `<animal>/compendium` and `analysis_files/compendium` (all 20 sessions, 1685 units).
- Medians (% change, ChR2 | Chrimson): dSPN 305 | -5; iSPN -1 | 253; muSPN 1050 | 743;
  untagged -24 | -16. % change scales roughly as 1/baseline by construction, so low-rate units
  dominate the top.
- User: "please do not log scale axes unless I ask for it" (saved as a feedback memory); also
  a smaller legend with a border, in the rightmost panel only.
  - Both axes are now linear. A near-silent iSPN (+8.8 million %) flattened everything, so the
    linear y axis stops at the 98th percentile (`Y_CAP_PERCENTILE`; 5283% for all animals),
    and the 43 / 24 units past it sit on the top edge as triangles, counted in the title.
  - Regenerated everywhere.
- User asked for y-axis breaks for "those 2 outlier iSPNs". They are not two: there is a
  continuous tail of near-silent units (iSPN 798: 1 spike in all quiet time -> +8.8M%; iSPN 205
  +1.4M%; ~60 units with baseline < 0.3 Hz at 5,000-500,000%). Options offered: broken axis at
  ~5,000%, excluding near-silent units, or an absolute-change (Hz) panel. **User chose the
  broken axis.**
  - `plot_scatter` breaks the y axis at `Y_BREAK_PCT` = 5000% whenever a unit passes it: a
    main panel from -100% to 5000%, and above it a linear panel with only the units past the
    break, with break marks. Figures with no such unit stay single-panel. The cap and
    triangles are gone.
  - Regenerated everywhere.
- User: "Change the peak stimulated firing rate calculation to the highest firing in 10ms in
  the stimulation".
  - `STIM_WINDOW_S` 0.1 -> 0.01: the highest 10-ms boxcar of the pooled 1-ms pulse response,
    1 ms after onset .. 1 ms before offset (DS13's 10-ms pulses: the whole 8 ms).
  - Medians (ChR2 | Chrimson): dSPN 624 | 73; iSPN 114 | 464; muSPN 1939 | 1755;
    untagged 58 | 71. Units > 5000%: 64 | 38.
  - Regenerated everywhere.

### 24. Compendium / ratio_tests rebinned at 100 ms — running (2026-10-05)
User: "Reanalyze everything covered by compendium/ratio analysis with 100ms binning instead of
50ms".
- New `pres_plot_utils.BIN_WIDTH = 0.1`, the default `bin_width` of all 14 binned analyses:
  heatmaps, PSTHs, rainbows, spn_summary, bars, the lick-bout PSTH and the binned-rate
  distributions. The z-score reference uses the same bins (`compute_unit_z`).
  `compendium_plots` passes no explicit bin_width.
- Smoke test DS48_20260923 ratio_tests: 76 files, 23,502 bins per unit.
- Running (scratchpad `rerun_bin100.sh`, logs `rerun_bin100/`): all 20 ratio_tests (DS13
  included), then the 6 pooled compendia (`rebuild_pooled.sh`). Each runs in its own process,
  with retries.
- **Done** (2026-10-05 ~17:40): all 20 ratio_tests and all 6 pooled compendia are at 100 ms.
  The cross-dataset compendium needed three tries: a segfault, then an OSError saving
  `spn_summary.png` (probably open in a viewer), then ok.

### 23. Distribution of the 50-ms binned rates behind the z-scores — done, awaiting review (2026-10-05)
User: for each session, plot the distribution of each unit's 50-ms binned firing rates (the
ones the z-score uses), one panel per neuron category, a thin coloured line per unit and the
grand average in black.
- New `pres_plot_utils.plot_binned_rate_distributions`:
  - The same bins as `compute_unit_z` (`get_trial_ephys`: task period, light pulses and
    inter-session gaps excluded).
  - Categories via `_unit_category` (dSPN, iSPN, muSPN, untagged).
  - A unit's line is coloured by its mean rate (log colour scale). Grand average = mean of the
    units' distributions. Log y axis.
  - The x axis runs to the largest per-unit 99.5th percentile count. Rates come in 20-Hz steps.
- Called from `compendium_plots.plot_compendium` (-> `binned_rate_distributions.png`).
- Drawn for all 20 sessions' ratio_tests, the 5 subject compendia (DS13 included) and the
  cross-dataset compendium (scratchpad `replot_binned.py`).
- **Bug found by it: pooled z-scores used a diluted reference.**
  - In a merged pseudo-session `compute_unit_z` binned every unit over *all* sessions'
    task time: 269,138 bins per unit in the DS48 compendium vs 47-67k per session; 783,343 in the
    cross-dataset one.
  - So the other sessions' time counted as silence, mu and sd were too low, and z was inflated
    in every z-scored plot of every subject and cross-dataset compendium. The per-session
    ratio_tests were unaffected.
  - **Fixed:** `merge_sessions` records each unit's `session_start`/`session_end`. New
    `unit_reference_counts` keeps each unit's bins within its own session's task period (its
    first cue - 10 s .. its last outcome + 10 s). `compute_unit_z` and the distribution plot
    both use it.
  - Check: DS48 0921+0923 pooled vs alone, mu/sd identical for 0921; within 0.07% / 2.2% for
    0923 (bin alignment at the session start).
  - All 6 pooled compendia rebuilt in full (scratchpad `rebuild_pooled.py` / `.sh`, one
    process each, with retries).
    - pynapple segfaults (issue 7) hit DS45 once and DS48 twice; each retry succeeded.
    - The cross-dataset compendium now reports a median of 46,953 bins per unit (its own
      session).
    - Awaiting the user's review.

### 22. Baseline firing of tagged SPNs: bimodal? -> bU / oU / muU — in progress (2026-10-05)
User: SPNs fire at low baseline rates. Pool the tagged units (iSPN, dSPN, muSPN), test the
baseline rate for bimodality, and if it is bimodal, set a metric-based cutoff that relabels
high-firing tagged units bU / oU / muU (blue / orange / multi; the user wrote both "muU" and
"mU" -- using muU). Revise the heatmaps and PSTHs to include muSPN, bU, oU and muU.
Plan:
- Analysis first, read-only in the scratchpad, while the retag runs. No shared-module edits
  until then.
- Baseline = the unit's rate outside every light pulse +-1 s, with blanked saturation time
  taken out of the denominator, per session. Sensitivity check: the rate in the 2 s before
  each cue.
- log10 rate; Ashman's D + density valley (as `_bimodal_split`) + a unimodality test (dip or
  Silverman). Points coloured by animal, so pooling does not fake a mode.
- If bimodal: the cutoff at the equal-posterior / density-minimum point, as a named constant.
  `isolate` keeps `tag_classif` and puts the relabelled class in `final_classif`. One shared
  class list and colour map in `pres_plot_utils`; the dSPN-iSPN difference rows unchanged.
  The user scoped the plots to ratio_tests / compendium.
- **Result** (scratchpad `spn_baseline.py`; outputs in
  `analysis_files/compendium/spn_baseline_rate/` (png + csv); 287 tagged SPNs in 20 sessions;
  DS15/45/48 still on old-rule tags):
  - **Not bimodal.** Baseline: median 3.1 Hz; a 2-Gaussian fit on log rate has Ashman's D 1.85
    (< 2), no valley (1.01); Silverman p(unimodal) 0.19. Pre-cue rate: the same (D 1.87,
    valley 1.01, p 0.18). One broad log-normal-like spread from 0.01 to 150 Hz.
  - But a long high tail: > 10 Hz 70 units (24%); > 20 Hz 53 (18%); > 40 Hz 36 (13%), almost
    all DS48 and DS45 (all three classes). Untagged single units: median 1.6 Hz, 19% > 10 Hz.
  - User: "don't log the firing rate for the bimodality analysis". On linear Hz (291 tagged
    SPNs, the retag partly through): 2-Gaussian fit 2.35 Hz (71%) + 41.8 Hz (29%), Ashman's D
    1.76, valley 0.65, Silverman p 0.43 (pre-cue: 1.75 / 0.65 / 0.41), so still not bimodal by
    the pipeline's criteria (D > 2 and valley < 0.15, Silverman p < 0.05). Shape: a tight 0-10 Hz
    peak plus a flat sparse 10-150 Hz plateau (DS48/DS45).
    Equal-posterior boundary 9.2 Hz (pre-cue 9.0). Offered: (a) the 9.2 Hz boundary,
    (b) the cell-type check, or (c) both. Figure: `spn_baseline_rate_linear.png`.
  - User asked for a csv of every SPN: `analysis_files/compendium/spn_table.csv` (scratchpad
    `spn_table.py`, final new-rule tags).
    - Columns: animal, recording, unit_id, ks_unit_ids, cell_type, baseline_hz, precue_hz,
      ramp_index_hz (compute_ramp_scores: 150-10 ms pre-lick minus the 1-s pre-cue rate,
      no-stim trials), lick_bout_modulated.
    - 292 SPNs (148 dSPN, 98 iSPN, 46 muSPN); 194 are lick-bout modulated.
  - Per the conditional request, no data-driven cutoff. Options put to the user: a striatal
    cell-type check of the high-rate tagged units (waveform width from the raw template, not
    SI peak_to_trough_duration; post-spike suppression; long-ISI share; bombcell's
    MSN/FSI/TAN rules), or a fixed literature-based rate cutoff.

### 21. DS13 first sort — running (2026-10-05)
User: "move on to resorting all of DS13" (never sorted: 0822, 0824, 0905; NP2014;
`hidden/` 0829/0902 and `training/` skipped).
- **Analog tone channel:** DS13 records the speaker's square wave on analog XA5. That path
  still classified by duration, so 0905 failed ("Multiple outcome tones found between cue
  at index 0").
  - `parse_nidq.parse_tones_analog` now goes through the same `_parse_tone_edges` as the
    digital line (auto: frequency for a square wave), with the rig's tonemap.
  - 0822 and 0824 give identical counts to before (113/30/66/17 and 122/31/78/12).
- **0905 solenoid:** it is on digital line 2, not analog XA4 (the default read 508,090 noise
  "openings"). Wrote `recording_files/DS13/DS13_20250905/nidaq_map.txt` (the defaults plus
  solenoid on din 2). It now parses 170 trials, 50/50 rewards matched.
- **Run** (scratchpad `sort_ds13.sh`, logs `sort_ds13/`): `si_pipeline.run_pipeline` per session
  (binary signals, saturation, preprocess+QC, KS4, analyzer, drift-aware cleaning, pynapple,
  summary, PSTHs), then `curate_subject --steps=summary,isolate,ratio_tests` + the DS13
  compendium. Started 2026-10-05 ~00:00.
- Untouched: DS13 0822/0824 have no nidaq_map (the defaults parse correctly).
- 2026-10-05 results:
  - Sorted and tagged: 0822 good+nsg 107+16 (dSPN 6); 0824 27+1 (none tagged);
    0905 54+3 (iSPN 1).
  - 0824's PSTH step crashed in a loky worker (issue 7). It is being redone via the
    curate_subject pynapple step, then isolate/ratio_tests/compendium again.
  - Low yield in 0824: 16% of it was blanked for whole-array saturation (584 s, mostly
    1777-3205 s), and 43 of its units are MUA on presence ratio alone. Possible fix:
    measure presence over the non-blanked time only. Not raised with the user yet.
- **Pulse length** (user, 2026-10-05: "DS13's optogenetics was run with 10ms pulses"; then "I am
  no longer sure if this is an issue"):
  - Measured: 0822 and 0824 use 10-ms pulses; **0905 uses 100 ms**.
  - TTFS (100-ms window) and `test_stim_firing` (window = the measured pulse duration) do not
    hard-code 100 ms.
  - But with 10-ms pulses the firing test's window is 1-10 ms, and Chrimson-driven units in
    0822 fire 6-58 ms after onset (845, 853, 1269, 1291, 1533), i.e. after the pulse.
- **Light artifacts:** sharp pile-ups 0.5-1.0 ms after pulse onset and ~0.75 ms after offset
  (0822 unit 683: 20 spikes in one 0.25-ms bin).
  - Units with an onset/offset pile-up in the first 1.5 ms: DS13 0822 ChR2 69/2; 0905 ChR2 87/33
    and Chrimson 55/52; DS15 0702 ChR2 21; DS45, DS48 and DS2 roughly 0-9.
  - TTFS takes the first spike after onset with no artifact skip, so an artifact can pass as a
    short-latency tag. The firing test skips only 1 ms.
  - Probably preprocessing's remove_artifacts misses DS13's artifacts (a delay after the NIDQ
    edge?).
  - Options put to the user: (1) the tagging tests ignore spikes within 1.5 ms of every
    onset/offset, and the firing-test window runs to max(pulse duration, 50 ms); (2) also fix
    the artifact removal and re-preprocess + re-sort DS13.
  - **User decided:** do not trust activity after light offset for classifying tagged units;
    ignore spikes within 1 ms of onset for TTFS. Then: "the ttfs test should look from 1ms after
    the onset of stim out to the end of stim, ignoring spikes that occur within 1ms before
    offset".
  - **Implemented** (`parse_opto_tagging`):
    - `LIGHT_ARTIFACT_MS = 1.0`; `pulse_duration_s(bs, opsin)` (the median of the paired
      on/off events), shared with `test_stim_firing`.
    - The TTFS classifier now uses (1 ms, pulse - 1 ms] after onset (1-9 ms for 10-ms pulses,
      1-99 ms for 100 ms; was 0-100 ms). The pre-stimulus control is the same span, ending 1 ms
      before onset, with censoring at the window end.
    - The firing test is unchanged: (1 ms, pulse].
    - Display-only TTFS plots (`plot_psth.py`, `plot_ttfs_cdf_single`) still use 0-100 ms.
  - **Impact** (scratchpad `ttfs_impact.py`): DS15 0702 0 of 127 TTFS classes change;
    DS48 0923 2 of 129 (758 and 1034, untagged -> iSPN).
  - DS13's pending isolate picks up the new rule. Queued after it (scratchpad
    `retag_all.sh`, logs `retag_all/`): isolate, ratio_tests and the subject compendium for
    DS2, DS15, DS45, DS48, then the cross-dataset compendium. DS13 is not in its session list
    yet; ask the user.
  - Retag done 12:51: every subject's isolate, ratio_tests and compendium plus the
    cross-dataset compendium are ok (668 lick-bout modulated units). Tag changes vs the old
    rule are small:
    - DS2 0511: dSPN 9 -> 10.
    - DS45 0814: iSPN 2 -> 3.
    - DS48 0921: iSPN 1 -> 3.
    - DS48 0923: iSPN 27 -> 28.
    - DS48 0924: iSPN 7 -> 8.
  - DS13 done under the new rule (10:37):
    - 0822: 123 single units, dSPN 1 (was 6 under the 0-100 ms window, which counted onset
      artifacts).
    - 0824: 28 single units, none tagged.
    - 0905: 57 single units, muSPN 1 (was iSPN 1).
    - The DS13 compendium is rebuilt. Re-tagging of the other subjects started 10:37.

### 20. Rainbow (latency-quintile) plots cut at the first lick — applied (2026-10-05)
User: each trial's contribution to the average should stop as soon as the lick happens, so
sensory and premotor activity are not mixed.
- New `pres_plot_utils.first_lick_latency` (first raw `lick_on` after the cue; the outcome
  latency is the fallback), `censor_after_first_lick` (bins ending after the lick -> NaN) and
  `nan_mean_ci`.
- Applied in `plot_cue_aligned_by_latency`, `plot_cue_aligned_nostim`,
  `plot_spn_cue_aligned_nostim`, and `plot_dspn_isPN_diff` (its cue-aligned latency
  figures). The pre-lick unit filter for unlabeled units is now NaN-aware.
- The first lick precedes the outcome tone it triggers by ~1 ms (DS48_20260923: median
  0.001 s, never after it).
- Each quintile's trace now ends at its latest lick. The last bins are averaged over few trials
  and swing up (premotor). Offered a minimum-trials rule or a pre-lick buffer; the user said
  "it's totally fine".
- A full regeneration was started, then stopped because the user asked why the bar plots were
  being redrawn too. `compendium_plots.plot_rainbows` / `replot_rainbows` now redraw only
  the rainbow figures (scratchpad `replot_rainbows.py`, log `replot_rainbows.log`): every
  ratio_tests folder, the 4 subject compendia and the cross-dataset one. DS2_20260505's
  ratio_tests, half-rebuilt when the full run was stopped, is redrawn whole.
  Done 22:40, all 22 ok (32 min).
- User asked for the same in `spn_summary`, and for the earliest-lick traces on top of the
  stack and the latest at the bottom.
  - `plot_spn_summary`'s cue-aligned column is now cut at the first lick, and it moved into
    `plot_rainbows`.
  - New `latency_zorder(norm, latency)` (z 2-3, earliest highest; CI bands one lower), applied
    to all 16 rainbow line, band and marker calls.
  - All rainbows redrawn again (log `replot_rainbows2.log`); all 22 ok. Awaiting review.

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
