# `parse_intan_input.py` vs. `parse_nidq.py`

Both modules turn a raw acquisition stream into a `pynapple.TsGroup` of behavioral
and opto-stimulation events (cues, licks, solenoid, trial outcomes, LED pulses).
`parse_nidq.py` is the original, written for the National Instruments DAQ /
Neuropixels rig; `parse_intan_input.py` is adapted from it for the Intan rig.

The two rigs record fundamentally different signals, so a handful of functions
(`parse_tones`, `parse_trials`, `get_opto_ts`, `get_binary_signals`) diverge in
non-trivial ways. The two differences below are the ones that matter for anyone
porting analysis between the two.

---

## 1. How a trial's outcome is determined

### nidq — outcome is *measured* from an explicit tone
On the nidq rig, **every trial contains a tone that marks its outcome.** Tones are
classified purely by duration in `parse_tones`:

| tone duration | meaning        |
|---------------|----------------|
| ~0.1 s        | start cue      |
| ~2 s          | timeout        |
| (onset coincides with a solenoid onset) | reward |

`parse_trials` then walks each outcome tone back to the nearest preceding start
tone, so start ↔ outcome pairing is read directly off the recorded tone channel.
Reward, timeout, and error are all physically present in the signal.

### intan — outcome is *inferred* from lick timing
On the Intan rig **there is no outcome tone.** `parse_tones` extracts only the
start cue (`start_tone_on` / `start_tone_off`). The outcome for each trial is
reconstructed behaviorally in `parse_trials`, which iterates through
`start_tone_ts` and, for each cue, assigns exactly one outcome:

- **reward** — the first lick falls **3.333–7 s** after the cue **and** a
  `solenoid_on` occurs within **0.1 s after** that lick.
- **early** — any other first lick (e.g. a premature lick < 3.333 s, or a lick in
  the reward window with no solenoid).
- **timeout** — **no lick** occurs within **7 s** of the cue; a synthetic outcome
  timestamp is placed at **cue + 7 s**.

Consequences:

- `trial_outcome_ts` is **derived**, not measured, and is guaranteed to be the
  same length as `start_tone_ts` (one outcome per cue — enforced by an assertion
  in `parse_trials`).
- The 3.333 s / 7 s constants and the 0.1 s solenoid window are the load-bearing
  parameters of the Intan pipeline. On nidq these thresholds don't exist because
  the tone tells you the answer.
- Intan trials therefore end either at the first lick (reward/early) or at
  cue + 7 s (timeout), whereas nidq trials end at the outcome tone.

---

## 2. LED / opto input channels

### nidq — one channel per LED color
The nidq rig wires **a single channel per opsin/LED color**, named by opsin:

```python
# parse_nidq.py
default_ainmap  = {'chrimson': 6, 'chr2': 7}
default_optoidx = {'chrimson': default_ints, 'chr2': default_ints}
```

`get_opto_ts` iterates those one-per-color channels directly.

### intan — a linked *pair* of drivers per color
The Intan rig drives **two LED drivers per color**. The two drivers in each pair
are hardware-linked so they fire simultaneously, and both feed the Intan ADC —
so there are two input channels per color instead of one:

```python
# parse_intan_input.py
default_adcmap = {'blue_1': 0, 'blue_2': 1, 'orange_1': 2, 'orange_2': 3}
default_optoidx = {'orange': [50,100,500,1000], 'blue': [50,100,500,1000]}
```

Note the two keyed layers this creates:

- **`adcmap`** is keyed **per channel** (`blue_1`, `orange_2`, …) → ADC index.
- **`optoidx`** is keyed **per color** (`blue`, `orange`) → the list of power
  levels, shared by both drivers of that color.

So mapping an Intan opto channel to its power list requires stripping the
`_1`/`_2` suffix to recover the color (`blue_1` → `blue`), a step that has no
analog on nidq where the channel key *is* the color.

Naming also shifts with the hardware: nidq names opto by opsin (`chrimson`,
`chr2`), Intan wires it by physical LED color (`orange`, `blue`).

Because the two drivers of a pair are hardware-matched (identical timing),
`get_opto_ts` emits **one representative channel per color** and relabels it by
**opsin** — `blue → chr2`, `orange → chrimson` (via `default_opsin_map`) — so the
resulting events (`chr2_trace`, `chr2_on_50`, `chrimson_off_200`, …) match the
nidq naming exactly and later analyses keyed on `chr2_*` / `chrimson_*` work
across both rigs unchanged. The `_1` / `_2` channel suffixes are collapsed and do
not appear in the `TsGroup`. Resolution steps: select channels whose name
contains a color from `optoidx.keys()`, dedupe to one per color, look up powers
via `optoidx[color]`, and label via `opsin_map[color]`.

---

## Summary

| | `parse_nidq.py` | `parse_intan_input.py` |
|---|---|---|
| Rig | NI-DAQ / Neuropixels | Intan |
| Trial outcome | explicit outcome tone (by duration + solenoid) | inferred from first-lick timing / solenoid, else cue + 7 s |
| `parse_tones` returns | cue + reward + timeout + error tones | start cue only |
| Trial pairing | outcome tone → nearest preceding cue | one synthesized outcome per cue |
| LED channels | 1 per color | matched pair per color, collapsed to 1 |
| Opto input keys | by opsin (`chrimson`, `chr2`) | `adcmap` per channel, `optoidx` per color (`orange`, `blue`) |
| Opto event labels | opsin (`chr2_on_50`, …) | opsin (`chr2_on_50`, …) — remapped from color for cross-compat |
