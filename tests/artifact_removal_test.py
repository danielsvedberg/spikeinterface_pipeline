"""Experiments on the remove_artifacts step, each a set of variants of the real protocol.

Every experiment starts from the session's own protocol
(si_pipeline.build_preprocessing_protocol) and derives its variants from it, so they
differ from the pipeline, and from each other, in exactly one thing:

    order   where remove_artifacts sits: first (the pipeline default), last (after
            common_reference), or dropped. Settled on DS45_20260811: first cuts the
            ringing outside the window 2.5-3x; last is identical to none outside it,
            because run after the bandpass it can only overwrite in-window samples.
    window  the width of the interpolated span, symmetric, with the step first.

Each experiment writes to qc_test/<experiment>/ in the recording folder: side-by-side
opto-triggered averages at a narrow and a wide window, the silence-period onset
average, and the two ringing measurements of tests/artifact_ringing.py. An experiment
with full_qc also runs the whole preprocessing QC per variant, in
qc_test/<experiment>/<variant>/ (the raw half once, since the raw input is identical).
Nothing is written to the preprocess folder; every chain here is lazy.

    python -m tests.artifact_removal_test [experiment] [session_folder]
"""
from pathlib import Path
import json
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import spikeinterface.full as si

from preprocessing import preprocess_qc as pqc
from preprocessing import detect_saturation as dsat
from preprocessing import get_stimulation_frames as gsf
from scripts import si_pipeline as sip
from utils import file_structure as fs
from tests import artifact_ringing as ring

matplotlib.use('Agg')

DEFAULT_FOLDER = Path(r"D:\experiment_data\recording_files\DS45\DS45_20260811")
# the opto averages are read at two widths: +/-5 ms is the artifact itself and the
# interpolated span; +/-50 ms is where bandpass ringing off an unremoved transient shows
OPTO_WINDOWS_MS = (5.0, 50.0)
MAX_OPTO_EVENTS = 500
MAX_LICK_EVENTS = 500


def reorder(protocol, position):
    """The protocol with remove_artifacts moved to the front, the back, or dropped."""
    ra = protocol['remove_artifacts']
    rest = {k: v for k, v in protocol.items() if k != 'remove_artifacts'}
    if position == 'first':
        return {'remove_artifacts': ra, **rest}
    if position == 'last':
        return {**rest, 'remove_artifacts': ra}
    if position == 'none':
        return rest
    raise ValueError(position)


def with_window(protocol, ms):
    """The protocol with remove_artifacts interpolating +/-ms around each trigger."""
    return {k: ({**v, 'ms_before': ms, 'ms_after': ms} if k == 'remove_artifacts' else v)
            for k, v in protocol.items()}


# variant name -> protocol transform; the names are folder names and plot labels
EXPERIMENTS = {
    'order': {'variants': {p: (lambda proto, p=p: reorder(proto, p))
                           for p in ('first', 'last', 'none')},
              'full_qc': True},
    'window': {'variants': {f'+/-{w:g}ms': (lambda proto, w=w: with_window(proto, w))
                            for w in (0.5, 0.75, 1.0)},
               'full_qc': False},
}


def build(raw_rec, protocol):
    """(pre_car_rec, final_rec), split the way run_pipeline splits it."""
    before, after = pqc.split_protocol(protocol, 'common_reference')
    pre_car = si.apply_preprocessing_pipeline(raw_rec, before) if before else raw_rec
    final = si.apply_preprocessing_pipeline(pre_car, after) if after else pre_car
    return pre_car, final


def compare_triggered(finals, triggers, ms_window, folder, name, title,
                      span=None, marks=()):
    """One panel per variant plus an overlay of their grand averages, on one y-scale.

    The per-variant panels share a y-axis on purpose, unlike the QC's own plots: the
    question here is which chain leaves the least trigger-locked signal, and that is a
    comparison of magnitudes.
    """
    colors = ring.variant_colors(finals)
    rec0 = next(iter(finals.values()))
    fs_hz = rec0.get_sampling_frequency()
    half = int(round(ms_window * fs_hz / 1000.0))
    t_ms = np.arange(-half, half) / fs_hz * 1000.0
    baseline = t_ms < t_ms[0] + 0.2 * (t_ms[-1] - t_ms[0])
    sel = pqc._select_events(rec0.get_num_samples(), np.asarray(triggers, np.int64),
                             half, MAX_OPTO_EVENTS)
    if sel.size == 0:
        return {'error': 'no usable triggers'}

    n = len(finals)
    fig, axs = plt.subplots(1, n + 1, figsize=(5.2 * (n + 1), 4.6), sharex=True,
                            sharey=True, constrained_layout=True)
    stats, grands = {'n_events': int(sel.size), 'ms_window': ms_window}, {}
    for ax, (variant, rec) in zip(axs, finals.items()):
        print(f'  compare: {name} / {variant}')
        a = pqc._event_average(rec, sel, half, baseline, in_uV=pqc._traces_in_uV(rec))
        grands[variant] = pqc._draw_event_panel(ax, t_ms, a, 'uV',
                                                shank_ids=pqc._shank_ids(rec))
        m = pqc._panel_metrics(grands[variant], fs_hz, 300.0, baseline)
        stats[variant] = m
        ax.set_title(f'{variant}\nswing {m["swing_uV"]:.1f} uV, '
                     f'above 300 Hz {m["peak_uV"]:.1f} uV (floor {m["hf_floor_uV"]:.1f})',
                     fontsize=9)
    ov = axs[-1]
    for variant, g in grands.items():
        ov.plot(t_ms, g, color=colors[variant], lw=1.5, label=variant)
    ov.set_title('average across channels, all variants', fontsize=9)
    ov.set_xlabel('time from trigger (ms)')
    for ax in axs:
        if span is not None:
            ax.axvspan(*span, color='tab:orange', alpha=0.12, zorder=0)
        for ms, label in marks:
            ax.axvline(ms, color='0.3', ls='--', lw=1.0, zorder=2, label=label)
        ax.legend(fontsize=7, loc='upper right')
    fig.suptitle(f'{title}, {sel.size} events, fully preprocessed', fontsize=10)
    pqc._save(fig, folder, name)
    return stats


def main(experiment='window', base_folder=DEFAULT_FOLDER):
    spec = EXPERIMENTS[experiment]
    si.set_global_job_kwargs(n_jobs=-1, chunk_duration='10s', progress_bar=True)
    folders = fs.SessionPaths(base_folder)
    out = folders.recording_folder / 'qc_test' / experiment
    out.mkdir(parents=True, exist_ok=True)

    spikeglx_folder = list(folders.recording_folder.glob('*_g*'))[0]
    raw_rec = si.read_spikeglx(spikeglx_folder, stream_id='imec0.ap')
    sf = raw_rec.get_sampling_frequency()
    on_t, off_t = gsf.get_stimulation_times(base_folder)
    artifact_idxs = sorted(int(t * sf) for t in on_t + off_t)
    # the pipeline's own cache: same folder, so this reuses a scan the pipeline already did
    sat = dsat.detect_saturation_cached(
        raw_rec, next((f for f in (folders.recording_folder / fs.SATURATION_FOLDER,
                                   folders.recording_folder / fs.PREPROCESS_FOLDER / 'qc')
                       if (f / 'saturation.npz').exists()),
                      folders.recording_folder / fs.SATURATION_FOLDER))
    base_protocol, bad_chids = sip.build_preprocessing_protocol(raw_rec, artifact_idxs, sat)
    # with no railed or manual channels the protocol leaves detection to each build,
    # which samples random chunks and need not agree with itself. Settle it once so the
    # variants -- and the two arms of the injection test -- share one channel set
    dib = base_protocol['detect_and_interpolate_bad_channels']
    if dib['bad_channel_ids'] is None:
        dib['bad_channel_ids'] = ring.detected_bad_channels(build(raw_rec, base_protocol)[0])
        print(f'bad channels pinned: {dib["bad_channel_ids"]}')
        bad_chids = dib['bad_channel_ids'] or None
    lick = {k: (t * sf).astype(int) for k, t in
            gsf.get_event_times(base_folder, events=('lick_on',)).items() if t.size}

    variants = list(spec['variants'])
    finals, summaries, protocols = {}, {}, {}
    for i, (variant, make) in enumerate(spec['variants'].items()):
        protocols[variant] = protocol = make(base_protocol)
        ra = protocol.get('remove_artifacts')
        print(f'\n=== {variant}: {" -> ".join(protocol)}'
              + (f"  (interpolating -{ra['ms_before']:g}/+{ra['ms_after']:g} ms)" if ra else '')
              + ' ===')
        pre_car, final = build(raw_rec, protocol)
        finals[variant] = final
        if spec['full_qc']:
            pqc.run_preprocessing_qc(
                raw_rec, protocol, out / variant, pre_car_rec=pre_car, final_rec=final,
                manual_bad_channel_ids=bad_chids, run_raw=(i == 0),
                rail=sat['rail'], event_triggers=lick, max_event_trials=MAX_LICK_EVENTS,
                # the saturation scan above has already read the whole file once
                full_saturation_scan=False)
            summaries[variant] = (out / variant / 'preprocessing_qc_summary.txt').read_text()

    _, span = ring.spans_of(protocols)
    compare = {}
    for w in OPTO_WINDOWS_MS:
        compare[f'opto_{w:g}ms'] = compare_triggered(
            finals, artifact_idxs, w, out, f'compare_opto_{w:g}ms',
            f'opto-triggered average +/-{w:g} ms', span=span)
    silenced = pqc._silenced_periods(base_protocol)
    if silenced is not None:
        pad = dsat.EPOCH_MARGIN_MS
        compare['silence_onset'] = compare_triggered(
            finals, silenced[:, 0], pad + 10.0, out, 'compare_silence_onset',
            'silence_periods onset average',
            marks=((pad, f'clipping begins (+{pad:g} ms)'),))

    obs = ring.observed_ringing(finals, artifact_idxs, silenced, protocols, out)
    inj = ring.injection_ringing(raw_rec, protocols, [int(t * sf) for t in on_t],
                                 artifact_idxs, silenced, sat['rail'], out)

    (out / 'compare_stats.json').write_text(json.dumps(
        {**compare, 'ringing_observed': obs, 'ringing_injected': inj},
        indent=2, default=float))
    w = max(len(v) for v in variants)
    lines = [f'experiment: {experiment}   session: {base_folder}', '']
    for key, st in compare.items():
        if 'error' in st:
            lines.append(f'{key}: {st["error"]}')
            continue
        lines.append(f'{key} ({st["n_events"]} events, +/-{st["ms_window"]:g} ms)')
        for v in variants:
            m = st[v]
            lines.append(f'  {v:>{w}}: swing {m["swing_uV"]:8.1f} uV   above 300 Hz '
                         f'{m["peak_uV"]:7.1f} uV   floor {m["hf_floor_uV"]:6.1f} uV')
    lines += ring.summary_lines(obs, inj)
    print('\n'.join(lines))
    for v in summaries:
        lines += ['', f'==================== {v} ====================', summaries[v]]
    (out / 'compare_summary.txt').write_text('\n'.join(lines))
    print(f'\nresults in {out}')


if __name__ == '__main__':  # required on Windows: job workers re-import this module
    main(sys.argv[1] if len(sys.argv) > 1 else 'window',
         Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_FOLDER)
