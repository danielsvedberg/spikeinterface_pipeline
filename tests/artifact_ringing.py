"""Filter ringing off the opto artifact, measured two ways, for each preprocessing variant.

observed_ringing -- the trigger-locked residual in the real data. Every channel is
    averaged over events on its own, and the channels are combined by RMS into an
    envelope r(t). Combining by RMS rather than averaging is the point: the common
    reference removes whatever the channels share, so an average across channels reads
    ~0 over residuals that are tens of uV on each channel. The floor is what averaging
    n events leaves of the noise, sigma/sqrt(n), measured off both far edges of the
    window; ringing is whatever clears it.

injection_ringing -- the paired test. A known artifact (the session's own raw
    opto-onset average) is added to the raw recording at fake trigger frames well
    clear of the real ones, the amplifier rail is reapplied, and each chain is run on
    the recording with and without it. The difference is the chain's response to the
    artifact and nothing else: no averaging is needed to remove neural signal or
    noise, so its extent can be read against the per-event noise the sorter sees.
    Everything else is held equal between the two arms -- the remove_artifacts trigger
    list (real + fake), the silenced periods, the bad-channel set -- and a zero-amplitude
    injection is run first to prove the difference really is zero when it should be.

The filter is zero-phase (filtfilt), so ringing runs before the event as well as after
it, and every extent is reported on both sides.
"""
import numpy as np
import matplotlib.pyplot as plt
import spikeinterface.full as si
from scipy.ndimage import uniform_filter1d

from preprocessing import preprocess_qc as pqc
from preprocessing import detect_saturation as dsat

def variant_colors(names):
    """One colour per variant, from the QC's categorical palette, in the given order."""
    return dict(zip(names, pqc.SHANK_COLORS + tuple(f'C{i}' for i in range(4, 10))))


def spans_of(protocols):
    """({variant: interpolated span or None}, common span) from each protocol.

    The common span is the union of every variant's own: the one stretch of samples
    that is excluded from the "outside" metrics for all of them alike, so those
    metrics compare the same samples across variants. A variant's own span is what
    its extents are read from -- for a narrower window, the samples between its edge
    and the common one are exactly the residue that choosing it leaves behind.
    """
    spans = {}
    for v, p in protocols.items():
        ra = p.get('remove_artifacts')
        spans[v] = (-ra.get('ms_before', 0.5), ra.get('ms_after', 0.5)) if ra else None
    own = [s for s in spans.values() if s is not None]
    common = (min(s[0] for s in own), max(s[1] for s in own)) if own else (0.0, 0.0)
    return {v: (s if s is not None else common) for v, s in spans.items()}, common
RING_WINDOW_MS = 50.0
# far-edge share of the window taken as the floor, on each side
EDGE_FRAC = 0.2
# template of the injected artifact, relative to the opto onset
TEMPLATE_MS = (-5.0, 25.0)
TEMPLATE_CORE_X = 4.0
HP_HZ = 300.0
# the transient's peak is looked for this close to the light edge
PEAK_SEARCH_MS = 2.0
# short, because the taper lies outside the artifact's core and is invented shape: at
# 1 ms it put a full millisecond of artificial artifact either side, straddling the
# very window edges being compared. 0.25 ms is still ~1.25 periods of the 5 kHz corner
TAPER_MS = 0.25
# fake events: this far from any real trigger, and this far from each other
FAKE_GUARD_MS = 300.0
MAX_FAKE_EVENTS = 150


def detected_bad_channels(rec):
    """The channel ids a lazy chain's detect_and_interpolate step settled on.

    Walks the chain back to that step. Pinning its answer into the protocol stops every
    later build from re-running a random-chunk detection that can land differently --
    fatal for a paired difference, merely confusing for a side-by-side comparison.
    """
    while rec is not None:
        if type(rec).__name__.startswith('DetectAndInterpolate'):
            return list(rec._kwargs['bad_channel_ids'])
        rec = rec._kwargs.get('recording', rec._kwargs.get('parent_recording'))
    return None


def envelope(avg):
    """RMS across channels of per-channel averages: (n_t, n_ch) -> (n_t,)."""
    return np.sqrt(np.mean(avg ** 2, axis=1))


def extent(t_ms, r, thresh, span):
    """(earliest, latest) time from the trigger, in ms, at which r exceeds thresh.

    Absolute times rather than distances from the interpolated span, so they read
    directly against anything else on the same clock -- the silence_periods pad in
    particular. Farthest crossing, not first drop below: a decaying oscillation dips
    through the threshold at every ripple of its envelope before it is actually done.
    Where nothing outside the span clears the threshold, the span's own edge is
    returned.
    """
    over = r > thresh
    pre = t_ms[over & (t_ms < span[0])]
    post = t_ms[over & (t_ms > span[1])]
    return (float(pre.min()) if pre.size else float(span[0]),
            float(post.max()) if post.size else float(span[1]))


def _window(fs, ms_window):
    half = int(round(ms_window * fs / 1000.0))
    t_ms = np.arange(-half, half) / fs * 1000.0
    edges = np.abs(t_ms) > (1.0 - EDGE_FRAC) * ms_window
    return half, t_ms, edges


def _clear_of(frames, periods, half):
    """Frames whose +/-half window touches no silenced period."""
    if periods is None or not len(periods):
        return np.ones(frames.size, bool)
    p = np.asarray(periods, np.int64)
    p = p[np.argsort(p[:, 0])]
    lo = np.searchsorted(p[:, 1], frames - half, side='right')
    hi = np.searchsorted(p[:, 0], frames + half, side='left')
    return hi <= lo


def _log_axes(ax, t_ms, span, unit):
    ax.set_yscale('log')
    ax.axvspan(*span, color='tab:orange', alpha=0.12, zorder=0,
               label=f'interpolated, widest variant ({span[0]:g} to +{span[1]:g} ms)')
    ax.set_xlabel('time from trigger (ms)')
    ax.set_ylabel(f'RMS across channels ({unit})')
    ax.set_xlim(t_ms[0], t_ms[-1])


def observed_ringing(finals, triggers, silenced, protocols, folder, max_events=500,
                     ms_window=RING_WINDOW_MS):
    """Envelope of the real trigger-locked residual, per variant. Returns stats."""
    spans, span = spans_of(protocols)
    colors = variant_colors(finals)
    rec0 = next(iter(finals.values()))
    fs = rec0.get_sampling_frequency()
    half, t_ms, edges = _window(fs, ms_window)
    trig = np.asarray(triggers, np.int64)
    trig = trig[(trig > half) & (trig < rec0.get_num_samples() - half)]
    clear = _clear_of(trig, silenced, half)
    # events touching a silenced span are part fill in every variant, which would
    # read as less ringing for reasons that have nothing to do with the filter
    sel = pqc._select_events(rec0.get_num_samples(), trig[clear], half, max_events)
    stats = {'n_triggers': int(trig.size), 'n_clear_of_silence': int(clear.sum()),
             'n_events': int(sel.size), 'ms_window': ms_window, 'common_span_ms': span,
             'variants': list(finals)}
    if sel.size == 0:
        stats['error'] = 'every trigger window overlaps a silenced period'
        return stats

    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    for variant, rec in finals.items():
        print(f'  ringing (observed): {variant}')
        in_uV = pqc._traces_in_uV(rec)
        r = envelope(pqc._event_average(rec, sel, half, edges, in_uV=in_uV))
        floor = float(np.sqrt(np.mean(r[edges] ** 2)))
        own = spans[variant]
        outside_own = (t_ms < own[0]) | (t_ms > own[1])
        outside = (t_ms < span[0]) | (t_ms > span[1])
        pre, post = extent(t_ms, r, 2.0 * floor, own)
        stats[variant] = {'floor_uV': floor, 'peak_uV': float(r.max()),
                          'span_ms': own,
                          'peak_outside_window_uV': float(r[outside_own].max()),
                          'excess_rms_outside_uV': float(np.sqrt(max(
                              np.mean(r[outside] ** 2) - floor ** 2, 0.0))),
                          'extent_ms': [pre, post]}
        c = colors[variant]
        ax.plot(t_ms, r, color=c, lw=1.2,
                label=f'{variant}: >2x floor from {pre:+.1f} to {post:+.1f} ms')
        ax.axhline(2.0 * floor, color=c, ls=':', lw=0.8)
    _log_axes(ax, t_ms, span, 'uV')
    ax.set_title(f'observed trigger-locked residual, {sel.size} events clear of '
                 f'silence_periods ({clear.sum()} of {trig.size})\n'
                 'dotted: 2x the averaging floor', fontsize=9)
    ax.legend(fontsize=7, loc='upper right')
    pqc._save(fig, folder, 'ringing_observed')
    return stats


def _fake_frames(real, n_samples, silenced, half, fs):
    """Event frames spaced FAKE_GUARD_MS apart inside the gaps between real triggers."""
    guard = int(round(FAKE_GUARD_MS * fs / 1000.0))
    bounds = np.r_[0, np.sort(np.asarray(real, np.int64)), n_samples]
    fakes = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a > 2 * guard:
            fakes.append(np.arange(a + guard, b - guard + 1, guard))
    fakes = np.concatenate(fakes) if fakes else np.array([], np.int64)
    fakes = fakes[(fakes > guard) & (fakes < n_samples - guard)]
    fakes = fakes[_clear_of(fakes, silenced, half)]
    if fakes.size > MAX_FAKE_EVENTS:
        fakes = fakes[np.linspace(0, fakes.size - 1, MAX_FAKE_EVENTS).astype(int)]
    return fakes


def _artifact_template(raw_rec, on_frames, fs):
    """The raw opto-onset average in counts, (n_t, n_ch), cut down to the artifact.

    Counts, not uV: it is added to the int16 raw stream, where the rail is.

    Only the part above HP_HZ is kept (see below), since that is the part that can
    excite the bandpass. A raw average over a few hundred events still carries their
    noise and LFP divided by sqrt(n) -- a few uV per channel over the whole template --
    and at the saturating scale that residue alone clears the per-event noise, which
    would read as ringing that lasts the length of the template. So the template is
    kept only over its core, the contiguous span around the onset transient where its
    RMS across channels stands above TEMPLATE_CORE_X times its own pre-trigger level,
    and tapered to zero over TAPER_MS either side. The taper keeps the cut from being a
    step the filter would ring on.
    """
    nb = int(round(-TEMPLATE_MS[0] * fs / 1000.0))
    na = int(round(TEMPLATE_MS[1] * fs / 1000.0))
    on = np.asarray(on_frames, np.int64)
    on = on[(on > nb) & (on < raw_rec.get_num_samples() - na)]
    base = np.zeros(nb + na, bool)
    base[:max(nb // 2, 1)] = True
    acc = None
    for t in on:
        x = raw_rec.get_traces(start_frame=int(t) - nb, end_frame=int(t) + na,
                               return_in_uV=False).astype(np.float64)
        x -= x[base].mean(axis=0, keepdims=True)
        acc = x if acc is None else acc + x
    tpl = acc / on.size
    # the fast part only. The raw average also carries a slow swing (below the
    # passband, removed by the bandpass whatever the order) that outlasts any template
    # window, so keeping it would make the cut at the window edge a step, and the
    # filter's response to that step would be read as ringing off the artifact
    tpl = tpl - uniform_filter1d(tpl, size=max(int(round(fs / HP_HZ)), 3), axis=0,
                                 mode='nearest')
    r = envelope(tpl)
    # the contiguous run around the transient's peak, not everything above threshold:
    # on a tagging session the average also holds light-evoked spiking a few ms later,
    # which is neural signal, not artifact, and must not be scaled up and injected
    below = r <= TEMPLATE_CORE_X * np.sqrt(np.mean(r[base] ** 2))
    near = np.abs(np.arange(r.size) - nb) <= int(round(PEAK_SEARCH_MS * fs / 1000.0))
    p = int(np.argmax(np.where(near, r, -np.inf)))
    left = np.flatnonzero(below[:p])
    right = np.flatnonzero(below[p:])
    c0 = left[-1] + 1 if left.size else 0
    c1 = p + right[0] if right.size else r.size
    nt = int(round(TAPER_MS * fs / 1000.0))
    w = np.zeros(tpl.shape[0])
    w[c0:c1] = 1.0
    ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, nt)))
    lo = ramp[max(nt - c0, 0):]
    w[c0 - lo.size:c0] = lo
    hi = ramp[::-1][:tpl.shape[0] - c1]
    w[c1:c1 + hi.size] = hi
    core_ms = [(c0 - nb) / fs * 1000.0, (c1 - nb) / fs * 1000.0]
    return tpl * w[:, None], nb, core_ms


def _paired_average(inj, base, frames, half, in_uV):
    acc = None
    for t in frames:
        a = inj.get_traces(start_frame=int(t) - half, end_frame=int(t) + half,
                           return_in_uV=in_uV).astype(np.float64)
        b = base.get_traces(start_frame=int(t) - half, end_frame=int(t) + half,
                            return_in_uV=in_uV).astype(np.float64)
        acc = (a - b) if acc is None else acc + (a - b)
    return acc / len(frames)


def injection_ringing(raw_rec, protocols, on_frames, real_triggers, silenced, rail,
                      folder, ms_window=RING_WINDOW_MS):
    """Paired injection test, per variant and per artifact scale. Returns stats.

    protocols: {variant: protocol} with bad channels already pinned. The fake frames
    are appended to each protocol's remove_artifacts triggers in both arms, so the
    step treats them exactly as it would a real opto event.
    """
    fs = raw_rec.get_sampling_frequency()
    n = raw_rec.get_num_samples()
    spans, span = spans_of(protocols)
    colors = variant_colors(protocols)
    half, t_ms, edges = _window(fs, ms_window)
    fakes = _fake_frames(real_triggers, n, silenced, half, fs)
    if fakes.size == 0:
        return {'error': 'no stimulation-free gaps long enough for fake events'}
    tpl, nbefore, core_ms = _artifact_template(raw_rec, on_frames, fs)
    gain = float(np.median(raw_rec.get_channel_gains()))
    peak_counts = float(np.abs(tpl).max())
    # scale 1 is the session's average artifact. Averaged clipped events sit below the
    # rail, so the second scale is chosen to put the template's peak past it, which is
    # the case silence_periods exists for
    scales = {'x0 (null check)': 0.0, 'x1 (average artifact)': 1.0}
    if rail is not None:
        scales['saturating'] = max(1.5 * rail / peak_counts, 1.5)
    stats = {'n_fake_events': int(fakes.size), 'template_peak_uV': peak_counts * gain,
             'template_core_ms': core_ms, 'rail_counts': rail, 'common_span_ms': span,
             'variants': list(protocols), 'scales': {}}
    print(f'  ringing (injected): {fakes.size} fake events, template peak '
          f'{peak_counts * gain:.0f} uV over {core_ms[0]:+.1f} to {core_ms[1]:+.1f} ms, '
          f'scales {list(scales.values())}')

    sorting = si.NumpySorting.from_samples_and_labels(
        [fakes], [np.zeros(fakes.size, int)], fs)
    clip = (lambda r: si.clip(r, a_min=-rail, a_max=rail)) if rail is not None else (lambda r: r)
    raw_base = clip(raw_rec)

    def chain(rec, variant):
        p = dict(protocols[variant])
        if 'remove_artifacts' in p:
            ra = dict(p['remove_artifacts'])
            ra['list_triggers'] = sorted(set(ra['list_triggers']) | set(fakes.tolist()))
            p['remove_artifacts'] = ra
        return si.apply_preprocessing_pipeline(rec, p)

    bases = {v: chain(raw_base, v) for v in protocols}
    in_uV = {v: pqc._traces_in_uV(r) for v, r in bases.items()}
    sigma = {v: float(np.median(si.get_noise_levels(r, return_in_uV=in_uV[v])))
             for v, r in bases.items()}
    stats['per_event_noise_uV'] = sigma
    outside = (t_ms < span[0]) | (t_ms > span[1])

    for label, scale in scales.items():
        inj_raw = clip(si.InjectTemplatesRecording(
            sorting, (tpl * scale)[None], nbefore=nbefore, parent_recording=raw_rec))
        # the envelope of what went in, in uV, for scale on the plot
        t_tpl = (np.arange(tpl.shape[0]) - nbefore) / fs * 1000.0
        clipped_in = np.clip(tpl * scale, -rail, rail) if rail is not None else tpl * scale
        entry = {'scale': scale}
        if rail is not None and scale > 0:
            railed = np.abs(tpl * scale).max(axis=1) >= rail
            if railed.any():
                entry['clipped_span_ms'] = [float(t_tpl[railed].min()),
                                            float(t_tpl[railed].max())]
        fig, axs = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True,
                                gridspec_kw={'width_ratios': [3, 2]})
        ax, zx = axs
        ex_ch = int(np.argmax(np.abs(tpl).max(axis=0)))
        for variant in protocols:
            print(f'  ringing (injected {label}): {variant}')
            d = _paired_average(chain(inj_raw, variant), bases[variant], fakes, half,
                                in_uV[variant])
            r = envelope(d)
            s = sigma[variant]
            if scale == 0.0:
                entry[variant] = {'max_abs_diff': float(np.abs(d).max())}
                continue
            own = spans[variant]
            outside_own = (t_ms < own[0]) | (t_ms > own[1])
            pre1, post1 = extent(t_ms, r, s, own)
            pre01, post01 = extent(t_ms, r, 0.1 * s, own)
            entry[variant] = {'peak_uV': float(r.max()), 'span_ms': own,
                              'peak_outside_window_uV': float(r[outside_own].max()),
                              'rms_outside_window_uV': float(np.sqrt(np.mean(r[outside] ** 2))),
                              'extent_over_noise_ms': [pre1, post1],
                              'extent_over_0.1noise_ms': [pre01, post01]}
            c = colors[variant]
            ax.plot(t_ms, np.maximum(r, 1e-3), color=c, lw=1.2,
                    label=f'{variant}: >noise {pre1:+.1f} to {post1:+.1f} ms, '
                          f'>0.1 noise {pre01:+.1f} to {post01:+.1f} ms')
            zoom = np.abs(t_ms) <= 10.0
            zx.plot(t_ms[zoom], d[zoom, ex_ch], color=c, lw=1.0, label=variant)
            for x in own:
                zx.axvline(x, color=c, lw=0.6, ls=':')
        stats['scales'][label] = entry
        plt_name = 'ringing_injected_' + label.split(' ')[0]
        if scale == 0.0:
            plt.close(fig)
            worst = max(entry[v]['max_abs_diff'] for v in protocols)
            if worst > 1e-6:
                raise RuntimeError(f'null injection left a {worst:g} difference: the two '
                                   'arms are not built identically, so no paired number '
                                   'can be trusted')
            print('  ringing: null injection difference is exactly zero on every variant')
            continue
        env_in = np.sqrt(np.mean((clipped_in * gain) ** 2, axis=1))
        ax.plot(t_tpl, np.maximum(env_in, 1e-3), color='0.55', lw=1.0,
                label='injected artifact, raw (before the chain)')
        s_med = float(np.median(list(sigma.values())))
        ax.axhline(s_med, color='k', ls='--', lw=0.8, label=f'per-event noise {s_med:.1f} uV')
        ax.axhline(0.1 * s_med, color='k', ls=':', lw=0.8, label='0.1x per-event noise')
        if 'clipped_span_ms' in entry:
            c0, c1 = entry['clipped_span_ms']
            pad = dsat.EPOCH_MARGIN_MS
            for x in (c0 - pad, c1 + pad):
                ax.axvline(x, color='tab:red', ls='--', lw=1.0)
            ax.plot([], [], color='tab:red', ls='--',
                    label=f'silence_periods would blank {c0 - pad:.1f} to {c1 + pad:.1f} ms')
        _log_axes(ax, t_ms, span, 'uV')
        ax.set_ylim(bottom=max(1e-2, 0.01 * s_med))
        ax.set_title(f'chain response to an injected artifact, {label} '
                     f'({fakes.size} fake events)', fontsize=9)
        ax.legend(fontsize=7, loc='upper right')
        zx.set_title(f'channel {raw_rec.channel_ids[ex_ch]} (largest artifact), +/-10 ms',
                     fontsize=9)
        zx.axvspan(*span, color='tab:orange', alpha=0.12, zorder=0)
        zx.set_xlabel('time from trigger (ms)')
        zx.set_ylabel('injected minus uninjected (uV)')
        zx.legend(fontsize=7)
        pqc._save(fig, folder, plt_name)
    return stats


def summary_lines(obs, inj):
    """Plain-text table of both measurements, for compare_summary.txt.

    "peak outside window" is outside each variant's own interpolated span; "RMS
    outside" is outside the common (widest) span, so it compares the same samples.
    """
    L = ['', 'RINGING']
    if 'error' in obs:
        L.append(f"observed: {obs['error']}")
    else:
        L.append(f"observed ({obs['n_events']} events; {obs['n_clear_of_silence']} of "
                 f"{obs['n_triggers']} triggers clear of silence_periods, "
                 f"+/-{obs['ms_window']:g} ms); extents are ms from the trigger")
        w = max(len(v) for v in obs['variants'])
        for v in obs['variants']:
            if v in obs:
                m = obs[v]
                L.append(f"  {v:>{w}}: floor {m['floor_uV']:.2f} uV  peak outside window "
                         f"{m['peak_outside_window_uV']:.2f} uV  excess RMS outside "
                         f"{m['excess_rms_outside_uV']:.2f} uV  >2x floor from "
                         f"{m['extent_ms'][0]:+.1f} to {m['extent_ms'][1]:+.1f} ms")
    if 'error' in inj:
        L.append(f"injected: {inj['error']}")
        return L
    L.append(f"injected ({inj['n_fake_events']} fake events, template peak "
             f"{inj['template_peak_uV']:.0f} uV, artifact core "
             f"{inj['template_core_ms'][0]:+.1f} to {inj['template_core_ms'][1]:+.1f} ms "
             f"+ {TAPER_MS:g} ms taper; per-event noise "
             + ', '.join(f"{v} {s:.1f}" for v, s in inj['per_event_noise_uV'].items())
             + ' uV)')
    for label, e in inj['scales'].items():
        if e['scale'] == 0.0:
            L.append(f"  {label}: difference exactly zero on every variant")
            continue
        cs = e.get('clipped_span_ms')
        L.append(f"  {label} (x{e['scale']:.2f})"
                 + (f", clipped {cs[0]:.1f} to {cs[1]:.1f} ms -> silence_periods would "
                    f"blank {cs[0] - dsat.EPOCH_MARGIN_MS:.1f} to "
                    f"{cs[1] + dsat.EPOCH_MARGIN_MS:.1f} ms" if cs else ''))
        w = max(len(v) for v in inj['variants'])
        for v in inj['variants']:
            if v in e:
                m = e[v]
                a, b = m['extent_over_noise_ms']
                c, d = m['extent_over_0.1noise_ms']
                L.append(f"    {v:>{w}}: peak outside window {m['peak_outside_window_uV']:.1f} uV  "
                         f"RMS outside {m['rms_outside_window_uV']:.2f} uV  "
                         f">noise {a:+.1f} to {b:+.1f} ms, >0.1 noise {c:+.1f} to {d:+.1f} ms")
    return L
