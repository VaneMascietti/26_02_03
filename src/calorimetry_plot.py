"""Render stored geometry; no detector or onset fitter is called here."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .calorimetry_store import load_bundle


def _broken(t, values, discontinuity=None):
    y = np.asarray(values, dtype=float).copy()
    steps = np.diff(t)
    positive = steps[steps > 0]
    if len(positive):
        indices = np.flatnonzero((steps <= 0) | (steps > 5 * np.median(positive))) + 1
        y[indices] = np.nan
    if discontinuity is not None:
        y[np.asarray(discontinuity, dtype=bool)] = np.nan
    return y


def _protect_sources(out, sets):
    target = Path(out).resolve()
    for item in sets:
        for source in item.metadata.get('sources', []):
            for key in ('path', 'archived_path'):
                if source.get(key) and target == Path(source[key]).resolve():
                    raise ValueError('Figure output cannot overwrite original or archived source data.')


def render_bundle(bundle, out, *, dataset_id=None, event_id=None, color='#554173',
                  line_width=1.7, line_style='-', font_size=18, legend_size=14, overwrite=False):
    payload, sets = load_bundle(bundle)
    if event_id:
        events = [e for e in payload['events'] if e['event_id'] == event_id]
        if not events:
            raise ValueError(f'Unknown event: {event_id}')
        dataset_id = events[0]['dataset_id']
    else:
        events = payload['events']
    if dataset_id is None and len(sets) > 1:
        raise ValueError('Multiple datasets: choose --dataset.')
    item = next((s for s in sets if dataset_id is None or s.id == dataset_id), None)
    if item is None:
        raise ValueError(f'Unknown dataset: {dataset_id}')
    events = [e for e in events if e['dataset_id'] == item.id]
    f = item.frame
    if event_id and events[0]['selection'].get('row_range') is not None:
        start, end = events[0]['selection']['row_range']
        f = f.iloc[int(start):int(end)+1]
    if event_id:
        low, high = events[0]['selection']['window_s']
        f = f.loc[(f.time_s >= low) & (f.time_s <= high)]
    if f.empty:
        raise ValueError('No signals in selected window.')
    out = Path(out)
    _protect_sources(out, sets)
    if out.exists() and not overwrite:
        raise FileExistsError(f'Figure exists: {out}; choose a new path or --overwrite.')
    # Figure paths must not replace saved results or raw files, even with --overwrite.
    if out.suffix.lower() not in ('.png', '.pdf', '.svg'):
        raise ValueError('Figure extension must be png, pdf or svg.')
    t = f.time_s.to_numpy(float)
    with plt.rc_context({'font.size': font_size}):
        fig, (ax, hf) = plt.subplots(2, 1, figsize=(16, 11), sharex=True, layout='constrained')
        temp_cols = [('sample_temperature_c', 'Sample temperature', '#285ba6'),
                     ('furnace_temperature_c', 'Furnace temperature', '#dc8430'),
                     ('huber_temperature_c', 'Huber temperature', '#16846b')]
        if not any(c in f for c, _, _ in temp_cols) and 'temperature_c' in f:
            temp_cols = [('temperature_c', f"Temperature ({item.metadata.get('temperature_source', 'instrument')})", '#285ba6')]
        for col, label, c in temp_cols:
            if col in f:
                ax.plot(t / 3600, _broken(t, f[col], f.get('source_discontinuity')), label=label, color=c, lw=line_width, ls=line_style)
        ax.set_ylabel(f"Temperature ({item.units.get('temperature_c', '°C')})")
        pressure = None
        if 'pressure' in f:
            pressure = ax.twinx()
            pressure.plot(t / 3600, _broken(t, f.pressure, f.get('source_discontinuity')), color='#555555', ls='--', lw=line_width, label='Pressure')
            pressure.set_ylabel(f"Pressure ({item.units['pressure']})")
            pressure.set_zorder(ax.get_zorder()-1)
            ax.patch.set_visible(False)
        hf.plot(t / 3600, _broken(t, f.heat_flow, f.get('source_discontinuity')), color=color, lw=line_width, ls=line_style, label='Original heat flow')
        hf.set_ylabel(f"Heat flow ({item.units['heat_flow']})")
        hf.set_xlabel('Time (h)')
        for e in events:
            fit = e.get('fit')
            if fit:
                for key, label, c in [('baseline', 'Baseline', '#16846b'), ('flank', 'Initial flank', '#dc8430')]:
                    if f'{key}_s' not in fit or f'{key}_coefficients' not in fit:
                        continue
                    low, high = fit[f'{key}_s']
                    mask = (t >= low) & (t <= high)
                    hf.scatter(t[mask] / 3600, f.heat_flow.to_numpy()[mask], s=10, color=c, zorder=3)
                    hf.axvspan(low / 3600, high / 3600, color=c, alpha=.1)
                    onset_time = (e.get('onset') or {}).get('time_s')
                    left = min(low, onset_time) if onset_time is not None else low
                    right = max(high, onset_time) if onset_time is not None else high
                    xx = np.linspace(left, right, 100)
                    coeff = fit[f'{key}_coefficients']
                    hf.plot(xx / 3600, np.polyval(coeff, xx), '--', color=c, lw=line_width,
                            label=f"{e['event_id']} {label.lower()}")
            onset = e.get('onset')
            if onset:
                x = onset['time_s'] / 3600
                for axis in [ax, hf]:
                    axis.axvline(x, color='#ab3354', ls='--', lw=1.4)
                hf.scatter(x, onset['heat_flow'], color='#ab3354', zorder=5)
                text = f"{e['event_id']} | {onset['time_s']:.2f} s"
                if onset['temperature_c']['value'] is not None:
                    text += f"\nT: {onset['temperature_c']['value']:.2f} {item.units.get('temperature_c', '°C')}"
                if onset['pressure']['value'] is not None:
                    text += f"\nP: {onset['pressure']['value']:.2f} {item.units['pressure']}"
                hf.annotate(text, (x, onset['heat_flow']), xytext=(5, 20), textcoords='offset points', fontsize=legend_size)
            elif event_id:
                hf.text(.02, .97, f"Not determined: {e.get('reason')}", transform=hf.transAxes,
                        va='top', fontsize=legend_size, wrap=True)
        handles, labels = ax.get_legend_handles_labels()
        if pressure:
            ph, pl = pressure.get_legend_handles_labels()
            handles += ph
            labels += pl
        if handles:
            ax.legend(handles, labels, loc='upper left', ncol=1, fontsize=legend_size)
        hf.legend(loc='best', fontsize=legend_size)
        for axis in [ax, hf]:
            axis.grid(alpha=.18)
            axis.tick_params(labelsize=font_size)
        fig.suptitle(f'{item.id} — Stored signals and onset geometry', fontsize=font_size + 4)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200)
        plt.close(fig)
    return out


def compare_events(pairs, out, *, overwrite=False, line_width=1.7, font_size=18):
    """Overlay original event windows aligned at saved onset, without reanalysis."""
    traces = []
    source_sets = []
    units = set()
    for bundle, event_id in pairs:
        payload, sets = load_bundle(bundle)
        source_sets.extend(sets)
        e = next((e for e in payload['events'] if e['event_id'] == event_id), None)
        if not e or not e.get('onset'):
            raise ValueError(f'Calculated saved onset required: {event_id}')
        item = next(s for s in sets if s.id == e['dataset_id'])
        low, high = e['selection']['window_s']
        f = item.frame
        if e['selection'].get('row_range') is not None:
            start, end = e['selection']['row_range']
            f = f.iloc[int(start):int(end)+1]
        f = f.loc[(f.time_s >= low) & (f.time_s <= high)]
        units.add(item.units['heat_flow'])
        traces.append((f, e, str(bundle)))
    if len(units) != 1 or len(traces) < 2:
        raise ValueError('Compare at least two events with identical HF units; no implicit normalization.')
    out = Path(out)
    _protect_sources(out, source_sets)
    if out.suffix.lower() not in ('.png', '.pdf', '.svg'):
        raise ValueError('Figure extension must be png, pdf or svg.')
    if out.exists() and not overwrite:
        raise FileExistsError(out)
    with plt.rc_context({'font.size': font_size}):
        fig, ax = plt.subplots(figsize=(12, 7), layout='constrained')
        for f, e, source in traces:
            t = f.time_s.to_numpy(float)
            ax.plot(t - e['onset']['time_s'], _broken(t, f.heat_flow, f.get('source_discontinuity')), lw=line_width,
                    label=f"{Path(source).name}: {e['event_id']}")
        ax.axvline(0, color='black', ls='--', lw=1)
        ax.set(xlabel='Time relative to saved onset (s)', ylabel=f'Heat flow ({next(iter(units))})',
               title='Event comparison — repeatability requires experimental review')
        ax.legend(fontsize=12)
        ax.grid(alpha=.2)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200)
        plt.close(fig)
    return out
