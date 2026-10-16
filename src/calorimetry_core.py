"""Shared time-domain calorimetry: selections, diagnostics and onset geometry.

Operational defaults are exploratory criteria, not scientifically validated rules.
Fits use original HF values; optional smoothing is used only for flank selection.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


def _local_peaks(values):
    """Acquisition-ordered local maxima, including the leading edge of plateaus."""
    values = np.asarray(values)
    return np.flatnonzero((values[1:-1] > values[:-2]) & (values[1:-1] >= values[2:])) + 1


REVIEWS = ('pending_review', 'accepted_event', 'noise', 'instrumental_oscillation')


@dataclass
class AnalysisConfig:
    noise_multiplier: float = 3.
    min_persistence_s: float = 2.
    smooth_s: float = 0.
    baseline_fraction: float = .2
    tangent_half_width_s: float = 2.
    min_fit_points: int = 4
    max_gap_factor: float = 5.
    max_gap_s: float | None = None
    candidate_separation_s: float = 10.
    oscillation_min_cycles: float = 3.
    oscillation_spectral_fraction: float = .5

    def validate(self):
        for key, value in asdict(self).items():
            if value is not None and (not np.isfinite(value) or value < 0):
                raise ValueError(f'Invalid configuration {key}: {value}')
        if not 0 < self.baseline_fraction < .5:
            raise ValueError('baseline_fraction must be between 0 and .5.')
        if self.min_fit_points < 3 or self.max_gap_factor <= 1 or self.noise_multiplier <= 0:
            raise ValueError('Require min_fit_points >= 3, max_gap_factor > 1 and noise_multiplier > 0.')
        if not 0 < self.oscillation_spectral_fraction <= 1:
            raise ValueError('oscillation_spectral_fraction must be in (0, 1].')


def bounds(value, label):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f'{label} must contain [start_s, end_s].')
    a, b = map(float, value)
    if not np.isfinite([a, b]).all() or a >= b:
        raise ValueError(f'{label} must have finite increasing bounds.')
    return a, b


def mask_range(t, interval):
    a, b = bounds(interval, 'range')
    return (t >= a) & (t <= b)


def gap_limit(t, config):
    steps = np.diff(t)
    positive = steps[steps > 0]
    limit = config.max_gap_factor * float(np.median(positive)) if len(positive) else 0.
    return min(limit, config.max_gap_s) if config.max_gap_s is not None else limit


def _fit(t, y, mask, config):
    if mask.sum() < config.min_fit_points:
        raise ValueError(f'Fit region needs at least {config.min_fit_points} finite points.')
    x, z = t[mask], y[mask]
    center = float(np.mean(x))
    slope, centered_intercept = np.polyfit(x - center, z, 1)
    intercept = centered_intercept - slope * center
    residual = z - (slope * x + intercept)
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    total = float(np.sum((z - np.mean(z)) ** 2))
    r2 = 1 - float(np.sum(residual ** 2)) / total if total > 0 else None
    return [float(slope), float(intercept)], {'rmse': rmse, 'r2': r2, 'points': int(mask.sum())}


def _intersection(base, flank):
    denominator = base[0] - flank[0]
    if abs(denominator) <= 1e-12 * max(1., abs(base[0]), abs(flank[0])):
        raise ValueError('Baseline and flank are effectively parallel.')
    return float((flank[1] - base[1]) / denominator)


def sample_signal(frame, name, target, config):
    """Only interpolate adjacent recorded rows; no bridging NaNs or time gaps."""
    if name not in frame:
        return {'value': None, 'method': 'unavailable', 'reason': f'{name} not recorded'}
    t = frame.time_s.to_numpy(float)
    y = pd.to_numeric(frame[name], errors='coerce').to_numpy(float)
    if not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        return {'value': None, 'method': 'unavailable', 'reason': 'Ambiguous or nonmonotonic time'}
    exact = np.flatnonzero(t == target)
    if len(exact):
        value = y[exact[0]]
        return {'value': float(value) if np.isfinite(value) else None,
                'method': 'measured' if np.isfinite(value) else 'unavailable',
                'reason': None if np.isfinite(value) else 'Missing measurement'}
    right = int(np.searchsorted(t, target))
    if right == 0 or right == len(t):
        return {'value': None, 'method': 'unavailable', 'reason': 'Outside record; no extrapolation'}
    left = right - 1
    if 'source_discontinuity' in frame and bool(frame.source_discontinuity.iloc[right]):
        return {'value': None, 'method': 'unavailable', 'reason': 'Omitted source rows or recorded-clock break'}
    if t[right] - t[left] > gap_limit(t, config) or not np.isfinite(y[[left, right]]).all():
        return {'value': None, 'method': 'unavailable', 'reason': 'Missing measurement or data gap'}
    value = y[left] + (target - t[left]) / (t[right] - t[left]) * (y[right] - y[left])
    return {'value': float(value), 'method': 'linear interpolation of adjacent recorded rows',
            'bracket_s': [float(t[left]), float(t[right])], 'reason': None}


def _persistence(t, active):
    best_points, best_duration = 0, 0.
    start = None
    for i, value in enumerate(active):
        if value and start is None:
            start = i
        if start is not None and (not value or i == len(active) - 1):
            end = i if value else i - 1
            best_points = max(best_points, end - start + 1)
            best_duration = max(best_duration, float(t[end] - t[start]))
            start = None
    return best_points, best_duration


def oscillation_indicator(t, residual, config):
    """Narrow-band periodicity hint, not an instrumental-origin classifier.

    Requires near-uniform sampling and at least three cycles. Nonstationarity,
    harmonics and genuine repeated physical events can produce false positives.
    """
    result = {'possible_oscillation': False, 'status': 'not_assessed',
              'limitation': 'Periodicity alone cannot identify instrumental origin; compare cycles and controls.'}
    if len(t) < 16 or np.any(np.diff(t) <= 0):
        return result
    dt = np.diff(t)
    if np.max(np.abs(dt - np.median(dt))) > .1 * np.median(dt):
        result['reason'] = 'Sampling is not sufficiently uniform (10% interval tolerance).'
        return result
    z = residual - np.polyval(np.polyfit(t - t[0], residual, 1), t - t[0])
    spectrum = abs(np.fft.rfft(z * np.hanning(len(z)))) ** 2
    spectrum[0] = 0
    if spectrum.sum() <= np.finfo(float).eps:
        result.update(status='assessed', spectral_fraction=0., cycles=0.)
        return result
    peak = int(np.argmax(spectrum))
    fraction = float(spectrum[max(1, peak-1):peak+2].sum() / spectrum.sum())
    cycles = float(peak * (t[-1] - t[0]) / (len(t) * np.median(dt)))
    zero_crossings = int(np.sum(np.signbit(z[1:]) != np.signbit(z[:-1])))
    possible = (cycles >= config.oscillation_min_cycles and
                fraction >= config.oscillation_spectral_fraction and
                zero_crossings >= 2 * config.oscillation_min_cycles)
    result.update(status='assessed', possible_oscillation=bool(possible),
                  spectral_fraction=fraction, cycles=cycles, zero_crossings=zero_crossings,
                  dominant_period_s=float(len(t) * np.median(dt) / peak) if peak else None)
    return result


def analyze_window(signal_set, selection, config=None):
    """One geometric estimate per selected event, independent of manual classification."""
    config = config or AnalysisConfig()
    config.validate()
    event_id = str(selection['event_id'])
    review = selection.get('manual_classification', 'pending_review')
    if review not in REVIEWS:
        raise ValueError(f'Unknown manual classification: {review}')
    result = {'event_id': event_id, 'dataset_id': signal_set.id,
              'run_id': selection.get('run_id', signal_set.id), 'cycle_id': selection.get('cycle_id'),
              'stage_id': selection.get('stage_id'), 'selection': dict(selection),
              'sources': signal_set.metadata.get('sources', []), 'units': signal_set.units,
              'temperature_source': signal_set.metadata.get('temperature_source', 'unavailable'),
              'endo_exo_convention': signal_set.metadata.get('endo_exo_convention', 'unknown'),
              'method': 'time-domain OLS baseline / initial-flank intersection',
              'parameters': asdict(config), 'smoothing': {'seconds': config.smooth_s,
                  'scope': 'centered time-window mean for automatic flank selection only; fits use original HF'},
              'manual_classification': review, 'automatic_classification': 'pending_review',
              'phase_hypothesis': selection.get('phase_hypothesis'),
              'phase_assignment_status': 'not_validated', 'status': 'not_determined',
              'reason': None, 'onset': None, 'diagnostics': {},
              'operational_criteria_status': 'pending experimental validation'}
    try:
        window = bounds(selection['window_s'], 'window_s')
        original = signal_set.frame
        if selection.get('row_range') is not None:
            start_row, end_row = selection['row_range']
            if (int(start_row) != start_row or int(end_row) != end_row or
                    start_row < 0 or end_row < start_row or end_row >= len(original)):
                raise ValueError('row_range must be inclusive valid acquisition-row indices.')
            original = original.iloc[int(start_row):int(end_row)+1]
        frame = original.loc[mask_range(original.time_s.to_numpy(float), window)].copy()
        t = frame.time_s.to_numpy(float)
        y = frame.heat_flow.to_numpy(float)
        if len(t) < 2 * config.min_fit_points:
            raise ValueError('Event window has too few points.')
        if not np.isfinite(t).all() or not np.isfinite(y).all():
            raise ValueError('Nonfinite time or HF inside event window; refine selection.')
        if np.any(np.diff(t) <= 0):
            raise ValueError('Non-increasing acquisition time; select a monotonic interval or explicitly reconstruct TRIOS time.')
        limit = gap_limit(t, config)
        if 'source_discontinuity' in frame and frame.source_discontinuity.iloc[1:].astype(bool).any():
            raise ValueError('Event window spans omitted source rows or a recorded-clock break; refine selection.')
        if np.any(np.diff(t) > limit):
            raise ValueError('Event window crosses a data gap; refine selection.')
        baseline = selection.get('baseline_s')
        if baseline is None:
            baseline = [float(t[0]), float(t[0] + config.baseline_fraction * (t[-1] - t[0]))]
        baseline = bounds(baseline, 'baseline_s')
        if baseline[0] < t[0] or baseline[1] > t[-1]:
            raise ValueError('Baseline must lie inside the event window.')
        bm = mask_range(t, baseline)
        base, base_quality = _fit(t, y, bm, config)
        result['fit'] = {'baseline_s': list(baseline), 'baseline_coefficients': base}
        residual = y - np.polyval(base, t)
        noise = residual[bm]
        sigma = max(float(np.std(noise, ddof=1)),
                    float(1.4826 * np.median(abs(noise - np.median(noise)))), np.finfo(float).eps)
        after = t > baseline[1]
        if not after.any():
            raise ValueError('No signal after baseline.')
        candidate = after  # diagnostics stay within this selected candidate window
        deviation = abs(residual[candidate])
        snr = float(np.max(deviation) / sigma) if len(deviation) else 0.
        points, duration = _persistence(t, candidate & (abs(residual) > config.noise_multiplier * sigma))
        bm_indices = np.flatnonzero(bm)
        mid = len(bm_indices) // 2
        slopes = []
        if mid >= config.min_fit_points:
            for indices in (bm_indices[:mid], bm_indices[mid:]):
                mask = np.zeros(len(t), dtype=bool)
                mask[indices] = True
                coeff, _ = _fit(t, y, mask, config)
                slopes.append(coeff[0])
        osc = oscillation_indicator(t[candidate], residual[candidate], config)
        result['diagnostics'] = {'baseline_noise_sigma': sigma, 'snr': snr,
                                 'persistence_points': points, 'persistence_s': duration,
                                 'baseline_fit': base_quality,
                                 'baseline_half_slopes': slopes,
                                 'baseline_slope_difference': abs(slopes[1]-slopes[0]) if slopes else None,
                                 'oscillation': osc}
        proposed = snr >= config.noise_multiplier and duration >= config.min_persistence_s
        result['automatic_classification'] = ('possible_oscillation' if osc['possible_oscillation'] else
                                              'event_candidate' if proposed else 'below_operational_criteria')
        polarity = selection.get('polarity', 'auto')
        if polarity not in ('auto', 'positive', 'negative'):
            raise ValueError('polarity must be auto, positive or negative; endo/exo is separate metadata.')
        flank = selection.get('flank_s')
        if flank is not None:
            flank = bounds(flank, 'flank_s')
            if flank[0] <= baseline[1] or flank[1] > t[-1]:
                raise ValueError('Flank must follow baseline and lie inside the event window.')
            fm = mask_range(t, flank)
            sign = (1 if polarity == "positive" else -1) if polarity != "auto" else (1 if np.mean(residual[fm]) >= 0 else -1)
        else:
            # First persistent threshold departure after baseline; never the global peak.
            smooth = residual.copy()
            if config.smooth_s:
                smooth = pd.Series(residual, index=pd.to_timedelta(t, unit='s')).rolling(
                    pd.Timedelta(seconds=config.smooth_s), center=True, min_periods=1).mean().to_numpy()
            directions = [1, -1] if polarity == 'auto' else [1 if polarity == 'positive' else -1]
            choices = []
            for sign in directions:
                active = after & (sign * smooth > config.noise_multiplier * sigma)
                starts = np.flatnonzero(active & ~np.r_[False, active[:-1]])
                for start in starts:
                    end = start
                    while end + 1 < len(t) and active[end + 1]:
                        end += 1
                    if t[end] - t[start] >= config.min_persistence_s:
                        choices.append((int(start), sign, int(end)))
                        break
            if not choices:
                raise ValueError('No persistent departure for automatic flank; select baseline_s and flank_s manually.')
            start, sign, end = min(choices)
            derivative = sign * np.gradient(smooth, t)
            # Stop at the first local maximum of signed signal after departure.
            peaks = _local_peaks(sign * smooth)
            following = peaks[peaks > start]
            stop = min(end, int(following[0])) if len(following) else end
            # Include the initial rise before the threshold crossing.
            low = max(int(np.searchsorted(t, baseline[1], side='right')),
                      int(np.searchsorted(t, t[start] - 2 * config.tangent_half_width_s)))
            center = low + int(np.argmax(derivative[low:stop+1]))
            fm = (t >= t[center] - config.tangent_half_width_s) & (t <= t[center] + config.tangent_half_width_s) & after
            flank = (float(t[fm][0]), float(t[fm][-1]))
        tangent, flank_quality = _fit(t, y, fm, config)
        result['fit'] = {'baseline_s': list(baseline), 'flank_s': list(flank),
                         'baseline_coefficients': base, 'flank_coefficients': tangent,
                         'selection': {'baseline': 'manual' if selection.get('baseline_s') is not None else 'initial window fraction',
                                       'flank': 'manual' if selection.get('flank_s') is not None else 'first persistent departure; steepest initial rise before first local maximum'}}
        result['diagnostics']['flank_fit'] = flank_quality
        onset = _intersection(base, tangent)
        if not baseline[1] <= onset <= flank[1] or not t[0] <= onset <= t[-1]:
            raise ValueError('Intersection lies outside the baseline-to-initial-flank interval; review fit regions.')
        if tangent[0] * sign <= base[0] * sign:
            raise ValueError('Flank does not depart from baseline in the selected direction.')
        onset_values = {name: sample_signal(frame, name, onset, config)
                        for name in ['temperature_c', 'sample_temperature_c', 'furnace_temperature_c',
                                     'huber_temperature_c', 'pressure']}
        sensitivity = []
        for indices in (bm_indices[:-1], bm_indices[1:]):
            if len(indices) >= config.min_fit_points:
                mask = np.zeros(len(t), dtype=bool)
                mask[indices] = True
                alternative, _ = _fit(t, y, mask, config)
                try:
                    sensitivity.append(_intersection(alternative, tangent))
                except ValueError:
                    pass
        flank_indices = np.flatnonzero(fm)
        for indices in (flank_indices[:-1], flank_indices[1:]):
            if len(indices) >= config.min_fit_points:
                mask = np.zeros(len(t), dtype=bool)
                mask[indices] = True
                alternative, _ = _fit(t, y, mask, config)
                try:
                    sensitivity.append(_intersection(base, alternative))
                except ValueError:
                    pass
        result['diagnostics']['onset_endpoint_trim_sensitivity_s'] = max(abs(np.array(sensitivity)-onset)) if sensitivity else None
        result['diagnostics']['sensitivity_definition'] = 'Trim one endpoint from each fit region in turn; not a confidence interval.'
        result['onset'] = {'time_s': onset, 'heat_flow': float(np.polyval(base, onset)), **onset_values}
        peak = np.flatnonzero(candidate)[np.argmax(sign * residual[candidate])]
        result['auxiliary_peak'] = {'time_s': float(t[peak]), 'heat_flow': float(y[peak]),
                                    'definition': 'largest signed baseline residual after baseline in this window; not onset'}
        result.update(status='calculated')
    except (ValueError, KeyError, IndexError, np.linalg.LinAlgError) as exc:
        result.update(status='not_determined', reason=str(exc))
    return result


def propose_windows(signal_set, config=None):
    """Propose unrestricted candidate windows. Does not accept or reject events."""
    config = config or AnalysisConfig()
    config.validate()
    frame = signal_set.frame
    t, y = frame.time_s.to_numpy(float), frame.heat_flow.to_numpy(float)
    if len(t) < 2 * config.min_fit_points or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        return {'candidates': [], 'reason': 'Select a finite monotonic time interval before automatic proposals.'}
    if not np.isfinite(y).all() or np.any(np.diff(t) > gap_limit(t, config)):
        return {'candidates': [], 'reason': 'Missing HF or data gaps; select intervals manually.'}
    # A rolling median removes slow baseline drift; this is proposal-only processing.
    width = max(config.candidate_separation_s * 4, config.tangent_half_width_s * 10)
    index = pd.to_timedelta(t, unit='s')
    trend = pd.Series(y, index=index).rolling(pd.Timedelta(seconds=width), center=True, min_periods=1).median().to_numpy()
    residual = y - trend
    sigma = max(float(1.4826 * np.median(abs(residual - np.median(residual)))), np.finfo(float).eps)
    dt = float(np.median(np.diff(t)))
    distance = max(1, int(config.candidate_separation_s / dt))
    magnitude = abs(residual)
    eligible = []
    for peak in _local_peaks(magnitude):
        low = max(0, peak - distance)
        high = min(len(t), peak + distance + 1)
        prominence = magnitude[peak] - max(float(magnitude[low:peak+1].min()),
                                           float(magnitude[peak:high].min()))
        if prominence >= config.noise_multiplier * sigma:
            eligible.append(int(peak))
    kept = []
    for peak in sorted(eligible, key=lambda i: magnitude[i], reverse=True):
        if all(abs(t[peak] - t[other]) >= config.candidate_separation_s for other in kept):
            kept.append(peak)
    peaks = sorted(kept)
    candidates = []
    for number, i in enumerate(peaks, 1):
        left = max(float(t[0]), float(t[i] - width))
        right = min(float(t[-1]), float(t[i] + width / 2))
        if right <= left:
            continue
        candidates.append({'event_id': f'candidate-{number:03d}', 'dataset_id': signal_set.id,
                           'window_s': [left, right], 'baseline_s': None, 'flank_s': None,
                           'manual_classification': 'pending_review', 'proposal': {
                               'method': 'absolute rolling-median residual prominence',
                               'candidate_time_s': float(t[i]), 'noise_sigma': sigma,
                               'parameters': asdict(config), 'status': 'review required'}})
    return {'candidates': candidates, 'reason': None,
            'limitations': 'Broad weak events can be missed; drift and oscillations can be proposed. Manual windows remain available.'}


def analyze_selections(sets, selections, config=None):
    ids = [str(s['event_id']) for s in selections]
    if len(ids) != len(set(ids)):
        raise ValueError('Event identifiers must be unique.')
    by_id = {s.id: s for s in sets}
    results = []
    for selection in selections:
        item = by_id.get(selection.get('dataset_id'))
        if item is None:
            raise ValueError(f"Unknown dataset_id: {selection.get('dataset_id')}")
        result = analyze_window(item, selection, config)
        # Validate the selected stage independently of geometric success.
        if selection.get('stage_id') is not None:
            stage = next((s for s in item.stages if str(s['id']) == str(selection['stage_id'])), None)
            result['stage'] = stage
            if stage is None:
                result.update(status='not_determined', reason='Unknown selected stage', onset=None)
            elif selection.get('window_s') is not None and (selection['window_s'][0] < stage['start_s'] or selection['window_s'][1] > stage['end_s']):
                result.update(status='not_determined', reason='Event window lies outside selected stage', onset=None)
            elif result.get('cycle_id') is None:
                result['cycle_id'] = stage.get('cycle_id')
        results.append(result)
    return results
