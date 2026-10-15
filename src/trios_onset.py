"""Time-domain onset detection for TRIOS cooling ramps."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .baseline import robust_polyfit
from .trios_parser import TriosSegment


@dataclass(slots=True)
class TriosOnset:
    event_number: int
    onset_elapsed_min: float
    onset_run_time_min: float
    onset_temperature_c: float
    onset_heat_flow_w_g: float
    uncertainty_time_min: float
    uncertainty_temperature_c: float
    peak_elapsed_min: float
    peak_temperature_c: float
    peak_heat_flow_w_g: float
    baseline_window_min: tuple[float, float]
    tangent_window_min: tuple[float, float]
    baseline_coefficients: tuple[float, float]
    tangent_coefficients: tuple[float, float]

    def as_dict(self) -> dict:
        values = asdict(self)
        for key in ["baseline_coefficients", "tangent_coefficients"]:
            slope, intercept = values.pop(key)
            values[f"{key}_slope"] = slope
            values[f"{key}_intercept"] = intercept
        return values


def _uniform_elapsed_time(time_min: np.ndarray) -> np.ndarray:
    """Restore acquisition order lost by the 0.01 min Excel display precision."""

    duration = float(time_min[-1] - time_min[0])
    return np.linspace(0.0, duration, len(time_min))


def first_exothermic_onset(segment: TriosSegment) -> TriosOnset | None:
    """Find the first positive heat-flow event using a time-domain tangent."""

    frame = segment.frame
    if len(frame) < 100:
        return None
    run_time = frame["time_min"].to_numpy(float)
    elapsed = _uniform_elapsed_time(run_time)
    temperature = frame["temperature_c"].to_numpy(float)
    heat_flow = frame["heat_flow_w_g"].to_numpy(float)
    duration = float(elapsed[-1])
    dt = float(np.median(np.diff(elapsed)))

    initial_mask = (elapsed >= 0.05 * duration) & (elapsed <= 0.40 * duration)
    initial = heat_flow[initial_mask]
    baseline_median = float(np.median(initial))
    mad_sigma = float(1.4826 * np.median(np.abs(initial - baseline_median)))
    dynamic_range = float(np.quantile(heat_flow, 0.999) - baseline_median)
    threshold = baseline_median + max(20.0 * mad_sigma, 0.01 * dynamic_range)

    smooth_window = max(5, int(round(0.02 / max(dt, np.finfo(float).eps))))
    if smooth_window % 2 == 0:
        smooth_window += 1
    smoothed = np.convolve(heat_flow, np.ones(smooth_window) / smooth_window, mode="same")
    search_mask = (elapsed > 0.05 * duration) & (elapsed < 0.95 * duration)
    crossings = np.flatnonzero(search_mask & (smoothed > threshold))
    if len(crossings) == 0:
        return None
    crossing = int(crossings[0])

    derivative = np.gradient(smoothed, elapsed)
    search_low = int(np.searchsorted(elapsed, elapsed[crossing] - 0.10))
    search_high = int(np.searchsorted(elapsed, elapsed[crossing] + 0.10))
    event_index = search_low + int(np.argmax(derivative[search_low:search_high]))

    baseline_low = float(elapsed[crossing] - 3.0)
    baseline_high = float(elapsed[crossing] - 0.30)
    baseline_mask = (elapsed >= baseline_low) & (elapsed <= baseline_high)
    if baseline_mask.sum() < 20:
        return None
    baseline_coeffs = robust_polyfit(elapsed[baseline_mask], heat_flow[baseline_mask], degree=1)

    tangent_half_window = 7
    tangent_low_index = max(0, event_index - tangent_half_window)
    tangent_high_index = min(len(elapsed), event_index + tangent_half_window + 1)
    tangent_coeffs = robust_polyfit(
        elapsed[tangent_low_index:tangent_high_index],
        heat_flow[tangent_low_index:tangent_high_index],
        degree=1,
    )
    denominator = float(baseline_coeffs[0] - tangent_coeffs[0])
    if abs(denominator) < 1e-12:
        return None
    onset_elapsed = float((tangent_coeffs[1] - baseline_coeffs[1]) / denominator)
    if not (baseline_high <= onset_elapsed <= elapsed[event_index] + 0.10):
        return None

    onset_temperature = float(np.interp(onset_elapsed, elapsed, temperature))
    onset_heat_flow = float(np.polyval(baseline_coeffs, onset_elapsed))
    peak_mask = (elapsed >= elapsed[crossing]) & (elapsed <= elapsed[crossing] + 2.0)
    peak_indices = np.flatnonzero(peak_mask)
    peak_index = int(peak_indices[np.argmax(heat_flow[peak_indices])])

    # The exported time column is rounded to 0.01 min.  Window sensitivity is
    # smaller than that, so the source precision controls the reported error.
    time_uncertainty = 0.01
    local_rate = np.gradient(temperature, elapsed)
    rate = float(np.nanmedian(np.abs(local_rate[baseline_mask])))
    temperature_uncertainty = max(0.01, rate * time_uncertainty)
    return TriosOnset(
        event_number=1,
        onset_elapsed_min=onset_elapsed,
        onset_run_time_min=float(run_time[0] + onset_elapsed),
        onset_temperature_c=onset_temperature,
        onset_heat_flow_w_g=onset_heat_flow,
        uncertainty_time_min=time_uncertainty,
        uncertainty_temperature_c=temperature_uncertainty,
        peak_elapsed_min=float(elapsed[peak_index]),
        peak_temperature_c=float(temperature[peak_index]),
        peak_heat_flow_w_g=float(heat_flow[peak_index]),
        baseline_window_min=(baseline_low, baseline_high),
        tangent_window_min=(
            float(elapsed[tangent_low_index]),
            float(elapsed[tangent_high_index - 1]),
        ),
        baseline_coefficients=(float(baseline_coeffs[0]), float(baseline_coeffs[1])),
        tangent_coefficients=(float(tangent_coeffs[0]), float(tangent_coeffs[1])),
    )
