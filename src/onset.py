"""Onset calculation by baseline-tangent intersection."""

from __future__ import annotations

import numpy as np

from .baseline import robust_polyfit


def tangent_onset(
    temperature_c: np.ndarray,
    heat_flow_mw: np.ndarray,
    baseline_mask: np.ndarray,
    event_index: int,
    tangent_half_window: int = 12,
) -> tuple[float | None, tuple[float, float] | None, tuple[float, float] | None]:
    """Return onset temperature and windows using baseline-tangent intersection."""

    temperature = np.asarray(temperature_c, dtype=float)
    heat_flow = np.asarray(heat_flow_mw, dtype=float)
    if len(temperature) < 8 or not (0 <= event_index < len(temperature)):
        return None, None, None

    base_coeffs = robust_polyfit(temperature[baseline_mask], heat_flow[baseline_mask], degree=1)
    grad = np.gradient(heat_flow)
    lo = max(0, event_index - 4 * tangent_half_window)
    hi = min(len(temperature), event_index + 4 * tangent_half_window + 1)
    if hi - lo < 5:
        return None, None, None
    steep_local = lo + int(np.argmax(np.abs(grad[lo:hi])))
    t0 = max(0, steep_local - tangent_half_window)
    t1 = min(len(temperature), steep_local + tangent_half_window + 1)
    if t1 - t0 < 4:
        return None, None, None

    tangent_coeffs = robust_polyfit(temperature[t0:t1], heat_flow[t0:t1], degree=1)
    denom = float(base_coeffs[0] - tangent_coeffs[0])
    if abs(denom) < 1e-12:
        return None, None, None
    onset = float((tangent_coeffs[1] - base_coeffs[1]) / denom)
    temp_min = float(np.nanmin(temperature))
    temp_max = float(np.nanmax(temperature))
    if not (temp_min - 2 <= onset <= temp_max + 2):
        return None, None, None
    base_window = (float(np.nanmin(temperature[baseline_mask])), float(np.nanmax(temperature[baseline_mask])))
    tangent_window = (float(np.nanmin(temperature[t0:t1])), float(np.nanmax(temperature[t0:t1])))
    return onset, base_window, tangent_window

