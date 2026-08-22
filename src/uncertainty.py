"""Simple uncertainty estimates for automatically determined onsets."""

from __future__ import annotations

import numpy as np


def onset_uncertainty(
    temperature_c: np.ndarray,
    onset_temp_c: float | None,
    noise_sigma: float,
    local_slope_mw_per_c: float | None = None,
) -> tuple[float | None, float | None, float | None]:
    if onset_temp_c is None:
        return None, None, None
    temperature = np.asarray(temperature_c, dtype=float)
    resolution = float(np.median(np.abs(np.diff(temperature)))) if len(temperature) > 1 else 0.0
    slope_term = abs(noise_sigma / local_slope_mw_per_c) if local_slope_mw_per_c else 0.0
    standard = max(resolution, slope_term, 0.02)
    return standard, onset_temp_c - 1.96 * standard, onset_temp_c + 1.96 * standard

