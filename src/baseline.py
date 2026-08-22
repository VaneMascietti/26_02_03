"""Baseline fitting utilities."""

from __future__ import annotations

import numpy as np


def robust_polyfit(x: np.ndarray, y: np.ndarray, degree: int = 1, iterations: int = 4) -> np.ndarray:
    """Small iteratively reweighted polynomial fit using Tukey-like clipping."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() <= degree + 1:
        return np.polyfit(x[mask], y[mask], min(0, max(mask.sum() - 1, 0))) if mask.any() else np.array([0.0])

    weights = np.ones(mask.sum())
    xm = x[mask]
    ym = y[mask]
    for _ in range(iterations):
        coeffs = np.polyfit(xm, ym, degree, w=weights)
        residuals = ym - np.polyval(coeffs, xm)
        mad = np.median(np.abs(residuals - np.median(residuals)))
        scale = 1.4826 * mad if mad > 0 else np.std(residuals)
        if scale <= 0:
            break
        weights = np.where(np.abs(residuals) <= 4 * scale, 1.0, 0.25)
    return coeffs


def evaluate(coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.polyval(coeffs, np.asarray(x, dtype=float))


def automatic_baseline_mask(temperature_c: np.ndarray, direction: str) -> np.ndarray:
    """Choose a conservative pre-event baseline region in temporal order."""

    n = len(temperature_c)
    mask = np.zeros(n, dtype=bool)
    if n == 0:
        return mask
    end = max(8, int(n * 0.35))
    mask[: min(end, n)] = True
    return mask

