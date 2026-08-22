"""Noise diagnostics and model comparison."""

from __future__ import annotations

import numpy as np

from .models import NoiseMetrics


def _max_run(mask: np.ndarray) -> int:
    best = current = 0
    for value in mask:
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _correlation_length(residuals: np.ndarray) -> int:
    residuals = residuals - np.mean(residuals)
    denom = float(np.dot(residuals, residuals))
    if denom <= 0:
        return 1
    for lag in range(1, min(200, len(residuals) // 2)):
        corr = float(np.dot(residuals[:-lag], residuals[lag:]) / denom)
        if corr < np.exp(-1):
            return lag
    return max(1, min(200, len(residuals) // 2))


def _bic(rss: float, n: int, k: int) -> float:
    rss = max(float(rss), np.finfo(float).eps)
    return n * np.log(rss / max(n, 1)) + k * np.log(max(n, 1))


def block_bootstrap_pvalue(
    baseline_residuals: np.ndarray,
    observed_max: float,
    block_size: int,
    iterations: int = 200,
    seed: int = 123,
) -> float:
    if len(baseline_residuals) < max(10, block_size * 2) or observed_max <= 0:
        return 1.0
    rng = np.random.default_rng(seed)
    residuals = np.asarray(baseline_residuals, dtype=float)
    starts = np.arange(0, len(residuals) - block_size + 1)
    exceed = 0
    for _ in range(iterations):
        chunks = []
        while sum(len(chunk) for chunk in chunks) < len(residuals):
            start = int(rng.choice(starts))
            chunks.append(residuals[start : start + block_size])
        sample = np.concatenate(chunks)[: len(residuals)]
        if np.max(np.abs(sample - np.mean(sample))) >= observed_max:
            exceed += 1
    return (exceed + 1) / (iterations + 1)


def analyse_noise(
    residuals: np.ndarray,
    baseline_mask: np.ndarray,
    event_mask: np.ndarray,
) -> NoiseMetrics:
    residuals = np.asarray(residuals, dtype=float)
    baseline = residuals[baseline_mask]
    event = residuals[event_mask]
    if len(baseline) < 3:
        baseline = residuals
    sigma = float(np.std(baseline, ddof=1)) if len(baseline) > 1 else 0.0
    rms = float(np.sqrt(np.mean(np.square(baseline)))) if len(baseline) else 0.0
    mad = float(np.median(np.abs(baseline - np.median(baseline)))) if len(baseline) else 0.0
    mad_sigma = 1.4826 * mad
    scale = max(mad_sigma, sigma, np.finfo(float).eps)
    max_abs = float(np.max(np.abs(event))) if len(event) else 0.0
    snr = max_abs / scale
    integrated_snr = float(np.abs(np.sum(event)) / (scale * np.sqrt(max(len(event), 1))))
    consecutive = _max_run(np.abs(residuals) > 2 * scale)
    corr_len = _correlation_length(baseline) if len(baseline) > 5 else 1

    rss_base = float(np.sum(np.square(residuals)))
    segmented = residuals.copy()
    if len(event):
        segmented[event_mask] -= float(np.mean(event))
    rss_segmented = float(np.sum(np.square(segmented)))
    bic_base = _bic(rss_base, len(residuals), 2)
    bic_segmented = _bic(rss_segmented, len(residuals), 4)
    p_value = block_bootstrap_pvalue(baseline, max_abs, max(3, corr_len))

    return NoiseMetrics(
        sigma=sigma,
        rms=rms,
        mad_sigma=mad_sigma,
        max_abs_deviation=max_abs,
        snr=float(snr),
        integrated_snr=integrated_snr,
        consecutive_points=consecutive,
        correlation_length=corr_len,
        bic_baseline=float(bic_base),
        bic_segmented=float(bic_segmented),
        delta_bic=float(bic_base - bic_segmented),
        bootstrap_p=float(p_value),
    )

