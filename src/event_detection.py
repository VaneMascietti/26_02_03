"""Candidate event detection for cooling thermograms."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .baseline import automatic_baseline_mask, evaluate, robust_polyfit
from .models import EventResult, NoiseMetrics
from .noise_analysis import analyse_noise
from .onset import tangent_onset
from .uncertainty import onset_uncertainty


DEFAULT_CONFIG = {
    "snr_detected": 3.0,
    "snr_doubtful": 2.0,
    "min_persistence_points": 8,
    "min_delta_bic": 10.0,
    "event_window_points": 80,
}


def _smooth(y: np.ndarray, window: int = 31) -> np.ndarray:
    if len(y) < 5:
        return y
    window = min(window, max(5, len(y) // 20))
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")


def _local_masks(n: int, index: int, half: int) -> tuple[np.ndarray, np.ndarray]:
    baseline = np.zeros(n, dtype=bool)
    event = np.zeros(n, dtype=bool)
    baseline[max(0, index - 10 * half) : max(0, index - 2 * half)] = True
    event[max(0, index - half) : min(n, index + half + 1)] = True
    if baseline.sum() < max(12, half):
        baseline[: max(8, min(index, 4 * half))] = True
    return baseline, event


def _candidate_indices(heat_flow: np.ndarray, half: int) -> list[int]:
    smoothed = _smooth(heat_flow)
    gradient = np.abs(np.gradient(smoothed))
    guard = max(10, int(0.05 * len(heat_flow)))
    gradient[:guard] = 0
    gradient[-guard:] = 0
    candidates: list[int] = []
    work = gradient.copy()
    for _ in range(4):
        idx = int(np.argmax(work))
        if work[idx] <= 0:
            break
        if all(abs(idx - existing) > 4 * half for existing in candidates):
            candidates.append(idx)
        work[max(0, idx - 4 * half) : min(len(work), idx + 4 * half + 1)] = 0
    return sorted(candidates)


def _classify(metrics: NoiseMetrics, config: dict[str, float]) -> tuple[str, str]:
    detected = (
        metrics.snr >= config["snr_detected"]
        and metrics.consecutive_points >= config["min_persistence_points"]
        and metrics.delta_bic >= config["min_delta_bic"]
    )
    persistent = metrics.consecutive_points >= max(3, int(config["min_persistence_points"]) // 2)
    model_hint = metrics.delta_bic > 0 or metrics.bootstrap_p < 0.2
    doubtful = (
        (metrics.snr >= config["snr_doubtful"] or metrics.integrated_snr >= config["snr_doubtful"])
        and persistent
        and model_hint
    )
    if detected:
        return "detected", "high"
    if doubtful:
        return "doubtful", "medium"
    return "not detected", "low"


def detect_events(segment: pd.DataFrame, direction: str = "cooling", config: dict | None = None) -> tuple[list[EventResult], NoiseMetrics | None, np.ndarray | None]:
    """Detect up to two candidate events in one segment.

    The labels LL/SL are heuristic suggestions based on temporal order during cooling.
    """

    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)

    if len(segment) < 30:
        return [], None, None

    temperature = segment["temperature_c"].to_numpy(float)
    heat_flow = segment["heat_flow_mw"].to_numpy(float)
    half = int(cfg["event_window_points"])
    candidates = _candidate_indices(heat_flow, half)
    if not candidates:
        baseline_mask = automatic_baseline_mask(temperature, direction)
        coeffs = robust_polyfit(temperature[baseline_mask], heat_flow[baseline_mask], degree=1)
        residuals = heat_flow - evaluate(coeffs, temperature)
        metrics = analyse_noise(residuals, baseline_mask, np.zeros(len(segment), dtype=bool))
        candidates = [int(len(segment) / 2)]
    scored: list[tuple[float, int, NoiseMetrics, np.ndarray, np.ndarray]] = []
    for idx in candidates:
        candidate_baseline, candidate_event = _local_masks(len(segment), idx, half)
        candidate_coeffs = robust_polyfit(temperature[candidate_baseline], heat_flow[candidate_baseline], degree=1)
        candidate_residuals = heat_flow - evaluate(candidate_coeffs, temperature)
        candidate_metrics = analyse_noise(candidate_residuals, candidate_baseline, candidate_event)
        score = candidate_metrics.snr + 0.1 * candidate_metrics.integrated_snr + max(candidate_metrics.delta_bic, 0) / 25
        scored.append((float(score), idx, candidate_metrics, candidate_baseline, candidate_residuals))
    scored.sort(key=lambda item: item[0], reverse=True)
    metrics = scored[0][2]
    residuals = scored[0][4]
    status, confidence = _classify(metrics, cfg)

    if status == "not detected":
        return [
            EventResult(
                event_label="candidate",
                status=status,
                confidence=confidence,
                onset_temp_c=None,
                onset_uncertainty_c=None,
                ci95_low_c=None,
                ci95_high_c=None,
                peak_temp_c=None,
                snr=metrics.snr,
                integrated_snr=metrics.integrated_snr,
                automatic_label="no event",
                notes="Deviation did not exceed configurable noise/model criteria.",
            )
        ], metrics, residuals

    selected = sorted(scored[:2], key=lambda item: item[1])
    selected_indices = [item[1] for item in selected]
    labels = ["event"] * len(selected_indices)
    if direction == "cooling" and selected_indices:
        if len(selected_indices) == 1:
            labels = ["TSL"]
        else:
            # During cooling, the higher-temperature candidate is LL and the
            # lower-temperature candidate is SL, even when the transitions are
            # far apart.  The confidence/status fields still communicate when
            # the higher-temperature feature is only a doubtful candidate.
            labels = ["TLL", "TSL", *["event"] * max(0, len(selected_indices) - 2)]

    results: list[EventResult] = []
    for i, idx in enumerate(selected_indices):
        local_baseline, local_mask = _local_masks(len(segment), idx, half)
        local_coeffs = robust_polyfit(temperature[local_baseline], heat_flow[local_baseline], degree=1)
        local_residuals = heat_flow - evaluate(local_coeffs, temperature)
        local_metrics = analyse_noise(local_residuals, local_baseline, local_mask)
        local_status, local_confidence = _classify(local_metrics, cfg)
        onset, base_window, tangent_window = tangent_onset(temperature, heat_flow, local_baseline, idx)
        slope = None
        if tangent_window:
            win = (temperature >= min(tangent_window)) & (temperature <= max(tangent_window))
            if win.sum() > 3:
                slope = float(np.polyfit(temperature[win], heat_flow[win], 1)[0])
        unc, ci_low, ci_high = onset_uncertainty(temperature, onset, local_metrics.mad_sigma, slope)
        results.append(
            EventResult(
                event_label=labels[i] if i < len(labels) else "event",
                status=local_status,
                confidence=local_confidence,
                onset_temp_c=onset,
                onset_uncertainty_c=unc,
                ci95_low_c=ci_low,
                ci95_high_c=ci_high,
                peak_temp_c=float(temperature[idx]),
                snr=local_metrics.snr,
                integrated_snr=local_metrics.integrated_snr,
                automatic_label=labels[i] if i < len(labels) else "event",
                baseline_window_c=base_window,
                tangent_window_c=tangent_window,
                notes="Automatic label is a heuristic suggestion; manual confirmation is required.",
            )
        )

    if len(results) == 2 and abs(results[0].peak_temp_c - results[1].peak_temp_c) < 0.25:
        for result in results:
            result.status = "doubtful"
            result.notes = "Overlapping events are not resolved."
    return results, metrics, residuals
