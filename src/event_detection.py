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
    # A low-temperature cooling scan can contain the LL and SL onsets in one
    # broad transition.  Keep this branch opt-in by temperature range so the
    # regular multi-event behavior used by the 50 -> 10 C scans is unchanged.
    "low_temperature_pair_max_c": 35.0,
    "low_temperature_ll_tangent_low_delta_c": 0.36,
    "low_temperature_ll_tangent_high_delta_c": 0.66,
    "low_temperature_ll_baseline_low_delta_c": 0.86,
    "low_temperature_ll_baseline_high_delta_c": 1.36,
    "low_temperature_sl_tangent_half_window": 36,
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


def _ll_tangent_onset(
    temperature: np.ndarray,
    heat_flow: np.ndarray,
    baseline_mask: np.ndarray,
    event_index: int,
    config: dict[str, float],
) -> tuple[float, tuple[float, float], tuple[float, float], float] | None:
    """Estimate LL from a local flank and a stable high-temperature baseline."""

    event_temperature = float(temperature[event_index])
    tangent_low = event_temperature + float(config["low_temperature_ll_tangent_low_delta_c"])
    tangent_high = event_temperature + float(config["low_temperature_ll_tangent_high_delta_c"])
    tangent_mask = (
        np.isfinite(temperature)
        & np.isfinite(heat_flow)
        & (temperature >= tangent_low)
        & (temperature <= tangent_high)
    )
    baseline_low = event_temperature + float(config["low_temperature_ll_baseline_low_delta_c"])
    baseline_high = event_temperature + float(config["low_temperature_ll_baseline_high_delta_c"])
    ll_baseline_mask = (
        np.isfinite(temperature)
        & np.isfinite(heat_flow)
        & (temperature >= baseline_low)
        & (temperature <= baseline_high)
    )
    if ll_baseline_mask.sum() < 8:
        ll_baseline_mask = baseline_mask
    if tangent_mask.sum() < 8 or ll_baseline_mask.sum() < 8:
        return None

    baseline_coeffs = robust_polyfit(
        temperature[ll_baseline_mask],
        heat_flow[ll_baseline_mask],
        degree=1,
    )
    tangent_coeffs = robust_polyfit(temperature[tangent_mask], heat_flow[tangent_mask], degree=1)
    denominator = float(baseline_coeffs[0] - tangent_coeffs[0])
    if abs(denominator) < 1e-12:
        return None
    onset = float((tangent_coeffs[1] - baseline_coeffs[1]) / denominator)
    data_low = float(np.nanmin(temperature))
    data_high = float(np.nanmax(temperature))
    if not (data_low - 2 <= onset <= data_high + 2):
        return None

    baseline_window = (
        float(np.nanmin(temperature[ll_baseline_mask])),
        float(np.nanmax(temperature[ll_baseline_mask])),
    )
    tangent_window = (
        float(np.nanmin(temperature[tangent_mask])),
        float(np.nanmax(temperature[tangent_mask])),
    )
    return onset, baseline_window, tangent_window, float(tangent_coeffs[0])


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

    low_temperature_pair = (
        direction == "cooling"
        and float(np.nanmax(temperature)) <= float(cfg["low_temperature_pair_max_c"])
    )
    if low_temperature_pair:
        # Select the candidate with the strongest segmented-vs-baseline
        # evidence.  Other gradients in the post-transition baseline (for
        # example near 13 C in 23_05_09) are not separate transitions.
        positive_bic = [item for item in scored if item[2].delta_bic > 0]
        primary = max(positive_bic or scored, key=lambda item: (item[2].delta_bic, item[0]))
        selected = [primary]
    else:
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
        tangent_half_window = (
            int(cfg["low_temperature_sl_tangent_half_window"])
            if low_temperature_pair
            else 12
        )
        onset, base_window, tangent_window = tangent_onset(
            temperature,
            heat_flow,
            local_baseline,
            idx,
            tangent_half_window=tangent_half_window,
        )
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

    if low_temperature_pair and results:
        broad = _ll_tangent_onset(
            temperature,
            heat_flow,
            selected[0][3],
            selected[0][1],
            cfg,
        )
        if broad is not None:
            onset, baseline_window, tangent_window, tangent_slope = broad
            unc, ci_low, ci_high = onset_uncertainty(
                temperature,
                onset,
                selected[0][2].mad_sigma,
                tangent_slope,
            )
            tll = EventResult(
                event_label="TLL",
                status="doubtful",
                confidence="medium",
                onset_temp_c=onset,
                onset_uncertainty_c=unc,
                ci95_low_c=ci_low,
                ci95_high_c=ci_high,
                peak_temp_c=float(temperature[selected[0][1]]),
                snr=selected[0][2].snr,
                integrated_snr=selected[0][2].integrated_snr,
                automatic_label="TLL",
                baseline_window_c=baseline_window,
                tangent_window_c=tangent_window,
                notes=(
                    "LL estimated from a tangent fitted to the local flank and a "
                    "stable high-temperature baseline; manual confirmation is required."
                ),
            )
            results.insert(0, tll)

    if (
        not low_temperature_pair
        and len(results) == 2
        and abs(results[0].peak_temp_c - results[1].peak_temp_c) < 0.25
    ):
        for result in results:
            result.status = "doubtful"
            result.notes = "Overlapping events are not resolved."
    return results, metrics, residuals
