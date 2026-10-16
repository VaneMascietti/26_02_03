"""Analysis helpers for Calvet HP data; kept separate from plotting."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd


def _sample_at_time(
    time_h: np.ndarray,
    values: np.ndarray,
    target_h: float,
) -> tuple[float | None, str, str | None]:
    valid = np.isfinite(time_h) & np.isfinite(values)
    time = time_h[valid]
    signal = values[valid]
    if not len(time):
        return None, "unavailable", "No finite measurements for this signal."

    order = np.argsort(time)
    time = time[order]
    signal = signal[order]
    exact = np.flatnonzero(np.isclose(time, target_h, rtol=0.0, atol=1e-10))
    if len(exact):
        return float(signal[exact[0]]), "measured", None

    right = int(np.searchsorted(time, target_h, side="right"))
    left = right - 1
    if left < 0 or right >= len(time):
        return None, "unavailable", "Onset lies outside the recorded signal; no extrapolation used."

    positive_steps = np.diff(time)
    positive_steps = positive_steps[positive_steps > 0]
    if not len(positive_steps):
        return None, "unavailable", "No positive time intervals are available for interpolation."
    max_gap = 5.0 * float(np.median(positive_steps))
    gap = float(time[right] - time[left])
    if gap <= 0 or gap > max_gap:
        return None, "unavailable", f"Onset is bracketed by a data gap ({gap:.6g} h); no interpolation used."

    fraction = (target_h - time[left]) / gap
    value = signal[left] + fraction * (signal[right] - signal[left])
    return float(value), "interpolated", None


def analyze_calvet_events(
    frame: pd.DataFrame,
    *,
    run_id: str,
    load_id: str | None = None,
    cycle_id: str | None = None,
    stage_id: str | None = None,
    method: str = "tangent",
    tangent_calculator: Callable[..., dict[str, Any]],
    threshold_calculator: Callable[..., dict[str, Any] | None],
    event_windows_h: Sequence[tuple[float, float]] | None = None,
    center_h: float | None = None,
    window_h: float = 2.0,
    flank_near_peak_frac: float = 0.15,
) -> list[dict[str, Any]]:
    """Calculate one or more Calvet heat-flow events without creating figures.

    Event windows use the frame's current ``t_h`` axis (zone-relative when the
    caller selected a zone). The returned original time is taken from
    ``t_h_original`` when available.
    """
    if method not in {"tangent", "simple"}:
        raise ValueError(f"Unsupported onset method: {method!r}")
    if "t_h" in frame:
        time_h = pd.to_numeric(frame["t_h"], errors="coerce").to_numpy(float)
    elif "t_s" in frame:
        time_h = pd.to_numeric(frame["t_s"], errors="coerce").to_numpy(float) / 3600.0
    else:
        raise ValueError("Calvet frame must contain t_h or t_s.")
    original_time_h = (
        pd.to_numeric(frame["t_h_original"], errors="coerce").to_numpy(float)
        if "t_h_original" in frame
        else time_h
    )
    heat_flow = pd.to_numeric(frame["HF_mW"], errors="coerce").to_numpy(float)
    if len(time_h) != len(heat_flow):
        raise ValueError("Time and HF signals must have the same length.")

    if event_windows_h:
        windows = [tuple(sorted((float(low), float(high)))) for low, high in event_windows_h]
    else:
        windows = [(float(np.nanmin(time_h)), float(np.nanmax(time_h)))] if len(time_h) else []

    results: list[dict[str, Any]] = []
    for event_number, window in enumerate(windows, 1):
        mask = (
            np.isfinite(time_h)
            & np.isfinite(heat_flow)
            & (time_h >= window[0])
            & (time_h <= window[1])
        )
        event_time = time_h[mask]
        event_hf = heat_flow[mask]
        base: dict[str, Any] = {
            "load_id": load_id,
            "run_id": run_id,
            "cycle_id": cycle_id,
            "stage_id": stage_id,
            "event_id": f"{run_id}:{stage_id or 'all'}:event-{event_number:02d}",
            "event_number": event_number,
            "event_window_start_h": window[0],
            "event_window_end_h": window[1],
            "time_unit": "h",
            "heat_flow_signal": "HF_mW",
            "heat_flow_unit": "mW",
            "temperature_signal": "T_muestra",
            "temperature_unit": "degC",
            "pressure_signal": "P_muestra",
            "pressure_unit": "bar",
            "method": method,
            "interpretation_hypothesis": None,
            "manual_review": "pending",
        }
        if len(event_time) < 5:
            results.append(
                {
                    **base,
                    "status": "not_determined",
                    "reason": f"Only {len(event_time)} valid points in event window; at least 5 required.",
                }
            )
            continue

        if method == "tangent":
            calculation = tangent_calculator(
                event_time,
                event_hf,
                event_polarity="auto",
                params={
                    "flank_force_near_peak": True,
                    "flank_near_peak_frac": float(flank_near_peak_frac),
                },
            )
            onset_h = calculation.get("x_onset", np.nan)
            peak_h = calculation.get("x_peak", np.nan)
            onset_hf = calculation.get("y_onset", np.nan)
            peak_hf = calculation.get("y_peak", np.nan)
            onset2_h = calculation.get("x_onset2", np.nan)
            onset2_hf = calculation.get("y_onset2", np.nan)
            onset2_ok = bool(calculation.get("onset2_ok", False))
            baseline_window = calculation.get("baseline_window_onset", calculation.get("baseline_window"))
            flank_window = calculation.get("flank_window_onset", calculation.get("flank_window"))
            success = bool(calculation.get("ok")) and np.isfinite(onset_h)
            reason = calculation.get("reason") or ",".join(calculation.get("reasons", [])) or None
            render_info = {
                "t_char_50": calculation.get("t_char_50", np.nan),
                "a_base": calculation.get("a_base", np.nan),
                "b_base": calculation.get("b_base", np.nan),
                "a_flank": calculation.get("a_flank", np.nan),
                "b_flank": calculation.get("b_flank", np.nan),
                "baseline_w0_h": (calculation.get("baseline_window") or (np.nan, np.nan))[0],
                "baseline_w1_h": (calculation.get("baseline_window") or (np.nan, np.nan))[1],
                "flank_w0_h": (calculation.get("flank_window") or (np.nan, np.nan))[0],
                "flank_w1_h": (calculation.get("flank_window") or (np.nan, np.nan))[1],
                "baseline_window_onset_h": baseline_window or (np.nan, np.nan),
                "flank_window_onset_h": flank_window or (np.nan, np.nan),
                "baseline_window_onset2_h": calculation.get("baseline_window_onset2", baseline_window)
                or (np.nan, np.nan),
                "flank_window_onset2_h": calculation.get("flank_window_onset2", flank_window)
                or (np.nan, np.nan),
            }
        else:
            calculation = threshold_calculator(
                event_time,
                event_hf,
                center_h=center_h,
                window_h=window_h,
            )
            success = calculation is not None
            if calculation is None:
                calculation = {}
            onset_h = calculation.get("t_start_h", np.nan)
            peak_h = calculation.get("t_peak_h", np.nan)
            onset_hf = calculation.get("hf_start", np.nan)
            peak_hf = calculation.get("hf_peak", np.nan)
            onset2_h = np.nan
            onset2_hf = np.nan
            onset2_ok = False
            baseline_window = None
            flank_window = None
            reason = None if success else "No threshold crossing met the configured criteria."
            render_info = {
                "t_char_50": np.nan,
                "a_base": np.nan,
                "b_base": np.nan,
                "a_flank": np.nan,
                "b_flank": np.nan,
                "baseline_w0_h": np.nan,
                "baseline_w1_h": np.nan,
                "flank_w0_h": np.nan,
                "flank_w1_h": np.nan,
                "baseline_window_onset_h": (np.nan, np.nan),
                "flank_window_onset_h": (np.nan, np.nan),
                "baseline_window_onset2_h": (np.nan, np.nan),
                "flank_window_onset2_h": (np.nan, np.nan),
            }

        absolute_onset_h = None
        if np.isfinite(onset_h):
            absolute_onset_h, _, _ = _sample_at_time(time_h, original_time_h, float(onset_h))

        temp_values = (
            pd.to_numeric(frame["T_muestra"], errors="coerce").to_numpy(float)
            if "T_muestra" in frame
            else np.full(len(frame), np.nan)
        )
        pressure_values = (
            pd.to_numeric(frame["P_muestra"], errors="coerce").to_numpy(float)
            if "P_muestra" in frame
            else np.full(len(frame), np.nan)
        )
        temperature, temperature_method, temperature_warning = _sample_at_time(
            time_h, temp_values, float(onset_h)
        ) if np.isfinite(onset_h) else (None, "unavailable", "Onset was not determined.")
        pressure, pressure_method, pressure_warning = _sample_at_time(
            time_h, pressure_values, float(onset_h)
        ) if np.isfinite(onset_h) else (None, "unavailable", "Onset was not determined.")
        warnings = [warning for warning in (temperature_warning, pressure_warning) if warning]
        if "T_horno" in frame:
            warnings.append("T_horno is available but was not substituted for T_muestra.")
        peak_temperature, peak_temperature_method, peak_temperature_warning = (
            _sample_at_time(time_h, temp_values, float(peak_h))
            if np.isfinite(peak_h)
            else (None, "unavailable", "Peak was not determined.")
        )
        if peak_temperature_warning:
            warnings.append(peak_temperature_warning)
        onset_datetime_elapsed_s = None
        if np.isfinite(onset_h) and "t_s_datetime" in frame:
            elapsed = pd.to_numeric(frame["t_s_datetime"], errors="coerce").to_numpy(float)
            onset_datetime_elapsed_s, _, _ = _sample_at_time(time_h, elapsed, float(onset_h))
        peak_datetime_elapsed_s = None
        if np.isfinite(peak_h) and "t_s_datetime" in frame:
            elapsed = pd.to_numeric(frame["t_s_datetime"], errors="coerce").to_numpy(float)
            peak_datetime_elapsed_s, _, _ = _sample_at_time(time_h, elapsed, float(peak_h))
        render_info.update(
            {
                "t_start_h": float(onset_h) if np.isfinite(onset_h) else np.nan,
                "hf_start": float(onset_hf) if np.isfinite(onset_hf) else np.nan,
                "t_peak_h": float(peak_h) if np.isfinite(peak_h) else np.nan,
                "hf_peak": float(peak_hf) if np.isfinite(peak_hf) else np.nan,
                "t_start2_h": float(onset2_h) if np.isfinite(onset2_h) else np.nan,
                "hf_start2": float(onset2_hf) if np.isfinite(onset2_hf) else np.nan,
                "onset2_ok": onset2_ok,
                "method": method,
            }
        )

        results.append(
            {
                **base,
                "status": "calculated" if success else "not_determined",
                "reason": reason,
                "onset_time_h": float(onset_h) if np.isfinite(onset_h) else None,
                "onset_original_time_h": absolute_onset_h,
                "onset_device_clock_elapsed_s": onset_datetime_elapsed_s,
                "onset_hf_mw": float(onset_hf) if np.isfinite(onset_hf) else None,
                "onset_temperature_c": temperature,
                "onset_temperature_method": temperature_method,
                "onset_pressure_bar": pressure,
                "onset_pressure_method": pressure_method,
                "peak_time_h": float(peak_h) if np.isfinite(peak_h) else None,
                "peak_device_clock_elapsed_s": peak_datetime_elapsed_s,
                "peak_hf_mw": float(peak_hf) if np.isfinite(peak_hf) else None,
                "peak_temperature_c": peak_temperature,
                "peak_temperature_method": peak_temperature_method,
                "observation": "Candidate extremum in the HF signal; not a phase assignment.",
                "onset2_time_h": float(onset2_h) if np.isfinite(onset2_h) else None,
                "onset2_hf_mw": float(onset2_hf) if np.isfinite(onset2_hf) else None,
                "onset2_definition": (
                    "Intersection of the fitted baseline and linear regression over the pre-peak flank; "
                    "legacy secondary geometric estimate, not an independent physical onset."
                    if method == "tangent"
                    else None
                ),
                "baseline_window_h": list(baseline_window) if baseline_window else None,
                "flank_window_h": list(flank_window) if flank_window else None,
                "parameters": {
                    "flank_near_peak_frac": float(flank_near_peak_frac),
                    "threshold_center_h": center_h,
                    "threshold_window_h": window_h,
                },
                "warnings": warnings,
                "legacy_plot_info": render_info,
            }
        )
    return results
