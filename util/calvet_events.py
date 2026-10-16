"""Calvet event records. Calculations have no plotting or phase-assignment side effects."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .calvet_onset import TANGENT_DEFAULTS, compute_onset_tangent, detect_peak_start
import inspect
from .synchronization import value_at_time


def analyze_event(frame: pd.DataFrame, spec: dict, identifiers: dict, max_gap_s=5.0) -> dict:
    method = spec.get("method", "tangent")
    if method not in {"tangent", "simple"}:
        raise ValueError(f"Unknown onset method: {method}")
    window = spec.get("interval_h", [float(frame.t_h.min()), float(frame.t_h.max())])
    if len(window) != 2 or window[0] >= window[1]:
        raise ValueError("Event interval requires start < end")
    part = frame.loc[frame.t_h.between(*window)].copy()
    params = dict(spec.get("params", {}))
    if method == "tangent":
        params = {**TANGENT_DEFAULTS, **params}
    else:
        params = {**{name: p.default for name, p in inspect.signature(detect_peak_start).parameters.items()
                     if p.default is not inspect.Parameter.empty}, **params}
    for key, mode, value_key in (("baseline_h", "baseline_window_mode", "baseline_manual_range"),
                                 ("flank_h", "search_flank_mode", "flank_manual_range")):
        if spec.get(key) is not None:
            if method == "simple":
                raise ValueError("El umbral legacy no acepta ventanas de ajuste; use tangent o params de umbral.")
            bounds = spec[key]
            if len(bounds) != 2 or bounds[0] >= bounds[1] or bounds[0] < window[0] or bounds[1] > window[1]:
                raise ValueError(f"{key} must be ordered and contained in interval_h")
            params.update({mode: "manual", value_key: bounds})
    polarity = spec.get("polarity", "auto")
    if polarity not in {"auto", "endo", "exo"}:
        raise ValueError("polarity must be auto, endo or exo")
    result = {**identifiers, "stage_id": spec.get("stage_id", identifiers.get("stage_id")),
              "event_id": str(spec.get("event_id", "event01")), "status": "undetermined",
              "method": method, "parameters": params, "requested_window_h": window,
              "requested_spec": spec, "signals": {"time": "t_h", "temperature": "T_muestra", "heat_flow": "HF_mW", "pressure": "P_muestra"},
              "units": {"time": "h", "temperature": "degC", "heat_flow": "mW", "pressure": "bar"},
              "manual_review": spec.get("manual_review", {"status": "pending"}),
              "detection_confidence": "not quantified", "phase_assignment": {"hypothesis": None, "confidence": "unassigned"},
              "equilibrium_validated": False, "warnings": [], "calculation": None,
              "onset": None, "onset2": None, "peak": None,
              "onset2_definition": "alternative baseline/flank-regression intersection for the same event"}
    def fail(reason):
        result["reason"] = reason
        result["warnings"].append(reason)
        return result
    if len(part) < 5:
        return fail("not_enough_points")
    cadence_h = part.t_h.diff().loc[lambda values: values > 0].median()
    if window[0] < part.t_h.min() - cadence_h or window[1] > part.t_h.max() + cadence_h:
        return fail("event_window_outside_selected_record")
    # Do not conceal missing rows, discontinuities, reversed clocks, or disjoint stages.
    t = part.t_h.to_numpy(float)
    if not np.isfinite(t).all() or np.any(np.diff(t) < 0):
        return fail("nonmonotonic_time; select a continuous interval")
    if np.any(np.diff(t)*3600 > max_gap_s):
        return fail("record_interruption; split event window")
    if "t_original_s" in part:
        original_dt = np.diff(part.t_original_s.to_numpy(float))
        if np.any(~np.isfinite(original_dt)) or np.any(original_dt < 0) or np.any(original_dt > max_gap_s):
            return fail("record_interruption in original clock; split event window")
    if "source_row" in part and np.any(np.diff(part.source_row) > 1):
        return fail("disjoint_selection; split event window")
    if "HF_mW" not in part or not np.isfinite(part.HF_mW).all():
        return fail("missing_heat_flow; split event window")
    duplicates = int(part.t_h.duplicated().sum())
    if duplicates:
        part = part.drop_duplicates("t_h", keep="first")
        result["warnings"].append(f"{duplicates} duplicate timestamps: first row used for fitting; all rows retained in signals.csv")
    if len(part) < 5:
        return fail("not_enough_unique_times")
    t = part.t_h.to_numpy(float)
    if method == "tangent":
        calc = compute_onset_tangent(t, part.HF_mW.to_numpy(float), polarity, params)
        calc = {k: v for k, v in calc.items() if k != "y_s"}
        onset_h, peak_h = calc.get("x_onset"), calc.get("x_peak")
        good = calc.get("ok", False)
    else:
        calc = detect_peak_start(t, part.HF_mW.to_numpy(float), **params)
        onset_h, peak_h = (calc.get("t_start_h"), calc.get("t_peak_h")) if calc else (None, None)
        good = calc is not None
    result["calculation"] = calc
    if not good:
        return fail(calc.get("reason", "threshold_not_found") if calc else "threshold_not_found")
    def point(time_h):
        return {"time_h": time_h, "temperature_c": value_at_time(part, "T_muestra", time_h, max_gap_s),
                "pressure_bar": value_at_time(part, "P_muestra", time_h, max_gap_s)}
    result.update(status="determined", reason="", onset=point(onset_h), peak=point(peak_h))
    result["observation"] = {"feature": "positive maximum" if calc.get("polarity") == "endo" else
                             ("negative minimum" if calc.get("polarity") == "exo" else "extremum"),
                             "polarity_convention": "endo=positive, exo=negative is the legacy computational convention; instrument sign unverified"}
    if calc.get("onset2_ok"):
        result["onset2"] = point(calc["x_onset2"])
    for name in ("onset", "peak"):
        for signal in ("temperature_c", "pressure_bar"):
            if result[name][signal]["value"] is None:
                result["warnings"].append(f"{name}: {signal} unavailable")
    return result
