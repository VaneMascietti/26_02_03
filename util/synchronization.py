"""Time diagnostics and bounded interpolation. No automatic alignment or thermal lag correction."""
from __future__ import annotations

import numpy as np
import pandas as pd


def time_diagnostics(frame: pd.DataFrame, time_col: str = "t_sync_s") -> dict:
    t = frame[time_col].to_numpy(float)
    dt = np.diff(t)
    positive = dt[dt > 0]
    cadence = float(np.median(positive)) if len(positive) else None
    gaps = np.where(dt > 5 * cadence)[0] if cadence else np.array([], dtype=int)
    result = {"time_column": time_col, "positive_median_dt_s": cadence,
            "duplicate_times": int((dt == 0).sum()), "backward_steps": int((dt < 0).sum()),
            "gaps": [{"after_s": float(t[i]), "before_s": float(t[i+1])} for i in gaps],
            "recorded_vs_legacy_end_s": float(frame.t_original_s.iloc[-1] - frame.t_acquisition_s.iloc[-1])
            if len(frame) and "t_acquisition_s" in frame else None}
    if "t_original_s" in frame and time_col != "t_original_s":
        result["recorded_clock"] = time_diagnostics(frame, "t_original_s")
    return result


def value_at_time(frame: pd.DataFrame, column: str, time_h: float | None, max_gap_s: float = 5.0) -> dict:
    """Read a measured row or interpolate adjacent rows only, preserving missing values/gaps."""
    missing = {"value": None, "source": "unavailable", "reason": "missing signal or time"}
    if column not in frame or time_h is None or not np.isfinite(time_h):
        return missing
    t = frame.t_h.to_numpy(float)
    y = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
    if not len(t) or np.any(~np.isfinite(t)) or np.any(np.diff(t) < 0):
        return {**missing, "reason": "empty or nonmonotonic time"}
    i = int(np.searchsorted(t, time_h))
    if i < len(t) and np.isclose(t[i], time_h, rtol=0, atol=1e-12):
        return {"value": float(y[i]) if np.isfinite(y[i]) else None,
                "source": "measured" if np.isfinite(y[i]) else "unavailable", "sample_time_h": float(t[i])}
    if i == 0 or i == len(t):
        return {**missing, "reason": "outside record; no extrapolation"}
    width_s = (t[i] - t[i-1]) * 3600
    if "source_row" in frame and frame.source_row.iloc[i] - frame.source_row.iloc[i-1] > 1:
        return {**missing, "reason": "missing original row between interpolation endpoints"}
    if "t_original_s" in frame:
        recorded_width = frame.t_original_s.iloc[i] - frame.t_original_s.iloc[i-1]
        if not np.isfinite(recorded_width) or recorded_width < 0 or recorded_width > max_gap_s:
            return {**missing, "reason": "interruption in original recorded clock"}
    if width_s > max_gap_s or width_s <= 0 or not np.isfinite(y[i-1:i+1]).all():
        return {**missing, "reason": "gap or missing adjacent measurement"}
    value = y[i-1] + (y[i]-y[i-1]) * (time_h-t[i-1]) / (t[i]-t[i-1])
    return {"value": float(value), "source": "interpolated", "method": "linear adjacent rows",
            "bracket_h": [float(t[i-1]), float(t[i])], "max_gap_s": max_gap_s}
