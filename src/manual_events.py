"""Independent Calisto event windows, using its existing robust line-fitting convention."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .baseline import robust_polyfit
from .models import EventResult


def analyze_manual_event(segment: pd.DataFrame, spec: dict) -> EventResult:
    result = EventResult(event_label=str(spec["event_id"]), status="undetermined", confidence="not quantified",
                         onset_temp_c=None, onset_uncertainty_c=None, ci95_low_c=None, ci95_high_c=None,
                         peak_temp_c=None, snr=np.nan, integrated_snr=np.nan, automatic_label="calorimetric candidate",
                         method="Calisto manual baseline/flank intersection", parameters={"robust_degree": 1},
                         requested_window_s=tuple(spec["interval_s"]), baseline_window_s=tuple(spec["baseline_s"]),
                         flank_window_s=tuple(spec["flank_s"]))
    lo, hi = spec["interval_s"]
    if not lo < hi:
        raise ValueError("Calisto interval_s requires start < end")
    part = segment.loc[segment.time_s.between(lo, hi)]
    def fail(reason):
        result.notes = reason
        return result
    if len(part) < 8:
        return fail("not_enough_points")
    steps = np.diff(part.time_s.to_numpy(float))
    positive = steps[steps > 0]
    if not len(positive) or np.any(steps < 0) or np.any(steps > 5*np.median(positive)):
        return fail("record_interruption; split event window")
    if lo < part.time_s.min()-np.median(positive) or hi > part.time_s.max()+np.median(positive):
        return fail("event_window_outside_selected_record")
    if not np.isfinite(part[["time_s", "temperature_c", "heat_flow_mw"]].to_numpy(float)).all():
        return fail("missing_required_measurement")
    fits = []
    windows_c = []
    for name in ("baseline_s", "flank_s"):
        bounds = spec[name]
        if len(bounds) != 2 or not lo <= bounds[0] < bounds[1] <= hi:
            raise ValueError(f"{name} must be inside interval_s")
        selected = part.loc[part.time_s.between(*bounds)]
        if len(selected) < 4 or selected.temperature_c.nunique() < 2:
            return fail(f"insufficient_temperature_variation_in_{name}")
        fits.append(robust_polyfit(selected.temperature_c.to_numpy(float), selected.heat_flow_mw.to_numpy(float), degree=1))
        windows_c.append((float(selected.temperature_c.min()), float(selected.temperature_c.max())))
    baseline, flank = fits
    denominator = float(baseline[0]-flank[0])
    if abs(denominator) < 1e-12:
        return fail("parallel_fits")
    onset = float((flank[1]-baseline[1])/denominator)
    if not part.temperature_c.min() <= onset <= part.temperature_c.max():
        return fail("onset_outside_event_window")
    residual = part.heat_flow_mw.to_numpy(float)-np.polyval(baseline, part.temperature_c)
    peak = int(np.argmax(np.abs(residual)))
    result.status = "determined"
    result.onset_temp_c = onset
    result.peak_temp_c = float(part.temperature_c.iloc[peak])
    result.peak_time_s = float(part.time_s.iloc[peak])
    result.baseline_window_c, result.tangent_window_c = windows_c
    result.fit_coefficients = {"baseline": baseline.tolist(), "flank": flank.tolist()}
    result.notes = "Explicit independent windows; peak is largest baseline deviation. Detection confidence and phase assignment require review."
    return result
