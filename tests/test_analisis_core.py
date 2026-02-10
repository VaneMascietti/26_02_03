from pathlib import Path

import numpy as np

from analisis import (
    _dedupe_repeated_suffix,
    _resolve_input_path,
    compute_onset_tangent,
    detect_zones_from_temperature,
)


def test_dedupe_repeated_suffix():
    assert _dedupe_repeated_suffix(Path("exp.lvm.lvm")) == Path("exp.lvm")
    assert _dedupe_repeated_suffix(Path("exp.pdf")) == Path("exp.pdf")


def test_resolve_input_path_with_expected_suffix(tmp_path):
    lvm = tmp_path / "sample.lvm"
    lvm.write_text("ok", encoding="utf-8")
    resolved = _resolve_input_path(tmp_path / "sample", expected_suffix=".lvm")
    assert resolved == lvm


def test_detect_zones_from_temperature_returns_expected_profiles():
    t = np.arange(0.0, 10.0, 1.0)
    temp = np.array([0, 2, 4, 6, 6, 6, 5, 3, 1, 0], dtype=float)
    zones = detect_zones_from_temperature(
        t_h=t,
        y=temp,
        rate_thresh_c_per_h=0.5,
        smooth_window=3,
        min_duration_h=0.0,
    )
    types = [z["type"] for z in zones]
    assert "Heating" in types
    assert "Cooling" in types
    assert "Isoterma" in types


def test_compute_onset_tangent_on_synthetic_peak():
    x = np.linspace(0.0, 10.0, 1001)
    baseline = 0.02 * x
    peak = 1.5 * np.exp(-0.5 * ((x - 7.0) / 0.35) ** 2)
    y = baseline + peak

    result = compute_onset_tangent(x, y, event_polarity="endo")

    assert result["ok"] is True
    assert np.isfinite(result["x_onset"])
    assert np.isfinite(result["x_peak"])
    assert result["x_onset"] < result["x_peak"]
    assert np.isfinite(result["flank_window"][0])
    assert np.isfinite(result["flank_window"][1])
