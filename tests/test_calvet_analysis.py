import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from process_dsc_data import compute_onset_tangent, detect_peak_start, save_calvet_event_bundle
from plot_calvet_results import render_saved_results
from util.calvet_analysis import analyze_calvet_events


def _fixed_tangent(time, heat_flow, **_kwargs):
    return {
        "ok": True,
        "x_onset": 1.5,
        "y_onset": 4.5,
        "x_peak": 2.5,
        "y_peak": 8.0,
        "x_onset2": 1.25,
        "y_onset2": 4.0,
        "onset2_ok": True,
        "baseline_window": (0.0, 1.0),
        "flank_window": (1.0, 2.0),
        "baseline_window_onset": (0.0, 1.0),
        "flank_window_onset": (1.0, 2.0),
        "baseline_window_onset2": (0.0, 1.0),
        "flank_window_onset2": (1.0, 2.0),
        "a_base": 1.0,
        "b_base": 2.0,
        "a_flank": 3.0,
        "b_flank": 2.0,
    }


def _frame():
    return pd.DataFrame(
        {
            "t_h": np.arange(5, dtype=float),
            "t_h_original": np.arange(5, dtype=float) + 10.0,
            "t_s_datetime": np.arange(5, dtype=float) * 3600.0,
            "t_s_sample_index": np.arange(5, dtype=float) * 1800.0,
            "P_muestra": [100.0, 101.0, 102.0, 103.0, 104.0],
            "T_muestra": [20.0, 21.0, 22.0, 23.0, 24.0],
            "T_horno": [19.0, 20.0, 21.0, 22.0, 23.0],
            "HF_mW": [1.0, 3.0, 8.0, 3.0, 1.0],
        }
    )


def test_onset_calculation_matches_legacy_numeric_output():
    x = np.linspace(0.0, 10.0, 1001)
    y = 0.02 * x + 1.5 * np.exp(-0.5 * ((x - 7.0) / 0.35) ** 2)
    legacy = compute_onset_tangent(
        x,
        y,
        event_polarity="auto",
        params={"flank_force_near_peak": True, "flank_near_peak_frac": 0.15},
    )
    frame = pd.DataFrame(
        {
            "t_h": x,
            "HF_mW": y,
            "T_muestra": 20 + x,
            "P_muestra": 100 + x,
        }
    )
    result = analyze_calvet_events(
        frame,
        run_id="R01",
        method="tangent",
        tangent_calculator=compute_onset_tangent,
        threshold_calculator=detect_peak_start,
    )[0]

    assert result["status"] == "calculated"
    assert result["onset_time_h"] == pytest.approx(legacy["x_onset"], abs=1e-12)
    assert result["peak_time_h"] == pytest.approx(legacy["x_peak"], abs=1e-12)
    assert result["onset2_definition"].startswith("Intersection of the fitted baseline")


def test_event_windows_have_independent_ids_and_interpolate_simultaneous_values():
    frame = _frame()
    results = analyze_calvet_events(
        frame,
        run_id="R01",
        load_id="L01",
        stage_id="3",
        method="tangent",
        tangent_calculator=_fixed_tangent,
        threshold_calculator=detect_peak_start,
        event_windows_h=[(0.0, 4.0), (0.0, 1.0)],
    )

    assert len(results) == 2
    assert results[0]["event_id"] == "R01:3:event-01"
    assert results[1]["event_id"] == "R01:3:event-02"
    assert results[0]["onset_temperature_c"] == pytest.approx(21.5)
    assert results[0]["onset_temperature_method"] == "interpolated"
    assert results[0]["onset_pressure_bar"] == pytest.approx(101.5)
    assert results[0]["onset_pressure_method"] == "interpolated"
    assert results[0]["onset_original_time_h"] == pytest.approx(11.5)
    assert results[0]["onset_device_clock_elapsed_s"] == pytest.approx(5400.0)
    assert results[1]["status"] == "not_determined"
    assert "at least 5" in results[1]["reason"]


def test_pressure_is_not_interpolated_across_large_data_gaps():
    frame = _frame()
    frame["t_h"] = [0.0, 1.0, 100.0, 101.0, 102.0]
    frame["t_h_original"] = frame["t_h"]
    frame["HF_mW"] = [1.0, 3.0, 8.0, 3.0, 1.0]
    result = analyze_calvet_events(
        frame,
        run_id="R02",
        method="tangent",
        tangent_calculator=lambda *_args, **_kwargs: {
            **_fixed_tangent(None, None),
            "x_onset": 50.0,
            "x_peak": 100.0,
        },
        threshold_calculator=detect_peak_start,
    )[0]

    assert result["onset_pressure_bar"] is None
    assert result["onset_pressure_method"] == "unavailable"
    assert any("data gap" in warning for warning in result["warnings"])


def test_saved_calvet_plot_renders_without_calculating_onsets(tmp_path: Path):
    frame = _frame()
    event = analyze_calvet_events(
        frame,
        run_id="R03",
        method="tangent",
        tangent_calculator=_fixed_tangent,
        threshold_calculator=detect_peak_start,
    )
    results_path = save_calvet_event_bundle(
        frame,
        event,
        tmp_path / "bundle",
        data_path=Path("source.lvm"),
        program_path=Path("program.pdf"),
        zone_info={"num": 3, "type": "Cooling"},
        method="tangent",
    )

    payload = json.loads(results_path.read_text(encoding="utf-8"))
    assert payload["time_basis"]["program_time_offset_applied_s"] == 0.0
    assert payload["time_basis"]["thermal_lag_correction_applied"] is False
    output = render_saved_results(results_path, mode="hf-temperature", out=tmp_path / "saved.png")

    assert output.is_file()
    assert output.stat().st_size > 0
