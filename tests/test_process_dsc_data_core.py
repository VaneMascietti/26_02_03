from pathlib import Path

import numpy as np
import pandas as pd

from process_dsc_data import (
    _dedupe_repeated_suffix,
    _extra_onsets_from_temperature_targets,
    _resolve_input_path,
    compute_onset_tangent,
    detect_zones_from_temperature,
    load_lvm,
    load_program_zones,
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


def test_load_program_zones_infers_header_when_page_prefix_appears(tmp_path, monkeypatch):
    pdf_path = tmp_path / "program.pdf"
    pdf_path.write_text("", encoding="utf-8")

    extracted = "\n".join(
        [
            "6 : Isoterma",
            "Total duration: 21600 s",
            "Page 37 : Cooling",
            "Total duration: 66000 s",
            "8 : Isoterma",
            "Total duration: 21600 s",
        ]
    )

    class _FakePage:
        def extract_text(self):
            return extracted

    class _FakeReader:
        def __init__(self, _):
            self.pages = [_FakePage()]

    class _FakePyPDF2:
        PdfReader = _FakeReader

    import sys

    monkeypatch.setitem(sys.modules, "PyPDF2", _FakePyPDF2)
    zones = load_program_zones(pdf_path)

    nums = [z["num"] for z in zones]
    assert nums == [6, 7, 8]
    z7 = next(z for z in zones if z["num"] == 7)
    assert z7["type"].lower().startswith("cooling")
    assert z7["duration_s"] == 66000


def test_load_program_zones_accepts_decimal_duration(tmp_path, monkeypatch):
    pdf_path = tmp_path / "program_decimal.pdf"
    pdf_path.write_text("", encoding="utf-8")

    extracted = "\n".join(
        [
            "1 : Inicio",
            "Total duration: 2666.7 s",
            "2 : Isoterma",
            "Total duration: 21600 s",
        ]
    )

    class _FakePage:
        def extract_text(self):
            return extracted

    class _FakeReader:
        def __init__(self, _):
            self.pages = [_FakePage()]

    class _FakePyPDF2:
        PdfReader = _FakeReader

    import sys

    monkeypatch.setitem(sys.modules, "PyPDF2", _FakePyPDF2)
    zones = load_program_zones(pdf_path)

    assert zones[0]["num"] == 1
    assert zones[0]["type"].lower().startswith("inicio")
    assert zones[0]["duration_s"] == 2666.7


def test_load_lvm_ignores_blank_leading_column_with_nul_footer(tmp_path):
    lvm_path = tmp_path / "sample.lvm"
    lvm_path.write_bytes(
        (
            b"\t300\t34.66\t307.4\t28030.5\t59.388103\t59.999531\t21.513342\t0.584247\t0.0\t2/27/2026 3:26:14 PM\n"
            b"\t300\t34.66\t307.3\t4.0\t19.914864\t19.997200\t6410.243652\t178.607569\t0.0\t2/27/2026 3:26:15 PM\n"
            b"\t300\t34.66\t307.2\t5.0\t19.914770\t19.997231\t6387.470215\t177.973052\t0.0\t2/27/2026 3:26:16 PM\n"
            b"\x00\x00\x00\t\t\t\t\t\t\t\t\t\n"
        )
    )

    df = load_lvm(lvm_path)

    assert len(df) == 2
    assert np.allclose(df["P_bomba_bar"].to_numpy(), [300.0, 300.0])
    assert np.allclose(df["P_muestra"].to_numpy(), [307.3, 307.2])
    assert np.allclose(df["T_muestra"].to_numpy(), [19.914864, 19.914770])
    assert np.allclose(df["HF_mW"].to_numpy(), [178.607569, 177.973052])


def test_extra_onsets_from_temperature_targets_maps_to_nearest_points():
    clean = pd.DataFrame(
        {
            "t_h": np.array([0.0, 1.0, 2.0, 3.0], dtype=float),
            "T_muestra": np.array([40.6, 16.1, 11.0, 5.0], dtype=float),
            "P_muestra": np.array([300.0, 280.0, 272.0, 260.0], dtype=float),
            "HF_mW": np.array([5.0, 6.0, 10.0, 2.0], dtype=float),
        }
    )
    extras = _extra_onsets_from_temperature_targets(
        clean,
        [40.64, 16.09, 10.98],
        excluded_indices=[1],
    )

    # index 1 is excluded, so only 40.6 and 11.0 remain
    assert len(extras) == 2
    assert extras[0]["idx"] == 0
    assert extras[1]["idx"] == 2
    assert abs(extras[0]["target_T_C"] - 40.64) < 1e-9
    assert abs(extras[1]["target_T_C"] - 10.98) < 1e-9
