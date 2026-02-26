from pathlib import Path

import numpy as np

from process_dsc_data import (
    _dedupe_repeated_suffix,
    _resolve_input_path,
    compute_onset_tangent,
    detect_zones_from_temperature,
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
