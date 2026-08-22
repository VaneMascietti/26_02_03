from pathlib import Path

import numpy as np
import pandas as pd

from cli import analyze
from src.event_detection import detect_events
from src.pdf_parser import parse_program_pdf
from src.txt_parser import read_txt


def test_parse_real_calisto_pdf_zones():
    program = parse_program_pdf("data/23_05_12.pdf")
    assert len(program.zones) >= 44
    assert len(program.cooling_zones) == 11
    first_cooling = program.cooling_zones[0]
    assert first_cooling.init_temp_c == 50
    assert first_cooling.final_temp_c == 10
    assert first_cooling.scan_rate_k_min == 0.3


def test_read_real_utf16_txt_with_trailing_delimiters():
    data = read_txt("data/23_05_12.txt")
    assert len(data.frame) > 500_000
    assert data.temperature_source == "furnace"
    assert data.frame["time_s"].iloc[0] == 0
    assert data.frame["heat_flow_mw"].notna().all()


def test_noise_only_signal_is_not_forced_to_event():
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(
        {
            "time_s": np.arange(2000),
            "temperature_c": np.linspace(50, 10, 2000),
            "heat_flow_mw": 0.01 * np.linspace(50, 10, 2000) + rng.normal(0, 0.001, 2000),
        }
    )
    results, metrics, _ = detect_events(frame, "cooling")
    assert metrics is not None
    assert results[0].status == "not detected"


def test_synthetic_event_is_detected_without_reference_temperature():
    rng = np.random.default_rng(123)
    temp = np.linspace(50, 10, 3000)
    hf = 0.002 * temp + rng.normal(0, 0.0008, len(temp))
    hf += -0.025 * np.exp(-0.5 * ((temp - 15.8) / 0.08) ** 2)
    frame = pd.DataFrame({"time_s": np.arange(len(temp)), "temperature_c": temp, "heat_flow_mw": hf})
    results, metrics, _ = detect_events(frame, "cooling")
    assert metrics is not None
    assert any(result.status in {"detected", "doubtful"} for result in results)
    assert any(result.peak_temp_c is not None and abs(result.peak_temp_c - 15.8) < 0.5 for result in results)


def test_cli_end_to_end_one_cycle(tmp_path: Path):
    output = analyze("data/23_05_12.pdf", "data/23_05_12.txt", tmp_path, max_cycles=1)
    results = output["results"]
    assert len(results) == 1
    assert (tmp_path / "results_by_cycle.csv").exists()
    assert (tmp_path / "thermal_program.csv").exists()
    assert (tmp_path / "analysis_config.json").exists()
