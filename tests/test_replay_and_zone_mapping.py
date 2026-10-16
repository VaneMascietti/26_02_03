from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.cycle_mapping import map_zones_to_data
from src.cycle_mapping import segment_for_assignment
from src.models import DSCData, EventResult, ProgramSummary, ThermalZone
from src.reporting import write_outputs
from src.replay import plot_saved_analysis
from src.txt_parser import read_txt


def test_map_zones_supports_manual_bounds_and_half_open_boundaries(tmp_path: Path):
    zones = [
        ThermalZone(1, "Cooling", "Cooling", "cooling", 50, 30, 2, 0.3, 1, 2, 0, 2, 1),
        ThermalZone(2, "Isotherm", "Isotherm", "isothermal", 30, 30, 2, 0, 1, 2, 2, 4, 1),
    ]
    program = ProgramSummary(tmp_path / "program.pdf", zones)
    frame = pd.DataFrame(
        {
            "time_s": [0.0, 1.0, 2.0, 3.0, 4.0],
            "source_row": [0, 1, 2, 3, 4],
        }
    )
    data = DSCData(
        tmp_path / "run.txt",
        frame,
        ["Time", "Sample Temperature", "Heat Flow"],
        "Time",
        "Sample Temperature",
        "Heat Flow",
        "sample",
    )

    automatic = map_zones_to_data(program, data)
    corrected = map_zones_to_data(program, data, {1: (1.0, 3.0)})

    assert (automatic[0].start_row, automatic[0].end_row) == (0, 1)
    assert (automatic[1].start_row, automatic[1].end_row) == (2, 4)
    assert (corrected[0].start_row, corrected[0].end_row) == (1, 2)
    assert corrected[0].selection_source == "manual"
    assert corrected[0].selected_start_s == 1.0


def test_non_monotonic_time_selection_keeps_only_matching_source_rows(tmp_path: Path):
    zone = ThermalZone(1, "Cooling", "Cooling", "cooling", 50, 30, 4, 0.3, 1, 4, 0, 4, 1)
    program = ProgramSummary(tmp_path / "program.pdf", [zone])
    frame = pd.DataFrame(
        {
            "time_s": [0.0, 1.0, 99.0, 2.0, 3.0],
            "source_row": [0, 1, 2, 3, 4],
        }
    )
    data = DSCData(
        tmp_path / "run.txt",
        frame,
        ["Time", "Sample Temperature", "Heat Flow"],
        "Time",
        "Sample Temperature",
        "Heat Flow",
        "sample",
    )

    assignment = map_zones_to_data(program, data)[0]
    segment = segment_for_assignment(data, assignment)

    assert segment["source_row"].tolist() == [0, 1, 3, 4]
    assert assignment.warning is not None
    assert "non-monotonic" in assignment.warning
    assert "disjoint source-row intervals" in assignment.warning
    assert assignment.as_dict()["selected_row_count"] == 4


def test_read_txt_retains_sample_and_furnace_temperature_separately(tmp_path: Path):
    source = tmp_path / "calisto.txt"
    source.write_text(
        "Time(s),Sample Temperature(C),Furnace Temperature(C),Heat Flow(mW)\n"
        "0,20,21,0.1\n"
        "1,19,20,0.2\n",
        encoding="utf-8",
    )

    data = read_txt(source)

    assert data.temperature_source == "sample"
    assert data.frame["temperature_c"].tolist() == [20.0, 19.0]
    assert data.frame["sample_temperature_c"].tolist() == [20.0, 19.0]
    assert data.frame["furnace_temperature_c"].tolist() == [21.0, 20.0]


def test_saved_calisto_results_replay_without_detection(tmp_path: Path, monkeypatch):
    outdir = tmp_path / "analysis"
    event = EventResult(
        event_label="event-01",
        status="detected",
        confidence="high",
        onset_temp_c=15.2,
        onset_uncertainty_c=0.1,
        ci95_low_c=15.0,
        ci95_high_c=15.4,
        peak_temp_c=14.8,
        snr=5.0,
        integrated_snr=10.0,
        automatic_label="event-01",
        baseline_window_c=(16.0, 18.0),
        tangent_window_c=(15.0, 15.5),
    ).as_dict()
    event.update({"cycle_id": 1, "stage_id": 4, "signal_file": "signals/cycle_01.csv"})
    signals = pd.DataFrame(
        {
            "source_row": [0, 1, 2],
            "temperature_c": [16.0, 15.0, 14.0],
            "heat_flow_mw": [0.0, 1.0, 0.0],
        }
    )
    write_outputs(
        outdir,
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        {},
        [],
        events=[event],
        signal_segments={"cycle_01.csv": signals},
    )
    observed = {}

    def capture(segment, results, output):
        observed["segment"] = segment
        observed["results"] = results
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"figure")
        return path

    monkeypatch.setattr("src.replay.plot_cycle", capture)
    rendered = plot_saved_analysis(outdir, cycles=[1])

    assert rendered and rendered[0].is_file()
    assert observed["results"][0].event_label == "event-01"
    assert observed["results"][0].onset_temp_c == pytest.approx(15.2)


def test_write_outputs_refuses_to_replace_existing_results(tmp_path: Path):
    options = (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}, [])
    write_outputs(tmp_path, *options)

    with pytest.raises(FileExistsError, match="not empty"):
        write_outputs(tmp_path, *options)
