import numpy as np
import pandas as pd
import pytest

from src.event_detection import detect_events
from src.txt_parser import read_txt


def test_txt_units_and_temperature_sources_are_explicit(tmp_path):
    source = tmp_path / "sample.txt"
    source.write_text("Time(min);Sample Temperature(K);Furnace Temperature(C);HeatFlow(uW);\n"
                      "0;293.15;21;1000;\n1;294.15;22;2000;\n")
    data = read_txt(source)
    assert data.frame.time_s.tolist() == [0, 60]
    assert data.frame.heat_flow_mw.tolist() == [1, 2]
    np.testing.assert_allclose(data.frame.temperature_c, [20, 21])
    assert data.frame.furnace_temperature_c.tolist() == [21, 22]
    assert data.temperature_source == "sample"
    assert data.frame.time_sync_s.tolist() == data.frame.time_original_s.tolist()


def test_unsupported_normalized_heat_flow_is_not_labeled_mw(tmp_path):
    source = tmp_path / "sample.txt"
    source.write_text("Time(s);Sample Temperature(C);HeatFlow(W/g)\n0;20;1\n")
    with pytest.raises(ValueError, match="Unsupported heat flow unit"):
        read_txt(source)


def test_thousands_separator_on_integer_time_rows_does_not_create_clock_resets(tmp_path):
    source = tmp_path / "sample.txt"
    source.write_text("Time(s);Sample Temperature(C);HeatFlow(mW)\n"
                      "1,001.7;20;1\n1,002;20;1\n1,002.3;20;1\n")
    data = read_txt(source)
    np.testing.assert_allclose(data.frame.time_s, [1001.7, 1002, 1002.3])
    assert data.frame.time_s.is_monotonic_increasing
    assert any("thousands grouping" in warning for warning in data.warnings)


def test_calisto_phase_heuristic_is_opt_in_and_numerically_identical():
    temp = np.linspace(50, 10, 3000)
    hf = .002*temp - .025*np.exp(-.5*((temp-15.8)/.08)**2)
    frame = pd.DataFrame({"time_s": np.arange(len(temp)), "temperature_c": temp, "heat_flow_mw": hf})
    generic, _, _ = detect_events(frame)
    heuristic, _, _ = detect_events(frame, config={"phase_heuristic": True})
    assert all(result.phase_hypothesis is None for result in generic)
    assert all(result.event_label.startswith("event") for result in generic)
    assert any(result.phase_hypothesis for result in heuristic)
    assert [r.onset_temp_c for r in generic] == [r.onset_temp_c for r in heuristic]


def test_calisto_discontinuous_record_is_undetermined():
    frame = pd.DataFrame({"time_s": np.r_[np.arange(100), np.arange(500, 600)],
                          "temperature_c": np.linspace(50, 10, 200), "heat_flow_mw": np.ones(200)})
    results, _, _ = detect_events(frame)
    assert results[0].status == "undetermined"
    assert "record_interruption" in results[0].notes


def test_calisto_multiple_manual_events_analyze_without_figures_and_render_saved(tmp_path, monkeypatch):
    import json
    import cli
    from src.models import DSCData, ProgramSummary, ThermalZone
    from src.manual_events import analyze_manual_event
    seconds = np.arange(2001, dtype=float)
    temp = 60-.02*seconds
    hf = .002*seconds+10*np.exp(-.5*((seconds-800)/35)**2)+20*np.exp(-.5*((seconds-1600)/35)**2)
    frame = pd.DataFrame({"time_s": seconds, "temperature_c": temp, "heat_flow_mw": hf,
                          "source_row": np.arange(len(temp))})
    specs = [{"event_id": f"A{i}", "stage_id": "zone1", "interval_s": [peak-300, peak+150],
              "baseline_s": [peak-280, peak-200], "flank_s": [peak-60, peak-5]}
             for i, peak in enumerate((800, 1600))]
    results = [analyze_manual_event(frame, spec) for spec in specs]
    assert all(r.status == "determined" for r in results)
    assert results[0].onset_temp_c > results[1].onset_temp_c
    assert all(r.phase_hypothesis is None for r in results)
    pdf, txt = tmp_path / "program.pdf", tmp_path / "signal.txt"
    pdf.write_text("test")
    txt.write_text("test")
    program = ProgramSummary(pdf, [ThermalZone(1, "Cooling", "Cooling", "cooling", 60, 20, 2000, -1.2, 1, 2001,
                                               start_s=0, end_s=2000, cycle=1)])
    data = DSCData(txt, frame, [], "time_s", "temperature_c", "heat_flow_mw", "sample")
    monkeypatch.setattr(cli, "parse_program_pdf", lambda _: program)
    monkeypatch.setattr(cli, "read_txt", lambda _: data)
    original_plot = cli.plot_cycle
    def forbidden(*args, **kwargs):
        raise AssertionError("Unexpected analysis/plot call")
    monkeypatch.setattr(cli, "detect_events", forbidden)
    monkeypatch.setattr(cli, "plot_cycle", forbidden)
    output = cli.analyze(pdf, txt, tmp_path / "analysis", config={"event_specs": specs}, make_plots=False)
    saved = json.loads(output["saved_results"].read_text())
    assert len(saved["segments"][0]["events"]) == 2
    assert saved["segments"][0]["observed_conditions"]["measured_rate_c_min"] == pytest.approx(-1.2)
    assert saved["programmed_zones"][0]["start_s"] == 0
    assert saved["segments"][0]["pressure"]["value"] is None
    assert (tmp_path / "analysis" / "events.csv").is_file()
    monkeypatch.setattr(cli, "read_txt", forbidden)
    monkeypatch.setattr(cli, "plot_cycle", original_plot)
    monkeypatch.chdir(tmp_path)
    out = cli.render_saved(output["saved_results"])
    assert (out / "zone_01.png").is_file()
