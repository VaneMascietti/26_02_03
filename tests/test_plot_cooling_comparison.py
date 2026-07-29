from pathlib import Path

import plot_cooling_comparison as comparison


def test_initial_heating_path_recognizes_aquecimento(monkeypatch):
    zones = [
        {"num": 1, "type": "isoterma", "start_h": 0.0, "end_h": 8.0},
        {"num": 2, "type": "aquecimento", "start_h": 8.0, "end_h": 21.0},
        {"num": 3, "type": "Isoterma", "start_h": 21.0, "end_h": 25.0},
    ]
    monkeypatch.setattr(comparison, "program_zone_spans", lambda _path: zones)

    assert comparison.initial_heating_path_span_h(Path("program.pdf")) == (0.0, 25.0)


def test_combined_pressure_temperature_time_plot_writes_a_figure(tmp_path: Path):
    data = {
        "path_time_h": [0.0, 1.0, 2.0],
        "P_muestra": [200.0, 210.0, 220.0],
        "T_muestra": [20.0, 30.0, 40.0],
        "segment": ["heating"] * 3,
        "label": ["E16 heating"] * 3,
    }
    outpath = tmp_path / "pressure_temperature_time.png"

    comparison.plot_pressure_temperature_vs_time(
        [comparison.pd.DataFrame(data)],
        pressure_col="P_muestra",
        temp_col="T_muestra",
        outpath=outpath,
        title="Test plot",
    )

    assert outpath.is_file()
    assert outpath.stat().st_size > 0
