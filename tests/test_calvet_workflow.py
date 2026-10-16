import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from util.calvet_events import analyze_event
from util.calvet_onset import compute_onset_tangent, detect_peak_start
from util.calvet_stages import describe_stages, select_stages
from util.experimental_metadata import experimental_metadata, write_json
from util.lvm_io import load_lvm
from util.synchronization import value_at_time


def signal_frame():
    seconds = np.arange(2001, dtype=float)
    t = seconds / 3600
    hf = .002 * seconds + 10*np.exp(-.5*((seconds-800)/35)**2) + 20*np.exp(-.5*((seconds-1600)/35)**2)
    return pd.DataFrame({"t_h": t, "t_sync_s": seconds, "t_original_s": seconds,
                         "source_row": np.arange(len(t)), "T_muestra": 60-.02*seconds,
                         "T_horno": 60-.021*seconds, "P_muestra": 500-.1*seconds, "HF_mW": hf})


def test_real_data_golden_preserves_legacy_tangent_and_threshold():
    expected = json.loads(Path("tests/fixtures/calvet_legacy_26_08_17.json").read_text())
    source = Path(expected["source"])
    if not source.exists():
        pytest.skip("Local instrument data is not distributed in git")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == expected["sha256"]
    frame = load_lvm(source, time_basis="legacy")
    part = frame.loc[frame.t_h.between(*expected["interval_h"])].iloc[::expected["stride"]]
    tangent = compute_onset_tangent(part.t_h, part.HF_mW)
    simple = detect_peak_start(part.t_h, part.HF_mW)
    for actual, golden in ((tangent, expected["tangent"]), (simple, expected["simple"])):
        for key, value in golden.items():
            if isinstance(value, (str, bool)):
                assert actual[key] == value
            else:
                np.testing.assert_allclose(actual[key], value, **expected["tolerance"])


def test_numerical_module_does_not_import_plotting():
    subprocess.run([sys.executable, "-c", "import sys; import util.calvet_events; assert 'matplotlib' not in sys.modules"], check=True)


def test_pressure_association_preserves_measured_interpolated_and_gaps():
    frame = pd.DataFrame({"t_h": np.array([0, 1, 2, 100])/3600, "P_muestra": [10, 12, np.nan, 50]})
    assert value_at_time(frame, "P_muestra", 1/3600)["source"] == "measured"
    interpolated = value_at_time(frame, "P_muestra", .5/3600)
    assert interpolated["source"] == "interpolated"
    assert interpolated["value"] == 11
    for when in (-1, 1.5, 50, 101):
        assert value_at_time(frame, "P_muestra", when/3600)["value"] is None
    assert value_at_time(frame, "missing", 0)["value"] is None


def test_stages_keep_programmed_and_measured_conditions_and_manual_limits():
    frame = signal_frame()
    spans = [{"num": 1, "type": "Cooling", "start_h": 0, "end_h": .3},
             {"num": 2, "type": "Isoterma", "start_h": .3, "end_h": .55}]
    stages = describe_stages(frame, spans)
    selected, summary = select_stages(frame, stages, kind="cooling", cycle=1, interval_h=[.1, .2])
    assert selected.t_h.min() >= .1 and selected.t_h.max() <= .2
    assert selected.t_original_s.iloc[0] == selected.t_h.iloc[0]*3600
    assert summary[0]["start_h"] == 0
    assert summary[0]["manual_limits_h"] == [.1, .2]
    assert summary[0]["measured_rate_c_min"] == pytest.approx(-1.2)
    assert stages[1]["programmed_kind"] == "isothermal"
    assert stages[1]["measured_kind"] == "cooling"
    with pytest.raises(ValueError, match="requiere --stage"):
        select_stages(frame, stages, cycle=1)
    with pytest.raises(ValueError, match="No hay etapas"):
        select_stages(frame, stages, numbers=[99])


def test_multiple_independent_events_and_simultaneous_pressure():
    frame = signal_frame()
    ids = {"load_id": "L", "run_id": "R", "stage_id": "zone1"}
    events = []
    for number, peak in enumerate((800, 1600)):
        spec = {"event_id": str(number), "interval_h": [(peak-300)/3600, (peak+150)/3600],
                "baseline_h": [(peak-280)/3600, (peak-200)/3600],
                "flank_h": [(peak-60)/3600, (peak-5)/3600], "polarity": "endo"}
        result = analyze_event(frame, spec, ids)
        assert result["status"] == "determined"
        onset = result["onset"]
        assert onset["pressure_bar"]["value"] == pytest.approx(500-360*onset["time_h"])
        assert onset["pressure_bar"]["source"] == "interpolated"
        assert result["phase_assignment"]["confidence"] == "unassigned"
        assert result["parameters"]["noise_k"] == 5
        events.append(result)
    assert events[0]["onset"]["time_h"] < events[1]["onset"]["time_h"]
    assert events[0]["calculation"]["baseline_window"] != events[1]["calculation"]["baseline_window"]


def test_event_rejects_gaps_missing_values_and_insufficient_data():
    frame = signal_frame()
    spec = {"interval_h": [0, 2000/3600]}
    broken = frame.drop(index=range(1000, 1100))
    assert analyze_event(broken, spec, {})["reason"].startswith("record_interruption")
    frame.loc[5, "HF_mW"] = np.nan
    assert analyze_event(frame, spec, {})["reason"].startswith("missing_heat_flow")
    assert analyze_event(frame.iloc[:2], spec, {})["reason"] == "not_enough_points"


def test_legacy_time_cannot_hide_interruptions_in_original_clock():
    frame = signal_frame()
    frame.loc[1000:, "t_original_s"] += 60
    spec = {"interval_h": [0, 2000/3600]}
    assert "original clock" in analyze_event(frame, spec, {})["reason"]
    assert value_at_time(frame, "P_muestra", 999.5/3600)["value"] is None


def test_recorded_and_legacy_time_are_explicit_and_gaps_preserved(tmp_path):
    source = tmp_path / "test.lvm"
    source.write_text("\n".join(f"300\t1\t200\t{i}\t20\t21\t0\t1\t8/17/2026 6:12:{second:02d} PM"
                                 for i, second in enumerate((15, 15, 16, 40))))
    recorded = load_lvm(source, time_basis="recorded")
    legacy = load_lvm(source)
    assert recorded.t_h.iloc[-1]*3600 == 25
    assert legacy.t_h.iloc[-1]*3600 == 1.5
    assert recorded.t_original_s.tolist() == legacy.t_original_s.tolist()
    assert recorded.attrs["provenance"]["offsets_s"] == {"all_signals": 0}
    assert recorded.attrs["units"]["HF_mW"] == "mW"
    assert recorded.attrs["units"]["P_muestra"] == "bar"


@pytest.mark.parametrize("kind", ["signals", "pt", "hf-time", "hf-temperature", "diagnostic"])
def test_saved_figures_never_recompute_onsets(tmp_path, monkeypatch, kind):
    import util.calvet_onset as onset
    import process_dsc_data as legacy
    from util.calvet_plotting import plot_saved
    frame = signal_frame()
    event = analyze_event(frame, {"event_id": "A", "interval_h": [500/3600, 950/3600],
                                 "baseline_h": [520/3600, 600/3600], "flank_h": [740/3600, 795/3600]}, {})
    frame.to_csv(tmp_path / "signals.csv", index=False)
    record = {"signals_file": "signals.csv", "stages": [], "events": [event]}
    write_json(tmp_path / "results.json", record)
    def forbidden(*args, **kwargs):
        raise AssertionError("Plotting attempted to recompute onset")
    for module in (onset, legacy):
        for name in ("compute_onset_tangent", "detect_peak_start", "analyze_peak"):
            monkeypatch.setattr(module, name, forbidden)
    assert plot_saved(tmp_path / "results.json", kind, "A").is_file()


@pytest.mark.parametrize("huber", ["present", "absent", "empty"])
def test_time_overview_has_three_aligned_axes_and_only_available_temperatures(huber):
    import matplotlib.pyplot as plt
    from util.calvet_plotting import time_signals_figure
    frame = signal_frame()
    if huber != "absent":
        frame["T_banho"] = frame.T_muestra-2 if huber == "present" else np.nan
    original = frame.copy(deep=True)
    fig = time_signals_figure(frame)
    pressure, temperature, heat_flow = fig.axes
    try:
        assert len(fig.axes) == 3
        assert pressure.get_shared_x_axes().joined(pressure, temperature)
        assert pressure.get_shared_x_axes().joined(pressure, heat_flow)
        np.testing.assert_allclose(pressure.lines[0].get_ydata(), frame.P_muestra)
        np.testing.assert_allclose(temperature.lines[0].get_ydata(), frame.T_muestra)
        np.testing.assert_allclose(heat_flow.lines[0].get_ydata(), frame.HF_mW)
        assert {line.get_label() for line in temperature.lines} == ({"Muestra", "Huber"} if huber == "present" else {"Muestra"})
        if huber == "present":
            np.testing.assert_allclose(temperature.lines[1].get_ydata(), frame.T_banho)
        for ax in fig.axes:
            for line in ax.lines:
                np.testing.assert_allclose(line.get_xdata(), frame.t_h)
        pd.testing.assert_frame_equal(frame, original)
    finally:
        plt.close(fig)


def test_retired_six_panel_commands_select_time_overview():
    from process_dsc_data import build_parser
    parser = build_parser()
    assert parser.parse_args(["--output", "diagnostic"]).output == ["signals"]
    assert "diagnostic-panel" not in parser.format_help()


@pytest.mark.parametrize("module_name", ["process_dsc_data", "analisis"])
def test_legacy_six_panel_functions_generate_three_time_axes(tmp_path, monkeypatch, module_name):
    import importlib
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    module = importlib.import_module(module_name)
    layouts = []
    original_save = Figure.savefig
    def capture(fig, *args, **kwargs):
        layouts.append(len(fig.axes))
        return original_save(fig, *args, **kwargs)
    monkeypatch.setattr(Figure, "savefig", capture)
    module.plot_panel_6(signal_frame(), tmp_path, outname="run_panel_6plots.png", show_onset=True)
    assert layouts == [3]
    assert (tmp_path / "run_signals_vs_time.png").is_file()
    assert not list(tmp_path.glob("*6plots*"))
    plt.close("all")


def test_metadata_preserves_both_mass_methods_without_inventing_composition(tmp_path):
    source = tmp_path / "metadata.json"
    source.write_text(json.dumps({"load_id": "L", "co2_mass_determinations": [
        {"method_id": "weighing", "mass_g": None}, {"method_id": "pump", "mass_g": None}]}))
    metadata = experimental_metadata(str(source), run_id="R")
    assert len(metadata["co2_mass_determinations"]) == 2
    assert metadata["compositions"] == []
    assert "compositions" in metadata["pending_fields"]
