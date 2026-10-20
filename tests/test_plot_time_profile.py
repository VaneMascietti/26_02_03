from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from plot_time_profile import _clean_profile, plot_time_profile


def test_time_profile_writes_sample_and_bath_temperature_plot(tmp_path: Path):
    data = pd.DataFrame(
        {
            "t_h": [0.0, 1.0, 2.0],
            "t_s": [0.0, 3600.0, 7200.0],
            "P_muestra": [200.0, 210.0, 220.0],
            "T_muestra": [20.0, 30.0, 40.0],
            "T_banho": [19.0, 25.0, 35.0],
        }
    )
    outpath = tmp_path / "profile.png"

    clean = _clean_profile(data, "h")
    plot_time_profile(
        clean,
        outpath,
        data_label="E01",
        unit="h",
        temperature="both",
        title=None,
    )

    assert outpath.is_file()
    assert outpath.stat().st_size > 0


def test_heat_flow_requested_limits_expand_to_preserve_every_point(tmp_path, monkeypatch):
    import matplotlib.figure

    frame = pd.DataFrame(
        {
            "plot_time": [0.0, 1.0, 2.0, 3.0],
            "T_muestra": [20.0, 30.0, 25.0, 40.0],
            "P_muestra": [100.0, 200.0, 150.0, 300.0],
            "HF_mW": [-11.0, -2.0, 6.0, 78.0],
        }
    )
    figures = []
    monkeypatch.setattr(
        matplotlib.figure.Figure,
        "savefig",
        lambda fig, *args, **kwargs: figures.append(fig),
    )

    plot_time_profile(
        frame,
        tmp_path / "profile.png",
        data_label="test",
        unit="h",
        temperature="sample",
        title=None,
        heat_flow=True,
        hf_ylim=(-12.0, 12.0),
    )

    heat_flow_axis = next(
        axis for axis in figures[0].axes if axis.get_ylabel().startswith("Heat flow (mW)")
    )
    low, high = heat_flow_axis.get_ylim()
    assert low <= -12 and high >= 78
    assert not any("Outside scale" in text.get_text() for text in heat_flow_axis.texts)


def test_heat_flow_autoscale_is_robust_to_isolated_extrema(tmp_path, monkeypatch):
    import matplotlib.figure
    import numpy as np

    hf = np.append(np.linspace(-5.0, 5.0, 1000), 78.5)
    frame = pd.DataFrame(
        {
            "plot_time": np.arange(len(hf), dtype=float),
            "T_muestra": np.linspace(20.0, 40.0, len(hf)),
            "P_muestra": np.linspace(100.0, 300.0, len(hf)),
            "HF_mW": hf,
        }
    )
    figures = []
    monkeypatch.setattr(
        matplotlib.figure.Figure,
        "savefig",
        lambda fig, *args, **kwargs: figures.append(fig),
    )

    plot_time_profile(
        frame,
        tmp_path / "profile.png",
        data_label="test",
        unit="h",
        temperature="sample",
        title=None,
        heat_flow=True,
    )

    heat_flow_axis = next(
        axis for axis in figures[0].axes if axis.get_ylabel().startswith("Heat flow (mW)")
    )
    low, high = heat_flow_axis.get_ylim()
    assert low < -5.0 and high > 78.5
    assert heat_flow_axis.get_yscale() == "symlog"
    np.testing.assert_array_equal(heat_flow_axis.lines[0].get_ydata(), hf)
    assert not any("Outside scale" in text.get_text() for text in heat_flow_axis.texts)


def test_onsets_interpolate_signals_and_reject_ambiguous_crossings():
    import pytest
    from plot_time_profile import onset_intersections

    data = pd.DataFrame({
        "plot_time": [0., 1., 2., 3.], "T_muestra": [30., 20., 30., 20.],
        "P_muestra": [310., 290., 310., 290.], "HF_mW": [2., 4., 2., 4.],
    })
    with pytest.raises(ValueError, match="2 cooling crossings"):
        onset_intersections(data, [25.])
    result = onset_intersections(data, [25.], time_max=1.)[0]
    assert result == {"plot_time": .5, "T_muestra": 25., "P_muestra": 300., "HF_mW": 3.}
    with pytest.raises(ValueError, match="0 cooling crossings"):
        onset_intersections(data, [10.])
