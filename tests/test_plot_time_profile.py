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
