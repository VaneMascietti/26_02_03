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
