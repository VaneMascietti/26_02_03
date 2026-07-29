import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import numpy as np
import pandas as pd

from util.data_filters import filter_pt_rows
from util.program_zones import zone_number_by_type, zone_numbers_by_type
from util.pt_plot import plot_pt_tracks


def test_filter_pt_rows_uses_inclusive_bounds_and_finite_values():
    df = pd.DataFrame(
        {
            "t_h": [0.0, 1.0, 2.0, 3.0],
            "T_muestra": [10.0, 20.0, np.inf, 30.0],
            "P_muestra": [100.0, 200.0, 300.0, 0.0],
        }
    )

    selected = filter_pt_rows(
        df,
        t_bounds=(10.0, 20.0),
        p_bounds=(100.0, 200.0),
    )
    assert len(selected) == 2
    assert np.allclose(selected["T_muestra"].to_numpy(), [10.0, 20.0])

    positive = filter_pt_rows(df, positive_pressure=True)
    assert np.all(positive["P_muestra"].to_numpy() > 0.0)


def test_zone_number_helpers_keep_program_order():
    zones = [
        {"num": 4, "type": "Cooling"},
        {"num": 5, "type": "Isoterma"},
        {"num": 6, "type": "Cooling"},
    ]

    assert zone_numbers_by_type(zones, "Cooling") == [4, 6]
    assert zone_number_by_type(zones, "Cooling", 2) == 6
    assert zone_number_by_type(zones, "Cooling", 3) is None


def test_shared_pt_plot_writes_a_figure(tmp_path: Path):
    track = pd.DataFrame(
        {
            "t_h": [0.0, 1.0, 2.0],
            "T_muestra": [10.0, 20.0, 30.0],
            "P_muestra": [100.0, 120.0, 140.0],
        }
    )
    outpath = tmp_path / "pt.png"

    plot_pt_tracks([(track, "E01")], outpath, include_time=True)

    assert outpath.is_file()
    assert outpath.stat().st_size > 0
