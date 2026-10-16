from pathlib import Path

import pandas as pd
import pytest

from process_experiments_from_metadata import (
    command_for_row,
    filter_metadata,
    load_metadata,
)


def test_load_metadata_validates_required_columns(tmp_path):
    metadata = tmp_path / "experiments.csv"
    metadata.write_text("experiment_id,file_lvm\nE08,26_03_30.lvm\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required columns"):
        load_metadata(metadata)


def test_filter_metadata_by_experiment_load_and_stage():
    df = pd.DataFrame(
        [
            {
                "experiment_id": "E08",
                "load_id": "L08",
                "run_id": "R01",
                "date": "2026-03-30",
                "stage": "heating",
                "file_lvm": "26_03_30.lvm",
                "file_program": "26_03_30.pdf",
                "notes": "",
            },
            {
                "experiment_id": "E08",
                "load_id": "L08",
                "run_id": "R02",
                "date": "2026-03-31",
                "stage": "cooling",
                "file_lvm": "26_03_31.lvm",
                "file_program": "26_03_31.pdf",
                "notes": "",
            },
            {
                "experiment_id": "E09",
                "load_id": "L09",
                "run_id": "R01",
                "date": "2026-04-06",
                "stage": "heating",
                "file_lvm": "26_04_06.lvm",
                "file_program": "26_04_06.pdf",
                "notes": "",
            },
        ]
    )

    selected = filter_metadata(df, experiment_id="E08", load_id="L08", stage="COOLING")

    assert len(selected) == 1
    assert selected.loc[0, "run_id"] == "R02"
    assert selected.loc[0, "file_lvm"] == "26_03_31.lvm"


def test_command_for_row_keeps_stage_zone_selection_explicit():
    row = pd.Series(
        {
            "stage": "heating",
            "load_id": "L08",
            "run_id": "R01",
            "file_lvm": "26_03_30.lvm",
            "file_program": "26_03_30.pdf",
        }
    )

    base = command_for_row(row, python_executable="python", process_script=Path("process_dsc_data.py"))
    with_zones = command_for_row(
        row,
        python_executable="python",
        process_script=Path("process_dsc_data.py"),
        select_stage_zones=True,
        extra_args=["--onset-method", "simple"],
    )

    assert base == [
        "python",
        "process_dsc_data.py",
        "--data",
        "26_03_30.lvm",
        "--program",
        "26_03_30.pdf",
        "--load-id",
        "L08",
        "--run-id",
        "R01",
        "--no-panel",
        "--analyze-events",
    ]
    assert with_zones == [
        "python",
        "process_dsc_data.py",
        "--data",
        "26_03_30.lvm",
        "--program",
        "26_03_30.pdf",
        "--load-id",
        "L08",
        "--run-id",
        "R01",
        "--all-heating",
        "--no-panel",
        "--analyze-events",
        "--onset-method",
        "simple",
    ]
