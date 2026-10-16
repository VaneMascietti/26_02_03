"""Map TXT rows to PDF thermal program zones."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping

import pandas as pd

from .models import DSCData, ProgramSummary, ZoneAssignment


def map_zones_to_data(
    program: ProgramSummary,
    data: DSCData,
    manual_bounds: Mapping[int, tuple[float, float]] | None = None,
) -> list[ZoneAssignment]:
    """Assign rows to program zones or explicit manually reviewed time bounds."""

    assignments: list[ZoneAssignment] = []
    frame = data.frame
    time_warnings: list[str] = []
    if not frame["time_s"].is_monotonic_increasing:
        time_warnings.append(
            "Raw TXT time is non-monotonic; matching rows are retained in source order and require review."
        )
    if frame["time_s"].duplicated().any():
        time_warnings.append(
            "Raw TXT time contains duplicate values; zone boundaries may be temporally ambiguous."
        )
    if manual_bounds:
        known_zones = {zone.zone_number for zone in program.zones}
        unknown_zones = set(manual_bounds).difference(known_zones)
        if unknown_zones:
            raise ValueError(f"Manual bounds refer to unknown zones: {sorted(unknown_zones)}")
    for zone_index, zone in enumerate(program.zones):
        manual = (manual_bounds or {}).get(zone.zone_number)
        start_s, end_s = manual or (zone.start_s, zone.end_s)
        if not start_s < end_s:
            raise ValueError(f"Invalid time bounds for zone {zone.zone_number}: {start_s}..{end_s}")
        final_zone = zone_index == len(program.zones) - 1
        right = frame["time_s"] <= end_s if final_zone else frame["time_s"] < end_s
        mask = (frame["time_s"] >= start_s) & right
        selected_rows = frame.loc[mask, "source_row"].astype(int).tolist()
        if not selected_rows:
            assignments.append(
                ZoneAssignment(
                    zone,
                    data.source,
                    None,
                    None,
                    "missing",
                    "No rows in selected time range.",
                    start_s,
                    end_s,
                    "manual" if manual else "program",
                )
            )
        else:
            row_warnings = list(time_warnings)
            if (
                time_warnings
                and any(
                    current - previous > 1
                    for previous, current in zip(selected_rows, selected_rows[1:])
                )
            ):
                row_warnings.append(
                    "Matching timestamps select disjoint source-row intervals; "
                    "program-zone alignment is not verified."
                )
            assignments.append(
                ZoneAssignment(
                    zone,
                    data.source,
                    selected_rows[0],
                    selected_rows[-1],
                    "manual" if manual else "automatic",
                    " ".join(row_warnings) or None,
                    start_s,
                    end_s,
                    "manual" if manual else "program",
                    selected_rows,
                )
            )
    return assignments


def assignments_frame(assignments: list[ZoneAssignment]) -> pd.DataFrame:
    return pd.DataFrame([assignment.as_dict() for assignment in assignments])


def segment_for_assignment(data: DSCData, assignment: ZoneAssignment) -> pd.DataFrame:
    if assignment.start_row is None or assignment.end_row is None:
        return data.frame.iloc[0:0].copy()
    if assignment.source_rows is not None:
        return data.frame[data.frame["source_row"].isin(assignment.source_rows)].copy()
    return data.frame[
        (data.frame["source_row"] >= assignment.start_row)
        & (data.frame["source_row"] <= assignment.end_row)
    ].copy()


def load_txt_folder(paths: list[str | Path]) -> list[Path]:
    """Expand file and folder inputs to TXT paths."""

    found: list[Path] = []
    for item in paths:
        path = Path(item)
        if path.is_dir():
            found.extend(sorted(path.glob("*.txt")))
        else:
            found.append(path)
    return found
