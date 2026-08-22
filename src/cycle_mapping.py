"""Map TXT rows to PDF thermal program zones."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .models import DSCData, ProgramSummary, ZoneAssignment


def map_zones_to_data(program: ProgramSummary, data: DSCData) -> list[ZoneAssignment]:
    """Assign rows to zones using PDF cumulative time limits."""

    assignments: list[ZoneAssignment] = []
    frame = data.frame
    for zone in program.zones:
        mask = (frame["time_s"] >= zone.start_s) & (frame["time_s"] <= zone.end_s)
        rows = frame.loc[mask, "source_row"]
        if rows.empty:
            assignments.append(
                ZoneAssignment(zone, data.source, None, None, "missing", "No rows in zone time range.")
            )
        else:
            assignments.append(
                ZoneAssignment(zone, data.source, int(rows.iloc[0]), int(rows.iloc[-1]), "automatic")
            )
    return assignments


def assignments_frame(assignments: list[ZoneAssignment]) -> pd.DataFrame:
    return pd.DataFrame([assignment.as_dict() for assignment in assignments])


def segment_for_assignment(data: DSCData, assignment: ZoneAssignment) -> pd.DataFrame:
    if assignment.start_row is None or assignment.end_row is None:
        return data.frame.iloc[0:0].copy()
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

