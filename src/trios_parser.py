"""Read legacy Excel exports produced by TA Instruments TRIOS."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd


@dataclass(slots=True)
class TriosSegment:
    """One tabular procedure segment from a TRIOS workbook."""

    sheet_name: str
    direction: str
    frame: pd.DataFrame
    time_unit: str
    temperature_unit: str
    heat_flow_unit: str

    def summary(self) -> dict[str, Any]:
        frame = self.frame
        return {
            "sheet": self.sheet_name,
            "direction": self.direction,
            "rows": len(frame),
            "time_start_min": float(frame["time_min"].iloc[0]),
            "time_end_min": float(frame["time_min"].iloc[-1]),
            "temperature_start_c": float(frame["temperature_c"].iloc[0]),
            "temperature_end_c": float(frame["temperature_c"].iloc[-1]),
            "temperature_min_c": float(frame["temperature_c"].min()),
            "temperature_max_c": float(frame["temperature_c"].max()),
            "heat_flow_unit": self.heat_flow_unit,
        }


def _normalise(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _find_header(sheet: Any) -> tuple[int, int, int, int] | None:
    """Return header row and Time/Temperature/Heat Flow column indexes."""

    for row_index in range(min(sheet.nrows, 50)):
        values = [str(sheet.cell_value(row_index, col)) for col in range(sheet.ncols)]
        normalised = [_normalise(value) for value in values]
        time_col = next((i for i, value in enumerate(normalised) if value == "time"), None)
        temperature_col = next(
            (i for i, value in enumerate(normalised) if "temperature" in value),
            None,
        )
        heat_flow_col = next(
            (i for i, value in enumerate(normalised) if "heatflow" in value),
            None,
        )
        if time_col is not None and temperature_col is not None and heat_flow_col is not None:
            return row_index, time_col, temperature_col, heat_flow_col
    return None


def _direction(temperature: pd.Series) -> str:
    values = temperature.to_numpy(float)
    edge = max(10, min(200, len(values) // 10))
    start = float(np.nanmedian(values[:edge]))
    end = float(np.nanmedian(values[-edge:]))
    delta = end - start
    if delta < -0.5:
        return "cooling"
    if delta > 0.5:
        return "heating"
    return "isothermal"


def _read_segment(sheet: Any, source: Path) -> TriosSegment | None:
    header = _find_header(sheet)
    if header is None:
        return None
    header_row, time_col, temperature_col, heat_flow_col = header
    units_row = header_row + 1
    units = [
        str(sheet.cell_value(units_row, column)).strip() if units_row < sheet.nrows else ""
        for column in (time_col, temperature_col, heat_flow_col)
    ]
    data_start = units_row + 1
    records = [
        (
            sheet.cell_value(row, time_col),
            sheet.cell_value(row, temperature_col),
            sheet.cell_value(row, heat_flow_col),
            row + 1,
        )
        for row in range(data_start, sheet.nrows)
    ]
    frame = pd.DataFrame(
        records,
        columns=["time_min", "temperature_c", "heat_flow_w_g", "source_row"],
    )
    for column in ["time_min", "temperature_c", "heat_flow_w_g"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["time_min", "temperature_c", "heat_flow_w_g"]).reset_index(drop=True)
    if frame.empty:
        return None
    frame["source_file"] = str(source)
    frame["source_sheet"] = sheet.name
    return TriosSegment(
        sheet_name=sheet.name,
        direction=_direction(frame["temperature_c"]),
        frame=frame,
        time_unit=units[0] or "min",
        temperature_unit=units[1] or "°C",
        heat_flow_unit=units[2] or "W/g",
    )


def read_trios_xls(path: str | Path) -> tuple[dict[str, Any], list[TriosSegment]]:
    """Return workbook metadata and every recognizable TRIOS data segment."""

    try:
        import xlrd
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("Reading TRIOS .xls files requires xlrd>=2.0.1") from exc

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    workbook = xlrd.open_workbook(source, on_demand=True)
    metadata: dict[str, Any] = {"source_file": str(source)}
    segments: list[TriosSegment] = []
    for sheet_name in workbook.sheet_names():
        sheet = workbook.sheet_by_name(sheet_name)
        segment = _read_segment(sheet, source)
        if segment is not None:
            segments.append(segment)
            continue
        if sheet_name.lower() == "details":
            for row in range(sheet.nrows):
                key = str(sheet.cell_value(row, 0)).strip() if sheet.ncols else ""
                if not key:
                    continue
                value = sheet.cell_value(row, 1) if sheet.ncols > 1 else ""
                metadata[key] = value
    workbook.release_resources()
    if not segments:
        raise ValueError(f"{source} does not contain recognizable TRIOS data sheets")
    return metadata, segments
