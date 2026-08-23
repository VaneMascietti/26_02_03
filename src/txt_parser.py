"""Read Calisto TXT exports without modifying the original data files."""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .models import DSCData, WarningLog


def detect_encoding(path: Path) -> str:
    sample = path.read_bytes()[:4096]
    if sample.startswith(b"\xff\xfe") or sample.count(b"\x00") > len(sample) // 5:
        return "utf-16"
    return "utf-8-sig"


def _pick_delimiter(header: str) -> str:
    candidates = [";", "\t", ","]
    return max(candidates, key=header.count)


def _to_float(value: object) -> float:
    if value is None:
        return np.nan
    text = str(value).strip()
    if text == "":
        return np.nan
    if "," in text and "." in text:
        text = text.replace(",", "")
    elif "," in text:
        text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return np.nan


def _normalise_column(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _read_rows(path: Path, warnings: WarningLog) -> tuple[list[str], list[list[str]]]:
    encoding = detect_encoding(path)
    text = path.read_text(encoding=encoding)
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"{path} is empty")

    delimiter = _pick_delimiter(lines[0])
    reader = csv.reader(io.StringIO("\n".join(lines)), delimiter=delimiter)
    rows = list(reader)
    header = [cell.strip() for cell in rows[0]]
    while header and header[-1] == "":
        header.pop()

    data_rows: list[list[str]] = []
    for row in rows[1:]:
        cells = [cell.strip() for cell in row]
        while len(cells) > len(header) and cells[-1] == "":
            cells.pop()
        if len(cells) != len(header):
            warnings.add(
                f"Skipped malformed row in {path.name}: expected {len(header)} fields, got {len(cells)}"
            )
            continue
        data_rows.append(cells)

    return header, data_rows


def _find_column(columns: list[str], variants: list[str]) -> str | None:
    normalised = {_normalise_column(col): col for col in columns}
    for variant in variants:
        key = _normalise_column(variant)
        for norm, original in normalised.items():
            if key in norm:
                return original
    return None


def read_txt(path: str | Path) -> DSCData:
    """Read one Calisto TXT file and return canonical columns."""

    source = Path(path)
    warnings = WarningLog()
    header, rows = _read_rows(source, warnings)
    raw = pd.DataFrame(rows, columns=header)

    time_col = _find_column(header, ["time"])
    sample_temp_col = _find_column(header, ["sample temperature", "sampletemperature"])
    furnace_temp_col = _find_column(header, ["furnace temperature", "furnacetemperature"])
    heat_col = _find_column(header, ["heatflow", "heat flow"])

    if time_col is None or heat_col is None:
        raise ValueError(f"{source} does not contain recognizable time and heat-flow columns")

    temperature_col = sample_temp_col or furnace_temp_col
    if temperature_col is None:
        raise ValueError(f"{source} does not contain a recognizable temperature column")

    temperature_source = "sample" if sample_temp_col else "furnace"
    if sample_temp_col is None:
        warnings.add("Sample Temperature is not available; Furnace Temperature was used.")

    time_scale = 60.0 if re.search(r"minute|\bmin\b", time_col, flags=re.IGNORECASE) else 1.0
    if time_scale != 1.0:
        warnings.add(f"{time_col} interpreted as minutes and converted to seconds.")

    frame = pd.DataFrame(
        {
            "time_s": raw[time_col].map(_to_float) * time_scale,
            "temperature_c": raw[temperature_col].map(_to_float),
            "heat_flow_mw": raw[heat_col].map(_to_float),
        }
    )
    frame["source_file"] = str(source)
    frame["source_row"] = np.arange(len(frame), dtype=int)

    missing = frame[["time_s", "temperature_c", "heat_flow_mw"]].isna().sum()
    for col, count in missing.items():
        if int(count):
            warnings.add(f"{col} contains {int(count)} missing/non-numeric values.")
    frame = frame.dropna(subset=["time_s", "temperature_c", "heat_flow_mw"]).reset_index(drop=True)

    if not frame["time_s"].is_monotonic_increasing:
        warnings.add("Time is not monotonically increasing; original row order was preserved.")
    duplicated = int(frame["time_s"].duplicated().sum())
    if duplicated:
        warnings.add(f"Detected {duplicated} duplicate time values.")

    dt = frame["time_s"].diff().dropna()
    if not dt.empty:
        median_dt = float(dt.median())
        if median_dt > 0:
            jumps = int((dt > 5 * median_dt).sum())
            if jumps:
                warnings.add(f"Detected {jumps} temporal jumps larger than 5x median dt.")

    return DSCData(
        source=source,
        frame=frame,
        original_columns=header,
        time_column=time_col,
        temperature_column=temperature_col,
        heat_flow_column=heat_col,
        temperature_source=temperature_source,
        warnings=warnings.messages,
    )
