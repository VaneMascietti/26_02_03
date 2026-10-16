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

    # Calisto exports may prepend metadata (creation date, mass, etc.) before
    # the actual table. Locate the first line containing both required fields
    # instead of assuming that the first non-empty line is the header.
    header_index = next(
        (
            index
            for index, line in enumerate(lines)
            if re.search(r"\btime\b", line, flags=re.IGNORECASE)
            and re.search(r"heat\s*flow|heatflow", line, flags=re.IGNORECASE)
        ),
        None,
    )
    if header_index is None:
        raise ValueError(f"{path} does not contain a recognizable table header")
    if header_index:
        warnings.add(
            f"Ignored {header_index} metadata line(s) before the table header in {path.name}."
        )

    table_lines = lines[header_index:]
    delimiter = _pick_delimiter(table_lines[0])
    reader = csv.reader(io.StringIO("\n".join(table_lines)), delimiter=delimiter)
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


def read_txt(path: str | Path, *, require_temperature: bool = True) -> DSCData:
    """Read one Calisto TXT file and return canonical columns."""

    source = Path(path)
    warnings = WarningLog()
    header, rows = _read_rows(source, warnings)
    raw = pd.DataFrame(rows, columns=header)

    time_col = _find_column(header, ["time"])
    sample_temp_col = _find_column(header, ["sample temperature", "sampletemperature"])
    furnace_temp_col = _find_column(header, ["furnace temperature", "furnacetemperature"])
    huber_temp_col = _find_column(header, ["huber temperature", "bath temperature"])
    setpoint_temp_col = _find_column(
        header,
        ["setpoint temperature", "target temperature", "programmed temperature"],
    )
    heat_col = _find_column(header, ["heatflow", "heat flow"])

    if time_col is None or heat_col is None:
        raise ValueError(f"{source} does not contain recognizable time and heat-flow columns")

    temperature_col = sample_temp_col or furnace_temp_col
    if temperature_col is None and require_temperature:
        raise ValueError(f"{source} does not contain a recognizable temperature column")

    temperature_source = "sample" if sample_temp_col else "furnace" if furnace_temp_col else "unavailable"
    if sample_temp_col is None and furnace_temp_col is not None:
        warnings.add("Sample Temperature is not available; Furnace Temperature was used.")

    time_scale = 60.0 if re.search(r"minute|\bmin\b", time_col, flags=re.IGNORECASE) else 1.0
    if time_scale != 1.0:
        warnings.add(f"{time_col} interpreted as minutes and converted to seconds.")

    frame = pd.DataFrame(
        {
            "time_s": raw[time_col].map(_to_float) * time_scale,
            "temperature_c": raw[temperature_col].map(_to_float) if temperature_col else np.nan,
            "heat_flow_mw": raw[heat_col].map(_to_float),
        }
    )
    for source_column, canonical_name in (
        (sample_temp_col, "sample_temperature_c"),
        (furnace_temp_col, "furnace_temperature_c"),
        (huber_temp_col, "huber_temperature_c"),
        (setpoint_temp_col, "setpoint_temperature_c"),
    ):
        if source_column is not None:
            frame[canonical_name] = raw[source_column].map(_to_float)
    frame["source_file"] = str(source)
    frame["source_row"] = np.arange(len(frame), dtype=int)

    missing = frame[["time_s", "temperature_c", "heat_flow_mw"]].isna().sum()
    for col, count in missing.items():
        if int(count):
            warnings.add(f"{col} contains {int(count)} missing/non-numeric values.")
    required_signals = ["time_s", "heat_flow_mw"]
    if require_temperature:
        required_signals.append("temperature_c")
    frame = frame.dropna(subset=required_signals).reset_index(drop=True)
    if temperature_col is None:
        frame = frame.drop(columns="temperature_c")
        warnings.add("Temperature is unavailable; time-domain analysis remains possible.")

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
