"""Read, locate, and select thermal-program zones from instrument PDFs."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
import re

import numpy as np
import pandas as pd


Zone = dict[str, float | int | str]


def load_program_zones(pdf_path: Path) -> list[Zone]:
    """Extract numbered zones and durations from a thermal-program PDF."""
    try:
        import PyPDF2
    except Exception:
        return []

    if not pdf_path.exists():
        return []

    reader = PyPDF2.PdfReader(str(pdf_path))
    text = "\n".join((page.extract_text() or "") for page in reader.pages)
    lines = text.splitlines()

    zone_headers: list[tuple[int, int, str]] = []
    for index, line in enumerate(lines):
        match = re.match(r"\s*(\d+)\s*:\s*(.+)\s*$", line)
        if match:
            zone_headers.append((index, int(match.group(1)), match.group(2).strip()))

    # Some PDF exports split headers at page boundaries. Infer a missing number
    # only when its neighbors make the sequence unambiguous.
    page_header_candidates: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        match = re.match(r"\s*Page\s+\d+\s*:\s*(.+)\s*$", line, flags=re.IGNORECASE)
        if not match:
            continue
        zone_type = match.group(1).strip()
        if zone_type.lower().startswith(("cooling", "heating", "isoterma", "isothermal", "return")):
            page_header_candidates.append((index, zone_type))

    if page_header_candidates and zone_headers:
        existing_numbers = {number for _, number, _ in zone_headers}
        explicit_by_index = sorted(zone_headers, key=lambda item: item[0])
        inferred_headers: list[tuple[int, int, str]] = []

        for index, zone_type in page_header_candidates:
            previous = None
            following = None
            for header in explicit_by_index:
                if header[0] < index:
                    previous = header
                elif header[0] > index:
                    following = header
                    break

            inferred_number = None
            if previous and following and following[1] - previous[1] == 2:
                inferred_number = previous[1] + 1
            elif previous and previous[1] + 1 not in existing_numbers:
                inferred_number = previous[1] + 1
            elif following and following[1] - 1 > 0 and following[1] - 1 not in existing_numbers:
                inferred_number = following[1] - 1

            if inferred_number and inferred_number not in existing_numbers:
                inferred_headers.append((index, inferred_number, zone_type))
                existing_numbers.add(inferred_number)

        if inferred_headers:
            zone_headers.extend(inferred_headers)
            zone_headers.sort(key=lambda item: item[0])

    zones: list[Zone] = []
    for index, (line_index, number, zone_type) in enumerate(zone_headers):
        next_line_index = zone_headers[index + 1][0] if index + 1 < len(zone_headers) else len(lines)
        duration: float | int | None = None
        for line in lines[line_index:next_line_index]:
            match = re.search(r"Total duration:\s*([0-9]+(?:\.[0-9]+)?)\s*s", line)
            if match:
                duration_value = float(match.group(1))
                duration = int(duration_value) if duration_value.is_integer() else duration_value
                break
        if duration is not None:
            zones.append({"num": number, "type": zone_type, "duration_s": duration})

    return zones


def program_total_seconds(pdf_path: Path) -> float | int | None:
    zones = load_program_zones(pdf_path)
    if not zones:
        return None
    return sum(zone["duration_s"] for zone in zones)


def program_zone_spans(pdf_path: Path, t_h: np.ndarray | None = None) -> list[Zone]:
    """Return zone start/end times in hours, optionally clipped to data time."""
    zones = load_program_zones(pdf_path)
    if not zones:
        return []

    spans: list[Zone] = []
    elapsed_s = 0.0
    for zone in zones:
        duration_s = float(zone["duration_s"])
        spans.append(
            {
                "num": int(zone["num"]),
                "type": str(zone["type"]),
                "start_h": elapsed_s / 3600.0,
                "end_h": (elapsed_s + duration_s) / 3600.0,
            }
        )
        elapsed_s += duration_s

    if t_h is None or len(t_h) == 0:
        return spans

    time_min, time_max = float(t_h[0]), float(t_h[-1])
    clipped: list[Zone] = []
    for zone in spans:
        start_h = float(zone["start_h"])
        end_h = float(zone["end_h"])
        if end_h <= time_min or start_h >= time_max:
            continue
        clipped.append(
            {
                "num": int(zone["num"]),
                "type": str(zone["type"]),
                "start_h": max(start_h, time_min),
                "end_h": min(end_h, time_max),
            }
        )
    return clipped


def first_zone_span_h(pdf_path: Path, type_terms: Iterable[str]) -> tuple[float, float] | None:
    """Find the first program zone whose type contains any requested term."""
    terms = tuple(term.lower() for term in type_terms)
    for span in program_zone_spans(pdf_path):
        zone_type = str(span["type"]).lower()
        if any(term in zone_type for term in terms):
            return float(span["start_h"]), float(span["end_h"])
    return None


def zone_numbers_by_type(zones: Iterable[Zone], zone_type: str) -> list[int]:
    return [
        int(zone["num"])
        for zone in zones
        if str(zone["type"]).lower().startswith(zone_type.lower())
    ]


def zone_number_by_type(zones: Iterable[Zone] | None, zone_type: str, ordinal: int | None) -> int | None:
    if zones is None or ordinal is None:
        return None
    matches = zone_numbers_by_type(zones, zone_type)
    if ordinal <= 0 or ordinal > len(matches):
        return None
    return matches[ordinal - 1]


def zone_label(zones: Iterable[Zone], zone_num: int) -> str:
    zone_list = list(zones)
    zone = next((item for item in zone_list if int(item["num"]) == zone_num), None)
    if zone is None:
        return f"Zona {zone_num}"
    zone_type = str(zone["type"])
    count = sum(
        1
        for item in zone_list
        if str(item["type"]) == zone_type and int(item["num"]) <= zone_num
    )
    return f"{zone_type} {count}"


def zone_type_to_english(zone_type: str) -> str:
    """Translate program zone names to English for plot titles."""
    normalized = (zone_type or "").strip().lower()
    mapping = {
        "isoterma": "Isotherm",
        "isothermal": "Isotherm",
        "heating": "Heating",
        "cooling": "Cooling",
        "return": "Return",
    }
    return mapping.get(normalized, zone_type)


def isotherm_spans_from_program(pdf_path: Path, t_h: np.ndarray | None = None) -> list[tuple[float, float]]:
    spans = [
        (float(zone["start_h"]), float(zone["end_h"]))
        for zone in program_zone_spans(pdf_path, t_h=t_h)
        if "isoterma" in str(zone["type"]).lower()
    ]
    return spans


def get_zone_spans(
    clean: pd.DataFrame,
    pdf_path: Path,
    fallback: Callable[[np.ndarray, np.ndarray], list[Zone]] | None = None,
) -> list[Zone]:
    time_h = clean["t_h"].to_numpy() if "t_h" in clean else clean["t_s"].to_numpy() / 3600.0
    zones = program_zone_spans(pdf_path, t_h=time_h)
    if zones or fallback is None:
        return zones
    return fallback(time_h, clean["T_horno"].to_numpy())


def select_zone(
    clean: pd.DataFrame,
    zone_num: int,
    pdf_path: Path,
    rebase_time: bool = True,
    fallback: Callable[[np.ndarray, np.ndarray], list[Zone]] | None = None,
) -> tuple[pd.DataFrame | None, Zone | None]:
    zones = get_zone_spans(clean, pdf_path, fallback=fallback)
    if not zones:
        return None, None
    zone = next((item for item in zones if int(item["num"]) == zone_num), None)
    if zone is None:
        return None, None

    original_start_h = float(zone["start_h"])
    original_end_h = float(zone["end_h"])
    mask = (clean["t_h"] >= original_start_h) & (clean["t_h"] <= original_end_h)
    subset = clean.loc[mask].copy()
    if rebase_time and not subset.empty:
        subset["t_h"] = subset["t_h"] - original_start_h
        if "t_s" in subset:
            subset["t_s"] = subset["t_s"] - original_start_h * 3600.0
        return subset, {
            "num": int(zone["num"]),
            "type": str(zone["type"]),
            "start_h": 0.0,
            "end_h": original_end_h - original_start_h,
            "offset_h": original_start_h,
            "orig_start_h": original_start_h,
            "orig_end_h": original_end_h,
        }

    selected_zone = dict(zone)
    selected_zone["offset_h"] = 0.0
    selected_zone["orig_start_h"] = original_start_h
    selected_zone["orig_end_h"] = original_end_h
    return subset, selected_zone
