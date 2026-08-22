"""Parse Calisto PDF thermal programs."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from .models import ProgramSummary, ThermalZone, WarningLog

LOGGER = logging.getLogger(__name__)


def _read_pdf_text(path: Path, warnings: WarningLog) -> str:
    try:
        try:
            import pymupdf as fitz  # type: ignore
        except ModuleNotFoundError:
            import fitz  # type: ignore

        with fitz.open(path) as doc:
            text = "\n".join(page.get_text() for page in doc)
        if text.strip():
            return text
    except Exception as exc:  # pragma: no cover - depends on optional install
        warnings.add(f"PyMuPDF unavailable or failed for {path.name}: {exc}")

    try:
        from PyPDF2 import PdfReader

        reader = PdfReader(str(path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        if text.strip():
            return text
    except Exception as exc:
        warnings.add(f"PyPDF2 failed for {path.name}: {exc}")

    warnings.add(
        "No extractable PDF text found. Install PyMuPDF or use OCR/Tesseract before analysis."
    )
    return ""


def _decimal(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip().replace(",", ".")
    try:
        return float(value)
    except ValueError:
        return None


def _normalise_kind(label: str, init: float | None, final: float | None) -> tuple[str, str]:
    lower = label.lower()
    if init is not None and final is not None:
        if final < init:
            return "Cooling", "cooling"
        if final > init:
            return "Heating", "heating"
    if "resfri" in lower or "cool" in lower:
        return "Cooling", "cooling"
    if "aquec" in lower or "heat" in lower:
        return "Heating", "heating"
    return "Isothermal", "isothermal"


def _parse_zone(block: str, zone_number: int, label: str) -> ThermalZone | None:
    duration = _decimal(_first(r"Total duration:\s*([\d,.]+)\s*s", block))
    acquisition = _decimal(_first(r"Acquisition Period:\s*([\d,.]+)\s*s", block))
    points_value = _first(r"Number of Points:\s*(\d+)", block)
    points = int(points_value) if points_value else None

    sequence = re.search(
        r"\n\s*1\s*\n\s*([+-]?[\d,.]+)\s+([+-]?[\d,.]+)\s+([+-]?[\d,.]+)\s+([+-]?[\d,.]+)",
        block,
    )
    if not sequence:
        LOGGER.debug("Skipping zone %s without complete sequence", zone_number)
        return None

    init_temp = _decimal(sequence.group(1))
    seq_duration = _decimal(sequence.group(2))
    final_temp = _decimal(sequence.group(3))
    scan_rate = _decimal(sequence.group(4))
    if duration is None:
        duration = seq_duration

    kind, direction = _normalise_kind(label, init_temp, final_temp)
    return ThermalZone(
        zone_number=zone_number,
        label=label.strip(),
        kind=kind,
        direction=direction,
        init_temp_c=init_temp,
        final_temp_c=final_temp,
        duration_s=duration,
        scan_rate_k_min=scan_rate,
        acquisition_period_s=acquisition,
        points=points,
    )


def _first(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    return match.group(1).strip() if match else None


def parse_program_pdf(path: str | Path) -> ProgramSummary:
    """Parse a Calisto PDF and infer cycles from the thermal sequence."""

    source = Path(path)
    warnings = WarningLog()
    text = _read_pdf_text(source, warnings)

    title = next((line.strip() for line in text.splitlines() if line.strip()), None)
    comment = _first(r"Comment:\s*(.+)", text)
    apparatus = _first(r"Apparatus:\s*\n?([^\n]+)", text)
    launch_date = _first(r"Launch date:\s*\n?([^\n]+)", text)

    headers = list(re.finditer(r"(?m)^\s*(\d+)\s*:\s*([^\n]+)\s*$", text))
    zones: list[ThermalZone] = []
    for index, header in enumerate(headers):
        start = header.end()
        end = headers[index + 1].start() if index + 1 < len(headers) else len(text)
        zone = _parse_zone(text[start:end], int(header.group(1)), header.group(2))
        if zone is not None:
            zones.append(zone)

    elapsed = 0.0
    cooling_cycle = 0
    for zone in zones:
        zone.start_s = elapsed
        elapsed += float(zone.duration_s or 0.0)
        zone.end_s = elapsed
        if zone.direction == "cooling":
            cooling_cycle += 1
            zone.cycle = cooling_cycle
        elif cooling_cycle:
            zone.cycle = cooling_cycle

    if not zones:
        warnings.add("No thermal zones were parsed from the PDF.")

    return ProgramSummary(
        source=source,
        zones=zones,
        title=title,
        comment=comment,
        apparatus=apparatus,
        launch_date=launch_date,
        warnings=warnings.messages,
    )
