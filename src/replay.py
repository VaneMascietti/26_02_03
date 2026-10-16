"""Plot persisted Calisto event results without rerunning detection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .models import EventResult
from .plotting import plot_cycle


def _optional_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _optional_window(value: Any) -> tuple[float, float] | None:
    if value is None or len(value) != 2:
        return None
    first, second = _optional_float(value[0]), _optional_float(value[1])
    if first is None or second is None:
        return None
    return first, second


def _event_from_dict(data: dict[str, Any]) -> EventResult:
    return EventResult(
        event_label=str(data["event_label"]),
        status=str(data["status"]),
        confidence=str(data["confidence"]),
        onset_temp_c=_optional_float(data.get("onset_temp_c")),
        onset_uncertainty_c=_optional_float(data.get("onset_uncertainty_c")),
        ci95_low_c=_optional_float(data.get("ci95_low_c")),
        ci95_high_c=_optional_float(data.get("ci95_high_c")),
        peak_temp_c=_optional_float(data.get("peak_temp_c")),
        snr=float(data.get("snr") or 0.0),
        integrated_snr=float(data.get("integrated_snr") or 0.0),
        automatic_label=str(data.get("automatic_label") or "unassigned"),
        manual_label=data.get("manual_label"),
        phase_hypothesis=data.get("phase_hypothesis"),
        phase_assignment_status=str(data.get("phase_assignment_status") or "pending_manual_review"),
        manual_onset_temp_c=_optional_float(data.get("manual_onset_temp_c")),
        calisto_onset_temp_c=_optional_float(data.get("calisto_onset_temp_c")),
        baseline_window_c=_optional_window(data.get("baseline_window_c")),
        tangent_window_c=_optional_window(data.get("tangent_window_c")),
        notes=str(data.get("notes") or ""),
    )


def _unique_figure_path(path: Path) -> Path:
    if not any(path.with_suffix(suffix).exists() for suffix in (".png", ".pdf", ".svg")):
        return path
    index = 2
    while True:
        candidate = path.with_name(f"{path.stem}_{index:02d}{path.suffix}")
        if not any(candidate.with_suffix(suffix).exists() for suffix in (".png", ".pdf", ".svg")):
            return candidate
        index += 1


def plot_saved_analysis(outdir: str | Path, cycles: list[int] | None = None) -> list[Path]:
    """Render cycle figures using only the stored signal and event files."""
    root = Path(outdir)
    event_file = root / "events.json"
    if not event_file.is_file():
        raise FileNotFoundError(f"Saved events file not found: {event_file}")
    payload = json.loads(event_file.read_text(encoding="utf-8"))
    selected = set(cycles or [])
    rows = payload.get("events", [])
    rendered: list[Path] = []
    figures = root / "figures" / "replayed"

    cycle_keys = sorted(
        {
            (int(row["cycle_id"]), int(row["stage_id"]), row["signal_file"])
            for row in rows
            if row.get("cycle_id") is not None and row.get("stage_id") is not None
        }
    )
    for cycle, zone, relative_signal in cycle_keys:
        if selected and cycle not in selected:
            continue
        signal_path = root / relative_signal
        if not signal_path.is_file():
            raise FileNotFoundError(f"Saved signal file not found: {signal_path}")
        signal = pd.read_csv(signal_path)
        event_rows = [
            row
            for row in rows
            if int(row["cycle_id"]) == cycle and int(row["stage_id"]) == zone
        ]
        results = [_event_from_dict(row) for row in event_rows]
        target = _unique_figure_path(figures / f"cycle_{cycle:02d}_zone_{zone:02d}.png")
        rendered.append(plot_cycle(signal, results, target))

    if selected and not rendered:
        available = sorted({int(row["cycle_id"]) for row in rows if row.get("cycle_id") is not None})
        raise ValueError(f"No saved results for cycles {sorted(selected)}; available cycles: {available}.")
    if not rendered:
        raise ValueError(f"No cycle results are available in {event_file}.")
    return rendered
