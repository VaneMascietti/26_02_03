"""Shared data models for DSC/uDSC analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(slots=True)
class WarningLog:
    """Collect non-fatal parsing and analysis warnings."""

    messages: list[str] = field(default_factory=list)

    def add(self, message: str) -> None:
        if message not in self.messages:
            self.messages.append(message)


@dataclass(slots=True)
class ThermalZone:
    """One Calisto thermal program zone."""

    zone_number: int
    label: str
    kind: str
    direction: str
    init_temp_c: float | None
    final_temp_c: float | None
    duration_s: float | None
    scan_rate_k_min: float | None
    acquisition_period_s: float | None
    points: int | None
    start_s: float = 0.0
    end_s: float = 0.0
    cycle: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "zone_number": self.zone_number,
            "label": self.label,
            "kind": self.kind,
            "direction": self.direction,
            "init_temp_c": self.init_temp_c,
            "final_temp_c": self.final_temp_c,
            "duration_s": self.duration_s,
            "scan_rate_k_min": self.scan_rate_k_min,
            "acquisition_period_s": self.acquisition_period_s,
            "points": self.points,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "cycle": self.cycle,
        }


@dataclass(slots=True)
class ProgramSummary:
    """Parsed Calisto program and experiment metadata."""

    source: Path
    zones: list[ThermalZone]
    title: str | None = None
    comment: str | None = None
    apparatus: str | None = None
    launch_date: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def cooling_zones(self) -> list[ThermalZone]:
        return [zone for zone in self.zones if zone.direction == "cooling"]

    @property
    def heating_zones(self) -> list[ThermalZone]:
        return [zone for zone in self.zones if zone.direction == "heating"]

    @property
    def isothermal_zones(self) -> list[ThermalZone]:
        return [zone for zone in self.zones if zone.direction == "isothermal"]

    @property
    def cycles(self) -> int:
        return len(self.cooling_zones)

    def zones_frame(self) -> pd.DataFrame:
        return pd.DataFrame([zone.as_dict() for zone in self.zones])


@dataclass(slots=True)
class DSCData:
    """Parsed DSC time series."""

    source: Path
    frame: pd.DataFrame
    original_columns: list[str]
    time_column: str
    temperature_column: str | None
    heat_flow_column: str
    temperature_source: str
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ZoneAssignment:
    """Mapping between one thermal zone and rows in a TXT data set."""

    zone: ThermalZone
    file: Path
    start_row: int | None
    end_row: int | None
    status: str
    warning: str | None = None
    selected_start_s: float | None = None
    selected_end_s: float | None = None
    selection_source: str = "program"
    source_rows: list[int] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "cycle": self.zone.cycle,
            "zone_number": self.zone.zone_number,
            "kind": self.zone.kind,
            "direction": self.zone.direction,
            "init_temp_c": self.zone.init_temp_c,
            "final_temp_c": self.zone.final_temp_c,
            "scan_rate_k_min": self.zone.scan_rate_k_min,
            "file": str(self.file),
            "start_row": self.start_row,
            "end_row": self.end_row,
            "selected_start_s": self.selected_start_s,
            "selected_end_s": self.selected_end_s,
            "selection_source": self.selection_source,
            "selected_row_count": len(self.source_rows) if self.source_rows is not None else None,
            "status": self.status,
            "warning": self.warning,
        }


@dataclass(slots=True)
class NoiseMetrics:
    """Noise and model comparison metrics."""

    sigma: float
    rms: float
    mad_sigma: float
    max_abs_deviation: float
    snr: float
    integrated_snr: float
    consecutive_points: int
    correlation_length: int
    bic_baseline: float
    bic_segmented: float
    delta_bic: float
    bootstrap_p: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "sigma": self.sigma,
            "rms": self.rms,
            "mad_sigma": self.mad_sigma,
            "max_abs_deviation": self.max_abs_deviation,
            "snr": self.snr,
            "integrated_snr": self.integrated_snr,
            "consecutive_points": self.consecutive_points,
            "correlation_length": self.correlation_length,
            "bic_baseline": self.bic_baseline,
            "bic_segmented": self.bic_segmented,
            "delta_bic": self.delta_bic,
            "bootstrap_p": self.bootstrap_p,
        }


@dataclass(slots=True)
class EventResult:
    """Candidate transition result for one zone."""

    event_label: str
    status: str
    confidence: str
    onset_temp_c: float | None
    onset_uncertainty_c: float | None
    ci95_low_c: float | None
    ci95_high_c: float | None
    peak_temp_c: float | None
    snr: float
    integrated_snr: float
    automatic_label: str
    manual_label: str | None = None
    phase_hypothesis: str | None = None
    phase_assignment_status: str = "pending_manual_review"
    manual_onset_temp_c: float | None = None
    calisto_onset_temp_c: float | None = None
    baseline_window_c: tuple[float, float] | None = None
    tangent_window_c: tuple[float, float] | None = None
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_label": self.event_label,
            "status": self.status,
            "confidence": self.confidence,
            "onset_temp_c": self.onset_temp_c,
            "onset_uncertainty_c": self.onset_uncertainty_c,
            "ci95_low_c": self.ci95_low_c,
            "ci95_high_c": self.ci95_high_c,
            "peak_temp_c": self.peak_temp_c,
            "snr": self.snr,
            "integrated_snr": self.integrated_snr,
            "automatic_label": self.automatic_label,
            "manual_label": self.manual_label,
            "phase_hypothesis": self.phase_hypothesis,
            "phase_assignment_status": self.phase_assignment_status,
            "detection_confidence": self.confidence,
            "phase_assignment_confidence": "not_assessed",
            "manual_onset_temp_c": self.manual_onset_temp_c,
            "calisto_onset_temp_c": self.calisto_onset_temp_c,
            "baseline_window_c": self.baseline_window_c,
            "tangent_window_c": self.tangent_window_c,
            "notes": self.notes,
        }
