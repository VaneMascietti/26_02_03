"""Export reproducible DSC analysis outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def write_outputs(
    outdir: str | Path,
    results: pd.DataFrame,
    zones: pd.DataFrame,
    assignments: pd.DataFrame,
    config: dict[str, Any],
    warnings: list[str],
    *,
    events: list[dict[str, Any]] | None = None,
    signal_segments: dict[str, pd.DataFrame] | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    out = Path(outdir)
    if out.exists() and any(out.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {out}. Choose a new --outdir or pass --overwrite."
        )
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "results_csv": out / "results_by_cycle.csv",
        "results_xlsx": out / "analysis_workbook.xlsx",
        "zones_csv": out / "thermal_program.csv",
        "assignments_csv": out / "zone_assignments.csv",
        "config_json": out / "analysis_config.json",
        "warnings_txt": out / "warnings.txt",
        "report_html": out / "report.html",
        "events_json": out / "events.json",
    }
    event_frame = pd.DataFrame(events or [])
    signal_dir = out / "signals"
    signal_dir.mkdir(exist_ok=True)
    for name, frame in (signal_segments or {}).items():
        path = signal_dir / name
        frame.to_csv(path, index=False)
    results.to_csv(paths["results_csv"], index=False)
    zones.to_csv(paths["zones_csv"], index=False)
    assignments.to_csv(paths["assignments_csv"], index=False)
    try:
        with pd.ExcelWriter(paths["results_xlsx"]) as writer:
            results.to_excel(writer, sheet_name="results_by_cycle", index=False)
            event_frame.to_excel(writer, sheet_name="events", index=False)
            zones.to_excel(writer, sheet_name="thermal_program", index=False)
            assignments.to_excel(writer, sheet_name="zone_assignments", index=False)
            pd.DataFrame([config]).to_excel(writer, sheet_name="parameters", index=False)
            pd.DataFrame({"warning": warnings}).to_excel(writer, sheet_name="warnings", index=False)
    except (ImportError, ModuleNotFoundError, ValueError):
        paths.pop("results_xlsx", None)
    paths["config_json"].write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    paths["warnings_txt"].write_text("\n".join(warnings), encoding="utf-8")
    paths["events_json"].write_text(
        json.dumps(
            {"schema_version": 1, "events": events or [], "signals": sorted((signal_segments or {}).keys())},
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    html = [
        "<html><head><meta charset='utf-8'><title>DSC analysis report</title></head><body>",
        "<h1>DSC analysis report</h1>",
        "<h2>Results</h2>",
        results.to_html(index=False),
        "<h2>Candidate events (phase assignment pending review)</h2>",
        event_frame.to_html(index=False),
        "<h2>Thermal program</h2>",
        zones.to_html(index=False),
        "<h2>Warnings</h2><ul>",
        *[f"<li>{warning}</li>" for warning in warnings],
        "</ul></body></html>",
    ]
    paths["report_html"].write_text("\n".join(html), encoding="utf-8")
    return paths
