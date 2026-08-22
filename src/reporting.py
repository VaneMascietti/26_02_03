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
) -> dict[str, Path]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "results_csv": out / "results_by_cycle.csv",
        "results_xlsx": out / "analysis_workbook.xlsx",
        "zones_csv": out / "thermal_program.csv",
        "assignments_csv": out / "zone_assignments.csv",
        "config_json": out / "analysis_config.json",
        "warnings_txt": out / "warnings.txt",
        "report_html": out / "report.html",
    }
    results.to_csv(paths["results_csv"], index=False)
    zones.to_csv(paths["zones_csv"], index=False)
    assignments.to_csv(paths["assignments_csv"], index=False)
    try:
        with pd.ExcelWriter(paths["results_xlsx"]) as writer:
            results.to_excel(writer, sheet_name="results_by_cycle", index=False)
            zones.to_excel(writer, sheet_name="thermal_program", index=False)
            assignments.to_excel(writer, sheet_name="zone_assignments", index=False)
            pd.DataFrame([config]).to_excel(writer, sheet_name="parameters", index=False)
            pd.DataFrame({"warning": warnings}).to_excel(writer, sheet_name="warnings", index=False)
    except (ImportError, ModuleNotFoundError, ValueError):
        paths.pop("results_xlsx", None)
    paths["config_json"].write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    paths["warnings_txt"].write_text("\n".join(warnings), encoding="utf-8")
    html = [
        "<html><head><meta charset='utf-8'><title>DSC analysis report</title></head><body>",
        "<h1>DSC analysis report</h1>",
        "<h2>Results</h2>",
        results.to_html(index=False),
        "<h2>Thermal program</h2>",
        zones.to_html(index=False),
        "<h2>Warnings</h2><ul>",
        *[f"<li>{warning}</li>" for warning in warnings],
        "</ul></body></html>",
    ]
    paths["report_html"].write_text("\n".join(html), encoding="utf-8")
    return paths
