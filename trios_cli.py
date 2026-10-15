"""Plot cooling ramps exported from TA Instruments TRIOS as legacy .xls files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.trios_parser import TriosSegment, read_trios_xls
from src.trios_onset import TriosOnset, first_exothermic_onset


BLUE = "#2166ac"
GRAY = "#777777"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot cooling ramps from a TRIOS .xls export.")
    parser.add_argument("--xls", required=True, help="TRIOS legacy Excel export")
    parser.add_argument("--outdir", default="out/trios_analysis", help="Output directory")
    parser.add_argument("--sheet", help="Optional exact sheet name to plot")
    return parser


def _cooling_arrow(ax: plt.Axes) -> None:
    ax.annotate(
        "Cooling",
        xy=(0.80, 0.94),
        xytext=(0.94, 0.94),
        xycoords="axes fraction",
        ha="center",
        va="center",
        fontsize=14,
        arrowprops={"arrowstyle": "->", "linewidth": 1.5, "color": "black"},
    )


def _time_cooling_arrow(ax: plt.Axes) -> None:
    ax.annotate(
        "Cooling",
        xy=(0.94, 0.94),
        xytext=(0.80, 0.94),
        xycoords="axes fraction",
        ha="center",
        va="center",
        fontsize=14,
        arrowprops={"arrowstyle": "->", "linewidth": 1.5, "color": "black"},
    )


def _dual_axis_arrows(temperature_ax: plt.Axes, heat_flow_ax: plt.Axes) -> None:
    """Connect each time-series curve visually to its corresponding Y axis."""

    temperature_ax.annotate(
        "",
        xy=(0.002, 0.86),
        xytext=(0.105, 0.86),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "->", "linewidth": 1.5, "color": GRAY},
    )
    heat_flow_ax.annotate(
        "",
        xy=(0.998, 0.105),
        xytext=(0.875, 0.050),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "->", "linewidth": 1.5, "color": "black"},
    )


def plot_cooling(segment: TriosSegment, out: Path, sample_name: str) -> Path:
    frame = segment.frame
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    ax.plot(frame["temperature_c"], frame["heat_flow_w_g"], color="black", linewidth=0.9)
    ax.set_xlim(float(frame["temperature_c"].min()), float(frame["temperature_c"].max()))
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel(f"Normalized heat flow ({segment.heat_flow_unit})")
    ax.set_title(sample_name)
    ax.text(0.03, 0.07, "Exo ↑", transform=ax.transAxes, fontsize=14)
    _cooling_arrow(ax)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.grid(False)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=600)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)
    return out


def plot_cooling_vs_time(
    segment: TriosSegment,
    out: Path,
    sample_name: str,
    onset: TriosOnset | None = None,
) -> Path:
    """Plot temperature and normalized heat flow against elapsed cooling time."""

    frame = segment.frame
    elapsed_time = frame["time_min"] - float(frame["time_min"].iloc[0])
    fig, temperature_ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    heat_flow_ax = temperature_ax.twinx()

    temperature_line = temperature_ax.plot(
        elapsed_time,
        frame["temperature_c"],
        color=GRAY,
        linewidth=1.1,
        label="Temperature",
    )[0]
    heat_flow_line = heat_flow_ax.plot(
        elapsed_time,
        frame["heat_flow_w_g"],
        color="black",
        linewidth=0.9,
        label="Normalized heat flow",
    )[0]

    temperature_ax.set_xlim(float(elapsed_time.min()), float(elapsed_time.max()))
    temperature_ax.set_xlabel("Cooling time (min)")
    temperature_ax.set_ylabel("Temperature (°C)", color="black")
    heat_flow_ax.set_ylabel(f"Normalized heat flow ({segment.heat_flow_unit})", color="black")
    temperature_ax.tick_params(axis="y", colors="black")
    heat_flow_ax.tick_params(axis="y", colors="black")
    temperature_ax.spines["left"].set_color("black")
    heat_flow_ax.spines["right"].set_color("black")
    temperature_ax.set_title(sample_name)
    heat_flow_ax.text(0.03, 0.07, "Exo ↑", transform=heat_flow_ax.transAxes, color="black", fontsize=14)
    _time_cooling_arrow(temperature_ax)
    _dual_axis_arrows(temperature_ax, heat_flow_ax)
    if onset is not None:
        heat_flow_ax.axvline(
            onset.onset_elapsed_min,
            color=BLUE,
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
        )
        heat_flow_ax.scatter(
            [onset.onset_elapsed_min],
            [onset.onset_heat_flow_w_g],
            s=55,
            facecolor="white",
            edgecolor=BLUE,
            linewidth=1.8,
            zorder=6,
        )
        heat_flow_ax.annotate(
            (
                rf"$t_{{onset}}$ = {onset.onset_elapsed_min:.2f} min"
                "\n"
                rf"$T_{{onset}}$ = {onset.onset_temperature_c:.3f} °C"
            ),
            xy=(onset.onset_elapsed_min, onset.onset_heat_flow_w_g),
            xytext=(0.63, 0.18),
            textcoords="axes fraction",
            color=BLUE,
            fontsize=12,
            ha="left",
            va="center",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 2.0},
            arrowprops={"arrowstyle": "->", "color": BLUE, "linewidth": 1.1},
        )
    temperature_ax.grid(False)
    heat_flow_ax.grid(False)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=600)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)
    return out


def analyze_trios(xls: str | Path, outdir: str | Path, sheet_name: str | None = None) -> dict:
    metadata, segments = read_trios_xls(xls)
    if sheet_name:
        selected = [segment for segment in segments if segment.sheet_name == sheet_name]
        if not selected:
            available = ", ".join(segment.sheet_name for segment in segments)
            raise ValueError(f"Sheet {sheet_name!r} was not found. Available data sheets: {available}")
        if selected[0].direction != "cooling":
            raise ValueError(f"Sheet {sheet_name!r} is classified as {selected[0].direction}, not cooling")
    else:
        selected = [segment for segment in segments if segment.direction == "cooling"]
    if not selected:
        raise ValueError("No cooling ramp was detected in the TRIOS workbook")

    out = Path(outdir)
    figures = out / "figures"
    tables = out / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    sample_name = str(metadata.get("Sample name") or metadata.get("Filename") or Path(xls).stem)
    rows: list[dict] = []
    onset_rows: list[dict] = []
    for index, segment in enumerate(selected, start=1):
        data_path = tables / f"cooling_{index:02d}.csv"
        temperature_figure_path = figures / f"cooling_{index:02d}_vs_temperature.png"
        time_figure_path = figures / f"cooling_{index:02d}_vs_time.png"
        segment.frame.to_csv(data_path, index=False)
        onset = first_exothermic_onset(segment)
        plot_cooling(segment, temperature_figure_path, sample_name)
        plot_cooling_vs_time(segment, time_figure_path, sample_name, onset)
        row = segment.summary()
        row.update(
            {
                "data_csv": str(data_path),
                "figure_vs_temperature": str(temperature_figure_path),
                "figure_vs_time": str(time_figure_path),
            }
        )
        rows.append(row)
        if onset is not None:
            onset_row = {"sheet": segment.sheet_name, **onset.as_dict()}
            onset_rows.append(onset_row)

    summary = pd.DataFrame(rows)
    summary_path = out / "trios_segments.csv"
    details_path = out / "trios_details.json"
    onsets_path = out / "trios_onsets.csv"
    summary.to_csv(summary_path, index=False)
    pd.DataFrame(onset_rows).to_csv(onsets_path, index=False)
    details_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return {
        "summary": summary,
        "summary_csv": summary_path,
        "details_json": details_path,
        "onsets_csv": onsets_path,
        "figures_dir": figures,
        "tables_dir": tables,
    }


def main() -> None:
    args = build_parser().parse_args()
    output = analyze_trios(args.xls, args.outdir, args.sheet)
    print(output["summary"].to_string(index=False))
    print(f"Outputs written to {Path(args.outdir).resolve()}")


if __name__ == "__main__":
    main()
