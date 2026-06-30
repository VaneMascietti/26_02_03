#!/usr/bin/env python3
"""Compare cooling portions of several LVM experiments.

The input CSV defines the experiments. The script generates three plots for the
same selected cooling data:

    HF vs temperature
    HF vs pressure
    pressure vs temperature
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_PROJECT = Path("/home/andrea/26_02_03")
PROJECT = DEFAULT_PROJECT
sys.path.insert(0, str(PROJECT))

from process_dsc_data import load_lvm, program_zone_spans  # noqa: E402


DEFAULT_SET = Path("data/plot_sets/cooling_4exp.csv")
DEFAULT_FIGDIR = Path("out/figures/cooling_comparison")
DEFAULT_TABLEDIR = Path("out/tables/cooling_comparison")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot HF-T, HF-P and P-T for cooling portions of several LVM files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--set", dest="set_path", type=Path, default=DEFAULT_SET)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--out-prefix", default="cooling_4exp")
    parser.add_argument("--figdir", type=Path, default=DEFAULT_FIGDIR)
    parser.add_argument("--tabledir", type=Path, default=DEFAULT_TABLEDIR)
    parser.add_argument("--hf-col", default="HF_mW")
    parser.add_argument("--temp-col", default="T_muestra")
    parser.add_argument("--pressure-col", default="P_muestra")
    parser.add_argument("--selection", choices=["program-first-cooling", "cooling-rate", "all"],
                        default="program-first-cooling",
                        help="How to select the data interval")
    parser.add_argument("--cooling-rate-threshold", type=float, default=-0.02,
                        help="Keep points with dT/dt below this value in C/h")
    parser.add_argument("--smooth-points", type=int, default=301,
                        help="Centered rolling window for cooling detection")
    parser.add_argument("--table-stride", type=int, default=50,
                        help="Save one every N selected points in the processed CSV")
    parser.add_argument("--all-points", action="store_true",
                        help="Alias for --selection all")
    parser.add_argument("--p-min", type=float, default=None)
    parser.add_argument("--p-max", type=float, default=None)
    parser.add_argument("--t-min", type=float, default=None)
    parser.add_argument("--t-max", type=float, default=None)
    parser.add_argument("--allow-missing", action="store_true", default=True,
                        help="Skip missing files instead of failing")
    return parser


def resolve_path(project: Path, value: str) -> Path:
    path = Path(str(value).strip())
    if path.is_absolute():
        return path
    candidate = project / path
    if candidate.exists():
        return candidate
    if path.suffix.lower() != ".lvm":
        candidate = project / "data" / "raw" / "lvm" / f"{path}.lvm"
        if candidate.exists():
            return candidate
    return project / path


def infer_program_pdf(project: Path, lvm_path: Path) -> Path:
    raw_pdf = project / "data" / "raw" / "pdf" / f"{lvm_path.stem}.pdf"
    if raw_pdf.exists():
        return raw_pdf
    return lvm_path.with_suffix(".pdf")


def load_set(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Set CSV not found: {path}")
    df = pd.read_csv(path)
    required = {"ID", "file", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in set CSV: {sorted(missing)}")
    return df


def first_cooling_span_h(program_pdf: Path) -> tuple[float, float] | None:
    spans = program_zone_spans(program_pdf)
    for span in spans:
        if "cool" in str(span["type"]).lower():
            return float(span["start_h"]), float(span["end_h"])
    return None


def first_heating_span_h(program_pdf: Path) -> tuple[float, float] | None:
    spans = program_zone_spans(program_pdf)
    for span in spans:
        ztype = str(span["type"]).lower()
        if "heat" in ztype or "inicio" in ztype:
            return float(span["start_h"]), float(span["end_h"])
    return None


def initial_heating_path_span_h(program_pdf: Path) -> tuple[float, float] | None:
    """Return the full initial heating path around the PDF's Inicio zone.

    These programs label the 20 C hold as Isoterma, the first ramp as Inicio,
    and the continued ramp/high-temperature segment as Isoterma. For the P-T
    path we want the whole initial path, matching the historical plots.
    """
    spans = program_zone_spans(program_pdf)
    for idx, span in enumerate(spans):
        ztype = str(span["type"]).lower()
        if "heat" not in ztype and "inicio" not in ztype:
            continue
        start_h = spans[idx - 1]["start_h"] if idx > 0 else span["start_h"]
        end_h = spans[idx + 1]["end_h"] if idx + 1 < len(spans) else span["end_h"]
        return float(start_h), float(end_h)
    return first_heating_span_h(program_pdf)


def rolling_temperature_rate(t_h: pd.Series, temp: pd.Series, window: int) -> pd.Series:
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    temp_smooth = temp.rolling(window, center=True, min_periods=max(3, window // 5)).median()
    rate = np.gradient(temp_smooth.to_numpy(dtype=float), t_h.to_numpy(dtype=float))
    return pd.Series(rate, index=temp.index)


def select_data(
    df: pd.DataFrame,
    temp_col: str,
    pressure_col: str,
    hf_col: str,
    selection: str,
    cooling_rate_threshold: float,
    smooth_points: int,
    program_pdf: Path | None,
    t_min: float | None,
    t_max: float | None,
    p_min: float | None,
    p_max: float | None,
) -> pd.DataFrame:
    required = [temp_col, pressure_col, hf_col, "t_h"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in LVM data: {missing}")

    out = df.copy()
    for col in [temp_col, pressure_col, hf_col, "t_h"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    valid = out[[temp_col, pressure_col, hf_col, "t_h"]].notna().all(axis=1)
    if t_min is not None:
        valid &= out[temp_col] >= t_min
    if t_max is not None:
        valid &= out[temp_col] <= t_max
    if p_min is not None:
        valid &= out[pressure_col] >= p_min
    if p_max is not None:
        valid &= out[pressure_col] <= p_max

    out = out.loc[valid].copy()
    if out.empty or selection == "all":
        return out

    if selection == "program-first-cooling":
        if program_pdf is None or not program_pdf.is_file():
            raise FileNotFoundError(f"Program PDF not found: {program_pdf}")
        span = first_cooling_span_h(program_pdf)
        if span is None:
            raise ValueError(f"No Cooling zone found in program PDF: {program_pdf}")
        start_h, end_h = span
        selected = out[(out["t_h"] >= start_h) & (out["t_h"] <= end_h)].copy()
        selected["program_start_h"] = start_h
        selected["program_end_h"] = end_h
        return selected

    out["dTdt_C_per_h"] = rolling_temperature_rate(out["t_h"], out[temp_col], smooth_points)
    cooling = out["dTdt_C_per_h"] <= cooling_rate_threshold
    return out.loc[cooling].copy()


def plot_xy(
    datasets: list[pd.DataFrame],
    x_col: str,
    y_col: str,
    xlabel: str,
    ylabel: str,
    title: str,
    outpath: Path,
    invert_x: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.2), constrained_layout=True)
    for data in datasets:
        ax.plot(
            data[x_col],
            data[y_col],
            ".",
            ms=1.4,
            alpha=0.55,
            label=data["label"].iloc[0],
            rasterized=True,
        )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(markerscale=5, fontsize=8)
    if invert_x:
        ax.invert_xaxis()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def select_time_span(df: pd.DataFrame, start_h: float, end_h: float) -> pd.DataFrame:
    return df[(df["t_h"] >= start_h) & (df["t_h"] <= end_h)].copy()


def main() -> int:
    args = build_parser().parse_args()
    project = args.project.resolve()
    os.chdir(project)

    set_path = args.set_path if args.set_path.is_absolute() else project / args.set_path
    figdir = args.figdir if args.figdir.is_absolute() else project / args.figdir
    tabledir = args.tabledir if args.tabledir.is_absolute() else project / args.tabledir
    figdir.mkdir(parents=True, exist_ok=True)
    tabledir.mkdir(parents=True, exist_ok=True)

    config = load_set(set_path)
    datasets: list[pd.DataFrame] = []
    pt_datasets: list[pd.DataFrame] = []
    summaries = []

    for _, row in config.iterrows():
        lvm_path = resolve_path(project, row["file"])
        program_pdf = (
            resolve_path(project, row["program_pdf"])
            if "program_pdf" in row and pd.notna(row.get("program_pdf"))
            else infer_program_pdf(project, lvm_path)
        )
        label = str(row["label"])
        exp_id = str(row["ID"])
        if not lvm_path.is_file():
            message = f"[WARN] {label}: LVM not found: {lvm_path}"
            if args.allow_missing:
                print(message)
                continue
            raise FileNotFoundError(message)

        raw = load_lvm(lvm_path, program_pdf=None)
        selection = "all" if args.all_points else args.selection
        data = select_data(
            raw,
            temp_col=args.temp_col,
            pressure_col=args.pressure_col,
            hf_col=args.hf_col,
            selection=selection,
            cooling_rate_threshold=args.cooling_rate_threshold,
            smooth_points=args.smooth_points,
            program_pdf=program_pdf,
            t_min=args.t_min,
            t_max=args.t_max,
            p_min=args.p_min,
            p_max=args.p_max,
        )
        if data.empty:
            print(f"[WARN] {label}: no points after filters")
            continue

        optional_cols = [
            col for col in ["dTdt_C_per_h", "program_start_h", "program_end_h"]
            if col in data.columns
        ]
        data = data[[args.temp_col, args.pressure_col, args.hf_col, "t_h"] + optional_cols].copy()
        data["ID"] = exp_id
        data["label"] = label
        data["file"] = str(lvm_path)
        data["program_pdf"] = str(program_pdf)
        datasets.append(data)

        pt_data = data.copy()
        if selection == "program-first-cooling":
            heating_lvm_path = (
                resolve_path(project, row["heating_file"])
                if "heating_file" in row and pd.notna(row.get("heating_file"))
                else lvm_path
            )
            heating_program_pdf = (
                resolve_path(project, row["heating_program_pdf"])
                if "heating_program_pdf" in row and pd.notna(row.get("heating_program_pdf"))
                else infer_program_pdf(project, heating_lvm_path)
            )
            heating_span = initial_heating_path_span_h(heating_program_pdf)
            if not heating_lvm_path.is_file():
                print(f"[WARN] {label}: heating LVM not found: {heating_lvm_path}")
            elif heating_span is not None:
                heating_raw = load_lvm(heating_lvm_path, program_pdf=None)
                heating = select_time_span(heating_raw, *heating_span)
                for col in [args.temp_col, args.pressure_col, args.hf_col, "t_h"]:
                    heating[col] = pd.to_numeric(heating[col], errors="coerce")
                valid_heating = heating[[args.temp_col, args.pressure_col, args.hf_col, "t_h"]].notna().all(axis=1)
                if args.t_min is not None:
                    valid_heating &= heating[args.temp_col] >= args.t_min
                if args.t_max is not None:
                    valid_heating &= heating[args.temp_col] <= args.t_max
                if args.p_min is not None:
                    valid_heating &= heating[args.pressure_col] >= args.p_min
                if args.p_max is not None:
                    valid_heating &= heating[args.pressure_col] <= args.p_max
                heating = heating.loc[valid_heating, [args.temp_col, args.pressure_col, args.hf_col, "t_h"]].copy()
                heating["program_start_h"] = heating_span[0]
                heating["program_end_h"] = heating_span[1]
                heating["ID"] = exp_id
                heating["label"] = f"{label} heating"
                heating["file"] = str(heating_lvm_path)
                heating["program_pdf"] = str(heating_program_pdf)
                pt_data = pd.concat([pt_data, heating], ignore_index=True, sort=False)
            else:
                print(f"[WARN] {label}: no initial heating path found in {heating_program_pdf}")
        pt_datasets.append(pt_data)

        summaries.append({
            "ID": exp_id,
            "label": label,
            "file": str(lvm_path),
            "program_pdf": str(program_pdf),
            "selection": selection,
            "program_start_h": data["program_start_h"].iloc[0] if "program_start_h" in data.columns else "",
            "program_end_h": data["program_end_h"].iloc[0] if "program_end_h" in data.columns else "",
            "n_points": len(data),
            "T_min_C": data[args.temp_col].min(),
            "T_max_C": data[args.temp_col].max(),
            "P_min_bar": data[args.pressure_col].min(),
            "P_max_bar": data[args.pressure_col].max(),
            "HF_min_mW": data[args.hf_col].min(),
            "HF_max_mW": data[args.hf_col].max(),
        })
        print(
            f"{label}: points={len(data)}, "
            f"T=[{data[args.temp_col].min():.3f},{data[args.temp_col].max():.3f}] C, "
            f"P=[{data[args.pressure_col].min():.3f},{data[args.pressure_col].max():.3f}] bar, "
            f"HF=[{data[args.hf_col].min():.4g},{data[args.hf_col].max():.4g}] mW"
        )

    if not datasets:
        raise SystemExit("No datasets to plot.")

    combined = pd.concat(datasets, ignore_index=True, sort=False)
    table_stride = max(1, int(args.table_stride))
    combined_for_table = pd.concat(
        [group.iloc[::table_stride] for _, group in combined.groupby("label", sort=False)],
        ignore_index=True,
        sort=False,
    )
    combined_path = tabledir / f"{args.out_prefix}_processed.csv"
    summary_path = tabledir / f"{args.out_prefix}_summary.csv"
    combined_for_table.to_csv(combined_path, index=False)
    pd.DataFrame(summaries).to_csv(summary_path, index=False)

    plot_xy(
        datasets,
        x_col=args.temp_col,
        y_col=args.hf_col,
        xlabel="Sample temperature (C)",
        ylabel=f"{args.hf_col} (mW)",
        title="Heat flow vs temperature during first cooling, β = 0.05 K/min",
        outpath=figdir / f"{args.out_prefix}_hf_vs_temperature.png",
        invert_x=True,
    )
    plot_xy(
        datasets,
        x_col=args.pressure_col,
        y_col=args.hf_col,
        xlabel="Sample pressure (bar)",
        ylabel=f"{args.hf_col} (mW)",
        title="Heat flow vs pressure during first cooling, β = 0.05 K/min",
        outpath=figdir / f"{args.out_prefix}_hf_vs_pressure.png",
    )
    plot_xy(
        pt_datasets,
        x_col=args.temp_col,
        y_col=args.pressure_col,
        xlabel="Sample temperature (C)",
        ylabel="Sample pressure (bar)",
        title="Pressure vs temperature: first cooling and initial heating",
        outpath=figdir / f"{args.out_prefix}_pressure_vs_temperature.png",
        invert_x=False,
    )

    print(f"Saved table: {combined_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved figures in: {figdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
