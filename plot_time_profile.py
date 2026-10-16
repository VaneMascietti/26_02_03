#!/usr/bin/env python3
"""Plot sample pressure and temperature profiles from a single LVM file."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from util.lvm_io import load_lvm
from util.path import get_data_path, get_output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Graficar P_muestra y temperaturas de muestra/bano vs tiempo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", required=True, help="Archivo .lvm a graficar.")
    parser.add_argument("--out", default=None, help="Ruta de salida PNG.")
    parser.add_argument("--label", default=None, help="Etiqueta para la leyenda.")
    parser.add_argument(
        "--time-unit",
        choices=["h", "min", "s"],
        default="h",
        help="Unidad del eje de tiempo.",
    )
    parser.add_argument("--time-min", type=float, default=None, help="Tiempo minimo a mostrar.")
    parser.add_argument("--time-max", type=float, default=None, help="Tiempo maximo a mostrar.")
    parser.add_argument(
        "--temperature",
        choices=["sample", "bath", "furnace", "both", "all"],
        default="both",
        help="Temperatura a graficar; both muestra muestra y bano.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Titulo opcional de la figura.",
    )
    parser.add_argument("--heat-flow", action="store_true", help="Add a lower heat-flow panel.")
    parser.add_argument("--time-source", choices=["sample", "device"], default="sample",
                        help="Use sample-index or recorded device elapsed time consistently.")
    parser.add_argument("--onset", type=float, action="append", default=[], metavar="TEMP_C",
                        help="Supplied sample-temperature onset; repeat for multiple onsets.")
    parser.add_argument("--onset-direction", choices=["cooling", "heating"], default="cooling")
    parser.add_argument("--onset-time-min", type=float, help="Start of onset search in selected time units.")
    parser.add_argument("--onset-time-max", type=float, help="End of onset search in selected time units.")
    parser.add_argument("--font-size", type=float, default=18, help="Axis label and tick size in points.")
    parser.add_argument("--legend-size", type=float, default=14)
    parser.add_argument("--line-width", type=float, default=1.7)
    for signal, style in [("temperature", "-"), ("pressure", "--"), ("heat-flow", "-")]:
        parser.add_argument(f"--{signal}-style", choices=["-", "--", ":", "-."], default=style)
    return parser


def _time_column(df: pd.DataFrame, unit: str) -> tuple[pd.Series, str]:
    if unit == "s":
        return df["t_s"], "Relative path time (s)"
    if unit == "min":
        return df["t_h"] * 60.0, "Relative path time (min)"
    return df["t_h"], "Relative path time (h)"


def _clean_profile(df: pd.DataFrame, unit: str) -> pd.DataFrame:
    out = df.copy()
    time, _ = _time_column(out, unit)
    out["plot_time"] = pd.to_numeric(time, errors="coerce")
    for col in ["P_muestra", "T_muestra", "T_banho", "T_horno"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.dropna(subset=["plot_time", "P_muestra", "T_muestra"])


def _apply_time_bounds(
    df: pd.DataFrame,
    time_min: float | None,
    time_max: float | None,
) -> pd.DataFrame:
    out = df
    if time_min is not None:
        out = out.loc[out["plot_time"] >= time_min]
    if time_max is not None:
        out = out.loc[out["plot_time"] <= time_max]
    return out


def onset_intersections(df, temperatures, direction="cooling", time_min=None, time_max=None):
    """Interpolate supplied onsets; refuse ambiguous or missing crossings."""
    records = []
    for target in temperatures:
        values = df["T_muestra"].to_numpy(float)
        if direction == "cooling":
            crosses = (values[:-1] >= target) & (values[1:] < target)
        else:
            crosses = (values[:-1] <= target) & (values[1:] > target)
        candidates = []
        for i in np.flatnonzero(crosses):
            left, right = df.iloc[i], df.iloc[i + 1]
            fraction = (target - left.T_muestra) / (right.T_muestra - left.T_muestra)
            record = {key: float(left[key] + fraction * (right[key] - left[key]))
                      for key in ["plot_time", "T_muestra", "T_horno", "T_banho", "P_muestra", "HF_mW"]
                      if key in df.columns}
            if time_min is not None and record["plot_time"] < time_min:
                continue
            if time_max is not None and record["plot_time"] > time_max:
                continue
            candidates.append(record)
        if len(candidates) != 1:
            raise ValueError(f"Onset {target} °C: found {len(candidates)} {direction} crossings; "
                             "select a unique interval with --onset-time-min/--onset-time-max.")
        records.append(candidates[0])
    return records


def plot_time_profile(
    df: pd.DataFrame, outpath: Path, *, data_label: str, unit: str,
    temperature: str, title: str | None, heat_flow=False, onsets=(),
    onset_direction="cooling", onset_time_min=None, onset_time_max=None,
    font_size=18, legend_size=14, line_width=1.7,
    temperature_style="-", pressure_style="--", heat_flow_style="-",
) -> None:
    selected = {
        "sample": ["T_muestra"], "bath": ["T_banho"], "furnace": ["T_horno"],
        "both": ["T_muestra", "T_banho"], "all": ["T_banho", "T_horno", "T_muestra"],
    }[temperature]
    for col in selected + (["HF_mW"] if heat_flow else []):
        if col not in df or not df[col].notna().any():
            raise ValueError(f"Required signal unavailable: {col}")
    records = onset_intersections(df, onsets, onset_direction, onset_time_min, onset_time_max)
    with plt.rc_context({"font.size": font_size}):
        fig, axes = plt.subplots(2 if heat_flow else 1, 1, figsize=(16, 11 if heat_flow else 7),
                                 sharex=True, squeeze=False, constrained_layout=True)
        ax = axes[0, 0]
        pressure = ax.twinx()
        pressure.set_zorder(ax.get_zorder() - 1)
        ax.patch.set_visible(False)
        names = {"T_banho": "Huber temperature", "T_horno": "Furnace temperature", "T_muestra": "Sample temperature"}
        colors = {"T_banho": "#16846b", "T_horno": "#dc8430", "T_muestra": "#285ba6"}
        for col in selected:
            ax.plot(df.plot_time, df[col], label=names[col], color=colors[col],
                    lw=line_width, ls=temperature_style)
        pressure.plot(df.plot_time, df.P_muestra, color="#555555", ls=pressure_style,
                      lw=line_width, label="Sample pressure")
        pressure.set_ylabel("Pressure (bar)")
        ax.set_ylabel("Temperature (°C)")
        ax.margins(y=.35)
        pressure.margins(y=.35)
        if heat_flow:
            hf = axes[1, 0]
            hf.plot(df.plot_time, df.HF_mW, color="#554173", lw=line_width,
                    ls=heat_flow_style, label="Heat flow")
            hf.set_ylabel("Heat flow (mW)")
            hf.legend(loc="upper left", fontsize=legend_size, frameon=False)
        for number, record in enumerate(records, 1):
            t = record["plot_time"]
            color = ["#ab3354", "#6550a1"][(number - 1) % 2]
            for axis in axes[:, 0]:
                axis.axvline(t, color=color, ls="--", lw=1.4)
            lines = [f"Onset {number} | {t:.3f} {unit}"]
            for col, label in [("T_muestra", "Sample"), ("T_horno", "Furnace"), ("T_banho", "Huber")]:
                if col in record:
                    lines.append(f"{label}: {record[col]:.2f} °C")
            lines.append(f"Pressure: {record['P_muestra']:.1f} bar")
            ax.annotate("\n".join(lines), xy=(t, record["T_muestra"]),
                        xytext=(.58 + .22 * ((number-1) % 2), .97 - .32 * ((number-1)//2)),
                        textcoords="axes fraction", va="top", fontsize=legend_size, color=color,
                        bbox=dict(boxstyle="round,pad=.4", fc="white", ec=color),
                        arrowprops=dict(arrowstyle="-", color=color))
            for col in selected:
                ax.scatter(t, record[col], color=colors[col], edgecolor="white", zorder=5)
            pressure.scatter(t, record["P_muestra"], color="#555555", edgecolor="white")
            if heat_flow:
                hf.scatter(t, record["HF_mW"], color=color, zorder=5)
                hf.annotate(f"Onset {number}", (t, record["HF_mW"]), xytext=(-8, 18),
                            textcoords="offset points", ha="right", color=color, fontsize=legend_size)
        for axis in axes[:, 0]:
            axis.grid(alpha=.18)
            axis.tick_params(labelsize=font_size)
        handles, labels = ax.get_legend_handles_labels()
        ph, pl = pressure.get_legend_handles_labels()
        ax.legend(handles + ph, labels + pl, loc="upper left", ncol=1,
                  frameon=False, fontsize=legend_size)
        axes[-1, 0].set_xlabel(f"Time ({unit})")
        ax.set_xlim(df.plot_time.min(), df.plot_time.max())
        fig.suptitle(title or f"{data_label} — Temperatures and pressure" +
                     (" with heat flow" if heat_flow else ""), fontsize=font_size + 4)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=220)
        plt.close(fig)
    if records:
        pd.DataFrame(records).to_csv(outpath.with_name(outpath.stem + "_onsets.csv"), index=False)


def main() -> int:
    args = build_parser().parse_args()
    data_path = get_data_path(args.data, "lvm")
    if not data_path.is_file():
        raise FileNotFoundError(f"LVM no encontrado: {data_path}")

    df = load_lvm(data_path, program_pdf=None)
    if args.time_source == "device":
        df["t_h"] = df["t_s_datetime"] / 3600.0
        df["t_s"] = df["t_s_datetime"]
    else:
        df["t_s"] = df["t_h"] * 3600.0
    clean = _clean_profile(df, args.time_unit)
    clean = _apply_time_bounds(clean, args.time_min, args.time_max)
    if clean.empty:
        raise SystemExit("No hay puntos validos despues de aplicar filtros.")

    outpath = (
        Path(args.out)
        if args.out
        else get_output_dir("figures", "time_profile")
        / f"{data_path.stem}_pressure_temperature_vs_time.png"
    )
    plot_time_profile(
        clean,
        outpath,
        data_label=args.label or data_path.stem,
        unit=args.time_unit,
        temperature=args.temperature,
        title=args.title,
        heat_flow=args.heat_flow, onsets=args.onset, onset_direction=args.onset_direction,
        onset_time_min=args.onset_time_min, onset_time_max=args.onset_time_max,
        font_size=args.font_size, legend_size=args.legend_size, line_width=args.line_width,
        temperature_style=args.temperature_style, pressure_style=args.pressure_style,
        heat_flow_style=args.heat_flow_style,
    )

    print(
        f"{data_path.stem}: points={len(clean)}, "
        f"t=[{clean['plot_time'].min():.3f},{clean['plot_time'].max():.3f}] {args.time_unit}, "
        f"T_sample=[{clean['T_muestra'].min():.3f},{clean['T_muestra'].max():.3f}] C, "
        f"P=[{clean['P_muestra'].min():.3f},{clean['P_muestra'].max():.3f}] bar"
    )
    print(f"Guardado: {outpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
