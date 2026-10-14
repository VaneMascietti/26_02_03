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
        choices=["sample", "bath", "furnace", "both"],
        default="both",
        help="Temperatura a graficar; both muestra muestra y bano.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Titulo opcional de la figura.",
    )
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


def plot_time_profile(
    df: pd.DataFrame,
    outpath: Path,
    *,
    data_label: str,
    unit: str,
    temperature: str,
    title: str | None,
) -> None:
    _, xlabel = _time_column(df, unit)
    fig, pressure_axis = plt.subplots(figsize=(8, 5.2), constrained_layout=True)
    temperature_axis = pressure_axis.twinx()
    color = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["black"])[0]

    pressure_axis.plot(
        df["plot_time"],
        df["P_muestra"],
        color=color,
        marker=".",
        linestyle="-",
        ms=1.4,
        alpha=0.8,
        label=f"{data_label}: pressure",
        rasterized=True,
    )

    if temperature in {"sample", "both"}:
        temperature_axis.plot(
            df["plot_time"],
            df["T_muestra"],
            color=color,
            linestyle=":",
            lw=1.0,
            alpha=0.6,
            label=f"{data_label}: sample temperature",
            rasterized=True,
        )
    if temperature in {"bath", "both"} and "T_banho" in df.columns:
        bath = df.dropna(subset=["T_banho"])
        temperature_axis.plot(
            bath["plot_time"],
            bath["T_banho"],
            color="#d95f02",
            linestyle="-",
            lw=1.0,
            alpha=0.8,
            label=f"{data_label}: bath temperature",
            rasterized=True,
        )
    if temperature == "furnace" and "T_horno" in df.columns:
        furnace = df.dropna(subset=["T_horno"])
        temperature_axis.plot(
            furnace["plot_time"],
            furnace["T_horno"],
            color="#d95f02",
            linestyle="-.",
            lw=1.0,
            alpha=0.7,
            label=f"{data_label}: bath temperature",
            rasterized=True,
        )

    pressure_axis.set_xlabel(xlabel)
    pressure_axis.set_ylabel("Sample pressure (bar)")
    temperature_axis.set_ylabel("Temperature (C)")
    pressure_axis.grid(True, alpha=0.3)
    handles, labels = pressure_axis.get_legend_handles_labels()
    temp_handles, temp_labels = temperature_axis.get_legend_handles_labels()
    pressure_axis.legend(
        handles + temp_handles,
        labels + temp_labels,
        fontsize=7,
        ncol=2,
        loc="best",
    )

    fig.suptitle(title or "Pressure and temperatures vs time", fontsize=12)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def main() -> int:
    args = build_parser().parse_args()
    data_path = get_data_path(args.data, "lvm")
    if not data_path.is_file():
        raise FileNotFoundError(f"LVM no encontrado: {data_path}")

    df = load_lvm(data_path, program_pdf=None)
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
