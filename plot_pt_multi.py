#!/usr/bin/env python3
"""Plot multiple P-T tracks from the 26_02_03 LVM parser."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_PROJECT = Path("/home/andrea/26_02_03")
PROJECT = DEFAULT_PROJECT
sys.path.insert(0, str(PROJECT))

from process_dsc_data import load_lvm  # noqa: E402
from util.path import get_data_path  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot several P-T paths together.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--track", nargs=3, action="append", required=True,
                        metavar=("LVM", "PDF", "LABEL"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--p-min", type=float, default=None)
    parser.add_argument("--p-max", type=float, default=None)
    parser.add_argument("--t-min", type=float, default=None)
    parser.add_argument("--t-max", type=float, default=None)
    parser.add_argument("--include-time", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    project = args.project.resolve()
    os.chdir(project)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.include_time:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        ax_pt, ax_time = axes
    else:
        fig, ax_pt = plt.subplots(figsize=(7, 5), constrained_layout=True)
        ax_time = None

    for lvm_name, pdf_name, label in args.track:
        lvm_path = get_data_path(lvm_name, "lvm")
        pdf_path = get_data_path(pdf_name, "pdf")
        if not lvm_path.is_file():
            raise FileNotFoundError(f"LVM not found: {lvm_path}")
        if not pdf_path.is_file():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        df = load_lvm(lvm_path, program_pdf=None)
        valid = (
            np.isfinite(df["T_muestra"])
            & np.isfinite(df["P_muestra"])
            & np.isfinite(df["t_h"])
        )
        if args.p_min is not None:
            valid &= df["P_muestra"] >= args.p_min
        if args.p_max is not None:
            valid &= df["P_muestra"] <= args.p_max
        if args.t_min is not None:
            valid &= df["T_muestra"] >= args.t_min
        if args.t_max is not None:
            valid &= df["T_muestra"] <= args.t_max
        data = df.loc[valid]
        if data.empty:
            print(f"{label}: no points after filters")
            continue

        ax_pt.scatter(
            data["T_muestra"],
            data["P_muestra"],
            s=0.6,
            alpha=0.6,
            label=label,
            rasterized=True,
        )
        if ax_time is not None:
            ax_time.scatter(
                data["t_h"],
                data["P_muestra"],
                s=0.6,
                alpha=0.6,
                label=label,
                rasterized=True,
            )
        print(
            f"{label}: points={len(data)}, "
            f"T=[{data['T_muestra'].min():.3f},{data['T_muestra'].max():.3f}] C, "
            f"P=[{data['P_muestra'].min():.3f},{data['P_muestra'].max():.3f}] bar"
        )

    ax_pt.set_title("P vs T")
    ax_pt.set_xlabel("T muestra (C)")
    ax_pt.set_ylabel("P muestra (bar)")
    ax_pt.grid(True, alpha=0.3)
    ax_pt.legend(markerscale=7, fontsize=8)

    if ax_time is not None:
        ax_time.set_title("P vs tiempo")
        ax_time.set_xlabel("Tiempo desde inicio (h)")
        ax_time.set_ylabel("P muestra (bar)")
        ax_time.grid(True, alpha=0.3)
        ax_time.legend(markerscale=7, fontsize=8)

    title_bits = []
    if args.p_min is not None or args.p_max is not None:
        p_min = args.p_min if args.p_min is not None else "-inf"
        p_max = args.p_max if args.p_max is not None else "inf"
        title_bits.append(f"P filter: {p_min}-{p_max} bar")
    if title_bits:
        fig.suptitle(" | ".join(title_bits), fontsize=10)

    fig.savefig(out, dpi=220)
    plt.close(fig)
    print(f"Guardado: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
