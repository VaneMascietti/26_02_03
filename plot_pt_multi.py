#!/usr/bin/env python3
"""Plot multiple P-T tracks from the 26_02_03 LVM parser."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")


DEFAULT_PROJECT = Path("/home/andrea/26_02_03")
PROJECT = DEFAULT_PROJECT
sys.path.insert(0, str(PROJECT))

from util.data_filters import filter_pt_rows  # noqa: E402
from util.lvm_io import load_lvm  # noqa: E402
from util.path import get_data_path  # noqa: E402
from util.pt_plot import plot_pt_tracks  # noqa: E402


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

    tracks = []
    for lvm_name, pdf_name, label in args.track:
        lvm_path = get_data_path(lvm_name, "lvm")
        pdf_path = get_data_path(pdf_name, "pdf")
        if not lvm_path.is_file():
            raise FileNotFoundError(f"LVM not found: {lvm_path}")
        if not pdf_path.is_file():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        df = load_lvm(lvm_path, program_pdf=None)
        data = filter_pt_rows(
            df,
            t_bounds=(args.t_min, args.t_max),
            p_bounds=(args.p_min, args.p_max),
        )
        if data.empty:
            print(f"{label}: no points after filters")
            continue
        tracks.append((data, label))

    title_bits = []
    if args.p_min is not None or args.p_max is not None:
        p_min = args.p_min if args.p_min is not None else "-inf"
        p_max = args.p_max if args.p_max is not None else "inf"
        title_bits.append(f"P filter: {p_min}-{p_max} bar")
    plot_pt_tracks(
        tracks,
        out,
        include_time=args.include_time,
        point_size=0.6,
        alpha=0.6,
        markerscale=7,
        xlabel="T muestra (C)",
        ylabel="P muestra (bar)",
        time_xlabel="Tiempo desde inicio (h)",
        time_ylabel="P muestra (bar)",
        positive_pressure=False,
        print_time_range=False,
        empty_message="no points after filters",
        suptitle=" | ".join(title_bits) if title_bits else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
