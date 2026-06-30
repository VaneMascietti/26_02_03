#!/usr/bin/env python3
"""Plot curated phase-transition points in P-T space.

Input is an editable CSV, and an optional literature CSV can be appended
without touching the plotting code. Expected columns:

    ID, transition_type, T_C, P_bar, source

Extra columns such as Ciclo, xCO2, DOI, Referencia and Metodo are preserved.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_INPUT = Path("data/phase_transitions/experimental_points.csv")
DEFAULT_LITERATURE = Path("data/phase_transitions/literature_points.csv")
DEFAULT_OUTDIR = Path("out/figures/phase_transitions")

TRANSITION_MARKERS = {
    "TLL": "o",
    "TSL": "s",
    "TIAC": "s",
    "secondary": "^",
    "LLV": "v",
    "SLL": "P",
    "SL": "X",
    "literature": "D",
}

LABEL_OFFSETS = [
    (7, 7),
    (7, -13),
    (-36, 8),
    (-36, -14),
    (12, 16),
    (-46, 16),
    (12, -22),
    (-46, -22),
]

ID_LABEL_OFFSETS = [
    (16, 10),
    (-44, -16),
    (14, 18),
    (16, -12),
    (-48, 8),
    (-48, -14),
]

ID_LABEL_OFFSETS_BY_ID = {
    "E05": (18, 18),
    "E06": (-48, -18),
    "E07": (22, -2),
    "E18": (28, -10),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot phase-transition points, separated from full P-T paths.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--literature", type=Path, default=None,
                        help="Optional literature CSV to append")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--out", default="phase_transition_points.png")
    parser.add_argument("--ids", nargs="*", default=None,
                        help="Optional subset of IDs, e.g. --ids E05 E06 E18")
    parser.add_argument("--exclude-ids", nargs="*", default=None,
                        help="IDs to remove after other filters, e.g. --exclude-ids E05")
    parser.add_argument("--cycles", nargs="*", default=None,
                        help="Optional subset of experimental cycles, e.g. --cycles C1")
    parser.add_argument("--transition-types", nargs="*", default=None,
                        help="Optional subset of transition types, e.g. --transition-types TLL")
    parser.add_argument("--xco2-values", nargs="*", type=float, default=None,
                        help="Optional subset of literature/experimental xCO2 values")
    parser.add_argument("--xco2-tol", type=float, default=5e-4,
                        help="Tolerance used by --xco2-values")
    parser.add_argument("--include-secondary", action="store_true")
    parser.add_argument("--include-literature", action="store_true")
    parser.add_argument("--label-points", action="store_true")
    parser.add_argument("--label-experimental-only", action="store_true",
                        help="With --label-points, label only non-literature points")
    parser.add_argument("--label-ids-once", action="store_true",
                        help="With --label-points, add one label per ID instead of one per point")
    parser.add_argument("--label-xco2", action="store_true",
                        help="Append molar composition xCO2 to point labels")
    parser.add_argument("--color-by-xco2", action="store_true",
                        help="Use xCO2 as color scale instead of color by experiment ID")
    parser.add_argument("--same-style-by-source", action="store_true",
                        help="Plot literature and experimental points with the same filled style")
    return parser


def load_points(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    df = pd.read_csv(path)
    required = {"ID", "transition_type", "T_C", "P_bar"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df["T_C"] = pd.to_numeric(df["T_C"], errors="coerce")
    df["P_bar"] = pd.to_numeric(df["P_bar"], errors="coerce")
    if "xCO2" in df.columns:
        df["xCO2"] = pd.to_numeric(df["xCO2"], errors="coerce")
    if "Ciclo" not in df.columns:
        df["Ciclo"] = ""
    df["transition_type"] = df["transition_type"].astype(str)
    return df.dropna(subset=["T_C", "P_bar"]).copy()


def combine_inputs(
    input_path: Path,
    literature_path: Path | None,
    include_literature: bool,
) -> pd.DataFrame:
    frames = [load_points(input_path)]
    if include_literature:
        lit_path = literature_path or DEFAULT_LITERATURE
        if not lit_path.is_file():
            raise FileNotFoundError(f"Literature CSV not found: {lit_path}")
        frames.append(load_points(lit_path))
    return pd.concat(frames, ignore_index=True, sort=False)


def filter_points(
    df: pd.DataFrame,
    ids: list[str] | None,
    exclude_ids: list[str] | None,
    cycles: list[str] | None,
    transition_types: list[str] | None,
    xco2_values: list[float] | None,
    xco2_tol: float,
    include_secondary: bool,
    include_literature: bool,
) -> pd.DataFrame:
    out = df.copy()
    source = out.get("source", pd.Series("", index=out.index)).astype(str).str.lower()
    if ids:
        keep_ids = out["ID"].isin(ids)
        keep_literature = include_literature & (source == "literature")
        out = out[keep_ids | keep_literature]
        source = out.get("source", pd.Series("", index=out.index)).astype(str).str.lower()
    if exclude_ids:
        out = out[~out["ID"].isin(exclude_ids)]
        source = out.get("source", pd.Series("", index=out.index)).astype(str).str.lower()
    if cycles:
        cycles_set = set(cycles)
        is_literature = source == "literature"
        out = out[out["Ciclo"].isin(cycles_set) | is_literature]
        source = out.get("source", pd.Series("", index=out.index)).astype(str).str.lower()
    if transition_types:
        transitions_set = {item.lower() for item in transition_types}
        out = out[out["transition_type"].str.lower().isin(transitions_set)]
    if xco2_values:
        if "xCO2" not in out.columns:
            raise ValueError("--xco2-values requires an xCO2 column")
        keep_xco2 = np.zeros(len(out), dtype=bool)
        xco2 = out["xCO2"].to_numpy(dtype=float)
        for value in xco2_values:
            keep_xco2 |= np.isclose(xco2, value, atol=xco2_tol, rtol=0.0)
        source = out.get("source", pd.Series("", index=out.index)).astype(str).str.lower()
        out = out[keep_xco2 | (source != "literature")]
    if not include_secondary:
        out = out[out["transition_type"].str.lower() != "secondary"]
    if not include_literature:
        source = out.get("source", pd.Series("", index=out.index)).astype(str).str.lower()
        out = out[source != "literature"]
    return out


def point_label(row: pd.Series, include_xco2: bool) -> str:
    cycle = str(row.get("Ciclo", "")).strip()
    label = f"{row['ID']} {cycle}".strip()
    if include_xco2 and "xCO2" in row and pd.notna(row["xCO2"]):
        label += f"\nxCO2={float(row['xCO2']):.4f}"
    return label


def plot_points(
    df: pd.DataFrame,
    outpath: Path,
    label_points: bool,
    label_experimental_only: bool,
    label_ids_once: bool,
    label_xco2: bool,
    color_by_xco2: bool,
    same_style_by_source: bool,
) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)

    ids = list(dict.fromkeys(df["ID"].astype(str)))
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["black"])
    color_by_id = {exp_id: colors[i % len(colors)] for i, exp_id in enumerate(ids)}
    scatter_for_colorbar = None
    if color_by_xco2:
        if "xCO2" not in df.columns or df["xCO2"].dropna().empty:
            raise ValueError("--color-by-xco2 requires numeric xCO2 values in the input CSV")
        xco2_min = float(df["xCO2"].min())
        xco2_max = float(df["xCO2"].max())
        if np.isclose(xco2_min, xco2_max):
            xco2_min -= 0.001
            xco2_max += 0.001

    label_count = 0
    rows_to_label = []
    for _, row in df.iterrows():
        exp_id = str(row["ID"])
        transition = str(row["transition_type"])
        marker = TRANSITION_MARKERS.get(transition, "o")
        source = str(row.get("source", ""))
        is_literature = source.lower() == "literature"
        if color_by_xco2:
            color = float(row["xCO2"]) if pd.notna(row.get("xCO2")) else np.nan
            color_kwargs = {
                "c": [color],
                "cmap": "viridis",
                "vmin": xco2_min,
                "vmax": xco2_max,
            }
        else:
            color_kwargs = {"color": color_by_id[exp_id]}

        if is_literature and not same_style_by_source:
            scatter_for_colorbar = ax.scatter(
                row["T_C"],
                row["P_bar"],
                marker=marker,
                s=30,
                alpha=0.42,
                linewidths=0.7,
                edgecolors="none",
                **color_kwargs,
            )
        elif "estimado" in source.lower():
            edgecolor = plt.get_cmap("viridis")(
                (float(row["xCO2"]) - xco2_min) / (xco2_max - xco2_min)
            ) if color_by_xco2 and pd.notna(row.get("xCO2")) else color_by_id[exp_id]
            ax.scatter(
                row["T_C"],
                row["P_bar"],
                marker=marker,
                s=70,
                facecolors="none",
                edgecolors=edgecolor,
                linewidths=2.0,
            )
        else:
            scatter_for_colorbar = ax.scatter(
                row["T_C"],
                row["P_bar"],
                marker=marker,
                s=62 if same_style_by_source else 68,
                edgecolors="none" if same_style_by_source else "black",
                linewidths=0.0 if same_style_by_source else 1.0,
                **color_kwargs,
            )

        if label_points and label_ids_once and not (label_experimental_only and is_literature):
            rows_to_label.append(row)
        elif label_points and not (label_experimental_only and is_literature):
            label = point_label(row, include_xco2=label_xco2)
            label_color = color_by_id[exp_id]
            offset = LABEL_OFFSETS[label_count % len(LABEL_OFFSETS)]
            label_count += 1
            ax.annotate(
                label,
                (row["T_C"], row["P_bar"]),
                xytext=offset,
                textcoords="offset points",
                fontsize=7,
                color=label_color,
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.78,
                },
            )

    if label_points and label_ids_once and rows_to_label:
        label_df = pd.DataFrame(rows_to_label)
        for label_count, (exp_id, group) in enumerate(label_df.groupby("ID", sort=False)):
            idx = group["P_bar"].astype(float).idxmax()
            row = group.loc[idx]
            offset = ID_LABEL_OFFSETS_BY_ID.get(
                str(exp_id),
                ID_LABEL_OFFSETS[label_count % len(ID_LABEL_OFFSETS)],
            )
            ax.annotate(
                str(exp_id),
                (row["T_C"], row["P_bar"]),
                xytext=offset,
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
                color=color_by_id[str(exp_id)],
                arrowprops={
                    "arrowstyle": "-",
                    "color": color_by_id[str(exp_id)],
                    "lw": 0.8,
                    "alpha": 0.85,
                },
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.86,
                },
            )

    if not color_by_xco2:
        for exp_id, color in color_by_id.items():
            ax.scatter([], [], color=color, label=exp_id)
    for transition, marker in TRANSITION_MARKERS.items():
        if transition in set(df["transition_type"]):
            ax.scatter([], [], color="black", marker=marker, label=transition)
    if not same_style_by_source:
        ax.scatter([], [], marker="o", s=68, color="white", edgecolors="black",
                   label="experimental")
        ax.scatter([], [], marker="o", s=30, color="gray", alpha=0.42,
                   label="literatura")
    ax.scatter([], [], marker="o", facecolors="none", edgecolors="black",
               label="P estimada")

    ax.set_title("P-T de puntos de transicion de fase")
    ax.set_xlabel("T (C)")
    ax.set_ylabel("P (bar)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc="upper right", framealpha=0.9)
    if color_by_xco2 and scatter_for_colorbar is not None:
        cb = fig.colorbar(scatter_for_colorbar, ax=ax)
        cb.set_label("xCO2")
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def main() -> int:
    args = build_parser().parse_args()
    df = combine_inputs(args.input, args.literature, args.include_literature)
    df = filter_points(
        df,
        ids=args.ids,
        exclude_ids=args.exclude_ids,
        cycles=args.cycles,
        transition_types=args.transition_types,
        xco2_values=args.xco2_values,
        xco2_tol=args.xco2_tol,
        include_secondary=args.include_secondary,
        include_literature=args.include_literature,
    )
    if df.empty:
        raise SystemExit("No points to plot after filters.")
    outpath = args.outdir / args.out
    plot_points(
        df,
        outpath,
        label_points=args.label_points,
        label_experimental_only=args.label_experimental_only,
        label_ids_once=args.label_ids_once,
        label_xco2=args.label_xco2,
        color_by_xco2=args.color_by_xco2,
        same_style_by_source=args.same_style_by_source,
    )
    print(f"Saved: {outpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
