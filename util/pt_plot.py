"""Shared rendering for sample-pressure trajectories in P-T and P-time space."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .data_filters import filter_pt_rows


def plot_pt_tracks(
    tracks: Iterable[tuple[pd.DataFrame, str]],
    outpath: Path,
    *,
    include_time: bool = False,
    show_initial_pressure: bool = False,
    point_size: float = 0.55,
    alpha: float = 0.65,
    markerscale: float = 8,
    xlabel: str = "T muestra (°C)",
    ylabel: str = "P muestra (bar)",
    time_xlabel: str = "Tiempo desde inicio (h)",
    time_ylabel: str = "P muestra (bar)",
    pt_title: str = "P vs T",
    time_title: str = "P vs tiempo",
    positive_pressure: bool = True,
    print_time_range: bool = True,
    empty_message: str = "no hay puntos P-T validos",
    suptitle: str | None = None,
) -> None:
    """Plot preloaded tracks while preserving each caller's labels and styles."""
    outpath.parent.mkdir(parents=True, exist_ok=True)
    if include_time:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        ax_pt, ax_time = axes
    else:
        fig, ax_pt = plt.subplots(figsize=(7, 5), constrained_layout=True)
        ax_time = None

    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["black"])
    for index, (raw_data, label) in enumerate(tracks):
        data = filter_pt_rows(raw_data, positive_pressure=positive_pressure)
        if data.empty:
            print(f"{label}: {empty_message}")
            continue

        color = colors[index % len(colors)]
        ax_pt.scatter(
            data["T_muestra"],
            data["P_muestra"],
            s=point_size,
            alpha=alpha,
            color=color,
            label=label,
            rasterized=True,
        )
        if ax_time is not None:
            ax_time.scatter(
                data["t_h"],
                data["P_muestra"],
                s=point_size,
                alpha=alpha,
                color=color,
                label=label,
                rasterized=True,
            )
        if show_initial_pressure:
            first = data.iloc[0]
            label_text = f"P0={first['P_muestra']:.1f} bar"
            ax_pt.scatter(
                [first["T_muestra"]],
                [first["P_muestra"]],
                s=36,
                color=color,
                edgecolors="black",
                linewidths=0.7,
                zorder=5,
            )
            ax_pt.annotate(
                label_text,
                (first["T_muestra"], first["P_muestra"]),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=8,
                color=color,
            )
            if ax_time is not None:
                ax_time.scatter(
                    [first["t_h"]],
                    [first["P_muestra"]],
                    s=36,
                    color=color,
                    edgecolors="black",
                    linewidths=0.7,
                    zorder=5,
                )
                ax_time.annotate(
                    label_text,
                    (first["t_h"], first["P_muestra"]),
                    xytext=(6, 6),
                    textcoords="offset points",
                    fontsize=8,
                    color=color,
                )

        summary = (
            f"{label}: points={len(data)}, "
            f"T=[{data['T_muestra'].min():.3f},{data['T_muestra'].max():.3f}] "
            f"C, P=[{data['P_muestra'].min():.3f},{data['P_muestra'].max():.3f}] bar"
        )
        if print_time_range:
            summary = (
                f"{label}: points={len(data)}, "
                f"t=[{data['t_h'].min():.3f},{data['t_h'].max():.3f}] h, "
                f"T=[{data['T_muestra'].min():.3f},{data['T_muestra'].max():.3f}] °C, "
                f"P=[{data['P_muestra'].min():.3f},{data['P_muestra'].max():.3f}] bar"
            )
        print(summary)

    ax_pt.set_title(pt_title)
    ax_pt.set_xlabel(xlabel)
    ax_pt.set_ylabel(ylabel)
    ax_pt.grid(True, alpha=0.3)
    ax_pt.legend(markerscale=markerscale, fontsize=8)

    if ax_time is not None:
        ax_time.set_title(time_title)
        ax_time.set_xlabel(time_xlabel)
        ax_time.set_ylabel(time_ylabel)
        ax_time.grid(True, alpha=0.3)
        ax_time.legend(markerscale=markerscale, fontsize=8)

    if suptitle:
        fig.suptitle(suptitle, fontsize=10)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)
    print(f"Guardado: {outpath}")
