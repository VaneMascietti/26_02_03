"""Render saved signals and calculated event records. Never computes onsets."""
from __future__ import annotations

import json
from pathlib import Path
import warnings
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _plot(ax, frame, x, y, **kwargs):
    if y not in frame:
        warnings.warn(f"Signal unavailable: {y}", stacklevel=2)
        return
    # Break line segments at missing values, clock reversals, and acquisition gaps.
    part = frame.copy()
    dt = part.t_h.diff() * 3600
    part.loc[(dt > 5) | (dt < 0), y] = np.nan
    if "t_original_s" in part:
        original_dt = part.t_original_s.diff()
        part.loc[(original_dt > 5) | (original_dt < 0), y] = np.nan
    if "source_row" in part:
        part.loc[part.source_row.diff() > 1, y] = np.nan
    ax.plot(part[x], part[y], linewidth=1, **kwargs)


def time_signals_figure(frame: pd.DataFrame, title: str | None = None):
    """Pressure, sample/Huber temperature and HF on three aligned time axes."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, constrained_layout=True)
    _plot(axes[0], frame, "t_h", "P_muestra", color="black", label="Muestra")
    _plot(axes[1], frame, "t_h", "T_muestra", color="tab:blue", label="Muestra")
    if "T_banho" in frame and np.isfinite(pd.to_numeric(frame["T_banho"], errors="coerce")).any():
        _plot(axes[1], frame, "t_h", "T_banho", color="seagreen", label="Huber")
    _plot(axes[2], frame, "t_h", "HF_mW", color="black", label="Heat flow")
    for ax, label in zip(axes, ("Presión de muestra / bar", "Temperatura / °C", "Heat flow / mW")):
        ax.set_ylabel(label)
        if ax.lines:
            ax.legend(loc="best")
        ax.grid(alpha=.2)
    axes[-1].set_xlabel("Tiempo de corrida / h")
    fig.suptitle(title or "Señales frente al tiempo")
    return fig


def plot_time_signals(frame: pd.DataFrame, out: str | Path, title: str | None = None) -> Path:
    """Save the time overview without calculating events or substituting temperatures."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig = time_signals_figure(frame, title)
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_saved(result_path, kind="signals", event_id=None) -> Path:
    # Saved commands using the retired panel name now render the time overview.
    if kind == "diagnostic":
        kind = "signals"
    result_path = Path(result_path)
    record = json.loads(result_path.read_text())
    frame = pd.read_csv(result_path.parent / record["signals_file"])
    outdir = result_path.parent / "figures"
    outdir.mkdir(exist_ok=True)
    suffix = "_" + re.sub(r"[^A-Za-z0-9_.-]", "_", event_id) if event_id else ""
    # Replots also avoid overwriting earlier figures.
    index = 1
    out = outdir / f"{kind}{suffix}.png"
    while out.exists():
        index += 1
        out = outdir / f"{kind}{suffix}_{index}.png"
    if kind == "signals":
        metadata = record.get("metadata", {})
        title = " · ".join(str(metadata[key]) for key in ("experiment_id", "run_id") if metadata.get(key))
        return plot_time_signals(frame, out, title=title or None)
    elif kind == "pt":
        fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
        stages = record["stages"] or [{"stage_id": "interval", "start_h": frame.t_h.min(), "end_h": frame.t_h.max()}]
        for stage in stages:
            part = frame.loc[frame.t_h.between(stage["start_h"], stage["end_h"])]
            _plot(ax, part, "T_muestra", "P_muestra", label=stage["stage_id"] + " " + stage.get("selection_kind", ""))
            good = part.dropna(subset=["T_muestra", "P_muestra"])
            if len(good) > 2:
                middle = good.iloc[int(.45*len(good)):int(.55*len(good))+1]
                a, b = middle.iloc[0], middle.iloc[-1]
                gaps = middle.t_h.diff()*3600
                continuous = not ((gaps > 5) | (gaps < 0)).any()
                if "t_original_s" in middle:
                    gaps = middle.t_original_s.diff()
                    continuous &= not ((gaps > 5) | (gaps < 0)).any()
                if "source_row" in middle:
                    continuous &= not (middle.source_row.diff() > 1).any()
                if continuous:
                    ax.annotate("", xy=(b.T_muestra, b.P_muestra), xytext=(a.T_muestra, a.P_muestra), arrowprops={"arrowstyle": "->"})
        for event in record["events"]:
            if not event.get("onset"):
                continue
            temp = event["onset"]["temperature_c"]["value"]
            pressure = event["onset"]["pressure_bar"]["value"]
            if temp is not None and pressure is not None:
                ax.scatter([temp], [pressure], facecolor="white", edgecolor="tab:purple", zorder=5)
                ax.annotate(event["event_id"], xy=(temp, pressure), xytext=(8, 8), textcoords="offset points", fontsize=9)
        ax.set(xlabel="Temperatura de muestra / °C", ylabel="Presión medida / bar")
        ax.legend()
        ax.grid(alpha=.2)
    elif kind in {"hf-time", "hf-temperature"}:
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        events = record["events"]
        if event_id:
            events = [e for e in events if e["event_id"] == event_id]
            if not events:
                plt.close(fig)
                raise ValueError(f"Event not found: {event_id}")
            frame = frame.loc[frame.t_h.between(*events[0]["requested_window_h"])]
        x = "t_h" if kind == "hf-time" else "T_muestra"
        _plot(ax, frame, x, "HF_mW", color="black", label="HF")
        for event in events:
            if not event.get("onset"):
                continue
            point = event["onset"]
            coordinate = point["time_h"] if x == "t_h" else point["temperature_c"]["value"]
            if coordinate is not None:
                ax.axvline(coordinate, linestyle="--", label=event["event_id"] + " onset calculado")
            calc = event["calculation"]
            if x == "t_h" and event["method"] == "tangent":
                window = event["requested_window_h"]
                xs = np.array(window)
                ax.plot(xs, calc["a_base"]+calc["b_base"]*xs, ":", label="Línea base guardada")
                for key, color in (("baseline_window", "tab:blue"), ("flank_window", "tab:orange")):
                    if calc.get(key):
                        ax.axvspan(*calc[key], color=color, alpha=.12)
                flank = np.array(calc["flank_window"])
                ax.plot(flank, calc["y0"] + calc["m_tan"]*(flank-calc["x0"]), "--", label="Tangente guardada")
        ax.set(xlabel="Tiempo de corrida / h" if x == "t_h" else "Temperatura de muestra / °C", ylabel="HF / mW")
        measured = frame.HF_mW.to_numpy(float)
        measured = measured[np.isfinite(measured)]
        if len(measured):
            low, high = float(measured.min()), float(measured.max())
            pad = max(.06*(high-low), .001)
            ax.set_ylim(low-pad, high+pad)
        ax.legend()
        ax.grid(alpha=.2)
    else:
        raise ValueError(f"Unknown plot: {kind}")
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def compare_saved(paths, out: Path, x_axis="T_muestra"):
    """Overlay saved selected runs/cycles, without reading original LVM or fitting events."""
    out = out.resolve()
    if not out.is_relative_to(Path("out").resolve()):
        raise ValueError("Output must be within out/")
    if out.exists():
        raise FileExistsError(out)
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for path in paths:
        path = Path(path)
        record = json.loads(path.read_text())
        frame = pd.read_csv(path.parent / record["signals_file"])
        label = "/".join(str(record["metadata"].get(k) or "pending") for k in ("load_id", "run_id"))
        _plot(ax, frame, x_axis, "HF_mW", label=label)
    ax.set(xlabel="Temperatura de muestra / °C" if x_axis == "T_muestra" else "Tiempo de corrida / h", ylabel="HF / mW")
    ax.legend()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out
