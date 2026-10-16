#!/usr/bin/env python3
"""Render saved Calvet HP signals and event results without recalculating onsets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from util.path import get_output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot stored Calvet event results without recalculating onsets."
    )
    parser.add_argument("--results", type=Path, required=True, help="Path to a saved results.json bundle.")
    parser.add_argument(
        "--mode",
        choices=["signals", "pt", "hf-time", "hf-temperature"],
        default="signals",
        help="Saved-signal view to render.",
    )
    parser.add_argument("--event-id", help="Optional event_id to annotate.")
    parser.add_argument(
        "--time-axis",
        choices=["selected", "sample-index", "device-clock"],
        default="selected",
        help="Time recorded in the saved frame; these are not synchronization corrections.",
    )
    parser.add_argument("--out", type=Path, help="Output image; defaults to a unique path under out/.")
    return parser


def _break_at_gaps(time: np.ndarray, values: np.ndarray) -> np.ndarray:
    result = values.astype(float, copy=True)
    steps = np.diff(time)
    positive = steps[np.isfinite(steps) & (steps > 0)]
    if len(positive):
        result[1:][steps > 5.0 * np.median(positive)] = np.nan
    return result


def _output_path(results_path: Path, mode: str) -> Path:
    folder = get_output_dir("figures", "calvet_results")
    base = folder / f"{results_path.parent.name}_{mode}.png"
    if not base.exists():
        return base
    index = 2
    while True:
        candidate = base.with_name(f"{base.stem}-{index:02d}{base.suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def render_saved_results(
    results_path: str | Path,
    *,
    mode: str = "signals",
    event_id: str | None = None,
    time_axis: str = "selected",
    out: str | Path | None = None,
) -> Path:
    """Render saved signals and annotations; this function never calculates onsets."""
    result_file = Path(results_path)
    payload = json.loads(result_file.read_text(encoding="utf-8"))
    signal_file = result_file.parent / payload["signals_file"]
    signals = pd.read_csv(signal_file)
    events = payload.get("events", [])
    if event_id is not None:
        events = [event for event in events if event.get("event_id") == event_id]
        if not events:
            raise ValueError(f"Event {event_id!r} is not present in {result_file}.")

    outpath = Path(out) if out else _output_path(result_file, mode)
    if outpath.exists():
        raise FileExistsError(f"Refusing to overwrite existing figure: {outpath}")
    outpath.parent.mkdir(parents=True, exist_ok=True)

    time_column, time_label = {
        "selected": ("t_h", "Selected-stage/sample-index time (h)"),
        "sample-index": ("t_h_original", "Sample-index time from record start (h)"),
        "device-clock": ("t_s_datetime", "Device-clock elapsed time from record start (h)"),
    }[time_axis]
    if time_column not in signals:
        raise ValueError(f"Requested time axis {time_axis!r} is not present in the saved signals.")
    time = pd.to_numeric(signals[time_column], errors="coerce").to_numpy(float)
    if time_axis == "device-clock":
        time = time / 3600.0
    hf = pd.to_numeric(signals.get("HF_mW"), errors="coerce").to_numpy(float)
    if mode == "signals":
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
        for column, label, unit in [
            ("P_muestra", "Sample pressure", "bar"),
            ("T_muestra", "Sample temperature", "°C"),
            ("T_horno", "Furnace temperature", "°C"),
            ("T_banho", "Bath temperature", "°C"),
        ]:
            if column in signals:
                values = pd.to_numeric(signals[column], errors="coerce").to_numpy(float)
                axis = axes[0] if column == "P_muestra" else axes[1]
                axis.plot(time, _break_at_gaps(time, values), label=label, linewidth=0.9)
                axis.set_ylabel(unit)
                axis.legend(loc="best", fontsize=8)
            else:
                print(f"WARNING: signal {column} is not available in saved data.")
        if "HF_mW" in signals:
            axes[2].plot(time, _break_at_gaps(time, hf), color="black", linewidth=0.8)
            axes[2].set_ylabel("HF (mW)")
        else:
            print("WARNING: signal HF_mW is not available in saved data.")
        axes[2].set_xlabel(time_label)
    elif mode == "pt":
        if "T_muestra" not in signals or "P_muestra" not in signals:
            raise ValueError("P-T plotting requires T_muestra and P_muestra in saved signals.")
        temperature = pd.to_numeric(signals["T_muestra"], errors="coerce").to_numpy(float)
        pressure = pd.to_numeric(signals["P_muestra"], errors="coerce").to_numpy(float)
        gap_time = pd.to_numeric(signals["t_h"], errors="coerce").to_numpy(float)
        pressure = _break_at_gaps(gap_time, pressure)
        mask = np.isfinite(temperature) & np.isfinite(pressure)
        if not mask.any():
            raise ValueError("No finite sample P-T points are available.")
        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
        ax.plot(temperature[mask], pressure[mask], color="black", linewidth=1.0)
        if mask.sum() > 1:
            indices = np.flatnonzero(mask)
            middle = indices[min(len(indices) // 2, len(indices) - 2)]
            ax.annotate(
                "",
                xy=(temperature[middle + 1], pressure[middle + 1]),
                xytext=(temperature[middle], pressure[middle]),
                arrowprops={"arrowstyle": "->", "color": "#d95f02"},
            )
        ax.set_xlabel("Sample temperature (°C)")
        ax.set_ylabel("Sample pressure (bar)")
        ax.grid(True, alpha=0.3)
    elif mode == "hf-time":
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        ax.plot(time, _break_at_gaps(time, hf), color="black", linewidth=0.8)
        ax.set_xlabel(time_label)
        ax.set_ylabel("HF (mW)")
    else:
        if "T_muestra" not in signals:
            raise ValueError("HF-temperature plotting requires measured T_muestra; no substitute is used.")
        temperature = pd.to_numeric(signals["T_muestra"], errors="coerce").to_numpy(float)
        hf = _break_at_gaps(time, hf)
        mask = np.isfinite(temperature) & np.isfinite(hf)
        if not mask.any():
            raise ValueError("No finite HF and sample-temperature points are available.")
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        ax.plot(temperature[mask], hf[mask], color="black", linewidth=0.8)
        ax.set_xlabel("Measured sample temperature (°C)")
        ax.set_ylabel("HF (mW)")

    if mode in {"hf-time", "hf-temperature"}:
        for event in events:
            if mode == "hf-time":
                onset_x = {
                    "selected": event.get("onset_time_h"),
                    "sample-index": event.get("onset_original_time_h"),
                    "device-clock": (
                        event.get("onset_device_clock_elapsed_s") / 3600.0
                        if event.get("onset_device_clock_elapsed_s") is not None
                        else None
                    ),
                }[time_axis]
            else:
                onset_x = event.get("onset_temperature_c")
            if onset_x is None or event.get("onset_hf_mw") is None:
                continue
            ax.scatter([onset_x], [event["onset_hf_mw"]], label=f"{event['event_id']} onset")
        if events:
            ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Calvet HP — {mode} — stored results")
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"Saved plot: {outpath}")
    return outpath


def main() -> int:
    args = build_parser().parse_args()
    render_saved_results(
        args.results,
        mode=args.mode,
        event_id=args.event_id,
        time_axis=args.time_axis,
        out=args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
