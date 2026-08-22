"""Publication-style DSC plotting."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .models import EventResult


def plot_cycle(segment: pd.DataFrame, results: list[EventResult], out: str | Path) -> Path:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    temp = segment["temperature_c"].to_numpy(float)
    hf = segment["heat_flow_mw"].to_numpy(float)

    axes[0].plot(temp, hf, color="black", lw=1)
    axes[0].set_xlabel("Temperature (°C)")
    axes[0].set_ylabel("Heat flow (mW)")
    axes[0].text(0.03, 0.95, "Exo ↑", transform=axes[0].transAxes, va="top")
    axes[0].annotate("Cooling", xy=(0.78, 0.1), xytext=(0.55, 0.1), xycoords="axes fraction", arrowprops={"arrowstyle": "->"})

    finite_onsets = [r.onset_temp_c for r in results if r.onset_temp_c is not None]
    if finite_onsets:
        center = float(np.mean(finite_onsets))
        width = max(2.0, float(np.ptp(finite_onsets)) + 1.0)
        mask = (temp >= center - width) & (temp <= center + width)
    else:
        mask = np.ones(len(temp), dtype=bool)
    axes[1].plot(temp[mask], hf[mask], color="black", lw=1)
    for result in results:
        if result.onset_temp_c is not None:
            axes[0].axvline(result.onset_temp_c, color="tab:red", lw=0.8)
            axes[1].axvline(result.onset_temp_c, color="tab:red", lw=0.8)
            axes[1].text(result.onset_temp_c, np.nanmax(hf[mask]), result.event_label, rotation=90, va="top")
    axes[1].set_xlabel("Temperature (°C)")
    axes[1].set_ylabel("Heat flow (mW)")

    for ax in axes:
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
        ax.tick_params(top=False, right=False)
        ax.grid(False)
        ax.invert_xaxis()

    fig.savefig(out_path, dpi=600)
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)
    return out_path
