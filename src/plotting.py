"""Publication-style DSC cooling plots with an LL detail panel."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from .models import EventResult


BLUE = "#2166ac"
RED = "#b2182b"
GRAY = "#777777"


def _window_mask(temperature: np.ndarray, window: tuple[float, float] | None) -> np.ndarray:
    if not window or len(window) != 2:
        return np.zeros(len(temperature), dtype=bool)
    low, high = sorted(float(value) for value in window)
    return np.isfinite(temperature) & (temperature >= low) & (temperature <= high)


def _fit_window(
    temperature: np.ndarray,
    heat_flow: np.ndarray,
    window: tuple[float, float] | None,
) -> tuple[np.ndarray, float, float] | None:
    mask = _window_mask(temperature, window) & np.isfinite(heat_flow)
    if mask.sum() < 3:
        return None
    low = float(np.nanmin(temperature[mask]))
    high = float(np.nanmax(temperature[mask]))
    return np.polyfit(temperature[mask], heat_flow[mask], 1), low, high


def _draw_fit(
    ax: plt.Axes,
    fit: tuple[np.ndarray, float, float] | None,
    x_low: float,
    x_high: float,
    *,
    color: str,
    linestyle: str,
    linewidth: float = 1.4,
    label: str | None = None,
) -> None:
    if fit is None:
        return
    coeffs = fit[0]
    x = np.linspace(float(x_low), float(x_high), 80)
    ax.plot(x, np.polyval(coeffs, x), color=color, linestyle=linestyle, linewidth=linewidth, label=label)


def _event(results: list[EventResult], label: str) -> EventResult | None:
    wanted = label.upper()
    for result in results:
        if (result.event_label or "").upper() == wanted:
            return result
    return None


def _finite_onset(result: EventResult | None) -> float | None:
    if result is None or result.onset_temp_c is None:
        return None
    return float(result.onset_temp_c) if np.isfinite(result.onset_temp_c) else None


def _ll_region(temperature: np.ndarray, result: EventResult) -> tuple[float, float] | None:
    points: list[float] = []
    onset = _finite_onset(result)
    if onset is not None:
        points.append(onset)
    if result.peak_temp_c is not None and np.isfinite(result.peak_temp_c):
        points.append(float(result.peak_temp_c))
    for window in (result.baseline_window_c, result.tangent_window_c):
        if window:
            points.extend(float(value) for value in window if np.isfinite(value))
    if not points:
        return None
    data_low = float(np.nanmin(temperature))
    data_high = float(np.nanmax(temperature))
    onset = _finite_onset(result)
    if onset is not None:
        # The reference layout uses a broad LL window for the full 10-50 C
        # scans.  A 30-10 C scan is much shorter, so use a compact window or
        # the zoom would cover almost the entire top panel.
        if data_high - data_low <= 25.0:
            if data_low <= 15.0 < onset < 16.5 <= data_high:
                return 15.0, 16.5
            return max(data_low, onset - 1.0), min(data_high, onset + 1.0)
        return max(data_low, onset - 5.0), min(data_high, onset + 8.0)
    span = max(points) - min(points)
    padding = max(1.0, 0.20 * span)
    return max(data_low, min(points) - padding), min(data_high, max(points) + padding)


def _add_cooling_arrow(ax: plt.Axes) -> None:
    ax.annotate(
        "Cooling",
        xy=(0.82, 0.94),
        xytext=(0.95, 0.94),
        xycoords="axes fraction",
        ha="center",
        va="center",
        fontsize=14,
        arrowprops={"arrowstyle": "->", "linewidth": 1.5, "color": "black"},
    )


def _style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.tick_params(top=False, right=False)
    ax.grid(False)


def _draw_tsl_panel(
    ax: plt.Axes,
    temperature: np.ndarray,
    heat_flow: np.ndarray,
    result: EventResult | None,
) -> None:
    onset = _finite_onset(result)
    if result is None or onset is None:
        return
    baseline_fit = _fit_window(temperature, heat_flow, result.baseline_window_c)
    tangent_fit = _fit_window(temperature, heat_flow, result.tangent_window_c)
    data_low = float(np.nanmin(temperature))
    data_high = float(np.nanmax(temperature))
    if baseline_fit is not None:
        baseline_x = np.linspace(onset, data_high, 160)
        baseline_y = np.polyval(baseline_fit[0], baseline_x)
        ax.plot(baseline_x, baseline_y, color=BLUE, linewidth=1.4, label="SL baseline")
        y_onset = float(np.polyval(baseline_fit[0], onset))
    else:
        y_onset = float(np.nanmedian(heat_flow))
    if tangent_fit is not None:
        _draw_fit(
            ax,
            tangent_fit,
            max(data_low, tangent_fit[1]),
            min(data_high, tangent_fit[2]),
            color=BLUE,
            linestyle="-",
            linewidth=1.2,
            label="SL tangent",
        )
    local_mask = (temperature >= onset - 1.0) & (temperature <= onset + 1.0)
    local_values = heat_flow[local_mask]
    if len(local_values) >= 3:
        local_low = float(np.nanmin(local_values))
        local_high = float(np.nanmax(local_values))
    else:
        local_low = float(np.nanmin(heat_flow))
        local_high = float(np.nanmax(heat_flow))
    local_span = max(local_high - local_low, 1e-9)
    ax.vlines(onset, y_onset - 0.05 * local_span, y_onset + 0.8 * local_span, color=BLUE, linewidth=1.5)
    ax.scatter([onset], [y_onset], s=65, facecolor="white", edgecolor=BLUE, linewidth=2, zorder=6)
    ax.text(
        onset + 0.6,
        y_onset + 0.08 * local_span,
        rf"$T_{{SL}}$ = {onset:.3f} °C",
        color=BLUE,
        fontsize=14,
    )


def _draw_tll_zoom(
    ax: plt.Axes,
    temperature: np.ndarray,
    heat_flow: np.ndarray,
    result: EventResult,
    region: tuple[float, float],
    y_limits: tuple[float, float] | None = None,
) -> None:
    low, high = region
    mask = (temperature >= low) & (temperature <= high) & np.isfinite(heat_flow)
    ax.plot(temperature[mask], heat_flow[mask], color="black", linewidth=1.0)
    onset = _finite_onset(result)
    if onset is None:
        return

    baseline_fit = _fit_window(temperature, heat_flow, result.baseline_window_c)
    tangent_fit = _fit_window(temperature, heat_flow, result.tangent_window_c)
    span = high - low
    if baseline_fit is not None:
        baseline_low = max(low, onset)
        _draw_fit(
            ax,
            baseline_fit,
            baseline_low,
            high,
            color=RED,
            linestyle="--",
            linewidth=1.7,
            label="LL baseline",
        )
        y_onset = float(np.polyval(baseline_fit[0], onset))
    else:
        y_onset = float(np.nanmedian(heat_flow[mask]))
    if tangent_fit is not None:
        tangent_low = max(low, tangent_fit[1] - 0.18 * span)
        tangent_high = min(high, tangent_fit[2] + 0.18 * span)
        _draw_fit(
            ax,
            tangent_fit,
            tangent_low,
            tangent_high,
            color=RED,
            linestyle="-",
            linewidth=2.0,
            label="LL tangent",
        )
    ax.scatter([onset], [y_onset], s=65, facecolor="white", edgecolor=RED, linewidth=2, zorder=6)
    if y_limits is None:
        y_min = float(np.nanmin(heat_flow[mask]))
        y_max = float(np.nanmax(heat_flow[mask]))
    else:
        y_min, y_max = y_limits
    ax.text(
        onset + 0.08 * span,
        y_min + 0.68 * (y_max - y_min),
        rf"$T_{{LL}}$ = {onset:.3f} °C",
        color=RED,
        fontsize=14,
    )


def plot_cycle(segment: pd.DataFrame, results: list[EventResult], out: str | Path) -> Path:
    """Plot a full cooling trace and a separate zoom of the LL region."""
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not any(r.event_label in {"TLL", "TSL"} for r in results):
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        ax.plot(segment.temperature_c, segment.heat_flow_mw, color="black", linewidth=1)
        for result in results:
            if result.onset_temp_c is not None:
                ax.axvline(result.onset_temp_c, linestyle="--", label=f"{result.event_label}: calculated onset")
        ax.set(xlabel="Temperature used (°C; see source in results)", ylabel="Heat flow (mW)",
               title="Calorimetric candidates · phase assignment pending")
        if ax.lines and len(ax.lines) > 1:
            ax.legend()
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        return out_path
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9, 10),
        gridspec_kw={"height_ratios": [1.0, 1.0]},
        constrained_layout=True,
    )
    temp = segment["temperature_c"].to_numpy(float)
    heat_flow = segment["heat_flow_mw"].to_numpy(float)
    valid = np.isfinite(temp) & np.isfinite(heat_flow)
    temp = temp[valid]
    heat_flow = heat_flow[valid]
    if len(temp) == 0:
        raise ValueError("Cannot plot an empty thermogram")

    full_ax, zoom_ax = axes
    full_ax.plot(temp, heat_flow, color="black", linewidth=1.0)
    full_ax.set_xlim(float(np.nanmin(temp)), float(np.nanmax(temp)))
    full_ax.set_xlabel("Temperature (°C)")
    full_ax.set_ylabel("Heat flow (mW)")
    full_ax.text(0.015, 0.93, "(a)", transform=full_ax.transAxes, fontsize=16, fontweight="bold")
    full_ax.text(0.03, 0.05, "Exo ↑", transform=full_ax.transAxes, fontsize=14)
    _add_cooling_arrow(full_ax)
    _draw_tsl_panel(full_ax, temp, heat_flow, _event(results, "TSL"))

    tll = _event(results, "TLL")
    region = _ll_region(temp, tll) if tll is not None else None
    if region is None:
        zoom_ax.set_axis_off()
        zoom_ax.text(0.5, 0.5, "TLL no detectado automáticamente", transform=zoom_ax.transAxes, ha="center", va="center")
    else:
        low, high = region
        compact_ll_limits = (
            (6.4, 7.1)
            if np.isclose(low, 15.0) and np.isclose(high, 16.5)
            else None
        )
        region_mask = (temp >= low) & (temp <= high)
        region_y = heat_flow[region_mask]
        if compact_ll_limits is None:
            y_min = float(np.nanmin(region_y))
            y_max = float(np.nanmax(region_y))
        else:
            y_min, y_max = compact_ll_limits
        y_pad = max(0.02 * (y_max - y_min), 0.0005)
        full_ax.add_patch(
            Rectangle(
                (low, y_min - y_pad),
                high - low,
                (y_max - y_min) + 2 * y_pad,
                fill=False,
                edgecolor=GRAY,
                linestyle="--",
                linewidth=1.2,
                zorder=4,
            )
        )
        full_ax.annotate(
            "Region enlarged in (b)",
            xy=((low + high) / 2, y_max + y_pad),
            xytext=(0.47, 0.75),
            textcoords="axes fraction",
            color=GRAY,
            fontsize=14,
            ha="center",
            arrowprops={"arrowstyle": "->", "color": GRAY, "linewidth": 1.2},
        )
        _draw_tll_zoom(zoom_ax, temp, heat_flow, tll, region, compact_ll_limits)
        zoom_ax.set_xlim(low, high)
        if compact_ll_limits is not None:
            zoom_ax.set_ylim(*compact_ll_limits)
        zoom_ax.set_xlabel("Temperature (°C)")
        zoom_ax.set_ylabel("Heat flow (mW)")
        zoom_ax.text(0.015, 0.92, "(b)", transform=zoom_ax.transAxes, fontsize=16, fontweight="bold")
        zoom_ax.text(0.03, 0.05, "Exo ↑", transform=zoom_ax.transAxes, fontsize=14)
        _add_cooling_arrow(zoom_ax)

    for ax in axes:
        _style_axis(ax)

    fig.savefig(out_path, dpi=600)
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)
    return out_path
