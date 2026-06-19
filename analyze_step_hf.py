import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from process_dsc_data import load_lvm, program_zone_spans
from util.path import get_data_path


DIAGNOSTIC_DEFAULT = "N2 cylinder closed during experiment"
HF_NOTE = "Diagnostic run: N2 cylinder closed / HF not quantitative"
RESIDUAL_NOTE = "Residual is diagnostic only because N2 atmosphere was not controlled."


def _require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _resolve_input(name: str | Path, kind: str, label: str) -> Path:
    path = get_data_path(name, kind)
    return _require_file(path, label)


def _check_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def _prep_dataframe(lvm_path: Path, program_path: Path | None) -> pd.DataFrame:
    df = load_lvm(lvm_path, program_pdf=None)
    _check_columns(
        df,
        ["t_h", "T_muestra", "T_horno", "P_muestra", "HF_mW", "datetime"],
        str(lvm_path),
    )
    out = df.copy()
    out["time_min"] = out["t_h"].to_numpy(dtype=float) * 60.0
    out["T_sample_C"] = pd.to_numeric(out["T_muestra"], errors="coerce")
    out["T_furnace_C"] = pd.to_numeric(out["T_horno"], errors="coerce")
    out["P_sample_bar"] = pd.to_numeric(out["P_muestra"], errors="coerce")
    out["HF_mW"] = pd.to_numeric(out["HF_mW"], errors="coerce")

    t = out["time_min"].to_numpy(dtype=float)
    t_unique = np.maximum.accumulate(t + np.arange(len(t)) * 1e-12)
    out["dTdt_C_min"] = np.gradient(out["T_sample_C"].to_numpy(dtype=float), t_unique)
    out["dPdt_bar_min"] = np.gradient(out["P_sample_bar"].to_numpy(dtype=float), t_unique)
    return out


def _classify_zone_type(ztype: str, mean_dtdt: float, threshold_c_min: float) -> str:
    z = (ztype or "").strip().lower()
    if "isoterma" in z or "isothermal" in z:
        return "isotherm"
    if "cooling" in z or "heating" in z or "return" in z:
        return "ramp"
    if np.isfinite(mean_dtdt):
        return "ramp" if abs(mean_dtdt) > threshold_c_min else "isotherm"
    return "unknown"


def _zones_from_pdf(df: pd.DataFrame, program_path: Path, threshold_c_min: float) -> list[dict]:
    spans = program_zone_spans(program_path, t_h=df["t_h"].to_numpy(dtype=float))
    zones = []
    for i, span in enumerate(spans, 1):
        start_min = float(span["start_h"]) * 60.0
        end_min = float(span["end_h"]) * 60.0
        mask = (df["time_min"] >= start_min) & (df["time_min"] <= end_min)
        if not mask.any():
            continue
        mean_dtdt = float(np.nanmedian(df.loc[mask, "dTdt_C_min"]))
        zones.append(
            {
                "zone_id": int(span.get("num", i)),
                "program_type": str(span.get("type", "")),
                "t_start_min": start_min,
                "t_end_min": end_min,
                "classification": _classify_zone_type(
                    str(span.get("type", "")), mean_dtdt, threshold_c_min
                ),
                "source": "pdf",
            }
        )
    return zones


def _zones_from_derivative(df: pd.DataFrame, threshold_c_min: float, min_duration_min: float = 1.0) -> list[dict]:
    dtdt = df["dTdt_C_min"].to_numpy(dtype=float)
    t = df["time_min"].to_numpy(dtype=float)
    labels = np.where(np.abs(dtdt) > threshold_c_min, "ramp", "isotherm")
    labels[~np.isfinite(dtdt)] = "unknown"

    raw = []
    start = 0
    current = labels[0]
    for i in range(1, len(labels)):
        if labels[i] != current:
            raw.append((current, start, i - 1))
            start = i
            current = labels[i]
    raw.append((current, start, len(labels) - 1))

    zones = []
    for label, s, e in raw:
        duration = float(t[e] - t[s])
        if duration < min_duration_min and zones:
            zones[-1]["t_end_min"] = float(t[e])
            continue
        zones.append(
            {
                "zone_id": len(zones) + 1,
                "program_type": "fallback_derivative",
                "t_start_min": float(t[s]),
                "t_end_min": float(t[e]),
                "classification": str(label),
                "source": "derivative",
            }
        )
    return zones


def _build_zones(df: pd.DataFrame, program_path: Path, threshold_c_min: float) -> list[dict]:
    zones = _zones_from_pdf(df, program_path, threshold_c_min)
    if zones:
        return zones
    return _zones_from_derivative(df, threshold_c_min)


def _assign_zones(df: pd.DataFrame, zones: list[dict]) -> pd.DataFrame:
    out = df.copy()
    out["zone_id"] = pd.NA
    out["zone_type"] = "unknown"
    for zone in zones:
        mask = (out["time_min"] >= zone["t_start_min"]) & (out["time_min"] <= zone["t_end_min"])
        out.loc[mask, "zone_id"] = int(zone["zone_id"])
        out.loc[mask, "zone_type"] = zone["classification"]
    out["zone_id"] = out["zone_id"].astype("Int64")
    return out


def _edge_value(df: pd.DataFrame, column: str, first: bool) -> float:
    vals = pd.to_numeric(df[column], errors="coerce").dropna()
    if vals.empty:
        return np.nan
    return float(vals.iloc[0] if first else vals.iloc[-1])


def _zone_summary(df: pd.DataFrame, zones: list[dict]) -> pd.DataFrame:
    rows = []
    for zone in zones:
        zdf = df.loc[
            (df["time_min"] >= zone["t_start_min"]) & (df["time_min"] <= zone["t_end_min"])
        ]
        if zdf.empty:
            continue
        rows.append(
            {
                "zone_id": int(zone["zone_id"]),
                "source": zone["source"],
                "program_type": zone["program_type"],
                "classification": zone["classification"],
                "t_start_min": float(zone["t_start_min"]),
                "t_end_min": float(zone["t_end_min"]),
                "duration_min": float(zone["t_end_min"] - zone["t_start_min"]),
                "T_start_C": _edge_value(zdf, "T_sample_C", True),
                "T_end_C": _edge_value(zdf, "T_sample_C", False),
                "P_start_bar": _edge_value(zdf, "P_sample_bar", True),
                "P_end_bar": _edge_value(zdf, "P_sample_bar", False),
                "HF_start_mW": _edge_value(zdf, "HF_mW", True),
                "HF_end_mW": _edge_value(zdf, "HF_mW", False),
                "mean_dTdt_C_min": float(np.nanmean(zdf["dTdt_C_min"])),
                "mean_dPdt_bar_min": float(np.nanmean(zdf["dPdt_bar_min"])),
            }
        )
    return pd.DataFrame(rows)


def _relaxation_fit(zdf: pd.DataFrame) -> dict:
    clean = zdf[["time_min", "HF_mW"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 20:
        return {"HF_inf_mW": np.nan, "A_mW": np.nan, "tau_min": np.nan, "fit_points": len(clean)}

    tail_n = max(5, int(round(len(clean) * 0.15)))
    hf_inf = float(np.nanmedian(clean["HF_mW"].iloc[-tail_n:]))
    t0 = float(clean["time_min"].iloc[0])
    x = clean["time_min"].to_numpy(dtype=float) - t0
    delta = clean["HF_mW"].to_numpy(dtype=float) - hf_inf
    scale = np.nanmax(np.abs(delta))
    if not np.isfinite(scale) or scale <= 0:
        return {"HF_inf_mW": hf_inf, "A_mW": np.nan, "tau_min": np.nan, "fit_points": 0}

    mask = (np.abs(delta) > 0.05 * scale) & np.isfinite(x) & np.isfinite(delta) & (x > 0)
    if mask.sum() < 10:
        return {"HF_inf_mW": hf_inf, "A_mW": np.nan, "tau_min": np.nan, "fit_points": int(mask.sum())}

    y = np.log(np.abs(delta[mask]))
    xm = x[mask]
    try:
        slope, intercept = np.polyfit(xm, y, 1)
    except Exception:
        return {"HF_inf_mW": hf_inf, "A_mW": np.nan, "tau_min": np.nan, "fit_points": int(mask.sum())}
    if not np.isfinite(slope) or slope >= -1e-9:
        tau = np.nan
    else:
        tau = float(-1.0 / slope)
    sign = np.sign(delta[mask][0]) if mask.any() else np.nan
    amp = float(sign * np.exp(intercept)) if np.isfinite(intercept) else np.nan
    return {"HF_inf_mW": hf_inf, "A_mW": amp, "tau_min": tau, "fit_points": int(mask.sum())}


def _isotherm_relaxation_table(df: pd.DataFrame, zones: list[dict]) -> pd.DataFrame:
    rows = []
    for zone in zones:
        if zone["classification"] != "isotherm":
            continue
        zdf = df.loc[
            (df["time_min"] >= zone["t_start_min"]) & (df["time_min"] <= zone["t_end_min"])
        ]
        fit = _relaxation_fit(zdf)
        rows.append(
            {
                "zone_id": int(zone["zone_id"]),
                "t_start_min": float(zone["t_start_min"]),
                "t_end_min": float(zone["t_end_min"]),
                "duration_min": float(zone["t_end_min"] - zone["t_start_min"]),
                **fit,
            }
        )
    return pd.DataFrame(rows)


def _fit_hf_baseline(
    df: pd.DataFrame,
    exclude_window: tuple[float, float],
    transient_exclude_min: float,
) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    t_low, t_high = sorted(exclude_window)
    fit_mask = (
        np.isfinite(out["HF_mW"])
        & np.isfinite(out["T_sample_C"])
        & np.isfinite(out["dTdt_C_min"])
        & np.isfinite(out["T_furnace_C"])
        & ~out["T_sample_C"].between(t_low, t_high, inclusive="both")
    )

    for zone_id in out["zone_id"].dropna().unique():
        zmask = out["zone_id"] == zone_id
        if not zmask.any():
            continue
        zstart = float(out.loc[zmask, "time_min"].min())
        fit_mask &= ~(zmask & (out["time_min"] <= zstart + transient_exclude_min))

    hf_abs = np.abs(out["HF_mW"] - np.nanmedian(out["HF_mW"]))
    limit = np.nanquantile(hf_abs, 0.99) if np.isfinite(hf_abs).any() else np.nan
    if np.isfinite(limit):
        fit_mask &= hf_abs <= limit

    design = pd.DataFrame(
        {
            "c0": 1.0,
            "T_sample": out["T_sample_C"],
            "dTdt": out["dTdt_C_min"],
            "T_furnace_minus_sample": out["T_furnace_C"] - out["T_sample_C"],
        }
    )
    x = design.loc[fit_mask].to_numpy(dtype=float)
    y = out.loc[fit_mask, "HF_mW"].to_numpy(dtype=float)
    if len(y) < 10:
        coefs = np.full(4, np.nan)
    else:
        coefs, *_ = np.linalg.lstsq(x, y, rcond=None)

    full_x = design.to_numpy(dtype=float)
    out["HF_base_mW"] = full_x @ coefs if np.all(np.isfinite(coefs)) else np.nan
    out["HF_res_mW"] = out["HF_mW"] - out["HF_base_mW"]
    info = {
        "c0": coefs[0],
        "c1_T_sample": coefs[1],
        "c2_dTdt": coefs[2],
        "c3_T_furnace_minus_sample": coefs[3],
        "fit_points": int(fit_mask.sum()),
    }
    return out, info


def _local_linear_baseline(zdf: pd.DataFrame) -> np.ndarray:
    design = pd.DataFrame(
        {
            "c0": 1.0,
            "T_sample": zdf["T_sample_C"],
            "dTdt": zdf["dTdt_C_min"],
            "T_furnace_minus_sample": zdf["T_furnace_C"] - zdf["T_sample_C"],
        }
    )
    mask = np.isfinite(design).all(axis=1) & np.isfinite(zdf["HF_mW"])
    if mask.sum() < 20:
        return np.full(len(zdf), float(np.nanmedian(zdf["HF_mW"])))
    coefs, *_ = np.linalg.lstsq(
        design.loc[mask].to_numpy(dtype=float),
        zdf.loc[mask, "HF_mW"].to_numpy(dtype=float),
        rcond=None,
    )
    if not np.all(np.isfinite(coefs)):
        return np.full(len(zdf), float(np.nanmedian(zdf["HF_mW"])))
    return design.to_numpy(dtype=float) @ coefs


def _isotherm_relaxation_baseline(zdf: pd.DataFrame) -> np.ndarray:
    fit = _relaxation_fit(zdf)
    hf_inf = fit.get("HF_inf_mW", np.nan)
    amp = fit.get("A_mW", np.nan)
    tau = fit.get("tau_min", np.nan)
    if not (np.isfinite(hf_inf) and np.isfinite(amp) and np.isfinite(tau) and tau > 0):
        return _local_linear_baseline(zdf)
    t = zdf["time_min"].to_numpy(dtype=float)
    t0 = float(np.nanmin(t))
    return hf_inf + amp * np.exp(-(t - t0) / tau)


def _fit_hf_baseline_by_zone(df: pd.DataFrame, zones: list[dict]) -> tuple[pd.DataFrame, dict]:
    """
    Build a program-aware baseline.

    Ramps get a local linear model. Isotherms get a local exponential
    relaxation model when stable, otherwise the same local linear fallback.
    This intentionally does not exclude LL by temperature; it lets the user
    inspect the LL interval in the residual instead of fitting around it.
    """
    out = df.copy()
    out["HF_base_zone_mW"] = np.nan
    info_rows = []

    for zone in zones:
        zmask = (
            (out["time_min"] >= zone["t_start_min"])
            & (out["time_min"] <= zone["t_end_min"])
        )
        zdf = out.loc[zmask].copy()
        if zdf.empty:
            continue
        if zone["classification"] == "isotherm":
            base = _isotherm_relaxation_baseline(zdf)
            model = "isotherm_relaxation"
        else:
            base = _local_linear_baseline(zdf)
            model = "local_linear"
        out.loc[zmask, "HF_base_zone_mW"] = base
        residual = zdf["HF_mW"].to_numpy(dtype=float) - base
        info_rows.append(
            {
                "zone_id": int(zone["zone_id"]),
                "classification": zone["classification"],
                "model": model,
                "points": int(len(zdf)),
                "residual_median_mW": float(np.nanmedian(residual)),
                "residual_q05_mW": float(np.nanquantile(residual, 0.05)),
                "residual_q95_mW": float(np.nanquantile(residual, 0.95)),
            }
        )

    out["HF_res_zone_mW"] = out["HF_mW"] - out["HF_base_zone_mW"]
    out["HF_base_mW"] = out["HF_base_zone_mW"]
    out["HF_res_mW"] = out["HF_res_zone_mW"]
    return out, {"mode": "zone", "zones": info_rows}


def _shade_zones(ax, zones: list[dict]) -> None:
    colors = {"ramp": "#f7d9d9", "isotherm": "#d9e8f7", "unknown": "#eeeeee"}
    for zone in zones:
        color = colors.get(zone["classification"], "#eeeeee")
        ax.axvspan(zone["t_start_min"], zone["t_end_min"], color=color, alpha=0.22, lw=0)
        ax.axvline(zone["t_start_min"], color="#777777", linewidth=0.5, alpha=0.45)


def _save_figure(fig, base_path: Path, transparent: bool = False) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base_path.with_suffix(".png"), dpi=220)
    fig.savefig(base_path.with_suffix(".pdf"))
    if transparent:
        fig.savefig(
            base_path.with_name(f"{base_path.stem}_transparent.png"),
            dpi=220,
            transparent=True,
        )
    plt.close(fig)


def _plot_pt(
    cooling: pd.DataFrame,
    heating: pd.DataFrame | None,
    outdir: Path,
    stem: str,
    transparent: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    if heating is not None:
        h = heating[np.isfinite(heating["T_sample_C"]) & np.isfinite(heating["P_sample_bar"])]
        ax.scatter(h["T_sample_C"], h["P_sample_bar"], s=0.5, alpha=0.45, label="heating", rasterized=True)
    c = cooling[np.isfinite(cooling["T_sample_C"]) & np.isfinite(cooling["P_sample_bar"])]
    ax.scatter(c["T_sample_C"], c["P_sample_bar"], s=0.5, alpha=0.55, label="step cooling", rasterized=True)
    for data, prefix in [(heating, "heating"), (cooling, "cooling")]:
        if data is None or data.empty:
            continue
        valid = data[np.isfinite(data["T_sample_C"]) & np.isfinite(data["P_sample_bar"])]
        if valid.empty:
            continue
        ax.scatter(valid["T_sample_C"].iloc[0], valid["P_sample_bar"].iloc[0], s=35, marker="o", color="green", zorder=5)
        ax.scatter(valid["T_sample_C"].iloc[-1], valid["P_sample_bar"].iloc[-1], s=35, marker="x", color="black", zorder=5)
        ax.annotate(f"{prefix} start", (valid["T_sample_C"].iloc[0], valid["P_sample_bar"].iloc[0]), xytext=(5, 5), textcoords="offset points", fontsize=8)
        ax.annotate(f"{prefix} end", (valid["T_sample_C"].iloc[-1], valid["P_sample_bar"].iloc[-1]), xytext=(5, -10), textcoords="offset points", fontsize=8)
    ax.set_xlabel("T_sample_C")
    ax.set_ylabel("P_sample_bar")
    ax.set_title("P-T path")
    ax.grid(True, alpha=0.3)
    ax.legend(markerscale=8, fontsize=8)
    _save_figure(fig, outdir / f"{stem}_pt_path", transparent=transparent)


def _plot_diagnostic(df: pd.DataFrame, zones: list[dict], outdir: Path, stem: str, note: str, transparent: bool) -> None:
    fig, axes = plt.subplots(5, 1, figsize=(10, 11), sharex=True, constrained_layout=True)
    for ax in axes:
        _shade_zones(ax, zones)
    axes[0].plot(df["time_min"], df["T_sample_C"], label="T_sample_C", linewidth=0.9)
    axes[0].plot(df["time_min"], df["T_furnace_C"], label="T_furnace_C", linewidth=0.9)
    axes[0].set_ylabel("T (°C)")
    axes[0].legend(fontsize=8)
    axes[1].plot(df["time_min"], df["P_sample_bar"], color="black", linewidth=0.8)
    axes[1].set_ylabel("P (bar)")
    axes[2].plot(df["time_min"], df["HF_mW"], color="#7b3294", linewidth=0.8)
    axes[2].set_ylabel("HF (mW)")
    axes[3].plot(df["time_min"], df["dTdt_C_min"], color="#1b9e77", linewidth=0.8)
    axes[3].set_ylabel("dT/dt\n(°C/min)")
    axes[4].plot(df["time_min"], df["dPdt_bar_min"], color="#d95f02", linewidth=0.8)
    axes[4].set_ylabel("dP/dt\n(bar/min)")
    axes[4].set_xlabel("time_min")
    fig.suptitle(f"{HF_NOTE}\n{note}", fontsize=11)
    _save_figure(fig, outdir / f"{stem}_diagnostic_multipanel", transparent=transparent)


def _plot_hf_vs_t(df: pd.DataFrame, outdir: Path, stem: str, window: tuple[float, float], transparent: bool) -> None:
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    valid = np.isfinite(df["T_sample_C"]) & np.isfinite(df["HF_mW"]) & np.isfinite(df["time_min"])
    sc = ax.scatter(
        df.loc[valid, "T_sample_C"],
        df.loc[valid, "HF_mW"],
        c=df.loc[valid, "time_min"],
        s=0.6,
        alpha=0.65,
        cmap="viridis",
        rasterized=True,
    )
    t_low, t_high = sorted(window)
    ax.axvspan(t_low, t_high, color="red", alpha=0.12, label=f"{t_low:g}-{t_high:g} °C")
    ax.set_xlabel("T_sample_C")
    ax.set_ylabel("HF_mW")
    ax.set_title("HF vs T - diagnostic only")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("time_min")
    _save_figure(fig, outdir / f"{stem}_hf_vs_T", transparent=transparent)


def _plot_hf_residual(df: pd.DataFrame, outdir: Path, stem: str, window: tuple[float, float], transparent: bool) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), constrained_layout=True)
    t_low, t_high = sorted(window)
    axes[0].plot(df["time_min"], df["HF_mW"], linewidth=0.8)
    axes[0].set_ylabel("HF_mW")
    axes[1].plot(df["time_min"], df["HF_base_mW"], linewidth=0.8)
    axes[1].set_ylabel("HF_base_mW")
    axes[2].plot(df["time_min"], df["HF_res_mW"], linewidth=0.8)
    axes[2].set_ylabel("HF_res_mW")
    valid = np.isfinite(df["T_sample_C"]) & np.isfinite(df["HF_res_mW"])
    axes[3].scatter(df.loc[valid, "T_sample_C"], df.loc[valid, "HF_res_mW"], s=0.6, alpha=0.65, rasterized=True)
    axes[3].axvspan(t_low, t_high, color="red", alpha=0.12, label=f"{t_low:g}-{t_high:g} °C")
    axes[3].set_xlabel("T_sample_C")
    axes[3].set_ylabel("HF_res_mW")
    axes[3].legend(fontsize=8)
    for ax in axes[:3]:
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("time_min")
    axes[3].grid(True, alpha=0.3)
    fig.suptitle(RESIDUAL_NOTE, fontsize=11)
    _save_figure(fig, outdir / f"{stem}_hf_residual", transparent=transparent)


def _plot_hf_residual_zoom(
    df: pd.DataFrame,
    outdir: Path,
    stem: str,
    window: tuple[float, float],
    ylim: tuple[float, float],
    transparent: bool,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    t_low, t_high = sorted(window)
    y_low, y_high = sorted(ylim)

    valid_time = np.isfinite(df["time_min"]) & np.isfinite(df["HF_res_mW"])
    axes[0].plot(df.loc[valid_time, "time_min"], df.loc[valid_time, "HF_res_mW"], linewidth=0.8)
    axes[0].set_xlabel("time_min")
    axes[0].set_ylabel("HF_res_mW")
    axes[0].set_ylim(y_low, y_high)
    axes[0].grid(True, alpha=0.3)

    valid_t = np.isfinite(df["T_sample_C"]) & np.isfinite(df["HF_res_mW"])
    axes[1].scatter(
        df.loc[valid_t, "T_sample_C"],
        df.loc[valid_t, "HF_res_mW"],
        s=0.7,
        alpha=0.65,
        rasterized=True,
    )
    axes[1].axvspan(t_low, t_high, color="red", alpha=0.12, label=f"{t_low:g}-{t_high:g} °C")
    axes[1].set_xlabel("T_sample_C")
    axes[1].set_ylabel("HF_res_mW")
    axes[1].set_ylim(y_low, y_high)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.suptitle(f"HF_res zoom ({y_low:g} to {y_high:g} mW) - diagnostic only", fontsize=11)
    y_low_txt = f"{y_low:g}".replace(".", "p")
    y_high_txt = f"{y_high:g}".replace(".", "p")
    _save_figure(
        fig,
        outdir / f"{stem}_hf_residual_zoom_{y_low_txt}_{y_high_txt}mW",
        transparent=transparent,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnostic HF analysis for stepwise cooling Calvet HP experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-cooling", required=True)
    parser.add_argument("--program-cooling", required=True)
    parser.add_argument("--data-heating", default=None)
    parser.add_argument("--program-heating", default=None)
    parser.add_argument("--exclude-transition-window", nargs=2, type=float, default=(25.0, 30.0))
    parser.add_argument("--outdir", type=Path, default=Path("out/figures/step_hf"))
    parser.add_argument("--diagnostic-note", default=DIAGNOSTIC_DEFAULT)
    parser.add_argument("--transparent", action="store_true")
    parser.add_argument("--ramp-threshold", type=float, default=0.03, help="Fallback |dT/dt| threshold in °C/min.")
    parser.add_argument("--transient-exclude-min", type=float, default=1.0)
    parser.add_argument(
        "--baseline-mode",
        choices=["zone", "global"],
        default="zone",
        help="zone usa modelos locales por rampa/isoterma; global usa el modelo lineal anterior.",
    )
    parser.add_argument(
        "--hf-res-ylim",
        nargs=2,
        type=float,
        default=None,
        metavar=("YMIN", "YMAX"),
        help="Crear una figura extra de HF_res con limite Y fijo.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    transition_window = tuple(float(v) for v in args.exclude_transition_window)

    cooling_lvm = _resolve_input(args.data_cooling, "lvm", "Cooling .lvm")
    cooling_pdf = _resolve_input(args.program_cooling, "pdf", "Cooling program PDF")
    heating = None
    if args.data_heating:
        heating_lvm = _resolve_input(args.data_heating, "lvm", "Heating .lvm")
        heating_pdf = _resolve_input(args.program_heating, "pdf", "Heating program PDF") if args.program_heating else None
        heating = _prep_dataframe(heating_lvm, heating_pdf)

    cooling = _prep_dataframe(cooling_lvm, cooling_pdf)
    zones = _build_zones(cooling, cooling_pdf, args.ramp_threshold)
    if not zones:
        raise RuntimeError("Could not segment cooling run from PDF or derivative fallback.")
    cooling = _assign_zones(cooling, zones)
    if args.baseline_mode == "zone":
        processed, baseline_info = _fit_hf_baseline_by_zone(cooling, zones)
    else:
        processed, baseline_info = _fit_hf_baseline(
            cooling,
            exclude_window=transition_window,
            transient_exclude_min=args.transient_exclude_min,
        )

    stem = cooling_lvm.stem
    figures_out = args.outdir
    tables_out = Path("out/tables/step_hf")
    tables_out.mkdir(parents=True, exist_ok=True)

    summary = _zone_summary(processed, zones)
    relaxation = _isotherm_relaxation_table(processed, zones)
    summary.to_csv(tables_out / f"{stem}_zone_summary.csv", index=False)
    relaxation.to_csv(tables_out / f"{stem}_isotherm_relaxation.csv", index=False)
    processed[
        [
            "time_min",
            "T_sample_C",
            "T_furnace_C",
            "P_sample_bar",
            "HF_mW",
            "dTdt_C_min",
            "dPdt_bar_min",
            "zone_id",
            "zone_type",
            "HF_base_mW",
            "HF_res_mW",
        ]
    ].to_csv(tables_out / f"{stem}_step_hf_processed.csv", index=False)

    _plot_pt(processed, heating, figures_out, stem, args.transparent)
    _plot_diagnostic(processed, zones, figures_out, stem, args.diagnostic_note, args.transparent)
    _plot_hf_vs_t(processed, figures_out, stem, transition_window, args.transparent)
    _plot_hf_residual(processed, figures_out, stem, transition_window, args.transparent)
    if args.hf_res_ylim is not None:
        _plot_hf_residual_zoom(
            processed,
            figures_out,
            stem,
            transition_window,
            tuple(float(v) for v in args.hf_res_ylim),
            args.transparent,
        )

    print(f"Diagnostic note: {args.diagnostic_note}")
    print(f"Zones: {len(zones)}")
    print(f"Baseline fit: {baseline_info}")
    print(f"Figures: {figures_out}")
    print(f"Tables: {tables_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
