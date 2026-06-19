#!/usr/bin/env python3
"""Dynamic HF deconvolution for step/ramp calorimetry data.

The script can either:

1. Use an existing residual signal, e.g. HF_res_mW, and deconvolve it.
2. Fit a simple dynamic baseline from T and a first-order response to dT/dt,
   then deconvolve the remaining excess.

First-order inverse used here:

    q_excess ~= y + tau * dy/dt

where y is the measured residual/excess HF and tau is the calorimeter response
time in minutes. Integrated heat is reported in mJ.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd


logger = logging.getLogger("dynamic_hf_deconvolution")


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)


def load_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def numeric_col(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        raise KeyError(f"Missing column {column!r}. Available columns: {list(df.columns)}")
    return pd.to_numeric(df[column], errors="coerce")


def standardize(
    raw: pd.DataFrame,
    time_col: str,
    temp_col: str,
    hf_col: str,
    furnace_col: str | None,
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "time_min": numeric_col(raw, time_col),
            "T_sample_C": numeric_col(raw, temp_col),
            "HF_signal_mW": numeric_col(raw, hf_col),
        }
    )
    if furnace_col and furnace_col in raw.columns:
        df["T_furnace_C"] = numeric_col(raw, furnace_col)
    else:
        df["T_furnace_C"] = np.nan

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["time_min", "T_sample_C", "HF_signal_mW"])
    df = df.sort_values("time_min").drop_duplicates(subset=["time_min"]).reset_index(drop=True)
    if len(df) < 5:
        raise ValueError("At least 5 valid points are required.")

    t = df["time_min"].to_numpy(dtype=float)
    if not np.all(np.diff(t) > 0):
        raise ValueError("time_min must be strictly increasing after cleaning.")
    T = df["T_sample_C"].to_numpy(dtype=float)
    df["dTdt_C_min"] = np.gradient(T, t)
    return df


def exp_response(signal: np.ndarray, time_min: np.ndarray, tau_min: float) -> np.ndarray:
    if tau_min <= 0:
        raise ValueError("tau_min must be positive.")
    out = np.zeros_like(signal, dtype=float)
    out[0] = signal[0]
    for i in range(1, len(signal)):
        dt = max(float(time_min[i] - time_min[i - 1]), 0.0)
        alpha = np.exp(-dt / tau_min)
        out[i] = alpha * out[i - 1] + (1.0 - alpha) * signal[i]
    return out


def smooth_by_time(values: np.ndarray, time_min: np.ndarray, smooth_min: float) -> np.ndarray:
    """Centered moving average over a time window, ignoring NaNs."""
    values = values.astype(float, copy=False)
    if smooth_min <= 0:
        return values.copy()
    half = 0.5 * smooth_min
    out = np.empty_like(values, dtype=float)
    finite = np.isfinite(values)
    safe_values = np.where(finite, values, 0.0)
    csum = np.concatenate(([0.0], np.cumsum(safe_values)))
    ccount = np.concatenate(([0], np.cumsum(finite.astype(int))))
    left = 0
    right = 0
    for i, t_i in enumerate(time_min):
        while left < len(time_min) and time_min[left] < t_i - half:
            left += 1
        while right + 1 < len(time_min) and time_min[right + 1] <= t_i + half:
            right += 1
        total = csum[right + 1] - csum[left]
        count = ccount[right + 1] - ccount[left]
        out[i] = float(total / count) if count else float("nan")
    return out


def cumulative_heat_mJ(power_mW: np.ndarray, time_min: np.ndarray) -> np.ndarray:
    heat = np.zeros_like(power_mW, dtype=float)
    for i in range(1, len(power_mW)):
        dt = float(time_min[i] - time_min[i - 1])
        heat[i] = heat[i - 1] + 0.5 * (power_mW[i] + power_mW[i - 1]) * dt * 60.0
    return heat


def trapezoid_heat_mJ(power_mW: np.ndarray, time_min: np.ndarray) -> float:
    """Integral trapezoidal total; 1 mW min = 60 mJ."""
    if len(power_mW) < 2:
        return float("nan")
    return float(cumulative_heat_mJ(power_mW, time_min)[-1])


def fit_dynamic_baseline(
    df: pd.DataFrame,
    tau_min: float,
    transition_window: tuple[float, float],
    use_furnace: bool,
) -> tuple[pd.DataFrame, dict[str, float | str]]:
    out = df.copy()
    t = out["time_min"].to_numpy(dtype=float)
    dTdt = out["dTdt_C_min"].to_numpy(dtype=float)
    out["dTdt_conv_C_min"] = exp_response(dTdt, t, tau_min)

    design = {
        "c0": np.ones(len(out), dtype=float),
        "T_sample": out["T_sample_C"].to_numpy(dtype=float),
        "dTdt_conv": out["dTdt_conv_C_min"].to_numpy(dtype=float),
    }
    if use_furnace and out["T_furnace_C"].notna().any():
        design["T_furnace_minus_sample"] = (
            out["T_furnace_C"].to_numpy(dtype=float)
            - out["T_sample_C"].to_numpy(dtype=float)
        )
    xdf = pd.DataFrame(design)

    t_low, t_high = sorted(transition_window)
    fit_mask = (
        np.isfinite(out["HF_signal_mW"])
        & np.isfinite(xdf).all(axis=1)
        & ~out["T_sample_C"].between(t_low, t_high, inclusive="both")
    )
    if fit_mask.sum() < len(xdf.columns) + 3:
        raise ValueError("Too few points to fit dynamic baseline.")

    x = xdf.loc[fit_mask].to_numpy(dtype=float)
    y = out.loc[fit_mask, "HF_signal_mW"].to_numpy(dtype=float)
    coefs, *_ = np.linalg.lstsq(x, y, rcond=None)

    out["HF_base_dyn_mW"] = xdf.to_numpy(dtype=float) @ coefs
    out["HF_excess_meas_mW"] = out["HF_signal_mW"] - out["HF_base_dyn_mW"]

    info: dict[str, float | str] = {
        "mode": "dynamic_baseline",
        "tau_min": float(tau_min),
        "fit_points": float(fit_mask.sum()),
    }
    info.update({name: float(coef) for name, coef in zip(xdf.columns, coefs, strict=True)})
    return out, info


def deconv_only(df: pd.DataFrame, tau_min: float) -> tuple[pd.DataFrame, dict[str, float | str]]:
    out = df.copy()
    out["HF_base_dyn_mW"] = 0.0
    out["HF_excess_meas_mW"] = out["HF_signal_mW"]
    return out, {"mode": "deconv_only", "tau_min": float(tau_min), "fit_points": 0.0}


def deconvolve_first_order(df: pd.DataFrame, tau_min: float, smooth_min: float) -> pd.DataFrame:
    out = df.copy()
    t = out["time_min"].to_numpy(dtype=float)
    y = out["HF_excess_meas_mW"].to_numpy(dtype=float)
    y_smooth = smooth_by_time(y, t, smooth_min)
    dy_dt = np.gradient(y_smooth, t)

    out["HF_excess_smooth_mW"] = y_smooth
    out["HF_excess_deconv_mW"] = y_smooth + tau_min * dy_dt
    out["Q_residual_cum_mJ"] = cumulative_heat_mJ(y_smooth, t)
    out["Q_deconv_cum_mJ"] = cumulative_heat_mJ(
        out["HF_excess_deconv_mW"].to_numpy(dtype=float), t
    )
    return out


def integrate_window(df: pd.DataFrame, window: tuple[float, float]) -> dict[str, float]:
    t_low, t_high = sorted(window)
    sub = df.loc[df["T_sample_C"].between(t_low, t_high, inclusive="both")]
    if len(sub) < 2:
        return {
            "T_low_C": t_low,
            "T_high_C": t_high,
            "points": float(len(sub)),
            "Q_residual_mJ": np.nan,
            "Q_deconv_mJ": np.nan,
        }
    t = sub["time_min"].to_numpy(dtype=float)
    q_res = sub["HF_excess_smooth_mW"].to_numpy(dtype=float)
    q_dec = sub["HF_excess_deconv_mW"].to_numpy(dtype=float)
    return {
        "T_low_C": t_low,
        "T_high_C": t_high,
        "points": float(len(sub)),
        "t_start_min": float(t[0]),
        "t_end_min": float(t[-1]),
        "Q_residual_mJ": trapezoid_heat_mJ(q_res, t),
        "Q_deconv_mJ": trapezoid_heat_mJ(q_dec, t),
    }


def make_figures(
    df: pd.DataFrame,
    fig_dir: Path,
    out_prefix: str,
    window: tuple[float, float],
) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        logger.warning("matplotlib unavailable (%r); skipping figures.", exc)
        return []

    saved: list[str] = []
    fig_dir.mkdir(parents=True, exist_ok=True)
    t_low, t_high = sorted(window)

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True, constrained_layout=True)
    axes[0].plot(df["time_min"], df["T_sample_C"], lw=0.9)
    axes[0].set_ylabel("T_sample_C")
    axes[1].plot(df["time_min"], df["HF_signal_mW"], lw=0.8, label="HF input")
    axes[1].plot(df["time_min"], df["HF_base_dyn_mW"], lw=0.9, label="dynamic baseline")
    axes[1].set_ylabel("HF_mW")
    axes[1].legend(fontsize=8)
    axes[2].plot(df["time_min"], df["HF_excess_smooth_mW"], lw=0.8)
    axes[2].axhline(0.0, color="black", lw=0.7)
    axes[2].set_ylabel("excess mW")
    axes[3].plot(df["time_min"], df["HF_excess_deconv_mW"], lw=0.8)
    axes[3].axhline(0.0, color="black", lw=0.7)
    axes[3].set_ylabel("deconv mW")
    axes[3].set_xlabel("time_min")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.suptitle("Dynamic HF treatment")
    path = fig_dir / f"{out_prefix}_dynamic_hf_time.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved.append(path)

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.scatter(df["T_sample_C"], df["HF_excess_smooth_mW"], s=1.2, alpha=0.5, label="smoothed excess")
    ax.scatter(df["T_sample_C"], df["HF_excess_deconv_mW"], s=1.0, alpha=0.45, label="deconvolved")
    ax.axvspan(t_low, t_high, color="red", alpha=0.12, label=f"{t_low:g}-{t_high:g} C")
    ax.axhline(0.0, color="black", lw=0.7)
    ax.set_xlabel("T_sample_C")
    ax.set_ylabel("HF_excess_mW")
    ax.set_title("Calorimetric excess vs temperature")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    path = fig_dir / f"{out_prefix}_dynamic_hf_vs_T.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved.append(path)
    return saved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dynamic convolution/deconvolution of HF in step calorimetry.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", help="CSV/XLSX with time, temperature and HF")
    parser.add_argument("--time-col", default="time_min")
    parser.add_argument("--temp-col", default="T_sample_C")
    parser.add_argument("--hf-col", default="HF_mW")
    parser.add_argument("--furnace-col", default="T_furnace_C")
    parser.add_argument("--tau-min", type=float, default=120.0,
                        help="First-order calorimeter response time in minutes")
    parser.add_argument("--smooth-min", type=float, default=2.0,
                        help="Centered smoothing window before deconvolution")
    parser.add_argument("--transition-window", nargs=2, type=float, default=(25.0, 30.0),
                        metavar=("T_LOW", "T_HIGH"))
    parser.add_argument("--deconv-only", action="store_true",
                        help="Treat --hf-col as an already baseline-corrected residual")
    parser.add_argument("--no-furnace", action="store_true",
                        help="Do not include T_furnace - T_sample in the dynamic baseline")
    parser.add_argument("--out", default="dynamic_hf", help="Output filename prefix")
    parser.add_argument("--fig-dir", type=Path, default=Path("out/figures/step_hf"))
    parser.add_argument("--table-dir", type=Path, default=Path("out/tables/step_hf"))
    parser.add_argument("--no-figs", action="store_true")
    return parser


def main() -> int:
    configure_logging()
    args = build_parser().parse_args()
    window = tuple(float(v) for v in args.transition_window)

    try:
        raw = load_table(args.input)
        df = standardize(raw, args.time_col, args.temp_col, args.hf_col, args.furnace_col)
        if args.deconv_only:
            processed, info = deconv_only(df, args.tau_min)
        else:
            processed, info = fit_dynamic_baseline(
                df,
                tau_min=args.tau_min,
                transition_window=window,
                use_furnace=not args.no_furnace,
            )
        info["smooth_min"] = float(args.smooth_min)
        processed = deconvolve_first_order(processed, args.tau_min, args.smooth_min)
        summary = integrate_window(processed, window)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        logger.error("[ERROR] %s", exc)
        return 1

    args.table_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.table_dir / f"{args.out}_dynamic_hf.csv"
    summary_path = args.table_dir / f"{args.out}_dynamic_summary.csv"
    processed.to_csv(csv_path, index=False)
    pd.DataFrame([{**info, **summary}]).to_csv(summary_path, index=False)

    logger.info("[Model] mode = %s | tau = %.3f min | smooth = %.3f min", info["mode"], args.tau_min, args.smooth_min)
    logger.info("[IO] dynamic CSV saved: %s", csv_path)
    logger.info("[IO] summary saved: %s", summary_path)
    logger.info(
        "[Q] %.2f-%.2f C | Q residual = %.3f mJ | Q deconv = %.3f mJ",
        summary["T_low_C"],
        summary["T_high_C"],
        summary["Q_residual_mJ"],
        summary["Q_deconv_mJ"],
    )

    if not args.no_figs:
        for fig_path in make_figures(processed, args.fig_dir, args.out, window):
            logger.info("[IO] figure saved: %s", fig_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
