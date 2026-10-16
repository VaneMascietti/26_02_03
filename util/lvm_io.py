"""LVM parsing shared by analysis and plotting scripts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .program_zones import program_total_seconds


LVM_SIGNAL_UNITS = {
    "P_bomba_bar": "bar",
    "P_muestra": "bar",
    "P_referencia": "bar",
    "V_bomba_ml": "mL",
    "T_muestra": "degC",
    "T_horno": "degC",
    "T_banho": "degC",
    "HF_mW": "mW",
    "HF_uV": "uV",
    "t_h": "h",
    "t_s": "s",
    "t_s_datetime": "s",
    "t_s_sample_index": "s",
}


def lvm_time_diagnostics(df: pd.DataFrame) -> dict[str, float | int | str]:
    """Compare device-clock elapsed time with the legacy sample-index time."""
    required = {"t_s_datetime", "t_s_sample_index"}
    if not required.issubset(df.columns):
        return {
            "status": "not_determined",
            "reason": "LVM frame does not expose both device-clock and sample-index time.",
        }
    clock = pd.to_numeric(df["t_s_datetime"], errors="coerce").to_numpy(float)
    index_time = pd.to_numeric(df["t_s_sample_index"], errors="coerce").to_numpy(float)
    valid = np.isfinite(clock) & np.isfinite(index_time)
    if valid.sum() < 2:
        return {
            "status": "not_determined",
            "reason": "Fewer than two finite timestamps are available.",
        }
    clock = clock[valid]
    index_time = index_time[valid]
    clock_steps = np.diff(clock)
    positive_steps = clock_steps[clock_steps > 0]
    drift = clock - index_time
    return {
        "status": "diagnostic_only",
        "sample_interval_assumption_s": 0.5,
        "device_clock_median_interval_s": (
            float(np.median(positive_steps)) if len(positive_steps) else float("nan")
        ),
        "device_clock_nonpositive_intervals": int(np.sum(clock_steps <= 0)),
        "device_clock_gaps_over_5x_median": (
            int(np.sum(clock_steps > 5.0 * np.median(positive_steps)))
            if len(positive_steps)
            else 0
        ),
        "device_clock_minus_sample_index_start_s": float(drift[0]),
        "device_clock_minus_sample_index_end_s": float(drift[-1]),
        "device_clock_minus_sample_index_min_s": float(np.min(drift)),
        "device_clock_minus_sample_index_max_s": float(np.max(drift)),
        "program_time_offset_applied_s": 0.0,
        "automatic_synchronization_applied": 0,
    }


def _series_is_effectively_blank(series: pd.Series) -> bool:
    normalized = (
        series.fillna("")
        .astype(str)
        .str.replace("\x00", "", regex=False)
        .str.strip()
    )
    return normalized.eq("").all()


def _drop_leading_counter_wrap(df: pd.DataFrame) -> pd.DataFrame:
    """Trim a short stale prefix left by an instrument-counter wrap."""
    if "_col4" not in df.columns or len(df) < 2:
        return df

    counter = pd.to_numeric(df["_col4"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(counter[0]):
        return df

    wrap_points = np.where(np.diff(counter) < -100.0)[0] + 1
    if wrap_points.size == 0:
        return df

    first_wrap = int(wrap_points[0])
    if first_wrap > 5 or not np.isfinite(counter[first_wrap]):
        return df
    if counter[0] < 100.0 or counter[first_wrap] > 10.0:
        return df
    return df.iloc[first_wrap:].reset_index(drop=True)


def _uses_trailing_sample_pressure(raw_df: pd.DataFrame) -> bool:
    """Identify 10-column exports with sample pressure in the trailing column."""
    if raw_df.shape[1] < 10:
        return False

    column_2 = pd.to_numeric(raw_df.iloc[:, 2], errors="coerce").dropna()
    column_8 = pd.to_numeric(raw_df.iloc[:, 8], errors="coerce").dropna()
    if column_2.empty or column_8.empty:
        return False

    q2_25, q2_75 = column_2.quantile([0.25, 0.75])
    q8_05, q8_25, q8_75, q8_95 = column_8.quantile([0.05, 0.25, 0.75, 0.95])
    column_2_iqr = float(q2_75 - q2_25)
    column_8_iqr = float(q8_75 - q8_25)
    trailing_is_pressure = 1.0 <= float(q8_05) and float(q8_95) <= 1000.0
    leading_is_nearly_constant = column_2_iqr <= 1.0
    trailing_has_signal = column_8_iqr >= 2.0
    return trailing_is_pressure and leading_is_nearly_constant and trailing_has_signal


def _uses_pressure_after_heatflow(raw_df: pd.DataFrame) -> bool:
    """Identify exports where the sample pressure follows the heat-flow columns."""
    if raw_df.shape[1] < 10:
        return False

    column_2 = pd.to_numeric(raw_df.iloc[:, 2], errors="coerce").dropna()
    column_7 = pd.to_numeric(raw_df.iloc[:, 7], errors="coerce").dropna()
    column_8 = pd.to_numeric(raw_df.iloc[:, 8], errors="coerce").dropna()
    if column_2.empty or column_7.empty or column_8.empty:
        return False

    q2_25, q2_75 = column_2.quantile([0.25, 0.75])
    q7_05, q7_25, q7_75, q7_95 = column_7.quantile([0.05, 0.25, 0.75, 0.95])
    q8_05, q8_95 = column_8.quantile([0.05, 0.95])
    column_2_iqr = float(q2_75 - q2_25)
    column_7_iqr = float(q7_75 - q7_25)

    counter_like = column_2_iqr >= 100.0
    pressure_like = 1.0 <= float(q7_05) and float(q7_95) <= 1000.0
    pressure_has_signal = column_7_iqr >= 2.0
    following_column_not_pressure = float(q8_05) < 1.0 or float(q8_95) < 100.0
    return counter_like and pressure_like and pressure_has_signal and following_column_not_pressure


def load_lvm(path: Path, program_pdf: Path | None = None) -> pd.DataFrame:
    """Load an instrument LVM export using the repository's canonical columns."""
    df = pd.read_csv(path, sep="\t", header=None, engine="python")
    while df.shape[1] >= 1 and _series_is_effectively_blank(df.iloc[:, 0]):
        df = df.iloc[:, 1:].copy()
        df.columns = range(df.shape[1])

    pressure_after_heatflow = _uses_pressure_after_heatflow(df)
    trailing_sample_pressure = (
        False if pressure_after_heatflow else _uses_trailing_sample_pressure(df)
    )
    datetime_column = df.shape[1] - 1
    if pressure_after_heatflow:
        rename_map = {
            0: "P_bomba_bar",
            1: "V_bomba_ml",
            2: "_col4",
            3: "T_muestra",
            4: "T_horno",
            5: "HF_mW",
            6: "HF_uV",
            7: "P_muestra_bar",
            8: "T_banho",
            datetime_column: "datetime",
        }
    else:
        rename_map = {
            0: "P_bomba_bar",
            1: "V_bomba_ml",
            2: "P_referencia_bar" if trailing_sample_pressure else "P_muestra_bar",
            3: "_col4",
            4: "T_muestra",
            5: "T_horno",
            6: "HF_uV",
            7: "HF_mW",
            datetime_column: "datetime",
        }
        if df.shape[1] >= 10:
            rename_map[datetime_column - 1] = (
                "P_muestra_bar" if trailing_sample_pressure else "P_referencia_bar"
            )
    df = df.rename(columns=rename_map)
    df = _drop_leading_counter_wrap(df)

    df["datetime"] = pd.to_datetime(
        df["datetime"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce"
    )
    df = df.loc[df["datetime"].notna()].copy()
    if df.empty:
        raise ValueError(f"No valid datetime rows found in {path}")
    df = df.reset_index(drop=True)
    initial_datetime = df["datetime"].iloc[0]
    df["t_s"] = (df["datetime"] - initial_datetime).dt.total_seconds()
    sample_time_s = np.arange(len(df), dtype=float) * 0.5
    df["t_h"] = sample_time_s / 3600.0
    df["t_s_datetime"] = df["t_s"]
    df["t_s_sample_index"] = sample_time_s
    df["t_h_original"] = df["t_h"]

    df["P_muestra"] = df["P_muestra_bar"]
    if "P_referencia_bar" in df.columns:
        df["P_referencia"] = df["P_referencia_bar"]

    total_seconds = program_total_seconds(program_pdf) if program_pdf is not None else None
    if total_seconds is not None:
        df = df.loc[sample_time_s <= total_seconds].copy()
    return df
