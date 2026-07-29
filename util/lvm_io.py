"""LVM parsing shared by analysis and plotting scripts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .program_zones import program_total_seconds


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


def load_lvm(path: Path, program_pdf: Path | None = None) -> pd.DataFrame:
    """Load an instrument LVM export using the repository's canonical columns."""
    df = pd.read_csv(path, sep="\t", header=None, engine="python")
    while df.shape[1] >= 1 and _series_is_effectively_blank(df.iloc[:, 0]):
        df = df.iloc[:, 1:].copy()
        df.columns = range(df.shape[1])

    trailing_sample_pressure = _uses_trailing_sample_pressure(df)
    datetime_column = df.shape[1] - 1
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

    df["P_muestra"] = df["P_muestra_bar"]
    if "P_referencia_bar" in df.columns:
        df["P_referencia"] = df["P_referencia_bar"]

    total_seconds = program_total_seconds(program_pdf) if program_pdf is not None else None
    if total_seconds is not None:
        df = df.loc[sample_time_s <= total_seconds].copy()
    return df
