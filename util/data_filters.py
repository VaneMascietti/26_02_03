"""Small, explicit numeric filters shared by plotting workflows."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd


Bounds = tuple[float | None, float | None]


def numeric_filter_mask(
    df: pd.DataFrame,
    columns: Iterable[str],
    bounds: Mapping[str, Bounds] | None = None,
    *,
    require_finite: bool = True,
) -> pd.Series:
    """Build an inclusive mask after numeric coercion without changing the frame."""
    column_list = list(columns)
    missing = [column for column in column_list if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    mask = pd.Series(True, index=df.index)
    for column in column_list:
        values = pd.to_numeric(df[column], errors="coerce")
        mask &= np.isfinite(values) if require_finite else values.notna()

    for column, (lower, upper) in (bounds or {}).items():
        if column not in df.columns:
            raise ValueError(f"Missing column: {column}")
        values = pd.to_numeric(df[column], errors="coerce")
        if lower is not None:
            mask &= values >= lower
        if upper is not None:
            mask &= values <= upper
    return mask


def filter_numeric_rows(
    df: pd.DataFrame,
    columns: Iterable[str],
    bounds: Mapping[str, Bounds] | None = None,
    *,
    require_finite: bool = True,
) -> pd.DataFrame:
    """Coerce selected columns and keep rows satisfying inclusive bounds."""
    column_list = list(columns)
    out = df.copy()
    for column in column_list:
        if column not in out.columns:
            raise ValueError(f"Missing column: {column}")
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out.loc[
        numeric_filter_mask(out, column_list, bounds, require_finite=require_finite)
    ].copy()


def filter_pt_rows(
    df: pd.DataFrame,
    *,
    temp_col: str = "T_muestra",
    pressure_col: str = "P_muestra",
    time_col: str = "t_h",
    t_bounds: Bounds = (None, None),
    p_bounds: Bounds = (None, None),
    require_time: bool = True,
    positive_pressure: bool = False,
) -> pd.DataFrame:
    """Return finite P-T rows with optional inclusive temperature/pressure bounds."""
    columns = [temp_col, pressure_col]
    if require_time:
        columns.append(time_col)
    out = filter_numeric_rows(
        df,
        columns,
        bounds={temp_col: t_bounds, pressure_col: p_bounds},
        require_finite=True,
    )
    if positive_pressure:
        out = out.loc[out[pressure_col] > 0].copy()
    return out


def select_time_span(
    df: pd.DataFrame,
    start_h: float,
    end_h: float,
    *,
    time_col: str = "t_h",
) -> pd.DataFrame:
    """Select an inclusive time interval without rebasing its time coordinate."""
    if time_col not in df.columns:
        raise ValueError(f"Missing column: {time_col}")
    return df.loc[(df[time_col] >= start_h) & (df[time_col] <= end_h)].copy()
