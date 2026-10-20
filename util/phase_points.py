"""Shared reader for curated experimental and literature P-T points."""
from pathlib import Path
import pandas as pd


def load_points(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    df = pd.read_csv(path)
    required = {"ID", "transition_type", "T_C", "P_bar"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df["T_C"] = pd.to_numeric(df["T_C"], errors="coerce")
    df["P_bar"] = pd.to_numeric(df["P_bar"], errors="coerce")
    if "xCO2" in df.columns:
        df["xCO2"] = pd.to_numeric(df["xCO2"], errors="coerce")
    if "Ciclo" not in df.columns:
        df["Ciclo"] = ""
    df["transition_type"] = df["transition_type"].astype(str)
    return df.dropna(subset=["T_C", "P_bar"]).copy()
