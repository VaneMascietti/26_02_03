import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from process_dsc_data import load_lvm, DEFAULT_PROGRAM

# NIST Phase Change Data (CO2) from NIST Chemistry WebBook, SRD 69
# https://webbook.nist.gov/cgi/cbook.cgi?ID=C124389&Mask=4
TRIPLE_T_K = 216.58
TRIPLE_P_BAR = 5.185
CRIT_T_K = 304.18
CRIT_P_BAR = 73.80
ONSET_T_C = 29.99
ONSET_P_BAR = 141.3

# Configura tus inputs aca y corre el script sin flags.
DEFAULT_CONFIG = {
    "data": "data/26_01_28_teste_oring_vitom.lvm",
    "program": DEFAULT_PROGRAM,
    "label": "Vitom + Niquel",
    "data2": None,
    "label2": "Nitrílica",
    "out": "output/compare_pt_path_nist/co2_phase_overlay.png",
    "nist_iso": [],
    "nist_iso_label": [],
    "nist_sat": "nist/co2_nist_sat.csv",
    "no_sat": False,
    "show_onset": False,
}

def _extract_density_from_rows(rows, header=None):
    if not rows:
        return None
    # Try header-based index if available
    if header:
        for idx, name in enumerate(header):
            name_l = name.lower()
            if "density" in name_l and "(l" in name_l and "g/ml" in name_l:
                vals = [
                    row[idx]
                    for row in rows
                    if idx < len(row) and row[idx] is not None
                ]
                if vals:
                    return float(np.median(np.array(vals, dtype=float)))

    max_len = max(len(r) for r in rows)
    best = None
    for j in range(2, max_len):
        vals = [r[j] for r in rows if len(r) > j and r[j] is not None]
        if len(vals) < max(5, len(rows) // 3):
            continue
        arr = np.array(vals, dtype=float)
        med = float(np.median(arr))
        if not (0.05 <= med <= 2.0):
            continue
        mad = float(np.median(np.abs(arr - med)))
        rel = mad / abs(med) if med != 0 else np.inf
        if best is None or rel < best[0]:
            best = (rel, med)
    return float(best[1]) if best else None


def load_nist_iso_file(path: Path):
    """
    Parse NIST IsoChor data (tab/space delimited or pasted text).
    Uses the first two numeric columns as Temperature (C) and Pressure (bar).
    Returns (df, density) where density is inferred from the file if possible.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    rows = []
    header = None
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        if ln[0] not in "0123456789-.":
            if header is None and "Temperature" in ln and "Pressure" in ln:
                header = re.split(r"\t+|\s{2,}", ln)
            continue
        parts = ln.replace(",", " ").split()
        vals = []
        for part in parts:
            try:
                vals.append(float(part))
            except ValueError:
                vals.append(None)
        rows.append(vals)
    if not rows:
        raise ValueError(f"No numeric rows found in {path}")
    t_list = []
    p_list = []
    for row in rows:
        nums = [v for v in row if v is not None]
        if len(nums) >= 2:
            t_list.append(nums[0])
            p_list.append(nums[1])
    if not t_list:
        raise ValueError(f"No T,P data found in {path}")
    df = pd.DataFrame({"T": t_list, "P": p_list})
    density = _extract_density_from_rows(rows, header)
    return df, density


def _infer_iso_label(path: Path) -> str:
    stem = path.stem
    m = re.search(r"[Dd](\d+)p(\d+)", stem)
    if m:
        return f"NIST isochore D={m.group(1)}.{m.group(2)} g/ml"
    m = re.search(r"[Dd]\s*=?\s*([0-9]*\.?[0-9]+)", stem)
    if m:
        return f"NIST isochore D={m.group(1)} g/ml"
    return f"NIST isochore {stem}"


def load_nist_sat_file(path: Path) -> pd.DataFrame:
    """
    Parse NIST vapor pressure (SatP) data from local file.
    Uses first two numeric columns as Temperature (C) and Pressure (bar).
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    rows = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        if ln[0] not in "0123456789-.":
            continue
        parts = ln.replace(",", " ").split()
        nums = []
        for part in parts:
            try:
                nums.append(float(part))
            except ValueError:
                continue
            if len(nums) >= 2:
                break
        if len(nums) >= 2:
            rows.append((nums[0], nums[1]))
    if not rows:
        raise ValueError(f"No numeric rows found in {path}")
    return pd.DataFrame(rows, columns=["T", "P"])


def plot_co2_phase_with_data(
    df,
    iso_curves,
    outpath,
    onset_point=None,
    label_data="Vitom + Niquel",
    df2=None,
    label_data2=None,
    sat_curve=None,
    title="CO2: P vs T with NIST isochores",
):
    fig, ax = plt.subplots(figsize=(8, 6))

    def _plot_experimental_track(df_track, label, line_kwargs, ref_kwargs):
        ax.plot(
            df_track["T_muestra"],
            df_track["P_muestra"],
            label=label,
            **line_kwargs,
        )
        if "P_referencia" in df_track.columns:
            ax.plot(
                df_track["T_muestra"],
                df_track["P_referencia"],
                label=f"{label} (Reference P)",
                **ref_kwargs,
            )

    # Experimental data
    _plot_experimental_track(
        df,
        label_data,
        line_kwargs={"color": "black", "linewidth": 1.2, "alpha": 0.9},
        ref_kwargs={
            "color": "dimgray",
            "linewidth": 1.2,
            "alpha": 0.9,
            "linestyle": ":",
        },
    )
    if df2 is not None:
        label2 = label_data2 or "Este trabajo (2)"
        _plot_experimental_track(
            df2,
            label2,
            line_kwargs={"linewidth": 1.2, "alpha": 0.85, "linestyle": "--"},
            ref_kwargs={"linewidth": 1.1, "alpha": 0.8, "linestyle": "-."},
        )
    if onset_point is not None:
        t_on, p_on = onset_point
        ax.scatter(
            [t_on],
            [p_on],
            color="#d35400",
            s=60,
            zorder=6,
            label=f"Onset ({label_data})",
        )

    # NIST isochoric curves
    for iso_df, label in iso_curves:
        ax.plot(
            iso_df["T"],
            iso_df["P"],
            linewidth=1.5,
            label=label,
        )
    if sat_curve is not None and not sat_curve.empty:
        ax.plot(
            sat_curve["T"],
            sat_curve["P"],
            linewidth=1.5,
            label="Vapor pressure (NIST)",
        )

    # Triple and critical points
    ax.scatter(
        [TRIPLE_T_K - 273.15],
        [TRIPLE_P_BAR],
        color="#cc0000",
        s=50,
        zorder=5,
        label="Triple point (NIST)",
    )
    crit_t_c = CRIT_T_K - 273.15
    crit_p = CRIT_P_BAR
    ax.scatter(
        [crit_t_c],
        [crit_p],
        color="#0b2e88",
        s=50,
        zorder=5,
        label="Critical point (NIST)",
    )
    # Reference lines from critical point (as in the example figure)
    ax.axvline(crit_t_c, color="#4a4a4a", linewidth=1.0, alpha=0.8)
    ax.axhline(crit_p, color="#4a4a4a", linewidth=1.0, alpha=0.8)

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Pressure (bar)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def run_with_config(cfg):
    data_path = Path(cfg["data"])
    program_value = cfg.get("program")
    program_path = Path(program_value) if program_value else None
    clean = load_lvm(data_path, program_pdf=program_path)
    clean2 = None
    if cfg.get("data2"):
        clean2 = load_lvm(Path(cfg["data2"]), program_pdf=program_path)

    outpath = Path(cfg["out"])
    outpath.parent.mkdir(parents=True, exist_ok=True)
    onset_point = (ONSET_T_C, ONSET_P_BAR) if cfg.get("show_onset", False) else None
    iso_curves = []
    nist_iso = cfg.get("nist_iso") or []
    nist_iso_label = cfg.get("nist_iso_label") or []
    if nist_iso:
        for i, iso_path in enumerate(nist_iso):
            path = Path(iso_path)
            iso_df, density = load_nist_iso_file(path)
            if i < len(nist_iso_label) and nist_iso_label[i]:
                label = nist_iso_label[i]
            elif density is not None:
                label = f"NIST isochore D={density:.5g} g/ml"
            else:
                label = _infer_iso_label(path)
            iso_curves.append((iso_df, label))

    sat_curve = None
    if not cfg.get("no_sat", False):
        sat_path = Path(cfg.get("nist_sat", "nist/co2_nist_sat.csv"))
        if sat_path.exists():
            sat_curve = load_nist_sat_file(sat_path)

    plot_co2_phase_with_data(
        clean,
        iso_curves,
        outpath,
        onset_point=onset_point,
        label_data=cfg.get("label", "Vitom + Niquel"),
        df2=clean2,
        label_data2=cfg.get("label2", "Nitrílica"),
        sat_curve=sat_curve,
    )
    print(f"Guardado: {outpath}")

def parse_args_to_config():
    import argparse

    parser = argparse.ArgumentParser(
        description="Grafica P vs T y superpone trayectorias isócoras de NIST."
    )
    parser.add_argument("--data", type=str, default=DEFAULT_CONFIG["data"])
    parser.add_argument("--program", type=str, default=DEFAULT_CONFIG["program"])
    parser.add_argument("--label", type=str, default=DEFAULT_CONFIG["label"])
    parser.add_argument("--data2", type=str, default=DEFAULT_CONFIG["data2"])
    parser.add_argument("--label2", type=str, default=DEFAULT_CONFIG["label2"])
    parser.add_argument("--out", type=str, default=DEFAULT_CONFIG["out"])
    parser.add_argument(
        "--nist-iso",
        action="append",
        default=DEFAULT_CONFIG["nist_iso"],
    )
    parser.add_argument(
        "--nist-iso-label",
        action="append",
        default=DEFAULT_CONFIG["nist_iso_label"],
    )
    parser.add_argument("--nist-sat", type=str, default=DEFAULT_CONFIG["nist_sat"])
    parser.add_argument("--no-sat", action="store_true")
    parser.add_argument("--show-onset", action="store_true")
    args = parser.parse_args()
    return {
        "data": args.data,
        "program": args.program,
        "label": args.label,
        "data2": args.data2,
        "label2": args.label2,
        "out": args.out,
        "nist_iso": args.nist_iso,
        "nist_iso_label": args.nist_iso_label,
        "nist_sat": args.nist_sat,
        "no_sat": args.no_sat,
        "show_onset": args.show_onset,
    }

def main():
    if len(sys.argv) > 1:
        cfg = parse_args_to_config()
    else:
        cfg = DEFAULT_CONFIG
    run_with_config(cfg)

if __name__ == "__main__":
    main()
