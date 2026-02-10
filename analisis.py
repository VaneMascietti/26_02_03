import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import re

DEFAULT_DATA = "data/26_01_28_teste_oring_vitom.lvm"
DEFAULT_PROGRAM = "data/26_01_28_teste_oring_vitom.pdf"


def _dedupe_repeated_suffix(path: Path) -> Path:
    """Collapse paths like foo.lvm.lvm -> foo.lvm."""
    fixed = path
    while (
        len(fixed.suffixes) >= 2
        and fixed.suffixes[-1].lower() == fixed.suffixes[-2].lower()
    ):
        fixed = Path(str(fixed)[: -len(fixed.suffix)])
    return fixed


def _resolve_input_path(path: Path, expected_suffix: str | None = None) -> Path:
    """
    Try common user input variants and return the first existing candidate.
    Falls back to the original path if nothing exists.
    """
    candidates = []

    def add_candidate(candidate: Path) -> None:
        if candidate not in candidates:
            candidates.append(candidate)

    add_candidate(path)
    deduped = _dedupe_repeated_suffix(path)
    add_candidate(deduped)

    if expected_suffix:
        expected_suffix = expected_suffix.lower()
        for base in list(candidates):
            if not base.name.lower().endswith(expected_suffix):
                add_candidate(Path(f"{base}{expected_suffix}"))
            # Fix names like foo.lvm.pdf -> foo.pdf
            if base.suffix.lower() == expected_suffix and len(base.suffixes) >= 2:
                add_candidate(base.with_suffix("").with_suffix(expected_suffix))

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return path

def load_lvm(path: Path, program_pdf: Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, engine="python")
    # LVM can start with a leading tab, creating an empty first column
    if df.shape[1] >= 1:
        first = df.iloc[:, 0]
        if first.isna().all() or first.astype(str).str.strip().eq("").all():
            df = df.iloc[:, 1:].copy()
            df.columns = range(df.shape[1])

    # Layout histórico: 9 columnas (última = datetime).
    # Layout nuevo: 10 columnas (penúltima = P de referencia, última = datetime).
    datetime_col = df.shape[1] - 1
    rename_map = {
        0: "P_bomba_bar",
        1: "V_bomba_ml",
        2: "P_muestra_bar",
        3: "_col4",
        4: "T_muestra",
        5: "T_horno",
        6: "HF_uV",
        7: "HF_mW",
        datetime_col: "datetime",
    }
    if df.shape[1] >= 10:
        rename_map[datetime_col - 1] = "P_referencia_bar"
    df = df.rename(columns=rename_map)

    df["datetime"] = pd.to_datetime(
        df["datetime"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce"
    )
    t0 = df["datetime"].iloc[0]
    df["t_s"] = (df["datetime"] - t0).dt.total_seconds()
    sample_time_s = np.arange(len(df), dtype=float) * 0.5
    df["t_h"] = sample_time_s / 3600.0

    df["P_muestra"] = df["P_muestra_bar"]
    if "P_referencia_bar" in df.columns:
        df["P_referencia"] = df["P_referencia_bar"]

    total_s = program_total_seconds(program_pdf) if program_pdf is not None else None
    if total_s is not None:
        df = df.loc[sample_time_s <= total_s].copy()

    return df

def detect_ramp_turns(t_h, y, min_sep_h=5.0):
    y = np.asarray(y, dtype=float)
    t = np.asarray(t_h, dtype=float)

    # suavizado simple
    if len(y) >= 11:
        k = 11
        ys = np.convolve(y, np.ones(k)/k, mode="same")
    else:
        ys = y

    dy = np.gradient(ys, t)
    s = np.sign(dy)
    s[s == 0] = 1
    changes = np.where(np.diff(s) != 0)[0] + 1

    turns = []
    last = -1e9
    for idx in changes:
        th = t[idx]
        if th - last < min_sep_h:
            continue
        if th < t.min() + 1 or th > t.max() - 1:
            continue
        turns.append(th)
        last = th
    return turns

def detect_isotherm_spans(t_h, y, max_rate_c_per_h=1.0, min_duration_h=0.0):
    t = np.asarray(t_h, dtype=float)
    y = np.asarray(y, dtype=float)

    # suavizado simple
    if len(y) >= 11:
        k = 11
        ys = np.convolve(y, np.ones(k) / k, mode="same")
    else:
        ys = y

    dy = np.gradient(ys, t)
    mask = np.abs(dy) <= max_rate_c_per_h

    spans = []
    if mask.any():
        idx = np.where(mask)[0]
        breaks = np.where(np.diff(idx) > 1)[0]
        start = idx[0]
        for b in breaks:
            end = idx[b]
            t0, t1 = t[start], t[end]
            if t1 - t0 >= min_duration_h:
                spans.append((t0, t1))
            start = idx[b + 1]
        end = idx[-1]
        t0, t1 = t[start], t[end]
        if t1 - t0 >= min_duration_h:
            spans.append((t0, t1))

    return spans

def detect_zones_from_temperature(
    t_h,
    y,
    rate_thresh_c_per_h=1.0,
    smooth_window=101,
    min_duration_h=0.05,
):
    t = np.asarray(t_h, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(y) < 3:
        return []

    if smooth_window < 3:
        smooth_window = 3
    if smooth_window % 2 == 0:
        smooth_window += 1
    if len(y) < smooth_window:
        smooth_window = max(3, len(y) // 2 * 2 + 1)

    if smooth_window >= 3:
        ys = np.convolve(y, np.ones(smooth_window) / smooth_window, mode="same")
    else:
        ys = y

    dy = np.gradient(ys, t)
    labels = np.where(
        dy > rate_thresh_c_per_h,
        "Heating",
        np.where(dy < -rate_thresh_c_per_h, "Cooling", "Isoterma"),
    )

    segments = []
    start = 0
    current = labels[0]
    for i in range(1, len(labels)):
        if labels[i] != current:
            segments.append((current, start, i - 1))
            current = labels[i]
            start = i
    segments.append((current, start, len(labels) - 1))

    zones = []
    for idx, (typ, s, e) in enumerate(segments, 1):
        zones.append(
            {
                "num": idx,
                "type": typ,
                "start_h": float(t[s]),
                "end_h": float(t[e]),
            }
        )

    def duration(z):
        return z["end_h"] - z["start_h"]

    if min_duration_h > 0 and len(zones) > 1:
        while True:
            short_idx = None
            shortest = None
            for i, z in enumerate(zones):
                d = duration(z)
                if d < min_duration_h:
                    if shortest is None or d < shortest:
                        shortest = d
                        short_idx = i
            if short_idx is None:
                break
            if short_idx == 0:
                zones[1]["start_h"] = zones[0]["start_h"]
                zones.pop(0)
            elif short_idx == len(zones) - 1:
                zones[-2]["end_h"] = zones[-1]["end_h"]
                zones.pop(-1)
            else:
                prev_d = duration(zones[short_idx - 1])
                next_d = duration(zones[short_idx + 1])
                if prev_d >= next_d:
                    zones[short_idx - 1]["end_h"] = zones[short_idx]["end_h"]
                    zones.pop(short_idx)
                else:
                    zones[short_idx + 1]["start_h"] = zones[short_idx]["start_h"]
                    zones.pop(short_idx)

    for i, z in enumerate(zones, 1):
        z["num"] = i

    return zones

def smooth_series(y, window):
    if window < 3:
        return y
    if window % 2 == 0:
        window += 1
    if len(y) < window:
        window = max(3, len(y) // 2 * 2 + 1)
    pad = window // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    return np.convolve(ypad, np.ones(window) / window, mode="valid")

def _savgol_coeffs(window, polyorder):
    if window < 3:
        raise ValueError("window must be >= 3")
    if window % 2 == 0:
        raise ValueError("window must be odd")
    if polyorder >= window:
        raise ValueError("polyorder must be < window")
    half = window // 2
    x = np.arange(-half, half + 1, dtype=float)
    # Vandermonde with powers 0..polyorder
    a = np.vander(x, polyorder + 1, increasing=True)
    # Pseudoinverse gives polynomial coefficients; first row yields value at x=0
    coeffs = np.linalg.pinv(a)[0]
    return coeffs

def _savgol_smooth(y, window, polyorder):
    y = np.asarray(y, dtype=float)
    if len(y) < 3:
        return y
    if window % 2 == 0:
        window += 1
    if window > len(y):
        window = max(3, len(y) // 2 * 2 + 1)
    if polyorder >= window:
        polyorder = max(1, window - 2)
    coeffs = _savgol_coeffs(window, polyorder)
    pad = window // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    return np.convolve(ypad, coeffs[::-1], mode="valid")

def _linear_fit(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return np.nan, np.nan
    b, a = np.polyfit(x, y, 1)
    return a, b

def _max_consecutive_true(mask):
    max_run = 0
    run = 0
    for v in mask:
        if v:
            run += 1
            if run > max_run:
                max_run = run
        else:
            run = 0
    return max_run

def _interpolate_crossing(x, y, target, direction):
    # direction: "up" for y increasing to target, "down" for decreasing
    for i in range(1, len(x)):
        y0, y1 = y[i - 1], y[i]
        if direction == "up":
            if (y0 < target <= y1) or (y0 > target >= y1):
                if y1 == y0:
                    return x[i]
                t = (target - y0) / (y1 - y0)
                return x[i - 1] + t * (x[i] - x[i - 1])
        else:
            if (y0 > target >= y1) or (y0 < target <= y1):
                if y1 == y0:
                    return x[i]
                t = (target - y0) / (y1 - y0)
                return x[i - 1] + t * (x[i] - x[i - 1])
    return np.nan

def compute_onset_tangent(
    x,
    y,
    event_polarity="auto",
    params=None,
):
    """
    Compute onset using the standard tangent method.

    Parameters
    ----------
    x, y : 1D arrays
        Monotonic x (time or temperature) and heat flow y.
    event_polarity : {"auto","endo","exo"}
        Endo = positive peak, Exo = negative peak.
    params : dict
        Optional parameters. Keys:
        - smooth_window_pts (odd int, default 51)
        - smooth_polyorder (default 3)
        - baseline_window_mode ("auto" or "manual", default "auto")
        - baseline_manual_range (x1, x2) if manual
        - search_flank_mode ("auto" or "manual", default "auto")
        - flank_manual_range (x1, x2) if manual
        - flank_force_near_peak (default True)
        - flank_near_peak_frac (default 0.15)
        - noise_k (default 5)
        - min_consecutive_pts (default 10)
    """
    defaults = {
        "smooth_window_pts": 51,
        "smooth_polyorder": 3,
        "baseline_window_mode": "auto",
        "baseline_manual_range": None,
        "baseline_force_near_peak": False,
        "baseline_near_peak_frac": 0.15,
        "search_flank_mode": "auto",
        "flank_manual_range": None,
        "flank_force_near_peak": True,
        "flank_near_peak_frac": 0.15,
        "noise_k": 5.0,
        "min_consecutive_pts": 10,
        "baseline_window_frac": 0.15,
        "baseline_min_pts": 20,
        "baseline_pre_margin_frac": 0.05,
        "flank_window_frac": 0.25,
        "flank_min_pts": 10,
        "max_tries": 3,
    }
    p = defaults if params is None else {**defaults, **params}

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y):
        raise ValueError("x and y must have same length")

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 5:
        return {
            "ok": False,
            "reason": "not_enough_points",
            "x_onset": np.nan,
            "y_onset": np.nan,
        }

    if np.any(np.diff(x) < 0):
        idx = np.argsort(x)
        x = x[idx]
        y = y[idx]

    x_min, x_max = float(x[0]), float(x[-1])
    x_range = x_max - x_min if x_max > x_min else 1.0

    y_s = _savgol_smooth(y, p["smooth_window_pts"], p["smooth_polyorder"])

    med = np.median(y_s)
    y_max = np.max(y_s)
    y_min = np.min(y_s)
    if event_polarity == "auto":
        if abs(y_max - med) >= abs(y_min - med):
            polarity = "endo"
        else:
            polarity = "exo"
    else:
        polarity = event_polarity

    if polarity == "endo":
        peak_idx = int(np.argmax(y_s))
    else:
        peak_idx = int(np.argmin(y_s))
    x_peak = float(x[peak_idx])
    y_peak = float(y_s[peak_idx])

    flank_margin = p["baseline_pre_margin_frac"] * x_range
    pre_mask = x <= (x_peak - flank_margin)
    if pre_mask.sum() < p["baseline_min_pts"]:
        early_mask = x <= (x_min + 0.2 * x_range)
        if early_mask.sum() >= p["baseline_min_pts"]:
            pre_mask = early_mask
        else:
            pre_mask = np.arange(len(x)) < min(len(x), p["baseline_min_pts"])

    # noise sigma (MAD) from early region
    sigma_mask = pre_mask
    if isinstance(sigma_mask, np.ndarray) and sigma_mask.dtype == bool:
        y_sigma = y_s[sigma_mask]
    else:
        y_sigma = y_s[sigma_mask]
    mad = np.median(np.abs(y_sigma - np.median(y_sigma)))
    sigma = 1.4826 * mad if mad > 0 else np.std(y_sigma)
    if sigma <= 0:
        sigma = 1e-9

    baseline_candidates = []
    if p["baseline_window_mode"] == "manual" and p["baseline_manual_range"] is not None:
        x1, x2 = p["baseline_manual_range"]
        bmask = (x >= x1) & (x <= x2)
        if bmask.sum() >= 2:
            baseline_candidates.append((0.0, np.where(bmask)[0][0], np.where(bmask)[0][-1]))
    else:
        pre_end = int(np.max(np.where(pre_mask)[0]))
        win = int(max(p["baseline_min_pts"], p["baseline_window_frac"] * (pre_end + 1)))
        win = min(win, pre_end + 1)
        if win >= 2:
            slopes = []
            resids = []
            starts = []
            for s in range(0, pre_end - win + 2):
                e = s + win - 1
                xw = x[s : e + 1]
                yw = y_s[s : e + 1]
                a, b = _linear_fit(xw, yw)
                fit = a + b * xw
                resid = yw - fit
                slopes.append(abs(b))
                resids.append(np.std(resid))
                starts.append((s, e))
            slope_scale = np.median(slopes) if np.median(slopes) > 0 else np.max(slopes) or 1.0
            resid_scale = np.median(resids) if np.median(resids) > 0 else np.max(resids) or 1.0
            for (s, e), sl, rs in zip(starts, slopes, resids):
                score = (sl / slope_scale) + (rs / resid_scale)
                baseline_candidates.append((score, s, e))
            baseline_candidates.sort(key=lambda t: t[0])
            if baseline_candidates and p.get("baseline_force_near_peak", True):
                # Prefer baseline windows in the final pre-peak segment so the
                # fitted baseline is not too far from the event.
                near_frac = float(p.get("baseline_near_peak_frac", 0.25))
                near_frac = min(max(near_frac, 0.01), 1.0)
                pre_start_x = float(x[0])
                pre_end_x = float(x[pre_end])
                near_start_x = pre_end_x - near_frac * (pre_end_x - pre_start_x)
                near_candidates = [
                    (score, s, e)
                    for (score, s, e) in baseline_candidates
                    if float(x[e]) >= near_start_x
                ]
                if near_candidates:
                    baseline_candidates = near_candidates

    if not baseline_candidates:
        return {
            "ok": False,
            "reason": "baseline_not_found",
            "x_onset": np.nan,
            "y_onset": np.nan,
        }

    def _attempt_with_window(s_idx, e_idx, flank_factor):
        xw = x[s_idx : e_idx + 1]
        yw = y_s[s_idx : e_idx + 1]
        a_base, b_base = _linear_fit(xw, yw)
        x_base_start = float(xw[0])
        x_base_end = float(xw[-1])

        if p["search_flank_mode"] == "manual" and p["flank_manual_range"] is not None:
            fx1, fx2 = p["flank_manual_range"]
            fmask = (x >= fx1) & (x <= fx2)
        else:
            pre_dist = max(1e-12, x_peak - x_base_end)
            dx = np.median(np.diff(x)) if len(x) > 1 else x_range / max(len(x), 1)
            min_width = max(dx * p["flank_min_pts"], 0.02 * x_range)
            w = max(p["flank_window_frac"] * pre_dist * flank_factor, min_width)
            f_start = max(x_base_end, x_peak - w)
            if p.get("flank_force_near_peak", True):
                near_frac = float(p.get("flank_near_peak_frac", 0.15))
                near_frac = min(max(near_frac, 0.01), 1.0)
                near_start = x_peak - near_frac * pre_dist
                f_start = max(f_start, near_start)
            fmask = (x >= f_start) & (x <= x_peak)
            if fmask.sum() < 3:
                # fallback: if near-peak constraint became too restrictive,
                # revert to the width-only window to keep the fit feasible.
                f_start = max(x_base_end, x_peak - w)
                fmask = (x >= f_start) & (x <= x_peak)

        if fmask.sum() < 3:
            return None

        dy_dx = np.gradient(y_s, x)
        if polarity == "endo":
            idx0 = int(np.argmax(dy_dx[fmask]))
        else:
            idx0 = int(np.argmin(dy_dx[fmask]))
        idxs = np.where(fmask)[0]
        idx0 = idxs[idx0]
        x0 = float(x[idx0])
        y0 = float(y_s[idx0])
        m_tan = float(dy_dx[idx0])

        denom = (b_base - m_tan)
        if abs(denom) < 1e-12:
            return None
        x_on = (y0 - m_tan * x0 - a_base) / denom
        y_on = a_base + b_base * x_on

        # baseline deviation for validation
        base_all = a_base + b_base * x
        dev = y_s - base_all
        if polarity == "endo":
            cond_base = dev >= p["noise_k"] * sigma
        else:
            cond_base = dev <= -p["noise_k"] * sigma

        ok = True
        reasons = []
        if not (x_min <= x_on <= x_peak and x_on <= x0):
            ok = False
            reasons.append("onset_out_of_bounds")

        cond = cond_base & (x >= x_on)
        max_run = _max_consecutive_true(cond)
        if max_run < p["min_consecutive_pts"]:
            ok = False
            reasons.append("insufficient_deviation")

        # Onset 2: intersection between baseline and linear regression on the flank
        a_flank, b_flank = _linear_fit(x[fmask], y_s[fmask])
        x_on2 = np.nan
        y_on2 = np.nan
        ok2 = True
        reasons2 = []
        denom2 = (b_base - b_flank)
        if abs(denom2) < 1e-12:
            ok2 = False
            reasons2.append("onset2_parallel_lines")
        else:
            x_on2 = (a_flank - a_base) / denom2
            y_on2 = a_base + b_base * x_on2
            if not (x_min <= x_on2 <= x_peak):
                ok2 = False
                reasons2.append("onset2_out_of_bounds")
        if ok2:
            cond2 = cond_base & (x >= x_on2)
            max_run2 = _max_consecutive_true(cond2)
            if max_run2 < p["min_consecutive_pts"]:
                ok2 = False
                reasons2.append("onset2_insufficient_deviation")

        return {
            "ok": ok,
            "reasons": reasons,
            "x_onset": float(x_on),
            "y_onset": float(y_on),
            "x_onset2": float(x_on2) if np.isfinite(x_on2) else np.nan,
            "y_onset2": float(y_on2) if np.isfinite(y_on2) else np.nan,
            "onset2_ok": ok2,
            "onset2_reasons": reasons2,
            "a_flank": float(a_flank),
            "b_flank": float(b_flank),
            "x0": x0,
            "y0": y0,
            "m_tan": m_tan,
            "a_base": float(a_base),
            "b_base": float(b_base),
            "baseline_window": (x_base_start, x_base_end),
            "flank_mask": fmask,
        }

    best = None
    tries = 0
    for score, s_idx, e_idx in baseline_candidates:
        for factor in (1.0, 1.5, 2.0):
            tries += 1
            if tries > p["max_tries"]:
                break
            res = _attempt_with_window(s_idx, e_idx, factor)
            if res is None:
                continue
            if best is None:
                best = res
            if res["ok"]:
                best = res
                tries = p["max_tries"]
                break
        if tries > p["max_tries"]:
            break

    if best is None:
        return {
            "ok": False,
            "reason": "tangent_not_found",
            "x_onset": np.nan,
            "y_onset": np.nan,
        }

    # T_char_50 on near-side flank
    base_all = best["a_base"] + best["b_base"] * x
    dev = y_s - base_all
    amp = y_peak - (best["a_base"] + best["b_base"] * x_peak)
    if polarity == "exo":
        amp = -amp
    amp = float(amp)
    t_char_50 = np.nan
    if amp > 0:
        target = 0.5 * amp
        if polarity == "endo":
            tmask = (x >= best["x_onset"]) & (x <= x_peak)
            t_char_50 = _interpolate_crossing(x[tmask], dev[tmask], target, "up")
        else:
            tmask = (x >= best["x_onset"]) & (x <= x_peak)
            t_char_50 = _interpolate_crossing(x[tmask], -dev[tmask], target, "up")

    flank_window = (np.nan, np.nan)
    fmask_best = best.get("flank_mask", None)
    if isinstance(fmask_best, np.ndarray) and fmask_best.dtype == bool and fmask_best.any():
        idx_f = np.where(fmask_best)[0]
        flank_window = (float(x[idx_f[0]]), float(x[idx_f[-1]]))

    result = {
        "ok": best["ok"],
        "reason": ",".join(best.get("reasons", [])) if not best["ok"] else "",
        "polarity": polarity,
        "x_onset": best["x_onset"],
        "y_onset": best["y_onset"],
        "x_onset2": best.get("x_onset2", np.nan),
        "y_onset2": best.get("y_onset2", np.nan),
        "onset2_ok": best.get("onset2_ok", False),
        "onset2_reason": ",".join(best.get("onset2_reasons", []))
        if not best.get("onset2_ok", False)
        else "",
        "a_flank": best.get("a_flank", np.nan),
        "b_flank": best.get("b_flank", np.nan),
        "x_peak": x_peak,
        "y_peak": y_peak,
        "x0": best["x0"],
        "y0": best["y0"],
        "m_tan": best["m_tan"],
        "a_base": best["a_base"],
        "b_base": best["b_base"],
        "baseline_window": best["baseline_window"],
        "flank_window": flank_window,
        # Se reporta por separado para facilitar trazabilidad en informes.
        "baseline_window_onset": best["baseline_window"],
        "flank_window_onset": flank_window,
        "baseline_window_onset2": best["baseline_window"],
        "flank_window_onset2": flank_window,
        "sigma": float(sigma),
        "t_char_50": float(t_char_50),
        "y_s": y_s,
    }
    return result

def plot_onset_tangent(x, y, result, ax=None):
    """
    Minimal helper to plot raw/smoothed signal with baseline and tangent.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.plot(x, y, color="gray", alpha=0.4, linewidth=1, label="raw")
    if "y_s" in result:
        ax.plot(x, result["y_s"], color="black", linewidth=1.2, label="smooth")
    if np.isfinite(result.get("a_base", np.nan)):
        a = result["a_base"]
        b = result["b_base"]
        ax.plot(x, a + b * x, color="green", linewidth=1, label="baseline")
    if np.isfinite(result.get("m_tan", np.nan)):
        x0 = result["x0"]
        y0 = result["y0"]
        m = result["m_tan"]
        xt = np.array([x0 - 0.1 * (x[-1] - x[0]), x0 + 0.1 * (x[-1] - x[0])])
        yt = y0 + m * (xt - x0)
        ax.plot(xt, yt, color="orange", linewidth=1, label="tangent")
    if np.isfinite(result.get("x_onset", np.nan)):
        ax.plot(result["x_onset"], result["y_onset"], "bo", label="onset")
    if np.isfinite(result.get("x_onset2", np.nan)):
        ax.plot(result["x_onset2"], result["y_onset2"], "mo", label="onset2")
    if np.isfinite(result.get("a_flank", np.nan)) and np.isfinite(result.get("b_flank", np.nan)):
        a_f = result["a_flank"]
        b_f = result["b_flank"]
        ax.plot(x, a_f + b_f * x, color="magenta", linestyle="--", linewidth=0.8, label="flank fit")
    if np.isfinite(result.get("x0", np.nan)):
        ax.plot(result["x0"], result["y0"], "ko", label="tangent point")
    if np.isfinite(result.get("t_char_50", np.nan)):
        ax.axvline(result["t_char_50"], color="purple", linestyle="--", linewidth=0.8, label="T_char_50")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(fontsize=8, loc="best")
    return ax

def detect_peak_start(
    t_h,
    y,
    smooth_window=51,
    z_thresh=1.0,
    min_frac_peak=0.01,
    pre_peak_frac=0.3,
    center_h=None,
    window_h=None,
    pre_peak_window_h=None,
    pre_peak_window_frac=0.1,
):
    t = np.asarray(t_h, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(y) == 0:
        return None

    if center_h is not None and window_h is not None:
        mask = (t >= center_h - window_h) & (t <= center_h + window_h)
        if mask.sum() < 5:
            return None
        t = t[mask]
        y = y[mask]

    ys = smooth_series(y, smooth_window)

    baseline0 = np.median(ys)
    residual0 = ys - baseline0
    peak_idx = None
    if center_h is not None:
        extrema = []
        for i in range(1, len(ys) - 1):
            if ys[i] > ys[i - 1] and ys[i] > ys[i + 1]:
                extrema.append(i)
            elif ys[i] < ys[i - 1] and ys[i] < ys[i + 1]:
                extrema.append(i)
        if extrema:
            peak_idx = min(extrema, key=lambda i: abs(t[i] - center_h))
    if peak_idx is None:
        peak_idx = int(np.argmax(np.abs(residual0)))

    pre_end = max(3, int(peak_idx * pre_peak_frac))
    if pre_end < 3:
        pre_end = min(len(ys), 50)
    pre_slice = ys[:pre_end] if pre_end > 0 else ys
    baseline = np.median(pre_slice)
    residual = ys - baseline
    mad = np.median(np.abs(pre_slice - baseline))
    sigma = 1.4826 * mad if mad > 0 else np.std(pre_slice - baseline)
    if sigma <= 0:
        sigma = 1e-9

    peak_val = ys[peak_idx]
    peak_amp = peak_val - baseline
    if abs(peak_amp) < max(z_thresh * sigma, 1e-6):
        return None

    direction = 1.0 if peak_amp >= 0 else -1.0
    threshold = baseline + direction * max(z_thresh * sigma, abs(peak_amp) * min_frac_peak)

    start_idx = 0
    if direction > 0:
        for i in range(peak_idx, -1, -1):
            if ys[i] <= threshold:
                start_idx = i
                break
    else:
        for i in range(peak_idx, -1, -1):
            if ys[i] >= threshold:
                start_idx = i
                break

    return {
        "t_start_h": float(t[start_idx]),
        "hf_start": float(ys[start_idx]),
        "t_peak_h": float(t[peak_idx]),
        "hf_peak": float(peak_val),
        "baseline": float(baseline),
        "threshold": float(threshold),
    }

def load_program_zones(pdf_path: Path):
    try:
        import PyPDF2
    except Exception:
        return []

    if not pdf_path.exists():
        return []

    reader = PyPDF2.PdfReader(str(pdf_path))
    text = "\n".join((page.extract_text() or "") for page in reader.pages)
    lines = text.splitlines()

    zone_headers = []
    for i, line in enumerate(lines):
        m = re.match(r"\s*(\d+)\s*:\s*(.+)\s*$", line)
        if m:
            zone_headers.append((i, int(m.group(1)), m.group(2).strip()))

    zones = []
    for idx, (i, num, typ) in enumerate(zone_headers):
        j = zone_headers[idx + 1][0] if idx + 1 < len(zone_headers) else len(lines)
        duration = None
        for line in lines[i:j]:
            m = re.search(r"Total duration:\s*(\d+)\s*s", line)
            if m:
                duration = int(m.group(1))
                break
        if duration is None:
            continue
        zones.append({"num": num, "type": typ, "duration_s": duration})

    return zones

def program_total_seconds(pdf_path: Path):
    zones = load_program_zones(pdf_path)
    if not zones:
        return None
    return sum(z["duration_s"] for z in zones)

def program_zone_spans(pdf_path: Path, t_h=None):
    zones = load_program_zones(pdf_path)
    if not zones:
        return []

    spans = []
    t_s = 0.0
    for z in zones:
        duration = z["duration_s"]
        start_h = t_s / 3600.0
        end_h = (t_s + duration) / 3600.0
        spans.append(
            {"num": z["num"], "type": z["type"], "start_h": start_h, "end_h": end_h}
        )
        t_s += duration

    if t_h is not None and len(t_h) > 0:
        t_min, t_max = float(t_h[0]), float(t_h[-1])
        clipped = []
        for z in spans:
            t0, t1 = z["start_h"], z["end_h"]
            if t1 <= t_min or t0 >= t_max:
                continue
            clipped.append(
                {
                    "num": z["num"],
                    "type": z["type"],
                    "start_h": max(t0, t_min),
                    "end_h": min(t1, t_max),
                }
            )
        spans = clipped

    return spans

def get_zone_spans(clean, pdf_path: Path):
    t_h = clean["t_h"].to_numpy() if "t_h" in clean else clean["t_s"].to_numpy() / 3600.0
    zones = program_zone_spans(pdf_path, t_h=t_h)
    if zones:
        return zones
    return detect_zones_from_temperature(t_h, clean["T_horno"].to_numpy())

def zone_label(zones, zone_num):
    zone = next((z for z in zones if z["num"] == zone_num), None)
    if zone is None:
        return f"Zona {zone_num}"
    ztype = zone["type"]
    count = sum(1 for z in zones if z["type"] == ztype and z["num"] <= zone_num)
    return f"{ztype} {count}"


def _zone_type_to_english(ztype: str) -> str:
    """Translate program zone type names to English for plot titles."""
    z = (ztype or "").strip().lower()
    mapping = {
        "isoterma": "Isotherm",
        "isothermal": "Isotherm",
        "heating": "Heating",
        "cooling": "Cooling",
        "return": "Return",
    }
    return mapping.get(z, ztype)


def isotherm_spans_from_program(pdf_path: Path, t_h=None):
    zones = load_program_zones(pdf_path)
    if not zones:
        return []

    spans = []
    t_s = 0.0
    for z in zones:
        duration = z["duration_s"]
        if "isoterma" in z["type"].lower():
            spans.append((t_s / 3600.0, (t_s + duration) / 3600.0))
        t_s += duration

    if t_h is not None and len(t_h) > 0:
        t_min, t_max = float(t_h[0]), float(t_h[-1])
        clipped = []
        for t0, t1 in spans:
            if t1 <= t_min or t0 >= t_max:
                continue
            clipped.append((max(t0, t_min), min(t1, t_max)))
        spans = clipped

    return spans

def select_zone(clean, zone_num, pdf_path: Path, rebase_time=True):
    zones = get_zone_spans(clean, pdf_path)
    if not zones:
        return None, None
    zone = next((z for z in zones if z["num"] == zone_num), None)
    if zone is None:
        return None, None

    orig_start = zone["start_h"]
    orig_end = zone["end_h"]
    mask = (clean["t_h"] >= zone["start_h"]) & (clean["t_h"] <= zone["end_h"])
    subset = clean.loc[mask].copy()
    if rebase_time and len(subset) > 0:
        subset["t_h"] = subset["t_h"] - zone["start_h"]
        if "t_s" in subset:
            subset["t_s"] = subset["t_s"] - zone["start_h"] * 3600.0
        zone = {
            "num": zone["num"],
            "type": zone["type"],
            "start_h": 0.0,
            "end_h": orig_end - orig_start,
            "offset_h": orig_start,
            "orig_start_h": orig_start,
            "orig_end_h": orig_end,
        }
    else:
        zone["offset_h"] = 0.0
        zone["orig_start_h"] = orig_start
        zone["orig_end_h"] = orig_end

    return subset, zone

def plot_panel_6(
    clean,
    outdir: Path,
    outname="panel_6plots.png",
    zone_info=None,
    show_zones=True,
    program_pdf: Path | None = None,
    onset_method="tangent",
    onset_center_h=None,
    onset_window_h=2.0,
    onset_flank_near_peak_frac=0.15,
    show_onset=False,
):
    outdir.mkdir(parents=True, exist_ok=True)

    if "t_h" in clean:
        t_h = clean["t_h"].to_numpy()
    else:
        t_h = clean["t_s"].to_numpy() / 3600.0
    turns = detect_ramp_turns(t_h, clean["T_horno"].to_numpy(), min_sep_h=5.0)
    peak_info = None
    onset_idx = None
    onset2_idx = None
    if show_onset:
        if onset_method == "tangent":
            res = compute_onset_tangent(
                t_h,
                clean["HF_mW"].to_numpy(),
                event_polarity="auto",
                params={
                    "flank_force_near_peak": True,
                    "flank_near_peak_frac": float(onset_flank_near_peak_frac),
                },
            )
            if res.get("ok") and np.isfinite(res.get("x_onset", np.nan)):
                peak_info = {
                    "t_start_h": float(res["x_onset"]),
                    "hf_start": float(res["y_onset"]),
                    "t_peak_h": float(res["x_peak"]),
                    "hf_peak": float(res["y_peak"]),
                    "t_char_50": float(res.get("t_char_50", np.nan)),
                    "t_start2_h": float(res.get("x_onset2", np.nan)),
                    "hf_start2": float(res.get("y_onset2", np.nan)),
                    "onset2_ok": bool(res.get("onset2_ok", False)),
                    "a_base": float(res.get("a_base", np.nan)),
                    "b_base": float(res.get("b_base", np.nan)),
                    "a_flank": float(res.get("a_flank", np.nan)),
                    "b_flank": float(res.get("b_flank", np.nan)),
                    "baseline_w0_h": float(res.get("baseline_window", (np.nan, np.nan))[0]),
                    "baseline_w1_h": float(res.get("baseline_window", (np.nan, np.nan))[1]),
                    "flank_w0_h": float(res.get("flank_window", (np.nan, np.nan))[0]),
                    "flank_w1_h": float(res.get("flank_window", (np.nan, np.nan))[1]),
                    "baseline_window_onset_h": tuple(res.get("baseline_window_onset", (np.nan, np.nan))),
                    "flank_window_onset_h": tuple(res.get("flank_window_onset", (np.nan, np.nan))),
                    "baseline_window_onset2_h": tuple(res.get("baseline_window_onset2", (np.nan, np.nan))),
                    "flank_window_onset2_h": tuple(res.get("flank_window_onset2", (np.nan, np.nan))),
                    "method": "tangent",
                }
                onset_idx = int(np.abs(t_h - peak_info["t_start_h"]).argmin())
                if peak_info.get("onset2_ok") and np.isfinite(peak_info.get("t_start2_h", np.nan)):
                    onset2_idx = int(np.abs(t_h - peak_info["t_start2_h"]).argmin())
        else:
            peak_info = detect_peak_start(
                t_h,
                clean["HF_mW"].to_numpy(),
                center_h=onset_center_h,
                window_h=onset_window_h,
            )
            if peak_info is not None:
                onset_idx = int(np.abs(t_h - peak_info["t_start_h"]).argmin())
    zone_spans = []
    iso_spans = []
    zone_colors = {}
    if show_zones:
        if program_pdf is None:
            program_pdf = Path(DEFAULT_PROGRAM)
        zone_spans = get_zone_spans(clean, program_pdf)
        if not zone_spans:
            iso_spans = detect_isotherm_spans(
                t_h, clean["T_horno"].to_numpy(), max_rate_c_per_h=1.0, min_duration_h=0.0
            )
        zone_colors = {
            "isoterma": "#d9d9d9",
            "heating": "#ffcccc",
            "cooling": "#cce0ff",
            "return": "#ccf2d9",
        }

    fig, axes = plt.subplots(3, 2, figsize=(10, 7), constrained_layout=True)
    if zone_info is not None:
        fig.suptitle(f"Zone {zone_info['num']} - {_zone_type_to_english(zone_info['type'])}")

    # 1) P vs t
    ax = axes[0, 0]
    ax.plot(t_h, clean["P_muestra"], color="black", label="Sample P")
    if "P_referencia" in clean.columns:
        ax.plot(
            t_h,
            clean["P_referencia"],
            color="dimgray",
            linestyle="--",
            linewidth=1.2,
            label="Reference P",
        )
    if zone_spans:
        for z in zone_spans:
            key = z["type"].lower().split()[0]
            color = zone_colors.get(key, "#eeeeee")
            ax.axvspan(z["start_h"], z["end_h"], color=color, alpha=0.18, linewidth=0)
    else:
        for t0, t1 in iso_spans:
            ax.axvspan(t0, t1, color="gray", alpha=0.15, linewidth=0)
    ax.set_xlabel("t (h)")
    ax.set_ylabel("Pressure (bar)")
    if "P_referencia" in clean.columns:
        ax.legend(loc="best", fontsize=8)

    # 2) P vs T
    ax = axes[0, 1]
    ax.plot(clean["T_muestra"], clean["P_muestra"], color="black", label="Sample P")
    if "P_referencia" in clean.columns:
        ax.plot(
            clean["T_muestra"],
            clean["P_referencia"],
            color="dimgray",
            linestyle="--",
            linewidth=1.2,
            label="Reference P",
        )
    ax.set_xlabel("T (°C)")
    ax.set_ylabel("Pressure (bar)")
    if "P_referencia" in clean.columns:
        ax.legend(loc="best", fontsize=8)

    # 3) T vs t
    ax = axes[1, 0]
    ax.plot(t_h, clean["T_muestra"], color="black", label="Sample T")
    ax.plot(t_h, clean["T_horno"], color="orange", label="Furnace T")
    if zone_spans:
        for z in zone_spans:
            key = z["type"].lower().split()[0]
            color = zone_colors.get(key, "#eeeeee")
            ax.axvspan(z["start_h"], z["end_h"], color=color, alpha=0.18, linewidth=0)
        # etiquetas de zonas desactivadas a pedido
    else:
        for t0, t1 in iso_spans:
            ax.axvspan(t0, t1, color="gray", alpha=0.15, linewidth=0)
    ax.set_xlabel("t (h)")
    ax.set_ylabel("Temperature (°C)")
    ax.legend(loc="best", fontsize=8)

    # 4) HF vs T
    ax = axes[1, 1]
    ax.plot(clean["T_muestra"], clean["HF_mW"], color="black")
    ax.axhline(0, linewidth=0.8)
    if peak_info is not None and onset_idx is not None:
        t_on = clean["T_muestra"].iloc[onset_idx]
        hf_on = clean["HF_mW"].iloc[onset_idx]
        ax.plot(
            clean["T_muestra"].iloc[onset_idx],
            clean["HF_mW"].iloc[onset_idx],
            "bo",
            markersize=4,
        )
        ax.annotate(
            f"{t_on:.2f} °C",
            xy=(t_on, hf_on),
            xytext=(12, 12),
            textcoords="offset points",
            fontsize=7,
            ha="left",
            va="bottom",
            color="blue",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
        )
    if peak_info is not None and onset2_idx is not None:
        t_on2 = clean["T_muestra"].iloc[onset2_idx]
        hf_on2 = clean["HF_mW"].iloc[onset2_idx]
        ax.plot(
            clean["T_muestra"].iloc[onset2_idx],
            clean["HF_mW"].iloc[onset2_idx],
            "mo",
            markersize=4,
        )
        ax.annotate(
            f"{t_on2:.2f} °C",
            xy=(t_on2, hf_on2),
            xytext=(12, -12),
            textcoords="offset points",
            fontsize=7,
            ha="left",
            va="top",
            color="magenta",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
        )
    ax.set_xlabel("T (°C)")
    ax.set_ylabel("HF (mW)")

    # 5) HF vs t (+ líneas verticales)
    ax = axes[2, 0]
    ax.plot(t_h, clean["HF_mW"], color="black")
    ax.axhline(0, linewidth=0.8)
    if zone_spans:
        for z in zone_spans:
            key = z["type"].lower().split()[0]
            color = zone_colors.get(key, "#eeeeee")
            ax.axvspan(z["start_h"], z["end_h"], color=color, alpha=0.18, linewidth=0)
    else:
        for t0, t1 in iso_spans:
            ax.axvspan(t0, t1, color="gray", alpha=0.15, linewidth=0)
    show_onset2_fit = False
    show_baseline_window = False
    if peak_info is not None and peak_info.get("method") == "tangent":
        a_base = peak_info.get("a_base", np.nan)
        b_base = peak_info.get("b_base", np.nan)
        a_flank = peak_info.get("a_flank", np.nan)
        b_flank = peak_info.get("b_flank", np.nan)
        x0 = peak_info.get("t_start2_h", np.nan)
        x1 = peak_info.get("t_peak_h", np.nan)
        if not np.isfinite(x0):
            x0 = peak_info.get("t_start_h", np.nan)
        if (
            np.isfinite(a_base)
            and np.isfinite(b_base)
            and np.isfinite(a_flank)
            and np.isfinite(b_flank)
            and np.isfinite(x0)
            and np.isfinite(x1)
            and x1 > x0
        ):
            x_fit = np.linspace(x0, x1, 120)
            ax.plot(
                x_fit,
                a_base + b_base * x_fit,
                linestyle="--",
                color="blue",
                linewidth=1.0,
                alpha=0.8,
                label="Baseline (onset2)",
            )
            ax.plot(
                x_fit,
                a_flank + b_flank * x_fit,
                linestyle="--",
                color="magenta",
                linewidth=1.0,
                alpha=0.8,
                label="Flank fit (onset2)",
            )
            show_onset2_fit = True
        xb0 = peak_info.get("baseline_w0_h", np.nan)
        xb1 = peak_info.get("baseline_w1_h", np.nan)
        if (
            np.isfinite(a_base)
            and np.isfinite(b_base)
            and np.isfinite(xb0)
            and np.isfinite(xb1)
            and xb1 > xb0
        ):
            x_base_win = np.linspace(xb0, xb1, 80)
            ax.plot(
                x_base_win,
                a_base + b_base * x_base_win,
                linestyle="--",
                color="navy",
                linewidth=1.4,
                alpha=0.95,
                label="Baseline (fit window)",
            )
            show_baseline_window = True
    if peak_info is not None:
        ax.plot(peak_info["t_start_h"], peak_info["hf_start"], "bo", markersize=5, zorder=5)
        ax.axvline(peak_info["t_start_h"], color="blue", linewidth=0.8, alpha=0.6)
        if onset2_idx is not None:
            ax.plot(peak_info["t_start2_h"], peak_info["hf_start2"], "mo", markersize=5, zorder=5)
            ax.axvline(peak_info["t_start2_h"], color="magenta", linewidth=0.8, alpha=0.6)
        ax.annotate(
            f"ONSET\\n{peak_info['t_start_h']:.2f} h\\n{peak_info['hf_start']:.3g} mW",
            xy=(peak_info["t_start_h"], peak_info["hf_start"]),
            xytext=(12, 12),
            textcoords="offset points",
            fontsize=7,
            ha="left",
            va="bottom",
            color="blue",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
        )
        if onset2_idx is not None:
            ax.annotate(
                f"ONSET2\\n{peak_info['t_start2_h']:.2f} h\\n{peak_info['hf_start2']:.3g} mW",
                xy=(peak_info["t_start2_h"], peak_info["hf_start2"]),
                xytext=(12, 28),
                textcoords="offset points",
                fontsize=7,
                ha="left",
                va="bottom",
                color="magenta",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
            )
    if show_onset2_fit or show_baseline_window:
        ax.legend(loc="lower left", fontsize=7)
    ax.set_xlabel("t (h)")
    ax.set_ylabel("HF (mW)")

    # 6) HF vs P
    ax = axes[2, 1]
    ax.plot(clean["P_muestra"], clean["HF_mW"], color="black")
    ax.axhline(0, linewidth=0.8)
    if peak_info is not None and onset_idx is not None:
        p_on = clean["P_muestra"].iloc[onset_idx]
        hf_on = clean["HF_mW"].iloc[onset_idx]
        ax.plot(
            clean["P_muestra"].iloc[onset_idx],
            clean["HF_mW"].iloc[onset_idx],
            "bo",
            markersize=4,
        )
        ax.annotate(
            f"{p_on:.2f} bar",
            xy=(p_on, hf_on),
            xytext=(12, 12),
            textcoords="offset points",
            fontsize=7,
            ha="left",
            va="bottom",
            color="blue",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
        )
    if peak_info is not None and onset2_idx is not None:
        p_on2 = clean["P_muestra"].iloc[onset2_idx]
        hf_on2 = clean["HF_mW"].iloc[onset2_idx]
        ax.plot(
            clean["P_muestra"].iloc[onset2_idx],
            clean["HF_mW"].iloc[onset2_idx],
            "mo",
            markersize=4,
        )
        ax.annotate(
            f"{p_on2:.2f} bar",
            xy=(p_on2, hf_on2),
            xytext=(12, -12),
            textcoords="offset points",
            fontsize=7,
            ha="left",
            va="top",
            color="magenta",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
        )
    ax.set_xlabel("P (bar)")
    ax.set_ylabel("HF (mW)")

    fig.savefig(outdir / outname, dpi=200)
    plt.close(fig)
    return peak_info


def _nearest_tp_at_time(clean: pd.DataFrame, t_on_h: float) -> tuple[float, float]:
    """Return (T, P) from the sample closest to the given time in hours."""
    idx = int(np.abs(clean["t_h"].to_numpy() - t_on_h).argmin())
    t_on = float(clean["T_muestra"].iloc[idx])
    p_on = float(clean["P_muestra"].iloc[idx])
    return t_on, p_on


def _print_onset_windows(kind: str, bw, fw, zone_name: str | None = None) -> None:
    """Print baseline/flank windows with the same format used in CLI output."""
    if not (
        np.all(np.isfinite(np.asarray(bw, dtype=float)))
        and np.all(np.isfinite(np.asarray(fw, dtype=float)))
    ):
        return

    zone_suffix = f" ({zone_name})" if zone_name else ""
    pair_sep = " " if zone_name else ", "
    print(
        f"Ventanas {kind}{zone_suffix}: "
        f"base=[{bw[0]:.4f},{bw[1]:.4f}] h{pair_sep}"
        f"pico=[{fw[0]:.4f},{fw[1]:.4f}] h"
    )


def print_peak_summary(clean: pd.DataFrame, peak_info: dict, zone_name: str | None = None) -> None:
    """Print onset/onset2 and window summary from `peak_info`."""
    zone_suffix = f" ({zone_name})" if zone_name else ""

    t_on, p_on = _nearest_tp_at_time(clean, peak_info["t_start_h"])
    print(
        f"HF onset{zone_suffix}: "
        f"t={peak_info['t_start_h']:.4f} h, "
        f"HF={peak_info['hf_start']:.6g} mW, "
        f"T={t_on:.3f} °C, P={p_on:.3f} bar"
    )
    _print_onset_windows(
        "onset",
        peak_info.get("baseline_window_onset_h", (np.nan, np.nan)),
        peak_info.get("flank_window_onset_h", (np.nan, np.nan)),
        zone_name=zone_name,
    )

    if peak_info.get("onset2_ok") and np.isfinite(peak_info.get("t_start2_h", np.nan)):
        t_on2, p_on2 = _nearest_tp_at_time(clean, peak_info["t_start2_h"])
        print(
            f"HF onset2{zone_suffix}: "
            f"t={peak_info['t_start2_h']:.4f} h, "
            f"HF={peak_info['hf_start2']:.6g} mW, "
            f"T={t_on2:.3f} °C, P={p_on2:.3f} bar"
        )
        _print_onset_windows(
            "onset2",
            peak_info.get("baseline_window_onset2_h", (np.nan, np.nan)),
            peak_info.get("flank_window_onset2_h", (np.nan, np.nan)),
            zone_name=zone_name,
        )

    t_char_50 = peak_info.get("t_char_50", np.nan)
    if np.isfinite(t_char_50):
        print(f"T_char_50{zone_suffix}: t={t_char_50:.4f} h")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Graficar panel 6 a partir de un .lvm.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=str, default=DEFAULT_DATA, help="Archivo .lvm a analizar")
    parser.add_argument("--program", type=str, default=DEFAULT_PROGRAM, help="PDF del programa térmico")
    parser.add_argument("--zone", type=int, default=None, help="Número de zona del programa (ej: 7)")
    parser.add_argument("--Cooling", type=int, default=None, help="Seleccionar Cooling N (ej: --Cooling 2)")
    parser.add_argument("--Heating", type=int, default=None, help="Seleccionar Heating N (ej: --Heating 1)")
    parser.add_argument("--Isoterma", type=int, default=None, help="Seleccionar Isoterma N (ej: --Isoterma 1)")
    parser.add_argument("--Return", type=int, default=None, help="Seleccionar Return N (ej: --Return 1)")
    parser.add_argument(
        "--no-rebase",
        action="store_true",
        help="Mantener tiempo absoluto al filtrar por zona",
    )
    parser.add_argument(
        "--onset-around",
        type=float,
        default=None,
        help="Hora (h) aproximada donde buscar el onset del pico HF",
    )
    parser.add_argument(
        "--onset-window",
        type=float,
        default=2.0,
        help="Ventana +/- horas alrededor de --onset-around",
    )
    parser.add_argument(
        "--onset-method",
        type=str,
        default="tangent",
        choices=["tangent", "simple"],
        help="Método de onset: tangent (tangente estándar) o simple (umbral)",
    )
    parser.add_argument(
        "--flank-near-peak-frac",
        type=float,
        default=0.15,
        help=(
            "Fracción del tramo pre-pico usada para forzar que la ventana del pico "
            "quede cerca del pico (menor = más cerca)"
        ),
    )
    args = parser.parse_args()

    raw_data_path = Path(args.data)
    raw_program_path = Path(args.program)
    data_path = _resolve_input_path(raw_data_path, expected_suffix=".lvm")
    program_path = _resolve_input_path(raw_program_path, expected_suffix=".pdf")

    if not data_path.exists():
        raise SystemExit(f"Archivo .lvm no encontrado: {raw_data_path}")

    if data_path != raw_data_path:
        print(f"Usando archivo de datos: {data_path}")
    if program_path != raw_program_path and program_path.exists():
        print(f"Usando PDF del programa: {program_path}")

    clean = load_lvm(data_path, program_pdf=program_path)
    zone_info = None
    def zone_num_from_profile(zones_all, ztype, ordinal):
        if zones_all is None or ordinal is None:
            return None
        matches = [z for z in zones_all if z["type"].lower().startswith(ztype.lower())]
        if ordinal <= 0 or ordinal > len(matches):
            return None
        return matches[ordinal - 1]["num"]

    zones_all = None
    zone_requested = (
        args.zone is not None
        or args.Cooling is not None
        or args.Heating is not None
        or args.Isoterma is not None
        or args.Return is not None
    )
    if zone_requested:
        # Cuando el usuario pide una zona por número/perfil, debe respetar
        # la numeración del PDF del programa y no caer al fallback automático.
        t_h = clean["t_h"].to_numpy() if "t_h" in clean else None
        zones_all = program_zone_spans(program_path, t_h=t_h)
        if not zones_all:
            raise SystemExit(
                "No se pudieron leer zonas del PDF del programa. "
                "Verifica PyPDF2 e intenta con el intérprete del entorno "
                "virtual (ej: .venv/bin/python)."
            )

    zone_num = None
    if args.zone is not None:
        available_nums = [z["num"] for z in zones_all] if zones_all else []
        if available_nums and args.zone not in available_nums:
            nums_txt = ", ".join(str(n) for n in available_nums)
            raise SystemExit(
                f"Zona {args.zone} no encontrada en el PDF. "
                f"Zonas disponibles: {nums_txt}."
            )
        zone_num = args.zone
    elif args.Cooling is not None:
        zone_num = zone_num_from_profile(zones_all, "Cooling", args.Cooling)
    elif args.Heating is not None:
        zone_num = zone_num_from_profile(zones_all, "Heating", args.Heating)
    elif args.Isoterma is not None:
        zone_num = zone_num_from_profile(zones_all, "Isoterma", args.Isoterma)
    elif args.Return is not None:
        zone_num = zone_num_from_profile(zones_all, "Return", args.Return)

    base_name = data_path.stem
    if zone_num is not None:
        clean, zone_info = select_zone(
            clean,
            zone_num,
            program_path,
            rebase_time=not args.no_rebase,
        )
        if zone_info is None:
            raise SystemExit("Zona no encontrada en el PDF.")
        zone_name = zone_label(zones_all, zone_num) if zones_all else f"Zona {zone_num}"
        outname = f"{base_name}_panel_6plots_zone{zone_info['num']}.png"
        peak_info = plot_panel_6(
            clean,
            Path("output/analisis"),
            outname=outname,
            zone_info=zone_info,
            show_zones=False,
            program_pdf=program_path,
            onset_method=args.onset_method,
            onset_center_h=args.onset_around,
            onset_window_h=args.onset_window,
            onset_flank_near_peak_frac=args.flank_near_peak_frac,
            show_onset=True,
        )
        if peak_info is not None:
            print_peak_summary(clean, peak_info, zone_name=zone_name)
    else:
        show_onset = args.onset_around is not None
        outname = f"{base_name}_panel_6plots.png"
        peak_info = plot_panel_6(
            clean,
            Path("output/analisis"),
            outname=outname,
            program_pdf=program_path,
            onset_method=args.onset_method,
            onset_center_h=args.onset_around,
            onset_window_h=args.onset_window,
            onset_flank_near_peak_frac=args.flank_near_peak_frac,
            show_onset=show_onset,
        )
        if show_onset and peak_info is not None:
            print_peak_summary(clean, peak_info, zone_name=None)
