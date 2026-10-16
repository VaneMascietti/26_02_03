"""Numerical Calvet onset methods, independent of matplotlib and file I/O.

onset2 intersects the baseline with a least-squares line over the flank.
It is an alternative estimate of the SAME event, not a second transition.
Original tangent and threshold algorithms are retained verbatim.
"""

import numpy as np

TANGENT_DEFAULTS = {
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
    defaults = TANGENT_DEFAULTS
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


def analyze_peak(clean, method="tangent", center_h=None, window_h=2.0, flank_near_peak_frac=0.15):
    """Compatibility adapter: compute legacy peak_info without plotting."""
    t_h = clean["t_h"].to_numpy()
    peak_info = None
    if method == "tangent":
        res = compute_onset_tangent(
            t_h,
            clean["HF_mW"].to_numpy(),
            event_polarity="auto",
            params={
                "flank_force_near_peak": True,
                "flank_near_peak_frac": float(flank_near_peak_frac),
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
    else:
        peak_info = detect_peak_start(
            t_h,
            clean["HF_mW"].to_numpy(),
            center_h=center_h,
            window_h=window_h,
        )
    return peak_info
