import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import argparse
from util.lvm_io import (
    _drop_leading_counter_wrap,
    _series_is_effectively_blank,
    _uses_trailing_sample_pressure,
    load_lvm,
)
from util.path import get_data_path, get_output_dir
from util.program_zones import (
    get_zone_spans as _get_zone_spans,
    isotherm_spans_from_program,
    load_program_zones,
    program_total_seconds,
    program_zone_spans,
    select_zone as _select_zone,
    zone_label,
    zone_number_by_type,
    zone_numbers_by_type,
    zone_type_to_english,
)
from util.pt_plot import plot_pt_tracks as _plot_pt_tracks

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

from util.calvet_onset import compute_onset_tangent, detect_peak_start, smooth_series, analyze_peak

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

def get_zone_spans(clean, pdf_path: Path):
    """Keep the historical temperature-derived fallback for malformed PDFs."""
    return _get_zone_spans(
        clean,
        pdf_path,
        fallback=detect_zones_from_temperature,
    )


def select_zone(clean, zone_num, pdf_path: Path, rebase_time=True):
    """Select a PDF zone while retaining the historical fallback behavior."""
    return _select_zone(
        clean,
        zone_num,
        pdf_path,
        rebase_time=rebase_time,
        fallback=detect_zones_from_temperature,
    )


def _zone_type_to_english(ztype: str) -> str:
    return zone_type_to_english(ztype)

def plot_panel_6(
    clean,
    outdir: Path,
    outname="signals_vs_time.png",
    zone_info=None,
    show_zones=True,
    program_pdf: Path | None = None,
    onset_method="tangent",
    onset_center_h=None,
    onset_window_h=2.0,
    onset_flank_near_peak_frac=0.15,
    show_onset=False,
    show_reference_pressure=False,
    extra_onset_temps_c=None,
    peak_info=None,
    calculate_onset=True,
):
    """Retired entry point: render pressure, sample/Huber T and HF against time."""
    from util.calvet_plotting import plot_time_signals
    outname = outname.replace("panel_6plots", "signals_vs_time")
    plot_time_signals(clean, Path(outdir) / outname)
    return peak_info


def _nearest_tp_at_time(clean: pd.DataFrame, t_on_h: float) -> tuple[float, float]:
    """Return (T, P) from the sample closest to the given time in hours."""
    idx = int(np.abs(clean["t_h"].to_numpy() - t_on_h).argmin())
    t_on = float(clean["T_muestra"].iloc[idx])
    p_on = float(clean["P_muestra"].iloc[idx])
    return t_on, p_on


def _extra_onsets_from_temperature_targets(clean, target_temps_c, excluded_indices=None):
    """
    Map target temperatures (°C) to nearest samples in the current dataset.
    Returns list of dicts with t/T/P/HF for annotation.
    """
    if not target_temps_c:
        return []

    t_arr = (
        clean["t_h"].to_numpy(dtype=float)
        if "t_h" in clean
        else clean["t_s"].to_numpy(dtype=float) / 3600.0
    )
    temp_arr = clean["T_muestra"].to_numpy(dtype=float)
    p_arr = clean["P_muestra"].to_numpy(dtype=float)
    hf_arr = clean["HF_mW"].to_numpy(dtype=float)

    finite_temp = np.isfinite(temp_arr)
    if not finite_temp.any():
        return []

    used = set()
    if excluded_indices is not None:
        used.update(i for i in excluded_indices if i is not None)

    out = []
    for target in target_temps_c:
        if target is None or not np.isfinite(float(target)):
            continue
        dist = np.abs(temp_arr - float(target))
        dist[~finite_temp] = np.inf
        idx = int(np.argmin(dist))
        if not np.isfinite(dist[idx]) or idx in used:
            continue
        used.add(idx)
        out.append(
            {
                "idx": idx,
                "target_T_C": float(target),
                "T_C": float(temp_arr[idx]),
                "P_bar": float(p_arr[idx]),
                "HF_mW": float(hf_arr[idx]),
                "t_h": float(t_arr[idx]),
            }
        )

    out.sort(key=lambda item: item["t_h"])
    return out


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

    extras = peak_info.get("extra_onsets", [])
    for i, onset_extra in enumerate(extras, 1):
        print(
            f"HF manual_temperature_target{i}{zone_suffix}: "
            f"t={onset_extra['t_h']:.4f} h, "
            f"HF={onset_extra['HF_mW']:.6g} mW, "
            f"T={onset_extra['T_C']:.3f} °C, "
            f"P={onset_extra['P_bar']:.3f} bar, "
            f"T_target={onset_extra['target_T_C']:.3f} °C"
        )


def _valid_sample_pressure_mask(clean: pd.DataFrame) -> pd.Series:
    from util.data_filters import numeric_filter_mask

    return numeric_filter_mask(
        clean,
        ["T_muestra", "P_muestra", "t_h"],
    ) & (pd.to_numeric(clean["P_muestra"], errors="coerce") > 0)


def plot_pt_tracks(
    tracks: list[tuple[pd.DataFrame, str]],
    outpath: Path,
    include_time: bool = False,
    show_initial_pressure: bool = False,
) -> None:
    """Backward-compatible wrapper around the shared P-T renderer."""
    _plot_pt_tracks(
        tracks,
        outpath,
        include_time=include_time,
        show_initial_pressure=show_initial_pressure,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Graficar presión, temperatura de muestra/Huber y HF frente al tiempo.",
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
        "--all-cooling",
        action="store_true",
        help="Generar gráficos para todas las zonas Cooling del programa.",
    )
    parser.add_argument(
        "--all-heating",
        action="store_true",
        help="Generar gráficos para todas las zonas Heating del programa.",
    )
    parser.add_argument(
        "--no-rebase",
        action="store_true",
        help="Mantener tiempo absoluto al filtrar por zona",
    )
    parser.add_argument(
        "--full-data",
        action="store_true",
        help="Graficar el .lvm completo sin recortar por la duracion del PDF del programa.",
    )
    parser.add_argument(
        "--plot-pt",
        action="store_true",
        help="Guardar solo P_muestra vs T_muestra en out/figures/pt_compare y salir.",
    )
    parser.add_argument(
        "--plot-pt-time",
        action="store_true",
        help="Guardar P_muestra vs T_muestra y P_muestra vs tiempo en una figura y salir.",
    )
    parser.add_argument("--data2", type=str, default=None, help="Segundo .lvm para superponer en --plot-pt.")
    parser.add_argument("--program2", type=str, default=None, help="PDF del programa para --data2.")
    parser.add_argument("--label", type=str, default=None, help="Etiqueta para --plot-pt.")
    parser.add_argument("--label2", type=str, default=None, help="Etiqueta para --data2 en --plot-pt.")
    parser.add_argument("--out", type=str, default=None, help="Ruta de salida para --plot-pt/--plot-pt-time.")
    parser.add_argument(
        "--show-initial-pressure",
        action="store_true",
        help="Marcar y etiquetar P0 en los graficos --plot-pt/--plot-pt-time.",
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
    parser.add_argument(
        "--show-reference-pressure",
        dest="show_reference_pressure",
        action="store_true",
        default=False,
        help="Mostrar presión de referencia en los gráficos P vs t y P vs T (por defecto no se muestra).",
    )
    parser.add_argument(
        "--extra-onset-temp",
        type=float,
        action="append",
        default=[],
        help=(
            "Temperatura objetivo (°C) para marcar observaciones manuales en HF. "
            "Puede repetirse: --extra-onset-temp 40.64 --extra-onset-temp 16.09"
        ),
    )
    from calvet_workflow import add_arguments
    add_arguments(parser)
    return parser


def resolve_data_program(data_arg: str, program_arg: str) -> tuple[Path, Path]:
    """Resolve data/PDF inputs using the project path conventions."""
    raw_data_path = Path(data_arg)
    raw_program_path = Path(program_arg)
    data_input = _resolve_input_path(raw_data_path, expected_suffix=".lvm")
    program_input = _resolve_input_path(raw_program_path, expected_suffix=".pdf")
    data_path = get_data_path(data_input, "lvm")
    program_path = get_data_path(program_input, "pdf")

    if not data_path.exists():
        raise SystemExit(f"Archivo .lvm no encontrado: {raw_data_path}")

    if data_path != raw_data_path:
        print(f"Usando archivo de datos: {data_path}")
    if program_path != raw_program_path and program_path.exists():
        print(f"Usando PDF del programa: {program_path}")

    return data_path, program_path


def default_pt_outpath(stems: list[str], include_time: bool, full_data: bool) -> Path:
    prefix = "pt_and_p_time" if include_time else "pt"
    suffix = "_".join(stems)
    tail = "all_points" if full_data else "program"
    return get_output_dir("figures", "pt_compare") / f"{prefix}_{suffix}_{tail}.png"


def zone_requested(args: argparse.Namespace) -> bool:
    return (
        args.zone is not None
        or args.Cooling is not None
        or args.Heating is not None
        or args.Isoterma is not None
        or args.Return is not None
        or args.all_cooling
        or args.all_heating
    )


def zone_num_from_profile(zones_all, ztype, ordinal):
    return zone_number_by_type(zones_all, ztype, ordinal)


def zone_nums_by_type(zones_all, ztype):
    return zone_numbers_by_type(zones_all or [], ztype)


def handle_pt_plot(args: argparse.Namespace, data_path: Path, program_path: Path) -> None:
    """Handle --plot-pt and --plot-pt-time, then exit the CLI path."""
    load_program_path = None if args.full_data else program_path
    tracks = [(load_lvm(data_path, program_pdf=load_program_path), args.label or data_path.stem)]
    stems = [data_path.stem]

    if args.data2:
        data2_path, program2_path = resolve_data_program(
            args.data2,
            args.program2 if args.program2 else args.program,
        )
        load_program2_path = None if args.full_data else program2_path
        tracks.append((load_lvm(data2_path, program_pdf=load_program2_path), args.label2 or data2_path.stem))
        stems.append(data2_path.stem)

    outpath = Path(args.out) if args.out else default_pt_outpath(
        stems,
        include_time=args.plot_pt_time,
        full_data=args.full_data,
    )
    if len(outpath.parts) == 1:
        outpath = get_output_dir("figures", "pt_compare") / outpath
    from util.experimental_metadata import new_output
    outdir = new_output(outpath.parent, data_path.stem, vars(args))
    outpath = outdir / outpath.name
    plot_pt_tracks(
        tracks,
        outpath,
        include_time=args.plot_pt_time,
        show_initial_pressure=args.show_initial_pressure,
    )
    print(f"Figura: {outpath}")


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    data_path, program_path = resolve_data_program(args.data, args.program)

    if args.plot_pt or args.plot_pt_time:
        handle_pt_plot(args, data_path, program_path)
        raise SystemExit(0)

    from calvet_workflow import process
    process(args, data_path, program_path)
    raise SystemExit(0)
