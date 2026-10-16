"""Command-line DSC/uDSC analyzer."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.cycle_mapping import assignments_frame, map_zones_to_data, segment_for_assignment
from src.event_detection import DEFAULT_CONFIG, detect_events
from src.pdf_parser import parse_program_pdf
from src.plotting import plot_cycle
from src.reporting import write_outputs
from src.txt_parser import read_txt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze Calisto DSC/uDSC PDF and TXT files.")
    parser.add_argument("--pdf", help="Calisto experiment PDF")
    parser.add_argument("--txt", nargs="+", help="TXT file(s) or folders")
    parser.add_argument("--outdir", default="out/dsc_analysis", help="Output directory")
    parser.add_argument("--load-id", help="Experimental charge ID for event traceability.")
    parser.add_argument("--run-id", help="Run ID; defaults to the TXT file stem.")
    parser.add_argument(
        "--zone-bounds-csv",
        type=Path,
        help="Optional manual zone bounds CSV with columns zone_number,start_s,end_s.",
    )
    parser.add_argument("--max-cycles", type=int, default=None, help="Limit cycles for quick validation")
    parser.add_argument(
        "--cycle",
        action="append",
        type=int,
        help="Cooling cycle number to analyze; may be repeated.",
    )
    parser.add_argument("--snr-detected", type=float, default=DEFAULT_CONFIG["snr_detected"])
    parser.add_argument("--snr-doubtful", type=float, default=DEFAULT_CONFIG["snr_doubtful"])
    parser.add_argument(
        "--phase-label-heuristic",
        action="store_true",
        help="Add LL/SL phase hypotheses based on cooling order; they remain pending review.",
    )
    parser.add_argument("--no-figures", action="store_true", help="Analyze and save results without figures.")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into a non-empty --outdir.")
    parser.add_argument(
        "--plot-saved",
        action="store_true",
        help="Render the already saved events and signals in --outdir without re-detecting events.",
    )
    return parser


def load_zone_bounds(path: str | Path) -> dict[int, tuple[float, float]]:
    frame = pd.read_csv(path)
    required = {"zone_number", "start_s", "end_s"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Zone bounds CSV is missing columns: {', '.join(sorted(missing))}")
    if frame["zone_number"].duplicated().any():
        raise ValueError("Zone bounds CSV contains duplicate zone_number values.")
    bounds: dict[int, tuple[float, float]] = {}
    for row in frame.itertuples(index=False):
        zone_number = int(row.zone_number)
        start_s = float(row.start_s)
        end_s = float(row.end_s)
        if not start_s < end_s:
            raise ValueError(f"Invalid manual bounds for zone {zone_number}: {start_s}..{end_s}")
        bounds[zone_number] = (start_s, end_s)
    return bounds


def analyze(
    pdf: str | Path,
    txt: str | Path,
    outdir: str | Path,
    max_cycles: int | None = None,
    config: dict | None = None,
    *,
    make_figures: bool = True,
    overwrite: bool = False,
    load_id: str | None = None,
    run_id: str | None = None,
    manual_zone_bounds: dict[int, tuple[float, float]] | None = None,
    cycle_numbers: list[int] | None = None,
) -> dict[str, Path | pd.DataFrame]:
    output_path = Path(outdir)
    if output_path.exists() and any(output_path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_path}. "
            "Choose a new --outdir or pass --overwrite."
        )
    program = parse_program_pdf(pdf)
    data = read_txt(txt)
    assignments = map_zones_to_data(program, data, manual_zone_bounds)
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    cfg["source_pdf"] = str(Path(pdf))
    cfg["source_data"] = str(data.source)
    cfg["time_unit"] = "s"
    cfg["original_columns"] = data.original_columns
    cfg["available_signal_columns"] = list(data.frame.columns)
    cfg["signal_units"] = {
        "time_s": "s",
        "temperature_c": "degC",
        "heat_flow_mw": "mW",
        "sample_temperature_c": "degC",
        "furnace_temperature_c": "degC",
        "huber_temperature_c": "degC",
        "setpoint_temperature_c": "degC",
    }
    cfg["selected_temperature_source"] = data.temperature_source
    cfg["synchronization"] = {
        "method": "Compare PDF cumulative elapsed time with raw TXT Time(s)",
        "offset_s": 0.0,
        "independent_sync_marker": None,
        "time_origin_assumed_to_match": True,
        "automatic_correction_applied": False,
        "thermal_lag_correction_applied": False,
    }

    rows: list[dict] = []
    event_rows: list[dict] = []
    signal_segments: dict[str, pd.DataFrame] = {}
    warnings = list(
        dict.fromkeys(
            [
                *program.warnings,
                *data.warnings,
                *(
                    assignment.warning
                    for assignment in assignments
                    if assignment.warning is not None
                ),
            ]
        )
    )
    figures_dir = Path(outdir) / "figures"
    cycle_count = 0
    for assignment in assignments:
        if assignment.zone.direction != "cooling" or assignment.status not in {"automatic", "manual"}:
            continue
        cycle_number = assignment.zone.cycle or cycle_count + 1
        if cycle_numbers and cycle_number not in cycle_numbers:
            continue
        cycle_count += 1
        if max_cycles is not None and cycle_count > max_cycles:
            break
        segment = segment_for_assignment(data, assignment)
        results, metrics, _ = detect_events(segment, assignment.zone.direction, cfg)
        cycle = cycle_number
        segment_name = f"cycle_{cycle:02d}_zone_{assignment.zone.zone_number:02d}.csv"
        signal_segments[segment_name] = segment
        for event_number, result in enumerate(results, 1):
            event = result.as_dict()
            event.update(
                {
                    "experiment_id": Path(pdf).stem,
                    "load_id": load_id,
                    "run_id": run_id or Path(txt).stem,
                    "cycle_id": cycle,
                    "stage_id": assignment.zone.zone_number,
                    "event_id": (
                        f"{load_id or Path(pdf).stem}:"
                        f"{run_id or Path(txt).stem}:cycle-{cycle:02d}:"
                        f"zone-{assignment.zone.zone_number:02d}:event-{event_number:02d}"
                    ),
                    "event_number": event_number,
                    "direction": assignment.zone.direction,
                    "source_file": str(data.source),
                    "signal_file": f"signals/{segment_name}",
                    "temperature_signal": data.temperature_column,
                    "temperature_source": data.temperature_source,
                    "temperature_unit": "degC",
                    "heat_flow_signal": data.heat_flow_column,
                    "heat_flow_unit": "mW",
                    "time_signal": data.time_column,
                    "time_unit": "s",
                    "pressure_signal": None,
                    "pressure_unit": None,
                    "pressure_at_onset": None,
                    "pressure_method": "unavailable",
                    "pressure_warning": "No pressure signal is present in this Calisto TXT data frame.",
                    "onset_time_s": None,
                    "peak_time_s": None,
                    "method": "tangent",
                    "parameters": cfg,
                    "manual_review": "pending",
                    "programmed_stage": assignment.zone.as_dict(),
                    "stage_selection": {
                        "source": assignment.selection_source,
                        "start_s": assignment.selected_start_s,
                        "end_s": assignment.selected_end_s,
                    },
                    "measured_temperature_start_c": (
                        float(segment["temperature_c"].iloc[0]) if len(segment) else None
                    ),
                    "measured_temperature_end_c": (
                        float(segment["temperature_c"].iloc[-1]) if len(segment) else None
                    ),
                    "warnings": list(data.warnings),
                }
            )
            if assignment.warning:
                event["warnings"] = list(
                    dict.fromkeys([*event["warnings"], assignment.warning])
                )
            if result.onset_temp_c is not None and len(segment):
                temp = segment["temperature_c"].to_numpy(float)
                onset_idx = int(abs(temp - result.onset_temp_c).argmin())
                event["onset_time_s"] = float(segment["time_s"].iloc[onset_idx])
                event["onset_time_method"] = "time of nearest measured temperature sample"
                event["onset_time_difference_c"] = float(temp[onset_idx] - result.onset_temp_c)
            if result.peak_temp_c is not None and len(segment):
                temp = segment["temperature_c"].to_numpy(float)
                peak_idx = int(abs(temp - result.peak_temp_c).argmin())
                event["peak_time_s"] = float(segment["time_s"].iloc[peak_idx])
                event["peak_time_method"] = "time of nearest measured temperature sample"
            event_rows.append(event)
        if metrics is not None:
            noise_dict = metrics.as_dict()
        else:
            noise_dict = {}
        figure = None
        if make_figures and len(segment):
            figure_path = figures_dir / f"cycle_{assignment.zone.cycle:02d}.png"
            if figure_path.exists() and not overwrite:
                figure_path = figure_path.with_name(
                    f"{figure_path.stem}_{len(list(figures_dir.glob(f'{figure_path.stem}*')))+1:02d}{figure_path.suffix}"
                )
            figure = plot_cycle(segment, results, figure_path)
        base = {
            "experiment": Path(pdf).stem,
            "composition": None,
            "cycle": cycle,
            "direction": assignment.zone.direction,
            "temperature_used": data.temperature_source,
            "zone_number": assignment.zone.zone_number,
            "figure": str(figure) if figure else None,
            "exclusion": False,
            "exclusion_reason": None,
        }
        by_label = {result.event_label: result for result in results}
        for label in ["TLL", "TSL"]:
            result = by_label.get(label)
            base[f"{label}_automatic"] = result.onset_temp_c if result else None
            base[f"{label}_manual"] = None
            base[f"{label}_calisto"] = None
            base[f"{label}_uncertainty"] = result.onset_uncertainty_c if result else None
            base[f"{label}_snr"] = result.snr if result else None
            base[f"{label}_confidence"] = result.confidence if result else "not evaluated"
            base[f"{label}_status"] = result.status if result else "not detected"
        base["TLL_minus_TSL"] = (
            base["TLL_automatic"] - base["TSL_automatic"]
            if base["TLL_automatic"] is not None and base["TSL_automatic"] is not None
            else None
        )
        base["observations"] = "; ".join(result.notes for result in results if result.notes)
        base["candidate_events"] = "; ".join(result.event_label for result in results)
        base.update(noise_dict)
        rows.append(base)

    results_frame = pd.DataFrame(rows)
    paths = write_outputs(
        outdir,
        results_frame,
        program.zones_frame(),
        assignments_frame(assignments),
        cfg,
        warnings,
        events=event_rows,
        signal_segments=signal_segments,
        overwrite=overwrite or make_figures,
    )
    return {"results": results_frame, **paths}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args()
    if args.plot_saved:
        from src.replay import plot_saved_analysis

        plot_saved_analysis(args.outdir, cycles=args.cycle)
        return
    if not args.pdf or not args.txt:
        raise SystemExit("--pdf and --txt are required unless --plot-saved is used.")
    if len(args.txt) != 1:
        raise SystemExit("This prototype supports one TXT file for automatic end-to-end analysis.")
    config = {
        "snr_detected": args.snr_detected,
        "snr_doubtful": args.snr_doubtful,
        "phase_label_heuristic": args.phase_label_heuristic,
    }
    try:
        manual_zone_bounds = (
            load_zone_bounds(args.zone_bounds_csv) if args.zone_bounds_csv else None
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Could not read manual zone bounds: {exc}") from exc
    output = analyze(
        args.pdf,
        args.txt[0],
        args.outdir,
        args.max_cycles,
        config,
        make_figures=not args.no_figures,
        overwrite=args.overwrite,
        manual_zone_bounds=manual_zone_bounds,
        cycle_numbers=args.cycle,
        load_id=args.load_id,
        run_id=args.run_id,
    )
    print(output["results"].to_string(index=False))
    print(f"Outputs written to {Path(args.outdir).resolve()}")


if __name__ == "__main__":
    main()
