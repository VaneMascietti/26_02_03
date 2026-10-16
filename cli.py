"""Command-line DSC/uDSC analyzer."""

from __future__ import annotations

import argparse
import logging
import json
from pathlib import Path

import pandas as pd
import numpy as np

from src.cycle_mapping import assignments_frame, map_zones_to_data, segment_for_assignment
from src.event_detection import DEFAULT_CONFIG, detect_events
from src.pdf_parser import parse_program_pdf
from src.plotting import plot_cycle
from src.reporting import write_outputs
from src.txt_parser import read_txt
from util.experimental_metadata import file_record, new_output, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze Calvet TXT/PDF exports from Calisto software.")
    parser.add_argument("--pdf", help="Calisto experiment PDF")
    parser.add_argument("--txt", nargs="+", help="TXT file(s) or folders")
    parser.add_argument("--outdir", default="out/dsc_analysis", help="Output directory")
    parser.add_argument("--max-cycles", type=int, default=None, help="Limit cycles for quick validation")
    parser.add_argument("--snr-detected", type=float, default=DEFAULT_CONFIG["snr_detected"])
    parser.add_argument("--snr-doubtful", type=float, default=DEFAULT_CONFIG["snr_doubtful"])
    parser.add_argument("--phase-heuristic", action="store_true", help="LL/SL order hypothesis; pending review")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--cycle", type=int)
    parser.add_argument("--zone", type=int)
    parser.add_argument("--direction", choices=["cooling", "heating", "isothermal"], default="cooling")
    parser.add_argument("--render", help="Render saved calisto_results.json without analysis")
    parser.add_argument("--events", help="JSON: events with stage_id, event_id, interval_s, baseline_s, flank_s")
    parser.add_argument("--metadata", help="Experimental load/run JSON")
    return parser


def analyze(pdf: str | Path, txt: str | Path, outdir: str | Path, max_cycles: int | None = None, config: dict | None = None, *, make_plots=True) -> dict[str, Path | pd.DataFrame]:
    outdir = Path(outdir)
    if (outdir / "calisto_results.json").exists() or (outdir / "results_by_cycle.csv").exists():
        raise FileExistsError(f"Analysis already exists: {outdir}; choose a new directory")
    outdir.mkdir(parents=True, exist_ok=True)
    program = parse_program_pdf(pdf)
    data = read_txt(txt)
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    original_program = program.zones_frame().to_dict("records")
    for zone in program.zones:
        override = cfg.get("zone_boundaries", {}).get(str(zone.zone_number))
        if override is not None:
            start, end = map(float, override)
            if not start < end:
                raise ValueError("Manual zone limits require start < end")
            zone.start_s, zone.end_s = start, end
    assignments = map_zones_to_data(program, data)

    rows: list[dict] = []
    warnings = [*program.warnings, *data.warnings]
    from util.experimental_metadata import experimental_metadata
    metadata = experimental_metadata(cfg.get("metadata"))
    metadata["experiment_id"] = metadata["experiment_id"] or Path(pdf).stem
    metadata["run_id"] = metadata["run_id"] or cfg.get("run_id", Path(txt).stem)
    metadata["pending_fields"] = [key for key in metadata["pending_fields"] if metadata[key] is None or metadata[key] == []]
    if metadata["pending_fields"]:
        warnings.append("Incomplete experimental metadata: " + ", ".join(metadata["pending_fields"]))
    figures_dir = Path(outdir) / "figures"
    saved_segments = []
    cycle_count = 0
    for assignment in assignments:
        if assignment.zone.direction != cfg.get("direction", "cooling") or assignment.status != "automatic":
            continue
        if cfg.get("cycle") is not None and assignment.zone.cycle != cfg["cycle"]:
            continue
        if cfg.get("zone") is not None and assignment.zone.zone_number != cfg["zone"]:
            continue
        cycle_count += 1
        if max_cycles is not None and cycle_count > max_cycles:
            break
        segment = segment_for_assignment(data, assignment)
        rate = None
        if len(segment) > 1 and segment.time_s.is_monotonic_increasing and segment.time_s.nunique() > 1:
            rate = float(np.polyfit((segment.time_s-segment.time_s.iloc[0])/60, segment.temperature_c, 1)[0])
        observed = {"measured_rate_c_min": rate,
                    "temperature_range_c": [float(segment.temperature_c.min()), float(segment.temperature_c.max())] if len(segment) else None,
                    "temperature_source": data.temperature_source}
        if cfg.get("event_specs") is not None:
            from src.manual_events import analyze_manual_event
            specs = [s for s in cfg["event_specs"] if s["stage_id"] == f"zone{assignment.zone.zone_number}"]
            results = [analyze_manual_event(segment, spec) for spec in specs]
            metrics = None
        else:
            results, metrics, _ = detect_events(segment, assignment.zone.direction, cfg)
        for result in results:
            if result.peak_temp_c is not None:
                peak_idx = int((segment.temperature_c-result.peak_temp_c).abs().argmin())
                result.peak_time_s = float(segment.time_s.iloc[peak_idx])
            if result.onset_temp_c is not None:
                # Only interpolate on a monotonic local recorded branch; never through gaps.
                timing_segment = segment
                if result.requested_window_s:
                    timing_segment = segment.loc[segment.time_s.between(*result.requested_window_s)]
                temp = timing_segment.temperature_c.to_numpy(float)
                target = result.onset_temp_c
                crossings = []
                for i in range(1, len(temp)):
                    if min(temp[i-1], temp[i]) <= target <= max(temp[i-1], temp[i]) and temp[i] != temp[i-1]:
                        dt = float(timing_segment.time_s.iloc[i]-timing_segment.time_s.iloc[i-1])
                        if 0 < dt <= 5:
                            crossings.append(float(timing_segment.time_s.iloc[i-1]+dt*(target-temp[i-1])/(temp[i]-temp[i-1])))
                if len(crossings) == 1:
                    result.onset_time_s = crossings[0]
        segment_name = f"zone_{assignment.zone.zone_number:02d}.csv"
        segment.to_csv(outdir / segment_name, index=False)
        saved_segments.append({"zone": assignment.zone.as_dict(), "signals_file": segment_name,
                               "observed_conditions": observed,
                               "events": [{**r.as_dict(), "event_id": r.event_label if cfg.get("event_specs") is not None else f"event{i:02d}",
                                           **{k: metadata[k] for k in ("experiment_id", "load_id", "run_id")},
                                           "stage_id": f"zone{assignment.zone.zone_number}",
                                           "analysis_config": cfg, "manual_review": "pending",
                                           "signals_used": {"time": data.time_column, "temperature": data.temperature_column,
                                                            "heat_flow": data.heat_flow_column},
                                           "units": {"time": "s", "temperature": "degC", "heat_flow": "mW"},
                                           "pressure": {"value": None, "source": "unavailable in Calisto TXT"},
                                           "observation": "maximum baseline deviation" if cfg.get("event_specs") is not None else "candidate slope change",
                                           "calculation_reason": r.notes if r.onset_temp_c is None else "",
                                           "peak_definition": "maximum baseline deviation" if cfg.get("event_specs") is not None else
                                           "legacy detector candidate at high gradient; not an independently fitted peak"}
                                          for i, r in enumerate(results, 1)],
                               "pressure": {"value": None, "source": "unavailable in Calisto TXT"}})
        if metrics is not None:
            noise_dict = metrics.as_dict()
        else:
            noise_dict = {}
        figure = None
        if len(segment) and make_plots:
            figure = plot_cycle(segment, results, figures_dir / f"zone_{assignment.zone.zone_number:02d}.png")
        base = {
            "experiment": Path(pdf).stem,
            "composition": None,
            "cycle": assignment.zone.cycle,
            "direction": assignment.zone.direction,
            "temperature_used": data.temperature_source,
            "zone_number": assignment.zone.zone_number,
            "figure": str(figure) if figure else None,
            "exclusion": False,
            "exclusion_reason": None,
            "event_count": len(results),
            "event_onsets_c": json.dumps([r.onset_temp_c for r in results]),
            "event_statuses": json.dumps([r.status for r in results]),
            "phase_heuristic_enabled": cfg["phase_heuristic"],
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
        base.update(noise_dict)
        rows.append(base)

    results_frame = pd.DataFrame(rows)
    pd.DataFrame([event for segment in saved_segments for event in segment["events"]]).to_csv(outdir / "events.csv", index=False)
    write_json(outdir / "calisto_results.json", {
        "schema_version": 1, "workflow": "calisto", "config": cfg,
        "sources": {"pdf": file_record(Path(pdf)), "txt": file_record(Path(txt))},
        "metadata": metadata,
        "temperature_source": data.temperature_source, "original_columns": data.original_columns,
        "units": {"time_s": "s", "temperature_c": "degC", "heat_flow_mw": "mW"},
        "time_alignment": "shared TXT time column; no offset or thermal-lag correction applied",
        "manual_zone_boundaries": cfg.get("zone_boundaries", {}), "segments": saved_segments,
        "programmed_zones": original_program,
        "warnings": warnings, "phase_assignment": "pending; calorimetric onset is not validated equilibrium"})
    paths = write_outputs(
        outdir,
        results_frame,
        program.zones_frame(),
        assignments_frame(assignments),
        cfg,
        warnings,
    )
    return {"results": results_frame, "saved_results": outdir / "calisto_results.json", **paths}


def render_saved(path):
    """Recreate figures from persisted signals and EventResult records."""
    from src.models import EventResult
    from dataclasses import fields
    path = Path(path)
    record = json.loads(path.read_text())
    out = new_output("out/analysis/calisto_replots", path.parent.name, record["config"])
    allowed = {f.name for f in fields(EventResult)}
    for segment in record["segments"]:
        frame = pd.read_csv(path.parent / segment["signals_file"])
        events = [EventResult(**{k: v for k, v in event.items() if k in allowed}) for event in segment["events"]]
        plot_cycle(frame, events, out / f"zone_{segment['zone']['zone_number']:02d}.png")
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args()
    if args.render:
        print(render_saved(args.render))
        return
    if not args.pdf or not args.txt:
        raise SystemExit("Provide --pdf and --txt, or --render saved_results.json")
    if len(args.txt) != 1:
        raise SystemExit("This prototype supports one TXT file for automatic end-to-end analysis.")
    config = {"snr_detected": args.snr_detected, "snr_doubtful": args.snr_doubtful,
              "phase_heuristic": args.phase_heuristic, "cycle": args.cycle, "zone": args.zone,
              "direction": args.direction}
    if args.metadata:
        config["metadata"] = args.metadata
        config["metadata_source"] = file_record(Path(args.metadata))
    if args.events:
        payload = json.loads(Path(args.events).read_text())
        config["event_specs"] = payload["events"] if isinstance(payload, dict) else payload
    outdir = new_output(args.outdir, Path(args.pdf).stem, config)
    output = analyze(args.pdf, args.txt[0], outdir, args.max_cycles, config, make_plots=not args.no_plots)
    print(output["results"].to_string(index=False))
    print(f"Outputs written to {outdir.resolve()}")


if __name__ == "__main__":
    main()
