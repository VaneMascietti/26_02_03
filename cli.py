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
    parser.add_argument("--pdf", required=True, help="Calisto experiment PDF")
    parser.add_argument("--txt", required=True, nargs="+", help="TXT file(s) or folders")
    parser.add_argument("--outdir", default="outputs/dsc_analysis", help="Output directory")
    parser.add_argument("--max-cycles", type=int, default=None, help="Limit cycles for quick validation")
    parser.add_argument("--snr-detected", type=float, default=DEFAULT_CONFIG["snr_detected"])
    parser.add_argument("--snr-doubtful", type=float, default=DEFAULT_CONFIG["snr_doubtful"])
    return parser


def analyze(pdf: str | Path, txt: str | Path, outdir: str | Path, max_cycles: int | None = None, config: dict | None = None) -> dict[str, Path | pd.DataFrame]:
    program = parse_program_pdf(pdf)
    data = read_txt(txt)
    assignments = map_zones_to_data(program, data)
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)

    rows: list[dict] = []
    warnings = [*program.warnings, *data.warnings]
    figures_dir = Path(outdir) / "figures"
    cycle_count = 0
    for assignment in assignments:
        if assignment.zone.direction != "cooling" or assignment.status != "automatic":
            continue
        cycle_count += 1
        if max_cycles is not None and cycle_count > max_cycles:
            break
        segment = segment_for_assignment(data, assignment)
        results, metrics, _ = detect_events(segment, assignment.zone.direction, cfg)
        if metrics is not None:
            noise_dict = metrics.as_dict()
        else:
            noise_dict = {}
        figure = None
        if len(segment):
            figure = plot_cycle(segment, results, figures_dir / f"cycle_{assignment.zone.cycle:02d}.png")
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
    paths = write_outputs(
        outdir,
        results_frame,
        program.zones_frame(),
        assignments_frame(assignments),
        cfg,
        warnings,
    )
    return {"results": results_frame, **paths}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args()
    if len(args.txt) != 1:
        raise SystemExit("This prototype supports one TXT file for automatic end-to-end analysis.")
    config = {"snr_detected": args.snr_detected, "snr_doubtful": args.snr_doubtful}
    output = analyze(args.pdf, args.txt[0], args.outdir, args.max_cycles, config)
    print(output["results"].to_string(index=False))
    print(f"Outputs written to {Path(args.outdir).resolve()}")


if __name__ == "__main__":
    main()

