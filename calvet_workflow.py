"""Calvet LVM workflow; TXT/PDF exports from its Calisto software have a separate reader."""
from __future__ import annotations

import json
import argparse
from pathlib import Path

from util.calvet_events import analyze_event
from util.calvet_stages import describe_stages, select_stages
from util.experimental_metadata import experimental_metadata, file_record, new_output, write_json
from util.lvm_io import load_lvm
from util.program_zones import program_zone_spans, zone_number_by_type, zone_numbers_by_type
from util.synchronization import time_diagnostics


def add_arguments(parser):
    parser.description = "Calvet HP: seleccionar etapas, guardar eventos y generar gráficos independientes."
    parser.add_argument("--output", choices=["signals", "pt", "hf-temperature", "hf-time", "none"],
                        type=lambda value: "signals" if value == "diagnostic" else value,
                        action="append", help="Salida independiente; repetible. Por defecto signals.")
    parser.add_argument("--no-plots", action="store_true", help="Guardar análisis sin figuras")
    parser.add_argument("--diagnostic-panel", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--time-basis", choices=["recorded", "legacy"], default="recorded",
                        help="recorded conserva el reloj exportado; legacy asume 0,5 s/fila")
    parser.add_argument("--time-range", nargs=2, type=float, metavar=("START_H", "END_H"))
    parser.add_argument("--stage-limits", nargs=2, type=float, metavar=("START_H", "END_H"),
                        help="Corregir manualmente los límites de una única etapa seleccionada")
    parser.add_argument("--stage", choices=["heating", "cooling", "isothermal"])
    parser.add_argument("--cycle", type=int, help="Ordinal dentro de --stage, desde 1")
    parser.add_argument("--event-window", nargs=2, type=float, metavar=("START_H", "END_H"))
    parser.add_argument("--event-id", default="event01")
    parser.add_argument("--baseline-window", nargs=2, type=float, metavar=("START_H", "END_H"))
    parser.add_argument("--flank-window", nargs=2, type=float, metavar=("START_H", "END_H"))
    parser.add_argument("--events", help="JSON con lista events, cada uno con ventanas independientes")
    parser.add_argument("--max-gap-s", type=float, default=5.0, help="Máxima separación contigua para ajustar/interpolar")
    parser.add_argument("--metadata", help="JSON de carga y corrida; campos desconocidos quedan pendientes")
    parser.add_argument("--experiment-id")
    parser.add_argument("--load-id")
    parser.add_argument("--run-id")
    parser.add_argument("--outdir", default="out/analysis/calvet")


def process(args, data_path: Path, program_path: Path):
    if args.full_data and any((args.zone, args.Cooling, args.Heating, args.Isoterma, args.Return,
                               args.all_cooling, args.all_heating, args.stage, args.cycle)):
        raise SystemExit("--full-data no se puede combinar con selección de zonas.")
    if args.max_gap_s <= 0:
        raise SystemExit("--max-gap-s debe ser positivo")
    data = load_lvm(data_path, None if args.full_data or args.stage_limits else program_path, time_basis=args.time_basis)
    # Keep un-clipped program limits to distinguish the plan from acquired conditions.
    spans = program_zone_spans(program_path)
    stages = describe_stages(data, spans)
    numbers = [args.zone] if args.zone is not None else []
    for flag, kind in (("Cooling", "Cooling"), ("Heating", "Heating"), ("Isoterma", "Isoterma"), ("Return", "Return")):
        ordinal = getattr(args, flag)
        if ordinal is not None:
            num = zone_number_by_type(spans, kind, ordinal)
            if num is None:
                raise SystemExit(f"{kind} {ordinal} no encontrado en PDF; use --stage para dirección observada.")
            numbers.append(num)
    if args.all_cooling:
        matches = zone_numbers_by_type(spans, "Cooling")
        if not matches:
            raise SystemExit("No hay zonas Cooling en el PDF")
        numbers += matches
    if args.all_heating:
        matches = zone_numbers_by_type(spans, "Heating")
        if not matches:
            raise SystemExit("No hay zonas Heating en el PDF; use --stage heating para dirección medida.")
        numbers += matches
    if args.stage_limits:
        candidates = [s for s in stages if (not numbers or s["num"] in numbers)
                      and (args.stage is None or s["selection_kind"] == args.stage)
                      and (args.cycle is None or s["cycle"] == args.cycle)]
        if len(candidates) != 1 or args.stage_limits[0] >= args.stage_limits[1]:
            raise SystemExit("--stage-limits requiere una única etapa seleccionada y límites inicio < fin.")
        candidate = candidates[0]
        candidate.update(program_start_h=candidate["start_h"], program_end_h=candidate["end_h"],
                         manual_limits_h=args.stage_limits, boundary_source="manual correction of PDF stage",
                         start_h=args.stage_limits[0], end_h=args.stage_limits[1])
    try:
        selected, selected_stages = select_stages(data, stages, numbers=numbers, kind=args.stage,
                                                  cycle=args.cycle, interval_h=args.time_range)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    metadata = experimental_metadata(args.metadata, experiment_id=args.experiment_id,
                                     load_id=args.load_id, run_id=args.run_id or data_path.stem)
    config = vars(args).copy()
    config["source"] = file_record(data_path)
    config["program"] = file_record(program_path)
    config["experimental_metadata"] = metadata
    config["event_definition_source"] = file_record(Path(args.events)) if args.events else None
    out = new_output(args.outdir, data_path.stem, config)
    specs = []
    if args.events:
        payload = json.loads(Path(args.events).read_text())
        specs = payload["events"] if isinstance(payload, dict) else payload
    if args.event_window or args.onset_around is not None:
        window = args.event_window
        if window is None:
            # The old onset-around option was relative to a selected zone when rebased.
            offset = selected_stages[0]["start_h"] if numbers and not args.no_rebase and len(selected_stages) == 1 else 0
            center = args.onset_around + offset
            window = [center-args.onset_window, center+args.onset_window]
        spec = {"event_id": args.event_id, "interval_h": window, "method": args.onset_method,
                "baseline_h": args.baseline_window, "flank_h": args.flank_window}
        if args.onset_method == "tangent":
            spec["params"] = {"flank_force_near_peak": True, "flank_near_peak_frac": args.flank_near_peak_frac}
        specs.append(spec)
    ids = {k: metadata[k] for k in ("experiment_id", "load_id", "run_id")}
    events = []
    event_ids = set()
    for i, spec in enumerate(specs, 1):
        spec = {"event_id": f"event{i:02d}", **spec}
        if spec["event_id"] in event_ids:
            raise ValueError("event_id must be unique within a run")
        event_ids.add(spec["event_id"])
        if "stage_id" not in spec:
            window = spec.get("interval_h", [selected.t_h.min(), selected.t_h.max()])
            matches = [s for s in selected_stages if s["start_h"] <= window[0] and s["end_h"] >= window[1]]
            spec["stage_id"] = matches[0]["stage_id"] if len(matches) == 1 else None
        event_frame = selected
        if spec.get("stage_id"):
            stage = next((s for s in selected_stages if s["stage_id"] == spec["stage_id"]), None)
            if stage is None:
                raise ValueError(f"Event stage not selected: {spec['stage_id']}")
            event_frame = selected.loc[selected.t_h.between(stage["start_h"], stage["end_h"])]
        event = analyze_event(event_frame, spec, ids, args.max_gap_s)
        if event["stage_id"] is None:
            event["warnings"].append("stage membership unresolved; provide stage_id or a contained event window")
        events.append(event)
    diagnostics = time_diagnostics(data)
    warnings = []
    recorded_diagnostics = diagnostics.get("recorded_clock", diagnostics)
    if recorded_diagnostics["duplicate_times"]:
        policy = "Event fits retain first row per timestamp" if args.time_basis == "recorded" else "Legacy analysis assumes 0.5 s per row"
        warnings.append(f"Recorded clock has repeated seconds. {policy}; no automatic offset.")
    if recorded_diagnostics["gaps"] or recorded_diagnostics["backward_steps"]:
        warnings.append("Time interruptions detected; select continuous event windows.")
    if metadata["pending_fields"]:
        warnings.append("Incomplete metadata limits composition/phase/equilibrium conclusions: " + ", ".join(metadata["pending_fields"]))
    if not spans:
        warnings.append("Program not available; no programmed stages assigned. Select --time-range manually.")
    # Missing signals are explicit, without substituting one temperature for another.
    for signal in ("T_muestra", "T_horno", "T_banho", "P_muestra", "P_referencia", "HF_mW"):
        if signal not in selected:
            warnings.append(f"Signal unavailable: {signal}")
    record = {"schema_version": 1, "workflow": "calvet_hp", "config": config, "metadata": metadata,
              "signals_file": "signals.csv", "units": data.attrs["units"], "provenance": data.attrs["provenance"],
              "synchronization": diagnostics, "program_stages": stages, "stages": selected_stages,
              "events": events, "warnings": warnings,
              "manual_temperature_targets": args.extra_onset_temp,
              "manual_targets_note": "Legacy temperature markers are manual observations, not calculated onsets."}
    selected.to_csv(out / "signals.csv", index=False)
    write_json(out / "results.json", record)
    write_json(out / "config.json", config)
    (out / "warnings.txt").write_text("\n".join(warnings))
    outputs = args.output or ["signals"]
    if args.diagnostic_panel and "signals" not in outputs:
        outputs.append("signals")
    if not args.no_plots:
        from util.calvet_plotting import plot_saved
        for kind in outputs:
            if kind != "none":
                plot_saved(out / "results.json", kind)
    for event in events:
        print(f"{event['event_id']}: {event['status']} {event.get('reason', '')}")
        if event["onset"]:
            print(f"  onset: {event['onset']}")
    for warning in warnings:
        print("Aviso:", warning)
    print(f"Resultados: {out / 'results.json'}")
    return out
