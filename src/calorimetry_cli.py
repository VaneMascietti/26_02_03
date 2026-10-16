"""Three shared actions: explore, analyze, plot (legacy CLIs remain available)."""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path

from .calorimetry_core import AnalysisConfig, analyze_selections, propose_windows
from .calorimetry_io import load_signals, SignalSet
from .calorimetry_store import load_bundle, save_bundle, write_json


def parser():
    p = argparse.ArgumentParser(description='Common TRIOS / Calisto / Calvet calorimetry workflow')
    sub = p.add_subparsers(dest='action', required=True)
    explore = sub.add_parser('explore', help='Load signals, preserve inputs and list stages; no onset fitting.')
    explore.add_argument('--format', required=True, choices=['trios', 'calisto', 'calvet'])
    explore.add_argument('--data', required=True)
    explore.add_argument('--program')
    explore.add_argument('--hf-unit', choices=['mW', 'W/g', 'mW/g'], help='Explicit HF unit declaration if export is ambiguous; no conversion.')
    explore.add_argument('--convention', default='unknown', choices=['unknown', 'endo_positive', 'exo_positive'])
    explore.add_argument('--trios-time', default='recorded', choices=['recorded', 'uniform'])
    explore.add_argument('--calvet-time', default='recorded', choices=['recorded', 'sample'],
                         help='Explicit sample option assumes a 0.5 s interval; recorded clock stays archived.')
    explore.add_argument('--outdir', required=True)
    explore.add_argument('--overwrite', action='store_true')
    analyze = sub.add_parser('analyze', help='Analyze an editable JSON list of any number of windows.')
    analyze.add_argument('--session', required=True)
    analyze.add_argument('--stages', help='Optional JSON list of reviewed stages: dataset_id,id,start_s,end_s,direction,cycle_id.')
    analyze.add_argument('--windows', help='JSON list; add, edit or remove entries to revise selections.')
    analyze.add_argument('--propose', action='store_true', help='Save automatic candidates for review, without analyzing them.')
    analyze.add_argument('--dataset', help='Dataset for automatic proposals; all if omitted.')
    analyze.add_argument('--stage', help='Stage ID for automatic proposals; optional.')
    analyze.add_argument('--direction', choices=['heating', 'cooling', 'isothermal'])
    analyze.add_argument('--outdir', required=True)
    analyze.add_argument('--overwrite', action='store_true')
    for f in fields(AnalysisConfig):
        analyze.add_argument('--' + f.name.replace('_', '-'), type=int if f.name == 'min_fit_points' else float,
                             default=getattr(AnalysisConfig(), f.name))
    plot = sub.add_parser('plot', help='Plot stored signals, regions, fit lines and intersections; no recalculation.')
    plot.add_argument('--session', required=True)
    plot.add_argument('--dataset')
    plot.add_argument('--event')
    plot.add_argument('--out', required=True)
    plot.add_argument('--color', default='#554173')
    plot.add_argument('--line-width', type=float, default=1.7)
    plot.add_argument('--line-style', choices=['-', '--', ':', '-.'], default='-')
    plot.add_argument('--font-size', type=float, default=18)
    plot.add_argument('--legend-size', type=float, default=14)
    plot.add_argument('--overwrite', action='store_true')
    plot.add_argument('--compare', nargs=2, action='append', metavar=('BUNDLE', 'EVENT_ID'),
                      help='Repeat at least twice to overlay saved events; no unit conversion.')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.action == 'explore':
        sets = load_signals(args.format, args.data, args.program, hf_unit=args.hf_unit,
                           convention=args.convention, trios_time=args.trios_time, calvet_time=args.calvet_time)
        path = save_bundle(args.outdir, sets, overwrite=args.overwrite)
        for item in sets:
            print(f'{item.id}: {len(item.frame)} rows; units={item.units}')
            print(json.dumps(item.stages, indent=2))
            for warning in item.metadata.get('warnings', []):
                print('WARNING:', warning)
            print('Time:', item.metadata['time_diagnostics'])
        print('Saved signals:', path)
    elif args.action == 'analyze':
        if not args.windows and not args.propose:
            raise ValueError('Supply --windows for manual analysis, or --propose for candidates only.')
        payload, sets = load_bundle(args.session)
        if args.stages:
            from .calorimetry_core import bounds
            stages = json.loads(Path(args.stages).read_text())
            seen = set()
            for stage in stages:
                dataset = next((s for s in sets if s.id == stage['dataset_id']), None)
                if dataset is None:
                    raise ValueError(f"Unknown stage dataset: {stage['dataset_id']}")
                bounds([stage['start_s'], stage['end_s']], 'stage bounds')
                if stage.get('direction') not in ('heating', 'cooling', 'isothermal', 'unknown'):
                    raise ValueError('Stage direction must be heating, cooling, isothermal or unknown.')
                identity = (dataset.id, str(stage['id']))
                if identity in seen:
                    raise ValueError('Duplicate stage identifier in stages file.')
                seen.add(identity)
                dataset.stages = [s for s in dataset.stages if str(s['id']) != str(stage['id'])]
                dataset.stages.append({**stage, 'source': 'manual stage revision'})
        config = AnalysisConfig(**{f.name: getattr(args, f.name) for f in fields(AnalysisConfig)})
        config.validate()
        selections = json.loads(Path(args.windows).read_text()) if args.windows else []
        if not isinstance(selections, list):
            raise ValueError('Windows JSON must be a list.')
        events = analyze_selections(sets, selections, config)
        proposals = []
        reports = []
        if args.propose:
            if args.dataset and not any(s.id == args.dataset for s in sets):
                raise ValueError(f'Unknown dataset: {args.dataset}')
            for item in sets:
                if args.dataset and item.id != args.dataset:
                    continue
                stages = [s for s in item.stages if (args.stage is None or str(s['id']) == args.stage)
                          and (args.direction is None or s['direction'] == args.direction)]
                if args.stage is not None or args.direction is not None:
                    if not stages:
                        raise ValueError(f'No matching stages in {item.id}')
                if not stages:
                    stages = [None]
                for stage in stages:
                    f = item.frame
                    if stage:
                        f = f.loc[f.time_s.between(stage['start_s'], stage['end_s'])]
                    subset = SignalSet(item.id, f, item.units, item.stages, item.metadata)
                    proposed = propose_windows(subset, config)
                    reports.append({'dataset_id': item.id, 'stage_id': stage['id'] if stage else None,
                                    'parameters': vars(args), **proposed})
                    for candidate in proposed['candidates']:
                        candidate['event_id'] = f"{item.id}:{stage['id'] if stage else 'all'}:{candidate['event_id']}"
                        if stage:
                            candidate['stage_id'] = stage['id']
                            candidate['cycle_id'] = stage.get('cycle_id')
                        proposals.append(candidate)
                    if proposed.get('reason'):
                        print(f"Proposal diagnostics ({item.id}): {proposed['reason']}")
        path = save_bundle(args.outdir, sets, events, selections, config,
                           overwrite=args.overwrite, parent=args.session)
        if args.propose:
            write_json(Path(args.outdir) / 'candidates.json', proposals)
            write_json(Path(args.outdir) / 'candidate_reports.json', reports)
            print('Review candidates.json and pass it as --windows in a new revision; proposals are not accepted events.')
        for event in events:
            print(event['event_id'], event['status'], event['reason'] or event['onset'])
        print('Saved analysis:', path)
    else:
        from .calorimetry_plot import render_bundle, compare_events
        if args.compare:
            path = compare_events(args.compare, args.out, overwrite=args.overwrite,
                                  font_size=args.font_size, line_width=args.line_width)
        else:
            path = render_bundle(args.session, args.out, dataset_id=args.dataset, event_id=args.event,
                                 color=args.color, line_width=args.line_width, line_style=args.line_style,
                                 font_size=args.font_size, legend_size=args.legend_size, overwrite=args.overwrite)
        print('Saved plot:', path)
    return 0
