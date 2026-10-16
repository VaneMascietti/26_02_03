"""Common Streamlit workflow, called by app.py; editors retain user selections."""
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from .calorimetry_core import AnalysisConfig, REVIEWS, analyze_selections, propose_windows
from .calorimetry_io import load_signals, SignalSet
from .calorimetry_store import load_bundle, save_bundle
from .calorimetry_plot import render_bundle, compare_events


def _editor_rows(selections):
    rows = []
    for s in selections:
        row = {k: s.get(k) for k in ['event_id', 'dataset_id', 'stage_id', 'cycle_id', 'run_id', 'manual_classification', 'phase_hypothesis', 'notes']}
        row['polarity'] = s.get('polarity', 'auto')
        for key in ['window', 'baseline', 'flank']:
            interval = s.get(f'{key}_s') or [None, None]
            row[f'{key}_start_s'], row[f'{key}_end_s'] = interval
        row['row_start'], row['row_end'] = s.get('row_range') or [None, None]
        row['proposal_json'] = json.dumps(s.get('proposal')) if s.get('proposal') else ''
        rows.append(row)
    return rows


def _selections(rows):
    selections = []
    for row in rows.to_dict('records'):
        if not row.get('event_id') or pd.isna(row.get('event_id')):
            continue
        s = {k: v for k, v in row.items() if not k.endswith(('_start_s', '_end_s'))
             and k not in ('proposal_json', 'row_start', 'row_end') and v is not None and not pd.isna(v)}
        for key in ['window', 'baseline', 'flank']:
            a, b = row.get(f'{key}_start_s'), row.get(f'{key}_end_s')
            if a is not None and b is not None and not pd.isna(a) and not pd.isna(b):
                s[f'{key}_s'] = [float(a), float(b)]
        if pd.notna(row.get('row_start')) and pd.notna(row.get('row_end')):
            s['row_range'] = [int(row['row_start']), int(row['row_end'])]
        if row.get('proposal_json'):
            s['proposal'] = json.loads(row['proposal_json'])
        s.setdefault('manual_classification', 'pending_review')
        s.setdefault('polarity', 'auto')
        selections.append(s)
    return selections


def main(st):
    st.caption('Shared time-domain analysis. Operational criteria require experimental validation. No automatic LL/SL assignment.')
    action = st.radio('Action', ['Explore signals', 'Select events and calculate onsets', 'Plot saved results'], horizontal=True)
    if action == 'Explore signals':
        format = st.selectbox('Instrument format', ['calvet', 'calisto', 'trios'])
        data = st.file_uploader('Instrument data', type={'calvet':['lvm'], 'calisto':['txt'], 'trios':['xls']}[format])
        program = st.file_uploader('Thermal program PDF', type=['pdf']) if format != 'trios' else None
        convention = st.selectbox('Recorded endo/exo convention', ['unknown', 'endo_positive', 'exo_positive'])
        unit = st.selectbox('Heat-flow unit declaration (no conversion)', ['from export', 'mW', 'W/g', 'mW/g'])
        trios_time = st.selectbox('TRIOS time', ['recorded', 'uniform']) if format == 'trios' else 'recorded'
        calvet_time = st.selectbox('Calvet time (sample assumes 0.5 s)', ['recorded', 'sample']) if format == 'calvet' else 'recorded'
        out = st.text_input('New session directory', 'out/common/session')
        if st.button('Load and save signals') and data:
            try:
                with TemporaryDirectory() as tmp:
                    source = Path(tmp) / Path(data.name).name
                    source.write_bytes(data.getvalue())
                    pdf = None
                    if program:
                        pdf = Path(tmp) / Path(program.name).name
                        pdf.write_bytes(program.getvalue())
                    sets = load_signals(format, source, pdf, hf_unit=None if unit == 'from export' else unit,
                                        convention=convention, trios_time=trios_time, calvet_time=calvet_time)
                    path = save_bundle(out, sets)
                st.session_state['common_bundle'] = str(path)
                st.success(f'Saved: {path}')
            except (ValueError, OSError) as exc:
                st.error(str(exc))
        bundle = st.session_state.get('common_bundle')
        if bundle:
            payload, sets = load_bundle(bundle)
            for item in sets:
                st.subheader(item.id)
                st.json({'units': item.units, 'metadata': item.metadata})
                st.dataframe(pd.DataFrame(item.stages))
                st.line_chart(item.frame.set_index('time_s')[['heat_flow']])
                temperatures = [c for c in item.frame if c.endswith('temperature_c') or c == 'temperature_c']
                if temperatures:
                    st.line_chart(item.frame.set_index('time_s')[temperatures])
                if 'pressure' in item.frame:
                    st.line_chart(item.frame.set_index('time_s')[['pressure']])
    elif action == 'Select events and calculate onsets':
        bundle = st.text_input('Saved session or analysis bundle', st.session_state.get('common_bundle', 'out/common/session'))
        if not (Path(bundle).is_file() or (Path(bundle) / 'results.json').is_file()):
            return
        try:
            payload, sets = load_bundle(bundle)
            dataset_id = st.selectbox('Dataset', [s.id for s in sets])
            item = next(s for s in sets if s.id == dataset_id)
            st.write('Stages from program/sheet (timing alignment requires review)')
            st.dataframe(pd.DataFrame(item.stages))
            stage_id = st.selectbox('Stage', ['manual bounds'] + [str(s['id']) for s in item.stages])
            stage = next((s for s in item.stages if str(s['id']) == stage_id), None)
            low = st.number_input('Stage start (s)', value=float(stage['start_s'] if stage else item.frame.time_s.min()))
            high = st.number_input('Stage end (s)', value=float(stage['end_s'] if stage else item.frame.time_s.max()))
            if low >= high:
                st.error('Stage start must precede end.')
                return
            selected_stage = {**(stage or {}), 'id': str(stage_id), 'start_s': low, 'end_s': high,
                              'direction': st.selectbox('Stage direction', ['heating', 'cooling', 'isothermal', 'unknown'],
                                                       index=['heating','cooling','isothermal','unknown'].index(stage.get('direction','unknown')) if stage else 3),
                              'source': 'user reviewed/edited stage bounds'}
            item.stages = [s for s in item.stages if str(s['id']) != stage_id] + [selected_stage]
            f = item.frame.loc[item.frame.time_s.between(low, high)]
            st.line_chart(f.set_index('time_s')[['heat_flow']])
            settings = {}
            with st.expander('Operational criteria and automatic selection settings'):
                for key, value in asdict(AnalysisConfig()).items():
                    if key == 'max_gap_s':
                        entered = st.number_input('Maximum gap (s), 0 = sampling-based limit', value=0., min_value=0.)
                        settings[key] = entered or None
                    else:
                        settings[key] = st.number_input(key, value=value, min_value=0 if isinstance(value, int) else 0.)
            config = AnalysisConfig(**settings)
            config.validate()
            key = f'windows:{bundle}:{item.id}:{stage_id}'
            if key not in st.session_state:
                st.session_state[key] = [s for s in payload['selections'] if s['dataset_id'] == item.id]
            st.write('Add/delete rows to add/remove events. Edit window, baseline and initial flank in seconds. Blank fit regions use automatic selection.')
            rows = _editor_rows(st.session_state[key])
            columns = ['event_id','dataset_id','stage_id','cycle_id','run_id','manual_classification','phase_hypothesis','notes','polarity',
                       'window_start_s','window_end_s','baseline_start_s','baseline_end_s','flank_start_s','flank_end_s','row_start','row_end','proposal_json']
            table = st.data_editor(pd.DataFrame(rows, columns=columns), num_rows='dynamic', key=key+':editor:'+str(st.session_state.get(key+':version',0)),
                                   column_config={'manual_classification':st.column_config.SelectboxColumn(options=list(REVIEWS)),
                                                  'polarity':st.column_config.SelectboxColumn(options=['auto','positive','negative'])},
                                   use_container_width=True)
            if st.button('Propose candidates for review'):
                proposals = propose_windows(SignalSet(item.id, f, item.units, item.stages, item.metadata), config)
                existing = _selections(table)
                ids = {s['event_id'] for s in existing}
                for candidate in proposals['candidates']:
                    name = candidate['event_id']
                    while name in ids:
                        name += '-new'
                    candidate['event_id'] = name
                    candidate['stage_id'] = stage_id
                    ids.add(name)
                    existing.append(candidate)
                st.session_state[key] = existing
                st.session_state[key+':version'] = st.session_state.get(key+':version',0) + 1
                st.session_state[key+':proposal_note'] = proposals.get('reason') or 'Candidates added for review.'
                st.rerun()
            if st.session_state.get(key+':proposal_note'):
                st.info(st.session_state[key+':proposal_note'])
            out = st.text_input('New analysis revision directory', 'out/common/revision')
            if st.button('Calculate, save and review fits'):
                selections = _selections(table)
                for s in selections:
                    s.setdefault('dataset_id', item.id)
                    s.setdefault('stage_id', stage_id)
                selections = [s for s in payload['selections'] if s['dataset_id'] != item.id] + selections
                events = analyze_selections(sets, selections, config)
                path = save_bundle(out, sets, events, selections, config, parent=bundle)
                st.session_state[key] = [s for s in selections if s['dataset_id'] == item.id]
                st.session_state['common_last_analysis'] = str(path)
                st.success(f'Saved: {path}')
            if st.session_state.get('common_last_analysis'):
                stored = st.session_state['common_last_analysis']
                saved, _ = load_bundle(stored)
                for event in saved['events']:
                    st.json(event)
                    with TemporaryDirectory() as tmp:
                        target = render_bundle(stored, Path(tmp)/'event.png', event_id=event['event_id'])
                        st.image(str(target))
        except (ValueError, OSError, KeyError) as exc:
            st.error(str(exc))
    else:
        bundle = st.text_input('Saved analysis bundle', st.session_state.get('common_last_analysis', 'out/common/revision'))
        if not (Path(bundle).is_file() or (Path(bundle) / 'results.json').is_file()):
            return
        try:
            payload, sets = load_bundle(bundle)
            dataset = st.selectbox('Dataset to plot', [s.id for s in sets])
            event = st.selectbox('Event (all = overview)', ['all'] + [e['event_id'] for e in payload['events'] if e['dataset_id'] == dataset])
            color = st.color_picker('Heat-flow color', '#554173')
            width = st.number_input('Line width', min_value=.1, value=1.7)
            style = st.selectbox('Line style', ['-', '--', ':', '-.'])
            font = st.number_input('Axis text size (pt)', min_value=6., value=18.)
            out = st.text_input('New figure path (.png, .pdf, .svg)', 'out/common/figure.png')
            if st.button('Plot stored results'):
                target = render_bundle(bundle, out, dataset_id=dataset, event_id=None if event == 'all' else event,
                                       color=color, line_width=width, line_style=style, font_size=font)
                st.success(str(target))
                if target.suffix.lower() == '.png':
                    st.image(str(target))
            with st.expander('Compare events from cycles or runs'):
                comparisons = st.text_area('JSON list of [bundle_path, event_id] pairs', '[]')
                compare_out = st.text_input('New comparison figure path', 'out/common/comparison.png')
                if st.button('Plot comparison'):
                    target = compare_events(json.loads(comparisons), compare_out, font_size=font, line_width=width)
                    st.success(str(target))
                    if target.suffix.lower() == '.png':
                        st.image(str(target))
        except (ValueError, OSError, KeyError) as exc:
            st.error(str(exc))
