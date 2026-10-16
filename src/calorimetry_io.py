"""Instrument adapters for the shared calorimetry workflow. No HF conversion."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import re

import numpy as np
import pandas as pd

from .pdf_parser import parse_program_pdf
from .txt_parser import read_txt
from .trios_parser import read_trios_xls
from util.lvm_io import load_lvm, lvm_time_diagnostics
from util.program_zones import program_zone_spans


@dataclass
class SignalSet:
    id: str
    frame: pd.DataFrame
    units: dict
    stages: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def fingerprint(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return {'path': str(path), 'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}


def _unit(text, override=None):
    if override:
        return override
    text = text.replace(' ', '').lower()
    for pattern, unit in [('mw/g', 'mW/g'), ('w/g', 'W/g'), ('mw', 'mW')]:
        if pattern in text:
            return unit
    raise ValueError('Heat-flow unit is not declared or supported; supply --hf-unit explicitly.')


def _clock_breaks(time):
    steps = np.diff(np.asarray(time, dtype=float))
    positive = steps[steps > 0]
    limit = 5 * float(np.median(positive)) if len(positive) else np.inf
    return np.r_[False, (steps < 0) | (steps > limit)]


def load_signals(format, source, program=None, *, hf_unit=None, convention='unknown',
                 trios_time='recorded', calvet_time='recorded'):
    """Preserve acquisition order and original columns; time conversion is explicit."""
    source = Path(source).resolve()
    sources = [fingerprint(source)]
    if program:
        sources.append(fingerprint(program))
    metadata = {'format': format, 'sources': sources, 'endo_exo_convention': convention,
                'heat_flow_conversion': 'none', 'warnings': [], 'unit_override': hf_unit}
    sets = []
    if format == 'trios':
        details, segments = read_trios_xls(source)
        for segment in segments:
            f = segment.frame.copy()
            # Older parser names describe conventional units, not a conversion.
            units = getattr(segment, 'declared_units', None)
            if not units or not units[0] or not units[1]:
                raise ValueError('TRIOS must declare time and temperature units.')
            time_unit = units[0].strip().lower()
            scale = {'min': 60., 's': 1., 'sec': 1.}.get(time_unit)
            if scale is None or units[1].strip() not in ('°C', 'C', 'degC', '℃'):
                raise ValueError(f'Unsupported TRIOS time/temperature units: {units[:2]}')
            unit = _unit(units[2], hf_unit)
            f['time_s'] = f.time_min * scale
            f['source_discontinuity'] = _clock_breaks(f.time_s)
            meta = {**metadata, 'workbook': details, 'time_source': 'recorded',
                    'temperature_source': 'instrument temperature; sample identity unverified',
                    'time_conversion': f'{time_unit} multiplied by {scale} to seconds'}
            if trios_time == 'uniform':
                if np.any(np.diff(f.time_s) < 0):
                    raise ValueError('Uniform reconstruction cannot repair TRIOS time resets; select source segments.')
                if f.time_s.iloc[-1] <= f.time_s.iloc[0]:
                    raise ValueError('Cannot reconstruct a nonpositive TRIOS duration.')
                f['recorded_time_s'] = f.time_s
                f['time_s'] = np.linspace(f.time_s.iloc[0], f.time_s.iloc[-1], len(f))
                meta['time_source'] = 'explicit uniform reconstruction between recorded endpoints'
                meta['warnings'] = ['Uniform time assumes constant acquisition; original rounded times retained.']
            f['heat_flow'] = f.heat_flow_w_g
            stage = {'id': segment.sheet_name, 'cycle_id': None, 'direction': segment.direction,
                     'start_s': float(f.time_s.iloc[0]), 'end_s': float(f.time_s.iloc[-1]),
                     'source': 'workbook sheet; direction inferred from temperature endpoints'}
            sets.append(SignalSet(segment.sheet_name, f, {'time_s': 's', 'temperature_c': units[1],
                                  'heat_flow': unit}, [stage], meta))
    elif format == 'calisto':
        if not program:
            raise ValueError('Calisto requires --program PDF.')
        data = read_txt(source, require_temperature=False)
        f = data.frame.copy()
        # read_txt preserves numeric HF values even when its legacy column says mW.
        unit = _unit(data.heat_flow_column, hf_unit)
        if data.temperature_column and not re.search(r'°\s*C|deg\s*C|\(C\)', data.temperature_column, re.I):
            raise ValueError('Calisto temperature must explicitly declare Celsius.')
        if not re.search(r'\(s\)|\bsec(?:ond)?s?\b|minute|\bmin\b', data.time_column, re.I):
            raise ValueError('Calisto time must explicitly declare seconds or minutes.')
        f['heat_flow'] = f.heat_flow_mw
        parsed = parse_program_pdf(program)
        stages = [{'id': str(z.zone_number), 'cycle_id': z.cycle, 'direction': z.direction,
                   'start_s': z.start_s, 'end_s': z.end_s,
                   'source': 'program PDF; timing alignment requires review'} for z in parsed.zones]
        # Retain any explicitly recorded pressure, with its declared unit.
        from .txt_parser import _read_rows, _to_float
        from .models import WarningLog
        header, rows = _read_rows(source, WarningLog())
        raw = pd.DataFrame(rows, columns=header)
        pressure_col = next((c for c in header if re.search(r'pressure', c, re.I)), None)
        signal_units = {'time_s': 's', 'temperature_c': '°C', 'heat_flow': unit}
        if pressure_col:
            match = re.search(r'\b(bar|MPa|kPa|Pa)\b', pressure_col, re.I)
            if not match:
                raise ValueError('Recorded pressure has no supported declared unit.')
            f['pressure'] = raw[pressure_col].map(_to_float).iloc[f.source_row.to_numpy()].to_numpy()
            signal_units['pressure'] = match.group(1)
        meta = {**metadata, 'warnings': [*data.warnings, *parsed.warnings],
                'temperature_source': data.temperature_source,
                'original_columns': data.original_columns, 'time_source': 'recorded',
                'time_conversion': 'TXT header minutes converted to seconds; otherwise seconds (legacy reader)'}
        sets.append(SignalSet(source.stem, f, signal_units, stages, meta))
    elif format == 'calvet':
        if not program:
            raise ValueError('Calvet requires --program PDF.')
        f = load_lvm(source, program_pdf=None)  # Do not truncate original signals to PDF duration.
        if calvet_time not in ('recorded', 'sample'):
            raise ValueError('Unknown Calvet time source.')
        f['source_discontinuity'] = _clock_breaks(f.t_s_datetime)
        f['time_s'] = f.t_s_datetime if calvet_time == 'recorded' else f.t_s_sample_index
        f['heat_flow'] = f.HF_mW
        mapping = {'T_muestra': 'sample_temperature_c', 'T_horno': 'furnace_temperature_c',
                   'T_banho': 'huber_temperature_c', 'P_muestra': 'pressure'}
        for col, name in mapping.items():
            if col in f:
                f[name] = f[col]
        if 'sample_temperature_c' in f:
            f['temperature_c'] = f.sample_temperature_c
        stages = []
        for z in program_zone_spans(Path(program)):
            kind = str(z['type']).lower()
            direction = next((d for d in ['heating', 'cooling'] if d in kind), 'unknown')
            if 'isot' in kind:
                direction = 'isothermal'
            stages.append({'id': str(z['num']), 'cycle_id': None, 'direction': direction,
                           'start_s': z['start_h'] * 3600, 'end_s': z['end_h'] * 3600,
                           'source': 'program PDF; timing alignment requires review'})
        if hf_unit and hf_unit != 'mW':
            raise ValueError('Calvet HF_mW is mW; unit override cannot relabel it.')
        meta = {**metadata, 'time_source': ('recorded device-clock elapsed seconds' if calvet_time == 'recorded'
                                else 'explicit sample-index time at assumed 0.5 s acquisition'),
                'temperature_source': 'sample' if 'temperature_c' in f else 'unavailable',
                'clock_diagnostics': lvm_time_diagnostics(f),
                'warnings': ['LVM column identities use the existing layout heuristic; review against instrument export.']}
        sets.append(SignalSet(source.stem, f, {'time_s': 's', 'heat_flow': 'mW',
                                             'temperature_c': '°C', 'pressure': 'bar'}, stages, meta))
    else:
        raise ValueError(f'Unknown format: {format}')
    for item in sets:
        if 'source_row' in item.frame:
            steps = item.frame.source_row.diff().fillna(1)
            item.frame['source_discontinuity'] = steps.ne(1) | item.frame.get('source_discontinuity', False)
        t = item.frame.time_s.to_numpy(float)
        item.metadata['time_diagnostics'] = {
            'nonpositive_intervals': int(np.sum(np.diff(t) <= 0)),
            'nonfinite_times': int(np.sum(~np.isfinite(t))),
            'source_discontinuities': int(item.frame.get('source_discontinuity', pd.Series(dtype=bool)).sum()),
            'policy': 'No sorting, automatic synchronization, or duplicate removal. Fit a strictly increasing interval.'}
        if not item.stages:
            item.metadata['warnings'].append('No PDF stages parsed; select stage bounds manually.')
    return sets
