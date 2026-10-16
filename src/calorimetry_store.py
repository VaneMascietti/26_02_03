"""Immutable-by-default shared bundles: original files, signals, selections, fits."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .calorimetry_io import SignalSet, fingerprint


SCHEMA = 'calorimetry-common-v1'


def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(json_value(value), indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def save_bundle(outdir, sets, events=(), selections=(), config=None, *, overwrite=False, parent=None):
    outdir = Path(outdir).resolve()
    # Never let an explicit overwrite target an input directory or an ancestor.
    source_paths = [Path(s['path']).resolve() for item in sets for s in item.metadata.get('sources', [])]
    if parent:
        parent_path = Path(parent).resolve()
        if parent_path.name == 'results.json':
            parent_path = parent_path.parent
        if outdir == parent_path or outdir in parent_path.parents or parent_path in outdir.parents:
            raise ValueError('Write a revision in a separate directory, outside the parent bundle.')
    for source in source_paths:
        if outdir == source or outdir in source.parents:
            raise ValueError('Output cannot contain or replace input files.')
    if outdir.exists() and any(outdir.iterdir()) and not overwrite:
        raise FileExistsError(f'Bundle already exists: {outdir}; choose a new directory or --overwrite.')
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / 'sources').mkdir(exist_ok=True)
    payload = {'schema': SCHEMA, 'created_utc': datetime.now(timezone.utc).isoformat(),
               'parent_bundle': str(parent) if parent else None, 'datasets': [],
               'events': list(events), 'selections': list(selections),
               'configuration': asdict(config) if config else None,
               'criteria_status': 'Operational, pending experimental validation; no automatic phase assignment.'}
    for index, item in enumerate(sets):
        csv_name = f'signals_{index:03d}.csv'
        item.frame.to_csv(outdir / csv_name, index=False)
        sources = []
        for n, source in enumerate(item.metadata.get('sources', [])):
            original = Path(source['path'])
            available = Path(source.get('archived_path', original))
            if not available.exists():
                raise FileNotFoundError(f'Cannot archive source: {available}')
            actual = fingerprint(available)
            if actual['sha256'] != source['sha256']:
                raise ValueError(f'Source changed since loading: {available}')
            relative = Path('sources') / f'{index:03d}_{n:02d}_{original.name}'
            shutil.copyfile(available, outdir / relative)
            sources.append({**source, 'archived_file': str(relative)})
        meta = {**item.metadata, 'sources': sources}
        payload['datasets'].append({'id': item.id, 'signals_file': csv_name,
                                    'signals_sha256': fingerprint(outdir / csv_name)['sha256'],
                                    'units': item.units, 'stages': item.stages, 'metadata': meta})
    write_json(outdir / 'results.json', payload)
    write_json(outdir / 'windows.json', list(selections))
    return outdir / 'results.json'


def load_bundle(path):
    path = Path(path).resolve()
    if path.is_dir():
        path = path / 'results.json'
    payload = json.loads(path.read_text(encoding='utf-8'))
    if payload.get('schema') != SCHEMA:
        raise ValueError('Not a common calorimetry bundle; use the legacy renderer for legacy results.')
    sets = []
    for entry in payload['datasets']:
        signal = (path.parent / entry['signals_file']).resolve()
        if path.parent not in signal.parents:
            raise ValueError('Signal path must remain inside bundle.')
        if fingerprint(signal)['sha256'] != entry['signals_sha256']:
            raise ValueError(f'Saved signals changed: {signal}')
        metadata = {**entry['metadata'], 'sources': [dict(s) for s in entry['metadata'].get('sources', [])]}
        for source in metadata['sources']:
            archive = (path.parent / source['archived_file']).resolve()
            if path.parent not in archive.parents:
                raise ValueError('Archived source must remain inside bundle.')
            source['archived_path'] = str(archive)
        sets.append(SignalSet(entry['id'], pd.read_csv(signal), entry['units'], entry['stages'], metadata))
    return payload, sets
