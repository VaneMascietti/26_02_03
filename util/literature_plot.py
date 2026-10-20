"""Literature overlay for existing P-T axes, independent of instrument readers."""
from pathlib import Path
import re
import hashlib

import numpy as np
import pandas as pd

from .phase_points import load_points


def load_extracted_workbook(path, system):
    """Adapt extracted workbooks without inferring phases from compositions.

    Sheet phase_type is authoritative. L1-L2-V is normalized to LLV; endpoints
    keep their own classes. Sheets lacking phase labels stay unclassified.
    Phase compositions remain distinct from overall composition xCO2.
    """
    if not system:
        raise ValueError('Excel literature requires --literature-system to declare the binary system.')
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    frames = []
    with pd.ExcelFile(path) as book:
        for sheet in book.sheet_names:
            raw = pd.read_excel(book, sheet_name=sheet)
            if not {'T_C', 'P_bar', 'source', 'doi'}.issubset(raw.columns):
                continue
            frame = raw.copy()
            for column in ['T_C', 'P_bar']:
                frame[column] = pd.to_numeric(frame[column], errors='raise')
            if not np.isfinite(frame[['T_C', 'P_bar']].to_numpy(float)).all():
                raise ValueError(f'Nonfinite literature coordinates in {sheet}')
            if 'T_K' in frame and not np.allclose(frame.T_K - 273.15, frame.T_C, atol=.011, rtol=0):
                raise ValueError(f'Inconsistent Celsius/Kelvin coordinates in {sheet}')
            if 'P_MPa' in frame and not np.allclose(frame.P_MPa * 10, frame.P_bar, atol=.011, rtol=0):
                raise ValueError(f'Inconsistent bar/MPa coordinates in {sheet}')
            original = frame.get('phase_type', pd.Series('unclassified', index=frame.index)).fillna('unclassified')
            frame['transition_original'] = original
            frame['transition_type'] = original.replace({'L1-L2-V': 'LLV',
                'K point / UCEP': 'UCEP', 'S-L1-L2-V Q point': 'Q point'})
            frame['ID'] = sheet.split('_', 1)[0]
            frame['Sistema'] = system
            frame['Referencia'] = frame['source']
            frame['source_original'] = frame['source']
            frame['source'] = 'literature'
            frame['DOI'] = frame['doi']
            frame['source_file'] = str(path.resolve())
            frame['source_sha256'] = digest
            frame['source_sheet'] = sheet
            frame['source_excel_row'] = np.arange(len(frame)) + 2
            frames.append(frame)
    if not frames:
        raise ValueError('No extracted P-T literature tables found in workbook.')
    return pd.concat(frames, ignore_index=True)


def load_literature(path, *, ids=None, transitions=None, compositions=None, system=None):
    path = Path(path)
    points = load_extracted_workbook(path, system) if path.suffix.lower() == ".xlsx" else load_points(path)
    if system and path.suffix.lower() != ".xlsx":
        if "Sistema" not in points:
            raise ValueError("CSV must declare Sistema to filter by system.")
        points = points.loc[points.Sistema.eq(system)]
    if 'source' in points:
        points = points.loc[points.source.str.lower().eq('literature')]
    if ids:
        unknown = set(ids) - set(points.ID)
        if unknown:
            raise ValueError(f'Unknown literature IDs: {sorted(unknown)}')
        points = points.loc[points.ID.isin(ids)]
    if transitions:
        unknown = set(transitions) - set(points.transition_type)
        if unknown:
            raise ValueError(f'Unknown literature transitions in selection: {sorted(unknown)}')
        points = points.loc[points.transition_type.isin(transitions)]
    if compositions:
        if 'xCO2' not in points:
            raise ValueError('Literature CSV has no xCO2 column.')
        keep = np.zeros(len(points), dtype=bool)
        for value in compositions:
            keep |= np.isclose(points.xCO2, value, atol=5e-5, rtol=0)
        points = points.loc[keep]
    if points.empty:
        raise ValueError('No literature points match the requested selection.')
    # Preserve identifiers, original transition names, composition, references and DOI.
    return points.copy()


def plot_literature(ax, points):
    """Plot unconnected symbols; never interpolate between literature compositions."""
    points = points.copy()
    points['_reference'] = points.ID.astype(str).str.replace(r'_x.*$', '', regex=True)
    points['_system'] = points.get('Sistema', pd.Series('System unspecified', index=points.index))
    colors = ['#16846b', '#ba5694', '#6d633b', '#408b9d']
    references = sorted(points._reference.unique())
    markers = {'TLL':'o', 'LL':'o', 'LLV':'v', 'SL':'s', 'SLL':'D', 'LV':'^'}
    for (reference, system, transition), group in points.groupby(['_reference','_system','transition_type'], sort=True):
        citation = re.sub(r'(?<=[A-Za-z])(?=\d{4})', ' ', reference)
        citation = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', citation)
        citation = citation.replace('van Der Steen', 'van der Steen')
        label = f'{citation} — {transition}'
        if 'xCO2' in group and group.xCO2.notna().any():
            low, high = group.xCO2.min(), group.xCO2.max()
            composition = f'{low:.4f}' if low == high else f'{low:.4f}–{high:.4f}'
            label += f' (xCO₂={composition})'
        if points._system.nunique() > 1:
            label += f' [{system}]'
        ax.scatter(group.T_C, group.P_bar, marker=markers.get(transition, 'd'),
                   s=32, facecolors='none', edgecolors=colors[references.index(reference) % len(colors)],
                   linewidths=1., label=label, zorder=3)
