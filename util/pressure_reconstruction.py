"""Documented reference-cycle pressure reconstruction; never refit instrument data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .lvm_io import load_lvm


def reconstruct_pressure(frame, recipe):
    """Interpolate measured P(T), then taper a supplied offset to zero.

    Windows and offset are supplied explicitly, not inferred or optimized.
    Missing-pressure rows are replaced in a copy; measured values remain intact.
    """
    start = pd.Timestamp(recipe['reference_start'])
    end = pd.Timestamp(recipe['reference_end'])
    cutoff = pd.Timestamp(recipe['last_measured_time'])
    if not start < end < cutoff:
        raise ValueError('Require reference_start < reference_end < last_measured_time.')
    reference = frame.loc[frame.datetime.between(start, end)]
    if len(reference) < 3 or not np.isfinite(reference[['T_muestra','P_muestra']].to_numpy(float)).all():
        raise ValueError('Reference requires finite measured T and P.')
    if (reference.P_muestra <= 0).any():
        raise ValueError('Reference contains invalid/nonpositive pressures.')
    times = reference.datetime.diff().dt.total_seconds().dropna()
    positive = times[times > 0]
    if (times < 0).any() or (len(positive) and (positive > 5 * positive.median()).any()):
        raise ValueError('Reference crosses a clock reset or acquisition gap.')
    target = frame.datetime > cutoff
    if not target.any() or not frame.loc[target,'P_muestra'].eq(recipe.get('missing_value',0.)).all():
        raise ValueError('All rows after last_measured_time must contain the documented missing-pressure value.')
    if recipe.get('expected_rows') is not None and target.sum() != recipe['expected_rows']:
        raise ValueError('Missing-pressure row count differs from recipe.')
    last = frame.loc[~target].iloc[-1]
    if last.datetime != cutoff:
        raise ValueError('Documented last measured timestamp not found.')
    anchor = float(last.T_muestra)
    taper = float(recipe['zero_offset_temperature_C'])
    offset = float(recipe['offset_at_cut_bar'])
    if not np.isfinite([anchor,taper,offset]).all() or anchor <= taper:
        raise ValueError('Invalid offset/taper temperatures.')
    # Duplicate temperature readings are grouped deterministically; do not
    # confuse this lookup curve with acquisition-order onset fitting.
    curve = reference.groupby('T_muestra',sort=True).P_muestra.median()
    temps = frame.loc[target,'T_muestra'].to_numpy(float)
    if not np.isfinite(temps).all() or temps.min() < curve.index.min() or temps.max() > curve.index.max():
        raise ValueError('Target temperatures outside measured reference; no extrapolation allowed.')
    right = np.searchsorted(curve.index.to_numpy(),temps,side='left')
    right = np.clip(right,1,len(curve)-1)
    gaps = curve.index.to_numpy()[right] - curve.index.to_numpy()[right-1]
    if np.any(gaps > recipe['max_reference_temperature_gap_C']):
        raise ValueError('Reference temperature gap too large for interpolation.')
    correction = offset * np.clip((temps-taper)/(anchor-taper),0,1)
    estimates = np.interp(temps,curve.index.to_numpy(),curve.to_numpy()) + correction
    if not np.isfinite(estimates).all() or np.any(estimates <= 0):
        raise ValueError('Reconstruction produced invalid pressures.')
    result = frame.copy()
    result['P_muestra_recorded'] = frame.P_muestra
    result['pressure_reconstructed'] = target
    result['pressure_origin'] = np.where(target,'reconstructed','measured')
    result['P_muestra'] = result.P_muestra.astype(float)
    result.loc[target,'P_muestra'] = estimates
    if 'P_muestra_bar' in result:
        result['P_muestra_bar'] = result.P_muestra_bar.astype(float)
        result.loc[target,'P_muestra_bar'] = estimates
    metadata = {'method':'linear interpolation of measured reference P(T) plus supplied temperature-tapered offset',
                'status':'estimated pressure, not a measurement; recreated from documented recipe',
                'recipe':recipe, 'reconstructed_rows':int(target.sum()),
                'duplicate_reference_temperatures':'median of recorded pressure at identical T',
                'reference_rows':len(reference), 'reference_temperature_range_C':[float(curve.index.min()),float(curve.index.max())],
                'anchor_temperature_C':anchor,'last_measured_pressure_bar':float(last.P_muestra),
                'predicted_pressure_at_anchor_bar':float(np.interp(anchor,curve.index,curve.values)+offset),
                'first_estimated_pressure_bar':float(estimates[0]),
                'estimated_pressure_range_bar':[float(estimates.min()),float(estimates.max())],
                'fit_performed':False, 'extrapolation_performed':False}
    return result,metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--recipe',type=Path,required=True)
    parser.add_argument('--outdir',type=Path,required=True)
    args=parser.parse_args()
    recipe=json.loads(args.recipe.read_text())
    source=args.data.resolve()
    out=args.outdir.resolve()
    if out in source.parents or out == source:
        raise ValueError('Output must be separate from raw input.')
    if out.exists():
        raise FileExistsError('Choose a new reconstruction output directory; previous results are preserved.')
    frame=load_lvm(source,layout=recipe['layout'])
    result,metadata=reconstruct_pressure(frame,recipe)
    # Preserve all original lines/fields, replacing only the pressure field on
    # explicitly reconstructed rows. Verify row alignment before writing.
    mask=result.pressure_reconstructed.to_numpy(bool)
    values=result.P_muestra.to_numpy(float)
    count=sum(1 for _ in source.open(newline=''))
    if count != len(result):
        raise ValueError('Raw lines do not align with parsed rows; refusing to alter columns.')
    out.mkdir(parents=True)
    derived=out/(source.stem+'_pressure_reconstructed.lvm')
    with source.open(newline='') as original, derived.open('w',newline='') as target:
        for index,line in enumerate(original):
            if mask[index]:
                fields=line.split('\t')
                field=recipe['raw_pressure_column_zero_based']
                if float(fields[field]) != recipe.get('missing_value',0.):
                    raise ValueError('Raw pressure field is inconsistent with reconstruction mask.')
                fields[field]=f'{values[index]:.6f}'
                line='\t'.join(fields)
            target.write(line)
    metadata['source_file']=str(source)
    metadata['source_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    metadata['derived_file']=derived.name
    metadata['derived_sha256']=hashlib.sha256(derived.read_bytes()).hexdigest()
    result[['datetime','T_muestra','P_muestra_recorded','P_muestra','pressure_origin']].to_csv(out/'pressure_provenance.csv',index=False)
    (out/'reconstruction.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False))
    print(json.dumps(metadata,indent=2,ensure_ascii=False))


if __name__ == '__main__':
    main()
