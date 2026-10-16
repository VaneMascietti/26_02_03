"""Selection of program zones and observed conditions on a shared time axis."""
from __future__ import annotations

import numpy as np
import pandas as pd


def measured_conditions(frame: pd.DataFrame) -> dict:
    valid = frame[["t_h", "T_muestra"]].dropna()
    if len(valid) < 2 or valid.t_h.iloc[-1] <= valid.t_h.iloc[0]:
        return {"measured_rate_c_min": None, "measured_kind": "undetermined"}
    rate = float(np.polyfit(valid.t_h - valid.t_h.iloc[0], valid.T_muestra, 1)[0] / 60)
    return {"measured_rate_c_min": rate,
            "measured_kind": "isothermal" if abs(rate) < 0.005 else ("heating" if rate > 0 else "cooling"),
            "measured_temperature_range_c": [float(valid.T_muestra.min()), float(valid.T_muestra.max())]}


def describe_stages(frame: pd.DataFrame, spans: list[dict]) -> list[dict]:
    stages = []
    ordinals = {"heating": 0, "cooling": 0, "isothermal": 0, "unclassified": 0}
    for span in spans:
        stage = dict(span)
        name = str(span["type"]).lower()
        kind = next((value for prefix, value in (("heating", "heating"), ("cooling", "cooling"),
                    ("isot", "isothermal")) if name.startswith(prefix)), "unclassified")
        part = frame.loc[frame.t_h.between(span["start_h"], span["end_h"])]
        observed = measured_conditions(part)
        # Inicio is retained literally; its observed direction is not a parsed setpoint.
        selection_kind = kind if kind != "unclassified" else observed["measured_kind"]
        ordinals.setdefault(selection_kind, 0)
        ordinals[selection_kind] += 1
        stages.append({**stage, "stage_id": f"zone{span['num']}", "programmed_kind": kind,
                       "selection_kind": selection_kind, "cycle": ordinals[selection_kind],
                       "cycle_definition": "ordinal within direction; not a verified thermodynamic cycle",
                       "boundary_source": "program PDF; start alignment not independently verified",
                       "programmed_rate_c_min": None, **observed})
    return stages


def select_stages(frame: pd.DataFrame, stages: list[dict], *, numbers=None, kind=None, cycle=None,
                  interval_h=None) -> tuple[pd.DataFrame, list[dict]]:
    unknown = set(numbers or []) - {s["num"] for s in stages}
    if unknown:
        raise ValueError(f"No hay etapas con números {sorted(unknown)}; disponibles: {[s['num'] for s in stages]}")
    if cycle is not None and kind is None and not numbers:
        raise ValueError("--cycle requiere --stage para definir el ordinal de la etapa.")
    selected = [dict(s) for s in stages if (not numbers or s["num"] in numbers)
                and (kind is None or s["selection_kind"] == kind)
                and (cycle is None or s["cycle"] == cycle)]
    if (numbers or kind or cycle is not None) and not selected:
        raise ValueError("No hay etapas que coincidan con la selección.")
    if interval_h is not None and (len(interval_h) != 2 or interval_h[0] >= interval_h[1]):
        raise ValueError("El intervalo debe tener inicio < fin (horas desde el inicio de la corrida).")
    mask = pd.Series(False, index=frame.index)
    if numbers or kind or cycle is not None:
        for stage in selected:
            mask |= frame.t_h.between(stage["start_h"], stage["end_h"])
    else:
        mask[:] = True
    if interval_h is not None:
        mask &= frame.t_h.between(*interval_h)
    result = frame.loc[mask].copy()
    if result.empty:
        raise ValueError("La selección no contiene datos.")
    for stage in selected:
        lo = max(stage["start_h"], float(result.t_h.min()))
        hi = min(stage["end_h"], float(result.t_h.max()))
        stage["selected_start_h"], stage["selected_end_h"] = lo, hi
        if interval_h is not None:
            stage["manual_limits_h"] = list(interval_h)
        stage.update(measured_conditions(result.loc[result.t_h.between(lo, hi)]))
    return result, [s for s in selected if s["selected_start_h"] <= s["selected_end_h"]]
