"""Experimental provenance with explicit pending values, without inferred compositions."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_json(path: Path, value):
    path.write_text(json.dumps(json_ready(value), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def file_record(path: Path | None) -> dict | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest()}


def new_output(root: str | Path, label: str, config: dict) -> Path:
    """Every invocation has its own configuration-labelled directory; never overwrite."""
    root = Path(root).resolve()
    allowed = Path("out").resolve()
    if not root.is_relative_to(allowed):
        raise ValueError("Las salidas deben estar dentro de out/.")
    fingerprint = hashlib.sha256(json.dumps(json_ready(config), sort_keys=True).encode()).hexdigest()[:10]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    result = root / f"{Path(label).stem}_{fingerprint}_{stamp}"
    result.mkdir(parents=True, exist_ok=False)
    return result


def experimental_metadata(path: str | None, *, experiment_id=None, load_id=None, run_id=None) -> dict:
    meta = json.loads(Path(path).read_text()) if path else {}
    defaults = {"experiment_id": experiment_id, "load_id": load_id, "run_id": run_id,
                "components": None, "loaded_masses": None, "co2_mass_determinations": [],
                "compositions": [], "effective_volume": None, "system_configuration": None,
                "program_calvet": None, "program_huber": None, "incidents": [],
                "manual_reviews": [], "uncertainties": None}
    for key, value in defaults.items():
        meta.setdefault(key, value)
    for key, value in (("experiment_id", experiment_id), ("load_id", load_id), ("run_id", run_id)):
        if value is not None:
            if meta[key] not in (None, value):
                raise ValueError(f"Conflicting metadata: {key}")
            meta[key] = value
    meta["pending_fields"] = [key for key in defaults if meta[key] is None or
                             (key in {"co2_mass_determinations", "compositions"} and not meta[key])]
    meta["limitations"] = "Exploratory data; phase assignment/equilibrium and enthalpy remain unvalidated."
    return meta
