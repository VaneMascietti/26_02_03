from pathlib import Path


_DATA_RAW_ROOT = Path("data/raw")
_DATA_LEGACY_ROOT = Path("data")
_OUT_ROOT = Path("out")

_KIND_TO_RAW_SUBDIR = {
    "lvm": "lvm",
    "pdf": "pdf",
}


def get_data_path(name: str | Path, kind: str) -> Path:
    """
    Resolve input data path with raw-first strategy and legacy fallback.

    Search order for data-like relative paths:
    1) data/raw/<kind>/<filename>
    2) data/<filename>

    If an explicit absolute/custom path is provided, it is respected.
    """
    if kind not in _KIND_TO_RAW_SUBDIR:
        raise ValueError(f"Unsupported kind: {kind!r}. Expected one of {tuple(_KIND_TO_RAW_SUBDIR)}")

    raw_subdir = _KIND_TO_RAW_SUBDIR[kind]
    in_path = Path(name)

    # For plain names or legacy data/ paths, prefer new raw layout.
    is_data_like_rel = (not in_path.is_absolute()) and (
        len(in_path.parts) == 1 or in_path.parts[0] == "data"
    )
    if is_data_like_rel:
        filename = in_path.name
        raw_candidate = _DATA_RAW_ROOT / raw_subdir / filename
        legacy_candidate = _DATA_LEGACY_ROOT / filename
        if raw_candidate.exists():
            return raw_candidate
        if legacy_candidate.exists():
            return legacy_candidate
        if in_path.exists():
            return in_path
        return raw_candidate

    # Respect explicit non-data paths first.
    if in_path.exists():
        return in_path

    # Convenience fallback by basename.
    raw_candidate = _DATA_RAW_ROOT / raw_subdir / in_path.name
    legacy_candidate = _DATA_LEGACY_ROOT / in_path.name
    if raw_candidate.exists():
        return raw_candidate
    if legacy_candidate.exists():
        return legacy_candidate
    return in_path


def ensure_out_layout() -> None:
    """Create standard output tree if missing."""
    (_OUT_ROOT / "figures").mkdir(parents=True, exist_ok=True)
    (_OUT_ROOT / "tables").mkdir(parents=True, exist_ok=True)
    (_OUT_ROOT / "reports").mkdir(parents=True, exist_ok=True)


def get_output_dir(category: str, subdir: str | None = None) -> Path:
    """Return an output directory under out/<category>[/<subdir>] creating it if needed."""
    ensure_out_layout()
    out_dir = _OUT_ROOT / category
    if subdir:
        out_dir = out_dir / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir
