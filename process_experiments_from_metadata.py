import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


DEFAULT_METADATA = Path("data/metadata/experiments.csv")
PROCESS_SCRIPT = Path("process_dsc_data.py")
REQUIRED_COLUMNS = {
    "experiment_id",
    "load_id",
    "run_id",
    "date",
    "stage",
    "file_lvm",
    "file_program",
    "notes",
}

STAGE_ZONE_FLAGS = {
    "heating": "--all-heating",
    "cooling": "--all-cooling",
}


def load_metadata(path: Path) -> pd.DataFrame:
    """Load and validate one-row-per-LVM experiment metadata."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        missing_txt = ", ".join(sorted(missing))
        raise ValueError(f"Metadata file is missing required columns: {missing_txt}")

    for column in REQUIRED_COLUMNS:
        df[column] = df[column].str.strip()

    empty_file_rows = df.index[(df["file_lvm"] == "") | (df["file_program"] == "")]
    if len(empty_file_rows) > 0:
        row_nums = ", ".join(str(i + 2) for i in empty_file_rows)
        raise ValueError(f"Rows with empty file_lvm or file_program: {row_nums}")

    return df


def filter_metadata(
    df: pd.DataFrame,
    experiment_id: str | None = None,
    load_id: str | None = None,
    run_id: str | None = None,
    stage: str | None = None,
) -> pd.DataFrame:
    filtered = df
    filters = {
        "experiment_id": experiment_id,
        "load_id": load_id,
        "run_id": run_id,
        "stage": stage,
    }
    for column, value in filters.items():
        if value is None:
            continue
        if column == "stage":
            filtered = filtered[filtered[column].str.lower() == value.lower()]
        else:
            filtered = filtered[filtered[column] == value]
    return filtered.reset_index(drop=True)


def command_for_row(
    row: pd.Series,
    python_executable: str,
    process_script: Path = PROCESS_SCRIPT,
    select_stage_zones: bool = False,
    extra_args: list[str] | None = None,
) -> list[str]:
    cmd = [
        python_executable,
        str(process_script),
        "--data",
        row["file_lvm"],
        "--program",
        row["file_program"],
    ]

    if select_stage_zones:
        stage_flag = STAGE_ZONE_FLAGS.get(row["stage"].lower())
        if stage_flag:
            cmd.append(stage_flag)

    if extra_args:
        cmd.extend(extra_args)

    return cmd


def run_commands(
    rows: pd.DataFrame,
    python_executable: str,
    process_script: Path,
    select_stage_zones: bool,
    extra_args: list[str],
    dry_run: bool,
    keep_going: bool,
) -> int:
    failures = 0

    for _, row in rows.iterrows():
        label = (
            f"{row['experiment_id']}/{row['load_id']}/{row['run_id']} "
            f"{row['date']} {row['stage']}"
        )
        cmd = command_for_row(
            row,
            python_executable=python_executable,
            process_script=process_script,
            select_stage_zones=select_stage_zones,
            extra_args=extra_args,
        )
        print(f"\n==> {label}")
        print(" ".join(cmd))

        if dry_run:
            continue

        completed = subprocess.run(cmd, check=False)
        if completed.returncode != 0:
            failures += 1
            if not keep_going:
                return completed.returncode

    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Procesar corridas .lvm listadas en data/metadata/experiments.csv.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--experiment-id", type=str, default=None)
    parser.add_argument("--load-id", type=str, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--stage", type=str, default=None)
    parser.add_argument("--python", dest="python_executable", default=sys.executable)
    parser.add_argument("--process-script", type=Path, default=PROCESS_SCRIPT)
    parser.add_argument(
        "--select-stage-zones",
        action="store_true",
        help=(
            "Para stage=heating usa --all-heating y para stage=cooling usa "
            "--all-cooling al llamar a process_dsc_data.py."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Mostrar comandos sin ejecutarlos.")
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continuar con las filas siguientes si una corrida falla.",
    )
    parser.add_argument(
        "process_args",
        nargs=argparse.REMAINDER,
        help=(
            "Argumentos extra para process_dsc_data.py. Usar despues de --, "
            "por ejemplo: -- --onset-method simple."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    extra_args = args.process_args
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    try:
        metadata = load_metadata(args.metadata)
        selected = filter_metadata(
            metadata,
            experiment_id=args.experiment_id,
            load_id=args.load_id,
            run_id=args.run_id,
            stage=args.stage,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    if selected.empty:
        print("No metadata rows matched the requested filters.")
        return 0

    return run_commands(
        selected,
        python_executable=args.python_executable,
        process_script=args.process_script,
        select_stage_zones=args.select_stage_zones,
        extra_args=extra_args,
        dry_run=args.dry_run,
        keep_going=args.keep_going,
    )


if __name__ == "__main__":
    raise SystemExit(main())
