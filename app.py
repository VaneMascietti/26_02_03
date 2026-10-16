"""Streamlit interface for the local DSC/uDSC analyzer."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from datetime import datetime
import json

import pandas as pd

from cli import analyze
from src.pdf_parser import parse_program_pdf
from src.txt_parser import read_txt


def main() -> None:
    try:
        import streamlit as st
    except ModuleNotFoundError as exc:  # pragma: no cover - user-facing dependency guard
        raise SystemExit("Install Streamlit with `pip install streamlit` to run app.py") from exc

    st.set_page_config(page_title="DSC/uDSC analyzer", layout="wide")
    st.title("DSC/uDSC analyzer")

    pdf_file = st.file_uploader("Calisto PDF", type=["pdf"])
    txt_file = st.file_uploader("Calisto TXT", type=["txt"])
    snr_detected = st.sidebar.number_input("Detected SNR threshold", value=3.0, min_value=0.5, step=0.1)
    snr_doubtful = st.sidebar.number_input("Doubtful SNR threshold", value=2.0, min_value=0.5, step=0.1)
    max_cycles = st.sidebar.number_input("Max cycles", value=3, min_value=1, step=1)
    load_id = st.sidebar.text_input("Load ID (optional)")
    run_id = st.sidebar.text_input("Run ID (optional)")
    phase_label_heuristic = st.sidebar.checkbox(
        "Include LL/SL phase hypotheses (manual review required)",
        value=False,
    )

    if not pdf_file or not txt_file:
        st.stop()

    with TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        pdf_path = tmpdir / pdf_file.name
        txt_path = tmpdir / txt_file.name
        pdf_path.write_bytes(pdf_file.getvalue())
        txt_path.write_bytes(txt_file.getvalue())

        program = parse_program_pdf(pdf_path)
        data = read_txt(txt_path)
        st.subheader("Thermal program")
        original_zones = program.zones_frame()
        edited_zones = st.data_editor(original_zones, use_container_width=True)
        st.subheader("TXT diagnostics")
        st.write(
            {
                "rows": len(data.frame),
                "temperature used": data.temperature_source,
                "original columns": data.original_columns,
            }
        )
        for warning in [*program.warnings, *data.warnings]:
            st.warning(warning)

        if st.button("Run analysis"):
            outdir = Path("out/streamlit_runs") / (
                f"{Path(pdf_file.name).stem}_{datetime.now().strftime('%Y%m%dT%H%M%S%f')}"
            )
            manual_bounds = {}
            if {"zone_number", "start_s", "end_s"}.issubset(edited_zones.columns):
                original_by_zone = original_zones.set_index("zone_number")
                for row in edited_zones.itertuples(index=False):
                    original = original_by_zone.loc[row.zone_number]
                    if row.start_s != original["start_s"] or row.end_s != original["end_s"]:
                        manual_bounds[int(row.zone_number)] = (
                            float(row.start_s),
                            float(row.end_s),
                        )
            output = analyze(
                pdf_path,
                txt_path,
                outdir,
                int(max_cycles),
                {
                    "snr_detected": snr_detected,
                    "snr_doubtful": snr_doubtful,
                    "phase_label_heuristic": phase_label_heuristic,
                },
                load_id=load_id or None,
                run_id=run_id or None,
                manual_zone_bounds=manual_bounds or None,
            )
            results = output["results"]
            if isinstance(results, pd.DataFrame):
                st.subheader("Results")
                st.data_editor(results, use_container_width=True)
            if "events_json" in output:
                st.subheader("Candidate events")
                event_file = output["events_json"]
                if isinstance(event_file, Path):
                    event_payload = json.loads(event_file.read_text(encoding="utf-8"))
                    st.dataframe(pd.DataFrame(event_payload["events"]), use_container_width=True)
            st.subheader("Editable zone table")
            st.dataframe(edited_zones, use_container_width=True)
            for figure in sorted((outdir / "figures").glob("*.png")):
                st.image(str(figure), caption=figure.name)


if __name__ == "__main__":
    main()
