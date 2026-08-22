"""Streamlit interface for the local DSC/uDSC analyzer."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

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
        edited_zones = st.data_editor(program.zones_frame(), use_container_width=True)
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
            outdir = Path("outputs/streamlit_run")
            output = analyze(
                pdf_path,
                txt_path,
                outdir,
                int(max_cycles),
                {"snr_detected": snr_detected, "snr_doubtful": snr_doubtful},
            )
            results = output["results"]
            if isinstance(results, pd.DataFrame):
                st.subheader("Results")
                st.data_editor(results, use_container_width=True)
            st.subheader("Editable zone table")
            st.dataframe(edited_zones, use_container_width=True)
            for figure in sorted((outdir / "figures").glob("*.png")):
                st.image(str(figure), caption=figure.name)


if __name__ == "__main__":
    main()

