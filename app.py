"""Streamlit interface for the local DSC/uDSC analyzer."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from cli import analyze
from src.pdf_parser import parse_program_pdf
from src.txt_parser import read_txt
from util.experimental_metadata import new_output


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
    phase_heuristic = st.sidebar.checkbox("LL/SL order hypothesis (requires review)", value=False)
    direction = st.sidebar.selectbox("Stage direction", ["cooling", "heating", "isothermal"])

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
        zones = program.zones_frame()
        edited_zones = st.data_editor(zones, use_container_width=True,
                                     disabled=[col for col in zones if col not in {"start_s", "end_s"}])
        selected_cycle = st.sidebar.selectbox("Cycle", [None, *sorted({z.cycle for z in program.zones if z.cycle is not None})],
                                             format_func=lambda value: "All" if value is None else str(value))
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

        with st.expander("Independent event windows (optional JSON)"):
            st.caption("Use stage_id=zoneN, event_id, interval_s, baseline_s and flank_s, in seconds from run start.")
            event_json = st.text_area("Events", value="", placeholder='{"events": []}')

        if st.button("Run analysis"):
            config = {"snr_detected": snr_detected, "snr_doubtful": snr_doubtful,
                      "phase_heuristic": phase_heuristic, "direction": direction, "cycle": selected_cycle,
                      "zone_boundaries": {str(int(row.zone_number)): [float(row.start_s), float(row.end_s)]
                                          for row in edited_zones.itertuples()
                                          if not zones.loc[zones.zone_number == row.zone_number, ["start_s", "end_s"]].iloc[0].tolist() == [row.start_s, row.end_s]}}
            if event_json.strip():
                import json
                try:
                    payload = json.loads(event_json)
                    config["event_specs"] = payload["events"] if isinstance(payload, dict) else payload
                except (ValueError, KeyError) as exc:
                    st.error(f"Invalid event JSON: {exc}")
                    st.stop()
            outdir = new_output("out/streamlit_run", pdf_file.name, config)
            output = analyze(
                pdf_path,
                txt_path,
                outdir,
                int(max_cycles),
                config,
            )
            st.session_state["calisto_output"] = output

    if "calisto_output" in st.session_state:
        output = st.session_state["calisto_output"]
        results = output["results"]
        if isinstance(results, pd.DataFrame):
            st.subheader("Results and manual review")
            editable = [col for col in results if col.endswith("_manual") or col in {"exclusion", "exclusion_reason", "observations"}]
            reviewed = st.data_editor(results, use_container_width=True,
                                      disabled=[col for col in results if col not in editable], key="manual_results")
            if st.button("Save manual review"):
                from datetime import datetime, timezone
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                target = Path(output["saved_results"]).parent / f"manual_review_{stamp}.csv"
                reviewed.to_csv(target, index=False)
                st.success(f"Manual review saved: {target}")
        for figure in sorted((Path(output["saved_results"]).parent / "figures").glob("*.png")):
            st.image(str(figure), caption=figure.name)


if __name__ == "__main__":
    main()
