from pathlib import Path

from co2_phase_plot import _infer_iso_label, load_nist_iso_file


def test_infer_iso_label_from_density_pattern():
    label = _infer_iso_label(Path("nist_iso_D0p83611.txt"))
    assert "0.83611" in label


def test_load_nist_iso_file_parses_tp_and_density(tmp_path):
    iso = tmp_path / "iso_test.txt"
    rows = ["Header"]
    rows.extend([f"{20 + i} {100 + i} 0.8" for i in range(6)])
    iso.write_text("\n".join(rows), encoding="utf-8")

    df, density = load_nist_iso_file(iso)

    assert list(df.columns) == ["T", "P"]
    assert len(df) == 6
    assert float(df["T"].iloc[0]) == 20.0
    assert float(df["P"].iloc[0]) == 100.0
    assert density == 0.8
