from pathlib import Path

from util.path import ensure_out_layout, get_data_path, get_output_dir


def test_get_data_path_fallback_to_legacy_data(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    legacy = Path("data") / "sample.lvm"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text("legacy", encoding="utf-8")

    resolved = get_data_path("sample.lvm", "lvm")
    assert resolved == legacy


def test_get_data_path_prefers_raw_layout(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    legacy = Path("data") / "sample.pdf"
    raw = Path("data/raw/pdf") / "sample.pdf"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    raw.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text("legacy", encoding="utf-8")
    raw.write_text("raw", encoding="utf-8")

    resolved = get_data_path("data/sample.pdf", "pdf")
    assert resolved == raw


def test_output_layout_helpers(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    ensure_out_layout()
    assert Path("out/figures").exists()
    assert Path("out/tables").exists()
    assert Path("out/reports").exists()

    figs = get_output_dir("figures", "process_dsc_data")
    assert figs == Path("out/figures/process_dsc_data")
    assert figs.exists()
