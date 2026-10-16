"""Interface smoke check plus roundtrip of editable manual windows."""
import pandas as pd
import pytest

from src.calorimetry_app import _editor_rows, _selections


def test_editor_retains_review_proposal_and_manual_fit_regions():
    windows = [{'event_id':'weak','dataset_id':'run','window_s':[0,100],
                'baseline_s':[5,30], 'flank_s':[43,55], 'row_range':[0,200],
                'manual_classification':'accepted_event','proposal':{'snr':1.2},
                'phase_hypothesis':None}]
    table = pd.DataFrame(_editor_rows(windows))
    table.loc[0,'flank_start_s'] = 44.
    restored = _selections(table)[0]
    assert restored['flank_s'] == [44,55]
    assert restored['row_range'] == [0,200]
    assert restored['manual_classification'] == 'accepted_event'
    assert restored['proposal'] == {'snr':1.2}
    assert _selections(table.iloc[0:0]) == []


def test_common_streamlit_three_actions(tmp_path):
    testing = pytest.importorskip('streamlit.testing.v1')
    from src.calorimetry_io import SignalSet
    from src.calorimetry_store import save_bundle
    item = SignalSet('run',pd.DataFrame({'time_s':[0.,1.,2.], 'heat_flow':[0.,1.,0.]}),
                     {'time_s':'s','heat_flow':'mW'})
    saved = save_bundle(tmp_path/'session',[item])
    app = testing.AppTest.from_file(__import__('pathlib').Path(__file__).resolve().parents[1] / 'app.py').run(timeout=15)
    assert not app.exception
    app.radio[0].set_value('Select events and calculate onsets').run()
    app.text_input[0].set_value(str(saved)).run(timeout=15)
    assert not app.exception
    assert not app.error
    app.radio[0].set_value('Plot saved results').run()
    app.text_input[0].set_value(str(saved)).run(timeout=15)
    assert not app.exception
