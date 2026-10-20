import numpy as np
import pandas as pd
import pytest

from util.pressure_reconstruction import reconstruct_pressure


def sample():
    times = pd.date_range('2026-01-01',periods=10,freq='1s')
    f=pd.DataFrame({'datetime':times,'T_muestra':[100,80,60,40,20,10,80,60,40,10],
                    'P_muestra':[400,320,240,160,80,40,310,0,0,0], 'HF_mW':np.arange(10.)})
    r={'reference_start':str(times[0]), 'reference_end':str(times[5]),'last_measured_time':str(times[6]),
       'zero_offset_temperature_C':10.,'offset_at_cut_bar':-10.,'max_reference_temperature_gap_C':21.,
       'expected_rows':3}
    return f,r


def test_reconstruction_preserves_measured_values_and_applies_documented_taper():
    f,r=sample()
    result,meta=reconstruct_pressure(f,r)
    assert f.P_muestra.tolist() == [400,320,240,160,80,40,310,0,0,0]
    assert result.P_muestra.iloc[:7].tolist() == f.P_muestra.iloc[:7].tolist()
    assert result.P_muestra.iloc[7:].tolist() == pytest.approx([240-10*50/70,160-10*30/70,40])
    assert result.pressure_origin.tolist() == ['measured']*7+['reconstructed']*3
    pd.testing.assert_series_equal(result.HF_mW,f.HF_mW)
    assert meta['reconstructed_rows'] == 3
    assert meta['fit_performed'] is False and meta['extrapolation_performed'] is False


def test_reconstruction_rejects_extrapolation_existing_pressure_and_gaps():
    f,r=sample();f.loc[9,'T_muestra']=0
    with pytest.raises(ValueError,match='no extrapolation'):
        reconstruct_pressure(f,r)
    f,r=sample();f.loc[8,'P_muestra']=20
    with pytest.raises(ValueError,match='missing-pressure value'):
        reconstruct_pressure(f,r)
    f,r=sample();r['max_reference_temperature_gap_C']=5
    with pytest.raises(ValueError,match='temperature gap'):
        reconstruct_pressure(f,r)


def test_profile_marks_reconstructed_pressure_separately(tmp_path,monkeypatch):
    import matplotlib.figure
    from plot_time_profile import _clean_profile, plot_time_profile
    f,r=sample()
    f,_=reconstruct_pressure(f,r)
    f['t_s']=(f.datetime-f.datetime.iloc[0]).dt.total_seconds()
    f['t_h']=f.t_s/3600
    f['T_horno']=f.T_muestra
    f['T_banho']=f.T_muestra
    captured=[]
    monkeypatch.setattr(matplotlib.figure.Figure,'savefig',lambda fig,*a,**k: captured.append(fig))
    plot_time_profile(_clean_profile(f,'s'),tmp_path/'plot.png',data_label='test',unit='s',
                      temperature='all',title=None,heat_flow=True,pt_panel=True)
    lines=[line for axis in captured[0].axes for line in axis.lines]
    labels=[line.get_label() for line in lines]
    assert 'Measured sample pressure' in labels
    assert 'Reconstructed pressure (estimate)' in labels
    assert any('reconstructed pressure' in label for label in labels)
    assert next(line for line in lines if line.get_label()=='Reconstructed pressure (estimate)').get_linestyle()=='--'
