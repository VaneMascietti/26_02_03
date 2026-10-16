"""Synthetic correctness checks, not experimental validation."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.calorimetry_core import (AnalysisConfig, analyze_window, analyze_selections,
                                  sample_signal, oscillation_indicator, propose_windows)
from src.calorimetry_io import SignalSet, load_signals, fingerprint
from src.calorimetry_store import save_bundle, load_bundle
from src.noise_analysis import analyse_noise


def signal(dt=.5, cooling=True, amplitude=1., noise=0., unit='mW'):
    t = np.arange(0, 120 + dt / 2, dt)
    peak = np.maximum(0, np.minimum((t - 40) / 20, (80 - t) / 20))
    y = .02 * t + amplitude * peak + np.random.default_rng(2).normal(0, noise, len(t))
    return SignalSet('run', pd.DataFrame({'time_s': t, 'heat_flow': y,
                     'temperature_c': 50 + (-.1 if cooling else .1) * t,
                     'pressure': 200 + t}), {'time_s': 's', 'heat_flow': unit,
                                            'temperature_c': '°C', 'pressure': 'bar'})


def selection(**kwargs):
    return {'event_id': 'one', 'dataset_id': 'run', 'window_s': [0., 100.],
            'baseline_s': [5., 30.], 'flank_s': [43., 55.], **kwargs}


@pytest.mark.parametrize('dt', [.1, .5, 2.])
@pytest.mark.parametrize('cooling', [True, False])
@pytest.mark.parametrize('amplitude', [.02, 2., -2.])
def test_shared_geometry_in_acquisition_order(dt, cooling, amplitude):
    event = analyze_window(signal(dt, cooling, amplitude), selection())
    assert event['status'] == 'calculated', event['reason']
    assert event['onset']['time_s'] == pytest.approx(40)
    assert event['onset']['temperature_c']['value'] == pytest.approx(46 if cooling else 54)
    assert event['onset']['pressure']['value'] == pytest.approx(240)
    assert event['phase_hypothesis'] is None
    assert event['manual_classification'] == 'pending_review'
    assert event['fit']['selection']['flank'] == 'manual'


def test_multiple_events_do_not_select_dominant_or_limit_to_two():
    t = np.arange(0., 400., .5)
    y = .01 * t
    windows = []
    for n, (start, amplitude) in enumerate([(40., 3.), (140., .02), (240., .5)], 1):
        y += amplitude * np.maximum(0, np.minimum((t-start)/20, (start+40-t)/20))
        windows.append(selection(event_id=f'e{n}', window_s=[start-35, start+45],
                                 baseline_s=[start-30, start-10], flank_s=[start+3, start+15]))
    item = SignalSet('run', pd.DataFrame({'time_s': t, 'heat_flow':y}), {'time_s':'s','heat_flow':'W/g'})
    results = analyze_selections([item], windows)
    assert len(results) == 3
    assert [e['onset']['time_s'] for e in results] == pytest.approx([40, 140, 240])
    assert all(e['onset']['temperature_c']['value'] is None for e in results)
    assert all(e['onset']['pressure']['value'] is None for e in results)
    assert all(e['units']['heat_flow'] == 'W/g' for e in results)


def test_manual_weak_signal_is_not_discarded_by_snr():
    item = signal(amplitude=.02, noise=.01)
    result = analyze_window(item, selection(manual_classification='accepted_event'),
                            AnalysisConfig(noise_multiplier=100))
    assert result['status'] == 'calculated', result['reason']
    assert result['manual_classification'] == 'accepted_event'
    assert result['automatic_classification'] == 'below_operational_criteria'


def test_persistence_is_local_and_does_not_join_disjoint_regions():
    y = np.zeros(100)
    y[50:90] = 30
    baseline = np.arange(100) < 20
    event = (np.arange(100) >= 20) & (np.arange(100) < 40)
    assert analyse_noise(y, baseline, event).consecutive_points == 0
    y[21:24] = 30
    y[26:29] = 30
    assert analyse_noise(y, baseline, event).consecutive_points == 3


def test_auto_flank_uses_first_departure():
    result = analyze_window(signal(), selection(baseline_s=[5,30], flank_s=None),
                            AnalysisConfig(min_persistence_s=2, tangent_half_width_s=3))
    assert result['status'] == 'calculated', result['reason']
    assert result['onset']['time_s'] == pytest.approx(40, abs=1.)
    assert result['fit']['flank_s'][1] < 60


def test_noise_failure_keeps_diagnostics_and_manual_review():
    item = signal(amplitude=0., noise=.001)
    result = analyze_window(item, selection(flank_s=None), AnalysisConfig(noise_multiplier=100))
    assert result['status'] == 'not_determined'
    assert result['onset'] is None
    assert result['diagnostics']['baseline_noise_sigma'] > 0
    assert result['manual_classification'] == 'pending_review'


def test_isothermal_and_oscillation_diagnostics():
    item = signal()
    item.frame['temperature_c'] = 20.
    assert analyze_window(item, selection())['onset']['temperature_c']['value'] == 20.
    t = np.arange(0., 200., .2)
    diagnostics = oscillation_indicator(t, np.sin(2*np.pi*t/10), AnalysisConfig())
    assert diagnostics['possible_oscillation']
    assert diagnostics['dominant_period_s'] == pytest.approx(10)
    irregular = t.copy()
    irregular[10:] += .5
    assert oscillation_indicator(irregular, np.sin(t), AnalysisConfig())['status'] == 'not_assessed'


@pytest.mark.parametrize('case', ['gap', 'duplicate', 'reversed', 'missing', 'omitted'])
def test_fit_fails_safely_on_bad_acquisition(case):
    item = signal()
    if case == 'gap':
        item.frame = item.frame.drop(index=range(60,80))
    elif case == 'duplicate':
        item.frame.loc[60,'time_s'] = item.frame.loc[59,'time_s']
    elif case == 'reversed':
        item.frame = item.frame.iloc[::-1]
    elif case == 'missing':
        item.frame.loc[60,'heat_flow'] = np.nan
    else:
        item.frame['source_discontinuity'] = False
        item.frame.loc[60,'source_discontinuity'] = True
    result = analyze_window(item, selection())
    assert result['status'] == 'not_determined'
    assert result['reason']
    assert result['onset'] is None


def test_interpolation_does_not_bridge_missing_values_or_gaps():
    f = pd.DataFrame({'time_s':[0., 1., 2., 20., 21.], 'temperature_c':[10., np.nan, 12., 20., 21.]})
    assert sample_signal(f,'temperature_c',.5,AnalysisConfig())['value'] is None
    assert sample_signal(f,'temperature_c',10,AnalysisConfig())['value'] is None
    assert sample_signal(f,'pressure',1,AnalysisConfig())['value'] is None
    assert sample_signal(f,'temperature_c',-1,AnalysisConfig())['value'] is None
    assert sample_signal(f,'temperature_c',20.5,AnalysisConfig())['value'] == 20.5


def test_manual_classes_and_failed_geometry_are_preserved():
    for review in ['noise','instrumental_oscillation','accepted_event','pending_review']:
        result = analyze_window(signal(), selection(manual_classification=review, flank_s=[10,20]))
        assert result['manual_classification'] == review
        assert result['status'] == 'not_determined'


def test_automatic_proposals_have_no_fixed_event_limit():
    t = np.arange(0., 500., .2)
    y = sum(a*np.exp(-((t-c)/3)**2) for a,c in [(1,60),(.1,160),(2,260),(.5,360)])
    item = SignalSet('run',pd.DataFrame({'time_s':t,'heat_flow':y}),{'heat_flow':'mW'})
    proposed = propose_windows(item, AnalysisConfig(candidate_separation_s=10))
    assert len(proposed['candidates']) >= 4
    assert all(c['manual_classification'] == 'pending_review' for c in proposed['candidates'])


def test_bundle_roundtrip_protects_inputs_and_results_and_plots_without_fitting(tmp_path, monkeypatch):
    import src.calorimetry_core as core
    from src.calorimetry_plot import render_bundle, compare_events
    item = signal()
    source = tmp_path/'source.raw'
    source.write_bytes(b'original instrument bytes')
    item.metadata['sources'] = [fingerprint(source)]
    events = analyze_selections([item],[selection()])
    saved = save_bundle(tmp_path/'first',[item],events,[selection()],AnalysisConfig())
    loaded, sets = load_bundle(saved)
    assert loaded['events'][0]['onset']['time_s'] == pytest.approx(40)
    pd.testing.assert_frame_equal(item.frame,sets[0].frame)
    with pytest.raises(FileExistsError):
        save_bundle(tmp_path/'first',[item])
    with pytest.raises(ValueError):
        save_bundle(tmp_path,[item],overwrite=True)
    with pytest.raises(ValueError):
        save_bundle(tmp_path/'first',sets,parent=saved,overwrite=True)
    source.unlink()  # Replay and revisions work from archived sources.
    revision = save_bundle(tmp_path/'revision', sets, events, [selection()], parent=saved)
    monkeypatch.setattr(core,'analyze_window',lambda *a,**k: pytest.fail('Plot recalculated onset'))
    fig = render_bundle(revision,tmp_path/'figure.png',event_id='one',line_style=':')
    assert fig.stat().st_size > 0
    with pytest.raises(FileExistsError):
        render_bundle(revision,fig)
    assert compare_events([(saved,'one'),(revision,'one')],tmp_path/'compare.svg').exists()
    assert json.loads(revision.read_text())['events'] == loaded['events']
    sets[0].units['heat_flow'] = 'W/g'
    other = save_bundle(tmp_path/'other',sets,events,[selection()])
    with pytest.raises(ValueError,match='identical HF units'):
        compare_events([(saved,'one'),(other,'one')],tmp_path/'bad.png')


def test_calisto_adapter_does_not_invent_sample_temperature_or_convert_hf(tmp_path, monkeypatch):
    from src.models import ProgramSummary
    import src.calorimetry_io as adapters
    source = tmp_path/'calisto.txt'
    source.write_text('Time(min);Furnace Temperature(°C);HeatFlow(W/g);Pressure(bar)\n0;30;1;200\n1;29;2;201\n')
    pdf = tmp_path/'program.pdf'
    pdf.write_bytes(b'placeholder; parser mocked')
    monkeypatch.setattr(adapters,'parse_program_pdf',lambda p: ProgramSummary(pdf,[]))
    item = load_signals('calisto',source,pdf)[0]
    assert item.frame.time_s.tolist() == [0,60]
    assert item.frame.heat_flow.tolist() == [1,2]
    assert item.frame.pressure.tolist() == [200,201]
    assert item.units['heat_flow'] == 'W/g'
    assert 'sample_temperature_c' not in item.frame
    assert item.metadata['temperature_source'] == 'furnace'


def test_trios_declared_units_and_explicit_time_reconstruction(tmp_path, monkeypatch):
    from src.trios_parser import TriosSegment
    import src.calorimetry_io as adapters
    source = tmp_path/'trios.xls'
    source.write_bytes(b'synthetic workbook mocked at parser boundary')
    segment = TriosSegment('cooling','cooling',pd.DataFrame({
        'time_min':[0.,0.,.01,.02], 'temperature_c':[30.,29.,28.,27.],
        'heat_flow_w_g':[0.,.1,.2,.3], 'source_row':[4,5,6,7]}),'min','°C','W/g',('min','°C','W/g'))
    monkeypatch.setattr(adapters,'read_trios_xls',lambda p: ({},[segment]))
    item = load_signals('trios',source)[0]
    assert item.metadata['time_diagnostics']['nonpositive_intervals'] == 1
    assert 'sample_temperature_c' not in item.frame
    uniform = load_signals('trios',source,trios_time='uniform')[0]
    assert np.diff(uniform.frame.time_s).min() > 0
    assert uniform.frame.recorded_time_s.tolist() == [0.,0.,.6,1.2]
    segment.declared_units = ('min','°C','')
    with pytest.raises(ValueError,match='Heat-flow unit'):
        load_signals('trios',source)


def test_stage_validation_and_unique_ids():
    item = signal()
    item.stages = [{'id':'cool','cycle_id':2,'direction':'cooling','start_s':0.,'end_s':100.}]
    event = analyze_selections([item],[selection(stage_id='cool')])[0]
    assert event['cycle_id'] == 2
    bad = analyze_selections([item],[selection(stage_id='missing')])[0]
    assert bad['status'] == 'not_determined' and bad['onset'] is None
    with pytest.raises(ValueError,match='unique'):
        analyze_selections([item],[selection(),selection()])


def test_row_range_can_select_a_monotonic_record_without_sorting():
    item = signal()
    original = item.frame.copy()
    item.frame = pd.concat([original, original], ignore_index=True)
    assert analyze_window(item, selection())['status'] == 'not_determined'
    event = analyze_window(item, selection(row_range=[len(original),len(item.frame)-1]))
    assert event['status'] == 'calculated', event['reason']
    assert event['onset']['time_s'] == pytest.approx(40)


def test_common_cli_manual_revision_and_replay(tmp_path):
    from src.calorimetry_cli import main
    item = signal()
    saved = save_bundle(tmp_path/'loaded',[item])
    windows = tmp_path/'windows.json'
    windows.write_text(json.dumps([selection(stage_id='manual')]))
    stages = tmp_path/'stages.json'
    stages.write_text(json.dumps([{'dataset_id':'run','id':'manual','start_s':0.,'end_s':100.,
                                   'direction':'cooling','cycle_id':'cycle-1'}]))
    assert main(['analyze','--session',str(saved),'--windows',str(windows),'--stages',str(stages),
                 '--outdir',str(tmp_path/'analyzed')]) == 0
    p, _ = load_bundle(tmp_path/'analyzed')
    assert p['events'][0]['cycle_id'] == 'cycle-1'
    assert main(['plot','--session',str(tmp_path/'analyzed'),'--event','one',
                 '--out',str(tmp_path/'event.pdf'),'--line-style=:']) == 0
    assert (tmp_path/'event.pdf').stat().st_size > 0


def test_renderer_never_overwrites_source_pdf(tmp_path):
    from src.calorimetry_plot import render_bundle
    item = signal()
    pdf = tmp_path/'instrument.pdf'
    pdf.write_bytes(b'original source')
    item.metadata['sources'] = [fingerprint(pdf)]
    saved = save_bundle(tmp_path/'bundle',[item])
    with pytest.raises(ValueError,match='source data'):
        render_bundle(saved,pdf,overwrite=True)
    assert pdf.read_bytes() == b'original source'


def test_calisto_without_any_temperature_keeps_hf_and_time(tmp_path, monkeypatch):
    from src.models import ProgramSummary
    import src.calorimetry_io as adapters
    source = tmp_path/'calisto.txt'
    source.write_text('Time(s);HeatFlow(mW)\n0;1\n1;2\n')
    pdf = tmp_path/'program.pdf'
    pdf.write_bytes(b'placeholder; parser mocked')
    monkeypatch.setattr(adapters,'parse_program_pdf',lambda p: ProgramSummary(pdf,[]))
    item = load_signals('calisto',source,pdf)[0]
    assert item.frame.heat_flow.tolist() == [1,2]
    assert 'temperature_c' not in item.frame
    assert 'sample_temperature_c' not in item.frame
    assert item.metadata['temperature_source'] == 'unavailable'


def test_manual_flank_respects_explicit_sign():
    event = analyze_window(signal(amplitude=-2),selection(polarity='positive'))
    assert event['status'] == 'not_determined'
    assert 'selected direction' in event['reason']
