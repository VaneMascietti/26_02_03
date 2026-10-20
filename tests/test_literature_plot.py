from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from util.literature_plot import load_literature, plot_literature


def test_filters_preserve_provenance_and_original_transitions(tmp_path):
    source = tmp_path/'literature.csv'
    pd.DataFrame({'ID':['A_x0.9','A_x0.8','B_x0.9'], 'transition_type':['TLL','SL','LLV'],
                  'T_C':[20,10,15], 'P_bar':[300,200,50], 'xCO2':[.9,.8,.9],
                  'source':['literature']*3, 'DOI':['a','b','c']}).to_csv(source,index=False)
    selected = load_literature(source,ids=['A_x0.9','A_x0.8'],transitions=['TLL'],compositions=[.9])
    assert selected.ID.tolist() == ['A_x0.9']
    assert selected.DOI.tolist() == ['a']
    assert selected.transition_type.tolist() == ['TLL']
    with pytest.raises(ValueError,match='Unknown literature IDs'):
        load_literature(source,ids=['missing'])
    with pytest.raises(ValueError,match='No literature points'):
        load_literature(source,compositions=[.5])
    fig,ax = plt.subplots()
    plot_literature(ax,selected)
    assert len(ax.lines) == 0
    assert len(ax.collections) == 1
    assert 'xCO₂=0.9000' in ax.get_legend_handles_labels()[1][0]
    plt.close(fig)


def test_local_csv_remains_compatible_with_existing_plotter():
    from plot_phase_transition_points import load_points
    path = Path('data/phase_transitions/literature_points.csv')
    pd.testing.assert_frame_equal(load_points(path),load_literature(path))


def test_nc15_workbook_keeps_only_explicit_llv_and_phase_compositions():
    path = Path('data/literature/CO2_nC15_literature_data_extracted.xlsx')
    points = load_literature(path, transitions=['LLV'], system='CO2+n-pentadecane')
    assert points.groupby('ID').size().to_dict() == {
        'Secuianu2010':15, 'Hottovy1981':7, 'vanDerSteen1989':8}
    assert set(points.Sistema) == {'CO2+n-pentadecane'}
    assert 'xCO2' not in points  # Phase compositions must not become overall composition.
    assert points.source_excel_row.ge(2).all()
    assert points.source_sha256.str.len().eq(64).all()
    assert points.DOI.notna().all()
    assert set(points.transition_original) == {'LLV','L1-L2-V'}
    with pytest.raises(ValueError,match='literature-system'):
        load_literature(path)
    with pytest.raises(ValueError,match='xCO2 column'):
        load_literature(path,system='CO2+n-pentadecane',compositions=[.9])
    endpoints = load_literature(path,system='CO2+n-pentadecane',transitions=['UCEP','Q point'])
    assert len(endpoints) == 3
