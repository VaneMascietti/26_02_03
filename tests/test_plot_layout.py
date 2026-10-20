import matplotlib.pyplot as plt
import numpy as np

from util.plot_layout import scale_heat_flow,place_inside_legend


def test_auto_scale_keeps_all_points_and_linear_mode_remains_available():
    fig,ax=plt.subplots()
    values=np.r_[np.linspace(-6,6,1000),78.52]
    ax.plot(np.arange(len(values)),values)
    assert scale_heat_flow(ax,values)=='symlog'
    assert ax.get_ylim()[0] < values.min() and ax.get_ylim()[1] > values.max()
    assert scale_heat_flow(ax,values,scale='linear')=='linear'
    assert ax.get_yscale()=='linear'
    assert ax.get_ylim()[0] < values.min() and ax.get_ylim()[1] > values.max()
    plt.close(fig)


def test_legend_placement_considers_other_axis_and_stays_inside():
    fig,ax=plt.subplots()
    twin=ax.twinx()
    x=np.linspace(0,1,100)
    ax.plot(x,np.zeros_like(x),label='Temperature')
    twin.plot(x,np.ones_like(x),label='Pressure')
    ax.set_ylim(-1,2)
    twin.set_ylim(-2,1.2)
    h,l=ax.get_legend_handles_labels();h2,l2=twin.get_legend_handles_labels()
    legend=place_inside_legend(ax,handles=h+h2,labels=l+l2,other_axes=[twin])
    fig.canvas.draw()
    box=legend.get_window_extent()
    area=ax.get_window_extent()
    assert area.contains(box.x0,box.y0) and area.contains(box.x1,box.y1)
    for axis in [ax,twin]:
        points=axis.lines[0].get_transform().transform(axis.lines[0].get_path().vertices)
        assert not any(box.contains(*point) for point in points)
    plt.close(fig)
