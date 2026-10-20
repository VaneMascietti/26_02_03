"""Display-only full-range scaling and legend placement for scientific plots."""
import numpy as np


def scale_heat_flow(axis, values, *, scale='auto', limits=None):
    """Keep all finite observations visible; compress extreme dynamic ranges only.

    Quantiles select the display transform, never select or discard observations.
    Explicit limits specify a minimum range and are expanded to include the data.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError('Heat-flow signal contains no finite values.')
    if scale not in ('auto','linear','symlog'):
        raise ValueError('Unknown HF display scale.')
    low, high = float(values.min()), float(values.max())
    core_low, core_high = np.quantile(values,[.005,.995])
    core_span = float(core_high-core_low)
    use_log = scale == 'symlog' or (scale == 'auto' and core_span > 0 and high-low > 3*core_span)
    if use_log:
        core_extent = max(abs(float(core_low)),abs(float(core_high)),np.finfo(float).eps)
        threshold = float(10**np.ceil(np.log10(core_extent)))
        axis.set_yscale('symlog',linthresh=threshold,base=10)
        axis.set_ylabel('Heat flow (mW)\nSymmetric log scale')
    else:
        axis.set_yscale('linear')
        axis.set_ylabel('Heat flow (mW)')
    if limits is not None:
        if not np.isfinite(limits).all() or limits[0] >= limits[1]:
            raise ValueError('--hf-ylim requires finite increasing limits.')
        low, high = min(low,float(limits[0])),max(high,float(limits[1]))
    # Pad in display coordinates, retaining negative and zero observations.
    transform = axis.yaxis.get_transform()
    ends = transform.transform(np.array([low,high]))
    pad = float(ends[1]-ends[0]) * .08
    if pad <= 0:
        pad = max(abs(float(ends[0]))*.05,1.)
    bounds = transform.inverted().transform([ends[0]-pad,ends[1]+pad])
    axis.set_ylim(*bounds)
    return 'symlog' if use_log else 'linear'


def _display_points(axes):
    samples=[]
    for axis in axes:
        for line in axis.lines:
            vertices=line.get_path().vertices
            # Keep all vertices of short paths and sample long acquired records.
            if len(vertices)>6000:
                vertices=vertices[np.linspace(0,len(vertices)-1,6000,dtype=int)]
            if len(vertices):
                samples.append(line.get_transform().transform(vertices))
        for collection in axis.collections:
            offsets=collection.get_offsets()
            if len(offsets):
                samples.append(collection.get_offset_transform().transform(offsets))
    return np.concatenate(samples) if samples else np.empty((0,2))


def place_inside_legend(axis, *, handles=None, labels=None, other_axes=(), fontsize=14, title=None, locations=None):
    """Choose an inside location considering curves on twin axes and annotations.

    If no location is free, reserve headroom and reconsider. Layout affects only
    axis limits and legend position, never the signal or calculated results.
    """
    if handles is None:
        handles,labels=axis.get_legend_handles_labels()
    if not handles:
        return None
    axes=[axis,*other_axes]
    fig=axis.figure
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    best=None
    for attempt in range(2):
        samples=_display_points(axes)
        texts=[text.get_window_extent(renderer) for ax in axes for text in ax.texts if text.get_visible()]
        candidates=[]
        columns = [1,2,3] if len(handles)>3 else [1]
        for count in columns:
            for location in (locations or [1,2,3,4,6,7,8,9,10]):
                legend=axis.legend(handles,labels,loc=location,fontsize=fontsize,title=title,
                                   ncol=count,frameon=True,facecolor='white',edgecolor='none',framealpha=.95)
                legend.set_in_layout(False)
                box=legend.get_window_extent(renderer).padded(5)
                area=axis.get_window_extent(renderer)
                if box.width>area.width or box.height>area.height:
                    continue
                inside=(samples[:,0]>=box.x0)&(samples[:,0]<=box.x1)&(samples[:,1]>=box.y0)&(samples[:,1]<=box.y1)
                overlap=sum(box.overlaps(text) for text in texts)
                candidates.append((int(inside.sum())+10000*overlap,count,location,box.height))
        if not candidates:
            raise ValueError('Legend cannot fit inside the panel at this font size; reduce --legend-size.')
        best=min(candidates,key=lambda entry:(entry[0],entry[1]))
        if best[0]==0 or attempt:
            break
        # Make a clear band above all data when every tested position overlaps.
        fraction=min(.7,best[3]/axis.get_window_extent(renderer).height+.04)
        for ax in axes:
            low,high=ax.get_ylim()
            transform=ax.yaxis.get_transform()
            ends=transform.transform(np.array([low,high]))
            data_high=float(transform.transform(ax.dataLim.ymax))
            upper=max(ends[1],(data_high-fraction*ends[0])/(1-fraction))
            ax.set_ylim(low,float(transform.inverted().transform(upper)))
        fig.canvas.draw()
        renderer=fig.canvas.get_renderer()
    legend=axis.legend(handles,labels,loc=best[2],fontsize=fontsize,title=title,ncol=best[1],
                       frameon=True,facecolor='white',edgecolor='none',framealpha=.95)
    legend.set_in_layout(False)
    return legend
