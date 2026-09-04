"""Plotting and computation helpers for analytic synaptic-field grids.

Used by downstream_plotting.ipynb (and compute_fields.py) in the analytic_results/
folder. Reuses the analytic get_synaptic_fields backend and the hex/degrees styling
developed for the T3 validation figures.
"""
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection

from flywire_tools.connectome import lattice_to_retinal, load_SynapticFields

# --- geometry / style (matches the T3 receptive-field figures) ---
DEG_PER_COLUMN = 5.0                          # assumed inter-ommatidial angle (degrees)
_NN_LATTICE = float(np.sqrt(1.5))             # adjacent-column distance in lattice units
DEG_PER_UNIT = DEG_PER_COLUMN / _NN_LATTICE   # scale lattice coords -> degrees
HEX_SCALE = 1.05                              # hexagon size relative to exact tiling
CMAP = "gray_r"                               # white (low) -> black (high)

# retinal column table (both hemispheres + many columnar types)
COLUMN_INFO_CSV = "visual_column_info.csv"

# retinal input channels used as stopping points (same as the T3 analysis)
STARTING_POINTS = ["L1", "L2", "L3", "L4", "L5", "R7", "R8"]


def lc_sort_key(name):
    """Sort key that orders types by the integer embedded in their name."""
    m = re.search(r"(\d+)", str(name))
    return int(m.group(1)) if m else float("inf")


def neuron_types(visual_neuron_types, family):
    """Sorted unique 'type' names for a given family (e.g. 'LC' or 'LPLC')."""
    types = visual_neuron_types[visual_neuron_types["family"] == family]["type"].dropna().unique()
    return sorted(types, key=lc_sort_key)


def aggregate_image(field):
    """Collapse a SynapticField (targets, W, H) to a single (W, H) image."""
    arr = np.asarray(field, dtype=float)
    return arr.reshape(arr.shape[0], -1).sum(0).reshape(arr.shape[-2:])


def field_degrees(field):
    """Return (xs_deg, ys_deg, values) for a SynapticField's aggregate image."""
    img = aggregate_image(field)
    xs, ys = lattice_to_retinal(np.asarray(field.ps), np.asarray(field.qs))
    return xs.ravel() * DEG_PER_UNIT, ys.ravel() * DEG_PER_UNIT, img.ravel()


def make_lognorm(value_arrays):
    """Shared LogNorm over the positive values pooled across several images.

    Uses robust percentile bounds (2nd-99.8th) so a few large values don't compress the
    rest of the range into darkness.
    """
    pooled = []
    for v in value_arrays:
        v = np.asarray(v).ravel()
        v = v[v > 0]
        if v.size:
            pooled.append(v)
    if not pooled:
        return None
    allv = np.concatenate(pooled)
    vmin = float(np.percentile(allv, 2))
    vmax = float(np.percentile(allv, 99.8))
    if vmin <= 0:
        vmin = float(allv.min())
    if vmin >= vmax:
        vmin = vmax * 1e-3 if vmax > 0 else 1.0
    return LogNorm(vmin=vmin, vmax=vmax)


def draw_hex(ax, field, norm, cmap=CMAP, axis_lim=None):
    """Draw one SynapticField as tiled hexagons (in degrees) on the given axis."""
    xs, ys, vals = field_degrees(field)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if axis_lim is not None:
        ax.set_xlim(-axis_lim, axis_lim)
        ax.set_ylim(-axis_lim, axis_lim)
    mask = vals > 0
    if not mask.any():
        return None
    radius = HEX_SCALE * DEG_PER_COLUMN / np.sqrt(3)
    hexes = [RegularPolygon((x, y), numVertices=6, radius=radius, orientation=np.pi / 6)
             for x, y in zip(xs[mask], ys[mask])]
    pc = PatchCollection(hexes, cmap=cmap, norm=norm, edgecolors="face")
    pc.set_array(vals[mask])
    ax.add_collection(pc)
    return pc


def draw_field_interp(ax, field, norm, cmap=CMAP, axis_lim=None, clip_zero=True, grid_res=80):
    """Draw a SynapticField as an interpolated pcolormesh.

    Scatter-interpolates the hexagonal lattice onto a regular Cartesian grid and
    renders it with ``pcolormesh(rasterized=True)``.  This gives smooth, readable
    images at small panel sizes and produces a single bitmap element per panel when
    saved as SVG, keeping file sizes manageable.

    Parameters
    ----------
    ax : matplotlib Axes
    field : SynapticField
    norm : matplotlib Normalize or LogNorm
    cmap : str
    axis_lim : float or None
        Symmetric degree limit for both axes.  ``None`` auto-fits.
    clip_zero : bool
        If True (default) clip values to ≥0 and mask zeros (for LogNorm / magma).
        Set False for signed/diverging data (SVD reconstructions).
    grid_res : int
        Number of grid points along each axis for the interpolation.
    """
    from scipy.interpolate import griddata

    xs, ys, vals = field_degrees(field)

    # Compute valid-pixel mask first so axis limits use only real data
    pos_mask = vals > 0 if clip_zero else np.isfinite(vals)

    if axis_lim is None:
        if pos_mask.any():
            xr = float(np.abs(xs[pos_mask]).max()) * 1.05
            yr = float(np.abs(ys[pos_mask]).max()) * 1.05
        else:
            xr = float(np.abs(xs).max()) * 1.05
            yr = float(np.abs(ys).max()) * 1.05
    else:
        xr = yr = float(axis_lim)

    xi = np.linspace(-xr, xr, grid_res)
    yi = np.linspace(-yr, yr, grid_res)
    Xi, Yi = np.meshgrid(xi, yi)

    if pos_mask.sum() < 3:
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-xr, xr)
        ax.set_ylim(-yr, yr)
        return

    Zi = griddata(
        (xs[pos_mask], ys[pos_mask]),
        vals[pos_mask],
        (Xi, Yi),
        method="linear",
        fill_value=np.nan,
    )
    if clip_zero:
        Zi = np.clip(Zi, 0.0, None)
        # Mask zeros and NaN fill-areas so the axes background shows through
        Zi_disp = np.ma.masked_where(~np.isfinite(Zi) | (Zi <= 0), Zi)
    else:
        # Mask only NaN fill-areas; actual zero values are meaningful
        Zi_disp = np.ma.masked_where(~np.isfinite(Zi), Zi)

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-xr, xr)
    ax.set_ylim(-yr, yr)
    ax.pcolormesh(xi, yi, Zi_disp, norm=norm, cmap=cmap,
                  shading="auto", rasterized=True)


def _field_to_base64(img, ref_fld, cmap, axis_lim, hover_px, clip_zero=True,
                     norm_mode="auto", mask=None):
    """Render a 2-D field image as a transparent base64 PNG without touching pyplot.

    Parameters
    ----------
    clip_zero : bool
        True (default) → LogNorm on positive values only (for real fields).
        False → symmetric Normalize centred at 0 (for signed SVD reconstructions).
    norm_mode : str
        ``'auto'``  — use clip_zero to decide between LogNorm and symmetric Normalize.
        ``'minmax'`` — Normalize(vmin, vmax) from valid pixels only (see *mask*).
    mask : ndarray of bool, shape (H, W), optional
        When norm_mode='minmax', restrict vmin/vmax computation to pixels where
        mask is True (i.e. exclude background zeros).
    """
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import LogNorm, Normalize
    import io, base64

    if norm_mode == "minmax":
        vals = img[mask] if (mask is not None) else img.ravel()
        vmin = float(vals.min()) if vals.size > 0 else 0.0
        vmax = float(vals.max()) if vals.size > 0 else 1.0
        if vmin == vmax:
            vmax = vmin + 1.0
        nrm = Normalize(vmin=vmin, vmax=vmax)
    elif clip_zero:
        vals = img[img > 0]
        if vals.size > 0:
            nrm = LogNorm(vmin=float(np.percentile(vals, 2)),
                          vmax=float(np.percentile(vals, 99.8)))
        else:
            nrm = Normalize(0, 1)
    else:
        vmax = float(np.abs(img).max()) or 1.0
        nrm = Normalize(vmin=-vmax, vmax=vmax)

    dpi = 72
    sz = hover_px / dpi
    fig_t = Figure(figsize=(sz, sz), dpi=dpi)
    FigureCanvasAgg(fig_t)
    ax_t = fig_t.add_axes([0, 0, 1, 1])  # fill figure, no margins
    fig_t.patch.set_alpha(0.0)
    ax_t.set_facecolor("none")

    # When a validity mask is provided, blank out background pixels so griddata
    # does not interpolate through them (zeros would contaminate the result).
    img_render = img.copy()
    if mask is not None:
        img_render[~mask] = np.nan
    fld = type(ref_fld)(img_render[None], ps=ref_fld.ps, qs=ref_fld.qs)
    draw_field_interp(ax_t, fld, nrm, cmap=cmap, axis_lim=axis_lim, clip_zero=clip_zero)
    # Zoom to actual content extent (ignores empty retinal lattice border)
    if axis_lim is None:
        xs_f, ys_f, vals_f = field_degrees(fld)
        pmask_f = (vals_f > 0) if clip_zero else np.isfinite(vals_f)
        if pmask_f.sum() > 2:
            content_half = max(float(np.abs(xs_f[pmask_f]).max()),
                               float(np.abs(ys_f[pmask_f]).max())) * 1.25
        else:
            content_half = max(abs(ax_t.get_xlim()[0]), abs(ax_t.get_xlim()[1]),
                               abs(ax_t.get_ylim()[0]), abs(ax_t.get_ylim()[1]))
    else:
        content_half = float(axis_lim)
    ax_t.set_aspect("auto")
    ax_t.set_xlim(-content_half, content_half)
    ax_t.set_ylim(-content_half, content_half)
    for spine in ax_t.spines.values():
        spine.set_visible(False)

    buf = io.BytesIO()
    fig_t.savefig(buf, format="png", pad_inches=0, transparent=True)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def plot_svd_eigenfields(
    neurons, scores, x, keep, all_fields,
    channel="L5",
    pc_x=0,
    pc_y=1,
    labels=None,
    cmap="magma",
    axis_lim=None,
    n_eigenfields=4,
    zoom_thr=0.08,
    figsize=(16, 12),
):
    """Scatter of SVD scores with eigenfields and peripheral reconstructed fields.

    Layout
    ------
    Top strip  : first *n_eigenfields* eigenfields (diverging colormap, ±symmetric).
    Centre     : scatter of PC *pc_x* vs PC *pc_y* with neuron labels coloured by
                 *labels* (if given).  An automatic zoom inset is added when two or
                 more labels fall within *zoom_thr* × axis-range of each other.
    Periphery  : 8 reconstructed fields at the corners / edges of the score space
                 (left/centre/right × top/middle/bottom, excluding centre).  Each
                 field is the SVD prediction using only PC *pc_x* and PC *pc_y*
                 (all other components set to zero).

    Parameters
    ----------
    neurons : list[str]
    scores  : (M, k) ndarray  SVD scores from :func:`svd_reduce`.
    x       : (M, n_keep) ndarray  z-scored RF matrix (used to recover Vt_k).
    keep    : bool mask (W*H,)
    all_fields : dict  neuron -> SynapticFields
    labels  : array-like or None  cluster labels for scatter colouring.
    zoom_thr : float  fraction of axis range; pairs closer than this get a zoom inset.

    Returns
    -------
    fig, ax_scatter, ax_eigenfields  (list of n_eigenfields axes)
    """
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import LogNorm, Normalize
    from matplotlib.patches import Rectangle

    k = scores.shape[1]
    n_neurons = len(neurons)

    # ------------------------------------------------------------------
    # 1. Recover Vt_k and reference geometry
    # ------------------------------------------------------------------
    _, _, Vt = np.linalg.svd(np.asarray(x, float), full_matrices=False)
    Vt_k = Vt[:k]

    ref_fld = all_fields[neurons[0]][channel]
    W, H = ref_fld.shape[-2], ref_fld.shape[-1]
    ps_ref, qs_ref = ref_fld.ps, ref_fld.qs

    # ------------------------------------------------------------------
    # 2. Figure layout
    #    Row 0: eigenfields (spans all cols)
    #    Row 1: TL  gap  TC  gap  TR
    #    Row 2: ML  [scatter spans 3 cols]  MR
    #    Row 3: BL  gap  BC  gap  BR
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(
        4, 5,
        figure=fig,
        height_ratios=[0.9, 0.9, 4.5, 0.9],
        width_ratios=[1.1, 0.12, 2.8, 0.12, 1.1],
        hspace=0.18, wspace=0.12,
        left=0.04, right=0.97, top=0.94, bottom=0.04,
    )
    # Eigenfield strip
    ef_gs = gridspec.GridSpecFromSubplotSpec(
        1, max(n_eigenfields, 1), subplot_spec=gs[0, :], wspace=0.12)
    ax_ef = [fig.add_subplot(ef_gs[0, i]) for i in range(n_eigenfields)]

    # Peripheral panels
    periph_cells = {
        "TL": (1, 0), "TC": (1, 2), "TR": (1, 4),
        "ML": (2, 0),               "MR": (2, 4),
        "BL": (3, 0), "BC": (3, 2), "BR": (3, 4),
    }
    ax_p = {name: fig.add_subplot(gs[r, c])
            for name, (r, c) in periph_cells.items()}

    # Main scatter
    ax_sc = fig.add_subplot(gs[2, 1:4])

    # ------------------------------------------------------------------
    # 3. Eigenfields (diverging: positive = region loaded by that PC)
    # ------------------------------------------------------------------
    n_ef = min(n_eigenfields, k)
    for i in range(n_ef):
        ef_flat = np.zeros(W * H)
        ef_flat[keep] = Vt_k[i]
        ef_2d = ef_flat.reshape(W, H)
        ef_fld = type(ref_fld)(ef_2d[None].astype(np.float32), ps=ps_ref, qs=qs_ref)
        vmax = float(np.abs(Vt_k[i]).max())
        ax_ef[i].set_title(f"EF {i + 1}", fontsize=9)
        draw_field_interp(ax_ef[i], ef_fld,
                          Normalize(vmin=-vmax, vmax=vmax),
                          cmap="RdBu_r", axis_lim=axis_lim)
    for i in range(n_ef, n_eigenfields):
        ax_ef[i].set_visible(False)

    # ------------------------------------------------------------------
    # 4. Scatter
    # ------------------------------------------------------------------
    sc_x = scores[:, pc_x]
    sc_y = scores[:, pc_y]

    if labels is not None:
        import colorsys
        uniq = sorted(set(labels))
        _PHI = 0.618033988749895
        lbl2col = {}
        for i, lbl in enumerate(uniq):
            h = (i * _PHI) % 1.0
            r, g, b = colorsys.hsv_to_rgb(h, 0.70, 1.0)
            lbl2col[lbl] = f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
        colors = [lbl2col[l] for l in labels]
    else:
        colors = ["#4477aa"] * n_neurons

    ax_sc.scatter(sc_x, sc_y, c=colors, s=45, zorder=3, edgecolors="none")
    for i, name in enumerate(neurons):
        ax_sc.text(sc_x[i], sc_y[i], name, fontsize=7.5,
                   ha="left", va="bottom", clip_on=True)
    ax_sc.set_xlabel(f"PC {pc_x + 1}", fontsize=11)
    ax_sc.set_ylabel(f"PC {pc_y + 1}", fontsize=11)
    ax_sc.axhline(0, color="#cccccc", lw=0.8, ls="--", zorder=1)
    ax_sc.axvline(0, color="#cccccc", lw=0.8, ls="--", zorder=1)

    xr = sc_x.max() - sc_x.min()
    yr = sc_y.max() - sc_y.min()
    pad = 0.09
    ax_sc.set_xlim(sc_x.min() - xr * pad, sc_x.max() + xr * pad)
    ax_sc.set_ylim(sc_y.min() - yr * pad, sc_y.max() + yr * pad)

    # ------------------------------------------------------------------
    # 5. Zoom inset for crowded labels
    # ------------------------------------------------------------------
    crowded_idx = set()
    for i in range(n_neurons):
        for j in range(i + 1, n_neurons):
            if (abs(sc_x[i] - sc_x[j]) < zoom_thr * xr and
                    abs(sc_y[i] - sc_y[j]) < zoom_thr * yr):
                crowded_idx.update([i, j])

    if crowded_idx:
        cx = float(np.mean([sc_x[i] for i in crowded_idx]))
        cy = float(np.mean([sc_y[i] for i in crowded_idx]))
        r = zoom_thr * max(xr, yr) * 2.0
        # Place inset in the opposite quadrant
        ix0 = 0.03 if cx > (sc_x.min() + xr * 0.5) else 0.55
        iy0 = 0.03 if cy > (sc_y.min() + yr * 0.5) else 0.55
        ax_ins = ax_sc.inset_axes([ix0, iy0, 0.42, 0.42])
        ax_ins.scatter(sc_x, sc_y, c=colors, s=45, zorder=3, edgecolors="none")
        for i, name in enumerate(neurons):
            ax_ins.text(sc_x[i], sc_y[i], name, fontsize=7, ha="left", va="bottom")
        ax_ins.set_xlim(cx - r, cx + r)
        ax_ins.set_ylim(cy - r, cy + r)
        ax_ins.set_xticks([])
        ax_ins.set_yticks([])
        for sp in ax_ins.spines.values():
            sp.set_color("#888888")
        # Dashed rectangle on main scatter indicating the zoom region
        ax_sc.add_patch(Rectangle(
            (cx - r, cy - r), 2 * r, 2 * r,
            lw=1, edgecolor="#888888", facecolor="none", ls="--", zorder=4))

    # ------------------------------------------------------------------
    # 6. Peripheral reconstructed fields
    #    Score vector: only pc_x and pc_y set; remaining PCs = 0.
    # ------------------------------------------------------------------
    x_min, x_max = float(sc_x.min()), float(sc_x.max())
    y_min, y_max = float(sc_y.min()), float(sc_y.max())
    x_mid = (x_min + x_max) / 2
    y_mid = (y_min + y_max) / 2

    periph_scores = {
        "TL": (x_min, y_max), "TC": (x_mid, y_max), "TR": (x_max, y_max),
        "ML": (x_min, y_mid),                         "MR": (x_max, y_mid),
        "BL": (x_min, y_min), "BC": (x_mid, y_min),  "BR": (x_max, y_min),
    }

    for name, (px, py) in periph_scores.items():
        sv = np.zeros(k)
        sv[pc_x] = px
        sv[pc_y] = py
        ff = np.zeros(W * H, dtype=np.float32)
        ff[keep] = np.clip(sv @ Vt_k, 0.0, None)
        pfld = type(ref_fld)(ff.reshape(W, H)[None], ps=ps_ref, qs=qs_ref)

        vals = ff[ff > 0]
        if vals.size > 0:
            pnorm = LogNorm(vmin=float(np.percentile(vals, 2)),
                            vmax=float(np.percentile(vals, 99.8)))
        else:
            pnorm = Normalize(0, 1)

        draw_field_interp(ax_p[name], pfld, pnorm, cmap=cmap, axis_lim=axis_lim)
        ax_p[name].set_title(f"({px:.2f}, {py:.2f})", fontsize=7)

    return fig, ax_sc, ax_ef


def plot_svd_scatter_plotly(
    neurons, scores, all_fields,
    channel="L5",
    pc_x=0,
    pc_y=1,
    labels=None,
    cmap="magma",
    axis_lim=None,
    hover_px=160,
    figsize_px=(950, 720),
):
    """Interactive Plotly scatter of SVD scores with hover synaptic-field thumbnails.

    Hovering over any neuron name reveals its scaled channel field as a small
    transparent PNG thumbnail embedded directly in the tooltip — no server needed.

    Parameters
    ----------
    neurons, scores, all_fields : as in other functions.
    channel : str
    pc_x, pc_y : int  0-indexed PC axes.
    labels : array-like or None  cluster labels for colouring and legend.
    cmap : str  colourmap for the thumbnail images.
    axis_lim : float or None  passed to :func:`draw_field_interp`.
    hover_px : int  thumbnail size in pixels.
    figsize_px : (width, height)  Plotly figure size in pixels.

    Returns
    -------
    plotly.graph_objects.Figure
        Call ``.show()`` in a Jupyter cell to display interactively.
    """
    import plotly.graph_objects as go
    import matplotlib.colors as mc

    ref_fld = all_fields[neurons[0]][channel]
    n = len(neurons)

    # ------------------------------------------------------------------
    # Render one thumbnail per neuron (uses FigureCanvasAgg, no pyplot)
    # ------------------------------------------------------------------
    hover_imgs = [
        _field_to_base64(
            or_over_targets(all_fields[neu][channel]),
            all_fields[neu][channel],
            cmap, axis_lim, hover_px,
        )
        for neu in neurons
    ]

    # ------------------------------------------------------------------
    # Colours
    # ------------------------------------------------------------------
    if labels is not None:
        import colorsys
        uniq = sorted(set(labels))
        _PHI = 0.618033988749895
        lbl2col = {}
        for i, lbl in enumerate(uniq):
            h = (i * _PHI) % 1.0
            r, g, b = colorsys.hsv_to_rgb(h, 0.70, 1.0)
            lbl2col[lbl] = f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
    else:
        lbl2col = None

    htmpl = (
        "<b>%{text}</b><br>"
        f'<img src="data:image/png;base64,%{{customdata}}" '
        f'width="{hover_px}" height="{hover_px}">'
        "<extra></extra>"
    )

    # One trace per cluster (gives a legend), or a single trace
    if labels is not None and lbl2col:
        traces = []
        for lbl in sorted(set(labels)):
            idx = [i for i, l in enumerate(labels) if l == lbl]
            traces.append(go.Scatter(
                x=scores[idx, pc_x].tolist(),
                y=scores[idx, pc_y].tolist(),
                mode="markers+text",
                name=str(lbl),
                text=[neurons[i] for i in idx],
                textposition="top center",
                textfont=dict(size=9),
                marker=dict(size=9, color=lbl2col[lbl]),
                customdata=[hover_imgs[i] for i in idx],
                hovertemplate=htmpl,
            ))
    else:
        traces = [go.Scatter(
            x=scores[:, pc_x].tolist(),
            y=scores[:, pc_y].tolist(),
            mode="markers+text",
            name="neurons",
            text=list(neurons),
            textposition="top center",
            textfont=dict(size=9),
            marker=dict(size=9, color="#4477aa"),
            customdata=hover_imgs,
            hovertemplate=htmpl,
        )]

    fig = go.Figure(data=traces)
    fig.update_layout(
        xaxis_title=f"PC {pc_x + 1}",
        yaxis_title=f"PC {pc_y + 1}",
        title=f"SVD score space  (PC{pc_x + 1} vs PC{pc_y + 1})",
        hovermode="closest",
        width=figsize_px[0],
        height=figsize_px[1],
        legend=dict(title="cluster") if labels is not None else {},
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#cccccc", line_width=1)
    fig.add_vline(x=0, line_dash="dash", line_color="#cccccc", line_width=1)
    return fig


def write_svd_scatter_html(
    neurons, scores, all_fields,
    channel="L5",
    pc_x=0,
    pc_y=1,
    labels=None,       # list; None entries → gray "unlabeled" trace
    cmap="magma",
    axis_lim=None,
    hover_px=160,
    figsize_px=(1100, 1000),
    output_path="svd_scatter.html",
    x=None,            # z-scored RF matrix; enables peripheral field images
    keep=None,         # bool mask (W*H,)
    img_frac=0.30,     # peripheral image size as fraction of max score range
    img_px=200,        # pixel size for peripheral images
    xaxis_reversed=False,  # invert the x axis
    yaxis_reversed=False,  # invert the y axis
    flip_pcs=None,         # list of PC indices whose sign to negate (arbitrary in SVD)
    field_mask=None,       # bool mask (H, W) for valid pixels; used to compute minmax norm
):
    """Standalone HTML scatter of SVD scores with hover thumbnails and optional
    peripheral SVD-reconstructed field images at the 8 extremes of score space.

    * ``labels`` may contain ``None`` entries — those neurons are shown gray.
    * Pass ``x`` and ``keep`` to add reconstructed field images around the plot.

    Returns
    -------
    str  Absolute path to the written HTML file.
    """
    import json as _json
    import os as _os
    import plotly.graph_objects as go
    import plotly.io as pio
    import matplotlib.colors as mc

    # Apply sign flips early so all downstream code (traces + peripheral images) is consistent
    if flip_pcs:
        scores = scores.copy()
        for pc in flip_pcs:
            scores[:, pc] = -scores[:, pc]

    # ------------------------------------------------------------------
    # 1. Hover thumbnails
    # ------------------------------------------------------------------
    hover_imgs = [
        _field_to_base64(
            or_over_targets(all_fields[neu][channel]),
            all_fields[neu][channel], cmap, axis_lim, hover_px,
        )
        for neu in neurons
    ]

    # ------------------------------------------------------------------
    # 2. Color mapping: None → gray, named labels → HSV (S=0.5, V=0.3..0.5)
    # ------------------------------------------------------------------
    if labels is not None:
        import colorsys
        named = sorted(set(l for l in labels if l is not None))
        _PHI = 0.618033988749895
        lbl2col = {}
        for i, lbl in enumerate(named):
            h = (i * _PHI) % 1.0
            r, g, b = colorsys.hsv_to_rgb(h, 0.70, 1.0)
            lbl2col[lbl] = f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
        lbl2col[None] = "#aaaaaa"
    else:
        named = []
        lbl2col = None

    # ------------------------------------------------------------------
    # 3. Build Plotly traces (named clusters first, unlabeled last)
    # ------------------------------------------------------------------
    traces = []
    per_trace_images = []

    if labels is not None:
        for lbl in named:
            idx = [i for i, l in enumerate(labels) if l == lbl]
            per_trace_images.append([hover_imgs[i] for i in idx])
            traces.append(go.Scatter(
                x=scores[idx, pc_x].tolist(),
                y=scores[idx, pc_y].tolist(),
                mode="markers+text",
                name=str(lbl),
                text=[neurons[i] for i in idx],
                textposition="top center",
                textfont=dict(size=9),
                marker=dict(size=9, color=lbl2col[lbl]),
                hoverinfo="text",
            ))
        none_idx = [i for i, l in enumerate(labels) if l is None]
        if none_idx:
            per_trace_images.append([hover_imgs[i] for i in none_idx])
            traces.append(go.Scatter(
                x=scores[none_idx, pc_x].tolist(),
                y=scores[none_idx, pc_y].tolist(),
                mode="markers+text",
                name="unlabeled",
                text=[neurons[i] for i in none_idx],
                textposition="top center",
                textfont=dict(size=9, color="#999999"),
                marker=dict(size=8, color="#cccccc", opacity=0.75),
                hoverinfo="text",
            ))
    else:
        per_trace_images.append(hover_imgs)
        traces.append(go.Scatter(
            x=scores[:, pc_x].tolist(),
            y=scores[:, pc_y].tolist(),
            mode="markers+text",
            name="neurons",
            text=list(neurons),
            textposition="top center",
            textfont=dict(size=9),
            marker=dict(size=9, color="#4477aa"),
            hoverinfo="text",
        ))

    # ------------------------------------------------------------------
    # 4. Build figure
    # ------------------------------------------------------------------
    fig = go.Figure(data=traces)
    fig.update_layout(
        xaxis_title=f"PC {pc_x + 1}",
        yaxis_title=f"PC {pc_y + 1}",
        title=f"SVD score space  (PC{pc_x + 1} vs PC{pc_y + 1})",
        hovermode="closest",
        autosize=True,
        legend=dict(title="region", font=dict(size=11)) if labels is not None else {},
        plot_bgcolor="white",
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#cccccc", line_width=1)
    fig.add_vline(x=0, line_dash="dash", line_color="#cccccc", line_width=1)

    # ------------------------------------------------------------------
    # 5. Peripheral SVD-reconstructed field images (if x + keep provided)
    # ------------------------------------------------------------------
    if x is not None and keep is not None:
        k = scores.shape[1]
        _, _, Vt = np.linalg.svd(np.asarray(x, float), full_matrices=False)
        Vt_k = Vt[:k].copy()
        # Negate Vt rows to match any sign flips already applied to scores above
        if flip_pcs:
            for pc in flip_pcs:
                if pc < len(Vt_k):
                    Vt_k[pc] = -Vt_k[pc]
        ref_fld = all_fields[neurons[0]][channel]
        W, H = ref_fld.shape[-2], ref_fld.shape[-1]

        sc_x = scores[:, pc_x]
        sc_y = scores[:, pc_y]
        x_min, x_max = float(sc_x.min()), float(sc_x.max())
        y_min, y_max = float(sc_y.min()), float(sc_y.max())
        x_mid = (x_min + x_max) / 2
        y_mid = (y_min + y_max) / 2
        axis_range = max(x_max - x_min, y_max - y_min)

        img_sz = img_frac * axis_range
        gap = 0.05 * axis_range

        periph_scores = {
            "TL": (x_min, y_max), "TC": (0.0,   y_max), "TR": (x_max, y_max),
            "ML": (x_min, 0.0),                          "MR": (x_max, 0.0),
            "BL": (x_min, y_min), "BC": (0.0,   y_min), "BR": (x_max, y_min),
        }
        # Anchor = (left edge, top edge) in data coordinates
        # TC/BC are centered at x=0; ML/MR are centered at y=0
        periph_anchor = {
            "TL": (x_min - gap - img_sz, y_max + gap + img_sz),
            "TC": (0.0 - img_sz / 2,     y_max + gap + img_sz),
            "TR": (x_max + gap,           y_max + gap + img_sz),
            "ML": (x_min - gap - img_sz, 0.0 + img_sz / 2),
            "MR": (x_max + gap,           0.0 + img_sz / 2),
            "BL": (x_min - gap - img_sz, y_min - gap),
            "BC": (0.0 - img_sz / 2,     y_min - gap),
            "BR": (x_max + gap,           y_min - gap),
        }

        for name, (px, py) in periph_scores.items():
            sv = np.zeros(k)
            sv[pc_x] = px
            sv[pc_y] = py
            ff = np.zeros(W * H, dtype=np.float32)
            ff[keep] = (sv @ Vt_k).astype(np.float32)  # no clipping — diverging
            # default mask: use keep reshaped, unless caller supplied field_mask
            _mask = field_mask if field_mask is not None else (
                keep.reshape(W, H) if keep is not None else None)
            b64 = _field_to_base64(ff.reshape(W, H), ref_fld,
                                   cmap, axis_lim, img_px,
                                   clip_zero=False, norm_mode="minmax",
                                   mask=_mask)
            xa, ya = periph_anchor[name]
            fig.add_layout_image(dict(
                source=f"data:image/png;base64,{b64}",
                xref="x", yref="y",
                x=xa, y=ya,
                sizex=img_sz, sizey=img_sz,
                xanchor="left", yanchor="top",
                sizing="contain",
                layer="above",
                opacity=1.0,
            ))

        extra = img_sz + gap + 0.04 * axis_range
        # Use equal spans centred on each axis mid-point → square data space.
        # This guarantees img_sz maps to the same pixel count on both axes,
        # so sizing="contain" fills the box exactly with no centering offset.
        # The half-span covers the wider score range plus the peripheral images.
        half = axis_range / 2 + extra
        x_range = ([x_mid + half, x_mid - half] if xaxis_reversed
                   else [x_mid - half, x_mid + half])
        y_range = ([y_mid + half, y_mid - half] if yaxis_reversed
                   else [y_mid - half, y_mid + half])
        fig.update_layout(
            xaxis=dict(range=x_range),
            # scaleanchor enforces equal pixels-per-unit so images stay square
            yaxis=dict(range=y_range, scaleanchor="x", scaleratio=1),
        )

    # ------------------------------------------------------------------
    # 6. Serialize → HTML
    # ------------------------------------------------------------------
    fig_json = pio.to_json(fig)
    images_js = _json.dumps(per_trace_images)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>SVD score space</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body {{ margin:0; padding:0; background:#fff; height:100vh; overflow:hidden; }}
  #plot {{ width:100%; height:100vh; }}
  #hover-box {{
    position: fixed;
    pointer-events: none;
    display: none;
    background: rgba(255,255,255,0.96);
    border: 1px solid #aaa;
    border-radius: 6px;
    padding: 7px 9px;
    box-shadow: 3px 4px 10px rgba(0,0,0,0.22);
    z-index: 9999;
  }}
  #hover-name {{
    font-family: Arial, sans-serif;
    font-size: 13px;
    font-weight: bold;
    margin-bottom: 5px;
    color: #222;
  }}
  #hover-pic {{ display: block; }}
</style>
</head>
<body>
<div id="plot"></div>
<div id="hover-box">
  <div id="hover-name"></div>
  <img id="hover-pic" src="" width="{hover_px}" height="{hover_px}">
</div>
<script>
var figData = {fig_json};
var images  = {images_js};

Plotly.newPlot('plot', figData.data, figData.layout, {{responsive:true}});

var plotDiv  = document.getElementById('plot');
var hoverBox = document.getElementById('hover-box');
var hoverPic = document.getElementById('hover-pic');
var hoverName= document.getElementById('hover-name');
var mx = 0, my = 0;

document.addEventListener('mousemove', function(e) {{
  mx = e.clientX; my = e.clientY;
  if (hoverBox.style.display === 'block') {{
    positionBox();
  }}
}});

function positionBox() {{
  var bw = hoverBox.offsetWidth  || {hover_px + 30};
  var bh = hoverBox.offsetHeight || {hover_px + 40};
  var vw = window.innerWidth, vh = window.innerHeight;
  var x = mx + 20, y = my - 10;
  if (x + bw > vw) x = mx - bw - 10;
  if (y + bh > vh) y = vh - bh - 10;
  hoverBox.style.left = x + 'px';
  hoverBox.style.top  = y + 'px';
}}

plotDiv.on('plotly_hover', function(data) {{
  var pt = data.points[0];
  var ti = pt.curveNumber;
  var pi = pt.pointIndex;
  if (images[ti] && images[ti][pi]) {{
    hoverPic.src = 'data:image/png;base64,' + images[ti][pi];
    hoverName.textContent = (pt.text instanceof Array) ? pt.text[pi] : (pt.text || '');
    hoverBox.style.display = 'block';
    positionBox();
  }}
}});

plotDiv.on('plotly_unhover', function() {{
  hoverBox.style.display = 'none';
}});
</script>
</body>
</html>"""

    out = _os.path.abspath(output_path)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)
    return out


def compute_or_load(connectome, neuron, hops, cache_dir="fields_us",
                    starting_points=STARTING_POINTS, side="right"):
    """UPSTREAM analytic synaptic fields for one neuron (target), cached as npz.

    Walks upstream from the neuron to the L1-R8 stopping points; the field is indexed by
    the input-cell (stopping) coordinate. Useful for the receptive-field analysis.
    """
    os.makedirs(cache_dir, exist_ok=True)
    fn = os.path.join(cache_dir, f"{neuron}_analytic.npz")
    if os.path.exists(fn):
        return load_SynapticFields(fn)
    paths = connectome.get_paths(neuron, direction="upstream", max_hops=hops)
    fields, _ = paths.get_synaptic_fields(
        method="analytic", stopping_points=list(starting_points), side=side)
    fields.save(fn)
    return fields


def _connectome_out_weight(connectome):
    """Total out-synapse weight per pre node in the full connectome (cached).

    Deduplicates (pre, post) to match the single-valued graph edge weights, then sums per
    pre. Used to normalize downstream transitions so mass can leak off the subgraph.
    """
    if not hasattr(connectome, "_out_weight_cache"):
        db = connectome.database
        dedup = db.drop_duplicates(["pre", "post"], keep="last")
        connectome._out_weight_cache = dedup.groupby("pre")["weight"].sum()
    return connectome._out_weight_cache


def _downstream_matrix_leaky(connectome, graph, node_ids, normalize=True, conditional=False):
    """Downstream (pre->post) transition matrix on a subgraph.

    normalize=False  : raw synapse weights; no probability interpretation.
    normalize=True, conditional=False (default, leaky)
        Rows divided by each node's TRUE out-degree in the FULL connectome.
        Rows sum to <= 1; the deficit is mass leaking to branches that don't
        reach the target.  Gives unconditional P(walk reaches target).
    normalize=True, conditional=True
        Rows divided by each node's out-degree within the SUBGRAPH only.
        Rows sum to 1; the walk is constrained to stay in the subgraph.
        Gives conditional P(walk reaches target | walk stays in subgraph).
    """
    import scipy.sparse as sp
    import networkx as nx
    N = len(node_ids)
    edf = nx.to_pandas_edgelist(graph)          # vectorized edge extraction (fast)
    if len(edf) == 0:
        return sp.csr_matrix((N, N), dtype=float)
    us = edf["source"].to_numpy()
    vs = edf["target"].to_numpy()
    ws = (edf["weight"].to_numpy(dtype=float) if "weight" in edf.columns
          else np.ones(len(edf), dtype=float))
    u_idx = np.searchsorted(node_ids, us)
    v_idx = np.searchsorted(node_ids, vs)
    W = sp.csr_matrix((ws, (u_idx, v_idx)), shape=(N, N))
    if not normalize:
        return W.tocsr()
    if conditional:
        # row-stochastic within the subgraph only
        row_sums = np.asarray(W.sum(axis=1)).ravel()
        inv = np.zeros(N)
        nz = row_sums > 0
        inv[nz] = 1.0 / row_sums[nz]
        return (sp.diags(inv) @ W).tocsr()
    # leaky: normalize by full-connectome out-degree
    out_weight = _connectome_out_weight(connectome)
    true_out = out_weight.reindex(node_ids).fillna(0.0).to_numpy()
    inv = np.zeros(N)
    nz = true_out > 0
    inv[nz] = 1.0 / true_out[nz]
    return (sp.diags(inv) @ W).tocsr()


def _side_criteria(neuron, side):
    """NeuronCriteria restricting a cell type to one optic lobe (endpoint eye isolation)."""
    from fafbseg import flywire
    return flywire.NeuronCriteria(type=neuron, side=side)


def _restrict_side(node_ids, graph, node_info, side):
    """Strict eye isolation: drop graph nodes annotated as the OPPOSITE optic lobe.

    Keeps same-side, 'center'/midline, and unannotated cells. Returns filtered
    (node_ids, graph, node_info). No-op if the annotations lack a 'side' column.
    """
    if "side" not in node_info.columns:
        return node_ids, graph, node_info
    opposite = {"left": "right", "right": "left"}.get(side)
    if opposite is None:
        return node_ids, graph, node_info
    smap = dict(zip(node_info["root_id"].values, node_info["side"].values))
    keep = np.sort(np.asarray([n for n in node_ids if smap.get(n) != opposite]))
    sub = graph.subgraph(keep).copy()
    ni = node_info[node_info["root_id"].isin(keep)]
    return keep, sub, ni


def compute_downstream_or_load(connectome, neuron, hops, cache_dir="fields_ds",
                               starting_points=STARTING_POINTS, side="right",
                               strict_side=False, normalize=True, conditional=False):
    """DOWNSTREAM analytic synaptic fields onto one neuron type, cached as npz.

    Builds the neuron's upstream input subgraph, then propagates DOWNSTREAM from each
    input-channel cell to the neuron (absorbing).

    normalize=False
        Raw synapse-weighted path counts; ``conditional`` is ignored.
        Cache: <neuron>_analytic_raw.npz
    normalize=True, conditional=False  (default — leaky / unconditional)
        Transitions divided by full-connectome out-degree; rows sum to <= 1.
        field[ch][p,q] = P(walk from (p,q) reaches neuron).
        Cache: <neuron>_analytic.npz
    normalize=True, conditional=True
        Transitions divided by subgraph out-degree; rows sum to 1.
        field[ch][p,q] = P(walk reaches neuron | walk constrained to subgraph).
        Cache: <neuron>_analytic_cond.npz
    """
    import scipy.sparse as sp
    from flywire_tools.connectome import SynapticField, SynapticFields
    os.makedirs(cache_dir, exist_ok=True)
    if not normalize:
        suffix = "_analytic_raw"
    elif conditional:
        suffix = "_analytic_cond"
    else:
        suffix = "_analytic"
    fn = os.path.join(cache_dir, f"{neuron}{suffix}.npz")
    if os.path.exists(fn):
        return load_SynapticFields(fn)
    # subgraph of cells on input paths to the neuron (targets restricted to one eye)
    paths = connectome.get_paths(_side_criteria(neuron, side),
                                 direction="upstream", max_hops=hops, skip_recurrents=False)
    node_ids, graph, node_info = paths.node_ids, paths.graph, paths.node_info
    if strict_side:
        node_ids, graph, node_info = _restrict_side(node_ids, graph, node_info, side)
    N = len(node_ids)
    # downstream transition matrix (leaky, conditional, or raw)
    P_down = _downstream_matrix_leaky(connectome, graph, node_ids, normalize=normalize, conditional=conditional)
    # the neuron's own cells are level 0 -> absorbing endpoint of the downstream walk
    target_ids = np.intersect1d(node_info[node_info.level == 0].root_id.values, node_ids)
    target_idx = np.searchsorted(node_ids, target_ids)
    n_tgt = len(target_idx)
    is_target = np.zeros(N, dtype=bool)
    is_target[target_idx] = True
    P_eff = (sp.diags((~is_target).astype(float)) @ P_down).tocsr()
    num_hops = max(int(np.abs(node_info.level.dropna().values).max()), 1)
    # retinal grid + channel source cells (coordinates come from the sources here)
    vci = pd.read_csv(COLUMN_INFO_CSV, index_col=0)
    all_ps, all_qs = vci[["p", "q"]].values.T
    pmin, pmax = int(all_ps.min()), int(all_ps.max()) + 1
    qmin, qmax = int(all_qs.min()), int(all_qs.max()) + 1
    width, height = pmax - pmin, qmax - qmin
    pvals, qvals = np.meshgrid(np.arange(pmin, pmax), np.arange(qmin, qmax), indexing="ij")
    types_col, hemi = vci["type"], vci["hemisphere"]
    grids = {}
    # 'all': normalized -> P(at least one channel reaches) = 1 - prod(1 - p_ch)
    #         unnormalized -> sum of raw synapse-weighted counts across channels
    none_reach = np.ones((n_tgt, width, height), dtype=np.float64)
    raw_sum = np.zeros((n_tgt, width, height), dtype=np.float64)
    for ch in starting_points:
        sel = vci[(types_col == ch) & (hemi == side)]
        if len(sel) == 0:
            continue
        sel = sel[np.isin(sel.index.values, node_ids)]
        if len(sel) == 0:
            continue
        s_idx = np.searchsorted(node_ids, sel.index.values)
        s_ps = sel["p"].values.astype(int)
        s_qs = sel["q"].values.astype(int)
        n_src = len(s_idx)
        dist = sp.csr_matrix((np.ones(n_src), (np.arange(n_src), s_idx)), shape=(n_src, N))
        # reach[c, t] = (probability | raw count) a walk from source c is absorbed at target t
        reach = np.zeros((n_src, n_tgt))
        for _ in range(num_hops):
            reach += np.asarray(dist[:, target_idx].todense())
            dist = dist @ P_eff
        reach += np.asarray(dist[:, target_idx].todense())
        grid = np.zeros((n_tgt, width, height), dtype=np.float32)
        for c in range(n_src):
            grid[:, s_ps[c] - pmin, s_qs[c] - qmin] += reach[c]
        field = SynapticField(grid, ps=pvals, qs=qvals)
        field.root_ids = np.asarray(target_ids)
        grids[ch] = field
        if normalize:
            none_reach *= (1.0 - np.clip(grid, 0.0, 1.0))
        else:
            raw_sum += grid
    all_grid = ((1.0 - none_reach) if normalize else raw_sum).astype(np.float32)
    all_field = SynapticField(all_grid, ps=pvals, qs=qvals)
    all_field.root_ids = np.asarray(target_ids)
    grids["all"] = all_field
    out = SynapticFields(grids)
    out.save(fn)
    return out


def compute_downstream_mc_or_load(connectome, neuron, hops, reps=2000,
                                  cache_dir="fields_ds_mc",
                                  starting_points=STARTING_POINTS, side="right", seed=0,
                                  strict_side=False, normalize=True, conditional=False):
    """Monte Carlo estimate of the downstream fields (killed random walk), for comparison.

    normalize=False
        Raw walker-hit counts.  ``conditional`` is ignored.
        Cache: <neuron>_mc_raw.npz
    normalize=True, conditional=False  (default — leaky / unconditional)
        Sampling uses the leaky matrix (full-connectome out-degree); walks die when they
        leave the subgraph.  Output is fraction of walks absorbed at the neuron.
        Cache: <neuron>_mc.npz
    normalize=True, conditional=True
        Sampling uses the subgraph-stochastic matrix; walks are constrained to the
        subgraph (no leakage).  Output is fraction of constrained walks absorbed.
        Cache: <neuron>_mc_cond.npz
    """
    import scipy.sparse as sp
    from flywire_tools.connectome import SynapticField, SynapticFields
    os.makedirs(cache_dir, exist_ok=True)
    if not normalize:
        suffix = "_mc_raw"
    elif conditional:
        suffix = "_mc_cond"
    else:
        suffix = "_mc"
    fn = os.path.join(cache_dir, f"{neuron}{suffix}.npz")
    if os.path.exists(fn):
        return load_SynapticFields(fn)
    paths = connectome.get_paths(_side_criteria(neuron, side),
                                 direction="upstream", max_hops=hops, skip_recurrents=False)
    node_ids, graph, node_info = paths.node_ids, paths.graph, paths.node_info
    if strict_side:
        node_ids, graph, node_info = _restrict_side(node_ids, graph, node_info, side)
    N = len(node_ids)
    P = _downstream_matrix_leaky(connectome, graph, node_ids, normalize=True, conditional=conditional)
    indptr, neighbors, probs = P.indptr, P.indices, P.data
    rowsum = np.asarray(P.sum(axis=1)).ravel()
    # per-row cumulative probabilities (vectorized) for sampling within each node's row
    csum = np.cumsum(probs) if probs.size else probs
    row_of_data = np.repeat(np.arange(N), np.diff(indptr))
    offset = np.zeros(N)
    starts = indptr[:-1]
    nz = starts > 0
    offset[nz] = csum[starts[nz] - 1]
    cum = (csum - offset[row_of_data]) if probs.size else probs
    target_ids = np.intersect1d(node_info[node_info.level == 0].root_id.values, node_ids)
    target_idx = np.searchsorted(node_ids, target_ids)
    n_tgt = len(target_idx)
    is_target = np.zeros(N, dtype=bool)
    is_target[target_idx] = True
    # map absorbing node -> its position among the target frames (-1 if not a target)
    target_pos = np.full(N, -1, dtype=np.int64)
    target_pos[target_idx] = np.arange(n_tgt)
    num_hops = max(int(np.abs(node_info.level.dropna().values).max()), 1)
    rng = np.random.default_rng(seed)
    vci = pd.read_csv(COLUMN_INFO_CSV, index_col=0)
    all_ps, all_qs = vci[["p", "q"]].values.T
    pmin, pmax = int(all_ps.min()), int(all_ps.max()) + 1
    qmin, qmax = int(all_qs.min()), int(all_qs.max()) + 1
    width, height = pmax - pmin, qmax - qmin
    pvals, qvals = np.meshgrid(np.arange(pmin, pmax), np.arange(qmin, qmax), indexing="ij")
    types_col, hemi = vci["type"], vci["hemisphere"]
    grids = {}
    # 'all': normalized -> P(at least one channel reaches) = 1 - prod(1 - p_ch)
    #         unnormalized -> sum of raw hit counts across channels
    none_reach = np.ones((n_tgt, width, height), dtype=np.float64)
    raw_sum = np.zeros((n_tgt, width, height), dtype=np.float64)
    for ch in starting_points:
        sel = vci[(types_col == ch) & (hemi == side)]
        if len(sel) == 0:
            continue
        sel = sel[np.isin(sel.index.values, node_ids)]
        if len(sel) == 0:
            continue
        s_idx = np.searchsorted(node_ids, sel.index.values)
        s_ps = sel["p"].values.astype(int)
        s_qs = sel["q"].values.astype(int)
        n_src = len(s_idx)
        state = np.repeat(s_idx, reps)                 # current node per walker
        src_of = np.repeat(np.arange(n_src), reps)     # source index each walker came from
        alive = np.ones(state.size, dtype=bool)
        # reached[t, c] = walks from source c absorbed at target t
        reached = np.zeros((n_tgt, n_src), dtype=float)
        for hop in range(num_hops + 1):
            at_t = alive & is_target[state]            # absorbed at the neuron
            if at_t.any():
                np.add.at(reached, (target_pos[state[at_t]], src_of[at_t]), 1.0)
                alive[at_t] = False
            if hop == num_hops or not alive.any():
                break
            idx = np.where(alive)[0]
            cur = state[idx]
            u = rng.random(idx.size)
            died = u >= rowsum[cur]                     # leaked out of the subgraph
            alive[idx[died]] = False
            surv = ~died
            sidx = idx[surv]
            cs = cur[surv]
            us = u[surv]
            if cs.size == 0:
                continue
            nxt = np.empty(cs.size, dtype=np.int64)
            order = np.argsort(cs, kind="stable")
            cs_o, us_o = cs[order], us[order]
            uniq, ustarts = np.unique(cs_o, return_index=True)
            ustarts = np.append(ustarts, cs_o.size)
            for k, node in enumerate(uniq):
                lo, hi = ustarts[k], ustarts[k + 1]
                a, b = indptr[node], indptr[node + 1]
                jj = np.searchsorted(cum[a:b], us_o[lo:hi], side="right")
                jj = np.clip(jj, 0, b - a - 1)
                nxt[order[lo:hi]] = neighbors[a + jj]
            state[sidx] = nxt
        reach = (reached / reps) if normalize else reached   # (n_tgt, n_src)
        grid = np.zeros((n_tgt, width, height), dtype=np.float32)
        for c in range(n_src):
            grid[:, s_ps[c] - pmin, s_qs[c] - qmin] += reach[:, c]
        field = SynapticField(grid, ps=pvals, qs=qvals)
        field.root_ids = np.asarray(target_ids)
        grids[ch] = field
        if normalize:
            none_reach *= (1.0 - np.clip(grid, 0.0, 1.0))
        else:
            raw_sum += grid
    all_grid = ((1.0 - none_reach) if normalize else raw_sum).astype(np.float32)
    all_field = SynapticField(all_grid, ps=pvals, qs=qvals)
    all_field.root_ids = np.asarray(target_ids)
    grids["all"] = all_field
    out = SynapticFields(grids)
    out.save(fn)
    return out


def plot_field_grid(all_fields, neurons, channels=None, out_png=None,
                    axis_lim=90, label_size=16, title=None, cmap=None,
                    row_height=2.2, col_width=2.2, hspace=None, wspace=None, 
                    lognorm=True):
    """Grid of analytic synaptic fields: one row per neuron, one column per channel.

    'all' is placed first. The seven input channels share one log color scale (pooled
    across every neuron); 'all' has its own shared scale. Each row's leftmost subplot is
    labelled with the neuron name (large ylabel). ``cmap`` overrides the module default
    (e.g. cmap='magma'). Use ``row_height`` and ``hspace`` to reduce vertical spacing
    (e.g. row_height=1.4, hspace=0.0).
    """
    if cmap is None:
        cmap = CMAP
    if channels is None:
        channels = ["all"] + STARTING_POINTS
    input_channels = [c for c in channels if c != "all"]
    # shared norms pooled across all neurons
    input_imgs, all_imgs = [], []
    for neuron in neurons:
        flds = all_fields[neuron]
        input_imgs += [aggregate_image(flds[c]) for c in input_channels if c in flds]
        if "all" in flds:
            all_imgs.append(aggregate_image(flds["all"]))
    if lognorm:
        norm_inputs = make_lognorm(input_imgs)
        norm_all = make_lognorm(all_imgs)
    else:
        norm_inputs = input_imgs
        norm_all = all_imgs
    nrows, ncols = len(neurons), len(channels)
    fig, axes = plt.subplots(nrows, ncols, figsize=(col_width * ncols, row_height * nrows),
                             squeeze=False, constrained_layout=True)
    if hspace is not None or wspace is not None:
        try:
            engine = fig.get_layout_engine()
            engine.set(hspace=0.0 if hspace is None else hspace,
                       wspace=0.0 if wspace is None else wspace)
        except Exception:
            pass
    for r, neuron in enumerate(neurons):
        flds = all_fields[neuron]
        for c, channel in enumerate(channels):
            ax = axes[r][c]
            norm = norm_all if channel == "all" else norm_inputs
            if channel in flds and norm is not None:
                draw_hex(ax, flds[channel], norm=norm, cmap=cmap, axis_lim=axis_lim)
            else:
                ax.set_aspect("equal")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlim(-axis_lim, axis_lim)
                ax.set_ylim(-axis_lim, axis_lim)
            if r == 0:
                ax.set_title(channel, fontsize=label_size)
            if c == 0:
                ax.set_ylabel(neuron, fontsize=label_size, rotation=90, labelpad=10)
    # two shared colorbars: 'all' (left) and input channels (right)
    if norm_all is not None:
        sm = ScalarMappable(norm=norm_all, cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, ax=axes[:, 0].tolist(), location="left", shrink=0.35,
                     label="all (log)")
    if norm_inputs is not None:
        sm = ScalarMappable(norm=norm_inputs, cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, ax=axes[:, 1:].ravel().tolist(), location="right", shrink=0.35,
                     label="input channel (log)")
    if title:
        fig.suptitle(title, fontsize=label_size + 4)
    if out_png:
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
    return fig, axes


def field_scalar(field, stat="mean"):
    """Summarize a field as a single scalar: per-target total, aggregated over targets.

    For each target cell we sum the first-passage field over space (its total input
    probability from this channel), then take the mean or median across target cells.
    """
    arr = np.asarray(field, dtype=float)
    per_target = arr.reshape(arr.shape[0], -1).sum(1)
    if stat == "mean":
        return float(per_target.mean())
    if stat == "median":
        return float(np.median(per_target))
    raise ValueError(f"unknown stat: {stat}")


def summary_table(all_fields, neurons, channels=None, stat="mean"):
    """DataFrame of field_scalar values with neurons as rows and channels as columns."""
    import pandas as pd
    if channels is None:
        channels = STARTING_POINTS + ["all"]
    data = {}
    for neuron in neurons:
        flds = all_fields[neuron]
        data[neuron] = {c: (field_scalar(flds[c], stat) if c in flds else np.nan)
                        for c in channels}
    return pd.DataFrame(data).T[channels]


# ----------------------------------------------------------------------------------------
# Phase 1 -- sensitivity / gain
# ----------------------------------------------------------------------------------------
def or_over_targets(field, normalize=True):
    """OR a per-target field (n_tgt, W, H) across target replicates -> (W, H).

    When normalize=True (default, fields are probabilities in [0,1]):
        out[p] = 1 - prod_t (1 - field[t, p]) = P(at least one target reached at pixel p).
    When normalize=False (fields are raw synapse counts):
        out[p] = sum_t field[t, p]  (total synapse-weighted paths to any target from pixel p).
    """
    if not normalize:
        return np.asarray(field).sum(axis=0).astype(float)
    return 1.0 - np.prod(1.0 - np.clip(field, 0.0, 1.0), axis=0)


def sensitivity_field(fields, normalize=True):
    """Per-pixel sensitivity for one neuron.

    When normalize=True (default): P(at least one input reaches at least one target cell),
    computed as OR over target replicates of the 'all' field.
    When normalize=False: total synapse-weighted paths to any target cell (sum over targets).
    Returns a (W, H) array aligned with the field's ps / qs.
    """
    return or_over_targets(fields["all"], normalize=normalize)


def bootstrap_ci(values, stat=np.mean, reps=10000, confidence=95, seed=0):
    """Bootstrap CI of a statistic (default mean) of a 1-D array. Returns (point, lo, hi)."""
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boot = stat(rng.choice(v, size=(reps, v.size), replace=True), axis=1)
    half = (100 - confidence) / 2
    lo, hi = np.percentile(boot, [half, 100 - half])
    return (float(stat(v)), float(lo), float(hi))


def sensitivity_summary(all_fields, neurons, reps=10000, confidence=95,
                        positive_only=True, normalize=True):
    """Per-neuron mean & median sensitivity with bootstrap CIs (mean vs median => skew).

    For each neuron: sensitivity_field -> flattened pixel values (positive-only by default).
    Reports the bootstrap mean and median with 95% CIs.

    normalize=True  (default): fields are probabilities; OR-over-targets formula applied.
    normalize=False: fields are raw synapse counts; sum-over-targets used instead.
    """
    import pandas as pd
    rows = {}
    for neuron in neurons:
        sens = sensitivity_field(all_fields[neuron], normalize=normalize).ravel()
        vals = sens[sens > 0] if positive_only else sens
        m, lo, hi = bootstrap_ci(vals, np.mean, reps=reps, confidence=confidence)
        md, mlo, mhi = bootstrap_ci(vals, np.median, reps=reps, confidence=confidence)
        rows[neuron] = dict(mean=m, lo=lo, hi=hi, median=md, med_lo=mlo, med_hi=mhi,
                            skew=m - md, n_px=int(vals.size))
    cols = ["mean", "lo", "hi", "median", "med_lo", "med_hi", "skew", "n_px"]
    return pd.DataFrame(rows).T[cols]


def plot_sensitivity_distributions(all_fields, neurons, summary=None, positive_only=True,
                                   ncols=6, panel=2.0, log=True):
    """Grid of per-neuron sensitivity distributions with mean + foreground-mean CI marks.

    One small panel per neuron: histogram of the ~Npx pixel probabilities, with the
    whole-field mean (blue) and foreground mean (red) and their bootstrap CIs overlaid.
    """
    if summary is None:
        summary = sensitivity_summary(all_fields, neurons, positive_only=positive_only)
    n = len(neurons)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * panel, nrows * panel),
                             squeeze=False)
    for i, neuron in enumerate(neurons):
        ax = axes[i // ncols][i % ncols]
        sens = sensitivity_field(all_fields[neuron]).ravel()
        vals = sens[sens > 0] if positive_only else sens
        if vals.size:
            bins = np.logspace(np.log10(max(vals.min(), 1e-6)), np.log10(vals.max()), 30) \
                if log else 30
            ax.hist(vals, bins=bins, color="0.7")
            if log:
                ax.set_xscale("log")
        r = summary.loc[neuron]
        ax.axvspan(r["lo"], r["hi"], color="tab:blue", alpha=0.25)
        ax.axvline(r["mean"], color="tab:blue", lw=1.5)
        ax.axvspan(r["med_lo"], r["med_hi"], color="tab:red", alpha=0.25)
        ax.axvline(r["median"], color="tab:red", lw=1.5)
        ax.set_title(neuron, fontsize=9)
        ax.set_yticks([])
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.tight_layout()
    return fig, axes


# ----------------------------------------------------------------------------------------
# Phase 2 -- input-channel pattern (gain-normalized) + scaled-combined assumption check
# ----------------------------------------------------------------------------------------
def gain_normalized_summary(all_fields, neurons, sens_summary, channels=STARTING_POINTS,
                            positive_only=True):
    """M x channels matrix of gain-normalized mean channel reach (Phase 2).

    For each neuron, each channel's OR-over-targets reach field is divided by the neuron's
    median sensitivity (its gain, from `sens_summary['median']`) and averaged over
    (positive) pixels. This factors out overall sensitivity so the matrix reflects the
    *pattern* of input-channel contributions rather than the gain.
    """
    import pandas as pd
    rows = {}
    for neuron in neurons:
        flds = all_fields[neuron]
        g = float(sens_summary.loc[neuron, "median"])
        row = {}
        for ch in channels:
            if ch in flds and g > 0:
                img = or_over_targets(flds[ch]) / g
                vals = img[img > 0] if positive_only else img.ravel()
                row[ch] = float(vals.mean()) if np.asarray(vals).size else 0.0
            else:
                row[ch] = np.nan
        rows[neuron] = row
    return pd.DataFrame(rows).T[list(channels)]


def combined_scaling_check(all_fields, neurons, channels=STARTING_POINTS, normalize=True):
    """Assumption check: is each channel's reach field ~ a SCALED copy of the combined `all`?

    For each neuron x channel, fits the best scalar a minimizing ||f_ch - a * f_all||
    (a = <f_ch, f_all> / <f_all, f_all>) on the pooled-over-targets fields, and reports:
      - resid: ||f_ch - a f_all|| / ||f_ch||   (0 = perfectly a scaled copy; larger = deviates)
      - corr:  Pearson correlation between f_ch and f_all over their union support.
    Low resid / high corr => the combined-field-scaling assumption holds; the exceptions are
    the high-resid, low-corr entries. Returns (resid_df, corr_df).

    ``normalize`` is forwarded to :func:`or_over_targets`: pass ``False`` for raw
    synapse-count fields (``True`` binarises counts via the probability-OR, which collapses
    the ``all`` field to a constant and makes every correlation NaN).
    """
    import pandas as pd
    resid_rows, corr_rows = {}, {}
    for neuron in neurons:
        flds = all_fields[neuron]
        base = or_over_targets(flds["all"], normalize=normalize).ravel()
        denom = float(base @ base)
        rr, cr = {}, {}
        for ch in channels:
            if ch not in flds:
                rr[ch] = np.nan
                cr[ch] = np.nan
                continue
            f = or_over_targets(flds[ch], normalize=normalize).ravel()
            a = (float(f @ base) / denom) if denom > 0 else 0.0
            res = f - a * base
            rr[ch] = float(np.linalg.norm(res) / (np.linalg.norm(f) + 1e-12))
            m = (f > 0) | (base > 0)
            cr[ch] = (float(np.corrcoef(f[m], base[m])[0, 1])
                      if m.sum() > 2 and f[m].std() > 0 and base[m].std() > 0 else np.nan)
        resid_rows[neuron] = rr
        corr_rows[neuron] = cr
    cols = list(channels)
    return pd.DataFrame(resid_rows).T[cols], pd.DataFrame(corr_rows).T[cols]


# ----------------------------------------------------------------------------------------
# Phase 3 -- spatial RF taxonomy (SVD + clustering)
# ----------------------------------------------------------------------------------------
def rf_matrix(all_fields, neurons, sens_summary=None, normalize="gain", mask=True, channel=None):
    """M x N matrix of per-neuron spatial RF fields for SVD / clustering.

    Rows = neurons, cols = retinal pixels. Each row is the neuron's field for ``channel``
    (OR over target replicates), or the full sensitivity field (OR of ``all``) when
    ``channel`` is None. Optionally divided by gain (``normalize='gain'``, the median from
    ``sens_summary``), z-scored per row (``'zscore'``), or left raw (``'none'``).
    If ``mask``, keeps only pixels active (>0) in at least one neuron. Returns (X, keep_mask).
    """
    if channel is None:
        rows = [sensitivity_field(all_fields[n]).ravel() for n in neurons]
    else:
        rows = [or_over_targets(all_fields[n][channel]).ravel() for n in neurons]
    M = np.vstack(rows).astype(float)                      # (n_neuron, n_px)
    keep = (M > 0).any(axis=0) if mask else np.ones(M.shape[1], dtype=bool)
    X = M[:, keep]
    if normalize == "gain" and sens_summary is not None:
        g = np.asarray([sens_summary.loc[n, "median"] for n in neurons], float)[:, None]
        X = X / np.where(g > 0, g, 1.0)
    elif normalize == "zscore":
        X = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-10)
    return X, keep


def svd_reduce(X, var_threshold=0.90):
    """Full SVD of X (M small); keep components for >= var_threshold explained variance.

    Returns (scores, s, explained_var, n_kept) where scores = U[:, :n] * s[:n] are the
    neurons embedded in the reduced spatial-basis space.
    """
    U, s, Vt = np.linalg.svd(np.asarray(X, float), full_matrices=False)
    ev = s ** 2 / np.sum(s ** 2)
    n = int(np.searchsorted(np.cumsum(ev), var_threshold) + 1)
    n = max(1, min(n, len(s)))
    return U[:, :n] * s[:n], s, ev, n


def cluster_rfs(scores, method="hdbscan", min_cluster_size=3, n_clusters=None):
    """Cluster neurons in reduced RF space using cosine geometry. Returns integer labels.

    method='hdbscan' -> sklearn HDBSCAN on a precomputed cosine-distance matrix (label -1 =
    noise). method='agglomerative' -> AgglomerativeClustering (average linkage, cosine).
    """
    from sklearn.metrics.pairwise import cosine_distances
    D = cosine_distances(scores)
    if method == "hdbscan":
        from sklearn.cluster import HDBSCAN
        return HDBSCAN(min_cluster_size=min_cluster_size,
                       metric="precomputed").fit_predict(D)
    from sklearn.cluster import AgglomerativeClustering
    return AgglomerativeClustering(n_clusters=n_clusters, metric="precomputed",
                                   linkage="average").fit_predict(D)


def plot_rf_taxonomy(all_fields, neurons, labels, ncols=8, panel=1.6, axis_lim=90,
                     cmap=CMAP, vmin=None, vmax=None, channel=None):
    """Plot each neuron's RF as a hex field, grouped/annotated by cluster label.

    Neurons are ordered by cluster; each panel titled '<neuron> [c<label>]'. Useful as the
    RF-shape taxonomy overview.

    Parameters
    ----------
    channel : str or None
        If given (e.g. ``'L5'``), plot the OR-over-replicates of that specific input channel
        instead of the full sensitivity field (OR of ``'all'``).

    If ``vmin``/``vmax`` are provided, one shared colormap normalization is applied across
    all plotted neurons. If both are None, each panel uses its own [0, max] normalization.
    """
    def _get_field(neuron):
        if channel is None:
            return sensitivity_field(all_fields[neuron])
        return or_over_targets(all_fields[neuron][channel])

    order = np.argsort(labels, kind="stable")
    ordered = [neurons[i] for i in order]
    lab = [labels[i] for i in order]
    n = len(ordered)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * panel, nrows * panel),
                             squeeze=False)
    global_norm = None
    if vmin is not None or vmax is not None:
        vals = []
        for neuron in ordered:
            img = _get_field(neuron).ravel()
            img = img[np.isfinite(img)]
            if img.size:
                vals.append(img)
        if vals:
            pooled = np.concatenate(vals)
            gvmin = float(pooled.min()) if vmin is None else float(vmin)
            gvmax = float(pooled.max()) if vmax is None else float(vmax)
            if gvmax <= gvmin:
                gvmax = gvmin + 1e-12
            from matplotlib.colors import Normalize
            global_norm = Normalize(gvmin, gvmax)
    for i, (neuron, cl) in enumerate(zip(ordered, lab)):
        ax = axes[i // ncols][i % ncols]
        img = _get_field(neuron)
        ref_key = channel if (channel is not None and channel in all_fields[neuron]) else "all"
        fld = all_fields[neuron][ref_key]
        field2d = type(fld)(img[None], ps=fld.ps, qs=fld.qs)
        if global_norm is None:
            vmax_local = float(img.max()) or 1.0
            from matplotlib.colors import Normalize
            norm = Normalize(0, vmax_local)
        else:
            norm = global_norm
        draw_hex(ax, field2d, norm=norm, cmap=cmap, axis_lim=axis_lim)
        ax.set_title(f"{neuron} [c{cl}]", fontsize=8)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.tight_layout()
    return fig, axes


# -------------------------------------------------------------------------------------
# Dendrogram + embedded hex-field figure
# -------------------------------------------------------------------------------------

def _get_node_leaves(Z, n_leaves, node_id):
    """Return sorted list of original leaf indices (0-based) under *node_id*.

    Leaves are node IDs 0..n_leaves-1; internal nodes are n_leaves..2*n_leaves-2.
    ``Z`` is the scipy linkage matrix (shape (n_leaves-1, 4)).
    """
    if node_id < n_leaves:
        return [int(node_id)]
    row = int(node_id - n_leaves)
    left  = int(Z[row, 0])
    right = int(Z[row, 1])
    return _get_node_leaves(Z, n_leaves, left) + _get_node_leaves(Z, n_leaves, right)


def _dendrogram_node_positions(Z, dend):
    """Map internal node IDs to ``(x_data, y_data)`` positions in the dendrogram.

    scipy's dendrogram encodes each merge as a U-shape whose four points are:

        (icoord[k][0], dcoord[k][0])  left  child bottom
        (icoord[k][1], dcoord[k][1])  left  junction  top   <- merge point
        (icoord[k][2], dcoord[k][2])  right junction  top
        (icoord[k][3], dcoord[k][3])  right child bottom

    The merge height ``dcoord[k][1]`` equals ``Z[j, 2]`` for some row j.  We match on
    height (unique in practice for cosine+average linkage) and place the panel at
    ``x = (icoord[k][1] + icoord[k][2]) / 2``, ``y = dcoord[k][1]``.

    Returns ``{internal_node_id: (x, y)}``.
    """
    n_leaves = len(Z) + 1
    # Build height -> (x_mid, y) from the dendrogram layout.
    # Use a list to tolerate duplicate heights (rare but possible).
    from collections import defaultdict
    height_to_pos = defaultdict(list)
    for ic, dc in zip(dend["icoord"], dend["dcoord"]):
        x_mid = (ic[1] + ic[2]) / 2.0
        y     = float(dc[1])          # == dc[2]
        height_to_pos[y].append((x_mid, y))

    # Sort each bucket by x so ties are popped in left→right order.
    for h in height_to_pos:
        height_to_pos[h].sort(key=lambda p: p[0])

    # scipy processes Z rows in bottom-up order when building the tree.
    # We need to assign a position to each row.  We group Z rows by height,
    # then zip them with the sorted positions at that height.
    from collections import defaultdict as _dd
    height_to_rows = _dd(list)
    for j, row in enumerate(Z):
        height_to_rows[float(row[2])].append(j)
    for h in height_to_rows:
        height_to_rows[h].sort()      # ascending row index = left-to-right merges

    node_pos = {}
    for h, positions in height_to_pos.items():
        rows = height_to_rows.get(h, [])
        for pos, row_j in zip(positions, rows):
            internal_id = n_leaves + row_j
            node_pos[internal_id] = pos
    return node_pos


def plot_dendrogram_with_fields(
    all_fields,
    neurons,
    scores,
    x=None,
    keep=None,
    channel="L5",
    n_clusters=32,
    log_norm=True,
    min_members=2,
    leaf_panel=0.45,
    axis_lim=None,
    cmap="magma",
    figsize=None,
    node_growth=3.0,
):
    """Draw a dendrogram with embedded channel hex fields at every leaf and internal node.

    The tree is drawn *horizontally* (root at the right, leaves down the left side) so
    the neuron names read left-to-right on the left spine.  Internal-node field panels
    are scaled in proportion to their cosine-distance (merge height) and then shrunk as
    needed by a greedy resolver so that no two panels overlap.

    Leaf panels show the actual OR-over-targets field.  Internal node panels show the
    mean of the leaf fields in that subtree — no SVD back-projection, no z-score
    artefacts.  All panels are placed at *exact* dendrogram node coordinates using
    matplotlib's ``transData → transFigure`` transform chain.

    Parameters
    ----------
    all_fields : dict
        Neuron -> SynapticFields containing *channel* (e.g. ``'L5'``).
    neurons : list[str]
        Ordered list of neuron names; row order of *scores*.
    scores : ndarray, shape (M, k)
        SVD scores from :func:`svd_reduce` — used only to build the linkage.
    x, keep : ignored
        Kept for backward compatibility.
    channel : str
        Which input channel to display.
    n_clusters : int
        Number of top-level clusters for branch colouring.
    log_norm : bool
        Use :class:`~matplotlib.colors.LogNorm` (shared across leaf panels).
    min_members : int
        Minimum subtree size to draw an internal node panel (2 = every merge).
    leaf_panel : float
        Approximate panel width in inches; drives figure width.
    axis_lim : float or None
        Visual-field extent (degrees).  ``None`` auto-fits every panel to its data.
    cmap : str
        Matplotlib colourmap name.
    figsize : tuple or None
        Override figure size; auto-computed if None.
    node_growth : float
        Controls how much larger the deepest internal-node panels are than the leaf
        panels.  A node at the maximum cosine distance is drawn ``(1 + node_growth)``
        times the leaf-panel size (before non-overlap shrinking).

    Returns
    -------
    fig, ax_dend, leaf_axes
        *leaf_axes* is a list in top-to-bottom display order.
    """
    from scipy.spatial.distance import squareform
    from scipy.cluster.hierarchy import linkage as scipy_linkage, dendrogram as scipy_dendrogram
    from sklearn.metrics.pairwise import cosine_distances
    from flywire_tools.connectome import SynapticField
    from collections import defaultdict

    n_leaves = len(neurons)

    # ------------------------------------------------------------------
    # 1. Linkage matrix
    # ------------------------------------------------------------------
    D = cosine_distances(scores)
    np.fill_diagonal(D, 0.0)
    Z = scipy_linkage(squareform(D), method="average")

    # ------------------------------------------------------------------
    # 2. Reference field geometry
    # ------------------------------------------------------------------
    ref_fld = all_fields[neurons[0]][channel]
    ps_ref  = ref_fld.ps
    qs_ref  = ref_fld.qs

    # ------------------------------------------------------------------
    # 3. Colour threshold for n_clusters top-level branches
    # ------------------------------------------------------------------
    idx = max(0, min(len(Z) - n_clusters, len(Z) - 1))
    color_threshold = float(Z[idx, 2]) * 0.9999

    # ------------------------------------------------------------------
    # 4. Pre-compute actual channel images for every leaf (no z-score)
    # ------------------------------------------------------------------
    leaf_imgs = [or_over_targets(all_fields[neurons[i]][channel])
                 for i in range(n_leaves)]

    # ------------------------------------------------------------------
    # 5. Shared LogNorm (colourbar only) + per-leaf norms for display
    # ------------------------------------------------------------------
    if log_norm:
        norm = make_lognorm(leaf_imgs)   # shared norm used for the colourbar
        if norm is None:
            from matplotlib.colors import Normalize
            all_vals = np.concatenate([img.ravel() for img in leaf_imgs])
            norm = Normalize(float(all_vals.min()), float(all_vals.max()))
    else:
        from matplotlib.colors import Normalize
        all_vals = np.concatenate([img.ravel() for img in leaf_imgs])
        norm = Normalize(0.0, float(np.percentile(all_vals[all_vals > 0], 99.5)))

    # Per-field norms: each panel auto-scales to its own dynamic range so
    # shape is visible regardless of the neuron's overall amplitude.
    def _per_field_norm(img):
        vals = img[img > 0]
        if vals.size == 0:
            return norm
        if log_norm:
            from matplotlib.colors import LogNorm
            vmin = float(np.percentile(vals, 2))
            vmax = float(np.percentile(vals, 99.8))
            if vmin <= 0:
                vmin = float(vals.min())
            if vmin >= vmax:
                vmax = vmin + 1e-12
            return LogNorm(vmin=vmin, vmax=vmax)
        else:
            from matplotlib.colors import Normalize
            return Normalize(0.0, float(np.percentile(vals, 99.5)))

    leaf_norms = [_per_field_norm(img) for img in leaf_imgs]

    # ------------------------------------------------------------------
    # 6. Figure — horizontal dendrogram (root at right, leaves down the left)
    # ------------------------------------------------------------------
    if figsize is None:
        fh = n_leaves * leaf_panel + 1.5
        fw = 14.0
        figsize = (fw, fh)
    fw, fh = float(figsize[0]), float(figsize[1])

    gap    = 0.008
    dend_l = 0.19                       # left region reserved for names + leaf panels
    dend_r = 0.985
    dend_b = 0.04
    dend_t = 0.93                       # leave a top strip for the colourbar

    fig = plt.figure(figsize=figsize)
    # colourbar strip along the top of the dendrogram body (pre-allocated so
    # ax_dend's bounding box is fixed for the lifetime of the figure)
    cbar_ax = fig.add_axes([dend_l, dend_t + 0.02, dend_r - dend_l, 0.012])
    ax_dend = fig.add_axes([dend_l, dend_b, dend_r - dend_l, dend_t - dend_b])

    # ------------------------------------------------------------------
    # 7. Draw dendrogram horizontally (root at right)
    # ------------------------------------------------------------------
    dend = scipy_dendrogram(
        Z,
        ax=ax_dend,
        orientation="right",
        no_labels=True,
        color_threshold=color_threshold,
        above_threshold_color="#888888",
    )
    # Thicken every dendrogram line
    for line in ax_dend.lines:
        line.set_linewidth(2.0)
    ax_dend.set_yticks([])
    ax_dend.set_xlabel("cosine distance", fontsize=14)
    ax_dend.tick_params(axis="x", labelsize=12, width=1.5, length=5)
    ax_dend.spines["bottom"].set_linewidth(1.5)
    for sp in ["top", "right", "left"]:
        ax_dend.spines[sp].set_visible(False)

    dist_max = float(Z[:, 2].max()) or 1.0

    # ------------------------------------------------------------------
    # 8. Coordinate helpers  (valid after dendrogram sets xlim/ylim)
    # ------------------------------------------------------------------
    def _to_fig(xd, yd):
        """Exact data → figure-fraction via matplotlib transforms."""
        disp = ax_dend.transData.transform((float(xd), float(yd)))
        frac = fig.transFigure.inverted().transform(disp)
        return float(frac[0]), float(frac[1])

    # Ensure leaves (distance 0) sit on the LEFT and the root on the right,
    # regardless of the scipy orientation convention.
    if _to_fig(0.0, 5.0)[0] > _to_fig(dist_max, 5.0)[0]:
        ax_dend.invert_xaxis()

    # scipy spaces leaves at y = 5, 15, 25, …  (leaf i at 5 + 10·i)
    leaf_spacing_fig = abs(_to_fig(0.0, 15.0)[1] - _to_fig(0.0, 5.0)[1])
    leaf_ph_fig = 0.9 * leaf_spacing_fig                 # square in inches
    leaf_pw_fig = leaf_ph_fig * fh / fw

    # leaf-panel strip sits just left of the tree (distance-0 edge)
    x0_fig     = _to_fig(0.0, 5.0)[0]
    leaf_right = x0_fig - gap
    leaf_left  = leaf_right - leaf_pw_fig

    # ------------------------------------------------------------------
    # 9. Leaf panels + neuron names on the left spine
    # ------------------------------------------------------------------
    leaf_axes = []
    placed = []   # (xc, yc, half_w, half_h) fig-fraction boxes for overlap tests
    for i, orig_idx in enumerate(dend["leaves"]):
        y_data = 5.0 + 10.0 * i
        _, yf_c = _to_fig(0.0, y_data)
        ax_leaf = fig.add_axes([leaf_left, yf_c - leaf_ph_fig / 2,
                                leaf_pw_fig, leaf_ph_fig])
        for sp in ax_leaf.spines.values():
            sp.set_visible(False)
        ax_leaf.set_xticks([])
        ax_leaf.set_yticks([])

        field2d = type(ref_fld)(leaf_imgs[orig_idx][None], ps=ps_ref, qs=qs_ref)
        draw_field_interp(ax_leaf, field2d, norm=leaf_norms[orig_idx], cmap=cmap, axis_lim=axis_lim)
        ax_leaf.set_ylabel(neurons[orig_idx], fontsize=9, rotation=0,
                           ha="right", va="center", labelpad=6)
        leaf_axes.append(ax_leaf)
        placed.append((leaf_left + leaf_pw_fig / 2, yf_c,
                       leaf_pw_fig / 2, leaf_ph_fig / 2))

    # ------------------------------------------------------------------
    # 10. Internal-node panels — size ∝ cosine distance, non-overlapping
    #
    # Horizontal U-shape: the vertical connector sits at x = merge distance,
    # spanning y = icoord[1]..icoord[2]; node centre = (distance, y_mid).
    # Desired panel size grows with distance; a greedy pass (largest first)
    # shrinks any panel that would overlap an already-placed one.
    # ------------------------------------------------------------------
    height_to_shapes = defaultdict(list)
    for ic, dc in zip(dend["icoord"], dend["dcoord"]):
        y_mid = (ic[1] + ic[2]) / 2.0
        dist  = float(dc[1])
        height_to_shapes[dist].append((y_mid, dist))
    for h in height_to_shapes:
        height_to_shapes[h].sort(key=lambda s: s[0])

    height_to_rows = defaultdict(list)
    for j, row in enumerate(Z):
        height_to_rows[float(row[2])].append(j)
    for h in height_to_rows:
        height_to_rows[h].sort()

    node_shapes = {}   # internal_node_id -> (y_mid, dist)
    for h, shapes in height_to_shapes.items():
        rows = height_to_rows.get(h, [])
        for shape, row_j in zip(shapes, rows):
            node_shapes[n_leaves + row_j] = shape

    # collect candidates, place biggest (deepest) first so smaller ones yield
    candidates = []
    for j in range(len(Z)):
        node_id = n_leaves + j
        if node_id not in node_shapes:
            continue
        leaf_idx = _get_node_leaves(Z, n_leaves, node_id)
        if len(leaf_idx) < min_members:
            continue
        y_mid, dist = node_shapes[node_id]
        candidates.append((dist, y_mid, node_id, leaf_idx))
    candidates.sort(key=lambda c: c[0], reverse=True)

    for dist, y_mid, node_id, leaf_idx in candidates:
        frac = dist / dist_max
        ph = leaf_ph_fig * (1.0 + node_growth * frac)    # desired height ∝ distance
        pw = ph * fh / fw                                 # square in inches
        hw, hh = pw / 2.0, ph / 2.0

        xf_c, yf_c = _to_fig(dist, y_mid)

        # greedy shrink so this panel does not overlap any already-placed panel
        for (pxc, pyc, phw, phh) in placed:
            dx = abs(xf_c - pxc)
            dy = abs(yf_c - pyc)
            if dx < hw + phw and dy < hh + phh:
                s_x = (dx - phw) / hw if hw > 0 else 0.0
                s_y = (dy - phh) / hh if hh > 0 else 0.0
                s = max(0.0, min(1.0, max(s_x, s_y)))
                hw *= s
                hh *= s
        if hw <= 1e-4 or hh <= 1e-4:
            continue

        # Centroid: straight mean of actual channel images — fully interpretable
        centroid_img = np.mean([leaf_imgs[i] for i in leaf_idx], axis=0)
        node_fld = SynapticField(centroid_img[None].astype(np.float32),
                                 ps=ps_ref, qs=qs_ref)

        xf0 = float(np.clip(xf_c - hw, 0.0, 1.0 - 2 * hw))
        yf0 = float(np.clip(yf_c - hh, dend_b, dend_t - 2 * hh))
        ax_node = fig.add_axes([xf0, yf0, 2 * hw, 2 * hh])
        ax_node.patch.set_alpha(0.0)
        for sp in ax_node.spines.values():
            sp.set_visible(False)
        ax_node.set_xticks([])
        ax_node.set_yticks([])

        node_vals = centroid_img[centroid_img > 0]
        if node_vals.size > 0 and log_norm:
            from matplotlib.colors import LogNorm
            nv_min = float(np.percentile(node_vals, 2))
            nv_max = float(np.percentile(node_vals, 99.8))
            if nv_min <= 0:
                nv_min = float(node_vals.min())
            if nv_min >= nv_max:
                nv_max = nv_min + 1e-12
            node_norm = LogNorm(vmin=nv_min, vmax=nv_max)
        else:
            node_norm = norm

        draw_field_interp(ax_node, node_fld, norm=node_norm, cmap=cmap, axis_lim=axis_lim)
        placed.append((xf_c, yf_c, hw, hh))

    # ------------------------------------------------------------------
    # 11. Shared colourbar  (horizontal strip along the top)
    # ------------------------------------------------------------------
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(f"{channel} input (a.u.)", fontsize=13)
    cbar.ax.tick_params(labelsize=11)
    cbar.ax.xaxis.set_label_position("top")
    cbar.ax.xaxis.set_ticks_position("top")

    return fig, ax_dend, leaf_axes


def rf_svd_model(all_fields, neurons, sens_summary=None, normalize="gain",
                 var_threshold=0.90, k=None, mask=True):
    """Low-rank SVD reconstruction of the per-neuron sensitivity fields (a denoised all_fields).

    Builds the M x N sensitivity RF matrix, truncates the SVD to `k` components (or enough
    for >= var_threshold explained variance if k is None), and reconstructs each neuron's
    sensitivity field from that rank-k model. If the matrix was gain-normalized the
    reconstruction is multiplied back by the per-neuron gain so it is in sensitivity-
    probability units, directly comparable to the original fields.

    Returns (svd_fields, info):
      - svd_fields: dict neuron -> SynapticFields with a single 'all' frame holding the
        rank-k reconstruction on the full retinal grid. Because it has an 'all' key with one
        target frame, sensitivity_field / rf_matrix / plot_rf_taxonomy work on it directly
        (note: they clip to [0, 1], so any small negative SVD residuals read as 0).
      - info: dict with scores (M x k), Vt (k x n_keep basis), ev, k, keep, s.
    """
    from flywire_tools.connectome import SynapticField, SynapticFields
    X, keep = rf_matrix(all_fields, neurons, sens_summary, normalize=normalize, mask=mask)
    U, s, Vt = np.linalg.svd(np.asarray(X, float), full_matrices=False)
    ev = s ** 2 / np.sum(s ** 2)
    if k is None:
        k = int(np.searchsorted(np.cumsum(ev), var_threshold) + 1)
    k = max(1, min(int(k), len(s)))
    scores = U[:, :k] * s[:k]                       # (M, k)
    Xhat = scores @ Vt[:k]                           # (M, n_keep) rank-k reconstruction
    if normalize == "gain" and sens_summary is not None:
        g = np.asarray([sens_summary.loc[n, "median"] for n in neurons], float)[:, None]
        Xhat = Xhat * g                              # back to probability units
    ref = all_fields[neurons[0]]["all"]
    width, height = ref.shape[-2:]
    svd_fields = {}
    for i, neuron in enumerate(neurons):
        flat = np.zeros(width * height, dtype=np.float32)
        flat[keep] = Xhat[i]
        field = SynapticField(flat.reshape(width, height)[None], ps=ref.ps, qs=ref.qs)
        svd_fields[neuron] = SynapticFields({"all": field})
    info = dict(scores=scores, Vt=Vt[:k], ev=ev, k=k, keep=keep, s=s)
    return svd_fields, info


# ----------------------------------------------------------------------------------------
# Phase 4 -- global (position x type x channel) tensor + HOSVD missing-value imputation
# ----------------------------------------------------------------------------------------
def _mode_mult(T, M, mode):
    """Multiply tensor ``T`` by matrix ``M`` along the given ``mode`` (axis)."""
    T2 = np.moveaxis(T, mode, 0)
    shp = T2.shape
    out = M @ T2.reshape(shp[0], -1)
    out = out.reshape((M.shape[0],) + shp[1:])
    return np.moveaxis(out, 0, mode)


def _mode_basis(T, mode, rank):
    """Top-``rank`` left singular vectors of ``T``'s mode-``mode`` unfolding."""
    unf = np.moveaxis(T, mode, 0).reshape(T.shape[mode], -1)
    U, _, _ = np.linalg.svd(unf, full_matrices=False)
    r = max(1, min(int(rank), U.shape[1]))
    return U[:, :r]


def _pick_ranks(T, var_threshold=0.9, max_ranks=None):
    """Per-mode multilinear ranks capturing >= ``var_threshold`` of each unfolding's energy."""
    ranks = []
    for mode in range(T.ndim):
        unf = np.moveaxis(T, mode, 0).reshape(T.shape[mode], -1)
        s = np.linalg.svd(unf, compute_uv=False)
        ev = np.cumsum(s ** 2) / max(float(np.sum(s ** 2)), 1e-12)
        r = int(np.searchsorted(ev, var_threshold) + 1)
        r = max(1, min(r, T.shape[mode]))
        if max_ranks is not None and max_ranks[mode] is not None:
            r = min(r, max_ranks[mode])
        ranks.append(r)
    return tuple(ranks)


def truncated_hosvd(T, ranks):
    """Truncated HOSVD (De Lathauwer) reconstruction of a dense tensor at ``ranks``.

    Projects ``T`` onto the leading mode-wise singular subspaces and returns the
    low-multilinear-rank reconstruction. Returns (T_hat, factors).
    """
    factors = [_mode_basis(T, mode, ranks[mode]) for mode in range(T.ndim)]
    T_hat = T
    for mode, U in enumerate(factors):
        T_hat = _mode_mult(T_hat, U @ U.T, mode)
    return T_hat, factors


def build_channel_tensor(all_fields, neurons, channels=None, normalize=True):
    """Assemble the global (positions x types x channels) tensor + observed mask.

    For each neuron type and channel the per-clone frames are pooled with ``or_over_targets``
    into one spatial field, so each type is modelled as a pooling unit. With ``normalize=True``
    (default) the pooled value is the OR-over-targets probability; with ``normalize=False`` it
    is the raw sum over targets (synapse-weighted path counts).
    Positions active in >= 1 (type, channel) form the retinal support (N). An entry is
    OBSERVED where its pooled value > 0 (exactly what plot_field_grid / draw_hex render as
    coloured) and MISSING (to be imputed) where the channel is absent or the pooled reach is
    0 within the support.

    Returns a dict with:
      T (N, M, K) pooled values, observed (N, M, K bool), keep (W*H bool support mask),
      ps, qs (W, H) coordinate grids, coords (N, 2) per-position (p, q), neurons, channels,
      shape (W, H).
    """
    if channels is None:
        channels = list(STARTING_POINTS)
    neurons = list(neurons)
    ref = all_fields[neurons[0]]["all"]
    width, height = ref.shape[-2:]
    npix = width * height
    M, K = len(neurons), len(channels)
    full = np.zeros((npix, M, K), dtype=np.float64)
    for j, neuron in enumerate(neurons):
        flds = all_fields[neuron]
        for kk, ch in enumerate(channels):
            if ch in flds:
                full[:, j, kk] = or_over_targets(flds[ch], normalize=normalize).ravel()
    keep = (full > 0).any(axis=(1, 2))              # positions active in >= 1 (type, channel)
    T = full[keep]                                   # (N, M, K)
    observed = T > 0
    coords = np.stack([np.asarray(ref.ps).ravel()[keep],
                       np.asarray(ref.qs).ravel()[keep]], axis=1)
    return dict(T=T, observed=observed, keep=keep, ps=ref.ps, qs=ref.qs, coords=coords,
                neurons=neurons, channels=list(channels), shape=(width, height))


def _init_missing(T, observed, method="fieldmean", seed=0):
    """Initial fill for missing entries.

    method='fieldmean' (default, recommended): per-(type, channel) observed mean.
    method='cross_channel': the mean of the OTHER observed channels at the same
        (position, type) -- uses cross-channel correlation from the start.
    method='random': uniform noise in each field's observed range (diagnostic only; in
        practice this converges WORSE than the mean init, so it is not recommended).
    method='global': the single global observed mean.
    """
    X = np.array(T, dtype=float)
    _, M, K = T.shape
    obs_global = float(T[observed].mean()) if observed.any() else 0.0
    if method == "cross_channel":
        rowobs = observed.sum(2)                                  # (N, M)
        rowmean = np.where(rowobs > 0, (T * observed).sum(2) / np.maximum(rowobs, 1),
                           obs_global)
        for kk in range(K):
            col = observed[:, :, kk]
            X[:, :, kk] = np.where(col, T[:, :, kk], rowmean)
        return X
    if method == "random":
        rng = np.random.default_rng(seed)
        for j in range(M):
            for kk in range(K):
                col = observed[:, j, kk]
                miss = ~col
                if miss.any():
                    lo, hi = ((T[col, j, kk].min(), T[col, j, kk].max())
                              if col.any() else (0.0, obs_global))
                    X[miss, j, kk] = rng.uniform(lo, hi, size=int(miss.sum()))
        return X
    if method != "fieldmean":
        X[~observed] = obs_global
        return X
    for j in range(M):
        for kk in range(K):
            col = observed[:, j, kk]
            miss = ~col
            if miss.any():
                X[miss, j, kk] = float(T[col, j, kk].mean()) if col.any() else obs_global
    return X


def hosvd_impute(T, observed, ranks=None, var_threshold=0.9, max_ranks=None,
                 n_iter=50, tol=1e-4, init="fieldmean", clip=(0.0, 1.0), seed=0,
                 normalize=True):
    """Iterative HOSVD missing-value imputation (hard-EM).

    Missing entries (~observed) are initialised with the per-field mean, then repeatedly:
    fit a truncated HOSVD to the current tensor, replace ONLY the missing entries with the
    reconstruction (observed entries stay exact), and clip to ``clip`` -- until the missing
    update norm drops below ``tol`` or ``n_iter`` is reached. If ``ranks`` is None it is
    chosen per mode from ``var_threshold`` (optionally capped by ``max_ranks``).

    When ``normalize=True`` (default) each (type×channel) column is z-score normalised
    before imputation and the result is unnormalised back to probability units afterward.

    Returns (X_filled, info) where info has ranks, n_iter, deltas (missing update norm per
    iter), obs_rmse (RMSE on observed entries per iter) and the final factors.
    """
    T = np.asarray(T, float)
    observed = np.asarray(observed, bool)
    if normalize:
        T_z, col_means, col_stds = normalize_tensor(T, observed)
        X_z, info = hosvd_impute(T_z, observed, ranks=ranks, var_threshold=var_threshold,
                                 max_ranks=max_ranks, n_iter=n_iter, tol=tol, init=init,
                                 clip=None, seed=seed, normalize=False)
        return np.clip(unnormalize_tensor(X_z, col_means, col_stds), 0.0, 1.0), info
    miss = ~observed
    X = _init_missing(T, observed, method=init, seed=seed)
    if ranks is None:
        ranks = _pick_ranks(X, var_threshold, max_ranks=max_ranks)
    deltas, obs_rmse, factors = [], [], None
    for _ in range(n_iter):
        Xhat, factors = truncated_hosvd(X, ranks)
        if clip is not None:
            Xhat = np.clip(Xhat, clip[0], clip[1])
        prev = X[miss]
        newX = np.where(observed, T, Xhat)
        denom = float(np.linalg.norm(prev)) + 1e-12
        deltas.append(float(np.linalg.norm(newX[miss] - prev) / denom))
        obs_rmse.append(float(np.sqrt(np.mean((Xhat[observed] - T[observed]) ** 2)))
                        if observed.any() else np.nan)
        X = newX
        if deltas[-1] < tol:
            break
    info = dict(ranks=ranks, n_iter=len(deltas), deltas=deltas, obs_rmse=obs_rmse,
                factors=factors)
    return X, info


def hosvd_cv(T, observed, coords=None, holdout=0.15, ranks=None, var_threshold=0.9,
             max_ranks=None, seed=0, n_iter=30, tol=1e-4, init="fieldmean",
             focus_channels=None):
    """Hold-out cross-validation of HOSVD imputation vs simple baselines.

    Only OBSERVED (non-missing) entries are ever scored: a random fraction of them is hidden,
    imputed from the rest, and compared to the truth (RMSE / R2 / Pearson r). Baselines:
    per-(type,channel) field mean, cross-channel mean at the same position/type, and (if
    ``coords`` given) the nearest observed position. If ``focus_channels`` is given, an extra
    'r2_focus' is reported over just those channel indices (e.g. the sparse channels where
    imputation actually matters). Returns {method: metrics}.
    """
    T = np.asarray(T, float)
    observed = np.asarray(observed, bool)
    rng = np.random.default_rng(seed)
    obs_idx = np.argwhere(observed)
    n_hold = max(1, int(holdout * len(obs_idx)))
    held = obs_idx[rng.choice(len(obs_idx), size=n_hold, replace=False)]
    hn, hm, hk = held[:, 0], held[:, 1], held[:, 2]
    truth = T[hn, hm, hk]
    O_train = observed.copy()
    O_train[hn, hm, hk] = False
    foc = None if focus_channels is None else np.isin(hk, list(focus_channels))

    def _scores(pred):
        pred = np.asarray(pred, float)

        def _r2(mask):
            t, p = truth[mask], pred[mask]
            ss = float(np.sum((t - t.mean()) ** 2))
            return float(1 - np.sum((t - p) ** 2) / ss) if ss > 0 else np.nan

        r = (float(np.corrcoef(truth, pred)[0, 1])
             if truth.std() > 0 and pred.std() > 0 else np.nan)
        d = dict(rmse=float(np.sqrt(np.mean((pred - truth) ** 2))),
                 r2=_r2(np.ones(len(truth), bool)), corr=r, n=int(len(truth)))
        if foc is not None and foc.any():
            d["r2_focus"] = _r2(foc)
        return d

    global_mean = float(T[O_train].mean()) if O_train.any() else 0.0
    Xfill, info = hosvd_impute(T, O_train, ranks=ranks, var_threshold=var_threshold,
                               max_ranks=max_ranks, n_iter=n_iter, tol=tol, init=init)
    out = {"hosvd": _scores(Xfill[hn, hm, hk])}

    fm = np.array([T[O_train[:, m, kk], m, kk].mean() if O_train[:, m, kk].any()
                   else global_mean for _, m, kk in held])
    out["field_mean"] = _scores(fm)

    cc = np.empty(len(held))
    for i, (n, m, kk) in enumerate(held):
        row = O_train[n, m, :]
        cc[i] = T[n, m, row].mean() if row.any() else fm[i]
    out["cross_channel_mean"] = _scores(cc)

    if coords is not None:
        coords = np.asarray(coords, float)
        nn = np.empty(len(held))
        for i, (n, m, kk) in enumerate(held):
            cand = np.where(O_train[:, m, kk])[0]
            if cand.size == 0:
                nn[i] = fm[i]
            else:
                d = np.sum((coords[cand] - coords[n]) ** 2, axis=1)
                nn[i] = T[cand[np.argmin(d)], m, kk]
        out["spatial_nn"] = _scores(nn)

    info["held_out"] = int(len(held))
    return out, info


def hosvd_select_ranks(T, observed, rank_grid=None, holdout=0.15, seed=0,
                       focus_channels=None, n_iter=30, tol=1e-4, init="fieldmean"):
    """Pick multilinear ranks by hold-out CV, maximising observed-only recovery.

    The variance-threshold heuristic (``_pick_ranks``) reads energy off the mean-filled
    tensor and tends to UNDER-select rank, so the imputation collapses toward the per-field
    mean. This instead hides a fraction of the observed entries and picks the rank triple that
    best recovers them (highest held-out R2). If ``focus_channels`` is given (e.g. the sparse
    channels), selection is driven by recovery on just those channels. Returns (best_ranks,
    table) where ``table`` is a DataFrame of r2_all / r2_focus / rmse per candidate.
    """
    import pandas as pd
    T = np.asarray(T, float)
    observed = np.asarray(observed, bool)
    N, M, K = T.shape
    if rank_grid is None:
        rN = sorted({min(r, N) for r in (15, 30, 60)})
        rM = sorted({min(r, M) for r in (8, 14)})
        rK = sorted({min(r, K) for r in (5, K)})
        rank_grid = [(a, b, c) for a in rN for b in rM for c in rK]
    rng = np.random.default_rng(seed)
    obs_idx = np.argwhere(observed)
    held = obs_idx[rng.choice(len(obs_idx), size=max(1, int(holdout * len(obs_idx))),
                              replace=False)]
    hn, hm, hk = held[:, 0], held[:, 1], held[:, 2]
    truth = T[hn, hm, hk]
    O_train = observed.copy()
    O_train[hn, hm, hk] = False
    foc = (np.ones(len(held), bool) if focus_channels is None
           else np.isin(hk, list(focus_channels)))

    def _r2(mask, pred):
        t, p = truth[mask], np.asarray(pred)[mask]
        ss = float(np.sum((t - t.mean()) ** 2))
        return float(1 - np.sum((t - p) ** 2) / ss) if ss > 0 else np.nan

    rows = []
    for r in rank_grid:
        Xf, _ = hosvd_impute(T, O_train, ranks=r, n_iter=n_iter, tol=tol, init=init)
        pred = Xf[hn, hm, hk]
        rows.append(dict(ranks=r, r2_all=_r2(np.ones(len(held), bool), pred),
                         r2_focus=_r2(foc, pred),
                         rmse=float(np.sqrt(np.mean((pred - truth) ** 2)))))
    table = pd.DataFrame(rows)
    key = "r2_focus" if focus_channels is not None else "r2_all"
    best = tuple(table.loc[table[key].idxmax(), "ranks"])
    return best, table


def svd_impute(T, observed, rank=None, var_threshold=0.9, n_iter=100, tol=1e-4,
               init="fieldmean", clip=(0.0, 1.0), seed=0, normalize=True):
    """Missing-value imputation by plain truncated-SVD matrix completion.

    Collapses the (N, M, K) tensor to an N x (M*K) matrix -- each column is one
    (type, channel) field over the retinal positions -- then runs the same hard-EM loop as
    ``hosvd_impute`` but with an ordinary truncated SVD: initialise missing entries, fit a
    rank-``rank`` SVD, replace ONLY the missing entries with the reconstruction, clip, repeat.
    Unlike the HOSVD this does not separate the type and channel modes; the collapsed column
    space is free (a single shared spatial basis with per-(type,channel) loadings). If
    ``rank`` is None it is taken from the cumulative singular-value energy (>= var_threshold)
    of the mean-filled matrix.

    When ``normalize=True`` (default) each (type×channel) column is z-score normalised
    before imputation and the result is unnormalised back to probability units afterward.

    Returns (X_filled (N, M, K), info).
    """
    T = np.asarray(T, float)
    observed = np.asarray(observed, bool)
    if normalize:
        T_z, col_means, col_stds = normalize_tensor(T, observed)
        X_z, info = svd_impute(T_z, observed, rank=rank, var_threshold=var_threshold,
                               n_iter=n_iter, tol=tol, init=init,
                               clip=None, seed=seed, normalize=False)
        return np.clip(unnormalize_tensor(X_z, col_means, col_stds), 0.0, 1.0), info
    N, M, K = T.shape
    A = T.reshape(N, M * K)
    OA = observed.reshape(N, M * K)
    X = _init_missing(T, observed, method=init, seed=seed).reshape(N, M * K)
    miss = ~OA
    if rank is None:
        s = np.linalg.svd(X, compute_uv=False)
        ev = np.cumsum(s ** 2) / max(float(np.sum(s ** 2)), 1e-12)
        rank = int(np.searchsorted(ev, var_threshold) + 1)
    rank = max(1, min(int(rank), min(A.shape)))
    deltas, obs_rmse = [], []
    for _ in range(n_iter):
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        Xhat = (U[:, :rank] * s[:rank]) @ Vt[:rank]
        if clip is not None:
            Xhat = np.clip(Xhat, clip[0], clip[1])
        prev = X[miss]
        newX = np.where(OA, A, Xhat)
        deltas.append(float(np.linalg.norm(newX[miss] - prev) / (np.linalg.norm(prev) + 1e-12)))
        obs_rmse.append(float(np.sqrt(np.mean((Xhat[OA] - A[OA]) ** 2))) if OA.any() else np.nan)
        X = newX
        if deltas[-1] < tol:
            break
    info = dict(rank=rank, n_iter=len(deltas), deltas=deltas, obs_rmse=obs_rmse)
    return X.reshape(N, M, K), info


def svd_select_rank(T, observed, ranks=None, holdout=0.15, seed=0, focus_channels=None,
                    n_iter=60, tol=1e-4, init="fieldmean"):
    """Pick the truncated-SVD rank for ``svd_impute`` by hold-out CV (observed-only).

    Hides a fraction of the observed (positive) entries and returns the rank that best
    recovers them (highest held-out R2), focused on ``focus_channels`` if given. Returns
    (best_rank, table) with r2_all / r2_focus / rmse per candidate rank.
    """
    import pandas as pd
    T = np.asarray(T, float)
    observed = np.asarray(observed, bool)
    N, M, K = T.shape
    if ranks is None:
        ranks = [r for r in (3, 5, 8, 10, 15, 20, 30) if r <= min(N, M * K)]
    rng = np.random.default_rng(seed)
    obs = np.argwhere(observed & (T > 0))
    held = obs[rng.choice(len(obs), max(1, int(holdout * len(obs))), replace=False)]
    hn, hm, hk = held[:, 0], held[:, 1], held[:, 2]
    truth = T[hn, hm, hk]
    foc = (np.ones(len(held), bool) if focus_channels is None
           else np.isin(hk, list(focus_channels)))
    O_tr = observed.copy()
    O_tr[hn, hm, hk] = False

    def _r2(mask, pred):
        tt, pp = truth[mask], np.asarray(pred)[mask]
        ss = float(np.sum((tt - tt.mean()) ** 2))
        return float(1 - np.sum((tt - pp) ** 2) / ss) if ss > 0 else np.nan

    rows = []
    for r in ranks:
        Xf, _ = svd_impute(T, O_tr, rank=r, n_iter=n_iter, tol=tol, init=init)
        pred = Xf[hn, hm, hk]
        rows.append(dict(rank=int(r), r2_all=_r2(np.ones(len(held), bool), pred),
                         r2_focus=_r2(foc, pred),
                         rmse=float(np.sqrt(np.mean((pred - truth) ** 2)))))
    table = pd.DataFrame(rows)
    key = "r2_focus" if focus_channels is not None else "r2_all"
    best = int(table.loc[table[key].idxmax(), "rank"])
    return best, table


def tensor_to_fields(T, keep, ps, qs, neurons, channels, add_all=True, combine="or"):
    """Rebuild an all_fields-style dict from a (N, M, K) tensor for plotting.

    Each (type, channel) becomes a single-frame SynapticField on the full W x H grid
    (positions outside ``keep`` are 0). If ``add_all`` an aggregated 'all' channel is added
    so plot_field_grid works with its default channel list: ``combine='or'`` (default) uses
    the probability OR over channels (assumes values in [0, 1]); ``combine='sum'`` adds the
    channels (for raw synapse-count fields). Returns dict type -> SynapticFields.
    """
    from flywire_tools.connectome import SynapticField, SynapticFields
    width, height = np.asarray(ps).shape[-2:]
    npix = width * height
    T = np.asarray(T, float)
    out = {}
    for j, neuron in enumerate(neurons):
        fields, chan_imgs = {}, []
        for kk, ch in enumerate(channels):
            flat = np.zeros(npix, dtype=np.float32)
            flat[keep] = T[:, j, kk]
            img = flat.reshape(width, height)
            fields[ch] = SynapticField(img[None], ps=ps, qs=qs)
            chan_imgs.append(img if combine == "sum" else np.clip(img, 0.0, 1.0))
        if add_all:
            if combine == "sum":
                all_img = np.sum(chan_imgs, axis=0).astype(np.float32)
            else:
                none_reach = np.ones((width, height), dtype=np.float64)
                for img in chan_imgs:
                    none_reach *= (1.0 - img)
                all_img = (1.0 - none_reach).astype(np.float32)
            fields["all"] = SynapticField(all_img[None], ps=ps, qs=qs)
        out[neuron] = SynapticFields(fields)
    return out


# ---------------------------------------------------------------------------
# Phase 4b -- 2D-structure-preserving completion (W x H x types x channels)
# ---------------------------------------------------------------------------

def build_channel_tensor_2d(all_fields, neurons, channels=None, normalize=True):
    """Assemble the (W x H x types x channels) tensor + observed mask, keeping the 2D grid.

    Unlike :func:`build_channel_tensor` (which collapses the retinal grid to an unordered
    list of support positions), this keeps the two retinal axes as *separate* tensor modes so
    a downstream HOSVD can factorise along x and y independently. Geometry is taken from the
    first requested channel -- the ``'all'`` field is never referenced. Values are pooled over
    targets with ``or_over_targets`` (probabilities if ``normalize`` else raw counts).

    Returns dict: T (W, H, M, K), observed (bool), support (W, H bool = active in >=1 slice),
    ps, qs, neurons, channels, shape.
    """
    if channels is None:
        channels = list(STARTING_POINTS)
    neurons = list(neurons)
    ref = all_fields[neurons[0]][channels[0]]          # geometry from a real channel, not 'all'
    width, height = ref.shape[-2:]
    M, K = len(neurons), len(channels)
    T = np.zeros((width, height, M, K), dtype=np.float64)
    for j, neuron in enumerate(neurons):
        flds = all_fields[neuron]
        for kk, ch in enumerate(channels):
            if ch in flds:
                T[:, :, j, kk] = or_over_targets(flds[ch], normalize=normalize)
    observed = T > 0
    support = observed.any(axis=(2, 3))                # (W, H) positions active in >=1 slice
    return dict(T=T, observed=observed, support=support, ps=ref.ps, qs=ref.qs,
                neurons=neurons, channels=list(channels), shape=(width, height))


def hosvd_impute_2d(T, observed, ranks=None, var_threshold=0.9, max_ranks=None,
                    n_iter=50, tol=1e-4, clip=None, zscore=True):
    """HOSVD missing-value imputation that KEEPS the 2D retinal grid (no x-y collapse).

    ``T`` is a (W, H, types, channels) tensor and ``observed`` its bool mask. Because the two
    spatial axes are separate tensor modes, the low-rank fit factorises along x and y
    independently -- a separable spatial basis that is far smoother than the unordered-position
    basis of the collapsed (positions x types x channels) version.

    Each (type, channel) field is z-scored across its OBSERVED spatial positions (so the fit
    captures shape, not magnitude) when ``zscore``. Missing entries -- including every position
    outside the field support, which is unobserved in all slices and therefore never
    influences the fit -- are initialised to the per-field mean, then a hard-EM loop fits a
    truncated 4-mode HOSVD, overwrites only the missing entries (observed stay exact) and
    optionally clips, until the missing-update norm drops below ``tol`` or ``n_iter`` is hit.
    Returns (X_filled, info).
    """
    T = np.asarray(T, float)
    observed = np.asarray(observed, bool)
    W, H, M, K = T.shape
    means = np.zeros((M, K))
    stds = np.ones((M, K))
    X = np.zeros_like(T)
    for j in range(M):
        for kk in range(K):
            obs = observed[:, :, j, kk]
            vals = T[:, :, j, kk][obs]
            mu = float(vals.mean()) if vals.size else 0.0
            sd = float(vals.std()) if vals.size > 1 else 1.0
            means[j, kk], stds[j, kk] = mu, (sd if sd > 1e-10 else 1.0)
            if zscore:
                X[:, :, j, kk] = np.where(obs, (T[:, :, j, kk] - mu) / stds[j, kk], 0.0)
            else:
                X[:, :, j, kk] = np.where(obs, T[:, :, j, kk], mu)
    miss = ~observed
    if ranks is None:
        ranks = _pick_ranks(X, var_threshold, max_ranks=max_ranks)
    deltas = []
    for _ in range(n_iter):
        Xhat, _factors = truncated_hosvd(X, ranks)
        prev = X[miss]
        newX = np.where(observed, X, Xhat)             # observed entries stay exact
        d = float(np.linalg.norm(newX[miss] - prev) / (np.linalg.norm(prev) + 1e-12))
        deltas.append(d)
        X = newX
        if d < tol:
            break
    if zscore:
        X = X * stds[None, None, :, :] + means[None, None, :, :]
    if clip is not None:
        X = np.clip(X, clip[0], clip[1])
    return X, dict(ranks=ranks, n_iter=len(deltas), deltas=deltas)


def tensor_to_fields_2d(T, ps, qs, neurons, channels, support=None, add_all=True,
                        combine="sum"):
    """Rebuild an all_fields-style dict from a (W, H, types, channels) tensor for plotting.

    ``support`` (W, H bool) masks out-of-field positions back to 0 (their imputed values are
    discarded). ``combine='sum'`` builds the 'all' channel as the channel sum (raw counts);
    ``combine='or'`` uses the probability OR. Returns dict type -> SynapticFields.
    """
    from flywire_tools.connectome import SynapticField, SynapticFields
    W, H, M, K = T.shape
    out = {}
    for j, neuron in enumerate(neurons):
        fields, chan_imgs = {}, []
        for kk, ch in enumerate(channels):
            img = np.array(T[:, :, j, kk], dtype=np.float32)
            if support is not None:
                img = np.where(support, img, 0.0).astype(np.float32)
            fields[ch] = SynapticField(img[None], ps=ps, qs=qs)
            chan_imgs.append(img)
        if add_all:
            if combine == "sum":
                all_img = np.sum(chan_imgs, axis=0).astype(np.float32)
            else:
                none_reach = np.ones((W, H), dtype=np.float64)
                for c in chan_imgs:
                    none_reach *= (1.0 - np.clip(c, 0.0, 1.0))
                all_img = (1.0 - none_reach).astype(np.float32)
            fields["all"] = SynapticField(all_img[None], ps=ps, qs=qs)
        out[neuron] = SynapticFields(fields)
    return out


# ---------------------------------------------------------------------------
# Phase 4c -- L5-anchored reduced-rank regression (channel = linear op of L5)
# ---------------------------------------------------------------------------

def _support_matrix(all_fields, neurons, channels, normalize=False):
    """(P support-pixels x M types x K channels) matrix + masks/geometry.

    Missing is defined PER CHANNEL: an entry is observed where the sum over that type's
    target cells (``or_over_targets(normalize=False)``) is > 0. ``support`` is the full-retina
    mask (active in >= 1 channel/type); only support pixels are kept as rows.
    """
    tens = build_channel_tensor_2d(all_fields, neurons, channels=channels, normalize=normalize)
    T4, support = tens["T"], tens["support"]
    W, H, M, K = T4.shape
    sup_idx = np.flatnonzero(support.ravel())
    Xp = T4.reshape(W * H, M, K)[sup_idx]
    obsP = Xp > 0
    ps = np.asarray(tens["ps"]).ravel()[sup_idx]
    qs = np.asarray(tens["qs"]).ravel()[sup_idx]
    return dict(Xp=Xp, obsP=obsP, sup_idx=sup_idx, P=sup_idx.size, shape=(W, H),
                support=support, ps=ps, qs=qs, ps_grid=tens["ps"], qs_grid=tens["qs"],
                neurons=list(neurons), channels=list(channels))


def build_smooth_basis(ps, qs, n_centers=6, scale=1.5, include_const=True, prune=1e-3):
    """Smooth 2D Gaussian-RBF basis over the support pixels: (P x r) columns.

    Places an ``n_centers`` x ``n_centers`` grid of RBFs across the (ps, qs) extent, width
    ``scale`` x grid-spacing. Columns with no nearby support are pruned. A constant column is
    prepended when ``include_const`` (captures the DC / scaled-copy term).
    """
    ps = np.asarray(ps, float); qs = np.asarray(qs, float)
    cx = np.linspace(ps.min(), ps.max(), n_centers)
    cy = np.linspace(qs.min(), qs.max(), n_centers)
    CX, CY = np.meshgrid(cx, cy)
    centers = np.stack([CX.ravel(), CY.ravel()], axis=1)
    dx = (ps.max() - ps.min()) / max(n_centers - 1, 1)
    sigma = max(scale * dx, 1e-6)
    d2 = (ps[:, None] - centers[None, :, 0]) ** 2 + (qs[:, None] - centers[None, :, 1]) ** 2
    B = np.exp(-d2 / (2 * sigma ** 2))
    B = B[:, B.max(axis=0) > prune]
    if include_const:
        B = np.hstack([np.ones((B.shape[0], 1)), B])
    return B


def _make_l5_basis(sm, basis="svd", anchor="L5", rank=None, var_threshold=0.95,
                   n_centers=6, rbf_scale=1.5, log=True):
    """Build the spatial basis V (P x r) and per-neuron L5 coefficients a (r x M)."""
    channels = sm["channels"]
    ci = channels.index(anchor)
    Xp = sm["Xp"]
    A = (np.log1p(Xp[:, :, ci]) if log else Xp[:, :, ci]).astype(float)   # (P, M) L5
    if basis == "svd":
        U, s, _ = np.linalg.svd(A, full_matrices=False)
        ev = s ** 2 / max(float(np.sum(s ** 2)), 1e-12)
        r = int(rank) if rank else int(np.searchsorted(np.cumsum(ev), var_threshold) + 1)
        r = max(1, min(r, U.shape[1]))
        V = U[:, :r]
    elif basis == "spline":
        V = build_smooth_basis(sm["ps"], sm["qs"], n_centers=n_centers, scale=rbf_scale)
    else:
        raise ValueError(f"unknown basis: {basis}")
    a = np.linalg.pinv(V) @ A                                            # (r, M) L5 in V
    return V, a, ci


def _l5_fit_predict(Xl, obs, ci, V, a, ridge=1e-2):
    """Fit channel operators C_c and predict every pixel. Xl is (P, M, K) in the fit space.

    For each channel c, solve  min_C  sum_n || (V C a_n - f_{n,c})_obs ||^2 + lam||C||^2
    (normal equations accumulated per neuron), then predict  V C a  for all neurons.
    Returns (pred (P, M, K), Cs list).
    """
    P, M, K = Xl.shape
    r = V.shape[1]
    pred = np.empty_like(Xl)
    Cs = []
    for kk in range(K):
        G = np.zeros((r * r, r * r)); b = np.zeros(r * r)
        for j in range(M):
            o = obs[:, j, kk]
            if not o.any():
                continue
            D = (V[o][:, :, None] * a[None, :, j][:, None, :]).reshape(int(o.sum()), r * r)
            G += D.T @ D
            b += D.T @ Xl[o, j, kk]
        lam = ridge * (np.trace(G) / (r * r) + 1e-12)
        C = (np.linalg.solve(G + lam * np.eye(r * r), b).reshape(r, r)
             if np.any(b) else np.zeros((r, r)))
        Cs.append(C)
        pred[:, :, kk] = V @ C @ a
    return pred, Cs


def l5_anchored_impute(all_fields, neurons, channels=None, anchor="L5", basis="svd",
                       rank=None, var_threshold=0.95, n_centers=6, rbf_scale=1.5,
                       ridge=1e-2, log=True, keep_observed=True, combine="sum"):
    """Impute per-channel missing pixels by regressing every channel on the L5 field.

    Each channel field is modelled as a linear operator applied to the neuron's L5 field in a
    reduced smooth spatial subspace V (``basis='svd'`` -> leading SVD modes of the L5 fields;
    ``basis='spline'`` -> fixed Gaussian-RBF surface). The operator is fit across neurons on
    the observed pixels, so entirely-absent channels are still predicted from that neuron's
    L5. Observed pixels are kept exact when ``keep_observed``. Returns (imputed_fields, info).
    """
    if channels is None:
        channels = list(STARTING_POINTS)
    sm = _support_matrix(all_fields, neurons, channels, normalize=False)
    Xl = (np.log1p(sm["Xp"]) if log else sm["Xp"].astype(float))
    V, a, ci = _make_l5_basis(sm, basis=basis, anchor=anchor, rank=rank,
                              var_threshold=var_threshold, n_centers=n_centers,
                              rbf_scale=rbf_scale, log=log)
    pred, Cs = _l5_fit_predict(Xl, sm["obsP"], ci, V, a, ridge=ridge)
    out = pred.copy()
    if keep_observed:
        out[sm["obsP"]] = Xl[sm["obsP"]]
    Ximp = np.clip(np.expm1(out) if log else out, 0.0, None)
    fields = _scatter_support_to_fields(Ximp, sm, add_all=True, combine=combine)
    info = dict(basis=basis, r=V.shape[1], ridge=ridge, Cs=Cs, sm=sm)
    return fields, info


def _scatter_support_to_fields(Ximp, sm, add_all=True, combine="sum"):
    """Rebuild an all_fields-style dict from a (P support-pixels x M x K) matrix."""
    from flywire_tools.connectome import SynapticField, SynapticFields
    W, H = sm["shape"]; sup_idx = sm["sup_idx"]
    ps_grid, qs_grid = sm["ps_grid"], sm["qs_grid"]
    P, M, K = Ximp.shape
    out = {}
    for j, neuron in enumerate(sm["neurons"]):
        fields, chan_imgs = {}, []
        for kk, ch in enumerate(sm["channels"]):
            flat = np.zeros(W * H, dtype=np.float32)
            flat[sup_idx] = Ximp[:, j, kk]
            img = flat.reshape(W, H)
            fields[ch] = SynapticField(img[None], ps=ps_grid, qs=qs_grid)
            chan_imgs.append(img)
        if add_all:
            if combine == "sum":
                all_img = np.sum(chan_imgs, axis=0).astype(np.float32)
            else:
                none_reach = np.ones((W, H), dtype=np.float64)
                for c in chan_imgs:
                    none_reach *= (1.0 - np.clip(c, 0.0, 1.0))
                all_img = (1.0 - none_reach).astype(np.float32)
            fields["all"] = SynapticField(all_img[None], ps=ps_grid, qs=qs_grid)
        out[neuron] = SynapticFields(fields)
    return out


def l5_anchored_cv(all_fields, neurons, channels=None, bases=("svd", "spline"), anchor="L5",
                   holdout=0.2, seed=0, log=True, ridge=1e-2, rank=None, var_threshold=0.95,
                   n_centers=6, rbf_scale=1.5):
    """Hold-out comparison of L5-anchored bases: hide observed non-anchor pixels and score.

    L5 (the anchor) is kept fully observed so the basis V and coefficients a are unchanged; a
    random ``holdout`` fraction of the *other* channels' observed pixels is hidden, the
    operators are refit on the rest, and recovery of the hidden pixels is scored in raw-count
    units (RMSE / R2 / Pearson r). Returns a DataFrame, one row per basis.
    """
    import pandas as pd
    if channels is None:
        channels = list(STARTING_POINTS)
    sm = _support_matrix(all_fields, neurons, channels, normalize=False)
    Xl = (np.log1p(sm["Xp"]) if log else sm["Xp"].astype(float))
    ci = channels.index(anchor)
    obs = sm["obsP"]
    cand = np.argwhere(obs)
    cand = cand[cand[:, 2] != ci]                       # never hide the anchor channel
    rng = np.random.default_rng(seed)
    held = cand[rng.choice(len(cand), max(1, int(holdout * len(cand))), replace=False)]
    hp, hj, hk = held[:, 0], held[:, 1], held[:, 2]
    obs_tr = obs.copy(); obs_tr[hp, hj, hk] = False
    truth = np.expm1(Xl[hp, hj, hk]) if log else Xl[hp, hj, hk]

    rows = []
    for basis in bases:
        V, a, _ = _make_l5_basis(sm, basis=basis, anchor=anchor, rank=rank,
                                 var_threshold=var_threshold, n_centers=n_centers,
                                 rbf_scale=rbf_scale, log=log)
        pred, _ = _l5_fit_predict(Xl, obs_tr, ci, V, a, ridge=ridge)
        p = np.clip(np.expm1(pred[hp, hj, hk]) if log else pred[hp, hj, hk], 0.0, None)
        ss = float(np.sum((truth - truth.mean()) ** 2))
        r2 = float(1 - np.sum((truth - p) ** 2) / ss) if ss > 0 else np.nan
        corr = (float(np.corrcoef(truth, p)[0, 1])
                if truth.std() > 0 and p.std() > 0 else np.nan)
        rows.append(dict(basis=basis, r=V.shape[1],
                         rmse=float(np.sqrt(np.mean((p - truth) ** 2))),
                         r2=r2, corr=corr, n_held=int(len(truth))))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Phase 4 — Gappy POD (non-iterative masked least-squares reconstruction)
# ---------------------------------------------------------------------------

def normalize_tensor(T, observed):
    """Z-score each (type×channel) field across N positions using only the observed values.

    Mirrors ``normalize='zscore'`` in :func:`rf_matrix` (Phase 3): centres and scales each
    field so the decomposition captures *shape* rather than *magnitude*.  Missing entries are
    set to 0 after normalisation (they are replaced by the EM / Gappy-POD fill anyway).

    Parameters
    ----------
    T        : ndarray (N, M, K)   raw tensor
    observed : ndarray (N, M, K)   bool mask

    Returns
    -------
    T_z       : ndarray (N, M, K)   z-scored tensor
    col_means : ndarray (M*K,)      per-column means  (observed pixels only)
    col_stds  : ndarray (M*K,)      per-column stds   (observed pixels only; ≥ 1e-10)
    """
    N = T.shape[0]
    A  = np.asarray(T, float).reshape(N, -1)       # (N, M*K)
    OA = np.asarray(observed, bool).reshape(N, -1)
    col_means = np.zeros(A.shape[1])
    col_stds  = np.ones(A.shape[1])
    for c in range(A.shape[1]):
        obs_vals = A[OA[:, c], c]
        if obs_vals.size > 1:
            col_means[c] = obs_vals.mean()
            col_stds[c]  = max(obs_vals.std(), 1e-10)
    T_z = ((A - col_means) / col_stds).reshape(T.shape)
    T_z[~np.asarray(observed, bool)] = 0.0
    return T_z, col_means, col_stds


def unnormalize_tensor(T_z, col_means, col_stds):
    """Reverse :func:`normalize_tensor`: multiply by col_stds and add col_means.

    Parameters
    ----------
    T_z       : ndarray (N, M, K)   normalised tensor
    col_means : ndarray (M*K,)      means returned by normalize_tensor
    col_stds  : ndarray (M*K,)      stds  returned by normalize_tensor

    Returns
    -------
    T : ndarray (N, M, K)   tensor in original units
    """
    N = T_z.shape[0]
    A_z = np.asarray(T_z, float).reshape(N, -1)
    return (A_z * col_stds + col_means).reshape(T_z.shape)


def build_channel_max_basis(T):
    """Per-type channel-max basis: pixel-wise max over the K channels for each type.

    Parameters
    ----------
    T : ndarray (N, M, K)   raw tensor (zeros where missing / unobserved)

    Returns
    -------
    basis : ndarray (N, M)  each column is the max-over-channels spatial profile for one type
    """
    return T.max(axis=2).astype(float)          # (N, M)


def gappy_pod_impute(T, observed, basis_mat, rank, clip=True):
    """Non-iterative Gappy-POD imputation via masked least-squares.

    Build a truncated SVD basis from ``basis_mat`` (shape N × C), then for every
    (type, channel) column that has missing entries solve the masked least-squares
    problem

        α̂ = argmin_α  ‖ U_obs α − f_obs ‖²

    where U_obs / f_obs are restricted to the observed rows, and back-project
    f̂ = U α̂ to fill the gaps.  Observed entries are kept exact.

    Parameters
    ----------
    T         : ndarray (N, M, K)   raw (or z-scored) tensor
    observed  : ndarray (N, M, K)   bool mask (True = observed / nonzero)
    basis_mat : ndarray (N, C)       pre-built basis columns
    rank      : int                  number of leading singular vectors to use
    clip      : bool                 clip filled values to [0, 1] (default True);
                                     set False when T is z-scored (clip after unnormalising)

    Returns
    -------
    X_filled : ndarray (N, M, K)   imputed tensor (observed entries unchanged)
    info     : dict                 rank, basis_shape (C), n_filled, obs_rmse
    """
    U, _, _ = np.linalg.svd(basis_mat, full_matrices=False)
    Ur = U[:, :rank]                                    # (N, rank)

    A  = np.asarray(T,        float).reshape(T.shape[0], -1)   # (N, M*K)
    OA = np.asarray(observed, bool ).reshape(T.shape[0], -1)   # (N, M*K)
    X  = A.copy()

    n_filled = 0
    for c in range(A.shape[1]):
        obs  = OA[:, c]
        miss = ~obs
        if not miss.any():
            continue                   # fully observed — nothing to do
        if obs.sum() < rank:
            continue                   # too few observations to fit; leave as-is
        coef, *_ = np.linalg.lstsq(Ur[obs], A[obs, c], rcond=None)
        fill = (Ur @ coef)[miss]
        X[miss, c] = np.clip(fill, 0.0, 1.0) if clip else fill
        n_filled += int(miss.sum())

    X_filled = X.reshape(T.shape)
    resid = X_filled[observed] - T[observed]
    info = dict(
        rank=rank,
        basis_shape=basis_mat.shape,
        n_filled=n_filled,
        obs_rmse=float(np.sqrt(np.mean(resid ** 2))),
    )
    return X_filled, info


def gappy_pod_per_type(T, observed, var_threshold=0.90, clip=True, normalize=True):
    """Per-type Gappy POD: build a basis *within each type's own footprint* from
    its observed channels and fill its missing channels by masked least-squares.

    Unlike the global approach this never generalises across neuron types — it
    exploits purely **inter-channel spatial correlation** within each type.

    When ``normalize=True`` (default) each (type×channel) column is z-score normalised
    before imputation and the result is unnormalised back to probability units afterward.

    Parameters
    ----------
    T             : ndarray (N, M, K)   raw tensor (zeros where missing)
    observed      : ndarray (N, M, K)   bool mask (True = observed)
    var_threshold : float               keep enough SVD components to explain
                                        this fraction of variance in each type's
                                        observed-channel basis (default 0.90)

    Returns
    -------
    X_filled : ndarray (N, M, K)   imputed tensor (observed entries unchanged)
    info     : dict                per-type rank dict, n_filled, obs_rmse
    """
    T = np.asarray(T, float)
    observed = np.asarray(observed, bool)
    if normalize:
        T_z, col_means, col_stds = normalize_tensor(T, observed)
        X_z, info = gappy_pod_per_type(T_z, observed, var_threshold=var_threshold,
                                       clip=False, normalize=False)
        return np.clip(unnormalize_tensor(X_z, col_means, col_stds), 0.0, 1.0), info
    N, M, K = T.shape
    X = np.asarray(T, float).copy()
    O = np.asarray(observed, bool)
    n_filled = 0
    per_type_rank = {}

    for m in range(M):
        # positions within this type's retinal footprint
        in_support = O[:, m, :].any(axis=1)     # (N,)
        if not in_support.any():
            continue
        pos = np.where(in_support)[0]            # indices in [0, N)
        T_m = T[pos, m, :]                       # (n_m, K) raw values
        O_m = O[pos, m, :]                       # (n_m, K) observed mask

        # use channels that are >= 50% observed within the footprint as basis
        obs_frac_k = O_m.mean(axis=0)            # (K,)
        basis_chan = obs_frac_k >= 0.50
        if basis_chan.sum() < 1:
            continue

        basis = T_m[:, basis_chan].astype(float) # (n_m, K_basis)
        if basis.shape[1] == 0 or basis.shape[0] < 2:
            continue

        U, S, _ = np.linalg.svd(basis, full_matrices=False)
        if S.sum() > 0:
            ev_cum = np.cumsum(S ** 2) / np.sum(S ** 2)
            r = max(1, int(np.searchsorted(ev_cum, var_threshold) + 1))
        else:
            r = 1
        r = min(r, U.shape[1])
        per_type_rank[m] = r
        Ur = U[:, :r]                            # (n_m, r)

        for k in range(K):
            obs_k  = O_m[:, k]
            miss_k = ~obs_k
            if not miss_k.any():
                continue
            if obs_k.sum() < r:
                continue
            coef, *_ = np.linalg.lstsq(Ur[obs_k], T_m[obs_k, k], rcond=None)
            fill = (Ur @ coef)[miss_k]
            X[pos[miss_k], m, k] = np.clip(fill, 0.0, 1.0) if clip else fill
            n_filled += int(miss_k.sum())

    resid = X[observed] - T[observed]
    info = dict(
        per_type_rank=per_type_rank,
        n_filled=n_filled,
        obs_rmse=float(np.sqrt(np.mean(resid ** 2))),
    )
    return X, info
