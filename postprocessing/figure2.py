#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 2 (results) -- boundary-contribution decomposition of the NUMBER DENSITY n.

One combined figure grouped by SPECIES for a single quantity: the reactant A number
density n_A on top, the product B number density n_B below (the axial flux Gamma_x is the
companion figure, figure3.py). Contour STYLE (font, horizontal colorbar above, ticks,
spacing) is inherited from extended_figure1.py. The opening does not feed B, so
n_B^open = Gamma_x,B^open = 0 is omitted.

build(kind) draws the SAME two-species layout for either quantity:
  figure2.py -> build("n")     : n_A (top block)        + n_B (bottom block)
  figure3.py -> build("Gamma") : Gamma_x,A (top block)  + Gamma_x,B (bottom block)

Guide lines (per block, species-aware):
  n     : upper/lower -> the n maximum ridge along x (dash-dot-dot) ; total(+zoom) -> the
          n minimum line for reactant A / the n maximum line for product B (dashed)
  Gamma : upper/lower -> where Gamma_x changes sign along x (dash-dot-dot) ; no total line

  Output: figures/figure2.png

    python3 figure2.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.transforms import Bbox

import sys as _sys
from pathlib import Path as _Path

_CODE_DIR = str(_Path(__file__).resolve().parents[1])  # folder of inversion_analytic.py
if _CODE_DIR not in _sys.path:
    _sys.path.insert(0, _CODE_DIR)
import inversion_analytic as analytic  # noqa: E402
import analytic_case_studies as acs
import dsmc_postprocessing as post
import extended_figure1 as ef1   # reuse ALL contour style constants (font, colorbar, ticks...)
from figure_common import (FIGURE_DIR, LABEL_FONTFAMILY, build_ar_cfg, build_kn_cfg,
                           defer_label, style_contour_axes as _sca, add_colorbar as _acb,
                           group_bbox_frac, draw_faint_panel)

ef1.QUAD_FONT = 100
# >>> COL_GAP is the ONLY interior horizontal gap that sets the figure width, so it is the knob
#     for "use more of \textwidth without changing anything else". The printed contour and font
#     sizes depend only on (LaTeX width fraction) / fig_w, so scaling fig_w by the SAME factor as
#     the width fraction leaves both untouched and the extra space goes into this gap alone.
ef1.COL_GAP = 5.33
# >>> Gap (points) between the contour box and its tick NUMBERS. Raise these if the first x tick
#     of a zoom panel (e.g. "2.6") and the bottom y tick ("-0.5") crowd each other at the corner.
#     Prefer raising YTICK_PAD: it moves the y numbers LEFT and costs no vertical room, whereas
#     XTICK_PAD pushes the x numbers AND the x label down into the next row's colorbar label.
ef1.XTICK_PAD = 34
ef1.YTICK_PAD = 55
ef1.CBAR_LABEL_DY_ZOOM = 0.6
ef1.M_LEFT, ef1.M_RIGHT, ef1.M_TOP, ef1.M_BOTTOM = 2.30, 2.60, 1.00, 1.70
# >>> Per-row TEXT blocks (inches). These are fixed room reserved for text, so they do NOT shrink
#     when the contour gets smaller -- lower them together with CONTOUR_H to keep the figure from
#     becoming relatively taller. ef1 defaults: 2.50 and 2.50.
ef1.CBAR_TOP_PAD = 2.50   # >>> room above each colorbar for its tick numbers + label
ef1.XLABEL_H = 2.50       # >>> room below each contour for its x-axis label


# ============================================================
# Case selection
# ============================================================
STUDY = "AR"        # "AR" -> build_ar_cfg(CASE_VALUE) (Kn=100); "Kn" -> build_kn_cfg (AR=5)
CASE_VALUE = 5
OUT_NAME = "figure2.png"    # output image name under FIGURE_DIR
# Both species A and B are drawn in one figure; the plotted QUANTITY is chosen by build(kind):
# "n" -> this file (Fig. 2, number density) ; "Gamma" -> figure3.py (Fig. 3, axial flux).

# Boundary contribution keys (analytic API): jet=opening, top=upper wall, bottom=lower
# wall, end=trench base. Panel placement is decided per species in _draw_block.

# Style constants reused from extended_figure1 (ef1).
# >>> CONTOUR SIZE: height of every contour panel in inches. The panel width
#     follows as H * AR, so lowering this shrinks the contours in BOTH directions while the fonts
#     keep their point size -> contours get smaller relative to the text.
CONTOUR_H = 3.00
H = CONTOUR_H
FONT = ef1.QUAD_FONT
COL_GAP = 8.6                         # >>> gap between the two contribution columns.
                                      #     Wider than ef1.COL_GAP because every panel
                                      #     here keeps its own y-axis label.
XLABEL_H = ef1.XLABEL_H
M_LEFT, M_RIGHT, M_TOP, M_BOTTOM = ef1.M_LEFT, ef1.M_RIGHT, ef1.M_TOP, ef1.M_BOTTOM
CBAR_BLOCK = ef1.CBAR_ABOVE_PAD + ef1.CBAR_W + ef1.CBAR_TOP_PAD
DPI = ef1.QUAD_DPI
# Guide-line colours: minimum-type lines white, maximum-type and zero-crossing lines black.
WALL_LINE_COLOR = "black"                          # n maximum ridge / Gamma_x sign change
TOTAL_MIN_COLOR = "white"                          # reactant A total: n minimum
TOTAL_MAX_COLOR = "black"                           # product  B total: n maximum

# ---- USER KNOBS ---------------------------------------------------------------------
ROW_GAP = 4.0         # >>> VERTICAL gap (inches) between the two rows of a block
BLOCK_GAP = 2.94      # >>> VERTICAL spacing (inches) between the species A and species B blocks,
                      #     in excess of the block height. build() prints the "tint gap" that
                      #     this actually leaves between the two block tints -- tune on that.
SUM_GAP_PROBE = 18.0  # room left for the sum arrow in the FIRST (measuring) pass only; the
                      #     second pass replaces it with the gap the measurement calls for
LETTER_SIZE_FACTOR = 1.45    # >>> panel-letter size as a multiple of the tick-number size (FONT).
LETTER_PAD_IN = 0.0          # >>> push the letter further out of its top left corner (inches)

YLABEL_PAD = 60.0            # >>> gap (points) between the y-axis label y/(2h) and the y-tick
                             #     numbers. Raising it widens the PNG, so the figure then fills
                             #     more of \textwidth at UNCHANGED contour and font size.
ZOOM_LETTER_DX = -5.0
LINE_LW = ef1.MINLINE_LW   # dashed guide-line thickness
# >>> Per-AR zoom RANGE: left edge of the zoom window in x/(2h) (right edge = base = AR).
#     None -> use the automatically computed window. Smaller value -> wider zoom window.
ZOOM_XLEFT = {1: None, 3: None, 5: None}

# >>> Faint grouping panels behind the four boundary contributions and behind the total,
#     with an arrow between, to make "open+upper+lower+base = tot" visually explicit.
FAINT = True                    # True -> draw the faint grouping panels; False -> plain figure
FAINT_ALPHA = 0.20              # opacity of the contributions / total faint panels (higher = bolder)
# Per-block hues so the species A block and the species B block are easy to tell apart -- and
# so neither looks like a neutral gray. index 0 = species A block, index 1 = species B block.
# Within a block the contributions panel and the total panel share the block's hue (the total a
# bit deeper), so the "components -> tot" grouping reads as ONE colour family joined by the arrow.
FAINT_CONTRIB = ["#5B9BD5", "#66BB6A"]   # contributions panel:  A -> blue,      B -> green
FAINT_TOTAL = ["#2E6DB4", "#2E8B45"]     # total(+zoom) panel :  A -> deep blue, B -> deep green
FAINT_ARROW = "#000000"         # colour of the sum arrow and of the formula under it
FAINT_PAD_IN = 0.20             # padding (inches) around each faint group
# The contributions->resulting-field arrow, drawn in the gap between the two faint groups and
# labelled with SUM_TEXT. It is as long as that caption plus ARROW_PAD_CHARS characters on each
# side, and the gap itself is then set to that length, so the arrow can never reach into a panel.
ARROW_PAD_CHARS = 2.0           # >>> caption characters of arrow sticking out on EACH side
ARROW_CLEAR_IN = 0.35           # >>> clearance (inches) kept between the arrow and each group
ARROW_TAIL_W = 55.0             # >>> arrow SHAFT thickness (points)
ARROW_HEAD_W = 130.0            # >>> arrow HEAD width (points)
ARROW_HEAD_L = 90.0             # >>> arrow HEAD length (points)
SUM_SPACE = r"\;"               # >>> space between the summation sign and the term it acts on;
                                #     r"\," is thinner, r"\quad" wider
SUM_TEXT_SIZE_FACTOR = 1.0      # >>> caption size as a multiple of the tick-number size (FONT)
SUM_TEXT_DY_IN = 2.6            # >>> caption distance below the arrow (inches)
# Whole-block background tint: a light container around each block + its arrow.
FAINT_BLOCK_COLORS = ["#3B75AF", "#4CA64C"]   # species A block (blue), species B block (green)
FAINT_BLOCK_ALPHA = 0.08
FAINT_BLOCK_PAD_IN = 0.30       # padding for the whole-block tint (larger, so a rim shows)
FAINT_BLOCK_GAP_IN = 0.128      # >>> smallest white gap left between the species A and species B
                                #     block tints (inches). Their facing edges grow towards each
                                #     other to close a wider gap, but never past FAINT_BLOCK_PAD_IN,
                                #     so no block tint carries a deep empty margin on one side.

# >>> Zoom marker: the dashed box on the total panel (e) + the arrow pointing to the zoom
#     panel (f).
ZOOM_MARKER_COLOR = "#F39200"   # box + arrow colour
ZOOM_BOX_LW = 13.0               # >>> dashed box line thickness   (raise = thicker)
ZOOM_ARROW_LW = 13.0             # >>> e->f arrow SHAFT thickness   (raise = thicker)
ZOOM_ARROW_HEAD = 85.0          # >>> e->f arrow HEAD size
# >>> Line style of the box outline + arrow shaft (the solid arrow HEAD is unaffected).
#     dashed = (0, (6, 4)) [current] ; dotted = (0, (1, 2)) or ":" ; dash-dot = "-." .
#     For thick lines the gap in the tuple is a multiple of the linewidth, so widen the second
#     number if the dots/dashes look too crowded.
ZOOM_LINESTYLE = (0, (6, 4))
ZOOM_BOX_MARGIN_IN = 0.07       # >>> how far outside the contour the dashed box sits (inches)
ZOOM_ARROW_PAD_IN = 1.0         # >>> arrow tip stops this far above the colorbar label of the zoom
ZOOM_ARROW_XF = 0.72            # >>> where the arrow tip sits across the zoom (0 = left, 1 = right)


def _case_cfg():
    return build_ar_cfg(int(CASE_VALUE)) if STUDY == "AR" else build_kn_cfg(str(CASE_VALUE))


# SI units for the colorbar labels, merged into the "(x10^n ...)" parentheses (Nature style).
UNITS = {"n": r"m$^{-3}$", "Gamma": r"m$^{-2}$ s$^{-1}$"}


def _sym(kind, species, sup=None):
    s = species
    if kind == "n":
        return (rf"$n_{{\mathrm{{{s}}}}}$" if sup is None
                else rf"$n_{{\mathrm{{{s}}}}}^{{\mathrm{{{sup}}}}}$")
    return (rf"$\Gamma_{{\mathrm{{x}},\mathrm{{{s}}}}}$" if sup is None
            else rf"$\Gamma_{{\mathrm{{x}},\mathrm{{{s}}}}}^{{\mathrm{{{sup}}}}}$")


def _line_extreme(x, y, Z, want_max):
    """Per-row argmax/argmin of Z along x -> (x_line, y_line)."""
    ix = np.argmax(Z, axis=1) if want_max else np.argmin(Z, axis=1)
    return np.asarray(x)[ix], np.asarray(y)


def _line_sign(x, y, Z):
    """Per-row first sign change (zero crossing) of Z along x -> (x_line, y_line)."""
    x = np.asarray(x, float); y = np.asarray(y, float); Z = np.asarray(Z, float)
    xl = np.full(y.size, np.nan)
    for i in range(y.size):
        r = Z[i]
        s = np.where(r[:-1] * r[1:] < 0)[0]
        if s.size:
            k = int(s[0]); f0, f1 = r[k], r[k + 1]
            xl[i] = x[k] + (x[k + 1] - x[k]) * f0 / (f0 - f1)
    return xl, y


def _panel(fig, fw, fh, x, y, w, X, Y, Z, Lref, AR, cfg, sym, letter, *,
           show_xlabel, show_ylabel, show_xtl, show_ytl, zoom_left=None, norm=None, line=None,
           line_style="--", line_color="black", h_in=None, cbar_nticks=3, unit=None):
    hp = H if h_in is None else h_in
    ax = fig.add_axes([x / fw, y / fh, w / fw, hp / fh])
    if norm is None:
        lv, vmin, vmax = post.compare_level_values(Z, Z, cfg["levels"])
        norm = mcolors.Normalize(vmin, vmax)
    else:
        vmin, vmax = float(norm.vmin), float(norm.vmax)
        lv = np.linspace(vmin, vmax, int(cfg["levels"]))
    ax.contourf(X / Lref, Y / Lref, Z, levels=lv, cmap="jet", norm=norm)
    if line is not None:
        ax.plot(np.asarray(line[0]) / Lref, np.asarray(line[1]) / Lref, color=line_color, ls=line_style,
                lw=LINE_LW, zorder=20)
    _sca(ax, AR, cfg, zoom_left=zoom_left, show_xlabel=show_xlabel, show_ylabel=show_ylabel,
         show_yticklabels=show_ytl, show_xticklabels=show_xtl, box_lw=ef1.CONTOUR_BOX_LW,
         xtick_pad=ef1.XTICK_PAD, ytick_pad=ef1.YTICK_PAD)
    if show_ylabel:
        ax.yaxis.labelpad = YLABEL_PAD
    sm = ScalarMappable(norm=norm, cmap="jet"); sm.set_array([])
    clen = ef1.CBAR_LEN_FRAC * w
    cx = x + ef1.CBAR_XF * (w - clen)
    cy = y + hp + ef1.CBAR_ABOVE_PAD
    cax = fig.add_axes([cx / fw, cy / fh, clen / fw, ef1.CBAR_W / fh])
    _acb(fig, cax, sm, vmin, vmax, sym, True, cfg, orientation="horizontal",
         nticks=cbar_nticks, decimals=ef1.CBAR_DECIMALS, label_dy_in=ef1.CBAR_LABEL_DY, unit=unit)
    # Fig. 2 and 3 place their own letters in build(); a caller that passes a letter here
    # (Extended Data Figs. 4 and 7) gets it from figure_common.draw_deferred_labels instead.
    defer_label(ax, letter, size=FONT)
    return ax




def _zoom_marker(fig, ax_tot, AR, zoom_left, fw, fh, x_z, y_z, w_z):
    """Dashed box over the zoomed range of the resulting field + an arrow down to the zoom."""
    md = ZOOM_BOX_MARGIN_IN / H          # inches -> data units (H inches per unit in both axes)
    ax_tot.add_patch(Rectangle((zoom_left - md, -0.5 - md), (AR - zoom_left) + 2 * md, 1.0 + 2 * md,
                               transform=ax_tot.transData, fill=False, edgecolor=ZOOM_MARKER_COLOR,
                               linestyle=ZOOM_LINESTYLE, linewidth=ZOOM_BOX_LW, zorder=25,
                               clip_on=False))
    # from the lower right corner of the box down to just above the colorbar label of the zoom, so
    # the arrow crosses neither that label nor its tick numbers
    pA = np.asarray(fig.transFigure.inverted().transform(
        ax_tot.transData.transform((AR + md, -0.5 - md))))
    pB = np.array([(x_z + ZOOM_ARROW_XF * w_z) / fw,
                   (y_z + H + CBAR_BLOCK + ZOOM_ARROW_PAD_IN) / fh])
    tA, tB = fig.transFigure.transform(pA), fig.transFigure.transform(pB)
    u = (tB - tA) / float(np.hypot(*(tB - tA)))
    pBase = fig.transFigure.inverted().transform(tB - (ZOOM_ARROW_HEAD * fig.dpi / 72.0) * u)
    # dashed shaft up to the head base, then only the solid filled head
    fig.add_artist(FancyArrowPatch(pA, pBase, transform=fig.transFigure, arrowstyle="-",
                                   linestyle=ZOOM_LINESTYLE, linewidth=ZOOM_ARROW_LW,
                                   color=ZOOM_MARKER_COLOR, zorder=35))
    fig.add_artist(FancyArrowPatch(pBase, pB, transform=fig.transFigure, arrowstyle="-|>",
                                   mutation_scale=ZOOM_ARROW_HEAD, linewidth=ZOOM_ARROW_LW,
                                   color=ZOOM_MARKER_COLOR, zorder=36))


def _sum_label(kind, species):
    return rf"$\sum_{{b}} {SUM_SPACE} {_sym(kind, species).strip('$')}^{{b}}$"


def _corner_letter(fig, rend, axes, text, fw, fh):
    """Panel letter at the TOP LEFT CORNER of a panel as a whole, measured over everything it
    draws: contour, y ticks and label, colorbar and colorbar label."""
    bb = group_bbox_frac(fig, axes, rend)
    return fig.text(bb.x0 - LETTER_PAD_IN / fw, bb.y1 + LETTER_PAD_IN / fh, text,
                    ha="left", va="top", fontsize=FONT * LETTER_SIZE_FACTOR,
                    fontweight="bold", fontfamily=LABEL_FONTFAMILY, zorder=40)


def _draw_block(fig, fw, fh, y_top, kind, species, d, gap, letters):
    """One species: the boundary contributions on the left, the resulting field (and, for n, its
    near-base zoom) on the right. Returns the panel groups and the geometry the gap check needs."""
    sol, fields, contrib, cfg = d["sol"], d["fields"], d["contrib"], d["cfg"]
    X, Y = np.asarray(sol["X"]), np.asarray(sol["Y"])
    x1d, y1d = np.asarray(sol["x"]), np.asarray(sol["y"])
    Lref = 2.0 * float(sol["h"]); AR = float(sol["Lm"]) / Lref
    key = "n" if kind == "n" else "nu"
    has_zoom = (kind == "n")
    zl_over = ZOOM_XLEFT.get(int(round(AR)))
    zoom_left = float(zl_over) if zl_over is not None else round(float(d["zoom_xlim"][0]) / Lref, 1)

    W = H * AR
    Wz = H * (AR - zoom_left)
    left_w = 2.0 * W + COL_GAP
    row_pitch = H + CBAR_BLOCK + XLABEL_H + ROW_GAP
    x_c1, x_c2 = M_LEFT, M_LEFT + W + COL_GAP
    x_mid = M_LEFT + 0.5 * (left_w - W)          # a single contour centred in the left block
    x_r = M_LEFT + left_w + gap
    x_z = x_r + 0.5 * (W - Wz)                   # the zoom, centred under the resulting field
    y_bot = y_top - row_pitch
    y_tot = y_top if has_zoom else y_top - 0.5 * row_pitch   # no zoom -> centre it vertically

    Zt = np.asarray(fields[species][key])
    # resulting-field guide line: reactant A depletes -> minimum ; product B accumulates -> maximum
    line_tot = _line_extreme(x1d, y1d, Zt, want_max=(species == "B")) if kind == "n" else None
    tot_color = TOTAL_MIN_COLOR if species == "A" else TOTAL_MAX_COLOR
    bd_style = (0, (7, 3, 1, 3, 1, 3))           # wall guide lines, distinct from the "--" above

    P = dict(fw=fw, fh=fh, X=X, Y=Y, Lref=Lref, AR=AR, cfg=cfg, show_xlabel=True,
             show_ylabel=True, show_xtl=True, show_ytl=True, unit=UNITS[kind])
    # the opening does not feed the product, so its base contribution is centred instead
    quad = ([("top", "upper", x_c1, y_top), ("bottom", "lower", x_c2, y_top),
             ("jet", "open", x_c1, y_bot), ("end", "end", x_c2, y_bot)] if species == "A" else
            [("top", "upper", x_c1, y_top), ("bottom", "lower", x_c2, y_top),
             ("end", "end", x_mid, y_bot)])
    contrib_groups = []
    for k, sup, xq, yq in quad:
        Z = np.asarray(contrib[species][k][key])
        line = None
        if sup in ("upper", "lower"):
            line = (_line_extreme(x1d, y1d, Z, want_max=True) if kind == "n"
                    else _line_sign(x1d, y1d, Z))
        n0 = len(fig.axes)
        _panel(fig, x=xq, y=yq, w=W, Z=Z, sym=_sym(kind, species, sup), letter="",
               line=line, line_style=bd_style, line_color=WALL_LINE_COLOR, **P)
        contrib_groups.append((next(letters), fig.axes[n0:]))
    n0 = len(fig.axes)
    ax_tot = _panel(fig, x=x_r, y=y_tot, w=W, Z=Zt, sym=_sym(kind, species), letter="",
                    line=line_tot, line_color=tot_color, **P)
    total_groups = [(next(letters), fig.axes[n0:])]
    if has_zoom:
        m = X >= zoom_left * Lref
        znorm = mcolors.Normalize(float(np.nanmin(Zt[m])), float(np.nanmax(Zt[m])))
        n0 = len(fig.axes)
        _panel(fig, x=x_z, y=y_bot, w=Wz, Z=Zt, sym=_sym(kind, species), letter="",
               zoom_left=zoom_left, norm=znorm, line=line_tot, line_color=tot_color,
               cbar_nticks=2, **P)
        total_groups.append((next(letters), fig.axes[n0:]))
        _zoom_marker(fig, ax_tot, AR, zoom_left, fw, fh, x_z, y_bot, Wz)
    return dict(contrib=contrib_groups, total=total_groups, species=species, kind=kind,
                # for n the resulting field and its zoom span the same two rows as the
                # contributions, so their tints are made equally tall; Gamma_x has a single
                # resulting field centred on those rows, whose tint fits that panel alone
                match_tint_heights=has_zoom,
                x_left_edge=x_c2 + W, x_right_edge=x_r)


def _finish_block(fig, fw, fh, rend, blk):
    """The three faint tints of one block -- contributions, resulting field, whole block -- and
    the sum arrow between the first two. Returns the gap this block needs for that arrow."""
    i = 0 if blk["species"] == "A" else 1
    cb = group_bbox_frac(fig, [a for _, ax in blk["contrib"] for a in ax], rend,
                         texts=[t for t, _ in blk["contrib_texts"]])
    tb = group_bbox_frac(fig, [a for _, ax in blk["total"] for a in ax], rend,
                         texts=[t for t, _ in blk["total_texts"]])
    if blk["match_tint_heights"]:
        # the contributions tint and the resulting-field tint of one block share one vertical extent
        y0, y1 = min(cb.y0, tb.y0), max(cb.y1, tb.y1)
        cb = Bbox.from_extents(cb.x0, y0, cb.x1, y1)
        tb = Bbox.from_extents(tb.x0, y0, tb.x1, y1)
    blk["bbox"] = Bbox.union([cb, tb])      # the whole-block tint is drawn by _draw_block_tints
    if FAINT:
        draw_faint_panel(fig, cb, FAINT_CONTRIB[i % len(FAINT_CONTRIB)], fw, fh,
                         alpha=FAINT_ALPHA, pad_in=FAINT_PAD_IN)
        draw_faint_panel(fig, tb, FAINT_TOTAL[i % len(FAINT_TOTAL)], fw, fh,
                         alpha=FAINT_ALPHA, pad_in=FAINT_PAD_IN)
    gap0, gap1 = cb.x1 * fw + FAINT_PAD_IN, tb.x0 * fw - FAINT_PAD_IN      # inches
    x_c = 0.5 * (gap0 + gap1)
    y = 0.5 * (min(cb.y0, tb.y0) + max(cb.y1, tb.y1))
    label = _sum_label(blk["kind"], blk["species"])
    cap = fig.text(x_c / fw, y - SUM_TEXT_DY_IN / fh, label, ha="center", va="top",
                   fontsize=FONT * SUM_TEXT_SIZE_FACTOR, color=FAINT_ARROW, zorder=35)
    w_cap = cap.get_window_extent(rend).transformed(fig.transFigure.inverted()).width * fw
    arrow_len = w_cap * (1.0 + 2.0 * ARROW_PAD_CHARS / max(len(label), 1))
    style = f"simple,tail_width={ARROW_TAIL_W},head_width={ARROW_HEAD_W},head_length={ARROW_HEAD_L}"
    fig.add_artist(FancyArrowPatch(((x_c - 0.5 * arrow_len) / fw, y),
                                   ((x_c + 0.5 * arrow_len) / fw, y), transform=fig.transFigure,
                                   arrowstyle=style, mutation_scale=1.0, facecolor=FAINT_ARROW,
                                   edgecolor=FAINT_ARROW, zorder=35, clip_on=False))
    over_left = cb.x1 * fw - blk["x_left_edge"]       # contributions reaching into the gap
    over_right = blk["x_right_edge"] - tb.x0 * fw     # resulting field reaching into the gap
    return over_left + over_right + arrow_len + 2.0 * (FAINT_PAD_IN + ARROW_CLEAR_IN)


def _draw_block_tints(fig, fw, fh, blocks):
    """The whole-block tint behind each species block. The facing edges of two blocks grow towards
    each other until exactly FAINT_BLOCK_GAP_IN of white is left between them, so the two tints
    always separate cleanly however tall the blocks turn out to be."""
    pad = FAINT_BLOCK_PAD_IN / fh
    tops = [pad] * len(blocks)
    bots = [pad] * len(blocks)
    for i in range(len(blocks) - 1):                       # blocks run top to bottom
        sep = blocks[i]["bbox"].y0 - blocks[i + 1]["bbox"].y1
        inner = min(pad, max(0.0, 0.5 * (sep - FAINT_BLOCK_GAP_IN / fh)))
        bots[i] = tops[i + 1] = inner
    for i, blk in enumerate(blocks):
        bb = blk["bbox"]
        px = FAINT_BLOCK_PAD_IN / fw
        draw_faint_panel(fig, Bbox([[bb.x0 - px, bb.y0 - bots[i]], [bb.x1 + px, bb.y1 + tops[i]]]),
                         FAINT_BLOCK_COLORS[i % len(FAINT_BLOCK_COLORS)], fw, fh,
                         alpha=FAINT_BLOCK_ALPHA, pad_in=0.0, zorder=-30)


def _render(kind, d, gap, out=None):
    """Build the whole figure for a given contributions->resulting-field gap. Returns the gap the
    measured group edges and caption width call for, so the caller can rebuild with it."""
    sol = d["sol"]
    Lref = 2.0 * float(sol["h"]); AR = float(sol["Lm"]) / Lref
    W = H * AR
    block_h = 2.0 * (CBAR_BLOCK + H + XLABEL_H) + ROW_GAP
    fw = M_LEFT + 2.0 * W + COL_GAP + gap + W + M_RIGHT
    fh = M_TOP + 2.0 * block_h + BLOCK_GAP + M_BOTTOM

    rc = {k: FONT for k in ("font.size", "axes.titlesize", "axes.labelsize",
                            "xtick.labelsize", "ytick.labelsize")}
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(fw, fh))
        letters = iter("abcdefghijklmnopqrstuvwxyz")
        blocks = []
        for bi, species in enumerate(("A", "B")):
            y_top = fh - M_TOP - bi * (block_h + BLOCK_GAP) - CBAR_BLOCK - H
            blocks.append(_draw_block(fig, fw, fh, y_top, kind, species, d, gap, letters))
        for ax in fig.axes:                      # let the tints show through everywhere
            ax.patch.set_alpha(0.0)
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        for blk in blocks:                       # letters first: the tints must enclose them
            for grp in ("contrib", "total"):
                blk[f"{grp}_texts"] = [(_corner_letter(fig, rend, ax, L, fw, fh), L)
                                       for L, ax in blk[grp]]
        need = max(_finish_block(fig, fw, fh, rend, blk) for blk in blocks)
        if FAINT:
            _draw_block_tints(fig, fw, fh, blocks)
        if out is not None:
            tint_gap = FAINT_BLOCK_GAP_IN if FAINT else (
                (blocks[0]["bbox"].y0 - blocks[1]["bbox"].y1) * fh)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=DPI, bbox_inches="tight", pad_inches=0.03)
            print(f"wrote {out}  ({fw:.1f} x {fh:.1f} in), A-B tint gap {tint_gap:.2f} in")
        plt.close(fig)
    return need


def build(kind="n") -> None:
    cfg_case = _case_cfg()
    sol, fields, diff, contrib, _sf, _sr = acs.solve_analytic_case(cfg_case, case_max_workers=8)
    _, _, minline = analytic.compute_min_diagnostics(sol, fields)
    cfg = analytic.get_case_plot_settings(cfg_case.aspect_ratio)["contour"]
    if cfg.get("zoom_xlim") is not None:
        zoom_xlim = tuple(float(v) for v in cfg["zoom_xlim"])
    else:
        zoom_xlim = analytic.compute_inversion_zoom_xlim(
            sol["x"], minline[0], zoom_factor=float(cfg.get("zoom_factor", 3.0)))
    d = dict(sol=sol, fields=fields, contrib=contrib, cfg=cfg, zoom_xlim=zoom_xlim)
    gap = _render(kind, d, SUM_GAP_PROBE)                    # pass 1: measure
    _render(kind, d, gap, out=FIGURE_DIR / OUT_NAME)         # pass 2: rebuild with that gap


def main() -> None:
    build("n")


if __name__ == "__main__":
    main()
