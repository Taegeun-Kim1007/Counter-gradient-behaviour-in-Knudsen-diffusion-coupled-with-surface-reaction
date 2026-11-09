#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extended Data Fig. 1 -- contour overview.

Everything is re-rendered from data so that the contour y-size, colorbar rectangle and font
are uniform across AR; the contours keep their true x:y ratio. The orange dashed box and
arrow mark the zoom window. Output: figures/figED1.png

    python3 extended_figure1.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle, ConnectionPatch
from matplotlib.transforms import Bbox

import dsmc_postprocessing as post
import analytic_case_studies as acs
import sys as _sys
from pathlib import Path as _Path

_CODE_DIR = str(_Path(__file__).resolve().parents[1])  # folder of inversion_analytic.py
if _CODE_DIR not in _sys.path:
    _sys.path.insert(0, _CODE_DIR)
import inversion_analytic as analytic  # noqa: E402

from figure_common import (
    FIGURE_DIR, defer_label, draw_deferred_labels, build_ar_cfg,
    style_contour_axes as _style_contour_axes,
    draw_split_scalar as _draw_split_scalar,
    draw_split_inversion as _draw_split_inversion,
    add_colorbar as _add_quad_colorbar,
    group_bbox_frac, draw_faint_panel,
)

# >>> Faint grouping panel behind each AR (ED Fig. 1) / Kn (ED Fig. 3) quad -- one distinct
#     colour per case (cycled in quad order).
FAINT = True
FAINT_COLORS = ["#3B75AF", "#519E3E", "#E8720C"]   # case 1, 2, 3 (blue, green, orange)
FAINT_ALPHA = 0.16
FAINT_PAD_IN = 0.22
FAINT_MATCH_ROW_WIDTH = True   # >>> a row holding a single quad (the AR 5 row) has its faint block
                               #     stretched to the same left and right edges as the widest row,
                               #     so all the faint blocks line up horizontally. False -> each
                               #     block hugs its own contours.


# ============================================================
# Configuration
# ============================================================

# Figure layout: which AR quads share each row.
QUAD_ROWS = ((1, 3), (5,))

OUT_PATH = FIGURE_DIR / "figED1.png"

# >>> Per-AR zoom RANGE: left edge of the zoom window in x/(2h) (right edge = base = AR).
#     None -> use the automatically computed window. Smaller value -> wider zoom window.
ZOOM_XLEFT = {1: None, 3: None, 5: None}

# --- Layout (inches). The image is scaled in LaTeX, so absolute inches only set
# resolution; what matters is the relative proportions and the uniform font. --------
QUAD_FONT = 80            # single font size for ALL text in the figure
H_CONTOUR = 4.2 * 0.90    # contour panel height (identical for every panel and AR)
# Horizontal colorbar placed ABOVE each contour (centred), numbers above the bar and
# the quantity label above the numbers.
CBAR_W = 0.5             # colorbar thickness
CBAR_LEN_FRAC = 0.8      # colorbar length = this x the contour's x-width
CBAR_DECIMALS = 2        # tick numbers to this many decimals (no x10^n exponent)
CBAR_ABOVE_PAD = 0.40    # gap between the contour top and the colorbar
CBAR_LABEL_DY = 1.6     # full contours: variable label this many inches above the colorbar
CBAR_LABEL_DY_ZOOM = 1.00  # zoom panels: separate value for the variable label
CBAR_TOP_PAD = 2.50      # vertical room above the colorbar for its numbers + label
XLABEL_H = 2.50          # room below each contour for its x-axis label
CBAR_XF = 0.5            # colorbar horizontal position above the contour (0.5 = centred)
COL_GAP = 6.60            # gap between the left and right columns of a quad
QUAD_GAP = 6.19           # gap between two quads in the same row
INNER_RG = 1.00           # gap between the two rows within a quad
SUPER_GAP = 1.655         # gap between successive quad rows
M_LEFT, M_RIGHT, M_TOP, M_BOTTOM = 2.30, 2.60, 1.00, 1.70
BOX_MARGIN_IN = 0.07      # how far the orange zoom box extends beyond the contour
# Panel letters are placed by figure_common.draw_deferred_labels: at the top left corner of each
# panel AS A WHOLE (contour + ticks + axis labels + colorbar and its label), one common line per
# row and one common x per column. LABEL_SIZE_FACTOR is the only knob left here.
LABEL_SIZE_FACTOR = 1.402 # >>> letter size as a multiple of QUAD_FONT (the tick-number size)
# Tick-number spacing (points) applied to EVERY contour panel.
XTICK_PAD = 35             # push every x-tick number DOWN, away from the axis
YTICK_PAD = 35             # push every y-tick number LEFT, away from the axis
QUAD_DPI = 50
ORANGE = "#F39200"        # zoom box / arrow colour
# Zoom box + arrow line thickness (points), set to MATCH figure2.py ZOOM_*_LW.
# Kept SEPARATE from BOX_LW below so the thicker zoom marker does not also thicken the contour
# panel borders / centerline / min line (those still use BOX_LW).
ZOOM_BOX_LW = 13.0        # zoom box thickness         (= figure2.ZOOM_BOX_LW)
ZOOM_ARROW_LW = 13.0      # zoom arrow shaft thickness (= figure2.ZOOM_ARROW_LW)
ARROW_HEAD = 85.0         # zoom arrow head size
BOX_LW = 6.0              # contour-panel border / centerline / min-line thickness (NOT the zoom marker)
# Where the arrow leaves the box bottom edge, as a fraction of the box width (0.5 = centre).
# AR3 uses 0.75 so the shaft clears the x-tick "2"; other ARs use ARROW_START_FRAC.
ARROW_START_FRAC_AR3 = 0.75
ARROW_START_FRAC = 0.5     # arrow START (top, on the Y_A box) fraction for non-AR3 quads
# arrow END (tip, on the zoom panel) fraction along the zoom window (0.5 = centre, ->1 = base/right).
# Per-AR dict {AR: frac}; may also be set to a single float (used by ED Fig. 3, whose quads
# are all AR5). Larger value -> tip further RIGHT.
ARROW_END_FRAC = {1: 0.85, 3: 0.95, 5: 0.85}
ARROW_LEN_SCALE = 1.3      # lengthen the arrow along its current direction (1.0 = unchanged; >1 longer)
# Contour line weights (points).
CONTOUR_BOX_LW = BOX_LW   # contour panel border (the box around each contour)
CENTERLINE_LW = BOX_LW    # the y=0 line splitting DSMC (top) / analytic (bottom)
MINLINE_LW = BOX_LW       # the dashed Y_A min line
MINLINE_COLOR = "white"   # colour of the Y_A (reactant A) min line (shared by ED Figs. 1 & 3)


# ============================================================
# Data + drawing helpers
# ============================================================

def _load_quad_data(aspect_ratio: int) -> dict:
    """Gather the DSMC (ensemble) and analytic fields needed for the quad figure."""
    case_cfg = build_ar_cfg(aspect_ratio)
    ens = post.build_ensemble_average_result(case_cfg)
    sol_d, fields_d, diff_d = ens.sol, ens.fields, ens.diff
    sol_a, fields_a, diff_a, _c, _sf, _sr = acs.solve_analytic_case(case_cfg, case_max_workers=8)
    _, _, minline_d = analytic.compute_min_diagnostics(sol_d, fields_d)
    _, _, minline_a = analytic.compute_min_diagnostics(sol_a, fields_a)
    cfg = analytic.get_case_plot_settings(case_cfg.aspect_ratio)["contour"]
    if cfg.get("zoom_xlim") is not None:
        zoom_xlim = tuple(float(v) for v in cfg["zoom_xlim"])
    else:
        zoom_xlim = analytic.compute_inversion_zoom_xlim(
            sol_d["x"], minline_d[0], minline_a[0], zoom_factor=float(cfg.get("zoom_factor", 3.0))
        )
    sign_d = post.compute_ya_min_based_inversion_x(sol_d, fields_d)["sign"]["x"]
    sign_a = analytic.compute_inversion_regions(sol_a, fields_a, diff_a, species="A")["sign"]["x"]
    return dict(
        sol_d=sol_d, sol_a=sol_a, fields_d=fields_d, fields_a=fields_a,
        diff_d=diff_d, diff_a=diff_a, minline_d=minline_d, minline_a=minline_a,
        zoom_xlim=zoom_xlim, sign_d=sign_d, sign_a=sign_a, cfg=cfg,
        Lref=2.0 * float(sol_d["h"]), Lm=float(sol_d["Lm"]),
        zoom_left_override=ZOOM_XLEFT.get(aspect_ratio),
    )


def _draw_quad(fig, fig_w, fig_h, data, x0, y_top, y_bot, labels):
    """Draw one AR's quad at the given inch positions (contours keep their true ratio)."""
    cfg = data["cfg"]
    AR = data["Lm"] / data["Lref"]
    _ov = data.get("zoom_left_override")
    zoom_left = float(_ov) if _ov is not None else round(float(data["zoom_xlim"][0]) / data["Lref"], 1)
    full_w = H_CONTOUR * AR
    zoom_w = H_CONTOUR * (AR - zoom_left)
    nt_full = 2 if AR <= 1.5 else 3      # AR1 -> 2 ticks, AR3/AR5 -> 3 ticks
    has_zoom = AR > 1.5                  # AR1: no zoom-in view
    # Vertical extent of the colorbar block above each contour. The zoom arrow points to
    # the top of the zoom panel's colorbar block.
    cbar_block = CBAR_ABOVE_PAD + CBAR_W + CBAR_TOP_PAD
    x_col1 = x0
    x_col2 = x0 + full_w + COL_GAP

    def ax_in(x, y, w, h):
        return fig.add_axes([x / fig_w, y / fig_h, w / fig_w, h / fig_h])

    def cbar_above(x_panel, y_panel, w_panel, sm, vmn, vmx, title, sci, nt, label_dy=CBAR_LABEL_DY):
        """Horizontal colorbar centred above the contour; length = 80% of its x-width;
        nt tick marks; plain decimals (no x10^n)."""
        clen = CBAR_LEN_FRAC * w_panel
        cx = x_panel + CBAR_XF * (w_panel - clen)
        cy = y_panel + H_CONTOUR + CBAR_ABOVE_PAD
        _add_quad_colorbar(fig, ax_in(cx, cy, clen, CBAR_W), sm, vmn, vmx, title, sci, cfg,
                           orientation="horizontal", nticks=nt,
                           decimals=CBAR_DECIMALS, label_dy_in=label_dy)

    # Top-left: Y_A with the orange zoom box (density fields go in the left column).
    ax_YA = ax_in(x_col1, y_top, full_w, H_CONTOUR)
    smY, vmnY, vmxY, _ = _draw_split_scalar(ax_YA, data, data["fields_d"]["YA"], data["fields_a"]["YA"],
                                            powernorm_gamma=cfg.get("mass_fraction_powernorm_gamma"),
                                            centerline_lw=CENTERLINE_LW, minline_lw=MINLINE_LW,
                                            minline_color=MINLINE_COLOR)
    _style_contour_axes(ax_YA, AR, cfg, show_xlabel=True, show_ylabel=True,
                        box_lw=CONTOUR_BOX_LW, xtick_pad=XTICK_PAD, ytick_pad=YTICK_PAD)   # top-left: keep y
    defer_label(ax_YA, next(labels), size=QUAD_FONT)
    cbar_above(x_col1, y_top, full_w, smY, vmnY, vmxY, r"$Y_{\mathrm{A}}$", False, nt_full)

    # Bottom-left: Y_A zoom, horizontally centred under the Y_A panel above (AR>1.5 only).
    if has_zoom:
        x_zoom = x_col1 + (full_w - zoom_w) / 2.0
        ax_zoom = ax_in(x_zoom, y_bot, zoom_w, H_CONTOUR)
        smZ, vmnZ, vmxZ, _ = _draw_split_scalar(ax_zoom, data, data["fields_d"]["YA"], data["fields_a"]["YA"],
                                                powernorm_gamma=cfg.get("mass_fraction_powernorm_gamma"),
                                                xlim_si=(zoom_left * data["Lref"], data["Lm"]),
                                                centerline_lw=CENTERLINE_LW, minline_lw=MINLINE_LW,
                                                minline_color=MINLINE_COLOR)
        _style_contour_axes(ax_zoom, AR, cfg, zoom_left=zoom_left, show_xlabel=True, show_ylabel=True,
                            box_lw=CONTOUR_BOX_LW, xtick_pad=XTICK_PAD, ytick_pad=YTICK_PAD)  # centred zoom: hide y label, keep tick numbers
        defer_label(ax_zoom, next(labels), size=QUAD_FONT)
        cbar_above(x_zoom, y_bot, zoom_w, smZ, vmnZ, vmxZ, r"$Y_{\mathrm{A}}$", False, 2,
                   label_dy=CBAR_LABEL_DY_ZOOM)   # zoom: min/max only

    # Top-right: j_x.
    ax_jx = ax_in(x_col2, y_top, full_w, H_CONTOUR)
    sm, vmn, vmx, _ = _draw_split_scalar(ax_jx, data, data["diff_d"]["A"]["ju_diff"], data["diff_a"]["A"]["ju_diff"],
                                         show_minline=False,   # j_x,A: no Y_A min line
                                         centerline_lw=CENTERLINE_LW, minline_lw=MINLINE_LW)
    _style_contour_axes(ax_jx, AR, cfg, show_xlabel=True, show_ylabel=True, box_lw=CONTOUR_BOX_LW,
                        xtick_pad=XTICK_PAD, ytick_pad=YTICK_PAD)
    defer_label(ax_jx, next(labels), size=QUAD_FONT)
    cbar_above(x_col2, y_top, full_w, sm, vmn, vmx,
               analytic.field_symbol("ju_diff", sp="A") + r" (kg m$^{-2}$ s$^{-1}$)", False, nt_full)

    x_inv = x_col2 if has_zoom else x_col1 + (full_w + COL_GAP) / 2.0
    ax_inv = ax_in(x_inv, y_bot, full_w, H_CONTOUR)
    _draw_split_inversion(ax_inv, data, show_minline=True,
                          centerline_lw=CENTERLINE_LW, minline_lw=MINLINE_LW,
                          minline_color=MINLINE_COLOR)
    _style_contour_axes(ax_inv, AR, cfg, show_xlabel=True, show_ylabel=True, box_lw=CONTOUR_BOX_LW,
                        xtick_pad=XTICK_PAD, ytick_pad=YTICK_PAD)
    defer_label(ax_inv, next(labels), size=QUAD_FONT)

    if not has_zoom:
        return   # AR1: no zoom-in view (and no orange box/arrow)

    # Orange zoom box on the Y_A panel (top-left), slightly larger than the marked region,
    # drawn in the Y_A axes' DATA coordinates (margin md from inches) so bbox_inches="tight"
    # measures it; clip_on=False lets it spill just outside the axes.
    md = BOX_MARGIN_IN / H_CONTOUR     # inches -> data units (1 data unit = H_CONTOUR in)
    ax_YA.add_patch(Rectangle((zoom_left - md, -0.5 - md),
                              (AR - zoom_left) + 2 * md, 1.0 + 2 * md,
                              transform=ax_YA.transData, fill=False, edgecolor=ORANGE,
                              linestyle=(0, (6, 4)), linewidth=ZOOM_BOX_LW, zorder=20, clip_on=False))

    # Orange dashed arrow from the CENTRE of the box bottom edge down to the centred
    # zoom panel (the centre lands between x ticks, so it stays clear of the numbers).
    # The shaft and head are SEPARATE artists: if a single dashed FancyArrow is used,
    # the dash pattern is applied to the arrowhead outline too and the thick, very large
    # head renders as a jagged/notched triangle. Here the shaft is dashed (no head) and
    # a short SOLID filled head is laid over the tip so the triangle stays clean.
    end_frac = ARROW_END_FRAC.get(int(round(AR)), 0.85) if isinstance(ARROW_END_FRAC, dict) else float(ARROW_END_FRAC)
    x_mid = zoom_left + end_frac * (AR - zoom_left)
    # Start fraction along the box bottom edge (AR3 shifted right to clear the x-tick "2").
    sfrac = ARROW_START_FRAC_AR3 if int(round(AR)) == 3 else ARROW_START_FRAC
    x_start = zoom_left + sfrac * (AR - zoom_left)
    pA = np.asarray(fig.transFigure.inverted().transform(
        ax_YA.transData.transform((x_start, -0.5 - md))))
    pB = np.asarray(fig.transFigure.inverted().transform(
        ax_zoom.transData.transform((x_mid, 0.5 + cbar_block / H_CONTOUR))))   # above the zoom colorbar variable
    pB = pA + ARROW_LEN_SCALE * (pB - pA)   # lengthen along the current direction (tip moves, start fixed)
    # Dashed shaft up to the head base, then only the solid filled head (length = the
    # triangle itself), so no visible solid straight segment remains before the tip.
    tA = fig.transFigure.transform(pA); tB = fig.transFigure.transform(pB)
    v = tB - tA; Lpx = float(np.hypot(*v))
    u = v / Lpx if Lpx else np.array([1.0, 0.0])
    pBase = fig.transFigure.inverted().transform(tB - (ARROW_HEAD * fig.dpi / 72.0) * u)
    fig.add_artist(ConnectionPatch(
        xyA=pA, coordsA=fig.transFigure, xyB=pBase, coordsB=fig.transFigure,
        arrowstyle="-", linestyle=(0, (6, 4)), linewidth=ZOOM_ARROW_LW, color=ORANGE, zorder=30))
    fig.add_artist(ConnectionPatch(
        xyA=pBase, coordsA=fig.transFigure, xyB=pB, coordsB=fig.transFigure,
        arrowstyle="-|>", mutation_scale=ARROW_HEAD, linewidth=ZOOM_ARROW_LW, linestyle="solid",
        color=ORANGE, zorder=31))


def build_extended_figure1(out_path: Path = OUT_PATH) -> None:
    """Combined contour overview laid out per QUAD_ROWS."""
    datas = {ar: _load_quad_data(ar) for row in QUAD_ROWS for ar in row}

    def quad_w(ar):
        full_w = H_CONTOUR * (datas[ar]["Lm"] / datas[ar]["Lref"])
        return 2.0 * full_w + COL_GAP     # colorbar is above the contour, not on the right

    def row_w(row):
        return sum(quad_w(ar) for ar in row) + QUAD_GAP * (len(row) - 1)

    # Each contour carries a horizontal colorbar above it (cbar_block) and an x-axis
    # label below it. The horizontal colorbar's vertical footprint is its thickness
    # (CBAR_W) plus room above for its tick/offset text (CBAR_TOP_PAD).
    cbar_block = CBAR_ABOVE_PAD + CBAR_W + CBAR_TOP_PAD
    super_h = 2.0 * H_CONTOUR + 2.0 * cbar_block + 2.0 * XLABEL_H + INNER_RG

    content_w = max(row_w(row) for row in QUAD_ROWS)
    fig_w = M_LEFT + content_w + M_RIGHT
    content_h = len(QUAD_ROWS) * super_h + (len(QUAD_ROWS) - 1) * SUPER_GAP
    fig_h = M_BOTTOM + content_h + M_TOP

    rc = {k: QUAD_FONT for k in ("font.size", "axes.titlesize", "axes.labelsize",
                                 "xtick.labelsize", "ytick.labelsize", "legend.fontsize")}
    labels = iter("abcdefghijklmnopqrstuvwxyz")
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(fig_w, fig_h))
        top = fig_h - M_TOP
        quad_axes = []
        for i, row in enumerate(QUAD_ROWS):
            super_top = top - i * (super_h + SUPER_GAP)         # top of the top colorbar
            y_top = super_top - cbar_block - H_CONTOUR          # top-row contour bottom
            y_bot = y_top - (H_CONTOUR + cbar_block + INNER_RG + XLABEL_H)  # bottom-row contour bottom
            # Centre rows that are narrower than the widest row.
            x = M_LEFT + (content_w - row_w(row)) / 2.0
            for ar in row:
                n0 = len(fig.axes)
                _draw_quad(fig, fig_w, fig_h, datas[ar], x, y_top, y_bot, labels)
                quad_axes.append(fig.axes[n0:])                 # this AR quad's axes
                x += quad_w(ar) + QUAD_GAP
        if FAINT:                                               # faint panel per AR quad, distinct colour
            fig.canvas.draw()
            rend = fig.canvas.get_renderer()
            bbs = [group_bbox_frac(fig, axs, rend) for axs in quad_axes]
            if FAINT_MATCH_ROW_WIDTH and len(bbs) > 1:
                gx0 = min(b.x0 for b in bbs); gx1 = max(b.x1 for b in bbs)
                k = 0
                for row in QUAD_ROWS:
                    if len(row) == 1:                          # alone on its row -> span the full width
                        b = bbs[k]
                        bbs[k] = Bbox.from_extents(gx0, b.y0, gx1, b.y1)
                    k += len(row)
            for j, bb in enumerate(bbs):
                draw_faint_panel(fig, bb, FAINT_COLORS[j % len(FAINT_COLORS)], fig_w, fig_h,
                                 alpha=FAINT_ALPHA, pad_in=FAINT_PAD_IN)
        draw_deferred_labels(fig, size_factor=LABEL_SIZE_FACTOR)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=QUAD_DPI, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    build_extended_figure1()


if __name__ == "__main__":
    main()
