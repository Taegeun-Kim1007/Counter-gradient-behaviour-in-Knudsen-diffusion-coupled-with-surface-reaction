#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Surface reaction rate along the trench wall, with boundary-source contributions.

Same case as Figure 2 (AR = 5, Kn = 100, kappa = 0.05). The three walls are unwrapped into one
along-wall axis: lower side wall -> trench base -> upper side wall, 0 at the CENTRE of the base,
positive toward the upper wall and negative toward the lower wall. The two dashed black lines
mark the base/side-wall corners (+/-0.5).

  Output: figures/figure4.png

    python3 figure4.py
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
import numpy as np

import sys as _sys
from pathlib import Path as _Path

_CODE_DIR = str(_Path(__file__).resolve().parents[1])  # folder of inversion_analytic.py
if _CODE_DIR not in _sys.path:
    _sys.path.insert(0, _CODE_DIR)
import inversion_analytic as analytic  # noqa: E402
import analytic_case_studies as acs
import figure2   # reuse the EXACT same case (STUDY, CASE_VALUE)
import figure_along_the_wall_schmatic as schematic   # along-wall coordinate schematic (optional inset)
from figure_common import FIGURE_DIR, defer_label, draw_deferred_labels

# ============================================================
# Configuration
# ============================================================
YLABEL = r"Surface reaction rate (m$^{-2}$ s$^{-1}$)"
ALONG_WALL_LABEL = r"$\ell/(2h)$"                 # along-wall arc coordinate (Methods keeps s)
OUT_NAME = "figure4.png"
# >>> which along-wall quantity to plot: "reaction_rate" (kappa * incident A flux, the default
#     Figure 4) or "wall_flux" (the incident molecular number flux on the wall). For wall_flux the
#     species is FLUX_SPECIES ("total" = A + B, "A", or "B") and the y-tick steps are auto-scaled.
QUANTITY = "reaction_rate"
FLUX_SPECIES = "total"
# >>> wall_flux direction: "incoming" (incident Gamma_imp) or "outgoing" (re-emitted flux). The
#     wall re-emits A_out = (1-kappa) A_in, B_out = kappa A_in + B_in, total_out = total_in (the model's
#     own wall boundary condition), so outgoing needs NO new solve -- just this algebra on the fluxes.
FLUX_DIRECTION = "incoming"
# --- wall-flux only: whitespace + guide placement (so total / A / B look identical) -------------
FLUX_A_MARGIN = 0.06          # panel a: bottom whitespace, as a fraction of the data range (also
                              #          the reaction-rate panel a bottom margin -- same tight look).
                              # NB: the zoom panels reuse Figure 4's own zoom rules (auto-scaled tick
                              #     steps), so there are no wall-flux-specific zoom knobs.
# panel a INSIDE wall-region guide (like Figure 4), as fractions of the data range. Side-wall labels
# go ABOVE the arrows, the trench-base label BELOW its arrow (separate rows -> no horizontal collision).
# Gamma_B and the total use the roomy placement below; only Gamma_A -- whose data sat too low, leaving
# a big empty top -- uses the tight ADAPTIVE placement (arrow just above the data, top just fits it).
FLUX_A_TOP = 0.26             # top whitespace above the data max (Gamma_B / total)
FLUX_GUIDE_ABOVE = 0.15       # guide arrows above the data max (Gamma_B / total)
FLUX_GUIDE_DY = 0.02          # side-wall labels: gap ABOVE the arrow
FLUX_GUIDE_DY_BASE = 0.05     # trench-base label: gap BELOW the arrow
# Gamma_A only: arrow height = above the global max AND above the base curve (so the below-base label
# clears it); top just fits the side labels. This pulls Gamma_A up without touching B / total.
FLUX_GUIDE_CLEAR = 0.03       # (Gamma_A) guide arrows: gap above the global data max
FLUX_LABEL_H = 0.08           # (Gamma_A) estimated label text height (fraction of the data range)
FLUX_TOP_PAD = 0.02           # (Gamma_A) small pad above the side-wall labels
# >>> optional along-wall coordinate schematic drawn to the RIGHT of panel a. It is redrawn as
#     vectors (not a shrunk raster), so its text keeps its absolute point size. SCH_GAP_IN is the
#     gap between panel a and the schematic.
ADD_SCHEMATIC = True         # draw the along-wall schematic to the RIGHT of panel a
LETTER_SIZE_FACTOR = 1.405   # >>> letter size as a multiple of FONT; ED Fig. 6 sets its own value
SCH_GAP_IN = 1.1
# crop the schematic's x-range (data pt units, native canvas is 0..598) to drop the empty left
# margin -- this narrows the gap to panel a AND shortens the schematic's aspect. No distortion.
SCH_XLIM = (118.0, 500.0)
# >>> overall composition: "a_right" (schematic right of a) | "sch_bottom"
#     (schematic as a bottom banner) | "sch_top" (schematic banner on top) | "two_col" (panel a +
#     schematic stacked on the left, zooms stacked in a column on the right).
LAYOUT = "a_right"
# --- schematic styling knobs (for trying prettier layouts) ------------------------------------
SCH_VALIGN = "center"        # vertical placement against panel a (a_right layout): center|top|bottom
SCH_DRAW_SCALE = 1.0         # extra size multiplier (text stays FONT pt, drawing grows -> roomier)
SCH_FRAME = False            # draw a faint rounded card behind the schematic
SCH_FRAME_COLOR = "#3B75AF"
SCH_FRAME_ALPHA = 0.06
SCH_FRAME_PAD_IN = 0.18
# >>> zoomed-in views of the trench base. False -> panel a (+schematic) only;
#     True -> also draw the ZOOM_GROUPS zoom panels below.
ZOOM_IN_VIEW = False
# All source lines are drawn in every panel; each zoom crops x AND y to the base region and one
# contribution's value band. Every panel keeps the
# SAME box aspect (1.25) and is just magnified (cropped), so nothing is stretched.
ALL_SERIES = ["total", "jet", "bottom", "top", "end"]
# Ignoring a
# source's own-wall segment makes the y-range track where that source actually varies (e.g. the
# base source peaks on the side walls near the corners, not on the base itself).
# (group, ignore_structural_zero_segments, y-mode). y-mode "cross" centres the panel on the
# upper/lower crossing at ell = 0; otherwise the group's band is framed symmetrically.
ZOOM_GROUPS = [
    (("jet",), False, None),
    (("bottom", "top", "end"), True, "cross"),
    (("total",), False, None),
]
STRUCT_ZERO = {"bottom": "bottom", "top": "top", "end": "end"}

# --- Layout (inches). Manual axes, ED-figure look. -----------------------------------
ASPECT = 10.0 / 8.0          # plot-box width : height -- SAME for all
FONT = 22.1
# >>> panel a wall-region guide (INSIDE the plot). WALL_GUIDE_Y is the DATA y-position of the
#     double-arrows -- change it to move the whole guide up/down.
WALL_GUIDE_Y = 1.33e24
WALL_GUIDE_DY = 0.02e24      # side-wall labels: gap ABOVE the arrow
WALL_GUIDE_DY_BASE = 0.06e24  # trench-base label: larger gap BELOW the arrow (clears it)
LINEWIDTH = 2.6
TOTAL_LW = LINEWIDTH         # total drawn at the SAME width as the source lines (uniform thickness)
# >>> Fig. 4 (reaction rate) only: star at the minimum of the resulting rate on EACH side wall
#     (the two minima are symmetric). Drawn in the total-curve colour. Not drawn for wall_flux (ED 6).
TOTAL_MIN_STAR = True
STAR_MS = LINEWIDTH * 5      # star marker size -- the same lw*5 convention as the other figures
BOX_LINEWIDTH = 3
WALL_DASH_LW = 3.0
MAIN_H = 4.8
MAIN_W = MAIN_H * ASPECT
ZOOM_H = 2.4                 # every zoom is a fixed 1.25 box, just cropped
ZOOM_W = ZOOM_H * ASPECT
BAND_PAD = 0.08              # padding around a group's value band (fraction of the band)
COL_GAP = 0.90
ROW_GAP = 1.15
M_LEFT, M_RIGHT, M_TOP, M_BOTTOM = 1.70, 0.35, 0.55, 0.95
LABEL_PAD_IN = 0.08
LABEL_YPOS = 1.05
# >>> ZOOM AMOUNT (user knob): the base zooms span x in +/- (0.5 + ZOOM_PAD). LARGER = LESS
#     zoomed (wider window). 0.25 -> +/-0.75 (tight), 0.5 -> +/-1.0, etc.
ZOOM_PAD = 0.45
# >>> Common y-tick spacing for ALL zoom panels (same interval everywhere). The y-limits are
#     snapped to whole multiples of this step so the ticks BRACKET every plotted value (nothing
#     sits above the top tick or below the bottom tick).
ZOOM_YTICK_STEP = 8.0e22
# >>> For the "cross"-centred zoom: shift the y-window UP by this much so the
#     curves sit a little LOWER in the panel. Positive = curves lower; 0 = exactly centred.
CROSS_YSHIFT = 3.0e22
# >>> cross zoom: 3 y-ticks whose outer two BRACKET the plotted band (nothing pokes past
#     them). The band is snapped out to a multiple of this quantum, then split into 3 even ticks.
CROSS_TICK_QUANTUM = 0.25e23
# >>> extra symmetric y-padding (in tick-steps) added by _sym_ylim so the bracketing ticks sit a
#     little INSIDE the box instead of exactly on the top/bottom edge.
EDGE_INSET_FRAC = 0.18
A_YTICK_STEP = 3.5e23       # >>> y-tick spacing for panel a
A_XTICK_N = 5              # >>> number of x-ticks on the full panel a (>3; evenly spaced)
PAD_INCHES = 0.03
DPI = 300

_CIDX = {"total": 0, "jet": 1, "bottom": 2, "top": 3, "end": 4}


def _nice_step(span, n_intervals):
    """A round tick step (1/2/2.5/5 x 10^k) giving roughly n_intervals intervals over `span`."""
    if not np.isfinite(span) or span <= 0:
        return 1.0
    raw = span / max(1, n_intervals)
    exp = np.floor(np.log10(raw))
    base = raw / 10.0 ** exp
    # finer candidate set so the bracketing tick lands close to the data max (tight margins)
    nice = next((m for m in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0) if base <= m), 10.0)
    return nice * 10.0 ** exp


def _series_map(data, kappa=0.0):
    """Along-wall series per source. Handles BOTH the reaction-rate object (local_rate[...]) and
    the wall-flux object (local_flux[species][...]); 'total' is the field total, the rest are the
    boundary-source contributions. For wall_flux, FLUX_DIRECTION selects incident ('incoming') or
    re-emitted ('outgoing') flux -- the latter is a linear combination of the incident A/B fluxes:
    A_out = (1-kappa) A_in, B_out = kappa A_in + B_in, total_out = total_in."""
    walls, parts = ("bottom", "end", "top"), ("jet", "bottom", "top", "end")
    if "local_rate" in data:                          # surface reaction rate (Figure 4 default)
        tot, comp = data["local_rate"], data.get("local_rate_components")
        smap = {"total": {w: np.asarray(tot[w], float) for w in walls}}
        if comp is not None:
            for part in parts:
                smap[part] = {w: np.asarray(comp[w][part], float) for w in walls}
        return smap

    # wall flux -----------------------------------------------------------------------------------
    lf, comp = data["local_flux"], data.get("local_flux_components")

    def incident(sp):                                 # incident series for one species
        m = {"total": {w: np.asarray(lf[sp][w], float) for w in walls}}
        if comp is not None:
            for part in parts:
                m[part] = {w: np.asarray(comp[sp][w][part], float) for w in walls}
        return m

    if FLUX_DIRECTION != "outgoing":
        return incident(FLUX_SPECIES)

    # outgoing (re-emitted) flux: combine the incident A/B series wall-line by wall-line
    A, B = incident("A"), incident("B")
    if FLUX_SPECIES == "A":                            # reflected A
        return {k: {w: (1.0 - kappa) * A[k][w] for w in walls} for k in A}
    if FLUX_SPECIES == "B":                            # reflected B + newly produced B
        return {k: {w: kappa * A[k][w] + B[k][w] for w in walls} for k in A}
    return incident("total")                           # total_out == total_in


def _draw(ax, smap, names, s_bot, s_end, s_top, colors, corner_nd, ignore=False):
    """Plot the given sources along the wall. When ignore=True, a source's own-wall segment is
    skipped -- so exactly the lines that set the y-range are drawn, and nothing pokes outside it."""
    for name in names:
        ser = smap[name]
        col = colors[_CIDX[name]]
        lw = TOTAL_LW if name == "total" else LINEWIDTH
        for seg, coord, values in (("bottom", s_bot, ser["bottom"]),
                                   ("end", s_end, ser["end"]),
                                   ("top", s_top, np.asarray(ser["top"])[::-1])):
            if ignore and STRUCT_ZERO.get(name) == seg:
                continue
            ax.plot(coord, values, color=col, lw=lw)
    for c in (-corner_nd, corner_nd):
        ax.axvline(c, color="k", linestyle="--", linewidth=WALL_DASH_LW, zorder=1)


def build() -> None:
    sol, _fields, _diff, _contrib, _sf, srr = acs.solve_analytic_case(figure2._case_cfg(), case_max_workers=8)
    s_bot, s_end, s_top, h = analytic.surface_wall_polyline_coordinates(sol)
    Lref = 2.0 * float(h)
    s_bot, s_end, s_top = s_bot / Lref, s_end / Lref, s_top / Lref
    AR = float(sol["Lm"]) / Lref
    corner_nd = float(h) / Lref                        # base/side-wall corner at +/- 0.5
    s_end_max = (float(sol["Lm"]) + float(h)) / Lref   # open wall ends at +/- (AR + 0.5)
    zx = 0.5 + ZOOM_PAD
    kappa = float(sol.get("reaction_probability", 0.0))
    smap = _series_map(srr if QUANTITY == "reaction_rate" else _sf, kappa)
    # y-tick steps: the tuned constants for the reaction rate, or auto-scaled to the flux magnitude
    if QUANTITY == "reaction_rate":
        a_ytick_step, zoom_ytick_step = A_YTICK_STEP, ZOOM_YTICK_STEP
        cross_quantum, cross_yshift = CROSS_TICK_QUANTUM, CROSS_YSHIFT
    else:
        _v = np.concatenate([np.concatenate([smap[n]["bottom"], smap[n]["end"], smap[n]["top"]])
                             for n in ALL_SERIES])
        _v = _v[np.isfinite(_v)]
        _span = float(_v.max() - _v.min())
        a_ytick_step = _nice_step(_span, 4)          # ~5 ticks on panel a
        zoom_ytick_step = _nice_step(_span, 16)
        cross_quantum = _nice_step(_span, 50)
        cross_yshift = 0.0
    colors = analytic.CUSTOM_COLORS
    ccfg = analytic.get_case_plot_settings(int(round(AR)))["centerline"]

    a_xlim = (-s_end_max * 1.02, s_end_max * 1.02)

    # y-range for each zoom, replicating analytic._collect_region_values: collect a group's
    # values inside the base x-window [-zx, zx], optionally skipping each source's own-wall
    # ("structural-zero") segment. Note the top segment is stored reversed vs s_top.
    def _band(group, ignore):
        vals = []
        for name in group:
            ser = smap[name]
            for seg, coord, values in (("bottom", s_bot, ser["bottom"]),
                                       ("end", s_end, ser["end"]),
                                       ("top", s_top, np.asarray(ser["top"])[::-1])):
                if ignore and STRUCT_ZERO.get(name) == seg:
                    continue
                m = (coord >= -zx) & (coord <= zx)
                if np.any(m):
                    vals.append(np.asarray(values)[m])
        v = np.concatenate(vals)
        v = v[np.isfinite(v)]
        return float(v.min()), float(v.max())

    def _sym_ylim(lo, hi, step):
        """y-limits with EQUAL empty space beyond the min and the max, while still keeping a
        tick just below the min and just above the max (so nothing pokes past a tick). The
        padding is the larger of the two gaps to the nearest bracketing ticks, applied
        symmetrically on both sides."""
        need_top = float(np.ceil(hi / step) * step - hi)     # gap up to the first tick >= max
        need_bot = float(lo - np.floor(lo / step) * step)    # gap down to the first tick <= min
        # +EDGE_INSET_FRAC*step so the bracketing ticks are inset from the edges, not sitting on them
        pad = max(need_top, need_bot, 0.12 * step) + EDGE_INSET_FRAC * step
        return lo - pad, hi + pad

    zoom_groups = ZOOM_GROUPS if ZOOM_IN_VIEW else []   # zoom panels only when requested
    nz = len(zoom_groups)
    zoom_row_w = nz * ZOOM_W + (nz - 1) * COL_GAP if nz else 0.0
    zoom_band = (ZOOM_H + ROW_GAP) if nz else 0.0       # vertical space used by the zoom row
    # scale the schematic so its text is EXACTLY this figure's FONT (its native text is FONT_SIZE);
    # scaling the whole box keeps the schematic's own proportions identical.
    sch_scale = (FONT / schematic.base.FONT_SIZE) * SCH_DRAW_SCALE
    sch_crop_w = SCH_XLIM[1] - SCH_XLIM[0]           # cropped width in the schematic's pt units
    has_sch = bool(ADD_SCHEMATIC)
    sch_w = (sch_crop_w / 72.0) * sch_scale if has_sch else 0.0
    sch_h = (schematic.base.HEIGHT_PT / 72.0) * sch_scale if has_sch else 0.0
    sch_gap = SCH_GAP_IN if has_sch else 0.0

    # ---- composition: place panel a, the zoom panels and the schematic (each keeps its own size) --
    sch_xy = None
    if LAYOUT == "sch_bottom":                        # a on top, zoom row, schematic banner bottom
        band = (ROW_GAP + sch_h) if has_sch else 0.0
        content_w = max(MAIN_W, zoom_row_w, sch_w)
        fig_w = M_LEFT + content_w + M_RIGHT
        fig_h = M_BOTTOM + band + zoom_band + MAIN_H + M_TOP
        a_xy = (M_LEFT + 0.5 * (content_w - MAIN_W), M_BOTTOM + band + zoom_band)
        zx0, zy = M_LEFT + 0.5 * (content_w - zoom_row_w), M_BOTTOM + band
        zoom_xy = [(zx0 + i * (ZOOM_W + COL_GAP), zy) for i in range(nz)]
        if has_sch:
            sch_xy = (M_LEFT + 0.5 * (content_w - sch_w), M_BOTTOM)
    elif LAYOUT == "sch_top":                         # schematic banner on top, a, then zoom row
        band = (ROW_GAP + sch_h) if has_sch else 0.0
        content_w = max(MAIN_W, zoom_row_w, sch_w)
        fig_w = M_LEFT + content_w + M_RIGHT
        fig_h = M_BOTTOM + zoom_band + MAIN_H + band + M_TOP
        a_xy = (M_LEFT + 0.5 * (content_w - MAIN_W), M_BOTTOM + zoom_band)
        zx0, zy = M_LEFT + 0.5 * (content_w - zoom_row_w), M_BOTTOM
        zoom_xy = [(zx0 + i * (ZOOM_W + COL_GAP), zy) for i in range(nz)]
        if has_sch:
            sch_xy = (M_LEFT + 0.5 * (content_w - sch_w), a_xy[1] + MAIN_H + ROW_GAP)
    elif LAYOUT == "two_col":                         # left: a over schematic; right: zooms stacked
        col_gap2 = 1.0
        left_w = max(MAIN_W, sch_w)
        fig_w = M_LEFT + left_w + col_gap2 + (ZOOM_W if nz else 0.0) + M_RIGHT
        left_h = MAIN_H + ((ROW_GAP + sch_h) if has_sch else 0.0)
        right_h = nz * ZOOM_H + (nz - 1) * ROW_GAP if nz else 0.0
        fig_h = M_BOTTOM + max(left_h, right_h) + M_TOP
        top = fig_h - M_TOP
        a_xy = (M_LEFT + 0.5 * (left_w - MAIN_W), top - MAIN_H)
        if has_sch:
            sch_xy = (M_LEFT + 0.5 * (left_w - sch_w), a_xy[1] - ROW_GAP - sch_h)
        rx = M_LEFT + left_w + col_gap2
        zoom_xy = [(rx, top - (i + 1) * ZOOM_H - i * ROW_GAP) for i in range(nz)]
    else:                                             # "a_right" (default): schematic right of a
        row_a_w = MAIN_W + sch_gap + sch_w
        content_w = max(zoom_row_w, row_a_w)
        fig_w = M_LEFT + content_w + M_RIGHT
        fig_h = M_BOTTOM + zoom_band + MAIN_H + M_TOP
        a_xy = (M_LEFT + 0.5 * (content_w - row_a_w), M_BOTTOM + zoom_band)
        zx0, zy = M_LEFT + 0.5 * (content_w - zoom_row_w), M_BOTTOM
        zoom_xy = [(zx0 + i * (ZOOM_W + COL_GAP), zy) for i in range(nz)]
        if has_sch:
            sy = {"top": a_xy[1] + (MAIN_H - sch_h), "bottom": a_xy[1]}.get(
                SCH_VALIGN, a_xy[1] + 0.5 * (MAIN_H - sch_h))
            sch_xy = (a_xy[0] + MAIN_W + sch_gap, sy)

    def ax_in(x, yb, w, hh):
        return fig.add_axes([x / fig_w, yb / fig_h, w / fig_w, hh / fig_h])

    def _style(ax, xticks, xlim, xlabel=True, ylabel=False):
        ax.set_xlim(*xlim)
        ax.xaxis.set_major_locator(mticker.FixedLocator(xticks))
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
        ax.set_xlabel(ALONG_WALL_LABEL if xlabel else "")
        if ylabel:
            ax.set_ylabel(YLABEL)
        analytic.apply_sci_offset(ax, axis="y", scilimits=(0, 0), useOffset=False)
        analytic.apply_axes_style(ax, BOX_LINEWIDTH, ccfg["tick_length"], ccfg["tick_width"])

    rc = {k: FONT for k in ("font.size", "axes.titlesize", "axes.labelsize",
                            "xtick.labelsize", "ytick.labelsize", "legend.fontsize")}
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(fig_w, fig_h))
        entries = []
        letters = iter("abcdefgh")                    # a = main panel, then schematic, then zooms
        sch_ax = None

        # (a) full along-wall, ALL contributions. More x-ticks than just the two ends + centre,
        # and y-ticks that bracket every value (nothing above the top tick).
        a_xticks = list(np.linspace(-s_end_max, s_end_max, A_XTICK_N))
        a_all = np.concatenate([np.concatenate([smap[n]["bottom"], smap[n]["end"], smap[n]["top"]])
                                for n in ALL_SERIES])
        a_all = a_all[np.isfinite(a_all)]
        ax_a = ax_in(a_xy[0], a_xy[1], MAIN_W, MAIN_H)
        if has_sch and sch_xy is not None:            # along-wall coordinate schematic
            sch_x, sch_y = sch_xy
            if SCH_FRAME:                              # faint rounded card behind the schematic
                p = SCH_FRAME_PAD_IN
                fig.add_artist(FancyBboxPatch(
                    ((sch_x - p) / fig_w, (sch_y - p) / fig_h),
                    (sch_w + 2 * p) / fig_w, (sch_h + 2 * p) / fig_h,
                    boxstyle="round,pad=0,rounding_size=0.012", transform=fig.transFigure,
                    facecolor=SCH_FRAME_COLOR, edgecolor=SCH_FRAME_COLOR, alpha=SCH_FRAME_ALPHA,
                    linewidth=0.0, zorder=-30, clip_on=False, mutation_aspect=fig_h / fig_w))
            sch_ax = ax_in(sch_x, sch_y, sch_w, sch_h)
            schematic.draw_schematic(ax=sch_ax)
            sch_ax.set_xlim(SCH_XLIM[0], SCH_XLIM[1])   # crop empty left margin (undistorted)
            for t in sch_ax.texts:                    # force schematic text to EXACTLY this FONT
                t.set_fontsize(FONT)
        _draw(ax_a, smap, ALL_SERIES, s_bot, s_end, s_top, colors, corner_nd)
        _style(ax_a, a_xticks, a_xlim, xlabel=True, ylabel=True)
        # panel a y-limits + the DATA-y position of the inside wall-region guide. Bottom whitespace is
        # the SAME tight 6% for both quantities; the top leaves room for the guide drawn INSIDE the box.
        lo_v, hi_v = float(a_all.min()), float(a_all.max())
        rng = hi_v - lo_v
        if QUANTITY == "reaction_rate":
            _, top = _sym_ylim(lo_v, hi_v, a_ytick_step)
            gy, gdy, gdy_base = WALL_GUIDE_Y, WALL_GUIDE_DY, WALL_GUIDE_DY_BASE
        elif FLUX_SPECIES == "A":
            # Gamma_A ONLY: pull the panel up (its data sat too low). Arrow just above the global max
            # AND above the base curve (so the below-base label clears it); top just fits the labels.
            base_all = np.concatenate([np.asarray(smap[n]["end"], float) for n in ALL_SERIES])
            base_max = float(base_all[np.isfinite(base_all)].max())
            gy = max(hi_v + FLUX_GUIDE_CLEAR * rng,
                     base_max + (FLUX_GUIDE_DY_BASE + FLUX_LABEL_H) * rng)
            gdy, gdy_base = FLUX_GUIDE_DY * rng, FLUX_GUIDE_DY_BASE * rng
            top = gy + (FLUX_GUIDE_DY + FLUX_LABEL_H + FLUX_TOP_PAD) * rng
        else:
            # Gamma_B and the total: the roomy placement (guide well clear of the curves).
            top = hi_v + FLUX_A_TOP * rng
            gy = hi_v + FLUX_GUIDE_ABOVE * rng
            gdy, gdy_base = FLUX_GUIDE_DY * rng, FLUX_GUIDE_DY_BASE * rng
        ax_a.set_ylim(lo_v - FLUX_A_MARGIN * rng, top)
        ax_a.yaxis.set_major_locator(mticker.MultipleLocator(a_ytick_step))

        # wall-region guide INSIDE panel a (both quantities, same style as Figure 4): the double-arrows
        # sit above the data; side-wall labels go ABOVE them, the trench-base label BELOW its arrow (a
        # separate row, so the three never collide horizontally) with a white bbox to clear any curve.
        for x0, x1, txt, above in ((-s_end_max, -corner_nd, "Lower side wall", True),
                                   (-corner_nd, corner_nd, "Closed end", False),
                                   (corner_nd, s_end_max, "Upper side wall", True)):
            ax_a.annotate("", xy=(x1, gy), xytext=(x0, gy),
                          arrowprops=dict(arrowstyle="<->", color="k", lw=1.8),
                          annotation_clip=False)
            if above:
                ax_a.text(0.5 * (x0 + x1), gy + gdy, txt, ha="center",
                          va="bottom", fontsize=FONT, clip_on=False)
            else:
                ax_a.text(0.5 * (x0 + x1), gy - gdy_base, txt, ha="center",
                          va="top", fontsize=FONT, clip_on=False, zorder=25,
                          bbox=dict(boxstyle="square,pad=0.12", facecolor="white", edgecolor="none"))
        if QUANTITY == "reaction_rate" and TOTAL_MIN_STAR:
            # star the minimum of the resulting rate on each side wall (symmetric pair)
            tot_b = np.asarray(smap["total"]["bottom"], float)
            tot_t = np.asarray(smap["total"]["top"], float)[::-1]     # plotted against s_top
            ib, it = int(np.nanargmin(tot_b)), int(np.nanargmin(tot_t))
            ax_a.plot([s_bot[ib], s_top[it]], [tot_b[ib], tot_t[it]], marker="*", ls="None",
                      color=colors[0], markersize=STAR_MS, zorder=30)

        entries.append((ax_a, next(letters), LABEL_YPOS))
        if sch_ax is not None:                        # the schematic is its own lettered subpanel
            entries.append((sch_ax, next(letters), LABEL_YPOS))

        # base zooms: SAME 1.25 box. ALL contributions are drawn together (so every
        # source is visible at once); the y-range is still focused on one group (with its own-wall
        # segment skipped when ignore=True), so contributions outside that band simply run off the
        # bottom/top -- that is fine, the point is to see all of them simultaneously.
        for i, (group, ignore, ymode) in enumerate(zoom_groups):
            ax = ax_in(zoom_xy[i][0], zoom_xy[i][1], ZOOM_W, ZOOM_H)
            _draw(ax, smap, ALL_SERIES, s_bot, s_end, s_top, colors, corner_nd)
            _style(ax, [-0.5, 0.0, 0.5], (-zx, zx), xlabel=True, ylabel=(i == 0))
            lo, hi = _band(group, ignore)
            # wall_flux uses the SAME zoom rules as the reaction rate (Figure 4). We compute Figure 4's
            # ORIGINAL window + ticks (unchanged magnification), then translate the window in y ONLY so
            # the middle tick lands at the panel centre -- a pure y-shift, NEVER a rescale.
            if ymode == "cross":
                # centre on the upper/lower crossing at ell = 0 (base centre)
                i0 = int(np.argmin(np.abs(s_end)))
                cross = float(np.asarray(smap["top"]["end"])[i0])
                half = max(cross - lo, hi - cross) + 0.12 * zoom_ytick_step
                q = cross_quantum
                t0, t2 = float(np.floor(lo / q) * q), float(np.ceil(hi / q) * q)
                ylo0 = min(cross - half + cross_yshift, t0)      # original window (sets the scale)
                yhi0 = max(cross + half + cross_yshift, t2)
                ticks = list(np.linspace(t0, t2, 3))
                m, h = ticks[1], yhi0 - ylo0
                ax.set_ylim(m - 0.5 * h, m + 0.5 * h)            # same height -> only shifted in y
                ax.yaxis.set_major_locator(mticker.FixedLocator(ticks))
                analytic.set_sci_axis_two_decimal(ax, axis="y")
            else:
                ylo0, yhi0 = _sym_ylim(lo, hi, zoom_ytick_step)  # original window (sets the scale)
                k0 = int(np.ceil(ylo0 / zoom_ytick_step - 1e-9))
                k1 = int(np.floor(yhi0 / zoom_ytick_step + 1e-9))
                ticks = [k * zoom_ytick_step for k in range(k0, k1 + 1)]
                if len(ticks) % 2 == 1:                          # a clear middle tick -> centre it
                    h, m = yhi0 - ylo0, ticks[len(ticks) // 2]   # same height -> only shifted in y
                    ax.set_ylim(m - 0.5 * h, m + 0.5 * h)
                else:                                            # no middle tick -> keep the window
                    ax.set_ylim(ylo0, yhi0)
                ax.yaxis.set_major_locator(mticker.FixedLocator(ticks))
            entries.append((ax, next(letters), LABEL_YPOS))

        for _e in entries:
            defer_label(_e[0], _e[1], size=FONT)
        draw_deferred_labels(fig, size_factor=LETTER_SIZE_FACTOR)

        out = FIGURE_DIR / OUT_NAME
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=DPI, bbox_inches="tight", pad_inches=PAD_INCHES)
        plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    build()


if __name__ == "__main__":
    main()
