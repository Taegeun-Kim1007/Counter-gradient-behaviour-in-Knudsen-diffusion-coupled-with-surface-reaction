#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extended Data Fig. 4 -- reaction-probability (kappa) study of the side-wall number-density
contributions for BOTH species, shown as CONTOURS with the max-ridge line.

How the maxima of the upper and lower side-wall contributions to the number density move as the
surface reaction probability kappa changes, at fixed aspect ratio 5 and Kn = 100 (same case as
Figures 2 and 3, only kappa varies). Same contour presentation as Fig. 2a,b -- each panel is a
wall contribution with the dot-dot-dashed max ridge (the per-y argmax of the contribution along x).

  6 rows x 2 columns. Columns = upper, lower side wall.
  Rows = reactant A (kappa = 0, 0.05, 0.5) as the top block, then product B (kappa = 0.05, 0.5, 1)
  as the bottom block -- the degenerate cases are dropped. A faint blue/green background groups the
  two species, and kappa is printed to the left of each row.

  For the reactant, the maxima sit at the mid-depth at kappa = 0 and shift towards the opening as
  kappa grows.

  build(kind) draws this SAME kappa x (upper, lower) layout for either quantity:
    build("n")     -> number density n   (this file, ED Fig. 4), guide line = max ridge
    build("Gamma") -> axial flux Gamma_x (extended_figure7.py, ED Fig. 7), guide line = sign change

  Output: figures/figED4.png

    python3 extended_figure4.py
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.transforms import Bbox
import numpy as np

import dsmc_postprocessing as post
import analytic_case_studies as acs
import sys as _sys
from pathlib import Path as _Path

_CODE_DIR = str(_Path(__file__).resolve().parents[1])  # folder of inversion_analytic.py
if _CODE_DIR not in _sys.path:
    _sys.path.insert(0, _CODE_DIR)
import inversion_analytic as analytic  # noqa: E402
import figure2  # reuse the EXACT Fig. 2 contour panel, ridge line, and style constants
from figure_common import FIGURE_DIR, draw_deferred_labels, group_bbox_frac, draw_faint_panel

# ---- Case sweep ---------------------------------------------------------------------
SPECIES_ROWS = ["A", "B"]                              # A block (top), then B block (bottom)
KAPPAS_BY_SPECIES = {"A": [0.0, 0.05, 0.5],            # per-species rows (degenerate cases dropped)
                     "B": [0.05, 0.5, 1.0]}
COLS = [("top", "upper"), ("bottom", "lower")]         # 2 columns: upper, lower side wall
ASPECT_RATIO = 5
KNUDSEN = 100.0
OUT_NAME = "figED4.png"

figure2.FONT = 105
# >>> ED figures keep their own contour size and column gap: figure2's values are tuned for the
#     main-figure page budget and would otherwise drag this figure along.
figure2.H = 4.20                    # >>> contour panel height (inches) for ED 4 / ED 7
figure2.YLABEL_PAD = 4.0            # >>> keep y/(2h) close to its ticks here: the wide pad used in
                                    #     Fig. 2 would run into the kappa row labels
figure2.ef1.XTICK_PAD = 28          # >>> ED tick pads: Fig. 2 widens these to clear its zoom panel,
figure2.ef1.YTICK_PAD = 28          #     which here would push y/(2h) into the kappa labels
H = figure2.H
FONT = figure2.FONT
COL_GAP = 5.21                      # >>> HORIZONTAL gap between the upper/lower contour columns
XLABEL_H = figure2.XLABEL_H
CBAR_BLOCK = figure2.CBAR_BLOCK
M_LEFT, M_RIGHT, M_TOP, M_BOTTOM = figure2.M_LEFT, figure2.M_RIGHT, figure2.M_TOP, figure2.M_BOTTOM
INNER_RG = 1.3                      # smaller row gap than figure2's, so the 6 rows stay compact
BD_STYLE = (0, (7, 3, 1, 3, 1, 3))  # dot-dot-dashed max ridge (identical to Fig. 2a,b)
LETTER_SIZE_FACTOR = 1.217          # >>> letter size as a multiple of figure2.FONT
LETTER_LEFT_FRAC = 0.5              # >>> shift every panel letter left by this fraction of its
                                    #     own width, clear of the contour block
KAPPA_LABEL_GAP_IN = 2.79           # >>> HORIZONTAL gap (inches) between the kappa row label and
                                    #     the LEFTMOST DRAWN element of that row -- the rotated
                                    #     y/(2h) label when the row has one, otherwise the outermost
                                    #     y-tick number. Measured after the figure is rendered, so
                                    #     every row ends up with the SAME visual gap no matter how
                                    #     wide its label or its tick numbers are. Lower = kappa sits
                                    #     closer to the contour block.
# --- Line thicknesses -------------------------------------------------------------
# Overriding figure2/ef1 here affects THIS run only, so Figs 2/3 and ED Figs. 1/3 keep their 6.0.
RIDGE_LW = 6.0                      # dot-dot-dashed max-ridge line width (figure2 default: ef1.BOX_LW = 6.0)
BOX_LW = 6.0                        # contour-panel box border width      (figure2 default: ef1.CONTOUR_BOX_LW = 6.0)
figure2.LINE_LW = RIDGE_LW
figure2.ef1.CONTOUR_BOX_LW = BOX_LW
# Faint background block per species, spanning each block's full tight bbox (panels, colorbars,
# tick labels, kappa labels) so nothing is left uncovered; a white gap is kept between the blocks.
FAINT_COLORS = {"A": "#3B75AF", "B": "#4CA64C"}
FAINT_ALPHA = 0.12
FAINT_PAD_IN = 0.30
FAINT_GAP_IN = 0.101                # >>> white gap left between the A and B blocks (inches);
                                    #     The facing edges of the two tints grow until exactly
                                    #     this much white is left, so the gap never depends on
                                    #     how tall the rows happen to be -- and never closes.


def _case_cfg(kappa):
    pressure = float(post.KN_REF * post.PRESSURE_REF_AT_KN_REF / KNUDSEN)
    return acs.build_ar_case_config(aspect_ratio=ASPECT_RATIO, reaction_probability=float(kappa),
                                    pressure=pressure, knudsen_number=KNUDSEN)


def build(kind="n") -> None:
    key = "n" if kind == "n" else "nu"          # contrib data key: number density or axial flux
    all_kappas = sorted({k for ks in KAPPAS_BY_SPECIES.values() for k in ks})
    solved = {k: acs.solve_analytic_case(_case_cfg(k), case_max_workers=8) for k in all_kappas}
    cfg = analytic.get_case_plot_settings(ASPECT_RATIO)["contour"]

    sol0 = solved[all_kappas[0]][0]
    Lref = 2.0 * float(sol0["h"]); AR = float(sol0["Lm"]) / Lref
    W = H * AR
    Pv = H + CBAR_BLOCK + XLABEL_H + INNER_RG
    rows = [(sp, k) for sp in SPECIES_ROWS for k in KAPPAS_BY_SPECIES[sp]]   # A block then B block
    nrow = len(rows); nA = len(KAPPAS_BY_SPECIES["A"])
    fig_w = M_LEFT + 2.0 * W + COL_GAP + M_RIGHT
    fig_h = M_TOP + CBAR_BLOCK + H + (nrow - 1) * Pv + XLABEL_H + M_BOTTOM

    x_up = M_LEFT
    x_low = M_LEFT + W + COL_GAP
    yc1 = fig_h - M_TOP - CBAR_BLOCK - H                 # top row's panel-bottom y

    rc = {k: FONT for k in ("font.size", "axes.titlesize", "axes.labelsize",
                            "xtick.labelsize", "ytick.labelsize")}
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(fig_w, fig_h))
        letters = iter("abcdefghijklmnopqrstuvwxyz")
        block_axes = {sp: [] for sp in SPECIES_ROWS}
        kappa_texts = {sp: [] for sp in SPECIES_ROWS}
        row_labels = []                                      # (left axes, kappa text) per row
        for i, (sp, kappa) in enumerate(rows):
            sol, fields, _d, contrib, _sf, _sr = solved[kappa]
            X, Y = np.asarray(sol["X"]), np.asarray(sol["Y"])
            x1d, y1d = np.asarray(sol["x"]), np.asarray(sol["y"])
            yb = yc1 - i * Pv
            show_xlabel = (i == nrow - 1)
            a_before = len(fig.axes)
            ax_left = None
            for c, (part, sup) in enumerate(COLS):
                Z = np.asarray(contrib[sp][part][key])
                zmin, zmax = float(np.nanmin(Z)), float(np.nanmax(Z))
                if zmax > zmin:
                    # n -> per-y argmax ridge ; Gamma_x -> where the axial flux changes sign
                    line = (figure2._line_extreme(x1d, y1d, Z, want_max=True) if kind == "n"
                            else figure2._line_sign(x1d, y1d, Z))
                    norm = None
                else:  # degenerate safeguard (no degenerate case is plotted with the chosen kappa)
                    line = None; norm = mcolors.Normalize(zmin, zmin + 1.0)
                ax = figure2._panel(fig, fig_w, fig_h, x_up if c == 0 else x_low, yb, W, X, Y, Z,
                                    Lref, AR, cfg, figure2._sym(kind, sp, sup), next(letters),
                                    show_xlabel=show_xlabel, show_ylabel=(c == 0),
                                    show_xtl=True, show_ytl=True, norm=norm,
                                    line=line, line_style=BD_STYLE, line_color="black",
                                    unit=figure2.UNITS[kind])
                if c == 0:
                    ax_left = ax
            # x here is provisional: it is re-solved below, once the row has been rendered, so that
            # the gap to the row's leftmost drawn artist is KAPPA_LABEL_GAP_IN in EVERY row.
            t = ax_left.text(0.0, 0.5, rf"$\kappa = {kappa:g}$", transform=ax_left.transAxes,
                             ha="right", va="center", fontsize=FONT)
            block_axes[sp].extend(fig.axes[a_before:])       # panels + colorbars added this row
            kappa_texts[sp].append(t)
            row_labels.append((ax_left, t))

        # Equal kappa gaps: put the RIGHT edge of each label KAPPA_LABEL_GAP_IN inches to the left of
        # the leftmost artist actually DRAWN on that row -- the rotated y/(2h) label when the row
        # carries one, otherwise its outermost y-tick number. A fixed axes-fraction offset instead
        # measures from the contour box, which leaves an unequal visual gap as soon as those artists
        # differ between rows. Needs a render first, so the text/tick extents are known.
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        for ax_l, t in row_labels:
            ab = ax_l.get_window_extent(rend)
            bbs = [ab]                                       # fallback: the contour box itself
            if ax_l.yaxis.label.get_text():
                bbs.append(ax_l.yaxis.label.get_window_extent(rend))
            bbs += [tl.get_window_extent(rend) for tl in ax_l.get_yticklabels() if tl.get_text()]
            left_px = min(b.x0 for b in bbs)
            t.set_x((left_px - KAPPA_LABEL_GAP_IN * fig.dpi - ab.x0) / ab.width)

        # faint background block per species (A rows on top, B rows below), covering every drawn
        # region (panels, colorbars, tick/axis labels, kappa), with a white gap kept between them.
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        a_bb = group_bbox_frac(fig, block_axes["A"], rend, texts=kappa_texts["A"])
        b_bb = group_bbox_frac(fig, block_axes["B"], rend, texts=kappa_texts["B"])
        padx, pady = FAINT_PAD_IN / fig_w, FAINT_PAD_IN / fig_h
        inner = max(0.0, 0.5 * (a_bb.y0 - b_bb.y1) - 0.5 * FAINT_GAP_IN / fig_h)
        x0 = min(a_bb.x0, b_bb.x0) - padx; x1 = max(a_bb.x1, b_bb.x1) + padx
        draw_faint_panel(fig, Bbox([[x0, a_bb.y0 - inner], [x1, a_bb.y1 + pady]]), FAINT_COLORS["A"],
                         fig_w, fig_h, alpha=FAINT_ALPHA, pad_in=0.0, zorder=-30)
        draw_faint_panel(fig, Bbox([[x0, b_bb.y0 - pady], [x1, b_bb.y1 + inner]]), FAINT_COLORS["B"],
                         fig_w, fig_h, alpha=FAINT_ALPHA, pad_in=0.0, zorder=-30)

        draw_deferred_labels(fig, size_factor=LETTER_SIZE_FACTOR, left_frac=LETTER_LEFT_FRAC)
        out = FIGURE_DIR / OUT_NAME
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=figure2.DPI, bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    build()


if __name__ == "__main__":
    main()
