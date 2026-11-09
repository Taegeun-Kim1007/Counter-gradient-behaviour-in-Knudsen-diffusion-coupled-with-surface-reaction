#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extended Data Fig. 3 -- Knudsen-number study, contour quads (same style as Extended Data Fig. 1).

The quad layout and every colorbar / font / line setting are imported from extended_figure1.py,
so the quads are styled identically to Extended Data Fig. 1. DSMC (ensemble average, upper half)
vs analytic (lower half). The number of ensemble runs used is printed for each Kn.
Output: figures/figED3.png

    python3 extended_figure3.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys as _sys
from pathlib import Path as _Path

_CODE_DIR = str(_Path(__file__).resolve().parents[1])  # folder of inversion_analytic.py
if _CODE_DIR not in _sys.path:
    _sys.path.insert(0, _CODE_DIR)
import inversion_analytic as analytic  # noqa: E402
import analytic_case_studies as acs
import dsmc_postprocessing as post

import extended_figure1 as ef1  # reuse the quad machinery + ALL layout/colorbar/font/line settings
from figure_common import (FIGURE_DIR, build_kn_cfg, draw_deferred_labels, group_bbox_frac,
                           draw_faint_panel)


# ============================================================
# Configuration
# ============================================================

# Each Kn is one AR5 quad. AR5 quads are wide (like Extended Data Fig. 1's AR5 row), so
# they are stacked one per row over Kn = 10, 1, 0.05.
QUAD_ROWS = (("10",), ("1",), ("0.05",))

# >>> Per-Kn zoom RANGE: left edge of the zoom window in x/(2h) (right edge = base = AR).
#     None -> use the automatically computed window. Smaller value -> wider zoom window.
ZOOM_XLEFT = {"0.05": None, "1": None, "10": None}
OUT_PATH = FIGURE_DIR / "figED3.png"

# --- Spacing knobs (inches). These override extended_figure1's values for THIS figure
# only; every other setting (colorbars, fonts, line widths, panel size) stays identical.
# Both the build layout and _draw_quad read these from ef1, so editing them here keeps
# the two consistent. Any other ef1 constant can be overridden the same way, e.g.
# ef1.H_CONTOUR (panel height), ef1.CBAR_ABOVE_PAD or ef1.XLABEL_H (vertical room).
INNER_GAP     = 1.0
ef1.INNER_RG  = INNER_GAP                        # top row -> bottom row within a Kn quad
ef1.SUPER_GAP = INNER_GAP * 2.92                         # bottom row -> next Kn's top row
ef1.COL_GAP   = 6.00     # >>> gap between the j_x and Y_A columns within a quad (raise to spread
                         #     the contours horizontally; the figure then needs a larger \includegraphics width)
ef1.QUAD_GAP  = 5.20     # gap between quads sharing one row (only matters if >1 Kn per row)
ef1.H_CONTOUR = 4.20
LETTER_SIZE_FACTOR = 1.261   # >>> letter size as a multiple of ef1.QUAD_FONT
ef1.M_LEFT, ef1.M_RIGHT, ef1.M_TOP, ef1.M_BOTTOM = 2.30, 2.60, 1.00, 1.70   # outer margins
ef1.ARROW_START_FRAC = 0.45   # zoom arrow START (top, on Y_A box) along the box (0.5 = centre)
ef1.ARROW_END_FRAC   = 0.8  # zoom arrow END (tip, on zoom panel); >0.5 shifts the tip right off the zoom x-tick "3"
ef1.ARROW_LEN_SCALE  = 1.25 # lengthen the zoom arrow along its current direction (1.0 = unchanged; >1 longer)
ef1.H_CONTOUR = 3.40        # >>> contour panel height (inches). Lower it to free
                            #     vertical room for the caption. The fonts keep their point size, so
                            #     drop the \includegraphics width in main.tex by the same factor to
                            #     keep the PRINTED contour and text sizes unchanged.
ef1.QUAD_FONT = 100         # single font size for ALL text (axes, ticks, colorbars, panel letters)
ef1.CBAR_LABEL_DY_ZOOM = 0.6   # ED Fig. 3 zoom colorbar variable label lower
# QUAD_FONT is large (100), so at the zoom panels' bottom-left corner the first x-tick ("2.6")
# and the bottom y-tick ("-0.5") crowd. Push the x-numbers further DOWN (clear of "-0.5") and the
# y-numbers a touch LEFT so the corner labels never touch.
ef1.XTICK_PAD = 55
ef1.YTICK_PAD = 36

# Room (inches) reserved BELOW the top row's contour. Every contour now carries
# its own x/(2h) label, so the top row needs the same room as the bottom row.
TOP_XLABEL_H = ef1.XLABEL_H

ef1.QUAD_DPI = 50

def _load_quad_data_kn(kn: str) -> dict:
    """Quad data for one Kn case (aspect ratio 5, kappa = 0.05): DSMC ensemble + analytic."""
    case_cfg = build_kn_cfg(kn)
    ens = post.build_ensemble_average_result(case_cfg)
    print(f"[ensemble] {case_cfg.study}/{case_cfg.case_name}: "
          f"averaged over {ens.info.case_count} DSMC run(s)", flush=True)
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
        zoom_left_override=ZOOM_XLEFT.get(kn),
    )


def build_extended_figure3(out_path: Path = OUT_PATH) -> None:
    """One AR5 quad per Kn, laid out per QUAD_ROWS, styled exactly like Extended Data Fig. 1."""
    datas = {kn: _load_quad_data_kn(kn) for row in QUAD_ROWS for kn in row}

    def quad_w(kn):
        full_w = ef1.H_CONTOUR * (datas[kn]["Lm"] / datas[kn]["Lref"])
        return 2.0 * full_w + ef1.COL_GAP     # colorbar is above the contour, not on the right

    def row_w(row):
        return sum(quad_w(kn) for kn in row) + ef1.QUAD_GAP * (len(row) - 1)

    cbar_block = ef1.CBAR_ABOVE_PAD + ef1.CBAR_W + ef1.CBAR_TOP_PAD
    # Top row reserves TOP_XLABEL_H below it; bottom row keeps ef1.XLABEL_H.
    super_h = 2.0 * ef1.H_CONTOUR + 2.0 * cbar_block + TOP_XLABEL_H + ef1.XLABEL_H + ef1.INNER_RG

    content_w = max(row_w(row) for row in QUAD_ROWS)
    fig_w = ef1.M_LEFT + content_w + ef1.M_RIGHT
    content_h = len(QUAD_ROWS) * super_h + (len(QUAD_ROWS) - 1) * ef1.SUPER_GAP
    fig_h = ef1.M_BOTTOM + content_h + ef1.M_TOP

    rc = {k: ef1.QUAD_FONT for k in ("font.size", "axes.titlesize", "axes.labelsize",
                                     "xtick.labelsize", "ytick.labelsize", "legend.fontsize")}
    labels = iter("abcdefghijklmnopqrstuvwxyz")
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(fig_w, fig_h))
        top = fig_h - ef1.M_TOP
        quad_axes = []
        for i, row in enumerate(QUAD_ROWS):
            super_top = top - i * (super_h + ef1.SUPER_GAP)        # top of this row's colorbar block
            y_top = super_top - cbar_block - ef1.H_CONTOUR         # top-row contour bottom
            y_bot = y_top - (ef1.H_CONTOUR + cbar_block + ef1.INNER_RG + TOP_XLABEL_H)  # bottom-row contour bottom
            x = ef1.M_LEFT + (content_w - row_w(row)) / 2.0
            for kn in row:
                n0 = len(fig.axes)
                ef1._draw_quad(fig, fig_w, fig_h, datas[kn], x, y_top, y_bot, labels)
                quad_axes.append(fig.axes[n0:])                    # this Kn quad's axes
                x += quad_w(kn) + ef1.QUAD_GAP
        if ef1.FAINT:                                              # faint panel per Kn quad, distinct colour
            fig.canvas.draw()
            rend = fig.canvas.get_renderer()
            for j, axs in enumerate(quad_axes):
                bb = group_bbox_frac(fig, axs, rend)
                draw_faint_panel(fig, bb, ef1.FAINT_COLORS[j % len(ef1.FAINT_COLORS)], fig_w, fig_h,
                                 alpha=ef1.FAINT_ALPHA, pad_in=ef1.FAINT_PAD_IN)
        draw_deferred_labels(fig, size_factor=LETTER_SIZE_FACTOR)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=ef1.QUAD_DPI, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    build_extended_figure3()


if __name__ == "__main__":
    main()
