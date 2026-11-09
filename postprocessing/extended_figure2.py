#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extended Data Fig. 2 -- centerline validation, as ONE combined figure.

Laid out 2 rows x 3 columns: the top row (a, b, c) is the centerline mass fraction Y
and the bottom row (d, e, f) is the centerline diffusive mass flux j_x, with aspect
ratios 1, 3, 5 across the columns. Every panel uses a manually placed axes of identical
size, so the plot box is the same whether or not a "x10^n" exponent is present. One
font size (FONT) is used for ALL text. Output: figures/figED2.png

    python3 extended_figure2.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figure_common import (
    FIGURE_DIR, defer_label, draw_deferred_labels, build_ar_cfg, load_case,
    draw_centerline_jx, draw_centerline_Y,
)


# ============================================================
# Configuration
# ============================================================

ASPECT_RATIOS = (1, 3, 5)
OUT_PATH = FIGURE_DIR / "figED2.png"

# --- Layout (inches). Manual axes => every plot box is identical. -----------------
FONT = 16                 # single font size for ALL text
LINEWIDTH = 1.5            # analytic (blue) centerline line width
DSMC_EXTRA_LW = 0.9       # DSMC (red) line is this many points THICKER than the blue line
BOX_LINEWIDTH = 2  # plot-box border width
PANEL_H = 2.0             # plot-box height (identical for every panel)
PANEL_W = PANEL_H         # plot box is square
COL_GAP = 1.00            # gap between the AR columns
ROW_GAP = 1.00            # gap between the j_x row and the mass-fraction row
M_LEFT, M_RIGHT, M_TOP, M_BOTTOM = 1.45, 0.35, 0.80, 0.95
LABEL_PAD_IN = 0.08
LETTER_SIZE_FACTOR = 1.202 # >>> letter size as a multiple of FONT
LETTER_DOWN_IN = 0.15     # >>> lower the letters of the panels that print an x10^n offset
                          #     (the j_x row), so both rows sit the same way over their content
LABEL_YPOS = 1.06
LABEL_YPOS_BOT = 1.06
PAD_INCHES = 0.02         # outer white-space padding when saving
DPI = 300


def build_extended_figure2(out_path: Path = OUT_PATH) -> None:
    datas = [load_case(build_ar_cfg(ar)) for ar in ASPECT_RATIOS]
    ncol = len(datas)        # AR1, AR3, AR5 across the columns
    nrow = 2

    fig_w = M_LEFT + ncol * PANEL_W + (ncol - 1) * COL_GAP + M_RIGHT
    fig_h = M_BOTTOM + nrow * PANEL_H + (nrow - 1) * ROW_GAP + M_TOP

    def ax_in(x, y, w, h):
        return fig.add_axes([x / fig_w, y / fig_h, w / fig_w, h / fig_h])

    rc = {k: FONT for k in ("font.size", "axes.titlesize", "axes.labelsize",
                            "xtick.labelsize", "ytick.labelsize", "legend.fontsize")}
    labels = iter("abcdefghijklmnop")
    top = fig_h - M_TOP
    # Two rows: mass fraction (a, b, c) on top, j_x (d, e, f) below. Only the bottom row
    # shows x/(2h); only the left column shows the y-axis variable label.
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(fig_w, fig_h))
        entries = []
        for row, draw in enumerate((draw_centerline_Y, draw_centerline_jx)):
            y = top - (row + 1) * PANEL_H - row * ROW_GAP
            show_xlabel = (row == nrow - 1)              # x/(2h) only on the bottom (j_x) row
            ypos = LABEL_YPOS if row == 0 else LABEL_YPOS_BOT
            for col, d in enumerate(datas):
                x = M_LEFT + col * (PANEL_W + COL_GAP)
                ax = ax_in(x, y, PANEL_W, PANEL_H)
                draw(ax, d, lw=LINEWIDTH, show_xlabel=show_xlabel, show_ylabel=(col == 0),
                     box_lw=BOX_LINEWIDTH, dsmc_extra=DSMC_EXTRA_LW)
                entries.append((ax, next(labels), ypos, col))
        for _e in entries:
            defer_label(_e[0], _e[1], size=FONT)
        draw_deferred_labels(fig, size_factor=LETTER_SIZE_FACTOR, down_offset_in=LETTER_DOWN_IN)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=DPI, bbox_inches="tight", pad_inches=PAD_INCHES)
        plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    build_extended_figure2()


if __name__ == "__main__":
    main()
