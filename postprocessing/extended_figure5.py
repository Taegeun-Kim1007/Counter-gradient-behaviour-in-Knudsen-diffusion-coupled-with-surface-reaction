#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extended Data Fig. 5 (number density) & Fig. 8 (axial flux) -- reaction-probability (kappa)
study, CENTERLINE decomposition, both species.

How the centerline (y = 0) boundary-contribution decomposition changes with the surface
reaction probability kappa, at fixed aspect ratio 5 and Kn = 100 (same case as Figures 2 and 3,
only kappa varies). Laid out like Extended Data Fig. 2: manually placed, identically sized
SQUARE panels, one font size, Nature-style lower-case panel labels.

  2 rows x 3 columns. Rows = species: reactant A (top), product B (bottom).
  Columns = kappa, chosen per species (the degenerate cases are dropped):
    reactant A: kappa = 0, 0.05, 0.5    ;    product B: kappa = 0.05, 0.5, 1.0
  kappa is printed inside each panel (top-right; top-left for the product flux). Each panel
  stacks total + open + lower + upper + base along the centerline.

  build(component): "n"  -> Extended Data Fig. 5 (number density, figED5.png)
                    "nu" -> Extended Data Fig. 8 (axial flux Gamma_x, figED8.png; see extended_figure8.py)

  n panels: a star marks the x of the MAXIMUM of the upper and of the lower wall contribution.
  Gamma panels: a black dashed line marks Gamma_x = 0, and stars mark where the upper and
  lower wall contributions to Gamma_x cross zero.

  Output: figures/figED5.png (n) or figures/figED8.png (Gamma)

    python3 extended_figure5.py
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

import dsmc_postprocessing as post
import analytic_case_studies as acs
import sys as _sys
from pathlib import Path as _Path

_CODE_DIR = str(_Path(__file__).resolve().parents[1])  # folder of inversion_analytic.py
if _CODE_DIR not in _sys.path:
    _sys.path.insert(0, _CODE_DIR)
import inversion_analytic as analytic  # noqa: E402
from matplotlib.transforms import Bbox
from figure_common import (FIGURE_DIR, defer_label, draw_deferred_labels, group_bbox_frac,
                           draw_faint_panel)

# ============================================================
# Configuration
# ============================================================
SPECIES_ROWS = ["A", "B"]                              # rows: reactant A (top), product B (bottom)
KAPPAS_BY_SPECIES = {"A": [0.0, 0.05, 0.5],            # per-species columns (degenerate cases dropped)
                     "B": [0.05, 0.5, 1.0]}
PARTS = ["jet", "bottom", "top", "end"]                # open, lower, upper, base (analytic API keys)
ASPECT_RATIO = 5                                       # same case as Figures 2 and 3
KNUDSEN = 100.0
OUT_NAME = None                                        # None -> auto ("figED5.png" for n, "figED8.png" for nu)

# --- Layout (inches). Manual axes => every plot box is identical.
FONT = 16
LINEWIDTH = 1.5                   # centerline line width
LOWER_EXTRA_LW = 1.2              # "lower" line drawn thicker so it is not hidden under "upper"
BOX_LINEWIDTH = 2
PANEL_H = 2.0
PANEL_W = PANEL_H
COL_GAP = 1.00
ROW_GAP = 0.90
M_LEFT, M_RIGHT, M_TOP, M_BOTTOM = 1.45, 0.35, 0.80, 0.95
LABEL_PAD_IN = 0.0
LABEL_YPOS = 1.06
ZERO_LW = 3.0                     # Gamma = 0 reference line width
STAR_MS = 8.0                     # star marker size
# kappa printed INSIDE each panel.
KAPPA_XY_RIGHT = (0.955, 0.945)
KAPPA_XY_LEFT = (0.045, 0.945)
KAPPA_Y_FRAC = 0.2               # right-side kappa: fraction up from the 2nd-from-top y-tick to the top tick
# Faint background block per species row, so the reactant (A) and product (B) rows read as groups.
# The block spans each row's full tight bbox (tick labels, axis labels, panel letters, kappa) so
# nothing is left uncovered; a white gap is kept between the two blocks.
FAINT = True
FAINT_COLORS = {"A": "#3B75AF", "B": "#4CA64C"}   # reactant A (blue), product B (green)
FAINT_ALPHA = 0.12
FAINT_PAD_IN = 0.16
LETTER_SIZE_FACTOR = 1.222       # >>> letter size as a multiple of FONT; ED Fig. 8 sets its own
FAINT_GAP_IN = 0.0180            # >>> white gap left between the A and B faint blocks (inches);
                                 #     the facing edges grow until exactly this much is left
PAD_INCHES = 0.03
DPI = 300


def _out_name(component):
    if OUT_NAME is not None:
        return OUT_NAME
    return "figED5.png" if component == "n" else "figED8.png"


def _case_cfg(kappa):
    pressure = float(post.KN_REF * post.PRESSURE_REF_AT_KN_REF / KNUDSEN)
    return acs.build_ar_case_config(aspect_ratio=ASPECT_RATIO, reaction_probability=float(kappa),
                                    pressure=pressure, knudsen_number=KNUDSEN)


def _lw(part):
    return LINEWIDTH + (LOWER_EXTRA_LW if part == "bottom" else 0.0)


def _kappa_label(ax, kappa, comp, sp):
    """Print kappa inside the panel."""
    if comp == "nu" and sp == "B":
        ax.text(KAPPA_XY_LEFT[0], KAPPA_XY_LEFT[1], rf"$\kappa = {kappa:g}$",
                transform=ax.transAxes, ha="left", va="top", fontsize=FONT)
        return
    yt = sorted(ax.get_yticks())
    if len(yt) >= 2 and yt[-1] != yt[-2]:
        yv = yt[-2] + KAPPA_Y_FRAC * (yt[-1] - yt[-2])
        ax.text(KAPPA_XY_RIGHT[0], yv, rf"$\kappa = {kappa:g}$",
                transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=FONT)
    else:
        ax.text(KAPPA_XY_RIGHT[0], KAPPA_XY_RIGHT[1], rf"$\kappa = {kappa:g}$",
                transform=ax.transAxes, ha="right", va="top", fontsize=FONT)


def _draw_panel(ax, sol, fields, contrib, sp, comp, colors, x_s, iy0, AR, ccfg,
                show_xlabel, show_ylabel):
    ax.plot(x_s, fields[sp][comp][iy0, :], color=colors[0], linewidth=LINEWIDTH,
            label=analytic.centerline_part_label("tot"))
    for k, part in enumerate(PARTS, start=1):
        # The opening does not feed the product, so its contribution to B is identically zero and
        # is not drawn (as in Figs. 2 and 3, which omit that panel). k stays tied to PARTS so the
        # colours of the remaining contributions do not shift.
        if sp == "B" and part == "jet":
            continue
        ax.plot(x_s, contrib[sp][part][comp][iy0, :], color=colors[k % len(colors)],
                linewidth=_lw(part), label=analytic.centerline_part_label(part))

    # n panels: star at the x of the MAXIMUM of the lower (yellow) and upper (green) wall
    # contribution. At the centerline the two are symmetric, so draw the lower one a touch
    # larger and the upper on top -> both stay visible even when they coincide.
    if comp == "n":
        for part, cidx, extra in (("bottom", 2, 2.0), ("top", 3, 0.0)):
            line = contrib[sp][part][comp][iy0, :]
            if np.all(~np.isfinite(line)) or np.nanmax(line) == np.nanmin(line):
                continue
            ix = int(np.nanargmax(line))
            ax.plot(x_s[ix], line[ix], marker="*", linestyle="None",
                    markersize=STAR_MS + extra, color=colors[cidx], zorder=6)
        # total n: star at the interior MINIMUM (reactant A) or MAXIMUM (product B). Skip if the
        # field is flat or the extremum sits at a boundary (e.g. A at kappa = 0, where n_A is uniform).
        tot = fields[sp][comp][iy0, :]
        if np.all(np.isfinite(tot)) and np.nanmax(tot) != np.nanmin(tot):
            it = int(np.nanargmin(tot) if sp == "A" else np.nanargmax(tot))
            if 0 < it < tot.size - 1:
                ax.plot(x_s[it], tot[it], marker="*", linestyle="None",
                        markersize=STAR_MS, color=colors[0], zorder=6)

    ax.set_xlabel(r"$x/(2h)$" if show_xlabel else "")
    _unit = {"n": r" (m$^{-3}$)", "nu": r" (m$^{-2}$ s$^{-1}$)"}.get(comp, "")
    # Each axis carries BOTH the resulting field and the per-boundary contributions, so the label
    # names both symbols: e.g. "n_A, n_A^b (m^-3)".
    _sym = analytic.field_symbol(comp, sp=sp)
    _sym_b = _sym[:-1] + r"^{b}$"
    ax.set_ylabel(_sym + ", " + _sym_b + _unit if show_ylabel else "")
    ax.set_xlim(-0.05 * AR, 1.05 * AR)
    ax.xaxis.set_major_locator(mticker.FixedLocator([0.0, AR / 2.0, AR]))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
    analytic.apply_sci_offset(ax, axis="y", scilimits=(0, 0))
    analytic.set_centerline_minmax_ticks(ax, axis="y", nticks=5, scientific=True)
    analytic.apply_axes_style(ax, BOX_LINEWIDTH, ccfg["tick_length"], ccfg["tick_width"])

    if comp == "nu":
        ax.axhline(0.0, color="k", linestyle="--", linewidth=ZERO_LW, zorder=1)
        # star at the x where the lower (yellow) and upper (green) wall contribution to
        # Gamma_x crosses zero. At the centerline the two coincide, so draw the lower a touch
        # larger and the upper on top -> both stay visible even when they overlap.
        for part, cidx, extra in (("bottom", 2, 2.0), ("top", 3, 0.0)):
            line = contrib[sp][part][comp][iy0, :]
            s = np.where(line[:-1] * line[1:] < 0)[0]
            if not s.size:
                continue
            k = int(s[0]); f0, f1 = line[k], line[k + 1]
            xc = x_s[k] + (x_s[k + 1] - x_s[k]) * f0 / (f0 - f1)
            ax.plot(xc, 0.0, marker="*", linestyle="None",
                    markersize=STAR_MS + extra, color=colors[cidx], zorder=6)


def build(component="n") -> None:
    # Solve each distinct kappa once (each solve returns both species).
    all_kappas = sorted({k for ks in KAPPAS_BY_SPECIES.values() for k in ks})
    solved = {k: acs.solve_analytic_case(_case_cfg(k), case_max_workers=8) for k in all_kappas}
    ccfg = analytic.get_case_plot_settings(ASPECT_RATIO)["centerline"]
    colors = analytic.CUSTOM_COLORS

    nrow = len(SPECIES_ROWS); ncol = len(KAPPAS_BY_SPECIES[SPECIES_ROWS[0]])
    fig_w = M_LEFT + ncol * PANEL_W + (ncol - 1) * COL_GAP + M_RIGHT
    fig_h = M_BOTTOM + nrow * PANEL_H + (nrow - 1) * ROW_GAP + M_TOP

    def ax_in(x, yb, w, h):
        return fig.add_axes([x / fig_w, yb / fig_h, w / fig_w, h / fig_h])

    rc = {k: FONT for k in ("font.size", "axes.titlesize", "axes.labelsize",
                            "xtick.labelsize", "ytick.labelsize", "legend.fontsize")}
    labels = iter("abcdefghijklmnopqrstuvwxyz")
    top = fig_h - M_TOP                            # top edge of the top row's panels
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(fig_w, fig_h))
        entries = []
        row_axes = {sp: [] for sp in SPECIES_ROWS}
        for r, sp in enumerate(SPECIES_ROWS):
            show_xlabel = (r == nrow - 1)          # x-axis label on the bottom row only
            for c, kappa in enumerate(KAPPAS_BY_SPECIES[sp]):
                sol, fields, _diff, contrib, _sf, _sr = solved[kappa]
                y = np.asarray(sol["y"]); iy0 = int(np.argmin(np.abs(y - 0.0)))
                Lref = 2.0 * float(sol["h"]); x_s = np.asarray(sol["x"]) / Lref
                AR = float(sol["Lm"]) / Lref
                x = M_LEFT + c * (PANEL_W + COL_GAP)
                yb = top - (r + 1) * PANEL_H - r * ROW_GAP
                ax = ax_in(x, yb, PANEL_W, PANEL_H)
                _draw_panel(ax, sol, fields, contrib, sp, component, colors, x_s, iy0, AR, ccfg,
                            show_xlabel=show_xlabel, show_ylabel=(c == 0))
                _kappa_label(ax, kappa, component, sp)
                entries.append((ax, next(labels), LABEL_YPOS, c))
                row_axes[sp].append(ax)
        for _e in entries:
            defer_label(_e[0], _e[1], size=FONT)

        if FAINT and len(SPECIES_ROWS) == 2:
            fig.canvas.draw()
            rend = fig.canvas.get_renderer()
            bbs = {sp: group_bbox_frac(fig, axs, rend, texts=[t for ax in axs for t in ax.texts])
                   for sp, axs in row_axes.items()}
            a_bb, b_bb = bbs[SPECIES_ROWS[0]], bbs[SPECIES_ROWS[1]]   # A (top) above B (bottom)
            padx, pady = FAINT_PAD_IN / fig_w, FAINT_PAD_IN / fig_h
            inner = max(0.0, 0.5 * (a_bb.y0 - b_bb.y1) - 0.5 * FAINT_GAP_IN / fig_h)
            x0 = min(a_bb.x0, b_bb.x0) - padx; x1 = max(a_bb.x1, b_bb.x1) + padx
            draw_faint_panel(fig, Bbox([[x0, a_bb.y0 - inner], [x1, a_bb.y1 + pady]]),
                             FAINT_COLORS[SPECIES_ROWS[0]], fig_w, fig_h, alpha=FAINT_ALPHA, pad_in=0.0, zorder=-30)
            draw_faint_panel(fig, Bbox([[x0, b_bb.y0 - pady], [x1, b_bb.y1 + inner]]),
                             FAINT_COLORS[SPECIES_ROWS[1]], fig_w, fig_h, alpha=FAINT_ALPHA, pad_in=0.0, zorder=-30)

        # AFTER the faint blocks: the letters are kept inside the tint that encloses their row,
        # and lined up as if the y ticks carried a minus sign, so this figure and its Gamma_x
        # counterpart (ED Fig. 8, which has negative ticks) match
        draw_deferred_labels(fig, size_factor=LETTER_SIZE_FACTOR, minus_compensate=True)

        out = FIGURE_DIR / _out_name(component)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=DPI, bbox_inches="tight", pad_inches=PAD_INCHES)
        plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    build("n")


if __name__ == "__main__":
    main()
