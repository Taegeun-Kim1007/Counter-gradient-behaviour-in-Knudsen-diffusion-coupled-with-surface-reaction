#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared helpers for the paper-figure builders.

Provides the output location, Nature-style panel-label helpers, case-config builders,
a DSMC(+analytic) field loader (averaged over the runs of each case), and centerline drawing
primitives. Output goes to figures/.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.transforms import ScaledTranslation, Bbox
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Nature Physics: sans-serif (Arial) for ALL figure text, maths included -- applied here at import
# time so EVERY figure builder that imports figure_common renders with the same font.
matplotlib.rcParams.update({
    "font.family": "Arial",
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
})

import dsmc_postprocessing as post
import analytic_case_studies as acs
import sys as _sys
from pathlib import Path as _Path

_CODE_DIR = str(_Path(__file__).resolve().parents[1])  # folder of inversion_analytic.py
if _CODE_DIR not in _sys.path:
    _sys.path.insert(0, _CODE_DIR)
import inversion_analytic as analytic  # noqa: E402


ROOT = Path(__file__).resolve().parent
FIGURE_DIR = ROOT / "figures"

# Nature figure lettering: lower-case bold sans-serif (Arial; fall back to the
# Helvetica-compatible faces / DejaVu Sans when Arial is not installed).
LABEL_FONTSIZE = 16
LABEL_FONTFAMILY = ["Arial"]
PANEL_HEIGHT_IN = 3.2     # height used when baking a label onto a single panel
DPI = 400


# ============================================================
# Saved-figure / label helpers
# ============================================================

def resolve_case_both_dir(study: str, case_value) -> Path:
    """Find the saved compare-figure '.../both' directory for a case."""
    study_root = post.figure_output_root(post.NO_SOLID_FIGURE_ROOT_SUFFIX) / study
    matches = sorted(study_root.glob(f"{case_value}_*"))
    matches = [m for m in matches if (m / "both").is_dir()]
    if not matches:
        raise FileNotFoundError(f"No saved case folder for {study}/{case_value} under {study_root}")
    return matches[0] / "both"


def place_label(ax, label, outside=False, size=None, ypos=None, xpos=None) -> None:
    """Nature-style panel label: lower-case, bold, sans-serif (Arial), no parentheses.
    outside=True places it just above the top-left corner (outside the axes). ypos/xpos
    override the vertical/horizontal position (axes fraction); a negative xpos moves the
    label left toward the y-axis label."""
    if outside:
        x, y, va = 0.0, 1.03, "bottom"
    else:
        x, y, va = 0.0, 1.0, "top"
    if ypos is not None:
        y, va = ypos, "bottom"
    if xpos is not None:
        x = xpos
    ax.text(
        x, y, str(label), transform=ax.transAxes,
        ha="left", va=va, fontsize=size if size is not None else LABEL_FONTSIZE,
        fontweight="bold", fontfamily=LABEL_FONTFAMILY, zorder=10,
    )


def place_panel_labels(fig, entries, pad_in=0.08, size=None) -> None:
    """Place Nature-style panel labels a FIXED distance (pad_in inches) to the LEFT of the
    left-most rendered content -- y-tick labels, the y-axis variable label and the x10^n
    offset text, whichever reaches furthest left. Measured from what is actually drawn, so the
    labels sit a uniform gap from the plot content regardless of tick-label width (a "-1.59"
    tick vs a "0.04" tick), instead of a hand-tuned constant that only matches some panels.

    entries: iterable of (ax, label, ypos) or (ax, label, ypos, group). Panels sharing a
    `group` (e.g. the column index) are ALIGNED to the SAME x -- the left-most content among
    them -- so a column of panels gets one common label x even if the rows have different tick
    widths. Without a group, each label aligns to its own panel. Panels in one group must
    share horizontal geometry (same x-position and width), as manual grid layouts do.
    Call ONCE, after every panel has been fully drawn."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    groups = {}
    order = []
    for e in entries:
        ax, label, ypos = e[0], e[1], e[2]
        group = e[3] if len(e) > 3 else id(ax)   # no group -> unique key -> per-panel
        if group not in groups:
            groups[group] = []
            order.append(group)
        groups[group].append((ax, label, ypos))
    for group in order:
        items = groups[group]
        # left-most content across the group, in axes fraction (comparable within a column).
        x_common = min(
            float(ax.transAxes.inverted().transform((ax.get_tightbbox(renderer).x0, 0.0))[0])
            for ax, _, _ in items
        )
        for ax, label, ypos in items:
            ax_w_px = float(ax.get_window_extent(renderer).width)
            pad_frac = (pad_in * fig.dpi) / ax_w_px if ax_w_px else 0.0
            place_label(ax, label, size=size, ypos=ypos, xpos=x_common - pad_frac)


# ----------------------------------------------------------------------------------------------
# Panel letters placed at the top left corner of a panel AS A WHOLE
#
# A figure calls defer_label() where it would have called place_label(), and draw_deferred_labels()
# once, just before savefig. The letters are then placed from what the figure actually rendered:
#
#   block  = the panel as a whole -- its contour or plot box, x and y tick numbers, x and y axis
#            labels, the x10^n offset, and the colorbar above it with the colorbar tick numbers
#            and its variable label. Annotations drawn inside or beside a panel (the kappa row
#            labels of ED Figs. 4 and 7, the in-panel kappa notes of ED Figs. 5 and 8) are NOT
#            part of the block, so they never drag a letter away from the panel.
#   place  = at the top left corner of that block. If the corner is taken by a wide colorbar label
#            (the narrow panels of ED Fig. 1) the letter slides left along the label line; if it is
#            taken by the panel's own top tick number (the line plots) the letter is lifted clear.
#   align  = panels of one row share one letter line and panels of one column share one letter x,
#            so a row stays aligned even where one panel carries a colorbar and another does not.
#   clamp  = a letter never pokes out of the faint tint that encloses its panels.
# ----------------------------------------------------------------------------------------------

_DEFERRED_LABELS = {}


def defer_label(ax, label, size=None) -> None:
    """Record a panel letter for `ax`; draw_deferred_labels() places it later."""
    if not str(label).strip():
        return
    _DEFERRED_LABELS.setdefault(ax.figure, []).append((ax, str(label), size))


def _pl_tick_boxes(a, rend):
    """Boxes of the tick numbers that are actually ON the axes. Matplotlib keeps a Text for every
    tick the locator produces, including ones outside the axis limits which are never drawn --
    counting those would push a panel block a whole tick past what the figure shows."""
    ab = a.get_window_extent(rend)
    out = []
    for t, axis in ([(t, "y") for t in a.get_yticklabels()]
                    + [(t, "x") for t in a.get_xticklabels()]):
        if t is None or not t.get_text() or not t.get_visible():
            continue
        b = t.get_window_extent(rend)
        c = 0.5 * (b.y0 + b.y1) if axis == "y" else 0.5 * (b.x0 + b.x1)
        lo, hi = (ab.y0, ab.y1) if axis == "y" else (ab.x0, ab.x1)
        if lo - 1.0 <= c <= hi + 1.0:
            out.append(b)
    return out


def _pl_text_boxes(a, rend, with_annotations=False):
    """The text an axes owns: tick numbers, axis labels and the x10^n offset."""
    items = [a.xaxis.label, a.yaxis.label, a.xaxis.get_offset_text(), a.yaxis.get_offset_text()]
    if with_annotations:
        items += list(a.texts)
    return (_pl_tick_boxes(a, rend)
            + [t.get_window_extent(rend) for t in items if t is not None and t.get_text()])


def _pl_owners(fig, rend, labelled):
    """Assign every unlabelled axes (a colorbar) to the labelled panel it belongs to: the one
    whose horizontal span holds it and whose centre is nearest."""
    own = {ax: [] for ax in labelled}
    for o in fig.axes:
        if o in labelled:
            continue
        ob = o.get_window_extent(rend)
        if ob is None:
            continue
        xc, yc = 0.5 * (ob.x0 + ob.x1), 0.5 * (ob.y0 + ob.y1)
        best, best_d = None, None
        for ax in labelled:
            ab = ax.get_window_extent(rend)
            if not (ab.x0 <= xc <= ab.x1):
                continue
            d = abs(yc - 0.5 * (ab.y0 + ab.y1))
            if best_d is None or d < best_d:
                best, best_d = ax, d
        if best is not None:
            own[best].append(o)
    return own


def _pl_minus_width(fig, rend, fs):
    """Width of a minus sign at the tick-number size."""
    t = fig.text(0.0, 0.0, "\u2212", fontsize=fs)
    w = t.get_window_extent(rend).width
    t.remove()
    return w


def _pl_block(fig, rend, ax, own, minus_compensate):
    """One panel as a whole -> (bbox, carries a colorbar)."""
    boxes = [ax.get_window_extent(rend)] + _pl_text_boxes(ax, rend)
    for o in own.get(ax, ()):
        boxes += [o.get_window_extent(rend)] + _pl_text_boxes(o, rend, with_annotations=True)
    bb = Bbox.union([b for b in boxes if b is not None])
    if minus_compensate:
        # line up as if the y tick numbers carried a minus sign, so a figure with only positive
        # ticks (ED Fig. 5) matches the one that has negative ticks (ED Fig. 8)
        ticks = [t for t in ax.get_yticklabels() if t.get_text() and t.get_visible()]
        tb = [t.get_window_extent(rend) for t in ticks]
        if tb and abs(min(b.x0 for b in tb) - bb.x0) < 1.0 and not any(
                t.get_text().lstrip().startswith(("-", "\u2212")) for t in ticks):
            bb = Bbox.from_extents(bb.x0 - _pl_minus_width(fig, rend, ticks[0].get_fontsize()),
                                   bb.y0, bb.x1, bb.y1)
    return bb, bool(own.get(ax))


def _pl_all_boxes(fig, rend, mine):
    """Everything already drawn, as (box, owner, kind). A letter sits where its own panel keeps
    its y-axis label and x10^n offset, so those two are skipped for that panel only."""
    out = []
    for a in fig.axes:
        out.append((a.get_window_extent(rend), a, "axes"))
        out += [(b, a, "tick") for b in _pl_tick_boxes(a, rend)]
        for t in list(a.texts):
            if t is not None and t.get_text():
                out.append((t.get_window_extent(rend), a, "tick"))
        for t, kind in ((a.yaxis.label, "corner"), (a.yaxis.get_offset_text(), "corner"),
                        (a.xaxis.get_offset_text(), "corner"), (a.xaxis.label, "tick")):
            if t is not None and t.get_text():
                out.append((t.get_window_extent(rend), a, kind))
    out += [(t.get_window_extent(rend), None, "tick") for t in fig.texts
            if t not in mine and t.get_text()]
    return [(b, o, k) for b, o, k in out if b is not None and b.width > 0 and b.height > 0]


def _pl_shrink(bb, d):
    """A box pulled in by d on every side, so boxes that merely touch do not count as overlapping."""
    return Bbox.from_extents(bb.x0 + d, bb.y0 + d, bb.x1 - d, bb.y1 - d)


def _pl_cluster(values, tol):
    """Map each index to the indices of its cluster (values within tol of each other)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    rep, group = {}, []
    for i in order:
        if group and values[i] - values[group[-1]] > tol:
            for j in group:
                rep[j] = group
            group = []
        group.append(i)
    for j in group:
        rep[j] = group
    return rep


def draw_deferred_labels(fig, *, size_factor=1.45, up_in=0.25, left_in=0.0, left_frac=0.0,
                         down_offset_in=0.0, minus_compensate=False, align_rows=True,
                         align_tol_in=0.8, touch_tol_in=0.12, clear_in=0.15,
                         faint_inset_in=0.04) -> None:
    """Place every letter recorded by defer_label() for `fig`. Call once, after everything else
    is drawn (the faint tints included) and before savefig.

      size_factor      letter size as a multiple of the size the figure asked for
      up_in            raise every letter above the top of its block (inches)
      left_in          push every letter left of its block (inches)
      left_frac        push every letter left by this fraction of its own width
      down_offset_in   lower the letters of panels that print an x10^n offset (inches)
      minus_compensate line the letters up as if the y tick numbers carried a minus sign
      align_rows       share one letter line per row / one letter x per column
    """
    items = _DEFERRED_LABELS.pop(fig, [])
    if not items:
        return
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    labelled = {ax for ax, _, _ in items}
    mine = set()
    boxes = _pl_all_boxes(fig, rend, mine)
    inv = fig.transFigure.inverted()
    tol, pad = align_tol_in * fig.dpi, clear_in * fig.dpi
    own = _pl_owners(fig, rend, labelled)
    blocks = [_pl_block(fig, rend, ax, own, minus_compensate) for ax, _, _ in items]

    xs = [b.x0 for b, _ in blocks]
    xrep = _pl_cluster(xs, tol) if align_rows else {i: [i] for i in range(len(xs))}
    xs = [min(xs[j] for j in xrep[i]) for i in range(len(xs))]

    def _test_box(t):
        b = t.get_window_extent(rend)
        return _pl_shrink(b, min(touch_tol_in * fig.dpi, 0.25 * min(b.width, b.height)))

    # first pass: where can each letter go, and how high must its top be
    plan = []
    for i, (ax, text, size) in enumerate(items):
        bb, has_cbar = blocks[i]
        fs = (size if size is not None else LABEL_FONTSIZE) * size_factor
        t = fig.text(float(inv.transform((xs[i], 0.0))[0]),
                     float(inv.transform((0.0, bb.y1))[1]), text, ha="left", va="top",
                     fontweight="bold", fontfamily=LABEL_FONTFAMILY, fontsize=fs, zorder=40)
        lb = _test_box(t)
        x_px, top = xs[i], bb.y1
        hit = [b for b, o, k in boxes if lb.overlaps(b) and not (o is ax and k == "corner")]
        if hit and has_cbar:
            # the colorbar label reaches the corner: keep the letter on the label line, slide left
            x_try = min(b.x0 for b in hit) - pad - lb.width
            t.set_position((float(inv.transform((x_try, 0.0))[0]),
                            float(inv.transform((0.0, top))[1])))
            lb2 = _test_box(t)
            if not [b for b, o, k in boxes if lb2.overlaps(b) and not (o is ax and k == "corner")]:
                x_px, hit = x_try, []
        if hit:
            top = bb.y1 + t.get_window_extent(rend).height     # no free corner: lift it clear
        plan.append([t, x_px, top])
        t.set_visible(False)

    # second pass: one letter line per row, then keep every letter inside its faint tint
    tops = [pl[2] for pl in plan]
    rows = [0.5 * (ax.get_window_extent(rend).y0 + ax.get_window_extent(rend).y1)
            for ax, _, _ in items]
    yrep = _pl_cluster(rows, tol) if align_rows else {i: [i] for i in range(len(items))}
    faints = [a.get_window_extent(rend) for a in fig.artists if isinstance(a, FancyBboxPatch)]
    inset = faint_inset_in * fig.dpi
    limits = []
    for ax, _, _ in items:
        ab = ax.get_window_extent(rend)
        cand = [f.y1 - inset for f in faints
                if f.x0 <= 0.5 * (ab.x0 + ab.x1) <= f.x1 and f.y0 <= ab.y0 and ab.y1 <= f.y1]
        limits.append(min(cand) if cand else None)
    for i, (t, x_px, _) in enumerate(plan):
        top = max(tops[j] for j in yrep[i]) + up_in * fig.dpi
        lim = [limits[j] for j in yrep[i] if limits[j] is not None]
        if lim:
            top = min(top, min(lim))
        if down_offset_in and items[i][0].yaxis.get_offset_text().get_text():
            top -= down_offset_in * fig.dpi
        t.set_visible(True)
        dx = left_in * fig.dpi + left_frac * t.get_window_extent(rend).width
        t.set_position((float(inv.transform((x_px - dx, 0.0))[0]),
                        float(inv.transform((0.0, top))[1])))
        mine.add(t)
        boxes.append((t.get_window_extent(rend), None, "tick"))


def add_label(src_png: Path, out_path: Path, label) -> None:
    """Bake a single Nature-style panel label onto an existing PNG and save it."""
    img = plt.imread(str(src_png))
    aspect = img.shape[1] / img.shape[0]
    fig = plt.figure(figsize=(PANEL_HEIGHT_IN * aspect, PANEL_HEIGHT_IN))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.imshow(img, aspect="auto", interpolation="none")
    ax.axis("off")
    place_label(ax, label)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# ============================================================
# Case configs + field loading (DSMC ensemble + analytic)
# ============================================================

def build_ar_cfg(aspect_ratio: int):
    """AR-study case (Kn = 100, kappa = 0.05)."""
    pressure = float(post.KN_REF * post.PRESSURE_REF_AT_KN_REF / 100.0)
    return acs.build_ar_case_config(
        aspect_ratio=aspect_ratio, reaction_probability=0.05,
        pressure=pressure, knudsen_number=100.0,
    )


def build_kn_cfg(kn_value):
    """Kn-study case (aspect ratio 5, kappa = 0.05). kn_value is the folder name string."""
    kn = float(kn_value)
    pressure = float(post.KN_REF * post.PRESSURE_REF_AT_KN_REF / kn)
    cfg = acs.build_ar_case_config(
        aspect_ratio=5, reaction_probability=0.05, pressure=pressure, knudsen_number=kn,
    )
    return dataclasses.replace(cfg, study="Kn", case_name=str(kn_value))


def _dsmc_fields(case_cfg):
    """DSMC fields for a case: average over the runs of the case in the dataset.
    Reports how many runs were averaged."""
    ens = post.build_ensemble_average_result(case_cfg)
    print(f"[ensemble] {case_cfg.study}/{case_cfg.case_name}: "
          f"averaged over {ens.info.case_count} DSMC run(s)", flush=True)
    return ens.sol, ens.fields, ens.diff


def load_case(case_cfg) -> dict:
    """Load DSMC + analytic fields and the plot settings for one case."""
    sol_d, fields_d, diff_d = _dsmc_fields(case_cfg)
    sol_a, fields_a, diff_a, _c, _sf, _sr = acs.solve_analytic_case(case_cfg, case_max_workers=8)
    _, _, minline_d = analytic.compute_min_diagnostics(sol_d, fields_d)
    _, _, minline_a = analytic.compute_min_diagnostics(sol_a, fields_a)
    settings = analytic.get_case_plot_settings(case_cfg.aspect_ratio)
    return dict(
        sol_d=sol_d, sol_a=sol_a, fields_d=fields_d, fields_a=fields_a,
        diff_d=diff_d, diff_a=diff_a, minline_d=minline_d, minline_a=minline_a,
        cfg=settings["contour"], ccfg=settings["centerline"],
        Lref=2.0 * float(sol_d["h"]), Lm=float(sol_d["Lm"]),
        iy_d=int(np.argmin(np.abs(sol_d["y"]))), iy_a=int(np.argmin(np.abs(sol_a["y"]))),
    )


# ============================================================
# Centerline drawing primitives (compare: DSMC red, analytic blue)
# ============================================================

def _style_centerline(ax, d, mass_fraction: bool, show_xlabel: bool = True, box_lw=None) -> None:
    AR = d["Lm"] / d["Lref"]
    ccfg = d["ccfg"]
    ax.set_xlabel(r"$x/(2h)$" if show_xlabel else "")
    ax.set_xlim(-0.05 * AR, 1.05 * AR)
    ax.xaxis.set_major_locator(mticker.FixedLocator([0.0, AR / 2.0, AR]))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
    if mass_fraction:
        analytic.set_centerline_mass_fraction_axis(ax, axis="y")
    else:
        analytic.apply_sci_offset(ax, axis="y", scilimits=(0, 0))
        analytic.set_centerline_minmax_ticks(ax, axis="y", nticks=5, scientific=True)
    blw = ccfg["box_linewidth"] if box_lw is None else box_lw
    analytic.apply_axes_style(ax, blw, ccfg["tick_length"], ccfg["tick_width"])


def draw_centerline_jx(ax, d, lw=None, show_xlabel=True, show_ylabel=True, box_lw=None, dsmc_extra=1.0) -> None:
    """Centerline diffusive mass flux j_x: reactant A (solid) and product B (dashed),
    DSMC (red) vs analytic (blue). lw overrides the analytic line width (None = case
    default); the DSMC (red) line is dsmc_extra points thicker than the analytic one."""
    Lref = d["Lref"]
    xd = np.asarray(d["sol_d"]["x"]) / Lref
    xa = np.asarray(d["sol_a"]["x"]) / Lref
    iy_d, iy_a = d["iy_d"], d["iy_a"]
    lw = float(d["ccfg"]["centerline_linewidth"] if lw is None else lw)
    lwd = lw + dsmc_extra
    cd, ca = post.DSMC_COMPARE_COLOR, post.ANALYTIC_COMPARE_COLOR
    ax.plot(xd, d["diff_d"]["A"]["ju_diff"][iy_d, :], color=cd, ls="-", lw=lwd)
    ax.plot(xd, d["diff_d"]["B"]["ju_diff"][iy_d, :], color=cd, ls="--", lw=lwd)
    ax.plot(xa, d["diff_a"]["A"]["ju_diff"][iy_a, :], color=ca, ls="-", lw=lw)
    ax.plot(xa, d["diff_a"]["B"]["ju_diff"][iy_a, :], color=ca, ls="--", lw=lw)
    jx_lab = (analytic.field_symbol("ju_diff", sp="A").rstrip("$") + r",\ "
              + analytic.field_symbol("ju_diff", sp="B").lstrip("$"))
    ax.set_ylabel(jx_lab + r" (kg m$^{-2}$ s$^{-1}$)" if show_ylabel else "")
    _style_centerline(ax, d, mass_fraction=False, show_xlabel=show_xlabel, box_lw=box_lw)


def draw_centerline_Y(ax, d, lw=None, show_xlabel=True, show_ylabel=True, box_lw=None, dsmc_extra=1.0) -> None:
    """Centerline mass fraction: Y_A (solid) and Y_B (dashed), DSMC (red) vs analytic
    (blue), with star markers at the Y_A minimum and Y_B maximum. lw overrides the
    analytic line width; the DSMC (red) line is dsmc_extra points thicker."""
    Lref = d["Lref"]
    xd = np.asarray(d["sol_d"]["x"]) / Lref
    xa = np.asarray(d["sol_a"]["x"]) / Lref
    iy_d, iy_a = d["iy_d"], d["iy_a"]
    lw = float(d["ccfg"]["centerline_linewidth"] if lw is None else lw)
    lwd = lw + dsmc_extra
    cd, ca = post.DSMC_COMPARE_COLOR, post.ANALYTIC_COMPARE_COLOR
    YAd, YBd = d["fields_d"]["YA"][iy_d, :], d["fields_d"]["YB"][iy_d, :]
    YAa, YBa = d["fields_a"]["YA"][iy_a, :], d["fields_a"]["YB"][iy_a, :]
    ax.plot(xd, YAd, color=cd, ls="-", lw=lwd)
    ax.plot(xd, YBd, color=cd, ls="--", lw=lwd)
    ax.plot(xa, YAa, color=ca, ls="-", lw=lw)
    ax.plot(xa, YBa, color=ca, ls="--", lw=lw)
    ms = lw * 5.0
    ax.plot(xd[np.nanargmin(YAd)], YAd[np.nanargmin(YAd)], marker="*", ls="None", ms=ms + 1, color=cd)
    ax.plot(xd[np.nanargmax(YBd)], YBd[np.nanargmax(YBd)], marker="*", ls="None", ms=ms + 1, color=cd)
    ax.plot(xa[np.nanargmin(YAa)], YAa[np.nanargmin(YAa)], marker="*", ls="None", ms=ms, color=ca)
    ax.plot(xa[np.nanargmax(YBa)], YBa[np.nanargmax(YBa)], marker="*", ls="None", ms=ms, color=ca)
    ax.set_ylabel(r"$Y_{\mathrm{A}},\ Y_{\mathrm{B}}$" if show_ylabel else "")
    _style_centerline(ax, d, mass_fraction=True, show_xlabel=show_xlabel, box_lw=box_lw)


# ============================================================
# Contour drawing primitives (DSMC top / analytic bottom split)
# ============================================================

def style_contour_axes(ax, AR: float, cfg, zoom_left=None, show_xlabel=True, show_ylabel=True,
                       show_yticklabels=True, show_xticklabels=True, box_lw=None,
                       xtick_pad=None, ytick_pad=None) -> None:
    ax.set_aspect("auto")
    ax.set_xlabel(r"$x/(2h)$" if show_xlabel else "")
    ax.set_ylabel(r"$y/(2h)$" if show_ylabel else "")
    ax.set_ylim(-0.5, 0.5)
    ax.yaxis.set_major_locator(mticker.FixedLocator([-0.5, 0.0, 0.5]))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
    if not show_yticklabels:
        ax.set_yticklabels([])     # keep tick marks, drop the numbers (shared with j_x,A)
    if zoom_left is None:
        ax.set_xlim(0.0, AR)
        nx = 3 if AR <= 1.5 else (4 if AR <= 3.5 else 6)
        ax.xaxis.set_major_locator(mticker.FixedLocator(np.linspace(0.0, AR, nx)))
    else:
        ax.set_xlim(zoom_left, AR)
        nz = {1: 2, 3: 3, 5: 3, 10: 5}.get(int(round(AR)), max(3, min(6, int(round(AR - zoom_left)) + 2)))
        ax.xaxis.set_major_locator(mticker.FixedLocator(np.round(np.linspace(zoom_left, AR, nz), 1)))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
    if not show_xticklabels:
        ax.set_xticklabels([])     # keep tick marks, drop the numbers (shared x across the stack)
    blw = cfg["box_linewidth"] if box_lw is None else box_lw
    analytic.apply_axes_style(ax, blw, cfg["tick_length"], cfg["tick_width"], tick_pad=6)
    if xtick_pad is not None:
        ax.tick_params(axis="x", pad=xtick_pad)
    if ytick_pad is not None:
        ax.tick_params(axis="y", pad=ytick_pad)


def draw_split_scalar(ax, data, Zd, Za, powernorm_gamma=None, xlim_si=None, show_minline=True,
                      centerline_lw=None, minline_lw=None, minline_color="k"):
    """Draw a DSMC(top)/analytic(bottom) split contourf into ax; return (sm, vmin, vmax, zoom_left)."""
    Lref, Lm = data["Lref"], data["Lm"]
    Xt, Yt, Zt = post.split_sol_field(data["sol_d"], Zd, upper=True)
    Xb, Yb, Zb = post.split_sol_field(data["sol_a"], Za, upper=False)
    zoom_left = None
    if xlim_si is not None:
        zoom_left = round(float(xlim_si[0]) / Lref, 1)
        win = (zoom_left * Lref, Lm)
        Xt, Yt, Zt = analytic.crop_contour_to_x_window(Xt, Yt, Zt, xlim=win)
        Xb, Yb, Zb = analytic.crop_contour_to_x_window(Xb, Yb, Zb, xlim=win)
    lv, vmin, vmax = post.compare_level_values(Zt, Zb, data["cfg"]["levels"])
    if powernorm_gamma is None or np.isclose(vmin, vmax):
        norm = mcolors.Normalize(vmin=float(vmin), vmax=float(vmax))
    else:
        norm = mcolors.PowerNorm(gamma=float(powernorm_gamma), vmin=float(vmin), vmax=float(vmax))
    ax.contourf(Xt / Lref, Yt / Lref, Zt, levels=lv, cmap="jet", norm=norm)
    ax.contourf(Xb / Lref, Yb / Lref, Zb, levels=lv, cmap="jet", norm=norm)
    clw = data["cfg"]["box_linewidth"] * 0.5 if centerline_lw is None else centerline_lw
    mlw = data["cfg"]["minline_linewidth"] if minline_lw is None else minline_lw
    ax.axhline(0.0, color="k", linewidth=clw, zorder=10)
    if show_minline:
        post.plot_half_minline(ax, post._scale_minline(data["minline_d"], Lref), upper=True, linewidth=mlw, color=minline_color)
        post.plot_half_minline(ax, post._scale_minline(data["minline_a"], Lref), upper=False, linewidth=mlw, color=minline_color)
    sm = ScalarMappable(norm=norm, cmap="jet")
    sm.set_array([])
    return sm, float(vmin), float(vmax), zoom_left


def draw_split_inversion(ax, data, show_minline=True, centerline_lw=None, minline_lw=None,
                         minline_color="k") -> None:
    Lref = data["Lref"]
    sd = np.where(np.asarray(data["sign_d"]) > 0, 1, -1).astype(int)
    sa = np.where(np.asarray(data["sign_a"]) > 0, 1, -1).astype(int)
    Xt, Yt, Zt = post.split_sol_field(data["sol_d"], sd, upper=True)
    Xb, Yb, Zb = post.split_sol_field(data["sol_a"], sa, upper=False)
    cmap = mcolors.ListedColormap([analytic.CUSTOM_COLORS[4], analytic.CUSTOM_COLORS[0]])
    lv = [-1.5, 0.5, 1.5]
    ax.contourf(Xt / Lref, Yt / Lref, Zt, levels=lv, cmap=cmap)
    ax.contourf(Xb / Lref, Yb / Lref, Zb, levels=lv, cmap=cmap)
    clw = data["cfg"]["box_linewidth"] * 0.5 if centerline_lw is None else centerline_lw
    mlw = data["cfg"]["minline_linewidth"] if minline_lw is None else minline_lw
    ax.axhline(0.0, color="k", linewidth=clw, zorder=10)
    if show_minline:
        post.plot_half_minline(ax, post._scale_minline(data["minline_d"], Lref), upper=True, linewidth=mlw, color=minline_color)
        post.plot_half_minline(ax, post._scale_minline(data["minline_a"], Lref), upper=False, linewidth=mlw, color=minline_color)


# ============================================================
# Faint "grouping" panels (behind a set of subplots).
# ============================================================
def group_bbox_frac(fig, axes, renderer=None, texts=()):
    """Union of the RENDERED TIGHT bboxes of `axes` (+ optional `texts`), in figure-fraction
    coords. Encloses every drawn region: tick labels, axis labels, colorbars, panel letters."""
    if renderer is None:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
    bbs = [a.get_tightbbox(renderer) for a in axes]
    bbs += [t.get_window_extent(renderer) for t in texts if t is not None]
    bbs = [b for b in bbs if b is not None]
    return Bbox.union(bbs).transformed(fig.transFigure.inverted())


def draw_faint_panel(fig, bbox, color, fw, fh, alpha=0.15, pad_in=0.18, rounding=0.008, zorder=-20):
    """Faint rounded rectangle behind a group of panels (bbox: figure-fraction Bbox)."""
    x0, y0, x1, y1 = bbox.extents
    dx, dy = pad_in / fw, pad_in / fh
    fig.add_artist(FancyBboxPatch((x0 - dx, y0 - dy), (x1 - x0) + 2 * dx, (y1 - y0) + 2 * dy,
                                  boxstyle=f"round,pad=0,rounding_size={rounding}",
                                  transform=fig.transFigure, facecolor=color, edgecolor=color,
                                  alpha=alpha, linewidth=0.0, zorder=zorder, clip_on=False,
                                  mutation_aspect=fh / fw))
    return bbox


def draw_group_arrow(fig, x, y_from, y_to, color, scale=42.0, lw=3.0):
    """Vertical arrow between two grouped regions (figure-fraction coords)."""
    fig.add_artist(FancyArrowPatch((x, y_from), (x, y_to), transform=fig.transFigure,
                                   arrowstyle="-|>", mutation_scale=scale, linewidth=lw,
                                   color=color, zorder=61, clip_on=False))


def add_colorbar(fig, cax, sm, vmin, vmax, title, scientific, cfg, orientation="vertical",
                 nticks=None, decimals=None, label_dy_in=0.0, box_lw=None, unit=None):
    dec = cfg["cbar_decimals"] if decimals is None else int(decimals)
    # Scientific colorbars carry the power of ten INSIDE the label's parentheses,
    # e.g. "n_A (x10^23)", instead of a separate floating x10^n offset text. The tick
    # numbers are then shown as the mantissa (value / 10^exp).
    exp = None
    if scientific:
        mx = max(abs(float(vmin)), abs(float(vmax)))
        exp = int(np.floor(np.log10(mx))) if mx > 0 else 0
        fmt = mticker.FuncFormatter(lambda v, pos, e=exp: f"{v / 10.0 ** e:.{dec}f}")
    else:
        fmt = mticker.FormatStrFormatter(f"%.{dec}f")     # plain decimals, no x10^n
    cb = fig.colorbar(sm, cax=cax, format=fmt, orientation=orientation)
    n = int(nticks) if nticks is not None else cfg["cbar_nticks"]
    ticks = [vmin] if np.isclose(vmin, vmax) else np.linspace(vmin, vmax, n)
    cb.set_ticks(ticks)
    blw = cfg["box_linewidth"] if box_lw is None else box_lw
    analytic.apply_axes_style(cb.ax, blw, cfg["tick_length"], cfg["tick_width"], tick_pad=6)
    # Nature style: units in parentheses in the label, merged with the power of ten when present,
    # e.g. "n_A (x10^23 m^-3)"; without a unit the "(x10^n)" form is kept.
    if exp is not None and exp != 0:
        inner = rf"$\times 10^{{{exp}}}$" + (f" {unit}" if unit else "")
        title_disp = f"{title} ({inner})"
    elif unit:
        title_disp = f"{title} ({unit})"
    else:
        title_disp = title
    if orientation == "horizontal":
        # Numbers ABOVE the bar; the quantity label centred ABOVE the numbers
        # (offset by label_dy_in inches so it never overlaps them).
        cb.ax.xaxis.set_ticks_position("top")
        cb.set_label("")
        trans = cb.ax.transAxes + ScaledTranslation(0.0, label_dy_in, fig.dpi_scale_trans)
        cb.ax.text(0.5, 1.0, title_disp, transform=trans, ha="center", va="bottom")
    else:
        cb.set_label(title_disp, rotation=90, labelpad=4)
    return cb
