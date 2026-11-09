#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auxiliary schematic -- along-wall coordinate direction.

Uses the trench geometry of figure_schematic_base.py, but omits the inlet/reaction
annotation. The wall-coordinate origin is at the trench base; the upper wall
direction is positive and the lower wall direction is negative.

Outputs:
  figures/figure_along_the_wall_schmatic.png

    python3 figure_along_the_wall_schmatic.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import figure_schematic_base as base


OUT_STEM = "figure_along_the_wall_schmatic"
WALL_COORD_LABEL = r"$\ell = 0$"
WALL_ARROW_WIDTH = 3.2
WALL_ARROW_HEAD_LEN = 14.0
WALL_ARROW_HEAD_HALF_HEIGHT = 7.0
WALL_ARROW_AXIS_GAP = 7.0
# >>> gap (page-pt) between each along-wall arrow's horizontal segment and ITS side wall. The two
#     arrows are placed the SAME distance outside their walls, so they no longer touch the wall
#     lines and the top/bottom spacing is symmetric.
WALL_ARROW_WALL_GAP = 12.0
# >>> shorten the (elongated) trench by moving its right side LEFT this many ppt-x units. The
#     opening/left side stays put; walls, base, coordinate axis and the right-hand labels shift by
#     the same amount so arrow-head SHAPES are preserved (translated, not squished). 0 = original.
SHORTEN_PT = 110.0
# >>> linewidth (page-pt) for the four lines that DEFINE the trench (opening / upper wall / lower
#     wall / base) -- kept thicker than the coordinate-axis strokes (~6.0 pt, set by the hard-coded
#     axis polygon widths below) so the trench geometry visually reads as the "primary" shape.
TRENCH_LW = 9.0
# >>> x position of the "trench opening" line -- the CENTRE of the y-axis polygon (see
#     add_axes_shortened: x in [144.257812, 150.25]) so the opening sits exactly on the axis
#     instead of a few tenths of a point off to one side. Both walls start exactly here too,
#     so the opening's thickness is centred symmetrically over both wall tips.
OPENING_X = (144.257812 + 150.25) / 2.0
# >>> the x-axis's page-y (shared with add_l_shaped_wall_arrows, and with the "Trench
#     opening"/"Trench base" label vertical placement below).
AXIS_Y = 121.089844
# >>> "Trench opening"/"Trench base" label placement. Horizontally, each sits LABEL_LINE_GAP away
#     from its own line -- measured from the label's near-edge GLYPH, which is exactly what
#     ha="left" (opening, near edge = the leading "T") / ha="right" (base, near edge = the trailing
#     "e") give. Vertically, each sits exactly
#     midway between the x-axis and its own wall line -- opening between the x-axis and the upper
#     wall, base between the x-axis and the lower wall -- see draw_schematic.
LABEL_LINE_GAP = 5.0
# >>> Side-wall label positions (page pt; increasing y moves the text DOWN). This ONE place controls
#     the labels in BOTH Fig. 4 and ED 6 (they embed this schematic). Rerun figure4.py + extended_figure6.py.
UPPER_WALL_LABEL_XY = (210.0, 52)
LOWER_WALL_LABEL_XY = (210.0, 202.682660)
# Boundary colours matched to the contribution graphs (analytic.CUSTOM_COLORS).
COL_OPENING = (0.8500, 0.3250, 0.0980)   # orange (opening / jet contribution)
COL_UPPER = (0.4660, 0.6740, 0.1880)     # green  (upper wall contribution)
COL_LOWER = (0.9290, 0.6940, 0.1250)     # yellow (lower wall contribution)
COL_BASE = (0.0000, 0.4470, 0.7410)      # blue   (base / end contribution)


def add_line_exact(ax, p0, p1, color, lw: float, zorder: int) -> None:
    """Like base.add_line, but with snap=False. matplotlib snaps axis-aligned Line2D strokes to
    the nearest pixel by default (to keep them crisp); that snapping is what left the opening/base
    caps looking a fraction of a pixel short of the wall's own (unsnapped, antialiased) edge even
    though their computed positions are mathematically exact -- see draw_schematic. Disabling snap
    forces the true sub-pixel position, matching the wall's antialiasing exactly."""
    ax.plot(
        [p0[0], p1[0]],
        [p0[1], p1[1]],
        color=color,
        linewidth=lw,
        solid_capstyle="butt",
        solid_joinstyle="miter",
        snap=False,
        zorder=zorder,
    )


def add_l_shaped_arrow(ax, x0: float, y_start: float, x_tip: float, y_corner: float) -> None:
    """One elbow arrow: a vertical segment down to the corner, then a horizontal segment to the
    arrowhead. The two segments are ONE Line2D (a single stroke, not two abutting filled
    rectangles), with a rounded join at the corner so the bend reads as one continuous line
    instead of two straight pieces glued together; the open ends stay flat (butt) so they meet
    the axis gap and the arrowhead cleanly."""
    x_head_base = x_tip + WALL_ARROW_HEAD_LEN
    ax.plot(
        [x0, x0, x_head_base],
        [y_start, y_corner, y_corner],
        color=base.BLACK,
        linewidth=WALL_ARROW_WIDTH,
        solid_capstyle="butt",
        solid_joinstyle="round",
        zorder=7,
    )
    base.add_polygon(
        ax,
        [
            (x_tip, y_corner),
            (x_head_base, y_corner - WALL_ARROW_HEAD_HALF_HEIGHT),
            (x_head_base, y_corner + WALL_ARROW_HEAD_HALF_HEIGHT),
        ],
        base.BLACK,
        zorder=7,
    )


def arrow_row_ys() -> tuple[float, float]:
    """The along-wall arrows' row positions (page pt): OUTSIDE their own side wall, i.e. the
    stacking order (centre outward) is wall line -> arrow -> label."""
    upper_wall_y = base.ppt_xy(0.0, base.UPPER_WALL_PPT_Y)[1]   # visually TOP wall (page y)
    lower_wall_y = base.ppt_xy(0.0, base.LOWER_WALL_PPT_Y)[1]   # visually BOTTOM wall (page y)
    top_y = upper_wall_y - WALL_ARROW_WALL_GAP
    bottom_y = lower_wall_y + WALL_ARROW_WALL_GAP
    return top_y, bottom_y


def add_l_shaped_wall_arrows(ax) -> None:
    x0 = 506.0 - SHORTEN_PT      # right (base) end of the along-wall arrows moves in with the base
    top_y, bottom_y = arrow_row_ys()
    left_x = 182.0               # left (opening) tip is fixed

    add_l_shaped_arrow(ax, x0, AXIS_Y - WALL_ARROW_AXIS_GAP, left_x, top_y)
    add_l_shaped_arrow(ax, x0, AXIS_Y + WALL_ARROW_AXIS_GAP, left_x, bottom_y)


def add_wall_direction_labels(ax) -> None:
    # Vertically CENTRED on the arrow row (va="center", not "baseline") so the "+"/"-" glyphs are
    # centred on the arrow itself rather than offset by the font's baseline-to-cap-height quirks.
    top_y, bottom_y = arrow_row_ys()
    ax.text(164.0, top_y, "+", ha="left", va="center", color=base.BLACK,
             fontsize=base.FONT_SIZE, fontfamily=base.FONT, zorder=8)
    ax.text(164.0, bottom_y, "-", ha="left", va="center", color=base.BLACK,
             fontsize=base.FONT_SIZE, fontfamily=base.FONT, zorder=8)


def add_wall_coordinate_label(ax) -> None:
    ax.text(
        511.0 - SHORTEN_PT,
        109.0,
        WALL_COORD_LABEL,
        ha="left",
        va="baseline",
        color=base.BLACK,
        fontsize=base.FONT_SIZE,
        fontfamily=base.FONT,
        zorder=8,
    )


def add_axes_shortened(ax) -> None:
    """Coordinate axes with the x-axis (only) shortened by SHORTEN_PT on the right. The shafts are
    thin Line2D strokes at the SAME width as the along-wall (ell) arrows, and the heads reuse the
    ell arrows' head dimensions, so the axes read at the same visual weight as the ell lines."""
    s = SHORTEN_PT
    x_c = OPENING_X                       # y-axis centreline (= 147.253906)
    # y-axis: thin shaft + ell-sized up arrow head.
    ax.plot([x_c, x_c], [221.382812, 20.562500 + WALL_ARROW_HEAD_LEN],
            color=base.BLACK, linewidth=WALL_ARROW_WIDTH,
            solid_capstyle="butt", zorder=2)
    base.add_polygon(ax, [(x_c - WALL_ARROW_HEAD_HALF_HEIGHT, 20.562500 + WALL_ARROW_HEAD_LEN),
                          (x_c, 20.562500),
                          (x_c + WALL_ARROW_HEAD_HALF_HEIGHT, 20.562500 + WALL_ARROW_HEAD_LEN)],
                     base.BLACK, zorder=2)
    # x-axis: left end fixed, right end pulled in by s; ell-sized right arrow head.
    tip_x = 568.824219 - s
    ax.plot([99.566406, tip_x - WALL_ARROW_HEAD_LEN], [AXIS_Y, AXIS_Y],
            color=base.BLACK, linewidth=WALL_ARROW_WIDTH,
            solid_capstyle="butt", zorder=2)
    base.add_polygon(ax, [(tip_x - WALL_ARROW_HEAD_LEN, AXIS_Y - WALL_ARROW_HEAD_HALF_HEIGHT),
                          (tip_x, AXIS_Y),
                          (tip_x - WALL_ARROW_HEAD_LEN, AXIS_Y + WALL_ARROW_HEAD_HALF_HEIGHT)],
                     base.BLACK, zorder=2)


def add_axis_labels_shortened(ax) -> None:
    base.add_text(ax, 561.795814 - SHORTEN_PT, 143.637487, r"$x$", base.BLACK)   # by the (moved) x tip
    base.add_text(ax, 131.557631, 16.708736, r"$y$", base.BLACK)


def draw_schematic(ax=None) -> plt.Figure:
    """Draw the along-wall schematic. With ax=None a standalone figure is made; pass an existing
    Axes to embed it (e.g. next to another figure's panel) -- text keeps its absolute point size,
    so it does NOT shrink when the host axes is smaller. The axes must have the schematic aspect
    (WIDTH_PT : HEIGHT_PT) to avoid distortion."""
    own = ax is None
    if own:
        plt.rcParams.update(
            {
                "font.family": base.FONT,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
                # Math (italic ell) in Arial too, matching every other figure.
                "mathtext.fontset": "custom",
                "mathtext.rm": base.FONT,
                "mathtext.it": f"{base.FONT}:italic",
                "mathtext.bf": f"{base.FONT}:bold",
                "mathtext.cal": base.FONT,
            }
        )
        fig = plt.figure(
            figsize=(base.WIDTH_PT / 72.0, base.HEIGHT_PT / 72.0),
            dpi=base.DPI,
        )
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    else:
        fig = ax.figure
    ax.set_xlim(0, base.WIDTH_PT)
    ax.set_ylim(base.HEIGHT_PT, 0)
    ax.axis("off")

    s = SHORTEN_PT
    base_x = 489.838277 - s      # trench-base x position; walls must stop HERE, not past it
    # Trench base and opening are the two "end caps" of the rectangle: each spans exactly from the
    # OUTER edge of the upper wall's stroke to the OUTER edge of the lower wall's stroke, with no
    # margin and no overshoot. Computed in PAGE space (post ppt_xy) and offset by TRENCH_LW/2
    # directly THERE -- offsetting before ppt_xy would get the offset scaled by PPT_SCALE (~0.9985),
    # falling ~0.006pt short of the wall's true edge, which antialiasing renders as a visible mistint
    # right at the tip even though it looks like an exact match away from the corner.
    opening_x_page, _ = base.ppt_xy(OPENING_X, 0.0)
    base_x_page, _ = base.ppt_xy(base_x, 0.0)
    upper_wall_page_y = base.ppt_xy(0.0, base.UPPER_WALL_PPT_Y)[1]
    lower_wall_page_y = base.ppt_xy(0.0, base.LOWER_WALL_PPT_Y)[1]
    # The wall strokes are TRENCH_LW PHYSICAL points wide, but the cap endpoints live in DATA
    # units. When the schematic is embedded at a scale other than 1 data-unit = 1 pt (Fig. 4 / ED 6
    # scale it by their FONT ratio), a plain TRENCH_LW/2 data offset misses the walls' true rendered
    # edge and the opening/base caps stick out past (or fall short of) the walls. Convert the
    # physical half-width into data units through the axes' actual y scale so the caps end EXACTLY
    # on the walls' outer edges at any embed scale.
    ax_h_in = ax.get_position().height * fig.get_size_inches()[1]
    pt_per_data = ax_h_in * 72.0 / base.HEIGHT_PT
    wall_half_data = (TRENCH_LW / 2.0) / pt_per_data
    cap_y_top = upper_wall_page_y - wall_half_data
    cap_y_bot = lower_wall_page_y + wall_half_data

    # Trench base (vertical line) -- moved LEFT with the shortening. Drawn TOPMOST (highest zorder
    # of anything in the schematic) so neither the coordinate axes nor the side walls cover it.
    # All four boundary lines (base/opening/upper/lower) use add_line_exact (snap=False) so their
    # antialiased edges are all computed from the same true sub-pixel position, instead of the
    # walls snapping to one pixel and the base/opening caps snapping to another.
    add_line_exact(
        ax,
        (base_x_page, cap_y_top),
        (base_x_page, cap_y_bot),
        COL_BASE,
        lw=TRENCH_LW,
        zorder=9,
    )
    add_axes_shortened(ax)

    # Trench side walls and opening (left, fixed) -- thicker than the coordinate axes so the
    # trench geometry reads as the primary shape. Side-wall right ends are clamped to base_x (the
    # base's own centreline) so they never cross past the base line; their left ends
    # both start exactly at OPENING_X, flush with the opening line's centreline.
    add_line_exact(
        ax,
        base.ppt_xy(OPENING_X, 56.280393),
        base.ppt_xy(base_x, 56.280393),
        COL_LOWER,
        lw=TRENCH_LW,
        zorder=3,
    )
    add_line_exact(
        ax,
        base.ppt_xy(OPENING_X, 151.078336),
        base.ppt_xy(base_x, 151.078336),
        COL_UPPER,
        lw=TRENCH_LW,
        zorder=3,
    )
    add_line_exact(
        ax,
        (opening_x_page, cap_y_top),
        (opening_x_page, cap_y_bot),
        COL_OPENING,
        lw=TRENCH_LW,
        zorder=4,
    )

    # Trench-wall labels. "Trench opening"/"Trench base" sit LABEL_LINE_GAP from their own line
    # (measured via ha="left"'s "T" for opening, ha="right"'s "e" for base) and, vertically, have
    # EQUAL whitespace above/below between the x-axis and their own wall line. That means centring
    # on the glyph ink itself (va="center", using matplotlib's actual rendered bbox), NOT on the
    # baseline -- with no descenders in either label, the ink sits entirely above the baseline, so
    # baseline-at-the-midpoint would put the ink noticeably above centre.
    opening_label_y = (upper_wall_page_y + AXIS_Y) / 2.0
    base_label_y = (lower_wall_page_y + AXIS_Y) / 2.0
    ax.text(
        opening_x_page + TRENCH_LW / 2.0 + LABEL_LINE_GAP,
        opening_label_y,
        "Trench opening",
        ha="left",
        va="center",
        color=COL_OPENING,
        fontsize=base.FONT_SIZE,
        fontfamily=base.FONT,
        zorder=5,
    )
    base.add_text(ax, UPPER_WALL_LABEL_XY[0], UPPER_WALL_LABEL_XY[1], "Upper side wall", COL_UPPER)
    base.add_text(ax, LOWER_WALL_LABEL_XY[0], LOWER_WALL_LABEL_XY[1], "Lower side wall", COL_LOWER)
    ax.text(
        base_x_page - TRENCH_LW / 2.0 - LABEL_LINE_GAP,
        base_label_y,
        "Closed end",
        ha="right",
        va="center",
        color=COL_BASE,
        fontsize=base.FONT_SIZE,
        fontfamily=base.FONT,
        fontweight="normal",
        zorder=8,
    )
    add_axis_labels_shortened(ax)
    base.add_origin_label(ax)

    add_l_shaped_wall_arrows(ax)
    add_wall_direction_labels(ax)
    add_wall_coordinate_label(ax)

    # Thin white cover strokes present in the PPT export, preserving the same clean edge.
    base.add_line(
        ax,
        base.ppt_xy(495.839682 - s, 78.118621),
        base.ppt_xy(495.839682 - s, 42.790924),
        "white",
        zorder=6,
    )
    base.add_line(
        ax,
        base.ppt_xy(495.839682 - s, 173.280405),
        base.ppt_xy(495.839682 - s, 137.948795),
        "white",
        zorder=6,
    )

    return fig


def main() -> None:
    base.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig = draw_schematic()
    out = base.FIGURE_DIR / f"{OUT_STEM}.png"
    fig.savefig(out, dpi=base.DPI, facecolor="white", edgecolor="white")
    print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
