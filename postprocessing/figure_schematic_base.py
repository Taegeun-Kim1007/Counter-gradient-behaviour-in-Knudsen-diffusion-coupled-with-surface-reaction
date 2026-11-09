#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Drawing helpers for the trench schematic. figure_along_the_wall_schmatic.py uses them for the
along-wall schematic shown with Fig. 4 and Extended Data Fig. 6. This module is not run on its own.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


ROOT = Path(__file__).resolve().parent
FIGURE_DIR = ROOT / "figures"

WIDTH_PT = 598.08
HEIGHT_PT = 226.80
DPI = 300   # >=300 dpi at the final printed width per the Nature Physics guideline

BLACK = "#000000"
RED = "#ff0000"
BLUE = "#0070c0"
OPENING_BLUE = "#4472c4"
OPENING_LW = 6.0
UPPER_WALL_PPT_Y = 151.078336
LOWER_WALL_PPT_Y = 56.280393
FONT = "Arial"
FONT_SIZE = 18
ORIGIN_LABEL_X = 126.0
ORIGIN_LABEL_Y = 143.0

# Manual label positions. Adjust these values if the y=h / y=-h labels need
# to move in the rendered figure. Coordinates are PDF/page points: increasing x
# moves right; increasing y moves down.
Y_H_LABEL_X = 91.802784
Y_H_LABEL_Y = 72.802449
Y_MINUS_H_LABEL_X = 83.703243
Y_MINUS_H_LABEL_Y = 189.167450
X_L_LABEL_X = 466.542339
X_L_LABEL_Y = 195.230124
# >>> Side-wall label positions (page pt; increasing y moves the text DOWN). Edit these to move
#     the "upper side wall" / "lower side wall" labels.
UPPER_WALL_LABEL_X = 258.562265
UPPER_WALL_LABEL_Y = 60.121955
LOWER_WALL_LABEL_X = 260.000053
LOWER_WALL_LABEL_Y = 195

# PowerPoint -> PDF transform used by the converted SVG for the stroked wall lines.
PPT_SCALE = 0.998464
PPT_Y0 = 226.525983


def ppt_xy(x: float, y: float) -> tuple[float, float]:
    """Convert the PowerPoint/PDF line coordinates into SVG/page coordinates."""
    return PPT_SCALE * x, PPT_Y0 - PPT_SCALE * y


def add_polygon(ax, points, color: str, zorder: int = 1) -> None:
    ax.add_patch(
        Polygon(points, closed=True, facecolor=color, edgecolor="none", zorder=zorder)
    )


def add_line(ax, p0, p1, color: str, lw: float = 6.0, zorder: int = 1) -> None:
    ax.plot(
        [p0[0], p1[0]],
        [p0[1], p1[1]],
        color=color,
        linewidth=lw,
        solid_capstyle="butt",
        solid_joinstyle="miter",
        zorder=zorder,
    )


def add_text(
    ax,
    x: float,
    y: float,
    text: str,
    color: str = BLACK,
    size: float = FONT_SIZE,
    zorder: int = 5,
) -> None:
    ax.text(
        x,
        y,
        text,
        ha="left",
        va="baseline",
        color=color,
        fontsize=size,
        fontfamily=FONT,
        fontweight="normal",
        zorder=zorder,
    )


def add_blue_arrow(ax, y_top: float, y_mid: float, y_bot: float) -> None:
    add_polygon(
        ax,
        [
            (134.433594, y_top + 11.984375),
            (157.886719, y_top + 11.984375),
            (157.886719, y_top + 5.992187),
            (134.433594, y_top + 5.992187),
        ],
        BLUE,
        zorder=4,
    )
    add_polygon(
        ax,
        [
            (154.890625, y_bot),
            (172.863281, y_mid),
            (154.890625, y_top),
        ],
        BLUE,
        zorder=4,
    )


def add_coordinate_axes(ax) -> None:
    # Coordinate axes with filled arrow heads.
    add_polygon(
        ax,
        [
            (150.25, 221.382812),
            (150.25, 35.539062),
            (144.257812, 35.539062),
            (144.257812, 221.382812),
        ],
        BLACK,
        zorder=2,
    )
    add_polygon(
        ax,
        [(156.238281, 38.535156), (147.253906, 20.562500), (138.265625, 38.535156)],
        BLACK,
        zorder=2,
    )
    add_polygon(
        ax,
        [
            (99.566406, 124.082031),
            (553.847656, 124.082031),
            (553.847656, 118.093750),
            (99.566406, 118.093750),
        ],
        BLACK,
        zorder=2,
    )
    add_polygon(
        ax,
        [
            (550.851562, 130.074219),
            (568.824219, 121.089844),
            (550.851562, 112.101562),
        ],
        BLACK,
        zorder=2,
    )


def add_axis_labels(ax) -> None:
    add_text(ax, 561.795814, 143.637487, r"$x$", BLACK)
    add_text(ax, 131.557631, 16.708736, r"$y$", BLACK)


def add_origin_label(ax) -> None:
    add_text(ax, ORIGIN_LABEL_X, ORIGIN_LABEL_Y, "0", BLACK, zorder=8)


def add_x_lm_label(ax) -> None:
    add_text(ax, X_L_LABEL_X, X_L_LABEL_Y, r"$x=L_{\mathrm{m}}$", BLACK)


def draw_schematic() -> plt.Figure:
    plt.rcParams.update(
        {
            "font.family": FONT,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            # Math (italic variables) in Arial too, matching every other figure.
            "mathtext.fontset": "custom",
            "mathtext.rm": FONT,
            "mathtext.it": f"{FONT}:italic",
            "mathtext.bf": f"{FONT}:bold",
        }
    )

    fig = plt.figure(figsize=(WIDTH_PT / 72.0, HEIGHT_PT / 72.0), dpi=DPI)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, WIDTH_PT)
    ax.set_ylim(HEIGHT_PT, 0)
    ax.axis("off")

    # Right trench wall (trench base). Drawn ABOVE the coordinate axes (zorder > 2) so the
    # x-axis does not cover it.
    add_line(
        ax,
        ppt_xy(489.838277, 153.359183),
        ppt_xy(489.838277, 58.975939),
        RED,
        zorder=3,
    )

    add_coordinate_axes(ax)

    # Trench side walls.
    add_line(
        ax,
        ppt_xy(147.480420, 56.280393),
        ppt_xy(493.148048, 56.280393),
        RED,
        zorder=3,
    )
    add_line(
        ax,
        ppt_xy(144.839646, 151.078336),
        ppt_xy(493.441467, 151.078336),
        RED,
        zorder=3,
    )

    # Labels.
    add_text(ax, Y_MINUS_H_LABEL_X, Y_MINUS_H_LABEL_Y, r"$y=-h$", BLACK)
    add_x_lm_label(ax)
    add_text(ax, Y_H_LABEL_X, Y_H_LABEL_Y, r"$y=h$", BLACK)
    add_text(ax, 5.727190, 107.718739, "Trench opening", BLUE)
    add_text(ax, 494.699026, 108.866973, "Trench base", RED)

    add_line(
        ax,
        ppt_xy(147.480420, UPPER_WALL_PPT_Y + OPENING_LW / 2.0),
        ppt_xy(147.480420, LOWER_WALL_PPT_Y - OPENING_LW / 2.0),
        OPENING_BLUE,
        lw=OPENING_LW,
        zorder=4,
    )

    add_text(ax, UPPER_WALL_LABEL_X, UPPER_WALL_LABEL_Y, "Upper side wall", RED)
    add_text(ax, LOWER_WALL_LABEL_X, LOWER_WALL_LABEL_Y, "Lower side wall", RED)
    add_text(
        ax,
        206.851809,
        112.840860,
        "Surface reaction : A(g) \u2192 B(g)",
        RED,
    )
    # x chosen so this line is CENTRED under "surface reaction : A(g) -> B(g)" above.
    add_text(ax, 237.24, 140.402463, r"Reaction probability : $\kappa$", RED)

    # Incoming arrows through the opening.
    add_blue_arrow(ax, 85.621094, 94.609375, 103.593750)
    add_blue_arrow(ax, 111.621094, 120.609375, 129.593750)
    add_blue_arrow(ax, 139.781250, 148.765625, 157.750000)

    add_axis_labels(ax)
    add_origin_label(ax)

    # Thin white cover strokes present in the PPT export, preserving the same clean edge.
    add_line(
        ax,
        ppt_xy(495.839682, 78.118621),
        ppt_xy(495.839682, 42.790924),
        "white",
        zorder=6,
    )
    add_line(
        ax,
        ppt_xy(495.839682, 173.280405),
        ppt_xy(495.839682, 137.948795),
        "white",
        zorder=6,
    )

    return fig

