#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extended Data Fig. 6 -- outgoing (re-emitted) reactant-A molecular flux along the trench wall.

The re-emitted (outgoing) reactant-A flux at the wall, in the same along-wall layout as Fig. 4,
with the along-wall schematic to the right of panel a. Reuses figure4's machinery; only the
reactant-A OUTGOING view is written here, as figED6.png.

  Output: figures/figED6.png

    python3 extended_figure6.py
"""

from __future__ import annotations

import figure4


def main() -> None:
    figure4.QUANTITY = "wall_flux"
    figure4.ADD_SCHEMATIC = True          # along-wall schematic to the right of panel a
    figure4.LAYOUT = "a_right"
    figure4.FLUX_DIRECTION = "outgoing"    # re-emitted (outgoing) flux
    figure4.FLUX_SPECIES = "A"             # reactant A
    figure4.YLABEL = r"Outgoing normal flux (m$^{-2}$ s$^{-1}$)"
    figure4.OUT_NAME = "figED6.png"
    figure4.LETTER_SIZE_FACTOR = 1.389
    figure4.build()


if __name__ == "__main__":
    main()
