#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extended Data Fig. 7 -- axial-molecular-flux analogue of Extended Data Fig. 4.

Same reaction-probability (kappa) study of the side-wall contributions for BOTH species,
shown as CONTOURS, with the SAME kappa x (upper, lower) layout, style and faint species
grouping as extended_figure4.py; only the plotted quantity changes from the number density
n (ED Fig. 4) to the axial molecular flux Gamma_x, and the per-panel guide line from the
max ridge to the axial-flux sign change (matching the Gamma_x convention of Fig. 3).

  Output: figures/figED7.png

    python3 extended_figure7.py
"""

from __future__ import annotations

import extended_figure4

extended_figure4.OUT_NAME = "figED7.png"


def main() -> None:
    extended_figure4.build("Gamma")


if __name__ == "__main__":
    main()
