#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure 3 -- axial-molecular-flux analogue of Figure 2.

Same two-species layout, style and guide lines as figure2.py; only the plotted quantity
changes from the number density n (Fig. 2) to the axial molecular flux Gamma_x. The
reactant A flux Gamma_x,A sits on top and the product B flux Gamma_x,B below, each
decomposed by boundary + total (no zoom row, since the flux has no near-base minimum).

  Output: figures/figure3.png

    python3 figure3.py
"""

from __future__ import annotations

import figure2

figure2.OUT_NAME = "figure3.png"
figure2.LETTER_SIZE_FACTOR = 1.469


def main() -> None:
    figure2.build("Gamma")


if __name__ == "__main__":
    main()
