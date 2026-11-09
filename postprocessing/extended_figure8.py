#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extended Data Fig. 8 -- axial-flux analogue of Extended Data Fig. 5.

Same two-species (A, B rows) x kappa-column centerline layout, style and guide markers as
extended_figure5.py; only the plotted quantity changes from the number density n (ED Fig. 5)
to the axial molecular flux Gamma_x.

  Output: figures/figED8.png

    python3 extended_figure8.py
"""

from __future__ import annotations

import extended_figure5


def main() -> None:
    extended_figure5.LETTER_SIZE_FACTOR = 1.241
    extended_figure5.build("nu")


if __name__ == "__main__":
    main()
