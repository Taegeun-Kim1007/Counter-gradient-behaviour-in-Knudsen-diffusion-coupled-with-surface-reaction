# Flux-based integral solution

Code for the flux-based integral solution in "Counter-gradient behaviour in Knudsen diffusion coupled with surface reaction". The scripts that produce the figures of the paper are in `postprocessing/` and are described in `postprocessing/README.md`.

Taegeun Kim
Department of Mechanical Engineering, Pohang University of Science and Technology, 77 Cheongam-ro, Nam-gu, Pohang, Gyeongbuk 37673, Republic of Korea
ktg1007@postech.ac.kr

## Files

| File | Purpose |
|---|---|
| `inversion_analytic.py` | The flux-based integral solution: solver, reconstruction of the fields, diffusive mass fluxes, contributions of the individual boundaries, counter-gradient region and plotting routines |
| `postprocessing/` | Scripts that produce the figures of the paper (see `postprocessing/README.md`) |

## Requirements

Tested with:

- Python 3.13.9
- numpy 2.3.5
- scipy 1.16.3
- matplotlib 3.10.8

## Usage

```bash
python inversion_analytic.py
```

- The case is set in the block under `if __name__ == "__main__":` at the end of the file:
  - `Aspect_ratio`: aspect ratio
  - `reaction_probability`: reaction probability kappa
  - `base_grid`: grid parameter (nx = base_grid × aspect ratio + 1, ny = base_grid + 1)
- The default case is an aspect ratio of 10 and kappa = 0.01.
- Figures are written to `figures/`. The `save_*` flags in the same block select which figures are written.
- This block runs only for aspect ratios that have plot settings in `PLOT_CASE_SETTINGS` (1, 3, 5, 10, 20, 40 and 100). For other aspect ratios, call the functions below directly.

## Settings used in the paper

- grid: nx = 200 × (aspect ratio) + 1, ny = 201 (for a non-integer aspect ratio, 200 × (aspect ratio) is rounded up)
- `n_quad_jet = 500`, `tol = 1e-15`, `max_iter = 300000`
- pressure at the opening: 1005 Pa

The integral solution has no intermolecular collisions and therefore does not depend on the Knudsen number.

## Main functions

| Function | Purpose |
|---|---|
| `solve_reactive_channel` | Solves for the outgoing fluxes at the boundaries by iteration |
| `compute_full_fields` | Reconstructs the number density and molecular flux fields inside the trench and the contribution of each boundary |
| `compute_diffusive_mass_flux_fields` | Computes the diffusive mass fluxes j_x and j_y |
| `compute_inversion_regions` | Finds the counter-gradient region (j_x · dY/dx > 0) and its area fraction |
| `compute_surface_flux`, `compute_surface_reaction_rate` | Incoming wall fluxes and surface reaction rate |

## License

This code is released under the MIT License. See `LICENSE` for the full text.

## Citation

(to be added once the paper is published)
