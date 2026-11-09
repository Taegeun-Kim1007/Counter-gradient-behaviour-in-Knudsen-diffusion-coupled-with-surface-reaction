# Figure scripts

Scripts that produce the figures of "Counter-gradient behaviour in Knudsen diffusion coupled with surface reaction". The integral solution is imported from `inversion_analytic.py` in the parent folder.

## Files

| File | Purpose |
|---|---|
| `analytic_case_studies.py` | Case settings of the paper (trench height, temperature, molecular properties, grid and solver settings) |
| `dsmc_postprocessing.py` | Reads the DSMC grid files and averages the runs of each case |
| `figure_common.py` | Plotting tools shared by the figure scripts |
| `figure2.py` … `figure4.py` | Figs. 2–4 |
| `figure_schematic_base.py`, `figure_along_the_wall_schmatic.py` | Schematic of the wall coordinate in Fig. 4 and Extended Data Fig. 6 |
| `extended_figure1.py` … `extended_figure8.py` | Extended Data Figs. 1–8 |
| `figure5_data.py` | Recomputes the data of Fig. 5 |
| `figure5.m` | Plots Fig. 5 (MATLAB) |

Fig. 1 is a schematic of the configuration and is not produced by this code.

## Requirements

Tested with:

- Python 3.13.9, numpy 2.3.5, scipy 1.16.3, matplotlib 3.10.8
- MATLAB R2022b (`figure5.m` only)

The figures use the Arial font. If Arial is not installed, matplotlib substitutes a similar sans-serif font.

## Data

Fig. 5 and Extended Data Figs. 1–3 read the published dataset (figshare, link to be added). Place the dataset folder next to the `code/` folder as `data/`.

```text
code/
├── inversion_analytic.py
└── postprocessing/   (this folder)
data/
├── dsmc/
└── analytic/
```

If the dataset is elsewhere, set the environment variable `COUNTER_GRADIENT_DATA` to its path. Without the dataset, the scripts for Extended Data Figs. 1–3 stop with an error that gives the expected location, and `figure5.m` stops because it cannot find the CSV file. The other figures do not need the dataset.

## Producing the figures

Run the scripts from this folder (`postprocessing/`). The figures are written to `postprocessing/figures/`.

| Figure | Command | Dataset needed |
|---|---|---|
| Fig. 2 | `python figure2.py` | No |
| Fig. 3 | `python figure3.py` | No |
| Fig. 4 | `python figure4.py` | No |
| Fig. 5 | `matlab -batch "run('figure5.m')"` | Yes (`analytic/inversion_region_surface_grid200.csv`) |
| Extended Data Figs. 1–3 | `python extended_figure1.py`, `extended_figure2.py`, `extended_figure3.py` | Yes (`dsmc/`) |
| Extended Data Figs. 4–8 | `python extended_figure4.py` … `extended_figure8.py` | No |

Extended Data Fig. 6 shows the outgoing normal flux of the reactant. `extended_figure6.py` uses `figure4.py`, which obtains this flux by multiplying the incoming flux returned by `compute_surface_flux` in `inversion_analytic.py` by (1 − kappa).

## Cases in each figure

| Figure | Aspect ratio | kappa |
|---|---|---|
| Figs. 2–4, Extended Data Fig. 6 | 5 | 0.05 |
| Extended Data Figs. 1, 2 | 1, 3, 5 | 0.05 |
| Extended Data Fig. 3 | 5 | 0.05 |
| Extended Data Figs. 4, 5, 7, 8 | 5 | 0, 0.05, 0.5, 1 |
| Fig. 5 | 0–100 | 0–1 |

The integral solution has no intermolecular collisions and therefore does not depend on the Knudsen number. Extended Data Fig. 3 compares the same integral solution with DSMC results at different Knudsen numbers.

## Recomputing the data of Fig. 5 (`figure5_data.py`)

**What it does**
- Solves the integral solution for each pair of aspect ratio and kappa and computes the extent of the counter-gradient region.
- Extent: the percentage of cells of the integral solution at which j_x,A · dY_A/dx > 0.
- By default it computes the pairs of the paper: 51 aspect ratios × 92 values of kappa.

**Relation to the published dataset**
- `analytic/inversion_region_surface_grid200.csv` in the dataset contains the results of this calculation.
- This script repeats the calculation so that those values can be checked.
- It neither reads nor modifies the published data file.

**Examples**

```bash
python figure5_data.py                                                          # all pairs
python figure5_data.py --aspect-ratios 5.054741255 --probabilities 0.05160531   # selected pairs
```

- Every combination of the values given to `--aspect-ratios` and `--probabilities` is computed.
- `--workers` sets the number of parallel workers used within one case (default 6).

**Output**
- The cases are computed one at a time, and each result is appended to the output file as soon as it is finished.
- On a rerun, pairs already in the file are skipped, so an interrupted run continues where it stopped.
- Location: `postprocessing/recomputed_data/inversion_region_surface_grid200.csv` (another path can be set with `--out`).
- The columns are those of the published CSV without three bookkeeping columns (`estimated_peak_memory_gb`, `measured_peak_memory_gb`, `case_runtime_seconds`).

**Cost**
- The cost grows with the aspect ratio.
- For the published dataset, the run times of all cases added up to about 1167 hours.
- A single case at an aspect ratio of 100 took about 4.7 hours and about 24 GB of memory.
- Computing a subset of the pairs is therefore recommended for checking the values.

`figure5.m` plots the CSV of the published dataset, not the recomputed file.
