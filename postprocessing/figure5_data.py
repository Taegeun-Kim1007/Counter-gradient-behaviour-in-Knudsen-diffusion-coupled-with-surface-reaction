#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fig. 5 data -- extent of the counter-gradient region over the aspect ratio and kappa.

For every (aspect ratio, kappa) pair the integral solution is solved on the grid used in the
paper (ceil(200 * aspect ratio) + 1 cells along the depth, 201 across the height) and the
percentage of cells where j_x,A * dY_A/dx > 0 is recorded. The default pairs are the 51 aspect
ratios and 92 reaction probabilities of the published data file
analytic/inversion_region_surface_grid200.csv. The full sweep is expensive (the cost grows
with the aspect ratio), so --aspect-ratios and --probabilities select a subset.

The cases are solved one at a time and each result is appended to the output file as soon as
it is finished. Pairs already in the output file are skipped, so an interrupted run continues
where it stopped. The published data file is never modified.

    python3 figure5_data.py                                   # full sweep
    python3 figure5_data.py --aspect-ratios 5 --probabilities 0.05

Output: recomputed_data/inversion_region_surface_grid200.csv (same columns as the published
file, without the bookkeeping columns).
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import analytic_case_studies as acs
import dsmc_postprocessing as post
import sys as _sys
from pathlib import Path as _Path

_CODE_DIR = str(_Path(__file__).resolve().parents[1])  # folder of inversion_analytic.py
if _CODE_DIR not in _sys.path:
    _sys.path.insert(0, _CODE_DIR)
import inversion_analytic as analytic  # noqa: E402

ROOT = Path(__file__).resolve().parent
OUT_PATH = ROOT / "recomputed_data" / "inversion_region_surface_grid200.csv"
BASE_GRID = 200

ASPECT_RATIOS = (
    0, 0.5, 0.707788153, 0.879350164, 1.058361703, 1.251080719, 1.46091431, 1.690491204,
    1.942197356, 2.218376019, 2.521421517, 2.853831711, 3.218242195, 3.617451763, 4.054443544,
    4.532404038, 5.054741255, 5.625102671, 6.247393437, 6.925795136, 7.664785303, 8.469157856,
    9.34404458, 10.29493778, 11.3277142, 12.44866032, 13.66449907, 14.98241824, 16.41010036,
    17.95575455, 19.62815009, 21.43665205, 23.39125902, 25.50264305, 27.78219189, 30.24205381,
    32.89518496, 35.75539952, 38.83742279, 42.15694733, 45.73069238, 49.57646674, 53.71323517,
    58.16118875, 62.94181918, 68.07799733, 73.59405635, 79.51587941, 85.87099246, 92.68866228,
    100,
)
PROBABILITIES = (
    0, 0.001, 0.001573191, 0.00208772, 0.002658209, 0.003306048, 0.004046727, 0.004894937,
    0.005865953, 0.006976221, 0.008243714, 0.009688201, 0.011331491, 0.013197676, 0.015313375,
    0.017708006, 0.020414059, 0.023467408, 0.026907637, 0.030778404, 0.035127827, 0.04000891,
    0.045480004, 0.05160531, 0.058455421, 0.066107917, 0.07464801, 0.08416924, 0.09477424,
    0.106575555, 0.119696541, 0.134272335, 0.150450909, 0.16839421, 0.188279403, 0.210300209,
    0.234668357, 0.261615166, 0.291393242, 0.324278325, 0.360571289, 0.4006003, 0.444723151,
    0.493329793, 0.546845058, 0.6, 0.605731621, 0.608213471, 0.616539378, 0.624979259,
    0.633534674, 0.642207205, 0.650998456, 0.659910051, 0.668943638, 0.670493178, 0.678100887,
    0.687383491, 0.696793165, 0.70633165, 0.716000708, 0.725802126, 0.735737718, 0.741677899,
    0.745809319, 0.756018791, 0.766368022, 0.776858925, 0.787493439, 0.798273531, 0.809201192,
    0.819882142, 0.820278443, 0.831507332, 0.842889935, 0.85, 0.854428355, 0.866124727,
    0.877981211, 0.89, 0.905754474, 0.91, 0.919585996, 0.929272971, 0.939061989, 0.94,
    0.948954125, 0.958950466, 0.969052108, 0.979260162, 0.989575749, 1,
)

COLUMNS = (
    "base_grid", "aspect_ratio", "probability", "nx", "ny", "knudsen_number",
    "inversion_region_percent", "inversion_region_fraction",
    "inversion_region_percent_x", "inversion_region_fraction_x",
    "positive_jx_dxa_dx_integral_over_total_area", "runtime_seconds",
)


def grid_shape(aspect_ratio: float) -> tuple[int, int]:
    nx = max(2, int(math.ceil(float(BASE_GRID) * float(aspect_ratio))) + 1)
    return nx, BASE_GRID + 1


def compute_record(aspect_ratio: float, probability: float, case_max_workers: int) -> dict[str, float]:
    started = time.perf_counter()
    pressure, knudsen_number = acs.resolve_fixed_pressure_and_kn()
    nx, ny = grid_shape(aspect_ratio)
    record = {
        "base_grid": BASE_GRID, "aspect_ratio": float(aspect_ratio), "probability": float(probability),
        "nx": nx, "ny": ny, "knudsen_number": float(knudsen_number),
    }
    # No trench or no reaction: there is no counter-gradient region.
    if float(aspect_ratio) == 0.0 or float(probability) == 0.0:
        record.update({key: 0.0 for key in COLUMNS[6:]})
        return record

    cfg = acs.build_ar_case_config(
        aspect_ratio=aspect_ratio, reaction_probability=probability,
        pressure=pressure, knudsen_number=knudsen_number,
    )
    number_density = cfg.pressure / post.BOLTZMANN_CONSTANT / cfg.temperature
    sol = analytic.solve_reactive_channel(
        nx=nx, ny=ny, h=cfg.half_width, Lm=cfg.depth,
        R_A=post.R_UNIVERSAL / cfg.molecular_weight_A * 1000.0,
        R_B=post.R_UNIVERSAL / cfg.molecular_weight_B * 1000.0,
        T_in=cfg.temperature, u_in=0.0, n_in_A=number_density, n_in_B=0.0,
        T_w=cfg.temperature, reaction_probability=cfg.reaction_probability,
        n_quad_jet=acs.N_QUAD_JET, tol=acs.TOL, max_iter=acs.MAX_ITER,
        verbose=False, max_workers=case_max_workers,
    )
    fields, _ = analytic.compute_full_fields(sol, cfg.mass_A, cfg.mass_B,
                                             max_workers=case_max_workers, verbose=False)
    diff = analytic.compute_diffusive_mass_flux_fields(fields, cfg.mass_A, cfg.mass_B)
    stats = analytic.compute_inversion_regions(sol, fields, diff, species="A")["stats"]
    record.update({
        "inversion_region_percent": stats["x"]["inversion_area_percent"],
        "inversion_region_fraction": stats["x"]["inversion_area_fraction"],
        "inversion_region_percent_x": stats["x"]["inversion_area_percent"],
        "inversion_region_fraction_x": stats["x"]["inversion_area_fraction"],
        "positive_jx_dxa_dx_integral_over_total_area":
            stats["x"]["positive_product_integral_over_total_area"],
        "runtime_seconds": time.perf_counter() - started,
    })
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--aspect-ratios", type=float, nargs="+", default=ASPECT_RATIOS)
    parser.add_argument("--probabilities", type=float, nargs="+", default=PROBABILITIES)
    parser.add_argument("--workers", type=int, default=6, help="threads/processes used inside one case")
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    pairs = [(float(a), float(k)) for a in args.aspect_ratios for k in args.probabilities]
    done = finished_pairs(args.out)
    todo = [pair for pair in pairs if pair not in done]
    print(f"{len(pairs)} case(s) requested: {len(pairs) - len(todo)} already in {args.out}, "
          f"{len(todo)} to compute", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.out.exists() or args.out.stat().st_size == 0
    with args.out.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        if write_header:
            writer.writeheader()
            handle.flush()
        for index, (aspect_ratio, probability) in enumerate(todo, start=1):
            record = compute_record(aspect_ratio, probability, args.workers)
            writer.writerow(record)
            handle.flush()   # saved before the next case starts
            print(f"[{index}/{len(todo)}] AR={aspect_ratio:g} kappa={probability:g} "
                  f"-> {record['inversion_region_percent']:.4f}%", flush=True)
    print(f"saved {args.out}", flush=True)


def finished_pairs(path: Path) -> set[tuple[float, float]]:
    """(aspect ratio, kappa) pairs already written to the output file."""
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise SystemExit(f"{path} has different columns; choose another file with --out.")
        return {(float(row["aspect_ratio"]), float(row["probability"])) for row in reader}


if __name__ == "__main__":
    main()
